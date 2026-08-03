"""
Router evaluation with two complementary metrics:
  1. Exact-match accuracy  — objective, reproducible (predicted == gold)
  2. LLM-as-a-judge         — a STRONGER model grades whether the routing was reasonable

Design principle: the judge model (gpt-4o) must be different from and stronger than
the router model (gpt-4o-mini). Judging a model with itself would be circular.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import json
from collections import defaultdict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Literal

from main import router_node
from src.state import AgentState

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
TEST_SET_PATH = "evaluation/router_testset.csv"   # adjust if yours differs
JUDGE_MODEL = "gpt-4o"                              # STRONGER than the router's gpt-4o-mini
VALID_INTENTS = ["billing", "technical_support", "info_lookup",
                 "escalation", "chitchat", "fallback"]

# NOTE: gold labels in the test set use "unknown"; the router's route_to uses "fallback"
# for that case. We normalise so they compare correctly.
GOLD_TO_ROUTE = {"unknown": "fallback"}


# ------------------------------------------------------------------
# The LLM judge
# ------------------------------------------------------------------
class JudgeVerdict(BaseModel):
    reasonable: bool = Field(description="Is the router's classification defensible for this message?")
    ideal_intent: str = Field(description="The intent the judge considers most correct")
    reasoning: str = Field(description="One sentence explaining the verdict")


JUDGE_SYSTEM = """You are an expert evaluator for a telecom customer-service chatbot's intent router.
The router assigns each user message to exactly one intent:
- billing: invoices, payments, charges, balances
- technical_support: connectivity, devices, outages, service problems
- info_lookup: plans, prices, packages, offers, roaming
- escalation: requests for a human, complaints
- chitchat: greetings, thanks, small talk
- fallback: out-of-scope, gibberish, or unclear messages

You are given the user message, any prior conversation context, the gold (human) label,
and the router's prediction. Judge whether the router's prediction is REASONABLE for the
message — not merely whether it string-matches the gold label. Some messages are genuinely
ambiguous and more than one intent may be defensible. Be fair but rigorous."""


def build_judge():
    return ChatOpenAI(model=JUDGE_MODEL, temperature=0.0).with_structured_output(JudgeVerdict)


def judge_one(judge, context, message, gold, predicted):
    ctx = f"\nPrior context:\n{context}\n" if context else ""
    user = (
        f"{ctx}\nUser message: {message!r}\n"
        f"Gold label: {gold}\n"
        f"Router predicted: {predicted}\n\n"
        f"Is the router's prediction reasonable?"
    )
    return judge.invoke([SystemMessage(content=JUDGE_SYSTEM), HumanMessage(content=user)])


# ------------------------------------------------------------------
# Main evaluation loop
# ------------------------------------------------------------------
def main():
    judge = build_judge()

    exact_correct = 0
    judge_reasonable = 0
    total = 0

    by_difficulty = defaultdict(lambda: {"exact": 0, "judge": 0, "n": 0})
    confusion = defaultdict(lambda: defaultdict(int))   # gold -> predicted -> count
    disagreements = []   # exact-wrong but judge-reasonable (the interesting cases)

    with open(TEST_SET_PATH, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        print("Detected columns:", reader.fieldnames, "\n")

        for row in reader:
            message = row["message"]
            gold = row["gold_intent"].strip()
            context = row.get("context", "").strip()
            difficulty = row.get("difficulty", "n/a").strip()

            # Normalise gold ("unknown") to router space ("fallback")
            gold_route = GOLD_TO_ROUTE.get(gold, gold)

            # Build state. Include context as prior turns if present so the router
            # can use conversation history (matches how it runs live).
            messages = []
            if context:
                # crude parse of "User: .. \n Assistant: .." lines into messages
                for line in context.splitlines():
                    line = line.strip()
                    if line.lower().startswith("user:"):
                        messages.append(HumanMessage(content=line[5:].strip()))
                    elif line.lower().startswith("assistant:"):
                        from langchain_core.messages import AIMessage
                        messages.append(AIMessage(content=line[10:].strip()))
            messages.append(HumanMessage(content=message))

            state: AgentState = {"messages": messages, "current_flow": None}
            result = router_node(state)
            predicted = result.get("route_to")

            # --- Metric 1: exact match ---
            is_exact = (predicted == gold_route)
            if is_exact:
                exact_correct += 1

            # --- Metric 2: LLM judge ---
            verdict = judge_one(judge, context, message, gold, predicted)
            if verdict.reasonable:
                judge_reasonable += 1

            # bookkeeping
            total += 1
            by_difficulty[difficulty]["n"] += 1
            by_difficulty[difficulty]["exact"] += int(is_exact)
            by_difficulty[difficulty]["judge"] += int(verdict.reasonable)
            confusion[gold_route][predicted] += 1

            if not is_exact and verdict.reasonable:
                disagreements.append({
                    "message": message, "gold": gold_route, "predicted": predicted,
                    "judge_ideal": verdict.ideal_intent, "why": verdict.reasoning,
                })

            # live progress line
            flag = "OK " if is_exact else ("~J " if verdict.reasonable else "XX ")
            print(f"[{flag}] {message[:45]:45} gold={gold_route:17} pred={predicted}")

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Total messages evaluated: {total}")
    print(f"Exact-match accuracy:     {exact_correct/total:.1%}  ({exact_correct}/{total})")
    print(f"LLM-judge reasonable:     {judge_reasonable/total:.1%}  ({judge_reasonable}/{total})")

    print("\nBy difficulty:")
    print(f"  {'level':10} {'n':>3}  {'exact':>7}  {'judge':>7}")
    for level in ["easy", "medium", "hard", "context"]:
        if level in by_difficulty:
            d = by_difficulty[level]
            print(f"  {level:10} {d['n']:>3}  {d['exact']/d['n']:>6.0%}  {d['judge']/d['n']:>6.0%}")

    print("\nConfusion (gold -> predicted), off-diagonal only:")
    for gold in sorted(confusion):
        for pred, count in sorted(confusion[gold].items()):
            if pred != gold:
                print(f"  {gold:17} -> {pred:17} : {count}")

    print(f"\nDisagreements (exact-wrong but judge-reasonable): {len(disagreements)}")
    for d in disagreements:
        print(f"  {d['message'][:50]!r}")
        print(f"     gold={d['gold']} pred={d['predicted']} judge_ideal={d['judge_ideal']}")
        print(f"     {d['why']}")

    # save full results for the thesis appendix
    with open("evaluation/results.json", "w", encoding="utf-8") as out:
        json.dump({
            "total": total,
            "exact_accuracy": exact_correct/total,
            "judge_reasonable_rate": judge_reasonable/total,
            "by_difficulty": {k: dict(v) for k, v in by_difficulty.items()},
            "disagreements": disagreements,
        }, out, indent=2, ensure_ascii=False)
    print("\nSaved detailed results to evaluation/results.json")


if __name__ == "__main__":
    main()
