"""
Three-provider blind judge panel for the intent router.

Addresses two separate requirements:
  - judges must be STRONGER than the agent's model (gpt-4o-mini)  -> all three are
  - judges must be DIFFERENT from the agent's model              -> three providers

Design difference from evaluate_intent.py: the judges here are BLIND. Each judge sees
only the user message, any prior context, and the intent taxonomy. It does NOT see the
gold label or the router's prediction, so it cannot anchor on either. Each judge
independently produces a label, exactly as a human annotator would.

That gives four things the original evaluation could not produce:
  1. Router accuracy with per-class precision/recall/F1 and macro-F1
  2. Each judge's own accuracy against the gold labels
  3. Fleiss' kappa across the three judges — do independent models agree with each other?
  4. Per-class unanimity — WHICH classes they agree on, which is the actual finding

Install:
    pip install langchain-openai langchain-anthropic langchain-google-genai

Environment:
    OPENAI_API_KEY, ANTHROPIC_API_KEY (only with credits, omitted this time), GOOGLE_API_KEY

Run from the project root:
    python evaluation/judge_panel.py                 # full test set, all judges + router
    python evaluation/judge_panel.py --n 20          # quick subset
    python evaluation/judge_panel.py --no-router     # judges only, skip loading the graph
"""

import argparse
import csv
import json
import math
import os
import random
from collections import Counter, defaultdict
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from dotenv import load_dotenv
load_dotenv()

TEST_SET_PATH = "evaluation/router_testset.csv"
OUT_PATH = "evaluation/judge_panel_results.json"

INTENTS = ["billing", "technical_support", "info_lookup",
           "escalation", "chitchat", "fallback"]

GOLD_TO_ROUTE = {"unknown": "fallback"}

# ------------------------------------------------------------------
# Judge panel configuration
#
# IMPORTANT: verify these model IDs against each provider's current docs before
# you run. Model strings change often and a wrong one fails at the first call.
# Whatever you use, record the exact strings and the run date in your thesis —
# judge model versions are part of the experimental setup.
# ------------------------------------------------------------------
PANEL = {
    "openai":    {"model": "gpt-5.6",           "cls": "ChatOpenAI"},
    #"anthropic": {"model": "claude-opus-5",     "cls": "ChatAnthropic"},
    "google":    {"model": "gemini-3.7-flash",  "cls": "ChatGoogleGenerativeAI"},
}


class BlindVerdict(BaseModel):
    intent: Literal["billing", "technical_support", "info_lookup",
                    "escalation", "chitchat", "fallback"] = Field(
        description="The single most appropriate intent for this user message")
    confidence: Literal["high", "medium", "low"] = Field(
        description="How clear-cut this classification is")
    reasoning: str = Field(description="One sentence explaining the choice")


# Deliberately identical to the taxonomy in evaluate_intent.py's JUDGE_SYSTEM, so any
# disagreement comes from the models rather than from a reworded rubric. Note that the
# chitchat and fallback definitions overlap for socially-directed out-of-scope messages
# ("tell me a joke") — that overlap is the thing under investigation, so do NOT fix it
# here. Fixing it would erase the effect you are trying to measure.
PANEL_SYSTEM = """You are an expert annotator for a telecom customer-service chatbot's intent router.
Assign the user's message to exactly one intent:
- billing: invoices, payments, charges, balances
- technical_support: connectivity, devices, outages, service problems
- info_lookup: plans, prices, packages, offers, roaming
- escalation: requests for a human, complaints
- chitchat: greetings, thanks, small talk
- fallback: out-of-scope, gibberish, or unclear messages

Choose the single best intent. If the message is genuinely ambiguous between two
intents, choose the one you consider most defensible and mark your confidence as low."""


def build_panel(only=None):
    """Instantiate one judge per provider. Missing packages/keys are skipped with a warning."""
    judges = {}
    for name, cfg in PANEL.items():
        if only and name not in only:
            continue
        try:
            if cfg["cls"] == "ChatOpenAI":
                from langchain_openai import ChatOpenAI
                llm = ChatOpenAI(model=cfg["model"], temperature=0.0)
            elif cfg["cls"] == "ChatAnthropic":
                from langchain_anthropic import ChatAnthropic
                llm = ChatAnthropic(model=cfg["model"], temperature=0.0)
            elif cfg["cls"] == "ChatGoogleGenerativeAI":
                from langchain_google_genai import ChatGoogleGenerativeAI
                llm = ChatGoogleGenerativeAI(model=cfg["model"], temperature=0.0)
            else:
                continue
            judges[name] = llm.with_structured_output(BlindVerdict)
            print(f"  judge ready: {name:10} ({cfg['model']})")
        except Exception as e:
            print(f"  SKIPPED {name}: {type(e).__name__}: {e}")
    if len(judges) < 2:
        raise SystemExit("Need at least 2 judges for agreement statistics.")
    return judges


def judge_blind(judge, context, message):
    ctx = f"\nPrior conversation:\n{context}\n" if context else ""
    user = f"{ctx}\nUser message: {message!r}\n\nWhich intent?"
    return judge.invoke([SystemMessage(content=PANEL_SYSTEM),
                         HumanMessage(content=user)])


def parse_context(context):
    """Same crude parse evaluate_intent.py uses, made robust to spacing/case."""
    msgs = []
    for line in context.splitlines():
        line = line.strip()
        low = line.lower()
        if low.startswith("user:"):
            msgs.append(HumanMessage(content=line.split(":", 1)[1].strip()))
        elif low.startswith("assistant:"):
            msgs.append(AIMessage(content=line.split(":", 1)[1].strip()))
    return msgs


# ------------------------------------------------------------------
# Agreement statistics
# ------------------------------------------------------------------
def fleiss_kappa(ratings):
    """ratings: list of per-item label lists, one label per rater. All items same n raters."""
    n_items = len(ratings)
    n_raters = len(ratings[0])
    cats = sorted({lab for r in ratings for lab in r})
    idx = {c: i for i, c in enumerate(cats)}

    counts = [[0] * len(cats) for _ in range(n_items)]
    for i, r in enumerate(ratings):
        for lab in r:
            counts[i][idx[lab]] += 1

    p_i = [(sum(c * c for c in row) - n_raters) / (n_raters * (n_raters - 1))
           for row in counts]
    p_bar = sum(p_i) / n_items
    p_j = [sum(counts[i][j] for i in range(n_items)) / (n_items * n_raters)
           for j in range(len(cats))]
    p_e = sum(p * p for p in p_j)
    if p_e >= 1.0:
        return 1.0
    return (p_bar - p_e) / (1 - p_e)


def cohen_kappa(a, b):
    """Two aligned label sequences."""
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in set(a) | set(b))
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def interpret_kappa(k):
    if k < 0.20:
        return "slight"
    if k < 0.40:
        return "fair"
    if k < 0.60:
        return "moderate"
    if k < 0.80:
        return "substantial"
    return "almost perfect"


def per_class_prf(gold, pred, labels):
    """Precision/recall/F1 per class, plus macro-F1."""
    out = {}
    f1s = []
    for lab in labels:
        tp = sum(g == lab and p == lab for g, p in zip(gold, pred))
        fp = sum(g != lab and p == lab for g, p in zip(gold, pred))
        fn = sum(g == lab and p != lab for g, p in zip(gold, pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        out[lab] = {"precision": prec, "recall": rec, "f1": f1, "support": tp + fn}
        if tp + fn:
            f1s.append(f1)
    return out, (sum(f1s) / len(f1s) if f1s else 0.0)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - m) / d), min(1.0, (c + m) / d))


# ------------------------------------------------------------------
def load_rows(n, seed):
    with open(TEST_SET_PATH, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if n and n < len(rows):
        by_class = defaultdict(list)
        for r in rows:
            by_class[r["gold_intent"].strip()].append(r)
        rng = random.Random(seed)
        per = max(1, n // len(by_class))
        sample = []
        for cls in sorted(by_class):
            items = by_class[cls][:]
            rng.shuffle(items)
            sample.extend(items[:per])
        rng.shuffle(sample)
        return sample[:n]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="0 = full test set")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-router", action="store_true",
                    help="skip running the router (judges only)")
    ap.add_argument("--judges", nargs="*", default=None,
                    help="subset of: openai anthropic google")
    args = ap.parse_args()

    print("Building judge panel:")
    judges = build_panel(only=args.judges)
    judge_names = list(judges)

    router_node = None
    if not args.no_router:
        from evaluate_intent import router_node as rn  # loads main.py / the graph
        router_node = rn

    rows = load_rows(args.n, args.seed)
    print(f"\nEvaluating {len(rows)} items "
          f"({len(rows) * len(judges) + (len(rows) if router_node else 0)} model calls)\n")

    records = []
    for i, row in enumerate(rows, 1):
        message = row["message"]
        gold = GOLD_TO_ROUTE.get(row["gold_intent"].strip(), row["gold_intent"].strip())
        context = row.get("context", "").strip()
        difficulty = row.get("difficulty", "n/a").strip()

        rec = {"message": message, "gold": gold, "difficulty": difficulty,
               "context": context, "judges": {}}

        if router_node:
            msgs = parse_context(context) + [HumanMessage(content=message)]
            state = {"messages": msgs, "current_flow": None}
            pred = router_node(state).get("route_to")
            if pred not in INTENTS:
                print(f"  WARNING: router returned unknown label {pred!r}")
            rec["router"] = pred

        for name, judge in judges.items():
            try:
                v = judge_blind(judge, context, message)
                rec["judges"][name] = {"intent": v.intent,
                                       "confidence": v.confidence,
                                       "reasoning": v.reasoning}
            except Exception as e:
                print(f"  {name} failed on item {i}: {e}")
                rec["judges"][name] = {"intent": None, "confidence": None,
                                       "reasoning": f"ERROR: {e}"}

        labels = [rec["judges"][n]["intent"] for n in judge_names]
        unanimous = len(set(labels)) == 1 and labels[0] is not None
        rec["unanimous"] = unanimous
        rec["matches_gold_all"] = unanimous and labels[0] == gold

        mark = "==" if rec["matches_gold_all"] else ("~~" if unanimous else "!!")
        print(f"[{mark}] {message[:38]:38} gold={gold:17} "
              + " ".join(f"{n[:2]}={rec['judges'][n]['intent']}" for n in judge_names))
        records.append(rec)

    # drop any item where a judge errored, for the statistics only
    clean = [r for r in records
             if all(r["judges"][n]["intent"] for n in judge_names)]
    if len(clean) < len(records):
        print(f"\nNote: {len(records) - len(clean)} items excluded from statistics (judge errors)")

    gold_seq = [r["gold"] for r in clean]
    judge_seqs = {n: [r["judges"][n]["intent"] for r in clean] for n in judge_names}
    ratings = [[r["judges"][n]["intent"] for n in judge_names] for r in clean]

    report = {"n": len(clean), "judges": {n: PANEL[n]["model"] for n in judge_names}}

    print("\n" + "=" * 70)
    print("JUDGE ACCURACY AGAINST GOLD (blind labelling)")
    for n in judge_names:
        k = sum(a == b for a, b in zip(judge_seqs[n], gold_seq))
        lo, hi = wilson(k, len(clean))
        ck = cohen_kappa(judge_seqs[n], gold_seq)
        print(f"  {n:10} {k:>3}/{len(clean)} = {k/len(clean):>6.1%}  "
              f"CI [{lo:.0%},{hi:.0%}]  kappa vs gold = {ck:.3f} ({interpret_kappa(ck)})")
        report.setdefault("judge_vs_gold", {})[n] = {
            "correct": k, "accuracy": k / len(clean), "ci95": [lo, hi], "cohen_kappa": ck}

    fk = fleiss_kappa(ratings)
    print(f"\nINTER-JUDGE AGREEMENT (Fleiss' kappa, {len(judge_names)} raters): "
          f"{fk:.3f} ({interpret_kappa(fk)})")
    unan = sum(r["unanimous"] for r in clean)
    print(f"Unanimous on {unan}/{len(clean)} items ({unan/len(clean):.1%})")
    report["fleiss_kappa"] = fk
    report["unanimous"] = {"count": unan, "rate": unan / len(clean)}

    print("\nPairwise Cohen's kappa:")
    for i, a in enumerate(judge_names):
        for b in judge_names[i + 1:]:
            ck = cohen_kappa(judge_seqs[a], judge_seqs[b])
            print(f"  {a:10} vs {b:10} {ck:.3f} ({interpret_kappa(ck)})")
            report.setdefault("pairwise_kappa", {})[f"{a}|{b}"] = ck

    # THE KEY TABLE: where does the panel agree, and where does it fracture?
    print("\nPER-CLASS PANEL BEHAVIOUR (by gold class)")
    print(f"  {'gold class':18} {'n':>3} {'unanimous':>10} {'all=gold':>9}")
    per_class = {}
    for lab in INTENTS:
        items = [r for r in clean if r["gold"] == lab]
        if not items:
            continue
        u = sum(r["unanimous"] for r in items)
        g = sum(r["matches_gold_all"] for r in items)
        print(f"  {lab:18} {len(items):>3} {u/len(items):>9.0%} {g/len(items):>8.0%}")
        per_class[lab] = {"n": len(items), "unanimous_rate": u / len(items),
                          "all_match_gold_rate": g / len(items)}
    report["per_class_panel"] = per_class

    if router_node:
        router_seq = [r["router"] for r in clean]
        k = sum(a == b for a, b in zip(router_seq, gold_seq))
        lo, hi = wilson(k, len(clean))
        prf, macro = per_class_prf(gold_seq, router_seq, INTENTS)
        print("\n" + "=" * 70)
        print(f"ROUTER vs GOLD: {k}/{len(clean)} = {k/len(clean):.1%}  CI [{lo:.0%},{hi:.0%}]")
        print(f"Macro-F1: {macro:.3f}   (accuracy alone hides per-class collapse)")
        print(f"  {'class':18} {'prec':>6} {'rec':>6} {'F1':>6} {'n':>4}")
        for lab, m in prf.items():
            if m["support"]:
                print(f"  {lab:18} {m['precision']:>6.2f} {m['recall']:>6.2f} "
                      f"{m['f1']:>6.2f} {m['support']:>4}")
        report["router"] = {"accuracy": k / len(clean), "ci95": [lo, hi],
                            "macro_f1": macro, "per_class": prf}

        print("\nRouter agreement with each judge (Cohen's kappa):")
        for n in judge_names:
            ck = cohen_kappa(router_seq, judge_seqs[n])
            agree = sum(a == b for a, b in zip(router_seq, judge_seqs[n])) / len(clean)
            print(f"  {n:10} agree {agree:.1%}  kappa {ck:.3f} ({interpret_kappa(ck)})")
            report.setdefault("router_vs_judge", {})[n] = {"agreement": agree, "kappa": ck}

    # Items where the panel split — the ambiguous boundary, empirically identified
    split = [r for r in clean if not r["unanimous"]]
    print(f"\nITEMS WHERE THE PANEL SPLIT: {len(split)}")
    for r in split:
        labs = ", ".join(f"{n}={r['judges'][n]['intent']}" for n in judge_names)
        rt = f" router={r['router']}" if router_node else ""
        print(f"  {r['message'][:46]!r}\n     gold={r['gold']}{rt}  {labs}")
    report["split_items"] = [
        {"message": r["message"], "gold": r["gold"],
         "router": r.get("router"),
         "labels": {n: r["judges"][n]["intent"] for n in judge_names}}
        for r in split]

    report["records"] = records
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
