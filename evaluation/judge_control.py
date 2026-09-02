

import argparse
import csv
import json
import math
import random
from collections import Counter

from langchain_core.messages import HumanMessage, SystemMessage

# Imports the SAME judge we used in the first evaluation, so this tests the actual metric
# and not a reconstruction of it.

from evaluate_intent import JUDGE_SYSTEM, build_judge, judge_one

TEST_SET_PATH = "evaluation/router_testset.csv"
OUT_PATH = "evaluation/judge_control_results.json"

# A deliberately wrong label for each gold class. Chosen to be indefensible
# rather than merely debatable: "why was I charged twice" -> chitchat,
# "asdfghjkl" -> escalation. If a judge waves these through, it waves anything
# through.
ABSURD = {
    "billing": "chitchat",
    "technical_support": "chitchat",
    "info_lookup": "escalation",
    "escalation": "chitchat",
    "chitchat": "billing",
    "unknown": "escalation",
    "fallback": "escalation",
}


def judge_one_without_gold(judge, context, message, predicted):
    """Mirrors judge_one() from evaluate_intent.py, minus the gold label line."""
    ctx = f"\nPrior context:\n{context}\n" if context else ""
    user = (
        f"{ctx}\nUser message: {message!r}\n"
        f"Router predicted: {predicted}\n\n"
        f"Is the router's prediction reasonable?"
    )
    return judge.invoke([SystemMessage(content=JUDGE_SYSTEM), HumanMessage(content=user)])


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def load_sample(n, seed):
    with open(TEST_SET_PATH, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    # Stratify by gold class so every intent is represented in the sample.
    by_class = {}
    for r in rows:
        by_class.setdefault(r["gold_intent"].strip(), []).append(r)
    rng = random.Random(seed)
    per_class = max(1, n // len(by_class))
    sample = []
    for cls in sorted(by_class):
        items = by_class[cls][:]
        rng.shuffle(items)
        sample.extend(items[:per_class])
    rng.shuffle(sample)
    return sample[:n]


def run_arm(judge, arm, sample):
    records = []
    approved = 0
    for row in sample:
        message = row["message"]
        gold = row["gold_intent"].strip()
        context = row.get("context", "").strip()

        predicted = gold if arm == "gold" else ABSURD[gold]

        if arm == "absurd_nogold":
            verdict = judge_one_without_gold(judge, context, message, predicted)
        else:
            verdict = judge_one(judge, context, message, gold, predicted)

        approved += int(verdict.reasonable)
        records.append({
            "message": message,
            "gold": gold,
            "shown_prediction": predicted,
            "reasonable": bool(verdict.reasonable),
            "judge_ideal": verdict.ideal_intent,
            "why": verdict.reasoning,
        })
        mark = "APPROVED" if verdict.reasonable else "rejected"
        print(f"  [{mark:8}] {message[:40]:40} shown={predicted}")
    return approved, records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="items per arm")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    judge = build_judge()
    sample = load_sample(args.n, args.seed)
    print(f"Sampled {len(sample)} items, stratified by gold class.\n")

    results = {}
    for arm in ["gold", "absurd", "absurd_nogold"]:
        print(f"--- arm: {arm} ---")
        approved, records = run_arm(judge, arm, sample)
        lo, hi = wilson(approved, len(sample))
        results[arm] = {
            "n": len(sample),
            "approved": approved,
            "approval_rate": approved / len(sample),
            "ci95": [lo, hi],
            "records": records,
        }
        print()

    print("=" * 64)
    print(f"{'arm':16} {'approved':>10}  {'rate':>7}  {'95% CI':>18}")
    for arm, r in results.items():
        lo, hi = r["ci95"]
        print(f"{arm:16} {r['approved']:>4}/{r['n']:<5} {r['approval_rate']:>7.0%}  "
              f"[{lo:>5.0%}, {hi:>5.0%}]")

    gold_rate = results["gold"]["approval_rate"]
    absurd_rate = results["absurd"]["approval_rate"]
    nogold_rate = results["absurd_nogold"]["approval_rate"]
    print(f"\nSeparation (gold - absurd): {gold_rate - absurd_rate:+.0%}")
    print(f"Gold-visibility effect (absurd_nogold - absurd): {nogold_rate - absurd_rate:+.0%}")

    # Which classes, if any, the judge is willing to rubber-stamp
    stamped = Counter(rec["gold"] for rec in results["absurd"]["records"] if rec["reasonable"])
    if stamped:
        print("\nAbsurd predictions approved, by gold class:")
        for cls, count in stamped.most_common():
            print(f"  {cls:18} {count}")

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
