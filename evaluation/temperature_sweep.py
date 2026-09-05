"""
Router temperature sweep — stability analysis.



Judges are NOT called here. Gold labels are fixed, the router is the system under
test, and holding the measuring instrument constant keeps the two from confounding.

Run from the project root:
    python evaluation/temperature_sweep.py                    # 0.0/0.3/0.7 x 3 runs
    python evaluation/temperature_sweep.py --runs 5
    python evaluation/temperature_sweep.py --temps 0.0 1.0
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict

from langchain_core.messages import AIMessage, HumanMessage

TEST_SET_PATH = "evaluation/router_testset.csv"
OUT_PATH = "evaluation/temperature_sweep_results.json"
CHECKPOINT_PATH = "evaluation/temperature_sweep_checkpoint.json"

INTENTS = ["billing", "technical_support", "info_lookup",
           "escalation", "chitchat", "fallback"]
GOLD_TO_ROUTE = {"unknown": "fallback"}


def parse_context(context):
    msgs = []
    for line in context.splitlines():
        line = line.strip()
        low = line.lower()
        if low.startswith("user:"):
            msgs.append(HumanMessage(content=line.split(":", 1)[1].strip()))
        elif low.startswith("assistant:"):
            msgs.append(AIMessage(content=line.split(":", 1)[1].strip()))
    return msgs


def set_router_temperature(main_mod, temp):
    """
    Rebuild main.router_llm at a new temperature.

    router_node() looks up the module-level name `router_llm` when it runs, so
    rebinding it here changes what the next call uses. This is a monkeypatch and it
    depends on main.py keeping that structure — if the router is ever refactored to
    capture the client in a closure or build it inside the node, this stops working
    SILENTLY (you would get identical results at every temperature). The guard below
    checks the object actually changed.
    """
    from langchain_openai import ChatOpenAI

    schema = getattr(main_mod, "IntentClassification", None)
    if schema is None:
        raise SystemExit(
            "Could not find IntentClassification in main.py — check the name and "
            "adjust set_router_temperature().")

    before = main_mod.router_llm
    main_mod.router_llm = ChatOpenAI(
        model="gpt-4o-mini", temperature=temp
    ).with_structured_output(schema)
    if main_mod.router_llm is before:
        raise SystemExit("Temperature patch had no effect — router_llm unchanged.")


def load_rows():
    with open(TEST_SET_PATH, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def run_once(main_mod, rows):
    """One full pass over the test set. Returns list of predicted labels."""
    preds = []
    for row in rows:
        context = row.get("context", "").strip()
        msgs = parse_context(context) + [HumanMessage(content=row["message"])]
        state = {"messages": msgs, "current_flow": None}
        try:
            pred = main_mod.router_node(state).get("route_to")
        except Exception as e:
            print(f"    error on {row['message'][:30]!r}: {e}")
            pred = None
        preds.append(pred)
    return preds


def accuracy(preds, gold):
    ok = sum(p == g for p, g in zip(preds, gold) if p is not None)
    n = sum(1 for p in preds if p is not None)
    return ok / n if n else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temps", type=float, nargs="*", default=[0.0, 0.3, 0.7])
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--no-patch", action="store_true",
                    help="Do not rebuild router_llm. Uses whatever temperature "
                         "main.py sets. Pass a single --temps value purely as a "
                         "label for the output, and edit main.py yourself.")
    ap.add_argument("--tag", default="",
                    help="Suffix for output filenames, e.g. --tag t07")
    args = ap.parse_args()

    import main as main_mod  # loads the graph

    rows = load_rows()
    gold = [GOLD_TO_ROUTE.get(r["gold_intent"].strip(), r["gold_intent"].strip())
            for r in rows]
    messages = [r["message"] for r in rows]

    total_calls = len(rows) * len(args.temps) * args.runs
    print(f"{len(rows)} items x {len(args.temps)} temperatures x {args.runs} runs "
          f"= {total_calls} router calls\n")

    # all_preds[temp][run] = list of labels
    all_preds = defaultdict(list)

    for temp in args.temps:
        if args.no_patch:
            print(f"  [--no-patch] using main.py's own temperature; "
                  f"labelling these runs as T={temp}")
        else:
            set_router_temperature(main_mod, temp)
        for run in range(args.runs):
            preds = run_once(main_mod, rows)
            all_preds[temp].append(preds)
            print(f"  T={temp}  run {run+1}/{args.runs}  "
                  f"accuracy = {accuracy(preds, gold):.1%}")
            with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
                json.dump({str(k): v for k, v in all_preds.items()}, f, indent=2)
        print()

    report = {"n_items": len(rows), "runs_per_temp": args.runs,
              "temperatures": args.temps, "by_temperature": {}}

    print("=" * 70)
    print("ACCURACY BY TEMPERATURE")
    print(f"  {'T':>5} {'mean':>8} {'sd':>8} {'min':>7} {'max':>7}")
    for temp in args.temps:
        accs = [accuracy(p, gold) for p in all_preds[temp]]
        sd = statistics.stdev(accs) if len(accs) > 1 else 0.0
        print(f"  {temp:>5} {statistics.mean(accs):>7.1%} {sd:>8.3f} "
              f"{min(accs):>6.1%} {max(accs):>6.1%}")
        report["by_temperature"][str(temp)] = {
            "accuracies": accs, "mean": statistics.mean(accs), "sd": sd}

    # Per-item stability: how many DISTINCT labels did each message receive
    # across every run at every temperature? 1 = perfectly stable.
    print("\nPER-ITEM STABILITY (distinct labels across all runs)")
    flips = {}
    for i, msg in enumerate(messages):
        labels = [all_preds[t][r][i]
                  for t in args.temps for r in range(args.runs)]
        flips[msg] = {"gold": gold[i], "labels": Counter(labels),
                      "n_distinct": len(set(labels))}

    unstable = {m: v for m, v in flips.items() if v["n_distinct"] > 1}
    print(f"  stable items:   {len(flips) - len(unstable)}/{len(flips)}")
    print(f"  unstable items: {len(unstable)}/{len(flips)}")

    # Instability by gold class — the number that answers the thesis question
    print("\n  instability by gold class:")
    by_class = defaultdict(lambda: [0, 0])
    for m, v in flips.items():
        by_class[v["gold"]][1] += 1
        if v["n_distinct"] > 1:
            by_class[v["gold"]][0] += 1
    for lab in INTENTS:
        if lab in by_class:
            bad, tot = by_class[lab]
            print(f"    {lab:18} {bad}/{tot} unstable ({bad/tot:.0%})")
    report["instability_by_class"] = {k: {"unstable": v[0], "n": v[1]}
                                      for k, v in by_class.items()}

    if unstable:
        print("\n  unstable items in detail:")
        for m, v in sorted(unstable.items(), key=lambda kv: -kv[1]["n_distinct"]):
            dist = ", ".join(f"{k}x{c}" for k, c in v["labels"].most_common())
            print(f"    {m[:46]!r}  gold={v['gold']}  -> {dist}")
    report["unstable_items"] = [
        {"message": m, "gold": v["gold"], "labels": dict(v["labels"])}
        for m, v in unstable.items()]

    out_path = OUT_PATH.replace(".json", f"_{args.tag}.json") if args.tag else OUT_PATH
    report["raw"] = {str(t): all_preds[t] for t in args.temps}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
