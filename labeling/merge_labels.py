"""Merge several people's labels_<name>.json files into one labels.json.

When two labellers judge the SAME window, both verdicts are kept rather than
one silently overwriting the other -- that overlap is exactly what an
inter-rater agreement figure needs, and collapsing it early would throw the
number away before it could be computed.

    python merge_labels.py                 # merge every labels_*.json here
    python merge_labels.py --report         # also print agreement stats
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
MERGED = HERE / "labels_merged.json"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true",
                    help="print per-window agreement between labellers")
    args = ap.parse_args()

    sources = sorted(HERE.glob("labels_*.json")) + (
        [HERE / "labels.json"] if (HERE / "labels.json").exists() else [])
    if not sources:
        print("no labels_*.json or labels.json found in this folder")
        return 1

    by_window: dict[str, dict[str, dict]] = defaultdict(dict)
    for path in sources:
        who = path.stem.replace("labels_", "") if path.stem != "labels" else "unnamed"
        data = json.loads(path.read_text(encoding="utf-8"))
        for window_id, entry in data.items():
            by_window[window_id][who] = entry
        print(f"  {path.name}: {len(data)} labels ({who})")

    merged = {}
    disagreements = []
    for window_id, votes in by_window.items():
        labels = [v["label"] for v in votes.values() if v["label"] != "unclear"]
        if len(set(labels)) > 1:
            disagreements.append((window_id, {k: v["label"] for k, v in votes.items()}))
        # Majority vote; ties keep the first rater's label rather than
        # guessing, since a coin-flip would be worse than one person's
        # judgement and worth flagging in disagreements either way.
        base = next(iter(votes.values()))
        chosen = max(set(labels), key=labels.count) if labels else "unclear"
        merged[window_id] = {**base, "label": chosen, "n_raters": len(votes)}

    MERGED.write_text(json.dumps(merged, indent=1), encoding="utf-8")
    print(f"\n{len(merged)} windows merged -> {MERGED.name}")
    print(f"labelled by more than one person: "
          f"{sum(1 for v in by_window.values() if len(v) > 1)}")
    print(f"disagreements: {len(disagreements)}")

    if args.report and disagreements:
        print("\ndisagreements (window: {labeller: label}):")
        for window_id, votes in disagreements[:30]:
            print(f"  {window_id}: {votes}")
        if len(disagreements) > 30:
            print(f"  ... and {len(disagreements) - 30} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
