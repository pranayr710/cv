"""How much do DIPSER's four expert raters agree with each other?

Our own labels re-judged blind agreed 65.8 per cent of the time, Cohen's kappa
0.312. Read alone that looks like careless labelling. The question it cannot
answer is whether this task is simply hard to agree on, and that is exactly
what a published dataset with several independent raters can settle.

DIPSER (Marquez-Carpintero et al., CC BY 4.0) labels in-person classroom
attention on a 1-5 scale with four expert raters plus the student's own
self-assessment. Computing their pairwise agreement gives a benchmark our
number can be read against -- and if experts also disagree a third of the
time, then a kappa near 0.3 is a property of the task rather than a fault in
how we collected labels.

Two comparisons are made, because they answer different questions:

* **On the 5-point scale**, which is what DIPSER actually asked for.
* **Collapsed to on/off task**, which is our question, since a model trained
  for a binary decision cannot be held to an ordinal rater's precision.

    python tools/dipser_agreement.py
    python tools/dipser_agreement.py --root labeling/images/1
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

REPORT = Path("outputs/dipser_agreement.json")

#: DIPSER attention runs 1 (not attending) to 5 (fully attending). Splitting
#: above the midpoint matches how our own on/off question was posed.
ON_TASK_FROM = 4


def kappa(a: list[int], b: list[int]) -> float:
    """Cohen's kappa for two raters over the same items."""
    labels = sorted(set(a) | set(b))
    n = len(a)
    if n == 0 or len(labels) < 2:
        return float("nan")
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    expected = sum((a.count(v) / n) * (b.count(v) / n) for v in labels)
    return (observed - expected) / (1 - expected) if expected < 1 else float("nan")


def load(root: Path) -> dict[str, dict[str, int]]:
    """``{rater: {timestamp: attention}}`` for every label file found."""
    out: dict[str, dict[str, int]] = {}
    for path in sorted(root.rglob("labels/*.json")):
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        by_time: dict[str, int] = {}
        for row in rows:
            value = row.get("attention")
            if value is None:
                continue
            try:
                by_time[str(row.get("datetime"))] = int(value)
            except (TypeError, ValueError):
                continue
        if by_time:
            out[f"{path.parent.parent.name}/{path.stem}"] = by_time
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("labeling/images/1"))
    args = ap.parse_args()

    raters = load(args.root)
    if len(raters) < 2:
        print(f"Fewer than two label files under {args.root}.")
        return 1

    # Group by subject so raters are only compared on the same student.
    by_subject: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)
    for key, values in raters.items():
        subject, name = key.split("/", 1)
        by_subject[subject][name] = values

    print(f"{len(raters)} label files across {len(by_subject)} subject(s)\n")

    five_k, five_a, bin_k, bin_a = [], [], [], []
    for subject, people in sorted(by_subject.items()):
        names = sorted(people)
        print(f"  {subject}: {', '.join(names)}")
        for one, two in combinations(names, 2):
            shared = sorted(set(people[one]) & set(people[two]))
            if len(shared) < 10:
                continue
            a = [people[one][t] for t in shared]
            b = [people[two][t] for t in shared]
            agree = sum(1 for x, y in zip(a, b) if x == y) / len(shared)
            k5 = kappa(a, b)

            ab = [1 if v >= ON_TASK_FROM else 0 for v in a]
            bb = [1 if v >= ON_TASK_FROM else 0 for v in b]
            agree_b = sum(1 for x, y in zip(ab, bb) if x == y) / len(shared)
            kb = kappa(ab, bb)

            five_a.append(agree)
            bin_a.append(agree_b)
            if k5 == k5:
                five_k.append(k5)
            if kb == kb:
                bin_k.append(kb)
            print(f"    {one:<18} vs {two:<18} n={len(shared):>4}  "
                  f"5-point {agree:.1%} (k={k5:+.3f})  "
                  f"on/off {agree_b:.1%} (k={kb:+.3f})")

    if not five_a:
        print("\n  No rater pair shared enough timestamps to compare.")
        return 1

    print(f"\n  {'':<28}{'agreement':>12}{'kappa':>10}")
    print(f"  {'DIPSER experts, 5-point':<28}{np.mean(five_a):>12.1%}"
          f"{np.mean(five_k) if five_k else float('nan'):>10.3f}")
    print(f"  {'DIPSER experts, on/off':<28}{np.mean(bin_a):>12.1%}"
          f"{np.mean(bin_k) if bin_k else float('nan'):>10.3f}")
    print(f"  {'our labels, on/off':<28}{0.658:>12.1%}{0.312:>10.3f}")

    print()
    ours = 0.312
    theirs = float(np.mean(bin_k)) if bin_k else float("nan")
    if theirs == theirs:
        if theirs <= ours + 0.1:
            print("  Expert raters on a published classroom dataset agree no")
            print("  better than we do. A kappa near 0.3 is therefore what this")
            print("  task looks like, not evidence of careless labelling, and a")
            print("  larger dataset would inherit the same ceiling.")
        else:
            print(f"  The experts agree substantially better ({theirs:.3f} "
                  f"against our {ours:.3f}).")
            print("  That gap is worth closing with a written labelling rule")
            print("  before collecting more judgements.")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({
        "root": str(args.root),
        "n_label_files": len(raters),
        "n_subjects": len(by_subject),
        "n_pairs": len(five_a),
        "five_point_agreement": round(float(np.mean(five_a)), 4),
        "five_point_kappa":
            None if not five_k else round(float(np.mean(five_k)), 4),
        "binary_agreement": round(float(np.mean(bin_a)), 4),
        "binary_kappa":
            None if not bin_k else round(float(np.mean(bin_k)), 4),
        "our_binary_agreement": 0.658,
        "our_binary_kappa": 0.312,
        "on_task_threshold": ON_TASK_FROM,
    }, indent=1), encoding="utf-8")
    print(f"\n  report -> {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
