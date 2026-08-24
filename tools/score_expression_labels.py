"""Score the expression-validation study: inter-rater agreement first, then
model accuracy against the agreed human labels -- in that order, because the
order is the point.

Per docs/LITERATURE_REVIEW.md section 4 (Whitehill et al. 2014's validation
template): human agreement sets the CEILING. If two people can't agree on a
crop, the model cannot be faulted for missing a signal that isn't reliably
readable in the first place, and the model's accuracy number is meaningless
without that context. So this reports:

    1. Cohen's kappa + raw percent agreement between the two labellers.
    2. Model accuracy against the AGREED subset only (disagreements are
       reported and excluded, not silently dropped, and not tie-broken by a
       third opinion that doesn't exist).
    3. Both broken out by the tiny/off-angle/frontal buckets from
       prepare_expression_labels.py, since accuracy is expected to vary
       sharply by condition -- an aggregate number would hide exactly the
       failure this whole study exists to find.
    4. A Wilson score interval on every accuracy figure -- with the sample
       sizes involved here (dozens per bucket), a bare point accuracy invites
       reading noise as a finding.

No sklearn dependency: kappa and the confusion matrix are computed directly
from their definitions, which are simple enough not to need one.

Run:
    python -m tools.score_expression_labels --dir outputs/expression_labels
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path

LABELS = ("happy", "sad", "neutral", "unclear")


def _load_labels(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as fh:
        return {row["crop_id"]: row["label"] for row in csv.DictReader(fh)}


def _cohen_kappa(a: list[str], b: list[str]) -> float:
    """Standard Cohen's kappa: (observed agreement - chance agreement) /
    (1 - chance agreement), chance agreement from each rater's own marginals."""
    n = len(a)
    if n == 0:
        return float("nan")
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    count_a, count_b = Counter(a), Counter(b)
    pe = sum((count_a[k] / n) * (count_b[k] / n) for k in set(a) | set(b))
    if pe >= 1.0:
        return 1.0  # both raters used exactly one label each -- no chance to disagree
    return (po - pe) / (1 - pe)


def _wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion -- stable at small n, unlike
    the naive normal approximation, which can extend past [0, 1]."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((centre - margin) / denom, (centre + margin) / denom)


def _confusion(true: list[str], pred: list[str], labels: tuple[str, ...]) -> dict:
    table = {t: Counter() for t in labels}
    for t, p in zip(true, pred):
        table.setdefault(t, Counter())[p] += 1
    return table


def _print_confusion(table: dict, labels: tuple[str, ...]) -> None:
    header = "true\\pred".ljust(10) + "".join(l[:7].rjust(9) for l in labels)
    print(header)
    for t in labels:
        row = table.get(t, Counter())
        print(t.ljust(10) + "".join(str(row.get(p, 0)).rjust(9) for p in labels))


def score(out_dir: Path, labellers: list[str]) -> None:
    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open(encoding="utf-8") as fh:
        manifest = list(csv.DictReader(fh))
    bucket_of = {row["crop_id"]: row["bucket"] for row in manifest}
    model_label = {row["crop_id"]: row["model_label"] for row in manifest}

    if len(labellers) != 2:
        raise SystemExit(
            f"Need exactly two labellers to compute inter-rater agreement, "
            f"got {labellers!r}."
        )
    label_files = {name: _load_labels(out_dir / f"labels_{name}.csv") for name in labellers}
    common = sorted(set(label_files[labellers[0]]) & set(label_files[labellers[1]]))
    if not common:
        raise SystemExit("No crop_ids labelled by both raters yet.")

    a = [label_files[labellers[0]][c] for c in common]
    b = [label_files[labellers[1]][c] for c in common]

    agree = sum(1 for x, y in zip(a, b) if x == y)
    kappa = _cohen_kappa(a, b)
    print(f"=== Inter-rater agreement ({labellers[0]} vs {labellers[1]}), "
          f"n={len(common)} ===")
    print(f"percent agreement: {100 * agree / len(common):.1f}%")
    print(f"Cohen's kappa:     {kappa:.3f}  "
          f"({'substantial+' if kappa >= 0.6 else 'below substantial -- see caveat'})")
    print()
    print("confusion (rater 1 rows vs rater 2 columns):")
    _print_confusion(_confusion(a, b, LABELS), LABELS)
    print()

    if kappa < 0.6:
        print(
            "CAVEAT: kappa below the 'substantial agreement' band (Landis & "
            "Koch). Per docs/LITERATURE_REVIEW.md section 4, this on its own "
            "is evidence the 3-class signal is not cleanly readable on this "
            "footage at this resolution -- report it as such rather than "
            "proceeding to a model-accuracy claim as if the ground truth "
            "were solid.\n"
        )

    agreed_crops = [c for c, x, y in zip(common, a, b) if x == y]
    excluded = len(common) - len(agreed_crops)
    print(f"=== Model accuracy vs. agreed human labels "
          f"(n={len(agreed_crops)}, {excluded} disagreements excluded, "
          f"not tie-broken) ===")

    human = [label_files[labellers[0]][c] for c in agreed_crops]
    pred = [model_label.get(c) or "none" for c in agreed_crops]
    correct = sum(1 for h, p in zip(human, pred) if h == p)
    lo, hi = _wilson_interval(correct, len(agreed_crops))
    print(f"overall: {correct}/{len(agreed_crops)} = "
          f"{100 * correct / len(agreed_crops):.1f}%  "
          f"(95% CI {100*lo:.1f}-{100*hi:.1f}%)")
    print()
    print("confusion (human rows vs model columns):")
    _print_confusion(_confusion(human, pred, LABELS + ("none",)), LABELS + ("none",))
    print()

    print("=== Accuracy by condition (this is the number that matters -- an "
          "aggregate hides exactly the failure this study exists to find) ===")
    by_bucket: dict[str, list[int]] = {}
    for c, h, p in zip(agreed_crops, human, pred):
        by_bucket.setdefault(bucket_of.get(c, "?"), []).append(1 if h == p else 0)
    for bucket, hits in sorted(by_bucket.items()):
        n = len(hits)
        k = sum(hits)
        lo, hi = _wilson_interval(k, n)
        print(f"  {bucket:<10} n={n:<4} {k}/{n} = {100*k/n:.1f}%  "
              f"(95% CI {100*lo:.1f}-{100*hi:.1f}%)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="outputs/expression_labels")
    parser.add_argument(
        "--labellers", nargs=2, required=True,
        help="The two --labeller names used with label_expressions.py.",
    )
    args = parser.parse_args()
    score(Path(args.dir), args.labellers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
