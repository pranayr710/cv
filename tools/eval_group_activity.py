"""Group-engagement evaluation harness for ClassGraph Stage 5.

Scores predicted clip-level group labels against OUC-CGE ground truth
(prepare the dataset first with ``tools/prepare_ouccge.py``). This is the
harness Person C owns; ``backend/group_activity.py``'s ARG+GCN predictions
are scored HERE, never with ad-hoc scripts, so every number in reports and
slides traces to one audited implementation of each metric.

Reporting discipline (from docs/PROJECT_PLAN.md section 5):
  * Accuracy is printed WITH its Wilson 95% CI — with ~3k clips but only ~17
    underlying sources, point estimates flatter the system.
  * Machine accuracy is printed NEXT TO a human-observer agreement figure
    when one is supplied (--human-agreement), never alone: "better than
    chance" is the wrong bar for a three-class ordinal task.
  * Abstentions are handled ONLY via the explicit --abstain flag. There is no
    silent default that quietly drops or counts them; the chosen policy is
    echoed in the output header.
  * Split integrity is re-checked here, not trusted from the manifest:
    any clip id appearing under two splits, or any source spanning splits,
    aborts the evaluation.

Prediction input: JSONL, one record per clip::

    {"clip_id": "clip_0001", "label": "high"}          # or "medium"/"low"
    {"clip_id": "clip_0002", "label": null}            # explicit abstention

Usage:
    python -m tools.eval_group_activity \
        --manifest data/OUC-CGE/prepared/manifest.csv \
        --predictions outputs/group_predictions.jsonl \
        [--split test] [--abstain exclude|wrong] \
        [--human-agreement 0.61]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

LABELS = ("high", "medium", "low")


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval on a binomial proportion. Returns (lo, hi).

    Used instead of the normal approximation because at our sample sizes the
    symmetric interval can dip below 0 or above 1 — an obviously broken thing
    to print next to an accuracy.
    """
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_manifest(path: Path) -> dict[str, dict[str, str]]:
    """clip_id -> {source_video,label,split}; enforces global uniqueness."""
    manifest: dict[str, dict[str, str]] = {}
    source_splits: dict[str, set[str]] = defaultdict(set)
    # utf-8-sig: tolerate BOMs from spreadsheet-exported manifests.
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        needed = {"clip_id", "source_video", "label", "split"}
        missing = needed - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{path}: manifest missing columns {sorted(missing)}")
        for row_no, row in enumerate(reader, 2):
            cid = row["clip_id"].strip()
            if cid in manifest:
                raise SystemExit(f"{path}:{row_no}: duplicate clip_id {cid!r}")
            label = row["label"].strip().lower()
            if label not in LABELS:
                raise SystemExit(f"{path}:{row_no}: bad label {row['label']!r}")
            manifest[cid] = {
                "source_video": row["source_video"].strip(),
                "label": label,
                "split": row["split"].strip().lower(),
            }
            source_splits[manifest[cid]["source_video"]].add(manifest[cid]["split"])
    leaked = {s: v for s, v in source_splits.items() if len(v) > 1}
    if leaked:
        raise SystemExit(
            f"ABORTING: split leakage detected — sources span multiple splits: "
            f"{dict(sorted(leaked.items()))}. Re-run tools.prepare_ouccge."
        )
    return manifest


def load_predictions(path: Path) -> list[dict[str, object]]:
    records = []
    with open(path, encoding="utf-8-sig") as fh:  # -sig: tolerate BOMs
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON ({exc})")
            label = rec.get("label")
            if label is not None:
                label = str(label).strip().lower()
                if label not in LABELS:
                    raise SystemExit(f"{path}:{line_no}: bad label {rec.get('label')!r}")
                rec["label"] = label
            if "clip_id" not in rec:
                raise SystemExit(f"{path}:{line_no}: missing clip_id")
            records.append(rec)
    return records


def evaluate(
    pairs: list[tuple[dict[str, object], dict[str, str]]],
    abstain_policy: str,
) -> dict[str, object]:
    """Core metric computation over (prediction, truth) pairs.

    ``abstain_policy``: 'exclude' drops abstained clips from the denominator
    (and reports their count loudly); 'wrong' charges them as errors — the
    conservative reading, appropriate whenever the deployment question is
    "can we rely on this output existing".
    """
    conf = Counter((str(p["label"]), t["label"]) for p, t in pairs if p["label"] is not None)
    n_abstained = sum(1 for p, _ in pairs if p["label"] is None)
    scored = [(p, t) for p, t in pairs if p["label"] is not None]
    correct = sum(1 for p, t in scored if p["label"] == t["label"])
    n_scored = len(scored)
    if abstain_policy == "wrong":
        charged = n_abstained
    else:
        charged = 0
    acc_denom = n_scored + charged
    acc_num = correct
    lo, hi = wilson_ci(acc_num, acc_denom)

    per_class: dict[str, dict[str, float | int]] = {}
    f1s = []
    for lab in LABELS:
        tp = conf[(lab, lab)]
        fp = sum(conf[(pred, lab)] for pred in LABELS) - tp
        fn = sum(conf[(lab, true)] for true in LABELS) - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[lab] = {
            "support": sum(conf[(pred, lab)] for pred in LABELS),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
        if per_class[lab]["support"]:
            f1s.append(f1)

    return {
        "n_pairs": len(pairs),
        "n_abstained": n_abstained,
        "n_scored": n_scored,
        "accuracy": round(correct / acc_denom, 4) if acc_denom else None,
        "wilson95": (round(lo, 4), round(hi, 4)),
        "macro_f1": round(sum(f1s) / len(f1s), 4) if f1s else None,
        "per_class": per_class,
        "confusion_rows_pred_cols_true": {
            pred: {true: conf[(pred, true)] for true in LABELS} for pred in LABELS
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", required=True, help="manifest.csv from tools.prepare_ouccge")
    parser.add_argument("--predictions", required=True, help="JSONL: {clip_id,label|null}")
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument(
        "--abstain",
        default=None,
        choices=["exclude", "wrong"],
        required=True,
        help=(
            "EXPLICIT policy for abstained clips — there is deliberately no "
            "default: 'exclude' drops them from the denominator, 'wrong' "
            "charges them as errors. The choice is printed in the report."
        ),
    )
    parser.add_argument(
        "--human-agreement",
        type=float,
        default=None,
        help=(
            "Human inter-rater agreement (e.g. Cohen's kappa from "
            "tools/boss_agreement.py or published OUC-CGE figures) to print "
            "alongside machine accuracy. Machine numbers are never reported "
            "alone for this task."
        ),
    )
    args = parser.parse_args(argv)

    manifest = load_manifest(Path(args.manifest))
    preds = load_predictions(Path(args.predictions))

    unknown_ids = sorted({p["clip_id"] for p in preds} - set(manifest))
    if unknown_ids:
        raise SystemExit(
            f"ABORTING: {len(unknown_ids)} predicted clip ids absent from "
            f"manifest, e.g. {unknown_ids[:5]}"
        )
    seen = [p["clip_id"] for p in preds]
    dupes = {cid for cid, n in Counter(seen).items() if n > 1}
    if dupes:
        raise SystemExit(f"ABORTING: duplicate predictions for {sorted(dupes)[:5]}")

    wanted = manifest.keys() if args.split == "all" else {
        cid for cid, row in manifest.items() if row["split"] == args.split
    }
    missing = sorted(wanted - set(seen))
    if missing:
        print(
            f"WARN: {len(missing)} manifest clips have no prediction, "
            f"e.g. {missing[:5]} — they count as neither correct nor wrong; "
            "report coverage alongside accuracy.",
            file=sys.stderr,
        )
    pairs = [(p, manifest[p["clip_id"]]) for p in preds if p["clip_id"] in wanted]
    results = evaluate(pairs, abstain_policy=args.abstain)

    print(f"split={args.split}  abstain={args.abstain}  (policy is part of the result)")
    print(f"pairs={results['n_pairs']}  scored={results['n_scored']}  abstained={results['n_abstained']}")
    acc = results["accuracy"]
    lo, hi = results["wilson95"]
    if acc is not None:
        print(f"accuracy = {acc:.3f}  [Wilson 95% CI {lo:.3f}, {hi:.3f}]")
    else:
        print("accuracy = undefined (nothing scored)")
    if results["macro_f1"] is not None:
        print(f"macro-F1 = {results['macro_f1']:.3f}   (classes: {'/'.join(LABELS)})")
    if args.human_agreement is not None:
        print(
            f"human inter-rater agreement (supplied) = {args.human_agreement:.3f}"
            "  <- machine accuracy must be read against THIS, not against chance"
        )
    else:
        print(
            "NOTE: no --human-agreement supplied. For a 3-class ordinal task "
            "this number must accompany machine accuracy before ANY external claim."
        )
    print("\nper-class (rows=prediction, cols=truth):")
    for pred in LABELS:
        cells = "  ".join(
            f"{true[:4]}={results['confusion_rows_pred_cols_true'][pred][true]:4d}"
            for true in LABELS
        )
        pc = results["per_class"][pred]
        print(f"  {pred:7s} {cells}   P={pc['precision']:.2f} R={pc['recall']:.2f} F1={pc['f1']:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
