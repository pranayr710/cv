"""Stop the sequence model reporting 10/10 for everything.

The aggregate model was calibrated; this one was not, and it shows. Measured
on the footage it was trained on, it puts 23% of windows at exactly 10/10.
Measured on a two-person webcam session it returned 10/10 for **every**
window -- the same score for a student on their phone in 64 frames and a
student attentive in 84, which is not a measurement.

This fits a Platt map on out-of-fold probabilities, grouped by student, so
the correction comes from windows the model being corrected never saw. The
map is written into the checkpoint beside the weights.

**The weights are not touched** and the comparison with the aggregate model
is unaffected: calibration is monotonic, so the ordering every reported
accuracy measured is identical before and after.

    python tools/train/fit_mamba_calibration.py --check
    python tools/train/fit_mamba_calibration.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from backend.engagement_sequence import (
    FRAME_FEATURE_NAMES,
    SEQ_LEN,
    sequences_from_graph,
)
from backend.mamba_score import DEFAULT_MODEL

LABELS = Path("labeling/labels.json")
GRAPH = Path("outputs/final2/live_graph.jsonl")
REPORT = Path("results/mamba_calibration_report.json")


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _spread(p):
    """How many of the ten score bands the probabilities actually reach."""
    score = np.clip(np.round(p * 9) + 1, 1, 10).astype(int)
    return {
        "distinct_scores": int(len(set(score.tolist()))),
        "at_10_of_10": round(float((score == 10).mean()), 4),
        "at_1_of_10": round(float((score == 1).mean()), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold

    from backend.mamba_model import MambaAttentionScorer, MambaConfig
    from tools.train.train_mamba_scorer import train_fold

    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    seqs = sequences_from_graph(GRAPH)
    X, y, g = [], [], []
    for entry in labels.values():
        if entry.get("label") not in ("on", "off"):
            continue
        key = (entry["person_id"], entry["start_ms"])
        if key not in seqs:
            continue
        X.append(seqs[key].steps)
        y.append(1 if entry["label"] == "on" else 0)
        g.append(entry["person_id"])
    X, y, g = np.array(X, dtype="float32"), np.array(y), np.array(g)
    print(f"  {len(y)} labelled windows, {len(set(g))} students, "
          f"{SEQ_LEN} steps x {len(FRAME_FEATURE_NAMES)} features")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    oof = np.zeros(len(y))
    for k, (tr, te) in enumerate(GroupKFold(5).split(X, y, g), 1):
        net = train_fold(X[tr], y[tr], args.epochs, args.seed, device)
        net.eval()
        with torch.no_grad():
            oof[te] = torch.sigmoid(
                net(torch.tensor(X[te], device=device))).cpu().numpy()
        print(f"    fold {k}/5 done ({len(te)} held out)")

    before = _spread(oof)
    platt = LogisticRegression().fit(_logit(oof).reshape(-1, 1), y)
    a = float(platt.coef_[0][0])
    b = float(platt.intercept_[0])
    after = _spread(platt.predict_proba(_logit(oof).reshape(-1, 1))[:, 1])

    acc = float(((oof > 0.5) == (y > 0.5)).mean())
    print(f"\n  {'':<22}{'raw':>10}{'calibrated':>14}")
    for key in ("distinct_scores", "at_10_of_10", "at_1_of_10"):
        print(f"  {key:<22}{before[key]:>10}{after[key]:>14}")
    print(f"\n  out-of-fold accuracy {acc:.3f} (unchanged: calibration is "
          "monotonic)")

    if args.check:
        print("\n  --check: nothing written.")
        return 0

    blob = torch.load(args.model, map_location="cpu", weights_only=False)
    blob["calibration"] = [a, b]
    torch.save(blob, args.model)
    print(f"\n  calibration a={a:.4f} b={b:.4f} -> {args.model}")
    print("  weights untouched.")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({
        "n_windows": int(len(y)),
        "n_students": int(len(set(g))),
        "evaluation": "out-of-fold, grouped by student",
        "out_of_fold_accuracy": round(acc, 4),
        "before": before,
        "after": after,
        "platt": {"a": round(a, 4), "b": round(b, 4)},
        "note": ("Ordering is unchanged, so the comparison with the "
                 "aggregate model stands. Only the reported score moves."),
    }, indent=1), encoding="utf-8")
    print(f"  report -> {REPORT}")
    assert MambaAttentionScorer and MambaConfig  # imported for the fold trainer
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
