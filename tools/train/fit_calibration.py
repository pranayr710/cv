"""Teach the model how sure it is actually entitled to be.

The attention score is a linear map of the model's probability, so a
saturated probability becomes a saturated score. Measured out of fold on 557
hand-labelled windows, grouped by student:

* the model claimed **more than 90% confidence on 65%** of windows, and was
  right on **59.9%** of those;
* windows it scored 0.98 were on task **62%** of the time;
* **58% of all scores came out as exactly 1 or exactly 10**.

That is what produced a student at 1/10 from this model and 10/10 from the
sequence model on the same window. Neither was disagreeing with the other.
Both were saturated, in opposite directions.

This fits a Platt map from the model's logit onto a probability that means
what it says, **nested**: the calibration for each outer fold is fitted on
inner folds of that fold's training data only, so no window is ever used to
calibrate a prediction about itself. The fitted map is then refitted on all
the data and written to the model file.

**The weights are not touched.** The on/off verdict still uses the raw
probability, whose ordering is what every quoted accuracy measured. Only the
1-10 score moves, and only because it was claiming a precision the model
does not have.

    python tools/train/fit_calibration.py --check
    python tools/train/fit_calibration.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from backend.engagement_features import iter_windows
from backend.engagement_model import DEFAULT_MODEL

LABELS = Path("labeling/labels.json")
GRAPH = Path("outputs/final2/live_graph.jsonl")
REPORT = Path("results/calibration_report.json")

#: Confidence bins the calibration error is measured over.
BINS = ((0.0, 0.1), (0.1, 0.3), (0.3, 0.7), (0.7, 0.9), (0.9, 1.01))


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _calibration_error(p, y) -> float:
    """Mean gap between what the model claims and what is true, by bin."""
    total = 0.0
    for lo, hi in BINS:
        sel = (p >= lo) & (p < hi)
        if sel.sum():
            total += abs(y[sel].mean() - p[sel].mean()) * sel.sum()
    return total / len(y)


def load():
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    truth = {(e["person_id"], e["start_ms"]): e["label"]
             for e in labels.values() if e.get("label") in ("on", "off")}
    X, y, g = [], [], []
    for w in iter_windows(GRAPH):
        t = truth.get((w.person_id, w.start_ms))
        if t is None:
            continue
        X.append(list(w.features))
        y.append(1 if t == "on" else 0)
        g.append(w.person_id)
    return np.array(X), np.array(y), np.array(g)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="store_true",
                    help="measure and report, change nothing")
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    X, y, g = load()
    print(f"  {len(y)} labelled windows, {len(set(g))} students")

    raw = np.zeros(len(y))
    cal = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        sc = StandardScaler().fit(X[tr])
        net = MLPClassifier((16,), max_iter=3000, random_state=args.seed)
        net.fit(sc.transform(X[tr]), y[tr])
        raw[te] = net.predict_proba(sc.transform(X[te]))[:, 1]

        # Nested: the map for this fold never sees this fold.
        inner_p, inner_y = [], []
        for itr, ite in GroupKFold(3).split(X[tr], y[tr], g[tr]):
            s2 = StandardScaler().fit(X[tr][itr])
            n2 = MLPClassifier((16,), max_iter=3000, random_state=args.seed)
            n2.fit(s2.transform(X[tr][itr]), y[tr][itr])
            inner_p += list(n2.predict_proba(s2.transform(X[tr][ite]))[:, 1])
            inner_y += list(y[tr][ite])
        platt = LogisticRegression().fit(
            _logit(np.array(inner_p)).reshape(-1, 1), inner_y)
        cal[te] = platt.predict_proba(_logit(raw[te]).reshape(-1, 1))[:, 1]

    def summarise(name, p):
        score = np.clip(np.round(p * 9) + 1, 1, 10).astype(int)
        return {
            "name": name,
            "accuracy": round(float(((p > .5) == (y > .5)).mean()), 4),
            "claims_over_90pct_sure": round(float(((p < .1) | (p > .9)).mean()), 4),
            "calibration_error": round(_calibration_error(p, y), 4),
            "scores_pinned_at_1_or_10": round(
                float(((score == 1) | (score == 10)).mean()), 4),
        }

    before, after = summarise("raw", raw), summarise("calibrated", cal)
    print(f"\n  {'':<26}{'raw':>10}{'calibrated':>14}")
    for k in ("accuracy", "claims_over_90pct_sure", "calibration_error",
              "scores_pinned_at_1_or_10"):
        print(f"  {k:<26}{before[k]:>10.3f}{after[k]:>14.3f}")

    print("\n  out-of-fold reliability, raw model")
    print(f"  {'bin':<14}{'n':>6}{'claims':>10}{'is':>10}")
    for lo, hi in BINS:
        sel = (raw >= lo) & (raw < hi)
        if sel.sum():
            print(f"  {f'{lo:.1f}-{hi:.2f}':<14}{sel.sum():>6}"
                  f"{raw[sel].mean():>10.3f}{y[sel].mean():>10.3f}")

    if args.check:
        print("\n  --check: nothing written.")
        return 0

    # The shipped map, refitted on everything now that it has been measured.
    final = LogisticRegression().fit(_logit(raw).reshape(-1, 1), y)
    a = float(final.coef_[0][0])
    b = float(final.intercept_[0])

    payload = json.loads(args.model.read_text(encoding="utf-8"))
    payload["calibration"] = [a, b]
    args.model.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"\n  calibration a={a:.4f} b={b:.4f} -> {args.model}")
    print("  weights untouched; the on/off verdict is unchanged.")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({
        "n_windows": int(len(y)),
        "n_students": int(len(set(g))),
        "evaluation": "out-of-fold, grouped by student; calibration nested",
        "before": before,
        "after": after,
        "platt": {"a": round(a, 4), "b": round(b, 4)},
        "note": ("The verdict still uses the raw probability, whose ordering "
                 "every quoted accuracy measured. Only the 1-10 score moves."),
    }, indent=1), encoding="utf-8")
    print(f"  report -> {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
