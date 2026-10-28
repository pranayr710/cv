"""Fit a second attention model that does not use peer context.

The shipping model reads 21 features, three of which describe the room: how
many peers are in frame, and what they are doing. Those are the features
that make it refuse a small room -- `peer_count` is n/(n+8), and its fitted
minimum of 0.333 means four peers, so below five people in frame the model
has never seen anything like the vector it is being handed.

That is the correct refusal for *that* model. It is not a reason the system
has to go blank. The other eighteen features are about the student alone and
are perfectly in range at any room size, so a model fitted on those works
everywhere.

Measured out of fold, grouped by student, on the 82 twice-confirmed labels:

    full (21 features)   accuracy 0.634   macro F1 0.620
    solo (18, no peers)  accuracy 0.598   macro F1 0.587

So peer context is worth about 0.03 macro F1, which is why the full model
remains the one used wherever it applies. The solo model is the fallback,
not the replacement, and the report says which one produced a score.

Writes models/window_engagement_model_solo.json, with its fitted range and
its calibration, so it arrives with the same two protections the full model
has rather than needing two more commands.

    python tools/train/train_solo_model.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from backend.engagement_features import (
    BASE_FEATURE_NAMES,
    BODY_FEATURE_NAMES,
    PEER_FEATURE_NAMES,
    iter_windows,
)
from backend.engagement_model import MLPEngagementModel

LABELS = Path("labeling/labels.json")
RECHECK = Path("labeling/recheck.json")
GRAPH = Path("outputs/final2/live_graph.jsonl")
OUT = Path("models/window_engagement_model_solo.json")
REPORT = Path("results/solo_model_report.json")

ALL_NAMES = (list(BASE_FEATURE_NAMES) + list(BODY_FEATURE_NAMES)
             + list(PEER_FEATURE_NAMES))
SOLO_NAMES = [n for n in ALL_NAMES if n not in set(PEER_FEATURE_NAMES)]


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def load():
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    truth = {(e["person_id"], e["start_ms"]): e["label"]
             for e in labels.values() if e.get("label") in ("on", "off")}
    trusted = set()
    if RECHECK.exists():
        for key, value in json.loads(
                RECHECK.read_text(encoding="utf-8")).items():
            entry = labels.get(key)
            if not entry:
                continue
            again = value.get("label") if isinstance(value, dict) else value
            if again == entry.get("label"):
                trusted.add((entry["person_id"], entry["start_ms"]))

    keep = [ALL_NAMES.index(n) for n in SOLO_NAMES]
    X, y, g, t = [], [], [], []
    for w in iter_windows(GRAPH):
        label = truth.get((w.person_id, w.start_ms))
        if label is None:
            continue
        X.append([w.features[i] for i in keep])
        y.append(1 if label == "on" else 0)
        g.append(w.person_id)
        t.append((w.person_id, w.start_ms) in trusted)
    return np.array(X), np.array(y), np.array(g), np.array(t)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, recall_score
    from sklearn.model_selection import GroupKFold
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    X, y, g, trusted = load()
    print(f"  {len(y)} labelled windows, {len(set(g))} students, "
          f"{len(SOLO_NAMES)} features (peers dropped)")

    # Out of fold first, because a number measured after fitting on
    # everything is not a measurement.
    oof = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        sc = StandardScaler().fit(X[tr])
        net = MLPClassifier((16,), max_iter=3000, random_state=args.seed)
        net.fit(sc.transform(X[tr]), y[tr])
        oof[te] = net.predict_proba(sc.transform(X[te]))[:, 1]

    def measure(mask):
        p, t = oof[mask], y[mask]
        pred = (p > 0.5).astype(int)
        return {
            "n": int(mask.sum()),
            "accuracy": round(float((pred == t).mean()), 4),
            "macro_f1": round(float(f1_score(t, pred, average="macro")), 4),
            "off_task_recall": round(
                float(recall_score(t, pred, pos_label=0)), 4),
        }

    scores = {"all": measure(np.ones(len(y), bool)),
              "trusted": measure(trusted)}
    for name, s in scores.items():
        print(f"  {name:<8} n={s['n']:<5} acc {s['accuracy']:.3f}   "
              f"macro F1 {s['macro_f1']:.3f}   "
              f"off-task recall {s['off_task_recall']:.3f}")

    # The shipped model, refit on everything now it has been measured.
    scaler = StandardScaler().fit(X)
    net = MLPClassifier((16,), max_iter=3000, random_state=args.seed)
    net.fit(scaler.transform(X), y)
    model = MLPEngagementModel(
        net.coefs_[0], net.intercepts_[0], net.coefs_[1], net.intercepts_[1],
        SOLO_NAMES, scaler.mean_, scaler.scale_,
        X.min(axis=0).tolist(), X.max(axis=0).tolist())

    # Same calibration the full model carries, fitted on the out-of-fold
    # probabilities so it corrects real overconfidence rather than the
    # in-sample kind, which is always zero.
    platt = LogisticRegression().fit(_logit(oof).reshape(-1, 1), y)
    model.calibration = (float(platt.coef_[0][0]), float(platt.intercept_[0]))

    model.save(OUT)
    print(f"\n  -> {OUT}  (range and calibration included)")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({
        "n_windows": int(len(y)),
        "n_students": int(len(set(g))),
        "features": SOLO_NAMES,
        "dropped": list(PEER_FEATURE_NAMES),
        "evaluation": "out-of-fold, grouped by student",
        "scores": scores,
        "calibration": [round(c, 4) for c in model.calibration],
        "role": ("fallback for rooms too small for the full model, which "
                 "needs five or more people in frame; the full model is "
                 "used wherever it applies"),
    }, indent=1), encoding="utf-8")
    print(f"  report -> {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
