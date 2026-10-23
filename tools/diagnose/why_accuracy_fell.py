"""Accuracy fell from 0.650 to 0.564 when the label set grew from 179 to 557.

More data making a model worse is usually one of three things, and they call
for different responses, so this separates them rather than guessing:

**The old number was measured on an easier subset.** The first 200 windows
were labelled in manifest order, which is (student, time) order -- so they came
from the first few students, in the opening minutes. If those windows are
easier, 0.650 was never a figure about the whole session and 0.564 is the
honest one.

**The new labels are noisier.** 405 windows judged in one sitting may carry
more rater fatigue than 200 judged across several. If so, the ceiling has
dropped rather than the model.

**It is seed noise.** 0.086 is large for that, but it costs little to rule out.

The split below is the decisive one: the same model, scored separately on the
windows that were labelled first and the windows that were labelled later.

    python tools/diagnose/why_accuracy_fell.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


LABELS = Path("labeling/labels.json")
MANIFEST = Path("labeling/manifest.json")

#: How many windows were in the label set when 0.650 was measured.
ORIGINAL_N = 200


def main() -> int:
    import warnings

    warnings.filterwarnings("ignore")
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import GroupKFold
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    store = json.loads(LABELS.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    order = {w["window_id"]: i for i, w in enumerate(manifest)}

    x, y, g, early = [], [], [], []
    for wid, entry in store.items():
        if entry["label"] not in ("on", "off"):
            continue
        x.append(entry["features"])
        y.append(1 if entry["label"] == "on" else 0)
        g.append(entry["person_id"])
        early.append(order.get(wid, 10**9) < ORIGINAL_N)

    x = np.array(x, float)
    y = np.array(y)
    g = np.array(g)
    early = np.array(early)

    print(f"{len(y)} usable labels across {len(set(g))} students")
    print(f"  labelled first ({ORIGINAL_N} windows) : {early.sum()}")
    print(f"  labelled later                       : {(~early).sum()}")
    print(f"  class balance, first  : {y[early].mean():.1%} on task")
    print(f"  class balance, later  : {y[~early].mean():.1%} on task")
    print(f"  students in the first : {len(set(g[early]))}")
    print(f"  students in the later : {len(set(g[~early]))}\n")

    def oof(xs, ys, gs, seed):
        pred = np.zeros(len(ys))
        for tr, te in GroupKFold(n_splits=5).split(xs, ys, gs):
            scaler = StandardScaler().fit(xs[tr])
            clf = MLPClassifier((16,), max_iter=3000, random_state=seed)
            clf.fit(scaler.transform(xs[tr]), ys[tr])
            pred[te] = clf.predict(scaler.transform(xs[te]))
        return pred

    print("  out-of-fold over 5 seeds, grouped by student\n")
    print(f"  {'trained and scored on':<34}{'accuracy':>10}{'macro F1':>11}")

    rows = [
        ("all 557 labels", np.ones(len(y), bool)),
        (f"the first {ORIGINAL_N} only", early),
        ("the later ones only", ~early),
    ]
    for name, mask in rows:
        if mask.sum() < 40 or len(set(g[mask])) < 5:
            print(f"  {name:<34}{'too few':>10}")
            continue
        accs, f1s = [], []
        for seed in range(5):
            pred = oof(x[mask], y[mask], g[mask], seed)
            accs.append(accuracy_score(y[mask], pred))
            f1s.append(f1_score(y[mask], pred, average="macro"))
        print(f"  {name:<34}{np.mean(accs):>10.3f}{np.mean(f1s):>11.3f}"
              f"   +/-{np.std(accs):.3f}")

    # Train on everything, then report accuracy separately on each half. This
    # is the question that matters: are the later windows harder, or did they
    # damage the model?
    print("\n  one model trained on all 557, scored on each half separately\n")
    accs_e, accs_l = [], []
    for seed in range(5):
        pred = oof(x, y, g, seed)
        accs_e.append(accuracy_score(y[early], pred[early]))
        accs_l.append(accuracy_score(y[~early], pred[~early]))
    print(f"  {'on the windows labelled first':<34}{np.mean(accs_e):>10.3f}")
    print(f"  {'on the windows labelled later':<34}{np.mean(accs_l):>10.3f}")

    gap = np.mean(accs_e) - np.mean(accs_l)
    print()
    if gap > 0.06:
        print(f"  The earlier windows are {gap:.3f} easier for the same model.")
        print("  The 0.650 reported at 179 labels was measured on that easier")
        print("  subset, so it was never a figure about the whole session.")
        print("  0.564 over all 557 is the honest number.")
    elif gap < -0.06:
        print("  The later windows score HIGHER, so they are not the problem.")
    else:
        print(f"  Both halves score within {abs(gap):.3f}, so the subset is not")
        print("  the explanation. Look at label noise or at the model instead.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
