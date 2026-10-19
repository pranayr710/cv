"""Do the early and late labels encode the same judgement?

A model trained on all 557 labels scores 0.500 -- chance -- on the 178 that
were labelled first, while scoring 0.586 on the 379 labelled later. Trained on
the early ones alone it reaches 0.601 on them. So the early labels are
learnable, and they are learnable *away* by the later ones.

That is the signature of rater drift: the same person applying a different
standard in a second sitting. It matters more than any accuracy figure,
because every number in the project is measured against these labels. If they
disagree with each other, the ceiling is not where it appears to be.

The decisive test is cross-prediction. Train only on the early labels and
predict the later ones, then the reverse. If each set predicts itself but not
the other, they are describing different functions.

Note on what this cannot do: students appear in both sets, so cross-prediction
here carries student leakage, which *inflates* the score. A low figure is
therefore conservative -- the true disagreement is at least this large.

    python tools/check_label_drift.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LABELS = Path("labeling/labels.json")
MANIFEST = Path("labeling/manifest.json")
ORIGINAL_N = 200


def main() -> int:
    import warnings

    warnings.filterwarnings("ignore")
    from sklearn.metrics import accuracy_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    store = json.loads(LABELS.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    order = {w["window_id"]: i for i, w in enumerate(manifest)}

    x, y, g, early, rule = [], [], [], [], []
    for wid, entry in store.items():
        if entry["label"] not in ("on", "off"):
            continue
        x.append(entry["features"])
        y.append(1 if entry["label"] == "on" else 0)
        g.append(entry["person_id"])
        early.append(order.get(wid, 10**9) < ORIGINAL_N)
        rule.append(entry.get("rule_verdict"))

    x, y, g = np.array(x, float), np.array(y), np.array(g)
    early = np.array(early)

    def fit_predict(train_mask, test_mask, seed):
        scaler = StandardScaler().fit(x[train_mask])
        clf = MLPClassifier((16,), max_iter=3000, random_state=seed)
        clf.fit(scaler.transform(x[train_mask]), y[train_mask])
        return clf.predict(scaler.transform(x[test_mask]))

    print(f"{len(y)} usable labels   early {early.sum()}   "
          f"late {(~early).sum()}\n")
    print("  cross-prediction (student leakage present, so these are"
          "\n  optimistic -- the real disagreement is at least this large)\n")
    print(f"  {'trained on':<16}{'scored on':<16}{'accuracy':>10}")
    for tr_name, tr, te_name, te in (
            ("early", early, "early (self)", early),
            ("early", early, "late", ~early),
            ("late", ~early, "late (self)", ~early),
            ("late", ~early, "early", early)):
        accs = [accuracy_score(y[te], fit_predict(tr, te, s))
                for s in range(5)]
        flag = "  <-- chance" if abs(np.mean(accs) - 0.5) < 0.03 else ""
        print(f"  {tr_name:<16}{te_name:<16}{np.mean(accs):>10.3f}{flag}")

    # The rule is a fixed function, so its agreement with each half is an
    # independent read on whether the labelling standard moved.
    print("\n  the rule is unchanged code, so its agreement with each half")
    print("  measures the labels rather than the model\n")
    for name, mask in (("early labels", early), ("late labels", ~early)):
        pairs = [(r, t) for r, t, m in zip(rule, y, mask)
                 if m and r in ("on", "off")]
        if pairs:
            agree = sum(1 for r, t in pairs if (r == "on") == bool(t))
            print(f"  rule agrees with {name:<16}{agree / len(pairs):>8.3f}"
                  f"   (n = {len(pairs)})")

    print("\n  on-task rate by half")
    for name, mask in (("early", early), ("late", ~early)):
        print(f"    {name:<8}{y[mask].mean():.1%}")

    # Students labelled in both sittings are where drift shows up cleanly.
    both = sorted(set(g[early]) & set(g[~early]))
    print(f"\n  {len(both)} students appear in both sittings")
    shifts = []
    for student in both:
        e = y[(g == student) & early]
        late_vals = y[(g == student) & ~early]
        if len(e) >= 5 and len(late_vals) >= 5:
            shifts.append((student, e.mean(), late_vals.mean(),
                           late_vals.mean() - e.mean()))
    if shifts:
        print(f"  {'student':>9}{'early on%':>11}{'late on%':>10}{'shift':>9}")
        for student, a, b, d in sorted(shifts, key=lambda r: -abs(r[3]))[:8]:
            print(f"  {student:>9}{a:>11.1%}{b:>10.1%}{d:>+9.1%}")
        mean_shift = float(np.mean([abs(d) for *_, d in shifts]))
        print(f"\n  mean absolute shift per student: {mean_shift:.1%}")
        if mean_shift > 0.20:
            print("  Large. The same students were judged differently between"
                  "\n  the two sittings, which is rater drift rather than the"
                  "\n  windows themselves being different.")
    print("\n  counts:", dict(Counter(
        "early" if e else "late" for e in early)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
