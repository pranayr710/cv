"""What the labels' own repeatability says about every accuracy in this project.

A re-check of 40 already-judged windows, shown without the earlier answer,
agreed with itself 25 times out of 38 -- 65.8 per cent, Cohen's kappa 0.312.
The flips were symmetric (7 on-to-off, 6 off-to-on), so this is noise rather
than a standard that shifted between sittings.

That number is a ceiling, and it reframes the rest. A model is scored against
these labels, so it cannot agree with them more often than they agree with
themselves. Reporting 0.64 as a shortfall from 1.0 is wrong; the question is
how close it sits to what is achievable.

This also gives a cleaner test set than anything available before. The windows
judged the same way twice are the ones whose labels are trustworthy, and a
model can be measured on those alone -- a small set, but one where a mistake
is the model's rather than the label's.

    python tools/evaluate/label_ceiling.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

warnings.filterwarnings("ignore")

LABELS = Path("labeling/labels.json")
RECHECK = Path("labeling/recheck.json")
REPORT = Path("results/label_ceiling_report.json")


def main() -> int:
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import GroupKFold
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    recheck = json.loads(RECHECK.read_text(encoding="utf-8"))

    pairs = [(w, labels[w]["label"], recheck[w]) for w in recheck
             if labels.get(w, {}).get("label") in ("on", "off")
             and recheck[w] in ("on", "off")]
    agreed = {w for w, a, b in pairs if a == b}
    n = len(pairs)
    rate = len(agreed) / n

    p_a = sum(1 for _, a, _ in pairs if a == "on") / n
    p_b = sum(1 for _, _, b in pairs if b == "on") / n
    expected = p_a * p_b + (1 - p_a) * (1 - p_b)
    kappa = (rate - expected) / (1 - expected)

    print(f"self-agreement over {n} re-judged windows: {rate:.1%} "
          f"(kappa {kappa:.3f})\n")

    # If each judgement is an independent draw with error rate e, agreement is
    # e^2 + (1-e)^2. Inverting gives the per-label error rate, and a perfect
    # model scored against labels that noisy tops out at 1 - e.
    disagree = 1 - rate
    err = (1 - (1 - 2 * disagree) ** 0.5) / 2
    print(f"  implied per-label error rate   : {err:.1%}")
    print(f"  a perfect model would score    : {1 - err:.3f} against "
          "these labels")
    print(f"  two independent raters would   : {rate:.3f} agree\n")

    x, y, g, trusted = [], [], [], []
    for wid, entry in labels.items():
        if entry["label"] not in ("on", "off"):
            continue
        x.append(entry["features"])
        y.append(1 if entry["label"] == "on" else 0)
        g.append(entry["person_id"])
        trusted.append(wid in agreed)
    x, y, g = np.array(x, float), np.array(y), np.array(g)
    trusted = np.array(trusted)

    def oof(seed):
        pred = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=5).split(x, y, g):
            scaler = StandardScaler().fit(x[tr])
            clf = MLPClassifier((16,), max_iter=3000, random_state=seed)
            clf.fit(scaler.transform(x[tr]), y[tr])
            pred[te] = clf.predict(scaler.transform(x[te]))
        return pred

    preds = [oof(s) for s in range(5)]
    all_acc = np.mean([accuracy_score(y, p) for p in preds])
    all_f1 = np.mean([f1_score(y, p, average="macro") for p in preds])

    print(f"  {'scored against':<36}{'accuracy':>10}{'macro F1':>11}")
    print(f"  {'all 557 labels':<36}{all_acc:>10.3f}{all_f1:>11.3f}")

    tr_acc = tr_f1 = float("nan")
    if trusted.sum() >= 15:
        tr_acc = np.mean([accuracy_score(y[trusted], p[trusted])
                          for p in preds])
        tr_f1 = np.mean([f1_score(y[trusted], p[trusted], average="macro")
                         for p in preds])
        print(f"  {'only labels confirmed twice':<36}{tr_acc:>10.3f}"
              f"{tr_f1:>11.3f}   (n = {int(trusted.sum())})")

    print(f"\n  achievable ceiling             : {1 - err:.3f}")
    print(f"  the model reaches              : {all_acc:.3f}  "
          f"({all_acc / (1 - err):.0%} of it)")
    print()
    if kappa < 0.4:
        print("  Kappa below 0.4 is 'fair' agreement at best. These labels do")
        print("  not support a confident accuracy claim, and more of them")
        print("  will not help -- the task needs a clearer rule for what")
        print("  counts as on task before more judgement is collected.")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({
        "n_rechecked": n,
        "self_agreement": round(rate, 4),
        "cohens_kappa": round(kappa, 4),
        "implied_label_error_rate": round(err, 4),
        "achievable_ceiling": round(1 - err, 4),
        "model_accuracy_all_labels": round(float(all_acc), 4),
        "model_macro_f1_all_labels": round(float(all_f1), 4),
        "model_accuracy_trusted_subset":
            None if tr_acc != tr_acc else round(float(tr_acc), 4),
        "n_trusted": int(trusted.sum()),
    }, indent=1), encoding="utf-8")
    print(f"  report -> {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
