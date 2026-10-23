"""Evaluate the attention model against labels known to be repeatable.

Every accuracy figure in this project was scored against single-pass labels
that, re-judged blind, agree with themselves 64.6 per cent of the time
(kappa 0.261, n = 127). A model measured against a target that noisy is
penalised for being right whenever the label is wrong, which is how 557
labels produced a *lower* number than 179 did.

The windows judged the same way twice are a different kind of test set. They
are the only labels here with evidence of being repeatable, so this trains on
everything else and reports performance on those alone.

Two things are reported rather than one, because on this subset they disagree:

**Accuracy flatters a majority guesser.** The trusted windows are 67 per cent
on task, so answering "on task" every time scores 0.671 while identifying no
disengaged student at all. Against that, the model's advantage is not
statistically significant.

**Macro F1 is the honest metric here.** It averages per class, so never
finding a disengaged student costs what it should. The majority guesser scores
0.401; the model scores 0.724.

Which matters depends on what the system is for, and this one exists to find
students who are disengaged -- the minority class by construction.

    python tools/evaluate/trusted_eval.py
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

warnings.filterwarnings("ignore")

LABELS = Path("labeling/labels.json")
RECHECK = Path("labeling/recheck.json")
REPORT = Path("outputs/trusted_eval_report.json")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    from scipy import stats
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        recall_score,
    )
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    if not RECHECK.exists():
        print(f"{RECHECK} not found. Run labeling/recheck.py first -- without "
              "it\nthere is no label here known to be repeatable.")
        return 1

    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    recheck = json.loads(RECHECK.read_text(encoding="utf-8"))

    def decided(wid: str) -> bool:
        return (labels.get(wid, {}).get("label") in ("on", "off")
                and recheck.get(wid) in ("on", "off"))

    trusted = [w for w in recheck
               if decided(w) and labels[w]["label"] == recheck[w]]
    conflicted = {w for w in recheck
                  if decided(w) and labels[w]["label"] != recheck[w]}

    def vec(ws):
        return (np.array([labels[w]["features"] for w in ws], float),
                np.array([1 if labels[w]["label"] == "on" else 0 for w in ws]))

    if len(trusted) < 30:
        print(f"Only {len(trusted)} trusted labels -- too few to measure on.")
        return 1

    x_test, y_test = vec(trusted)
    # Conflicted windows are excluded from training as well: a label the rater
    # contradicted is not evidence, and keeping it would teach the model the
    # noise this evaluation is trying to see past.
    train = [w for w in labels if labels[w]["label"] in ("on", "off")
             and w not in set(trusted) and w not in conflicted]

    print(f"  re-judged      : {sum(1 for w in recheck if decided(w))}")
    print(f"  agreed twice   : {len(trusted)}  (the test set)")
    print(f"  contradicted   : {len(conflicted)}  (dropped from training)")
    print(f"  trained on     : {len(train)}")
    print(f"  test set is {y_test.mean():.1%} on task\n")

    preds = []
    for seed in range(args.seeds):
        x, y = vec(train)
        scaler = StandardScaler().fit(x)
        clf = MLPClassifier((16,), max_iter=3000, random_state=seed)
        clf.fit(scaler.transform(x), y)
        preds.append(clf.predict(scaler.transform(x_test)))

    always_on = np.ones(len(y_test), int)
    acc = float(np.mean([accuracy_score(y_test, p) for p in preds]))
    f1 = float(np.mean([f1_score(y_test, p, average="macro") for p in preds]))
    off_recall = float(np.mean(
        [recall_score(y_test, p, pos_label=0) for p in preds]))

    print(f"  {'':<28}{'accuracy':>10}{'macro F1':>11}{'off-task recall':>18}")
    print(f"  {'always answer on task':<28}"
          f"{accuracy_score(y_test, always_on):>10.3f}"
          f"{f1_score(y_test, always_on, average='macro'):>11.3f}"
          f"{0.0:>18.3f}")
    print(f"  {'the model':<28}{acc:>10.3f}{f1:>11.3f}{off_recall:>18.3f}")

    cm = confusion_matrix(y_test, preds[0])
    print("\n  confusion, seed 0        pred off   pred on")
    print(f"    actually off {'':<8}{cm[0, 0]:>8}{cm[0, 1]:>11}")
    print(f"    actually on  {'':<8}{cm[1, 0]:>8}{cm[1, 1]:>11}")

    lo, hi = stats.binomtest(
        round(acc * len(y_test)), len(y_test)).proportion_ci()
    base = max(y_test.mean(), 1 - y_test.mean())
    p_acc = stats.binomtest(round(acc * len(y_test)), len(y_test), p=base,
                            alternative="greater").pvalue
    print(f"\n  accuracy 95% CI [{lo:.3f}, {hi:.3f}]")
    print(f"  against the {base:.3f} majority baseline: p = {p_acc:.3f}"
          f"  ({'significant' if p_acc < 0.05 else 'not significant'})")

    print("\n  Accuracy alone does not separate the model from guessing on a")
    print(f"  {y_test.mean():.0%}/{1 - y_test.mean():.0%} split. Macro F1 does: "
          f"{f1:.3f} against "
          f"{f1_score(y_test, always_on, average='macro'):.3f}, because the")
    print(f"  model finds {cm[0, 0]} of {cm[0].sum()} disengaged students and a "
          "majority")
    print("  guesser finds none. That is the case the system exists for.")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({
        "n_trusted": len(trusted),
        "n_conflicted": len(conflicted),
        "n_train": len(train),
        "test_on_task_rate": round(float(y_test.mean()), 4),
        "model_accuracy": round(acc, 4),
        "model_macro_f1": round(f1, 4),
        "model_off_task_recall": round(off_recall, 4),
        "baseline_accuracy": round(float(base), 4),
        "baseline_macro_f1": round(
            float(f1_score(y_test, always_on, average="macro")), 4),
        "accuracy_ci": [round(float(lo), 4), round(float(hi), 4)],
        "accuracy_vs_baseline_p": round(float(p_acc), 4),
        "note": "Trusted = judged identically on two independent passes. "
                "Accuracy is not significant against the majority class on "
                "this split; macro F1 and off-task recall are what separate "
                "the model from guessing.",
    }, indent=1), encoding="utf-8")
    print(f"\n  report -> {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
