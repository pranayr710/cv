"""Test whether something beats logistic regression on the window features.

Logistic regression was chosen on principle -- 179 samples over 16 correlated
features will fit the rater rather than the task if given enough capacity --
but a principle is not a measurement. This runs the alternatives under exactly
the protocol the shipped model is judged by: out-of-fold, folds split by
student, so overlapping windows of one person cannot straddle the split.

Two numbers are reported per candidate, because accuracy alone hides the thing
that matters at this sample size. The mean is what the model scores; the spread
across folds is how much that mean can be trusted. A model at 0.63 with a fold
range of 0.30 has not learned something a model at 0.60 with a range of 0.08
missed -- it has found a split it happens to suit.

    python tools/compare_classifiers.py
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.train_window_model import load_labelled

REPORT = Path("outputs/classifier_comparison.json")


def candidates() -> dict:
    """The alternatives worth trying at this sample size.

    Deep networks are excluded deliberately. With 179 examples, a network has
    more parameters than data and its score would describe the initialisation.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        GradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    def scaled(model):
        return make_pipeline(StandardScaler(), model)

    return {
        "logistic regression (shipped)":
            scaled(LogisticRegression(max_iter=2000, C=1.0)),
        "logistic regression, C=0.1":
            scaled(LogisticRegression(max_iter=2000, C=0.1)),
        "linear SVM":
            scaled(SVC(kernel="linear", probability=True)),
        "RBF SVM":
            scaled(SVC(kernel="rbf", probability=True)),
        "linear discriminant":
            scaled(LinearDiscriminantAnalysis()),
        "gaussian naive bayes":
            scaled(GaussianNB()),
        "k-nearest neighbours (k=9)":
            scaled(KNeighborsClassifier(n_neighbors=9)),
        "random forest":
            RandomForestClassifier(n_estimators=400, random_state=0),
        "extra trees":
            ExtraTreesClassifier(n_estimators=400, random_state=0),
        "gradient boosting":
            GradientBoostingClassifier(random_state=0),
        "small MLP (16)":
            scaled(MLPClassifier(hidden_layer_sizes=(16,), max_iter=3000,
                                 random_state=0)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.parse_args()

    from sklearn.model_selection import GroupKFold

    try:
        x, y, groups, rule = load_labelled()
    except (FileNotFoundError, ValueError) as exc:
        print(f"\n  {exc}")
        return 1

    n_folds = min(5, len(set(groups)))
    print(f"{len(y)} labels, {len(set(groups))} students, "
          f"{n_folds} folds split by student\n")

    # The rule, on the windows it decided, for reference.
    decided = [(p, t) for p, t in zip(rule, y) if p is not None]
    rule_acc = (sum(1 for p, t in decided if (p == "on") == bool(t))
                / len(decided)) if decided else float("nan")

    splits = list(GroupKFold(n_splits=n_folds).split(x, y, groups))
    results = {}
    for name, model in candidates().items():
        folds = []
        for tr, te in splits:
            model.fit(x[tr], y[tr])
            folds.append(float((model.predict(x[te]) == y[te]).mean()))
        results[name] = {
            "mean": round(statistics.fmean(folds), 4),
            "worst_fold": round(min(folds), 4),
            "best_fold": round(max(folds), 4),
            "spread": round(max(folds) - min(folds), 4),
        }

    order = sorted(results, key=lambda k: -results[k]["mean"])
    print(f"  {'model':<32} {'mean':>7} {'worst':>7} {'best':>7} {'spread':>8}")
    print(f"  {'-' * 32} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 8}")
    for name in order:
        r = results[name]
        print(f"  {name:<32} {r['mean']:>7.3f} {r['worst_fold']:>7.3f} "
              f"{r['best_fold']:>7.3f} {r['spread']:>8.3f}")
    print(f"\n  {'rule baseline (reference)':<32} {rule_acc:>7.3f}")

    best = order[0]
    shipped = "logistic regression (shipped)"
    gain = results[best]["mean"] - results[shipped]["mean"]
    print(f"\n  best: {best}  ({gain:+.3f} against the shipped model)")
    if gain < 0.02:
        print("  Within noise at this sample size. Not worth switching.")
    elif results[best]["spread"] > 2 * results[shipped]["spread"]:
        print("  Higher mean but a much wider fold spread -- the gain may be "
              "one\n  favourable split rather than a better model.")
    else:
        print("  A real improvement. Worth switching, with the caveat that 179"
              "\n  labels cannot separate close candidates confidently.")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({
        "n_labels": len(y),
        "n_students": len(set(groups)),
        "protocol": "out-of-fold, grouped by student",
        "rule_baseline": round(float(rule_acc), 4),
        "models": results,
        "best": best,
    }, indent=1), encoding="utf-8")
    print(f"\n  report: {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
