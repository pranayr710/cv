"""Fit the window-level engagement model, and measure it against the rules.

The point of this is not an accuracy figure. It is the comparison: on the same
windows, judged against the same human labels, does a model fitted to the
features beat the precedence rule that currently ships? If it does not, the
rules stay, and that is a result worth having.

Two things are done carefully because they decide whether the number means
anything.

**Splitting is by student.** Windows overlap by 50% and a student appears in
many consecutive ones, so a random split puts near-identical windows of the
same person on both sides. Grouping by person_id is the same precaution the
image scorer needed, for the same reason.

**The rule baseline is scored on exactly the windows the model is scored on.**
Comparing a model's test-set accuracy against the rule's accuracy over all
windows would be comparing two different questions.

Abstentions are reported, never silently counted as errors or dropped. Both
the rules and the model are allowed to decline, and a system that answers
less often but is right more often when it does answer is a real trade a
reader should get to see.

    python tools/train_window_model.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Running this file directly puts tools/ on the path, not the project root, so
# `backend` would not import. Matches the bootstrap the other tools use.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.engagement_features import FEATURE_NAMES
from backend.engagement_model import EngagementModel

LABELS = Path("labeling/labels.json")
REPORT = Path("outputs/window_model_report.json")


def load_labelled() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Features, binary target, student id and the rule's verdict per window.

    Returns only windows a human called on or off. "unclear" is excluded from
    training for the reason it exists: it is the rater declining to decide, and
    forcing it into a class would put their uncertainty into the target.
    """
    if not LABELS.exists():
        raise FileNotFoundError(
            f"{LABELS} not found. Label windows with "
            "labeling/label_gui.py first.")
    store = json.loads(LABELS.read_text(encoding="utf-8"))

    x, y, groups, rule = [], [], [], []
    for entry in store.values():
        if entry["label"] not in ("on", "off"):
            continue
        names = entry.get("feature_names") or list(FEATURE_NAMES)
        if names != list(FEATURE_NAMES):
            raise ValueError(
                "labels were made against a different feature set; re-run "
                "labeling/prepare_package.py and relabel, or the weights will "
                "be attached to the wrong columns")
        x.append(entry["features"])
        y.append(1 if entry["label"] == "on" else 0)
        groups.append(entry["person_id"])
        rule.append(entry.get("rule_verdict"))
    return (np.array(x, dtype=float), np.array(y), np.array(groups), rule)


def score(pred: list[str | None], truth: np.ndarray) -> dict[str, float]:
    """Accuracy over the windows a predictor actually decided.

    Abstentions are excluded from accuracy and reported separately, so a
    predictor cannot look better by refusing the hard ones without that
    refusal being visible.
    """
    decided = [(p, t) for p, t in zip(pred, truth) if p is not None]
    if not decided:
        return {"accuracy": float("nan"), "decided": 0,
                "abstained": len(pred), "coverage": 0.0}
    correct = sum(1 for p, t in decided if (p == "on") == bool(t))
    return {
        "accuracy": round(correct / len(decided), 4),
        "decided": len(decided),
        "abstained": len(pred) - len(decided),
        "coverage": round(len(decided) / len(pred), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-labels", type=int, default=60,
                    help="refuse to train below this many usable labels")
    args = ap.parse_args()

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    try:
        x, y, groups, rule = load_labelled()
    except (FileNotFoundError, ValueError) as exc:
        # An expected state, not a crash: nothing has been labelled yet.
        print(f"\n  {exc}")
        return 1
    print(f"{len(y)} usable labels ({y.sum()} on, {len(y) - y.sum()} off) "
          f"across {len(set(groups))} students")
    if len(y) < args.min_labels:
        print(f"\n  too few to train on -- need at least {args.min_labels}.")
        print("  Label more windows with labeling/label_gui.py.")
        return 1
    if len(set(groups)) < 3:
        print("\n  too few distinct students to split by student.")
        return 1

    n_folds = min(5, len(set(groups)))
    oof: list[str | None] = [None] * len(y)
    for tr, te in GroupKFold(n_splits=n_folds).split(x, y, groups):
        scaler = StandardScaler().fit(x[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0)
        clf.fit(scaler.transform(x[tr]), y[tr])
        model = EngagementModel(clf.coef_[0], float(clf.intercept_[0]),
                                list(FEATURE_NAMES), scaler.mean_,
                                scaler.scale_)
        for i in te:
            oof[i] = model.classify(x[i]).label

    learned = score(oof, y)
    rules = score(rule, y)
    print(f"\n  evaluated out-of-fold over {n_folds} folds, split by student\n")
    print(f"  {'':<22} {'accuracy':>9} {'decided':>9} {'abstained':>10}")
    print(f"  {'rule (precedence)':<22} {rules['accuracy']:>9.3f} "
          f"{rules['decided']:>9} {rules['abstained']:>10}")
    print(f"  {'learned model':<22} {learned['accuracy']:>9.3f} "
          f"{learned['decided']:>9} {learned['abstained']:>10}")

    verdict = ("the learned model" if learned["accuracy"] > rules["accuracy"]
               else "the rules")
    print(f"\n  better on this evidence: {verdict}")

    # Final fit on everything, for the model that would actually ship.
    scaler = StandardScaler().fit(x)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(scaler.transform(x), y)
    final = EngagementModel(clf.coef_[0], float(clf.intercept_[0]),
                            list(FEATURE_NAMES), scaler.mean_, scaler.scale_)
    path = final.save()
    print(f"  model written to {path}")

    order = sorted(zip(FEATURE_NAMES, clf.coef_[0]), key=lambda kv: -abs(kv[1]))
    print("\n  what it leans on (standardised weights):")
    for name, weight in order[:8]:
        print(f"    {name:<24} {weight:+.3f}")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({
        "n_labels": len(y),
        "n_students": len(set(groups)),
        "class_balance": {"on": int(y.sum()), "off": int(len(y) - y.sum())},
        "evaluation": "out-of-fold, grouped by student",
        "rule_baseline": rules,
        "learned_model": learned,
        "better": verdict,
        "weights": {n: round(float(w), 4) for n, w in order},
    }, indent=1), encoding="utf-8")
    print(f"\n  report: {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
