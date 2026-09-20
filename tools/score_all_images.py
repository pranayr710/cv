"""Score every image 1-10, and test whether a direct regressor beats calibration.

Two questions, answered together.

**Can the 350 hand-scored images label the other 12,232?** Yes, and no
propagation step is needed: a model trained on the folder labels and calibrated
against the 350 already maps any image to a 1-10 score. This writes that score
out for the whole set.

**Should the generated scores then be trained on?** No. A model fitted to its
own predictions learns nothing it did not already believe; it sharpens its
biases, including the wrong ones, and reports the resulting confidence as
progress. The generated scores are an output, not training data.

**What IS worth testing** is whether the 350 ordinal scores can train a scorer
directly -- a ridge regression on the same frozen features, fitted to the human
1-5 values rather than to the binary folder labels. That is a different model
answering the same question, so the two can be compared honestly. Both are
evaluated out-of-fold over near-duplicate groups.

    python tools/score_all_images.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from tools.train_engagement_scorer import (
    CALIB_MANIFEST,
    DATA,
    SCORES,
    duplicate_groups,
    list_images,
    load_or_build,
)

OUT_CSV = Path("outputs/image_engagement_scores.csv")
OUT_REPORT = Path("outputs/score_method_comparison.json")


def to_ten(values: np.ndarray) -> np.ndarray:
    """Map a 1-5 judgement onto the 1-10 scale, at the very end."""
    return np.clip(np.round((values - 1) / 4 * 9 + 1), 1, 10)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.parse_args()

    from scipy.stats import spearmanr
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.model_selection import GroupKFold

    paths = list_images(DATA)
    labels = np.array([1 if "Engagement" in p.parts and "Disengagement"
                       not in p.parts else 0 for p in paths])
    features = load_or_build(paths, refresh=False)

    calib = json.loads(CALIB_MANIFEST.read_text(encoding="utf-8"))
    human = json.loads(SCORES.read_text(encoding="utf-8"))
    by_source = {Path(e["source"]).as_posix(): e["file"] for e in calib}

    idx, y_human = [], []
    for i, p in enumerate(paths):
        name = by_source.get(p.as_posix())
        if name and name in human:
            idx.append(i)
            y_human.append(human[name]["score_5"])
    idx = np.array(idx)
    y_human = np.array(y_human, dtype=float)
    scored = np.zeros(len(paths), dtype=bool)
    scored[idx] = True
    print(f"{len(paths)} images, {len(idx)} with a human score\n")

    groups = duplicate_groups([paths[i] for i in idx])
    n_folds = min(5, len(set(groups)))
    x_rest, y_rest = features[~scored], labels[~scored]

    # --- method A: binary classifier, then isotonic calibration ----------- #
    # --- method B: ridge fitted straight to the human 1-5 values ---------- #
    pred_a = np.zeros(len(idx))
    pred_b = np.zeros(len(idx))
    for tr, te in GroupKFold(n_splits=n_folds).split(idx, y_human, groups):
        clf = LogisticRegression(max_iter=2000).fit(
            np.vstack([x_rest, features[idx[tr]]]),
            np.concatenate([y_rest, labels[idx[tr]]]))
        p_tr = clf.predict_proba(features[idx[tr]])[:, 1]
        iso = IsotonicRegression(y_min=1, y_max=5, out_of_bounds="clip")
        iso.fit(p_tr, y_human[tr])
        pred_a[te] = iso.predict(clf.predict_proba(features[idx[te]])[:, 1])

        ridge = Ridge(alpha=10.0).fit(features[idx[tr]], y_human[tr])
        pred_b[te] = np.clip(ridge.predict(features[idx[te]]), 1, 5)

    results = {}
    for name, pred in (("calibrated classifier", pred_a),
                       ("direct ridge on 350 scores", pred_b)):
        rho, _ = spearmanr(pred, y_human)
        mae = float(np.mean(np.abs(pred - y_human)))
        results[name] = {"spearman_rho": round(float(rho), 4),
                         "mae_points_of_5": round(mae, 3)}
        print(f"  {name:<28} rho {rho:.3f}   MAE {mae:.2f}")

    better = max(results, key=lambda k: results[k]["spearman_rho"])
    print(f"\n  better on this evidence: {better}")

    # --- score every image with the stronger method ----------------------- #
    if better.startswith("calibrated"):
        clf = LogisticRegression(max_iter=2000).fit(features, labels)
        p_all = clf.predict_proba(features)[:, 1]
        iso = IsotonicRegression(y_min=1, y_max=5, out_of_bounds="clip")
        iso.fit(clf.predict_proba(features[idx])[:, 1], y_human)
        score_5 = iso.predict(p_all)
    else:
        ridge = Ridge(alpha=10.0).fit(features[idx], y_human)
        score_5 = np.clip(ridge.predict(features), 1, 5)
    score_10 = to_ten(score_5)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["path", "folder_label", "score_5", "score_10",
                         "human_scored"])
        for i, p in enumerate(paths):
            writer.writerow([
                p.as_posix(),
                "Engagement" if labels[i] else "Disengagement",
                round(float(score_5[i]), 3),
                int(score_10[i]),
                int(scored[i]),
            ])

    hist = {int(v): int((score_10 == v).sum())
            for v in np.unique(score_10)}
    print(f"\n  wrote {len(paths)} scores to {OUT_CSV}")
    print("  1-10 distribution:")
    for value in sorted(hist):
        bar = "#" * max(1, int(40 * hist[value] / max(hist.values())))
        print(f"    {value:>2}  {hist[value]:>5}  {bar}")

    OUT_REPORT.write_text(json.dumps({
        "n_images": len(paths),
        "n_human_scored": len(idx),
        "evaluation": "out-of-fold over near-duplicate groups",
        "methods": results,
        "used_for_scoring": better,
        "note": "Generated scores are an output, not training data. Refitting "
                "on them would sharpen the model's existing biases and report "
                "the result as progress.",
    }, indent=1), encoding="utf-8")
    print(f"  report: {OUT_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
