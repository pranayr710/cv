"""Is the label-spreading gain real, or an artefact of one fold assignment?

Label spreading is deterministic and GroupKFold does not shuffle, so running it
under several seeds returned the identical number five times. That is not
stability -- it is the same computation repeated, and it leaves the gain of
+0.036 over the baseline with no error bar to judge it against.

So the randomness is moved to where it belongs: which students land in which
fold. Permuting that gives a genuine spread for both methods, measured on the
same splits, and the sensitivity sweep over n_neighbors says whether the result
survives a different choice of graph.

One property is checked rather than assumed: label spreading is
*transductive*. The test windows must be present in the graph when it is fit,
unlabelled. That is legitimate -- their labels are never used -- but it means
the method cannot score a new student without being refit, which is a real
deployment constraint and not a detail.

    python tools/diagnose/check_label_spreading.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tools.train.semi_supervised import fit_mlp, load, predict_mlp

REPORT = Path("outputs/label_spreading_check.json")


def permuted_folds(groups: np.ndarray, n_folds: int,
                   rng: np.random.Generator) -> list[tuple[np.ndarray, np.ndarray]]:
    """Group-aware folds with the students shuffled between them.

    GroupKFold assigns deterministically by group size, so every call returns
    the same partition. Shuffling the students first is what turns repeated
    runs into independent draws.
    """
    students = np.array(sorted(set(groups)))
    rng.shuffle(students)
    buckets = np.array_split(students, n_folds)
    folds = []
    for held in buckets:
        te = np.where(np.isin(groups, held))[0]
        tr = np.where(~np.isin(groups, held))[0]
        if len(te) and len(tr) and len(set(groups[tr])) >= 2:
            folds.append((tr, te))
    return folds


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeats", type=int, default=12)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.semi_supervised import LabelSpreading

    xl, yl, gl, _sl, xu, _su = load()
    print(f"{len(yl)} labelled, {len(xu)} unlabelled, {len(set(gl))} students")
    print(f"{args.repeats} independent fold permutations\n")

    base_a, base_f, ls_a, ls_f = [], [], [], []
    for rep in range(args.repeats):
        rng = np.random.default_rng(rep)
        folds = permuted_folds(gl, args.folds, rng)
        p_base = np.full(len(yl), np.nan)
        p_ls = np.full(len(yl), np.nan)

        for tr, te in folds:
            p_base[te] = predict_mlp(fit_mlp(xl[tr], yl[tr], rep), xl[te])

            x_all = np.vstack([xl[tr], xl[te], xu])
            y_all = np.concatenate([yl[tr], np.full(len(te) + len(xu), -1)])
            scaler = StandardScaler().fit(x_all)
            ls = LabelSpreading(kernel="knn", n_neighbors=7, alpha=0.2)
            ls.fit(scaler.transform(x_all), y_all)
            p_ls[te] = ls.label_distributions_[len(tr):len(tr) + len(te), 1]

        seen = ~np.isnan(p_base)
        for store_a, store_f, p in ((base_a, base_f, p_base),
                                    (ls_a, ls_f, p_ls)):
            pred = (p[seen] >= 0.5).astype(int)
            store_a.append(accuracy_score(yl[seen], pred))
            store_f.append(f1_score(yl[seen], pred, average="macro"))
        print(f"  permutation {rep:>2}: baseline {base_a[-1]:.3f}   "
              f"label spreading {ls_a[-1]:.3f}")

    base_a, ls_a = np.array(base_a), np.array(ls_a)
    base_f, ls_f = np.array(base_f), np.array(ls_f)
    print(f"\n  {'':<24}{'accuracy':>18}{'macro F1':>18}")
    print(f"  {'baseline (labels only)':<24}"
          f"{base_a.mean():>11.3f} +/-{base_a.std():<4.3f}"
          f"{base_f.mean():>11.3f} +/-{base_f.std():<4.3f}")
    print(f"  {'label spreading':<24}"
          f"{ls_a.mean():>11.3f} +/-{ls_a.std():<4.3f}"
          f"{ls_f.mean():>11.3f} +/-{ls_f.std():<4.3f}")

    diff = ls_a - base_a
    # Paired across permutations, because both methods saw the same folds.
    wins = int((diff > 0).sum())
    se = diff.std(ddof=1) / np.sqrt(len(diff)) if len(diff) > 1 else float("nan")
    t = diff.mean() / se if se else float("nan")
    print(f"\n  paired difference {diff.mean():+.3f} +/-{diff.std():.3f}"
          f"   (t = {t:.2f}, wins {wins}/{len(diff)})")
    verdict = ("a real gain" if abs(t) > 2.2 and wins >= 0.75 * len(diff)
               else "within noise")
    print(f"  verdict: {verdict}")

    print("\n  sensitivity to the graph (n_neighbors):")
    for k in (3, 5, 7, 11, 15, 21):
        accs = []
        for rep in range(4):
            rng = np.random.default_rng(rep)
            p = np.full(len(yl), np.nan)
            for tr, te in permuted_folds(gl, args.folds, rng):
                x_all = np.vstack([xl[tr], xl[te], xu])
                y_all = np.concatenate([yl[tr], np.full(len(te) + len(xu), -1)])
                sc = StandardScaler().fit(x_all)
                m = LabelSpreading(kernel="knn", n_neighbors=k, alpha=0.2)
                m.fit(sc.transform(x_all), y_all)
                p[te] = m.label_distributions_[len(tr):len(tr) + len(te), 1]
            seen = ~np.isnan(p)
            accs.append(accuracy_score(yl[seen], (p[seen] >= 0.5).astype(int)))
        print(f"    n_neighbors={k:<3} accuracy {np.mean(accs):.3f}")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({
        "n_labelled": len(yl),
        "n_unlabelled": len(xu),
        "permutations": args.repeats,
        "evaluation": "out-of-fold, students permuted between folds",
        "baseline": {"accuracy": round(float(base_a.mean()), 4),
                     "accuracy_sd": round(float(base_a.std()), 4),
                     "macro_f1": round(float(base_f.mean()), 4)},
        "label_spreading": {"accuracy": round(float(ls_a.mean()), 4),
                            "accuracy_sd": round(float(ls_a.std()), 4),
                            "macro_f1": round(float(ls_f.mean()), 4)},
        "paired_difference": round(float(diff.mean()), 4),
        "paired_t": None if np.isnan(t) else round(float(t), 3),
        "wins": wins,
        "verdict": verdict,
        "caveat": "label spreading is transductive: the windows to be scored "
                  "must be present, unlabelled, when it is fit. It cannot "
                  "score a new student without refitting.",
    }, indent=1), encoding="utf-8")
    print(f"\n  report -> {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
