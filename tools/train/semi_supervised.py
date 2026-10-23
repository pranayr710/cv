"""Can the 405 unlabelled windows be labelled by the 179 that are?

This is the question semi-supervised learning exists to answer, and it has a
real failure mode that looks like success: a model trained on its own confident
guesses becomes more confident without becoming more correct, and reports the
confidence as progress. So every method here is measured the same way -- out of
fold, on human labels only, split by student -- and the pseudo-labelling happens
*inside* each training fold. Pseudo-labelling once up front would let the test
students' windows influence the labels the model then trains on, and the figure
would be meaningless.

Four methods, in increasing order of how much they trust the model:

**Baseline.** The 179 human labels alone. Everything else must beat this or it
is not worth the complexity.

**Self-training.** Label the unlabelled windows the model is most confident
about, add them, refit. The classic method and the one most prone to
compounding its own errors.

**Co-training.** Two models that read genuinely different views of the same
window -- the MLP sees 18 aggregate features, the Mamba sees the 48-step
sequence -- and a pseudo-label is accepted only where both are confident *and*
agree. Disagreement between two different views is evidence the window is hard,
which is exactly when a guess should not be trusted. This is the method this
data actually suits, because the two views were built independently.

**Label spreading.** No pseudo-labels at all: labels diffuse through a
similarity graph over all 584 windows at once. It makes a different assumption
-- that nearby windows share a label -- which may or may not hold here.

    python tools/train/semi_supervised.py
    python tools/train/semi_supervised.py --seeds 10 --confidence 0.85
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.engagement_features import iter_windows

LABELS = Path("labeling/labels.json")
GRAPH = Path("outputs/final2/live_graph.jsonl")
REPORT = Path("results/semi_supervised_report.json")


def load() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                    np.ndarray, np.ndarray]:
    """Labelled features/target/group, and the unlabelled features.

    Sequences are returned alongside so co-training has its second view.
    """
    from backend.engagement_sequence import sequences_from_graph

    store = json.loads(LABELS.read_text(encoding="utf-8"))
    human = {(v["person_id"], v["start_ms"]): v["label"]
             for v in store.values()}
    seqs = sequences_from_graph(GRAPH)

    xl, yl, gl, sl = [], [], [], []
    xu, su = [], []
    for w in iter_windows(GRAPH):
        key = (w.person_id, w.start_ms)
        if key not in seqs:
            continue
        label = human.get(key)
        if label in ("on", "off"):
            xl.append(w.features)
            yl.append(1 if label == "on" else 0)
            gl.append(w.person_id)
            sl.append(seqs[key].steps)
        elif label is None:
            # "unclear" is excluded from both sides: the rater looked and
            # declined, so it is not unlabelled, it is known-ambiguous.
            xu.append(w.features)
            su.append(seqs[key].steps)
    return (np.array(xl, float), np.array(yl), np.array(gl),
            np.array(sl, dtype=np.float32), np.array(xu, float),
            np.array(su, dtype=np.float32))


def fit_mlp(x: np.ndarray, y: np.ndarray, seed: int, weights=None):
    """An MLP on aggregate features, wrapped with its scaler."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(x)
    clf = MLPClassifier((16,), max_iter=3000, random_state=seed)
    clf.fit(scaler.transform(x), y)
    return scaler, clf


def predict_mlp(model, x: np.ndarray) -> np.ndarray:
    scaler, clf = model
    return clf.predict_proba(scaler.transform(x))[:, 1]


def fit_mamba(steps: np.ndarray, y: np.ndarray, seed: int, device: str):

    from tools.train.train_mamba_scorer import train_fold
    return train_fold(steps, y, 150, seed, device)


def predict_mamba(model, steps: np.ndarray, device: str) -> np.ndarray:
    import torch

    model.eval()
    with torch.no_grad():
        return torch.sigmoid(
            model(torch.tensor(steps, device=device))).cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--confidence", type=float, default=0.90,
                    help="how sure a model must be to pseudo-label a window")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    import torch
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.semi_supervised import LabelSpreading

    device = "cuda" if torch.cuda.is_available() else "cpu"
    xl, yl, gl, sl, xu, su = load()
    print(f"{len(yl)} labelled ({yl.sum()} on, {len(yl) - yl.sum()} off), "
          f"{len(xu)} unlabelled, {len(set(gl))} students | {device}")
    print(f"pseudo-label confidence threshold: {args.confidence}\n")

    results: dict[str, dict] = {}

    def record(name: str, accs: list[float], f1s: list[float],
               extra: dict | None = None) -> None:
        results[name] = {
            "accuracy": round(float(np.mean(accs)), 4),
            "accuracy_sd": round(float(np.std(accs)), 4),
            "macro_f1": round(float(np.mean(f1s)), 4),
            "macro_f1_sd": round(float(np.std(f1s)), 4),
            **(extra or {}),
        }

    base_a, base_f = [], []
    self_a, self_f, self_added = [], [], []
    co_a, co_f, co_added = [], [], []
    spread_a, spread_f = [], []

    for seed in range(args.seeds):
        p_base = np.zeros(len(yl))
        p_self = np.zeros(len(yl))
        p_co = np.zeros(len(yl))
        p_spread = np.zeros(len(yl))
        added_self = added_co = 0

        for tr, te in GroupKFold(n_splits=args.folds).split(xl, yl, gl):
            # --- baseline -------------------------------------------------
            model = fit_mlp(xl[tr], yl[tr], seed)
            p_base[te] = predict_mlp(model, xl[te])

            # --- self-training --------------------------------------------
            pu = predict_mlp(model, xu)
            confident = np.abs(pu - 0.5) >= (args.confidence - 0.5)
            if confident.sum():
                x_aug = np.vstack([xl[tr], xu[confident]])
                y_aug = np.concatenate([yl[tr], (pu[confident] >= 0.5).astype(int)])
                added_self += int(confident.sum())
                p_self[te] = predict_mlp(fit_mlp(x_aug, y_aug, seed), xl[te])
            else:
                p_self[te] = p_base[te]

            # --- co-training ----------------------------------------------
            mam = fit_mamba(sl[tr], yl[tr], seed, device)
            pu_m = predict_mamba(mam, su, device)
            sure_a = np.abs(pu - 0.5) >= (args.confidence - 0.5)
            sure_b = np.abs(pu_m - 0.5) >= (args.confidence - 0.5)
            agree = (pu >= 0.5) == (pu_m >= 0.5)
            take = sure_a & sure_b & agree
            if take.sum():
                x_aug = np.vstack([xl[tr], xu[take]])
                y_aug = np.concatenate([yl[tr], (pu[take] >= 0.5).astype(int)])
                added_co += int(take.sum())
                p_co[te] = predict_mlp(fit_mlp(x_aug, y_aug, seed), xl[te])
            else:
                p_co[te] = p_base[te]

            # --- label spreading ------------------------------------------
            x_all = np.vstack([xl[tr], xl[te], xu])
            y_all = np.concatenate(
                [yl[tr], np.full(len(te) + len(xu), -1)])
            scaler = StandardScaler().fit(x_all)
            ls = LabelSpreading(kernel="knn", n_neighbors=7, alpha=0.2)
            ls.fit(scaler.transform(x_all), y_all)
            p_spread[te] = ls.label_distributions_[
                len(tr):len(tr) + len(te), 1]

        for store_a, store_f, p in (
                (base_a, base_f, p_base), (self_a, self_f, p_self),
                (co_a, co_f, p_co), (spread_a, spread_f, p_spread)):
            pred = (p >= 0.5).astype(int)
            store_a.append(accuracy_score(yl, pred))
            store_f.append(f1_score(yl, pred, average="macro"))
        self_added.append(added_self / args.folds)
        co_added.append(added_co / args.folds)
        print(f"  seed {seed}: baseline {base_a[-1]:.3f}  self {self_a[-1]:.3f}"
              f"  co {co_a[-1]:.3f}  spread {spread_a[-1]:.3f}")

    record("baseline (179 human labels)", base_a, base_f)
    record("self-training", self_a, self_f,
           {"pseudo_labels_added": round(float(np.mean(self_added)), 1)})
    record("co-training (MLP + Mamba agree)", co_a, co_f,
           {"pseudo_labels_added": round(float(np.mean(co_added)), 1)})
    record("label spreading", spread_a, spread_f)

    print(f"\n  {'':<34}{'accuracy':>10}{'macro F1':>11}{'added':>9}")
    for name, r in results.items():
        added = r.get("pseudo_labels_added", "")
        print(f"  {name:<34}{r['accuracy']:>10.3f}{r['macro_f1']:>11.3f}"
              f"{added!s:>9}")

    base = results["baseline (179 human labels)"]
    best = max(results, key=lambda k: results[k]["accuracy"])
    gain = results[best]["accuracy"] - base["accuracy"]
    spread_sd = base["accuracy_sd"] + results[best]["accuracy_sd"]
    print(f"\n  best: {best}")
    if best == "baseline (179 human labels)":
        print("  No method beat the labels alone. The unlabelled windows did"
              "\n  not help, and that is the result.")
    elif gain < spread_sd:
        print(f"  Gain is {gain:+.3f} against a combined spread of "
              f"{spread_sd:.3f}:\n  within noise, so not a real improvement.")
    else:
        print(f"  Gain is {gain:+.3f}, larger than the combined spread of "
              f"{spread_sd:.3f}.")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({
        "n_labelled": len(yl),
        "n_unlabelled": len(xu),
        "n_students": len(set(gl)),
        "seeds": args.seeds,
        "confidence_threshold": args.confidence,
        "evaluation": "out-of-fold on human labels, grouped by student; "
                      "pseudo-labelling done inside each training fold",
        "results": results,
        "best": best,
    }, indent=1), encoding="utf-8")
    print(f"\n  report -> {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
