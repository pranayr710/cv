"""Train an image-level engagement scorer and calibrate it to a 1-10 scale.

The folder labels are binary, so a model trained on them emits a probability,
not a score. The probability is already ordered -- 0.93 is more confident than
0.41 -- but "more confident" is not the same as "twice as engaged", and
presenting a raw probability as a 1-10 score asserts a calibration nobody
checked.

So this does two separable things:

1. Train a classifier on the binary folder labels, held out honestly.
2. Fit the map from its probability to a human 1-5 judgement, using the
   calibration set, and report how well the resulting score tracks a person.

The calibration images are excluded from training. They are the only evidence
that the score means anything, and a model scored on images it was fitted to
would report a number about its own memory.

Features come from an ImageNet-pretrained ResNet18 rather than from
fine-tuning. With ~12,000 images of a task this subjective, a linear model on
frozen features is the more honest baseline: it cannot memorise the way a
fully fine-tuned network can, its probabilities are better behaved before
calibration, and if it already separates the classes then a heavier model was
never the missing piece.

    python tools/train_engagement_scorer.py
    python tools/train_engagement_scorer.py --refresh-features
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DATA = Path("labeling/images/Final Dataset 256/Final Dataset 256")
CALIB_MANIFEST = Path("labeling/images/calibration/manifest.json")
SCORES = Path("labeling/scores.json")
CACHE = Path("outputs/engagement_features.npz")
REPORT = Path("outputs/engagement_scorer_report.json")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
BATCH = 64


def list_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*")
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def extract_features(paths: list[Path]) -> np.ndarray:
    """512-d penultimate activations from an ImageNet ResNet18."""
    import torch
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision import models, transforms

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    net = models.resnet18(weights=weights)
    net.fc = torch.nn.Identity()
    net.eval().to(device)

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    class Images(Dataset):
        def __len__(self) -> int:
            return len(paths)

        def __getitem__(self, i: int):
            with Image.open(paths[i]) as im:
                return tf(im.convert("RGB"))

    loader = DataLoader(Images(), batch_size=BATCH, num_workers=0)
    out = []
    with torch.no_grad():
        for n, batch in enumerate(loader, 1):
            out.append(net(batch.to(device)).cpu().numpy())
            if n % 20 == 0:
                print(f"    {n * BATCH}/{len(paths)}")
    return np.concatenate(out)


def load_or_build(paths: list[Path], refresh: bool) -> np.ndarray:
    keys = np.array([str(p) for p in paths])
    if CACHE.exists() and not refresh:
        cached = np.load(CACHE, allow_pickle=True)
        if len(cached["keys"]) == len(keys) and (cached["keys"] == keys).all():
            print(f"  features loaded from {CACHE}")
            return cached["features"]
        print("  cache is stale, rebuilding")
    print(f"  extracting features for {len(paths)} images...")
    features = extract_features(paths)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, features=features, keys=keys)
    return features



def duplicate_groups(paths: list[Path], distance: int = 3) -> np.ndarray:
    """Cluster near-identical images so copies cannot straddle a split.

    This dataset holds roughly 2,000 distinct pictures repeated about six times
    each with small variations. Split at random, near-copies land on both sides
    and the model is tested on images it effectively memorised -- which is
    exactly what a 96.8% accuracy on a task humans agree on 84% of the time
    was measuring.

    Deleting the copies would throw away 84% of the data. Grouping them keeps
    every image for training while making the test set genuinely unseen.

    Returns:
        A group id per image, for GroupShuffleSplit.
    """
    from PIL import Image

    hashes = []
    for path in paths:
        try:
            with Image.open(path) as im:
                px = list(im.convert("L").resize((9, 8),
                                                 Image.Resampling.LANCZOS)
                          .getdata())
            bits = 0
            for row in range(8):
                for col in range(8):
                    bits = (bits << 1) | int(px[row * 9 + col]
                                             > px[row * 9 + col + 1])
            hashes.append(bits)
        except Exception:  # noqa: BLE001 - an unreadable file gets its own group
            hashes.append(None)

    parent = list(range(len(paths)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    by_hash: dict[int, list[int]] = {}
    for i, h in enumerate(hashes):
        if h is not None:
            by_hash.setdefault(h, []).append(i)
    # Exact hash matches first, then the more expensive near-match pass over
    # one representative per distinct hash.
    for members in by_hash.values():
        for other in members[1:]:
            union(members[0], other)
    keys = list(by_hash)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            if (a ^ b).bit_count() <= distance:
                union(by_hash[a][0], by_hash[b][0])

    return np.array([find(i) for i in range(len(paths))])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh-features", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from scipy.stats import spearmanr
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import (
        GroupShuffleSplit,
        train_test_split,
    )

    if not SCORES.exists():
        print(f"{SCORES} not found -- score the calibration set first.")
        return 1

    paths = list_images(DATA)
    labels = np.array([1 if "Engagement" in p.parts and "Disengagement"
                       not in p.parts else 0 for p in paths])
    print(f"{len(paths)} images: {labels.sum()} engaged, "
          f"{len(labels) - labels.sum()} disengaged\n")

    features = load_or_build(paths, args.refresh_features)

    # The calibration images must not train the model that will be judged on
    # them. They are identified by their original path, recorded when sampled.
    calib = json.loads(CALIB_MANIFEST.read_text(encoding="utf-8"))
    calib_sources = {Path(e["source"]).as_posix() for e in calib}
    is_calib = np.array([p.as_posix() in calib_sources for p in paths])
    print(f"  held out of training: {is_calib.sum()} calibration images\n")

    x_pool, y_pool = features[~is_calib], labels[~is_calib]
    pool_paths = [p for p, c in zip(paths, is_calib) if not c]

    print("  clustering near-duplicates so copies cannot straddle the split...")
    groups = duplicate_groups(pool_paths)
    print(f"  {len(x_pool)} images fall into {len(set(groups))} distinct groups")

    # Reported for contrast: the naive number this dataset invites.
    xa, xb, ya, yb = train_test_split(
        x_pool, y_pool, test_size=0.2, random_state=args.seed, stratify=y_pool)
    naive = LogisticRegression(max_iter=2000).fit(xa, ya)
    naive_acc = accuracy_score(yb, naive.predict(xb))

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2,
                                 random_state=args.seed)
    train_idx, test_idx = next(splitter.split(x_pool, y_pool, groups))
    x_train, x_test = x_pool[train_idx], x_pool[test_idx]
    y_train, y_test = y_pool[train_idx], y_pool[test_idx]

    print(f"  train {len(x_train)}   test {len(x_test)}  (no group shared)")
    clf = LogisticRegression(max_iter=2000, C=1.0)
    clf.fit(x_train, y_train)

    p_test = clf.predict_proba(x_test)[:, 1]
    acc = accuracy_score(y_test, p_test >= 0.5)
    auc = roc_auc_score(y_test, p_test)
    # Both are printed because the gap between them IS the finding: the first
    # is what this dataset hands you if you split it the obvious way.
    print(f"\n  accuracy, random split (LEAKY): {naive_acc:.3f}")
    print(f"  accuracy, grouped split      : {acc:.3f}")
    print(f"  ROC AUC, grouped split       : {auc:.3f}")

    # --- calibration against human judgement ------------------------------ #
    human = json.loads(SCORES.read_text(encoding="utf-8"))
    by_source = {Path(e["source"]).as_posix(): e["file"] for e in calib}
    idx, y_human = [], []
    for i, p in enumerate(paths):
        if not is_calib[i]:
            continue
        name = by_source.get(p.as_posix())
        if name and name in human:
            idx.append(i)
            y_human.append(human[name]["score_5"])
    idx = np.array(idx)
    y_human = np.array(y_human, dtype=float)
    p_calib = clf.predict_proba(features[idx])[:, 1]
    print(f"\n  calibration images with a human score: {len(idx)}")

    rho, pval = spearmanr(p_calib, y_human)
    print(f"  Spearman rho (probability vs human 1-5): {rho:.3f} "
          f"(p={pval:.2g})")

    # Isotonic keeps the ordering and fits the shape, without assuming the
    # relationship is a straight line -- there is no reason it should be.
    iso = IsotonicRegression(y_min=1, y_max=5, out_of_bounds="clip")
    iso.fit(p_calib, y_human)
    fitted = iso.predict(p_calib)
    mae = float(np.mean(np.abs(fitted - y_human)))
    print(f"  mean absolute error after calibration : {mae:.2f} points (of 5)")

    # 1-5 becomes 1-10 only at the very end, as a presentation step.
    score_10 = np.clip(np.round((fitted - 1) / 4 * 9 + 1), 1, 10)
    print(f"  resulting 1-10 scores span {int(score_10.min())}"
          f"-{int(score_10.max())}, mean {score_10.mean():.1f}")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({
        "n_images": len(paths),
        "n_train": len(x_train),
        "n_test": len(x_test),
        "held_out_accuracy_grouped": round(float(acc), 4),
        "held_out_accuracy_random_leaky": round(float(naive_acc), 4),
        "held_out_roc_auc": round(float(auc), 4),
        "n_calibration_scored": len(idx),
        "spearman_rho_prob_vs_human": round(float(rho), 4),
        "spearman_p": float(pval),
        "calibrated_mae_points_of_5": round(mae, 3),
        "backbone": "resnet18 (ImageNet), frozen",
        "classifier": "logistic regression",
    }, indent=1), encoding="utf-8")
    print(f"\n  report: {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
