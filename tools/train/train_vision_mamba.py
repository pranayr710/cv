"""Train Vision Mamba on the engagement image set, against the ResNet18 baseline.

The point is the comparison. A Vision Mamba accuracy figure on its own says
nothing, because the number that matters is whether scanning patches with a
state-space model beats frozen ImageNet features and a logistic regression --
the 0.664 that currently stands.

Two things decide whether the figure means anything, and both are inherited
from the earlier work rather than reinvented:

**Splitting is by near-duplicate group.** 83.7% of these images were near
copies of another before deduplication, and copies remain within what is left.
A random split puts the same photograph on both sides and returns 0.968, which
is what this project measured before grouping and is the reason the honest
figure is 0.664.

**The 350 human scores are used for calibration, not accuracy.** Binary folder
labels say engaged or not; they cannot say how much. Spearman correlation
against the human 1-5 judgement is reported separately from accuracy, because
a model can sort images correctly and still be miscalibrated.

    python tools/train/train_vision_mamba.py --limit 2000 --epochs 6   # quick check
    python tools/train/train_vision_mamba.py                           # full run
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.vision_mamba import VisionMamba, VisionMambaConfig

SCORES = Path("outputs/image_engagement_scores.csv")
HUMAN = Path("labeling/scores.csv")
REPORT = Path("outputs/vision_mamba_report.json")
MODEL_OUT = Path("outputs/vision_mamba.pt")

#: ImageNet channel statistics, so input is normalised identically to the
#: ResNet18 route this is being compared against.
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class ImageSet(Dataset):
    """Images and their binary folder label, with light augmentation.

    Augmentation is horizontal flip and brightness jitter only. A classroom or
    e-learning portrait has a canonical orientation, so rotation or a vertical
    flip would synthesise images the model will never meet.
    """

    def __init__(self, paths: list[str], labels: np.ndarray, size: int,
                 train: bool) -> None:
        self.paths = paths
        self.labels = labels
        self.size = size
        self.train = train
        self._mean = torch.tensor(MEAN).view(3, 1, 1)
        self._std = torch.tensor(STD).view(3, 1, 1)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, float]:
        from PIL import Image

        img = Image.open(self.paths[i]).convert("RGB").resize(
            (self.size, self.size), Image.BILINEAR)
        x = torch.from_numpy(
            np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        if self.train:
            if torch.rand(1).item() < 0.5:
                x = x.flip(-1)
            x = (x * (0.8 + 0.4 * torch.rand(1).item())).clamp(0, 1)
        return (x - self._mean) / self._std, float(self.labels[i])


def load_rows(limit: int) -> tuple[list[str], np.ndarray, dict[str, float]]:
    """Image paths, binary labels, and the human 1-5 score where one exists."""
    rows = list(csv.DictReader(SCORES.open(encoding="utf-8")))
    if limit:
        # Evenly spread rather than the first N, so a quick run still covers
        # both classes and every age band.
        step = max(1, len(rows) // limit)
        rows = rows[::step][:limit]
    paths = [r["path"] for r in rows]
    y = np.array([1 if r["folder_label"] == "Engagement" else 0 for r in rows])

    human: dict[str, float] = {}
    if HUMAN.exists():
        for r in csv.DictReader(HUMAN.open(encoding="utf-8")):
            if r.get("score_5"):
                human[Path(r["source"]).name] = float(r["score_5"])
    return paths, y, human


def evaluate(model: VisionMamba, loader: DataLoader,
             device: str) -> tuple[np.ndarray, np.ndarray]:
    """Held-out probabilities and their true labels."""
    model.eval()
    probs: list[float] = []
    truth: list[float] = []
    with torch.no_grad():
        for xb, yb in loader:
            probs.extend(torch.sigmoid(model(xb.to(device))).cpu().numpy())
            truth.extend(yb.numpy())
    return np.array(probs), np.array(truth)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0,
                    help="use only this many images (0 = all)")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from scipy.stats import spearmanr
    from sklearn.model_selection import GroupShuffleSplit

    from tools.train.train_engagement_scorer import duplicate_groups

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    paths, y, human = load_rows(args.limit)
    print(f"{len(paths)} images ({y.sum()} engagement, {len(y) - y.sum()} "
          f"disengagement), {len(human)} human-scored | device {device}")

    print("  clustering near-duplicates so copies cannot straddle the split...")
    groups = duplicate_groups([Path(p) for p in paths])
    print(f"  {len(set(groups))} distinct groups")

    tr, te = next(GroupShuffleSplit(
        n_splits=1, test_size=0.2,
        random_state=args.seed).split(paths, y, groups))
    print(f"  train {len(tr)}  test {len(te)}  (no group shared)")

    cfg = VisionMambaConfig(image_size=args.size)
    model = VisionMamba(cfg).to(device)
    print(f"  Vision Mamba: {model.n_params:,} parameters, "
          f"{cfg.n_patches} patches\n")

    train_dl = DataLoader(
        ImageSet([paths[i] for i in tr], y[tr], args.size, True),
        batch_size=args.batch, shuffle=True)
    test_dl = DataLoader(
        ImageSet([paths[i] for i in te], y[te], args.size, False),
        batch_size=args.batch)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    loss_fn = nn.BCEWithLogitsLoss()

    best = 0.0
    probs = np.array([])
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss) * len(yb)
        sched.step()

        probs, truth = evaluate(model, test_dl, device)
        acc = float(((probs >= 0.5) == truth).mean())
        best = max(best, acc)
        print(f"  epoch {epoch:>2}/{args.epochs}  loss {total / len(tr):.4f}  "
              f"held-out accuracy {acc:.3f}")

    # Calibration against whichever human scores fall in the test split.
    names = [Path(paths[i]).name for i in te]
    pairs = [(p, human[n]) for p, n in zip(probs, names) if n in human]
    rho = float("nan")
    if len(pairs) >= 10:
        rho = float(spearmanr([a for a, _ in pairs],
                              [b for _, b in pairs]).statistic)

    print(f"\n  {'':<34}{'accuracy':>10}")
    print(f"  {'ResNet18 frozen + logistic':<34}{0.664:>10.3f}   (grouped)")
    print(f"  {'Vision Mamba (this run)':<34}{acc:>10.3f}   (grouped)")
    if not math.isnan(rho):
        print(f"\n  Spearman vs {len(pairs)} human scores in the test split: "
              f"{rho:.3f}   (ResNet18 route: 0.556)")

    torch.save({"state_dict": model.state_dict(), "config": vars(cfg)},
               MODEL_OUT)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({
        "n_images": len(paths),
        "n_groups": len(set(groups)),
        "n_train": len(tr),
        "n_test": len(te),
        "evaluation": "held out by near-duplicate group",
        "parameters": model.n_params,
        "epochs": args.epochs,
        "image_size": args.size,
        "vision_mamba_accuracy": round(acc, 4),
        "vision_mamba_best_epoch_accuracy": round(best, 4),
        "resnet18_logistic_accuracy": 0.6636,
        "spearman_vs_human": None if math.isnan(rho) else round(rho, 4),
        "n_human_in_test": len(pairs),
    }, indent=1), encoding="utf-8")
    print(f"\n  model  -> {MODEL_OUT}\n  report -> {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
