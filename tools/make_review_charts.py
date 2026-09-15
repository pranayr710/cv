"""Chart the fine-tuning run for the review deck, from results.csv.

Three charts, deliberately three and not one. Loss and mAP share an x-axis but
nothing else: putting them on one plot needs two y-scales, and a dual-axis chart
lets the reader infer a relationship from where two lines happen to cross, which
is a property of the scaling, not of the data.

    python tools/make_review_charts.py
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = Path("runs/behaviour/merged4_aug")
OUT = Path("ppt_assets")

# The deck's palette. Assigned in fixed order and never cycled, so a series
# keeps its hue no matter how many are drawn.
INK = "#10273F"
BODY = "#3C5066"
MUTE = "#8496A6"
GRID = "#E6ECF2"
SERIES = ("#0E7C86", "#B47A14", "#1F6F50", "#A9332A")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": GRID,
    "axes.labelcolor": BODY,
    "text.color": INK,
    "xtick.color": MUTE,
    "ytick.color": MUTE,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def style(ax, ylabel):
    """Recessive frame: the data is the ink, the scaffolding is not."""
    ax.set_ylabel(ylabel, fontsize=10.5, color=BODY)
    ax.set_xlabel("epoch", fontsize=10.5, color=BODY)
    ax.grid(axis="y", color=GRID, linewidth=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0, labelsize=9.5)


def read_rows():
    f = RUN / "results.csv"
    if not f.exists():
        print(f"no results at {f}")
        return None
    return list(csv.DictReader(f.open()))


def losses(rows):
    """Validation loss only. Training loss falls by construction; the question
    a reviewer is asking is whether it fell on data the model never saw."""
    ep = [int(r["epoch"]) for r in rows]
    fig, ax = plt.subplots(figsize=(6.4, 3.5), dpi=200)
    for i, (key, label) in enumerate((("val/box_loss", "box"),
                                      ("val/cls_loss", "cls"),
                                      ("val/dfl_loss", "dfl"))):
        vals = [float(r[key]) for r in rows]
        ax.plot(ep, vals, color=SERIES[i], linewidth=2, label=label)
        # Direct label at the line's end beats hunting a legend swatch.
        ax.annotate(label, (ep[-1], vals[-1]), xytext=(6, 0),
                    textcoords="offset points", color=SERIES[i],
                    fontsize=10, fontweight="bold", va="center")
    style(ax, "validation loss")
    ax.set_xlim(min(ep), max(ep) + 3)
    ax.set_xticks([1, 10, 20, 30, max(ep)])
    ax.set_title("Validation loss by epoch", fontsize=12.5, color=INK,
                 fontweight="bold", loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(OUT / "train_loss.png", bbox_inches="tight")
    plt.close(fig)
    print("  -> train_loss.png")


def metrics(rows):
    """mAP at two IoU regimes. Same units, so one axis is honest."""
    ep = [int(r["epoch"]) for r in rows]
    m50 = [float(r["metrics/mAP50(B)"]) for r in rows]
    m5095 = [float(r["metrics/mAP50-95(B)"]) for r in rows]
    best_i = max(range(len(rows)), key=lambda i: m5095[i])

    fig, ax = plt.subplots(figsize=(6.4, 3.5), dpi=200)
    ax.plot(ep, m50, color=SERIES[0], linewidth=2, label="mAP@50")
    ax.plot(ep, m5095, color=SERIES[1], linewidth=2, label="mAP@50-95")
    ax.axvline(ep[best_i], color=MUTE, linewidth=1, linestyle="--")
    # Parked low and left of the rule: at the marker it sat on top of the
    # mAP@50 line, which is the collision this chart exists to avoid.
    ax.annotate(f"best epoch {ep[best_i]}\nmAP@50 {m50[best_i]:.3f}\n"
                f"mAP@50-95 {m5095[best_i]:.3f}",
                (ep[best_i], 0.02), xytext=(-108, 0),
                textcoords="offset points", color=INK, fontsize=9,
                fontweight="bold", va="bottom")
    for series, vals, i in (("mAP@50", m50, 0), ("mAP@50-95", m5095, 1)):
        ax.annotate(series, (ep[-1], vals[-1]), xytext=(6, 0),
                    textcoords="offset points", color=SERIES[i],
                    fontsize=10, fontweight="bold", va="center")
    style(ax, "mean average precision")
    ax.set_ylim(0, 0.75)
    ax.set_xlim(min(ep), max(ep) + 7)
    # Ticks stop where the data stops; an axis running to 50 would
    # imply epochs that were never run.
    ax.set_xticks([1, 10, 20, 30, max(ep)])
    ax.set_title("Detection quality by epoch", fontsize=12.5, color=INK,
                 fontweight="bold", loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(OUT / "train_metrics.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  -> train_metrics.png  (best epoch {ep[best_i]})")


def ablation():
    """What dataset preparation bought, measured on the same architecture."""
    labels = ["mAP@50", "mAP@50-95", "precision", "recall"]
    before = [0.4148, 0.1879, 0.589, 0.381]
    after = [0.6071, 0.2688, 0.650, 0.586]

    fig, ax = plt.subplots(figsize=(6.6, 3.5), dpi=200)
    xs = range(len(labels))
    w = 0.34
    b1 = ax.bar([x - w / 2 for x in xs], before, w, color=MUTE,
                label="8 classes, 423 images", zorder=3)
    b2 = ax.bar([x + w / 2 for x in xs], after, w, color=SERIES[0],
                label="4 classes, 877 images", zorder=3)
    for bars in (b1, b2):
        for rect in bars:
            ax.annotate(f"{rect.get_height():.3f}",
                        (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8.5, color=BODY)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels)
    style(ax, "score")
    ax.set_xlabel("")
    ax.set_ylim(0, 0.75)
    ax.legend(frameon=False, fontsize=9.5, loc="upper right",
              labelcolor=BODY)
    ax.set_title("What dataset preparation bought — same model, same weights",
                 fontsize=12.5, color=INK, fontweight="bold", loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(OUT / "dataset_ablation.png", bbox_inches="tight")
    plt.close(fig)
    print("  -> dataset_ablation.png")


def main() -> int:
    OUT.mkdir(exist_ok=True)
    rows = read_rows()
    if rows is None:
        return 1
    losses(rows)
    metrics(rows)
    ablation()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
