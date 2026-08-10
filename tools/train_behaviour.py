"""Fine-tune YOLOv11 to detect students *and* their behaviour in one pass.

Replaces the book-proximity proxy, which was measured against human labels and
failed: precision 31.9%, recall 20.7%, F1 25.1% for "is this student writing"
(see ``tools/eval_detection.py`` and CHALLENGES_AND_SOLUTIONS.md §12). COCO's
`book` class was the only positive-evidence signal available without labels;
labels now exist, so the behaviour is learned directly instead of inferred from
a proxy object.

A useful side effect: because every labelled box *is* a student, this one model
outputs student boxes and behaviour together, rather than needing a separate
person detector plus a behaviour rule on top.

Deliberate choices worth knowing about:

* **Clip-wise validation** (``tools/make_split.py``) -- validation clips are
  classrooms the model has never seen. A random frame split would put
  near-duplicate frames on both sides and inflate every number.
* **Small-data settings.** 423 training images from 8 clips is very little for
  detection. Training starts from COCO-pretrained weights, runs with early
  stopping, and a fixed seed so a rerun is comparable.
* **imgsz 960, not the pipeline's 1920.** Purely a 6.4 GB VRAM limit. The
  mismatch is real and is re-measured at the pipeline's own resolution in
  ``tools/eval_behaviour.py`` rather than assumed away.

Run:
    python -m tools.train_behaviour
    python -m tools.train_behaviour --model yolo11s.pt --epochs 60
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def train(
    data: Path,
    model_name: str,
    epochs: int,
    imgsz: int,
    batch: int,
    patience: int,
    name: str,
    workers: int,
) -> None:
    from ultralytics import YOLO

    if not data.is_file():
        raise FileNotFoundError(
            f"{data} not found — run `python -m tools.make_split` first."
        )

    model = YOLO(model_name)
    model.train(
        data=str(data),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        patience=patience,
        project=str(ROOT / "runs" / "behaviour"),
        name=name,
        exist_ok=True,
        # Fixed seed, but NOT deterministic=True. Measured on this machine:
        # deterministic=True (plus cache="ram") gave 551 s/epoch, which is 15
        # hours for 100 epochs on 423 images -- it forces cuDNN onto
        # reproducible-but-slow kernels and disables autotuning. The seed alone
        # keeps runs close enough to compare while staying usable.
        seed=0,
        deterministic=False,
        # cache="ram" was also dropped: 423 frames at 1920x1080 decode to ~2.6 GB
        # and the resulting memory pressure cost more than the decode it saved.
        cache=False,
        workers=workers,
        val=True,
        plots=True,
        verbose=True,
    )
    print(f"\nweights: {ROOT / 'runs' / 'behaviour' / name / 'weights' / 'best.pt'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="dataset/behaviour/data.yaml")
    parser.add_argument("--model", default="yolo11m.pt")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--name", default="yolo11m_960")
    args = parser.parse_args()

    train(
        Path(args.data),
        args.model,
        args.epochs,
        args.imgsz,
        args.batch,
        args.patience,
        args.name,
        args.workers,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
