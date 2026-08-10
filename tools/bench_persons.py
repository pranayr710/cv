"""Sweep YOLO person-detection recall across model size and inference size.

Motivated by a finding from ``tools/bench_faces.py``: SCRFD finds 434 faces
across the 13 dataset images while YOLOv11m finds only 264 persons, and the
unmatched faces were visually confirmed to be real students in crowded back
rows. Person detection, not face detection, is now the pipeline's weakest
stage -- so it needs the same measured sweep that ``imgsz`` got originally.

SCRFD's face count is used here as an *approximate* independent reference for
how many students are actually present. It is not ground truth (it has its own
misses and can produce false positives), but it is a far better yardstick than
the earlier hand-counted single image, and it is measured the same way for
every configuration, so the comparison between configurations is fair.

Run:
    python -m tools.bench_persons
    python -m tools.bench_persons --models yolo11m.pt yolo11l.pt --sizes 1280 1536
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import replace
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"


def sweep(models: list[str], sizes: list[int], reference: bool) -> None:
    import cv2

    from backend.config import CONFIG
    from backend.detection import Detector

    images = sorted(DATASET.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No .jpg images found in {DATASET}")
    frames = {}
    for path in images:
        frame = cv2.imread(str(path))
        if frame is None:
            logger.warning("Skipping unreadable image: %s", path)
            continue
        frames[path.name] = frame

    ref_faces = {}
    if reference:
        from backend.face_detect import FaceDetector

        fd = FaceDetector()
        for name, frame in frames.items():
            ref_faces[name] = len(fd.detect(frame))
        print(f"\nSCRFD reference face count (approximate student count): "
              f"{sum(ref_faces.values())}")

    print(f"\n{'model':<14} {'imgsz':>6} {'persons':>8} {'books':>6} "
          f"{'phones':>7} {'ms/img':>8}")
    for model in models:
        for size in sizes:
            cfg = replace(CONFIG.detection, weights=model, imgsz=size)
            try:
                detector = Detector(cfg)
            except (FileNotFoundError, RuntimeError, ImportError) as exc:
                print(f"{model:<14} {size:>6}   FAILED: {exc}")
                continue

            n_persons = n_books = n_phones = 0
            t0 = time.perf_counter()
            for frame in frames.values():
                persons, objects = detector.detect(frame)
                n_persons += len(persons)
                n_books += sum(1 for o in objects if o.cls == "book")
                n_phones += sum(1 for o in objects if o.cls == "cell phone")
            elapsed = (time.perf_counter() - t0) / len(frames) * 1000

            print(f"{model:<14} {size:>6} {n_persons:>8} {n_books:>6} "
                  f"{n_phones:>7} {elapsed:>8.0f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+",
                       default=["yolo11m.pt", "yolo11l.pt", "yolo11x.pt"])
    parser.add_argument("--sizes", nargs="+", type=int,
                       default=[1280, 1536, 1920])
    parser.add_argument("--no-reference", action="store_true",
                       help="Skip the SCRFD face-count reference.")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)
    sweep(args.models, args.sizes, not args.no_reference)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
