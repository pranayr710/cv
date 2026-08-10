"""Characterise a Roboflow/YOLO-format labelled dataset before trusting it.

Run before using a new dataset for anything, because the two things that most
often invalidate a result are invisible in the file count: **near-duplicate
frames** extracted from the same video (which leak between train and val and
inflate every score), and a **class distribution** so skewed that overall
accuracy is meaningless.

Reports: class distribution, boxes per image, image resolution, box size
relative to the frame, and how many images come from each source video.

Run:
    python -m tools.analyse_labelled --root dataset/dataset
"""

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path


def _source_clip(stem: str) -> str:
    """Strip Roboflow's per-image hash and frame index to get the source clip.

    Roboflow names exported frames like
    ``ammad-seminar-stand_mp4-0003_jpg.rf.<hash>.jpg``. Frames sharing a prefix
    came from the same video and are near-duplicates of each other.

    Args:
        stem: The image filename stem.

    Returns:
        A best-effort source-clip identifier.
    """
    name = re.sub(r"_jpg\.rf\.[0-9a-zA-Z]+$", "", stem)
    return re.sub(r"[-_]\d{3,}$", "", name)


def analyse(root: Path) -> None:
    import cv2
    import yaml

    data_yaml = root / "data.yaml"
    names: list[str] = []
    if data_yaml.is_file():
        with data_yaml.open(encoding="utf-8") as fh:
            names = list(yaml.safe_load(fh).get("names", []))

    images_dir, labels_dir = root / "images", root / "labels"
    images = sorted(images_dir.glob("*.jpg")) + sorted(images_dir.glob("*.png"))
    if not images:
        raise FileNotFoundError(f"No images found in {images_dir}")

    class_counts: Counter[str] = Counter()
    boxes_per_image: list[int] = []
    resolutions: Counter[str] = Counter()
    clips: Counter[str] = Counter()
    rel_heights: list[float] = []
    images_per_class: defaultdict[str, set[str]] = defaultdict(set)
    unlabelled = 0

    for img_path in images:
        clips[_source_clip(img_path.stem)] += 1

        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        resolutions[f"{w}x{h}"] += 1

        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.is_file():
            unlabelled += 1
            boxes_per_image.append(0)
            continue

        rows = [
            line.split()
            for line in label_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        boxes_per_image.append(len(rows))
        for row in rows:
            cls_id = int(float(row[0]))
            cls_name = names[cls_id] if cls_id < len(names) else f"id{cls_id}"
            class_counts[cls_name] += 1
            images_per_class[cls_name].add(img_path.stem)
            if len(row) >= 5:
                rel_heights.append(float(row[4]))

    total_boxes = sum(class_counts.values())
    print(f"\n=== {root} ===")
    print(f"images: {len(images)}   labelled boxes: {total_boxes}   "
          f"images with no label file: {unlabelled}")

    print("\n--- class distribution ---")
    print(f"{'class':<16}{'boxes':>7}{'share':>8}{'images':>8}")
    for cls_name, count in class_counts.most_common():
        share = count / total_boxes * 100 if total_boxes else 0
        print(f"{cls_name:<16}{count:>7}{share:>7.1f}%{len(images_per_class[cls_name]):>8}")

    if boxes_per_image:
        srt = sorted(boxes_per_image)
        n = len(srt)
        print("\n--- students (boxes) per image ---")
        print(f"  min {srt[0]}   median {srt[n // 2]}   "
              f"mean {sum(srt) / n:.1f}   max {srt[-1]}")
        print(f"  images with 1 box: {sum(1 for b in srt if b == 1)}"
              f"   with >=10: {sum(1 for b in srt if b >= 10)}"
              f"   with >=20: {sum(1 for b in srt if b >= 20)}")

    if rel_heights:
        srt = sorted(rel_heights)
        n = len(srt)
        print("\n--- box height as a fraction of frame height ---")
        print(f"  min {srt[0]:.3f}   median {srt[n // 2]:.3f}   max {srt[-1]:.3f}")
        print(f"  boxes under 10% of frame height: "
              f"{sum(1 for v in srt if v < 0.10)} "
              f"({sum(1 for v in srt if v < 0.10) / n * 100:.1f}%)")

    print("\n--- resolutions ---")
    for res, count in resolutions.most_common(6):
        print(f"  {res:<12}{count:>5}")

    print(f"\n--- source clips (near-duplicate risk): {len(clips)} distinct ---")
    for clip, count in clips.most_common(12):
        print(f"  {clip:<44}{count:>4} frames")
    if len(clips) > 12:
        print(f"  ... and {len(clips) - 12} more")
    biggest = clips.most_common(1)[0][1] if clips else 0
    print(f"\n  largest single clip is {biggest / len(images) * 100:.1f}% of all images")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="dataset/dataset")
    args = parser.parse_args()
    analyse(Path(args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
