"""Build a clip-wise train/val split for the labelled behaviour dataset.

**Why not a random frame split.** The 481 images come from only 11 source
videos; the largest single clip is 24.5% of them. Consecutive frames of the same
video are near-identical, so a random frame-level split puts near-duplicates of
the same moment on both sides and the model scores well by memorising rather
than generalising. Splitting by **clip** means validation clips are classrooms
the model has never seen — the only split that answers the question we actually
care about.

The cost is honest and worth stating: with 11 clips, a held-out set is 2-3
classrooms. Validation numbers will be noisy, and a single unusual clip can move
them several points. That is a property of the data, not of the split.

**All 8 classes are kept**, including `handrise` (26 boxes) and `stand` (60),
even though those are far too few to learn well. Dropping them would leave real
students *unlabelled* in the images they appear in, which actively teaches the
detector to suppress standing and hand-raising students — worse than having a
class with weak metrics. Their per-class numbers are reported and should be
treated as unmeasured, not as results.

Run:
    python -m tools.make_split --root dataset/dataset --out dataset/behaviour
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from tools.analyse_labelled import _source_clip

#: Clips held out for validation. Chosen (not random) so the held-out set spans
#: more than one filming session and contains the behaviours we care about most
#: -- `write`/`read`/`using_device` -- rather than whichever clips a seed picked.
#: Verified by the class-distribution printout at the end of a run.
DEFAULT_VAL_CLIPS: tuple[str, ...] = ("owais-class-2_mp4", "taimoor-class-3_mp4")


def build(root: Path, out: Path, val_clips: tuple[str, ...]) -> None:
    with (root / "data.yaml").open(encoding="utf-8") as fh:
        names = list(yaml.safe_load(fh).get("names", []))

    images = sorted((root / "images").glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No images in {root / 'images'}")

    by_clip: defaultdict[str, list[Path]] = defaultdict(list)
    for path in images:
        by_clip[_source_clip(path.stem)].append(path)

    unknown = [c for c in val_clips if c not in by_clip]
    if unknown:
        raise ValueError(
            f"Requested val clips not present: {unknown}. "
            f"Available: {sorted(by_clip)}"
        )

    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val"):
        (out / split / "images").mkdir(parents=True)
        (out / split / "labels").mkdir(parents=True)

    counts: dict[str, Counter[str]] = {"train": Counter(), "val": Counter()}
    n_images = Counter()
    for clip, clip_images in by_clip.items():
        split = "val" if clip in val_clips else "train"
        for img_path in clip_images:
            label_src = root / "labels" / f"{img_path.stem}.txt"
            shutil.copy2(img_path, out / split / "images" / img_path.name)
            if label_src.is_file():
                shutil.copy2(label_src, out / split / "labels" / label_src.name)
                for line in label_src.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if not parts:
                        continue
                    cls_id = int(float(parts[0]))
                    counts[split][
                        names[cls_id] if cls_id < len(names) else f"id{cls_id}"
                    ] += 1
            n_images[split] += 1

    # Paths are absolute so the yaml works regardless of Ultralytics' cwd or its
    # configured datasets_dir, which is a common source of silent "found 0
    # images" runs.
    (out / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(out.resolve()),
                "train": "train/images",
                "val": "val/images",
                "nc": len(names),
                "names": names,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    print(f"clips: {len(by_clip)}   val clips: {list(val_clips)}")
    print(f"images: train {n_images['train']}   val {n_images['val']}")
    print(f"\n{'class':<16}{'train':>8}{'val':>8}")
    for name in names:
        print(f"{name:<16}{counts['train'][name]:>8}{counts['val'][name]:>8}")
    print(f"{'TOTAL':<16}{sum(counts['train'].values()):>8}"
          f"{sum(counts['val'].values()):>8}")

    missing = [n for n in names if counts["val"][n] == 0]
    if missing:
        print(f"\nWARNING: no val boxes for {missing} -- those classes cannot "
              f"be measured by this split at all.")
    print(f"\nwrote {out / 'data.yaml'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="dataset/dataset")
    parser.add_argument("--out", default="dataset/behaviour")
    parser.add_argument("--val-clips", nargs="+", default=list(DEFAULT_VAL_CLIPS))
    args = parser.parse_args()
    build(Path(args.root), Path(args.out), tuple(args.val_clips))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
