"""Merge two independently-collected classroom datasets into one 4-class set.

Why this is now possible, when it was previously blocked
--------------------------------------------------------

An earlier attempt to use the second dataset was rejected (see
``CHALLENGES_AND_SOLUTIONS.md`` section 18) because of a **label-density
conflict**: the second dataset labels only notable behaviours (phone, sleeping,
hand-raising) and leaves attentive students entirely unannotated, while the
first labels every visible student including plain ``look_forward``. Merging
directly would teach the model that an attentive student -- boxed in one set,
unboxed in the other -- is background.

Dropping ``look_forward`` dissolves that conflict, and dropping it is correct
independently: :mod:`backend.behaviour` already **suppresses** ``look_forward``
and ``turn_head``, because calibrated head pose measures head orientation far
better (F1 63.2% vs the behaviour model's 25.0% on the same boxes). The
behaviour model was carrying, and being dominated by, a class the pipeline
throws away -- ``look_forward`` alone was 2384 of 4603 boxes (52%).

With those classes gone, both datasets label the same four behaviours
consistently, "no detection" cleanly means "nothing notable", and the training
set grows 4.9x:

============  =====  =====  ======
class          ours    new  merged
============  =====  =====  ======
read            344   1063    1407
write           320    975    1295
sleep           232   1474    1706
using_device    522   1975    2497
TOTAL          1418   5487    6905
============  =====  =====  ======

``using_device`` -- the pipeline's worst class at ~21% recall -- gains the most
in absolute terms (522 -> 2497), which is where improvement is most needed.

Splitting
---------

Held out **by source**, never by frame: the first dataset's frames come from 11
videos and the second's from distinct photo sessions, so a frame-level split
would put near-duplicates on both sides and inflate every score. The second
dataset's own ``test`` split is kept entirely out of training so it remains a
true out-of-distribution generalization check.

Run:
    python -m tools.merge_behaviour_datasets --out dataset/behaviour_merged
"""

from __future__ import annotations

import argparse
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from tools.analyse_labelled import _source_clip

#: The classes the pipeline actually surfaces. Order defines the new class ids.
CANONICAL: tuple[str, ...] = ("read", "sleep", "using_device", "write")

#: Source-label -> canonical name, per dataset. Anything not listed is dropped.
OURS_MAP = {
    "read": "read",
    "sleep": "sleep",
    "using_device": "using_device",
    "write": "write",
}
NEW_MAP = {
    "Reading": "read",
    "Sleeping": "sleep",
    "Using Phone": "using_device",
    "Writing": "write",
}

#: Clips from the first dataset held out for validation (same choice as
#: tools/make_split.py, so results stay comparable to the earlier model).
VAL_CLIPS: tuple[str, ...] = ("owais-class-2_mp4", "taimoor-class-3_mp4")


def _remap_label_file(
    src: Path, names: list[str], mapping: dict[str, str]
) -> list[str]:
    """Rewrite one YOLO label file to canonical class ids, dropping others.

    Args:
        src: Source ``.txt`` label file.
        names: The source dataset's class-name list.
        mapping: Source-name -> canonical-name for classes to keep.

    Returns:
        Rewritten label lines. Empty if no kept class appears -- the caller
        decides whether to copy an image with no remaining labels.
    """
    out: list[str] = []
    if not src.is_file():
        return out
    for line in src.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        source_name = names[int(float(parts[0]))]
        canon = mapping.get(source_name)
        if canon is None:
            continue
        out.append(" ".join([str(CANONICAL.index(canon)), *parts[1:]]))
    return out


def build(out_root: Path, keep_empty: bool) -> None:
    ours = Path("dataset/dataset")
    new = Path("dataset/23-08")

    ours_names = yaml.safe_load((ours / "data.yaml").read_text(encoding="utf-8"))["names"]
    new_names = yaml.safe_load((new / "data.yaml").read_text(encoding="utf-8"))["names"]

    if out_root.exists():
        shutil.rmtree(out_root)
    for split in ("train", "val"):
        (out_root / split / "images").mkdir(parents=True)
        (out_root / split / "labels").mkdir(parents=True)

    counts: dict[str, Counter[str]] = {"train": Counter(), "val": Counter()}
    images: Counter[str] = Counter()
    dropped_empty = 0

    def emit(img: Path, lines: list[str], split: str, prefix: str) -> None:
        nonlocal dropped_empty
        if not lines and not keep_empty:
            dropped_empty += 1
            return
        stem = f"{prefix}_{img.stem}"
        shutil.copy2(img, out_root / split / "images" / f"{stem}{img.suffix}")
        (out_root / split / "labels" / f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        images[split] += 1
        for ln in lines:
            counts[split][CANONICAL[int(ln.split()[0])]] += 1

    # --- dataset 1: split by source clip -----------------------------------
    by_clip: defaultdict[str, list[Path]] = defaultdict(list)
    for img in sorted((ours / "images").glob("*.jpg")):
        by_clip[_source_clip(img.stem)].append(img)
    for clip, clip_images in by_clip.items():
        split = "val" if clip in VAL_CLIPS else "train"
        for img in clip_images:
            lines = _remap_label_file(
                ours / "labels" / f"{img.stem}.txt", ours_names, OURS_MAP
            )
            emit(img, lines, split, "d1")

    # --- dataset 2: its own train/valid go to train; its test is NEVER used
    # here, so it stays a clean out-of-distribution check.
    for src_split in ("train", "valid"):
        for img in sorted((new / src_split / "images").glob("*.jpg")):
            lines = _remap_label_file(
                new / src_split / "labels" / f"{img.stem}.txt", new_names, NEW_MAP
            )
            emit(img, lines, "train", "d2")

    (out_root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(out_root.resolve()),
                "train": "train/images",
                "val": "val/images",
                "nc": len(CANONICAL),
                "names": list(CANONICAL),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    print(f"images: train {images['train']}  val {images['val']}")
    print(f"images dropped for having no kept label: {dropped_empty}"
          f"{' (kept as negatives instead)' if keep_empty else ''}")
    print(f"\n{'class':<14}{'train':>8}{'val':>8}")
    for name in CANONICAL:
        print(f"{name:<14}{counts['train'][name]:>8}{counts['val'][name]:>8}")
    print(f"{'TOTAL':<14}{sum(counts['train'].values()):>8}"
          f"{sum(counts['val'].values()):>8}")
    missing = [n for n in CANONICAL if counts["val"][n] == 0]
    if missing:
        print(f"\nWARNING: no val boxes for {missing} -- unmeasurable by this split.")
    print(f"\nwrote {out_root / 'data.yaml'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="dataset/behaviour_merged")
    parser.add_argument(
        "--keep-empty", action="store_true",
        help=(
            "Keep images that have no remaining label as explicit negatives. "
            "Off by default: dataset 1 labels every student, so an image whose "
            "only labels were look_forward is genuinely 'attentive class', but "
            "including thousands of them would swamp the four real classes."
        ),
    )
    args = parser.parse_args()
    build(Path(args.out), args.keep_empty)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
