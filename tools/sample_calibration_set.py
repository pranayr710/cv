"""Pick a representative subset to hand-score, and copy it out for labelling.

The scoring model is trained on the folder labels, which are binary. A 1-10
score needs something the binary labels cannot supply: evidence that the
model's confidence actually tracks how engaged a person looks. That evidence is
a few hundred images a human has scored on an ordinal scale.

Which few hundred matters. Taking the first N, or a flat random draw, hands you
a set dominated by whichever stratum happens to be largest -- here the 5-10 and
11-15 age groups have 1000 images each while 16-21 Boy has 150, so a flat draw
would be about seven times more likely to ask about one than the other, and the
calibration would be fitted mostly to children.

So the draw is stratified across class x age x gender, proportional to each
stratum's share but with a floor, so small strata are represented rather than
rounded away. Sampling is seeded, so the same set comes out on every run and
the labels stay attached to the images they were made for.

    python tools/sample_calibration_set.py
    python tools/sample_calibration_set.py --n 400 --seed 7
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path("labeling/images/Final Dataset 256/Final Dataset 256")
OUT = Path("labeling/images/calibration")
MANIFEST = OUT / "manifest.json"

#: Smallest number of images to take from any stratum that has any at all.
#: Without this, a stratum holding 1% of the data rounds to two or three images
#: and the calibration says nothing about it.
MIN_PER_STRATUM = 8

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def strata(root: Path) -> dict[tuple[str, str, str], list[Path]]:
    """Group every image by (class, age band, gender) from its path."""
    groups: dict[tuple[str, str, str], list[Path]] = defaultdict(list)
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        parts = path.relative_to(root).parts
        if len(parts) < 4:
            continue
        groups[(parts[0], parts[1], parts[2])].append(path)
    return groups


def allocate(groups: dict[tuple[str, str, str], list[Path]],
             total: int) -> dict[tuple[str, str, str], int]:
    """How many to draw from each stratum.

    Proportional to size, floored at :data:`MIN_PER_STRATUM`, then trimmed back
    from the largest strata if the floors push the total over budget.
    """
    overall = sum(len(v) for v in groups.values())
    if overall == 0:
        return {}
    quota = {k: max(MIN_PER_STRATUM, round(total * len(v) / overall))
             for k, v in groups.items()}
    for key, cap in ((k, len(v)) for k, v in groups.items()):
        quota[key] = min(quota[key], cap)

    # Trim the largest strata first, so the floors survive the correction.
    while sum(quota.values()) > total:
        biggest = max(quota, key=lambda k: quota[k])
        if quota[biggest] <= MIN_PER_STRATUM:
            break
        quota[biggest] -= 1
    return quota


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=350,
                    help="how many images to draw (default 350)")
    ap.add_argument("--seed", type=int, default=0,
                    help="sampling seed; the same seed gives the same set")
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    if not args.root.is_dir():
        print(f"dataset not found at {args.root}")
        return 1

    groups = strata(args.root)
    if not groups:
        print("no images found")
        return 1
    quota = allocate(groups, args.n)

    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = []
    print(f"{sum(len(v) for v in groups.values())} images in "
          f"{len(groups)} strata -> drawing {sum(quota.values())}\n")
    for key in sorted(groups):
        cls, age, gender = key
        picks = rng.sample(groups[key], quota[key])
        print(f"  {cls:<14} {age:<14} {gender:<5} "
              f"{len(groups[key]):>5} -> {len(picks):>3}")
        for path in picks:
            # Flat destination with the stratum encoded in the name, so the
            # labelling tool needs no directory logic and a stray file cannot
            # lose its provenance.
            dest_name = f"{cls}__{age}__{gender}__{path.name}".replace(" ", "-")
            shutil.copy2(path, args.out / dest_name)
            manifest.append({
                "file": dest_name,
                "source": str(path).replace("\\", "/"),
                "folder_label": cls,
                "age_band": age,
                "gender": gender,
            })

    rng.shuffle(manifest)  # present them mixed, not class by class
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    print(f"\n  {len(manifest)} images copied to {args.out}")
    print(f"  manifest: {MANIFEST}")
    print("\n  The folder label is recorded but NOT shown while scoring -- "
          "seeing it\n  first would anchor the score to the label being "
          "validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
