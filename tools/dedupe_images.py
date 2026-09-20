"""Find and remove duplicate images in a labelled image dataset.

Duplicates in a classification set are not merely wasted disk. A frame repeated
twelve times is twelve votes for one example, and if the copies straddle a
train/test split the model is scored partly on pictures it memorised.

Three kinds are treated differently, because they are not the same problem:

* **Exact duplicates within one class** -- byte-identical files under the same
  label. Safe to collapse to one copy.
* **Exact duplicates across classes** -- the same bytes filed under two
  different labels. This is a contradiction in the dataset, not a duplicate,
  and deleting either copy silently decides which label was right. These are
  reported and quarantined, never auto-resolved.
* **Near-duplicates** -- visually the same image re-encoded or resized, so the
  bytes differ. Caught by a perceptual hash, opt-in via ``--near``, because the
  threshold is a judgement call rather than a fact.

Nothing is deleted. Removals are MOVED to a quarantine folder, so a bad call is
undone by moving them back.

    python tools/dedupe_images.py "labeling/images/Final Dataset 256"
    python tools/dedupe_images.py "<dir>" --apply
    python tools/dedupe_images.py "<dir>" --near 3 --apply

Do NOT run this over a video frame sequence such as a DIPSER subject's
images/. Consecutive frames there are near-identical by design, and collapsing
them destroys the time series the windowing depends on.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
from collections import defaultdict
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

#: Where the class name sits in the path, relative to the scanned root. For
#: ``<root>/<inner>/Engagement/<age>/<gender>/x.jpg`` the class is 2 levels in.
#: Overridable, since not every dataset nests the same way.
DEFAULT_CLASS_DEPTH = 1


def file_hash(path: Path) -> str:
    """MD5 of the file's bytes. Collisions are not a practical concern here."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def perceptual_hash(path: Path) -> int | None:
    """A 64-bit difference hash: is this the same picture, re-encoded?

    Downscales to 9x8 greyscale and records whether each pixel is brighter than
    the one to its right. Resizing, re-compression and small quality changes
    leave the comparisons intact; a genuinely different picture does not.

    Returns ``None`` if the file cannot be read as an image.
    """
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover - Pillow ships with the project
        return None
    try:
        with Image.open(path) as im:
            small = im.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
            pixels = list(small.getdata())
    except Exception:  # noqa: BLE001 - an unreadable file is not fatal here
        return None
    bits = 0
    for row in range(8):
        for col in range(8):
            left = pixels[row * 9 + col]
            right = pixels[row * 9 + col + 1]
            bits = (bits << 1) | int(left > right)
    return bits


def class_of(path: Path, root: Path, depth: int) -> str:
    """The label a file sits under, or ``"?"`` if the path is too shallow."""
    parts = path.relative_to(root).parts
    return parts[depth] if len(parts) > depth else "?"


def keeper(group: list[Path]) -> Path:
    """Which copy of a duplicate set to keep.

    Largest file first -- for re-encoded copies that is the least-compressed
    one -- then shortest path, then alphabetical, so the choice is the same on
    every run rather than depending on directory order.
    """
    return min(group, key=lambda p: (-p.stat().st_size, len(str(p)), str(p)))


def scan(root: Path, depth: int) -> list[Path]:
    return sorted(p for p in root.rglob("*")
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="directory to scan")
    ap.add_argument("--apply", action="store_true",
                    help="actually move duplicates (default: report only)")
    ap.add_argument("--near", type=int, default=None, metavar="N",
                    help="also treat images within Hamming distance N (try 3) "
                         "as duplicates; slower, and the threshold is a "
                         "judgement call")
    ap.add_argument("--class-depth", type=int, default=DEFAULT_CLASS_DEPTH,
                    help="path component holding the class label")
    ap.add_argument("--quarantine", type=Path, default=None,
                    help="where removals go (default: <root>/../_duplicates)")
    args = ap.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        print(f"not a directory: {root}")
        return 1
    quarantine = (args.quarantine or root.parent / "_duplicates").resolve()

    files = scan(root, args.class_depth)
    print(f"scanning {len(files)} images under {root.name}\n")

    by_hash: dict[str, list[Path]] = defaultdict(list)
    for p in files:
        by_hash[file_hash(p)].append(p)

    same_class: list[list[Path]] = []
    cross_class: list[list[Path]] = []
    for group in by_hash.values():
        if len(group) < 2:
            continue
        labels = {class_of(p, root, args.class_depth) for p in group}
        (cross_class if len(labels) > 1 else same_class).append(group)

    if args.near is not None:
        # Only compare files that survived exact dedup, so the expensive pass
        # runs on the smaller set.
        survivors = [keeper(g) for g in same_class] + \
                    [g[0] for g in by_hash.values() if len(g) == 1]
        buckets: dict[int, list[Path]] = defaultdict(list)
        for p in survivors:
            h = perceptual_hash(p)
            if h is not None:
                buckets[h].append(p)
        hashes = list(buckets)
        seen: set[int] = set()
        for i, a in enumerate(hashes):
            if a in seen:
                continue
            near = [a]
            for b in hashes[i + 1:]:
                if b not in seen and (a ^ b).bit_count() <= args.near:
                    near.append(b)
                    seen.add(b)
            if len(near) > 1:
                merged = [p for h in near for p in buckets[h]]
                labels = {class_of(p, root, args.class_depth) for p in merged}
                (cross_class if len(labels) > 1 else same_class).append(merged)

    removable = [p for g in same_class for p in g if p != keeper(g)]
    per_class: dict[str, int] = defaultdict(int)
    totals: dict[str, int] = defaultdict(int)
    for p in files:
        totals[class_of(p, root, args.class_depth)] += 1
    for p in removable:
        per_class[class_of(p, root, args.class_depth)] += 1

    print(f"  duplicate groups within a class : {len(same_class)}")
    print(f"  redundant copies to remove      : {len(removable)} "
          f"({100 * len(removable) / max(len(files), 1):.1f}%)")
    print(f"  images remaining after removal  : {len(files) - len(removable)}\n")
    print("  per class:")
    for c in sorted(totals):
        kept = totals[c] - per_class[c]
        print(f"    {c:<18} {totals[c]:>6} -> {kept:>6}  "
              f"(-{per_class[c]}, {100 * per_class[c] / totals[c]:.1f}%)")

    if cross_class:
        print(f"\n  !! {len(cross_class)} groups hold the SAME image under "
              f"DIFFERENT labels.")
        print("     These are contradictions, not duplicates. Neither copy is "
              "removed;\n     resolve them by hand or drop both.")
        for group in cross_class[:5]:
            labels = sorted({class_of(p, root, args.class_depth) for p in group})
            print(f"       {labels}: {group[0].name}")
        if len(cross_class) > 5:
            print(f"       ... and {len(cross_class) - 5} more")
        listing = quarantine.parent / "cross_class_conflicts.txt"

    if not args.apply:
        print("\n  dry run -- nothing moved. Re-run with --apply to act.")
        return 0

    quarantine.mkdir(parents=True, exist_ok=True)
    moved = 0
    for p in removable:
        dest = quarantine / p.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            dest = dest.with_name(f"{dest.stem}__{moved}{dest.suffix}")
        shutil.move(str(p), str(dest))
        moved += 1

    if cross_class:
        listing.parent.mkdir(parents=True, exist_ok=True)
        with listing.open("w", encoding="utf-8") as fh:
            for group in cross_class:
                labels = sorted({class_of(p, root, args.class_depth)
                                 for p in group})
                fh.write(f"{labels}\n")
                for p in group:
                    fh.write(f"    {p}\n")
        print(f"\n  conflicts listed in {listing}")

    print(f"\n  moved {moved} duplicates to {quarantine}")
    print("  they are not deleted -- move them back to undo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
