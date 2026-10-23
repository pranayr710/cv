"""Are the high-overlap box pairs one student twice, or two students?

NMS runs at iou=0.50, so no pair of person boxes can exceed that -- which
means a double detection cannot look like a near-copy. It gets clipped to just
under the NMS ceiling instead, landing in the same 0.4-0.5 band as a student
genuinely sitting behind another.

IoU alone therefore cannot separate the two cases on this pipeline's output,
and a gate set anywhere in that band would merge real students as readily as
it would repair split ones. Shape is the discriminator: two detections of one
person have nearly the same width, height and centre, while a student behind
another is smaller and offset, because they are further from the camera.

    python tools/diagnose/inspect_overlaps.py outputs/final2/raw.jsonl
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.identity import _iou

#: Pairs below this are plainly separate people and not worth examining.
LOOK_ABOVE: float = 0.35


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw", type=Path, nargs="?",
                    default=Path("outputs/final2/raw.jsonl"))
    args = ap.parse_args()

    size_ratios: list[float] = []
    centre_shifts: list[float] = []
    same_track: int = 0
    examined: int = 0

    for line in args.raw.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        persons = [p for p in json.loads(line).get("persons", [])
                   if p.get("bbox")]
        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                a, b = persons[i], persons[j]
                if _iou(a["bbox"], b["bbox"]) < LOOK_ABOVE:
                    continue
                examined += 1
                ah, bh = a["bbox"][3], b["bbox"][3]
                aw, bw = a["bbox"][2], b["bbox"][2]
                if not ah or not bh:
                    continue
                # 1.0 means identically sized; a student one row back is
                # measurably smaller.
                size_ratios.append(min(ah, bh) / max(ah, bh))
                acx = a["bbox"][0] + aw / 2
                bcx = b["bbox"][0] + bw / 2
                acy = a["bbox"][1] + ah / 2
                bcy = b["bbox"][1] + bh / 2
                diag = max(ah, bh)
                centre_shifts.append(
                    ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5 / diag)
                if a.get("person_id") == b.get("person_id"):
                    same_track += 1

    if not size_ratios:
        print(f"No pairs above IoU {LOOK_ABOVE} in {args.raw}.")
        return 0

    print(f"{examined} pairs above IoU {LOOK_ABOVE} in {args.raw}\n")
    print("  height ratio (1.0 = same size)")
    print(f"    median {statistics.median(size_ratios):.3f}   "
          f"min {min(size_ratios):.3f}   max {max(size_ratios):.3f}")
    print("  centre separation, as a fraction of box height")
    print(f"    median {statistics.median(centre_shifts):.3f}   "
          f"min {min(centre_shifts):.3f}   max {max(centre_shifts):.3f}")
    print(f"\n  pairs already sharing a person_id: {same_track}")

    # A double detection would be near-identical in both measures.
    near_copies = sum(1 for r, c in zip(size_ratios, centre_shifts)
                      if r > 0.9 and c < 0.15)
    print(f"  pairs that look like one person twice "
          f"(height ratio > 0.9 and centre shift < 0.15): {near_copies}")
    print()
    if near_copies == 0:
        print("  None. Every high-overlap pair differs in size or position by")
        print("  more than a duplicate box would, so these are students")
        print("  occluding each other, not the detector seeing double.")
    else:
        print(f"  {near_copies} pairs are plausible double detections.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
