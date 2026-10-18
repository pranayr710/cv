"""How often does the detector put two boxes on one student?

The IoU gate in :mod:`backend.identity` exists to stop that case from
recording a cannot-link edge between a person and themselves. Whether it
changes anything on a given recording depends entirely on whether the case
occurs there, and on the audited clip it clearly did -- 11 to 17 ids for 8
people, flat across every match_threshold.

On other footage it may not occur at all, and then the gate is correct and
inert. Those two outcomes look identical from the id count alone, so this
counts the overlapping pairs directly from a finished run rather than
inferring them.

    python tools/measure_double_detection.py outputs/final2/raw.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.identity import OVERLAP_SAME_PERSON, _iou


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw", type=Path, nargs="?",
                    default=Path("outputs/final2/raw.jsonl"),
                    help="a raw.jsonl written by the pipeline")
    args = ap.parse_args()

    if not args.raw.exists():
        print(f"{args.raw} not found.")
        return 1

    frames = 0
    pairs = 0
    suppressed = 0
    worst = 0.0
    affected_frames = 0
    histogram = {"0.0-0.2": 0, "0.2-0.4": 0, "0.4-0.55": 0, "0.55+": 0}

    for line in args.raw.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        persons = [p for p in row.get("persons", []) if p.get("bbox")]
        frames += 1
        hit_here = False
        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                iou = _iou(persons[i]["bbox"], persons[j]["bbox"])
                pairs += 1
                worst = max(worst, iou)
                if iou >= OVERLAP_SAME_PERSON:
                    suppressed += 1
                    hit_here = True
                    histogram["0.55+"] += 1
                elif iou >= 0.4:
                    histogram["0.4-0.55"] += 1
                elif iou >= 0.2:
                    histogram["0.2-0.4"] += 1
                else:
                    histogram["0.0-0.2"] += 1
        if hit_here:
            affected_frames += 1

    print(f"{frames} frames, {pairs} person-pairs examined")
    print(f"gate threshold: IoU >= {OVERLAP_SAME_PERSON}\n")
    print("  overlap distribution")
    for band, count in histogram.items():
        share = count / pairs if pairs else 0.0
        print(f"    IoU {band:<9} {count:>7}  {share:>6.2%}")
    print(f"\n  highest overlap seen        : {worst:.3f}")
    print(f"  pairs the gate suppresses   : {suppressed}")
    print(f"  frames containing one       : {affected_frames}"
          f"  ({affected_frames / frames:.2%})" if frames else "")

    print()
    if suppressed == 0:
        print("  The detector never double-boxed a student in this recording,")
        print("  so the gate is inert here. That is not evidence the fix is")
        print("  unnecessary -- it is evidence this footage does not trigger")
        print("  the bug. The audited clip did, and the unit tests pin the")
        print("  behaviour for when it recurs.")
    else:
        print(f"  {suppressed} cannot-link edges would have been recorded")
        print("  between a student and themselves. Each one splits that")
        print("  student into two ids permanently.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
