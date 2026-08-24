"""Render a contact sheet per person_id so identity can be VERIFIED, not assumed.

The pipeline reports "45 raw tracks -> 10 stable person ids" on real footage.
That is an improvement, but on its own it is not evidence of correctness: 10
could be right, or it could be two students merged into one id plus one student
split across three. Nothing measured so far distinguishes those cases, because
there is no identity ground truth for this footage.

This makes the question answerable by eye. For each ``person_id`` it crops every
face that id was seen with, across the whole video, and tiles them into one
image. Then the two failure modes are directly visible:

* **A merge** (two different students sharing one id) shows up as a sheet
  containing two visibly different people.
* **A split** (one student across several ids) shows up as two different sheets
  containing the same person.

Both are errors, and they are not symmetric. A merge silently attributes one
student's behaviour to another and is the worse failure; a split under-counts
continuity but never mixes two people's data. Reporting them separately matters.

This is a manual visual audit, not an automated metric, and is reported as such
-- a small-N eyeball check that yields a concrete count ("of N ids, M pure"),
which is strictly better than an unverified improvement claim.

Run:
    python -m tools.audit_identity --jsonl outputs/wa_full_video.jsonl \\
        --video "dataset/23-08/vedio/WhatsApp Video 2026-08-23 at 10.16.09.mp4"
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Max crops per contact sheet. Enough to judge identity without an unreadable
#: wall of near-identical frames; sampled evenly across the id's lifetime so the
#: sheet spans the whole appearance rather than one moment.
MAX_CROPS = 12
CROP_PX = 96


def build_sheets(jsonl: Path, video: Path, out_dir: Path) -> None:
    import cv2

    records = [
        json.loads(line)
        for line in jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    # person_id -> [(frame_id, face_bbox), ...]
    sightings: dict[int, list[tuple[int, list[int]]]] = defaultdict(list)
    for rec in records:
        for person in rec["persons"]:
            pid = person.get("person_id")
            face = person.get("face")
            if pid is None or not face or not face.get("bbox"):
                continue
            sightings[pid].append((rec["frame_id"], face["bbox"]))

    if not sightings:
        raise SystemExit(
            "No person_id + face pairs in this JSONL -- nothing to audit. "
            "(Was it produced before the identity wiring?)"
        )

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video}")

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(sightings)} person_ids with at least one face\n")

    for pid in sorted(sightings, key=lambda k: (k < 0, abs(k))):
        entries = sorted(sightings[pid])
        # Even sampling across the id's whole lifetime, not the first N frames.
        step = max(1, len(entries) // MAX_CROPS)
        picked = entries[::step][:MAX_CROPS]

        crops = []
        for frame_id, bbox in picked:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = cap.read()
            if not ok:
                continue
            x, y, w, h = bbox
            # A little context around the face makes a person far easier to
            # recognise by eye than a tight crop does.
            pad = int(max(w, h) * 0.25)
            fh, fw = frame.shape[:2]
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(fw, x + w + pad), min(fh, y + h + pad)
            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            crops.append(cv2.resize(crop, (CROP_PX, CROP_PX)))

        if not crops:
            print(f"  person_id {pid}: no readable crops, skipped")
            continue

        cols = min(6, len(crops))
        rows = (len(crops) + cols - 1) // cols
        sheet = cv2.copyMakeBorder(
            cv2.vconcat([
                cv2.hconcat(
                    crops[r * cols:(r + 1) * cols]
                     + [crops[0] * 0] * (cols - len(crops[r * cols:(r + 1) * cols]))
                )
                for r in range(rows)
            ]),
            28, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0),
        )
        verified = "face-verified" if pid > 0 else "NOT face-verified"
        cv2.putText(
            sheet, f"person_id {pid}  ({len(entries)} sightings, {verified})",
            (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
        )
        dest = out_dir / f"person_{pid:+03d}.jpg"
        cv2.imwrite(str(dest), sheet)
        print(f"  person_id {pid:>3}: {len(entries):>3} sightings, "
              f"{len(crops)} crops -> {dest.name}")

    cap.release()
    print(f"\nContact sheets in {out_dir}")
    print("Audit by eye and record, per id: is it ONE person (pure), two people "
          "merged, or the same person also appearing on another sheet (split)?")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", default="outputs/wa_full_video.jsonl")
    parser.add_argument(
        "--video",
        default="dataset/23-08/vedio/WhatsApp Video 2026-08-23 at 10.16.09.mp4",
    )
    parser.add_argument("--out", default="outputs/identity_audit")
    args = parser.parse_args()
    build_sheets(Path(args.jsonl), Path(args.video), ROOT / args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
