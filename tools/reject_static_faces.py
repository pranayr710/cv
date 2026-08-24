"""Flag identities that are printed faces (posters) rather than students.

Runs :func:`backend.identity.is_static_face` over a finished JSONL run, using
the source video to sample each identity's face crops, and rewrites the JSONL
with poster identities marked. Kept as a separate post-processing pass rather
than folded into the live pipeline for two reasons: it needs an identity's WHOLE
lifetime to judge invariance (unavailable mid-stream), and it needs to re-read
the video for pixels the JSONL deliberately does not store.

Found necessary by ``tools/audit_identity.py``, which showed two wall posters
being tracked and profiled as students for 27 and 20 sightings on real footage.

Marked, not deleted: each affected person gets ``"is_poster": true`` so the
rejection is visible and auditable. ``backend.student_profile`` then excludes
them from the student roster.

Run:
    python -m tools.reject_static_faces \\
        --jsonl outputs/wa_full_video.jsonl \\
        --out outputs/wa_no_posters.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

#: Crops sampled per identity, spread evenly across its lifetime. More than
#: enough for a stable invariance estimate without re-decoding the whole video.
SAMPLE_CROPS = 10


def reject(jsonl: Path, video: Path, out: Path) -> None:
    import cv2

    from backend.config import CONFIG
    from backend.identity import appearance_invariance, is_static_face

    records = [
        json.loads(line)
        for line in jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    sightings: dict[int, list[tuple[int, list[int]]]] = defaultdict(list)
    for rec in records:
        for person in rec["persons"]:
            pid, face = person.get("person_id"), person.get("face")
            if pid is None or not face or not face.get("bbox"):
                continue
            sightings[pid].append((rec["frame_id"], face["bbox"]))

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video}")

    posters: set[int] = set()
    print(f"{'id':>5}{'sightings':>11}{'invariance':>13}  verdict")
    for pid in sorted(sightings, key=lambda k: (k < 0, abs(k))):
        entries = sorted(sightings[pid])
        step = max(1, len(entries) // SAMPLE_CROPS)
        crops = []
        for frame_id, (x, y, w, h) in entries[::step][:SAMPLE_CROPS]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ok, frame = cap.read()
            if not ok:
                continue
            crop = frame[max(0, y):y + h, max(0, x):x + w]
            if crop.size:
                crops.append(crop)

        score = appearance_invariance(crops)
        static = is_static_face(crops, CONFIG.identity)
        if static:
            posters.add(pid)
        score_s = f"{score:.3f}" if score is not None else "n/a"
        verdict = (
            "POSTER (rejected)" if static
            else ("student" if score is not None else "too few crops to judge")
        )
        print(f"{pid:>5}{len(entries):>11}{score_s:>13}  {verdict}")
    cap.release()

    for rec in records:
        for person in rec["persons"]:
            person["is_poster"] = person.get("person_id") in posters

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")

    print(f"\n{len(posters)} identity/identities rejected as printed faces: "
          f"{sorted(posters) or 'none'}")
    print(f"wrote {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument(
        "--video",
        default="dataset/23-08/vedio/WhatsApp Video 2026-08-23 at 10.16.09.mp4",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    reject(Path(args.jsonl), Path(args.video), Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
