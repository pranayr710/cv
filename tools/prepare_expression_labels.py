"""Sample face crops for the expression-validation study, stratified by the
conditions the literature review flagged as where accuracy actually drops.

Never trust AffectNet-trained accuracy to transfer to classroom footage --
docs/LITERATURE_REVIEW.md section 4 cites HSEmotion's own team finding their
larger, higher-benchmark-accuracy models generalise WORSE cross-dataset, and
occlusion/off-angle studies reporting 57-61% relative accuracy drops. The only
way to know our own number is to measure it on our own crops against human
judgement.

Stratification (approximating the review's "near-frontal >=50px / off-angle /
<35px" bins from data we already compute, no new detection needed):
    tiny       face height < 35px
    off-angle  |yaw| >= 20 degrees (checked first, so a small AND off-angle
               face is not double counted -- tininess is usually the harder
               condition, so it takes priority)
    frontal    everything else (>=35px, |yaw| < 20)

Output is written entirely to local disk. Nothing here uploads or publishes
face crops anywhere -- see the privacy section of docs/LITERATURE_REVIEW.md
(problem 7) for why that boundary matters for real, if anonymised, students.

Run:
    python -m tools.prepare_expression_labels --jsonl outputs/wa_no_posters.jsonl \\
        --n 120 --out outputs/expression_labels
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TINY_PX = 35
OFFANGLE_DEG = 20.0
DISPLAY_PX = 220  # upsampled crop size so a human can actually judge it


def _bucket(face_h: float, yaw: float | None) -> str:
    if face_h < TINY_PX:
        return "tiny"
    if yaw is not None and abs(yaw) >= OFFANGLE_DEG:
        return "off-angle"
    return "frontal"


def prepare(jsonl: Path, video: Path, out: Path, n: int, seed: int) -> None:
    import cv2

    records = [
        json.loads(line)
        for line in jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    candidates: list[dict] = []
    for rec in records:
        for person in rec["persons"]:
            face = person.get("face")
            if not face or not face.get("bbox"):
                continue
            fx, fy, fw, fh = face["bbox"]
            if fw <= 0 or fh <= 0:
                continue
            hp = person.get("head_pose") or {}
            expr = person.get("expression") or {}
            candidates.append({
                "frame_id": rec["frame_id"],
                "person_id": person.get("person_id"),
                "bbox": (fx, fy, fw, fh),
                "bucket": _bucket(fh, hp.get("yaw")),
                "model_label": expr.get("label"),
                "model_confidence": expr.get("confidence"),
            })

    if not candidates:
        raise SystemExit(
            "No face crops with a bbox in this JSONL -- nothing to sample. "
            "Was expression/face wiring run for this file?"
        )

    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for c in candidates:
        by_bucket[c["bucket"]].append(c)

    rng = random.Random(seed)
    # Proportional to what exists, but every non-empty bucket gets at least a
    # floor so a rare-but-important condition (e.g. very few off-angle faces)
    # isn't sampled down to zero and silently excluded from the study.
    buckets = [b for b in ("tiny", "off-angle", "frontal") if by_bucket[b]]
    floor = max(5, n // (4 * max(1, len(buckets))))
    picked: list[dict] = []
    remaining = n
    for i, b in enumerate(buckets):
        pool = by_bucket[b]
        share = max(floor, round(n * len(pool) / len(candidates)))
        share = min(share, len(pool), remaining if i == len(buckets) - 1 else share)
        picked.extend(rng.sample(pool, min(share, len(pool))))
        remaining -= share
    rng.shuffle(picked)

    print(f"{len(candidates)} candidate face crops across buckets: "
          + ", ".join(f"{b}={len(by_bucket[b])}" for b in buckets))
    print(f"sampled {len(picked)} crops for labelling")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video}")

    crops_dir = out / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.csv"

    manifest_rows = []
    written = 0
    for idx, c in enumerate(picked):
        cap.set(cv2.CAP_PROP_POS_FRAMES, c["frame_id"])
        ok, frame = cap.read()
        if not ok:
            continue
        x, y, w, h = c["bbox"]
        crop = frame[max(0, y):y + h, max(0, x):x + w]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (DISPLAY_PX, DISPLAY_PX), interpolation=cv2.INTER_CUBIC)
        crop_id = f"crop_{idx:04d}"
        cv2.imwrite(str(crops_dir / f"{crop_id}.png"), crop)
        manifest_rows.append({
            "crop_id": crop_id,
            "bucket": c["bucket"],
            "model_label": c["model_label"] or "",
            "model_confidence": c["model_confidence"] if c["model_confidence"] is not None else "",
            # Kept for our own traceability only -- never shown to labellers,
            # never leaves this machine, and identifies nothing beyond a
            # within-video track number already scoped as anonymous.
            "_frame_id": c["frame_id"],
            "_person_id": c["person_id"],
        })
        written += 1
    cap.release()

    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"wrote {written} crops to {crops_dir}")
    print(f"wrote manifest to {manifest_path}")
    print(
        "\nNext: two people independently run, e.g.\n"
        "  python -m tools.label_expressions --labeller you\n"
        "  python -m tools.label_expressions --labeller rater2\n"
        "then:\n"
        "  python -m tools.score_expression_labels"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", default="outputs/wa_no_posters.jsonl")
    parser.add_argument(
        "--video",
        default="dataset/23-08/vedio/WhatsApp Video 2026-08-23 at 10.16.09.mp4",
    )
    parser.add_argument("--out", default="outputs/expression_labels")
    parser.add_argument("--n", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    prepare(Path(args.jsonl), Path(args.video), Path(args.out), args.n, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
