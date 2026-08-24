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

# Context padding around the tight face bbox. A tight crop throws away cues
# (hair, shoulders, mouth corners cut off) that help a human read expression
# even when the face itself is low-resolution -- the same reasoning
# tools/audit_identity.py uses padding for, applied here for the same reason.
PAD_FRAC = 0.35

# Lanczos resamples sharper than cubic at this magnification (crops are
# routinely upsampled 6-9x from a <35px source), and a mild unsharp mask on
# top recovers perceived edge contrast that heavy upsampling smooths away.
# Neither invents detail that was not in the source -- a face that is
# genuinely too small/blurry to read will still be too small/blurry to read,
# and should be labelled "unclear" rather than guessed at. That is a real
# finding for this study, not a tooling failure to hide.
SHARPEN_AMOUNT = 0.6


def _crop_for_display(frame, bbox: tuple[int, int, int, int]):
    """Crop with context padding, then upsample as legibly as the source
    allows. Never claims to recover detail the sensor never captured."""
    import cv2

    x, y, w, h = bbox
    pad_x, pad_y = int(w * PAD_FRAC), int(h * PAD_FRAC)
    fh, fw = frame.shape[:2]
    x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
    x1, y1 = min(fw, x + w + pad_x), min(fh, y + h + pad_y)
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return None

    resized = cv2.resize(crop, (DISPLAY_PX, DISPLAY_PX), interpolation=cv2.INTER_LANCZOS4)
    blurred = cv2.GaussianBlur(resized, (0, 0), sigmaX=1.2)
    sharpened = cv2.addWeighted(resized, 1 + SHARPEN_AMOUNT, blurred, -SHARPEN_AMOUNT, 0)
    return sharpened


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
    # floor so a rare-but-important condition (e.g. very few frontal faces)
    # isn't sampled down to near-zero and silently excluded from the study.
    #
    # The floor is RESERVED upfront, before any proportional allocation --
    # measured bug in an earlier version: allocating tiny and off-angle first
    # (in a fixed iteration order) could exhaust the budget before frontal's
    # turn, so its floor guarantee only ever got whatever was left over. At
    # n=120 that left frontal with 3 crops instead of its intended floor of
    # 10, silently defeating the whole point of having a floor.
    buckets = [b for b in ("tiny", "off-angle", "frontal") if by_bucket[b]]
    floor = max(5, n // (4 * max(1, len(buckets))))
    reserved = {b: min(floor, len(by_bucket[b])) for b in buckets}
    extra_budget = max(0, n - sum(reserved.values()))

    extra_pool_total = sum(len(by_bucket[b]) - reserved[b] for b in buckets)
    shares = dict(reserved)
    if extra_pool_total > 0:
        for b in buckets:
            available = len(by_bucket[b]) - reserved[b]
            if available <= 0:
                continue
            extra = round(extra_budget * available / extra_pool_total)
            shares[b] += min(extra, available)

    picked: list[dict] = []
    for b in buckets:
        share = min(shares[b], len(by_bucket[b]))
        picked.extend(rng.sample(by_bucket[b], share))
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
        crop = _crop_for_display(frame, c["bbox"])
        if crop is None:
            continue
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
