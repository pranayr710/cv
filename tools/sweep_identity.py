"""Sweep identity settings against a KNOWN headcount, from a single video pass.

`IdentityConfig.match_threshold` has been 0.35 since it was written -- a ballpark
figure for this embedding family, never calibrated against this project's own
footage, because there was no identity ground truth to calibrate against. There
is now: the audited clip contains 7 students and 1 teacher, confirmed by the
person who recorded it. That makes the threshold gradeable for the first time.

Sweeping it naively means re-running the whole pipeline per candidate value,
which is dominated by detection and face embedding -- the parts the threshold
does not affect at all. So this does the expensive work ONCE, holds the
embeddings in memory, and re-resolves identity per candidate. Pose, expression
and behaviour are skipped entirely; none of them influence identity.

Deliberately in memory and never on disk. backend/identity.py states that no
face embedding is written to output, and a sweep cache would quietly break that
for the sake of convenience.

Reported per candidate:
    ids         distinct face-verified identities passing the frame minimum
    error       ids minus the true student count -- the number that matters
    dup frames  frames containing a repeated id, which must stay 0
    no id %     detections left unidentified, i.e. students dropped from output

Run:
    python -m tools.sweep_identity --students 7 \\
        --video "dataset/23-08/vedio/WhatsApp Video 2026-08-23 at 10.16.09.mp4"
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_THRESHOLDS = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)


def collect(video: Path, sample_rate: int, limit: int | None) -> list[dict]:
    """One expensive pass: per frame, the track ids and face embeddings only."""
    import cv2

    from backend.detection import Detector
    from backend.face import FaceAnalyzer
    from backend.tracking import PersonTracker

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video}")

    frames: list[dict] = []
    detector = Detector()
    tracker = PersonTracker()
    index = 0
    with FaceAnalyzer() as faces:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % sample_rate == 0:
                persons, _objects = detector.detect(frame)
                boxes = [p.bbox for p in persons]
                results = faces.analyze(frame, boxes)
                frames.append({
                    "frame_id": index,
                    "track_ids": tracker.update(persons),
                    "embeddings": [r.embedding for r in results],
                    "scores": [r.score for r in results],
                })
                if len(frames) % 25 == 0:
                    print(f"  ...{len(frames)} frames collected", flush=True)
                if limit and len(frames) >= limit:
                    break
            index += 1
    cap.release()
    total = sum(len(f["track_ids"]) for f in frames)
    print(f"collected {len(frames)} frames, {total} person detections "
          f"(sample_rate={sample_rate})\n")
    return frames


def evaluate(frames: list[dict], cfg, min_frames: int) -> dict:
    """Re-resolve identity under `cfg` and grade the result."""
    from backend.identity import TwoPassIdentityResolver

    resolver = TwoPassIdentityResolver(cfg)
    keyed = []
    for f in frames:
        keys = resolver.keys_for(f["track_ids"], f["embeddings"], f["scores"])
        resolver.observe(keys, f["embeddings"], f["scores"])
        keyed.append(keys)
    mapping = resolver.finalise()

    seen = Counter()
    dup_frames = 0
    unidentified = total = 0
    for keys in keyed:
        here = Counter()
        for key in keys:
            total += 1
            pid = mapping.get(key) if key is not None else None
            if pid is None:
                unidentified += 1
                continue
            here[pid] += 1
            seen[pid] += 1
        if any(n > 1 for pid, n in here.items() if pid > 0):
            dup_frames += 1

    ids = [p for p, n in seen.items() if p > 0 and n >= min_frames]
    return {
        "ids": len(ids),
        "dup_frames": dup_frames,
        "unidentified_pct": 100 * unidentified / total if total else 0.0,
        "id_list": sorted(ids),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        default="dataset/23-08/vedio/WhatsApp Video 2026-08-23 at 10.16.09.mp4",
    )
    parser.add_argument(
        "--students", type=int, required=True,
        help="True student count for this clip, excluding the instructor.",
    )
    parser.add_argument(
        "--instructors", type=int, default=1,
        help="True instructor count; added to --students for the target, since "
             "identity resolution has no notion of role.",
    )
    parser.add_argument("--sample-rate", type=int, default=30)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--thresholds", default=",".join(str(t) for t in DEFAULT_THRESHOLDS)
    )
    args = parser.parse_args()

    from backend.config import CONFIG

    target = args.students + args.instructors
    frames = collect(Path(args.video), args.sample_rate, args.limit)
    min_frames = CONFIG.profile.min_frames_for_profile

    print(f"target: {target} people ({args.students} students + "
          f"{args.instructors} instructor)\n")
    print(f"{'thresh':>7}{'ids':>6}{'error':>7}{'dup frames':>12}{'no id %':>9}")
    rows = []
    for thresh in (float(t) for t in args.thresholds.split(",")):
        cfg = replace(CONFIG.identity, match_threshold=thresh)
        r = evaluate(frames, cfg, min_frames)
        err = r["ids"] - target
        rows.append((thresh, r, abs(err)))
        print(f"{thresh:>7.2f}{r['ids']:>6}{err:>+7}{r['dup_frames']:>12}"
              f"{r['unidentified_pct']:>9.1f}")

    best = min(rows, key=lambda x: (x[2], x[0]))
    print(f"\nclosest to truth: threshold {best[0]:.2f} -> {best[1]['ids']} ids "
          f"(target {target}), ids {best[1]['id_list']}")
    if any(r[1]["dup_frames"] for r in rows):
        print("NOTE: a non-zero dup-frames column means the co-occurrence "
              "constraint is not holding -- that is a bug, not a tuning choice.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
