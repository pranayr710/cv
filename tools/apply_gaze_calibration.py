"""Recompute gaze_label in a finished JSONL using a calibrated yaw reference.

Purely a post-processing pass: raw yaw/pitch were already measured correctly
by SixDRepNet and are left untouched. Only the *label* derived from them
(:func:`backend.headpose.classify_gaze`) is recomputed, using
``tools/calibrate_gaze.py``'s reference for this camera instead of the
uncalibrated default. No detection, tracking, or re-identification is re-run
-- this is why it takes seconds, not the original run's several minutes.

Exists because a real mistake was caught rather than shipped: a finished run
was reported (and a per-student profile built from it) BEFORE calibration was
applied, and the "concentration" numbers in that profile were consequently
near-meaningless for this camera (~100% almost everywhere, from mislabelling
most attending students as "right" instead of "teacher"). This closes that
gap on the already-computed data instead of re-running the whole pipeline.

Run:
    python -m tools.apply_gaze_calibration \\
        --jsonl outputs/wa_continuous_v2.jsonl \\
        --out outputs/wa_continuous_calibrated.jsonl \\
        --yaw-reference 48.7
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path


def apply(jsonl_path: Path, out_path: Path, yaw_reference: float) -> None:
    from backend.config import CONFIG
    from backend.headpose import classify_gaze

    cfg = replace(CONFIG.headpose, yaw_reference_deg=yaw_reference)

    changed = 0
    total = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open(encoding="utf-8") as src, out_path.open(
        "w", encoding="utf-8"
    ) as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            for person in record["persons"]:
                hp = person.get("head_pose")
                if hp is None:
                    continue
                total += 1
                old_label = hp["gaze_label"]
                # yaw/pitch are the model's raw measurement -- never touched,
                # only the derived label is recomputed under calibration.
                hp["gaze_label"] = classify_gaze(hp["yaw"], hp["pitch"], cfg)
                if hp["gaze_label"] != old_label:
                    changed += 1
            dst.write(json.dumps(record) + "\n")

    print(f"{jsonl_path} -> {out_path}")
    print(f"gaze_label recomputed with yaw_reference_deg={yaw_reference}")
    print(f"{changed}/{total} face readings changed label ({changed/total*100:.1f}%)"
          if total else "no head_pose readings found")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--yaw-reference", type=float, required=True)
    args = parser.parse_args()
    apply(Path(args.jsonl), Path(args.out), args.yaw_reference)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
