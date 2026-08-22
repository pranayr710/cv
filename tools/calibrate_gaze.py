"""Derive the per-camera yaw reference from an existing Stage 1 JSONL run.

Closes the loop on a real bug: :func:`backend.headpose.classify_gaze` treats yaw
~0 as "attending", which assumes the camera sits where the teacher and board
are. On a corner-mounted camera that is false, and it fails silently -- one real
clip labelled 320 of 383 faces (84%) as looking "right" while the students were
in fact facing a board off-frame.

``HeadPoseConfig.yaw_reference_deg`` fixes that, but only if someone knows what
to set it to. This reads a finished JSONL run, estimates the reference from the
yaw angles already in it (no re-inference), and shows what the gaze distribution
would become -- so the value is chosen from evidence and its effect is visible
before it is adopted.

**Check the "after" split against one rendered frame before trusting it.** The
estimate assumes most students face the front most of the time. That holds for
an ordinary lesson and breaks for group-work footage where the class is turned
toward each other -- in which case this will confidently return the wrong
reference.

Run:
    python -m tools.calibrate_gaze --jsonl outputs/part1_check.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import replace
from pathlib import Path


def calibrate(jsonl: Path) -> int:
    from backend.config import CONFIG
    from backend.headpose import classify_gaze, estimate_yaw_reference

    if not jsonl.is_file():
        raise FileNotFoundError(f"JSONL not found: {jsonl}")

    pairs: list[tuple[float, float]] = []
    frames = 0
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        frames += 1
        for person in json.loads(line).get("persons", []):
            hp = person.get("head_pose")
            if hp:
                pairs.append((float(hp["yaw"]), float(hp["pitch"])))

    if not pairs:
        print(f"No head_pose entries in {jsonl} -- nothing to calibrate from.")
        return 1

    yaws = [y for y, _ in pairs]
    reference = estimate_yaw_reference(yaws)
    print(f"{jsonl}: {frames} frames, {len(pairs)} faces with head pose")

    if reference is None:
        print("\nToo few samples to estimate a reference honestly. Run on more "
              "footage rather than accepting a noisy value.")
        return 1

    ordered = sorted(yaws)
    n = len(ordered)
    print(f"\nyaw distribution: min {ordered[0]:.1f}  p10 {ordered[n // 10]:.1f}  "
          f"median {reference:.1f}  p90 {ordered[9 * n // 10]:.1f}  "
          f"max {ordered[-1]:.1f}")

    before = Counter(classify_gaze(y, p, CONFIG.headpose) for y, p in pairs)
    after_cfg = replace(CONFIG.headpose, yaw_reference_deg=reference)
    after = Counter(classify_gaze(y, p, after_cfg) for y, p in pairs)

    print(f"\n{'label':<10}{'current':>10}{'calibrated':>13}")
    for label in ("teacher", "left", "right", "down", "back"):
        b = before[label] / n * 100
        a = after[label] / n * 100
        print(f"{label:<10}{b:>9.1f}%{a:>12.1f}%")

    dominant = max(before, key=before.get)
    if before[dominant] / n > 0.6:
        print(f"\nDiagnosis: '{dominant}' currently accounts for "
              f"{before[dominant] / n * 100:.0f}% of faces. A gaze signal that "
              f"returns one value that often is not discriminating -- consistent "
              f"with an off-centre camera rather than with the students.")

    print(f"\nTo adopt, set in backend/config.py HeadPoseConfig:\n"
          f"    yaw_reference_deg: float = {reference:.1f}")
    print("\nVerify the calibrated split against one rendered frame first: this "
          "assumes most students face the front most of the time, which is "
          "false for group-work footage.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", required=True)
    args = parser.parse_args()
    return calibrate(Path(args.jsonl))


if __name__ == "__main__":
    raise SystemExit(main())
