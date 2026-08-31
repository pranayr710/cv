"""Show how the reported scores move with the temporal thresholds.

Two constants decide when looking away stops being a glance and starts being
disengagement, and both were unvalidated numbers for most of this project's
life. ``sustained_attention_seconds`` was 90, which is far outside anything the
classroom literature supports and meant the sustained-distraction flag almost
never fired.

A single defended value is still a choice, so this measures the alternative:
how much does each reported figure actually move across the plausible range? A
score that barely moves between 2 and 20 seconds is robust and the exact value
does not matter much. One that swings wildly is being decided by the constant
rather than by the student, and should be reported with that caveat.

Run:
    python -m tools.sweep_temporal --graph outputs/lecture/live_graph.jsonl
"""

from __future__ import annotations

import argparse
import logging
import statistics
from dataclasses import replace
from pathlib import Path

#: Candidate glance filters, in seconds. 0 is the old behaviour (every off-task
#: frame counts); 5 is the behavioural-observation benchmark; 20 is the
#: classroom observation interval.
GLANCE_SECONDS = (0.0, 2.0, 5.0, 10.0, 20.0)


def main() -> int:
    """Sweep the temporal thresholds and print how the numbers respond.

    Returns:
        A process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", default="outputs/lecture/live_graph.jsonl",
                        help="A finished session's scene-graph JSONL.")
    parser.add_argument("--seconds", default=",".join(str(s) for s in GLANCE_SECONDS))
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    from backend.config import CONFIG
    from backend.student_profile import build_profiles

    graph = Path(args.graph)
    if not graph.exists():
        raise SystemExit(f"No session graph at {graph}")

    print(f"{graph}\n")
    print(f"{'glance filter':>14}{'students':>10}{'mean on-task':>14}"
          f"{'median':>9}{'spread':>9}")

    baseline = None
    for seconds in (float(s) for s in args.seconds.split(",")):
        config = replace(
            CONFIG, temporal=replace(CONFIG.temporal, min_off_task_seconds=seconds))
        profiles = build_profiles(graph, config)
        scores = [p["on_task_pct"] for p in profiles.values()
                  if p.get("is_student") and p.get("on_task_pct") is not None]
        if not scores:
            print(f"{seconds:>13.0f}s{0:>10}{'-':>14}")
            continue
        mean = statistics.mean(scores)
        if baseline is None:
            baseline = mean
        print(f"{seconds:>13.0f}s{len(scores):>10}{mean:>13.1f}%"
              f"{statistics.median(scores):>8.1f}%"
              f"{max(scores) - min(scores):>8.1f}")

    print("\nA figure that moves little across this range is robust to the "
          "threshold.\nOne that swings is being decided by the constant "
          "rather than by the student.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
