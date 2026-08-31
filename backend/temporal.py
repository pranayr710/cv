"""Stage 4: Temporal sequence analysis.

Consumes Stage 3 Scene Graph JSONL, applies rolling-window tracking to compute
temporal/rolling attributes, and outputs Stage 4 JSONL along with a summary report.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import deque
from itertools import combinations
from pathlib import Path

from backend.config import CONFIG, Config

logger = logging.getLogger(__name__)


class _StudentState:
    """Internal rolling state for a student node."""

    def __init__(self) -> None:
        self.history: deque[tuple[int, str | None, bool | None]] = deque()  # (timestamp_ms, engagement, eyes_closed)
        self.calibration_start_ms: int | None = None
        self.calibration_on_task: int = 0
        self.calibration_graded: int = 0
        self.calibration_baseline: float | None = None

        # Streak tracking for sustained distraction
        self.streak_start_ms: int | None = None
        self.was_sustained_distracted: bool = False
        self.distraction_event_count: int = 0

        # Overall summary stats
        self.first_seen_ms: int | None = None
        self.last_seen_ms: int | None = None
        self.total_on_task: int = 0
        self.total_graded: int = 0


class _PairState:
    """Internal rolling state for a student pair."""

    def __init__(self) -> None:
        self.history: deque[tuple[int, bool]] = deque()  # (timestamp_ms, oriented)

        # Streak tracking for sustained interaction
        self.streak_start_ms: int | None = None
        self.was_sustained_interacting: bool = False
        self.interaction_event_count: int = 0


class TemporalTracker:
    """Orchestrates rolling-window trackers for students and pairs."""

    def __init__(self, config: Config | None = None) -> None:
        self.cfg = config if config is not None else CONFIG
        self._students: dict[int, _StudentState] = {}
        self._pairs: dict[tuple[int, int], _PairState] = {}

    def update_frame(self, graph: dict) -> dict:
        """Process one frame's scene graph and add rolling/temporal features.

        Args:
            graph: Stage 3 scene graph dict.

        Returns:
            The mutated graph with temporal features populated.
        """
        timestamp_ms = graph["timestamp_ms"]
        nodes = graph.get("nodes", [])
        edges = graph.get("edges", [])

        # 1. Update Student States
        visible_node_ids = set()
        for node in nodes:
            node_id = node["id"]
            visible_node_ids.add(node_id)
            state = self._students.setdefault(node_id, _StudentState())

            if state.first_seen_ms is None:
                state.first_seen_ms = timestamp_ms
            state.last_seen_ms = timestamp_ms

            features = node["features"]
            engagement = features.get("engagement")
            eyes_closed = features.get("eyes_closed")

            # Update overall aggregates
            if engagement in ("on", "off"):
                state.total_graded += 1
                if engagement == "on":
                    state.total_on_task += 1

            # Append to history
            state.history.append((timestamp_ms, engagement, eyes_closed))

            # Prune old history
            window_start = timestamp_ms - int(self.cfg.temporal.window_seconds * 1000)
            while state.history and state.history[0][0] < window_start:
                state.history.popleft()

            # Personal Baseline Calibration
            if state.calibration_start_ms is None:
                state.calibration_start_ms = timestamp_ms

            if state.calibration_baseline is None:
                elapsed_calib = timestamp_ms - state.calibration_start_ms
                if engagement in ("on", "off"):
                    state.calibration_graded += 1
                    if engagement == "on":
                        state.calibration_on_task += 1

                if elapsed_calib >= self.cfg.attention.calibration_seconds * 1000:
                    if state.calibration_graded > 0:
                        state.calibration_baseline = state.calibration_on_task / state.calibration_graded
                    else:
                        state.calibration_baseline = 0.0

            # Compute rolling engagement percentage
            on_count = sum(1 for _, eng, _ in state.history if eng == "on")
            off_count = sum(1 for _, eng, _ in state.history if eng == "off")
            total_graded = on_count + off_count
            rolling_engagement_pct = on_count / total_graded if total_graded > 0 else None

            # Compute sustained distraction (majority off-task for sustained_attention_seconds)
            total_visible = len(state.history)
            majority_off = total_visible > 0 and (off_count / total_visible) >= self.cfg.attention.off_task_majority_fraction

            if majority_off:
                if state.streak_start_ms is None:
                    state.streak_start_ms = timestamp_ms
            else:
                state.streak_start_ms = None

            is_sustained_distracted = False
            if state.streak_start_ms is not None:
                elapsed_distracted = timestamp_ms - state.streak_start_ms
                if elapsed_distracted >= self.cfg.temporal.sustained_attention_seconds * 1000:
                    is_sustained_distracted = True

            # Track transitions to count distraction events
            if is_sustained_distracted and not state.was_sustained_distracted:
                state.distraction_event_count += 1
            state.was_sustained_distracted = is_sustained_distracted

            # Compute sustained eye closure (majority eyes closed in window)
            eyes_closed_count = sum(1 for _, _, ec in state.history if ec is True)
            eyes_graded = sum(1 for _, _, ec in state.history if ec is not None)
            is_eyes_closed_sustained = None
            if eyes_graded > 0:
                is_eyes_closed_sustained = (eyes_closed_count / eyes_graded) >= 0.5

            # Populate features
            features["rolling_engagement_pct"] = rolling_engagement_pct
            features["is_sustained_distracted"] = is_sustained_distracted
            features["is_eyes_closed_sustained"] = is_eyes_closed_sustained

        # 2. Update Pair States
        # Find which pairs are mutually oriented this frame
        oriented_pairs = set()
        for edge in edges:
            if edge["type"] == "mutual_orientation":
                id_a, id_b = edge["source"], edge["target"]
                pair_key = (min(id_a, id_b), max(id_a, id_b))
                oriented_pairs.add(pair_key)

        # Update rolling state for every visible/active pair in the frame
        visible_pairs = set()
        for node_a, node_b in combinations(nodes, 2):
            id_a, id_b = node_a["id"], node_b["id"]
            pair_key = (min(id_a, id_b), max(id_a, id_b))
            visible_pairs.add(pair_key)

            state_p = self._pairs.setdefault(pair_key, _PairState())
            oriented_now = pair_key in oriented_pairs
            state_p.history.append((timestamp_ms, oriented_now))

            # Prune old history
            window_start = timestamp_ms - int(self.cfg.temporal.window_seconds * 1000)
            while state_p.history and state_p.history[0][0] < window_start:
                state_p.history.popleft()

            # Compute rolling interaction fraction
            oriented_count = sum(1 for _, o in state_p.history if o)
            rolling_interaction_fraction = oriented_count / len(state_p.history) if state_p.history else 0.0

            # Compute sustained interaction (majority oriented for sustained_interaction_seconds)
            majority_oriented = len(state_p.history) > 0 and rolling_interaction_fraction >= self.cfg.peer_interaction.majority_fraction

            if majority_oriented:
                if state_p.streak_start_ms is None:
                    state_p.streak_start_ms = timestamp_ms
            else:
                state_p.streak_start_ms = None

            is_sustained_interacting = False
            if state_p.streak_start_ms is not None:
                elapsed_interaction = timestamp_ms - state_p.streak_start_ms
                if elapsed_interaction >= self.cfg.temporal.sustained_interaction_seconds * 1000:
                    is_sustained_interacting = True

            # Track transitions to count interaction events
            if is_sustained_interacting and not state_p.was_sustained_interacting:
                state_p.interaction_event_count += 1
            state_p.was_sustained_interacting = is_sustained_interacting

        # Populate Edge Features
        for edge in edges:
            id_a, id_b = edge["source"], edge["target"]
            pair_key = (min(id_a, id_b), max(id_a, id_b))
            state_p = self._pairs.get(pair_key)
            if state_p:
                oriented_count = sum(1 for _, o in state_p.history if o)
                rolling_interaction_fraction = oriented_count / len(state_p.history) if state_p.history else 0.0

                edge["features"]["rolling_interaction_fraction"] = rolling_interaction_fraction
                edge["features"]["is_sustained_interaction"] = state_p.was_sustained_interacting

        return graph

    def get_student_profiles(self) -> list[dict]:
        """Build individual summaries from accumulated student states."""
        profiles = []
        for node_id, state in self._students.items():
            duration_ms = (state.last_seen_ms - state.first_seen_ms) if (state.last_seen_ms is not None and state.first_seen_ms is not None) else 0
            final_engagement_pct = state.total_on_task / state.total_graded if state.total_graded > 0 else None
            deviation = None
            if final_engagement_pct is not None and state.calibration_baseline is not None:
                deviation = final_engagement_pct - state.calibration_baseline

            profiles.append({
                "node_id": node_id,
                "duration_ms": duration_ms,
                "calibration_baseline": state.calibration_baseline,
                "final_engagement_pct": final_engagement_pct,
                "deviation_from_baseline": deviation,
                "sustained_distractions_count": state.distraction_event_count,
            })
        return sorted(profiles, key=lambda p: p["node_id"])

    def get_classroom_summary(self) -> dict:
        """Build class-level summary statistics."""
        profiles = self.get_student_profiles()
        graded_profiles = [p for p in profiles if p["final_engagement_pct"] is not None]

        avg_engagement = (
            sum(p["final_engagement_pct"] for p in graded_profiles) / len(graded_profiles)
            if graded_profiles
            else None
        )
        total_sustained_distractions = sum(p["sustained_distractions_count"] for p in profiles)

        # Count pairs with at least one sustained interaction
        interacting_pairs = sum(1 for state in self._pairs.values() if state.interaction_event_count > 0)

        return {
            "students_considered": len(profiles),
            "average_engagement": avg_engagement,
            "total_sustained_distractions": total_sustained_distractions,
            "interacting_pairs_count": interacting_pairs,
        }


def process_jsonl(input_path: Path, output_path: Path, config: Config | None = None) -> TemporalTracker:
    """Read Stage 3 JSONL, run temporal tracking, and write Stage 4 JSONL."""
    cfg = config if config is not None else CONFIG
    tracker = TemporalTracker(cfg)
    with input_path.open("r", encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            graph = json.loads(line)
            augmented = tracker.update_frame(graph)
            fout.write(json.dumps(augmented) + "\n")
    return tracker


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m backend.temporal",
        description="Convert Stage 3 Scene Graph JSONL into Stage 4 Temporal JSONL and report summary."
    )
    parser.add_argument("--jsonl", required=True, type=str, help="Path to input Stage 3 JSONL file.")
    parser.add_argument("--out", required=True, type=str, help="Path to output Stage 4 JSONL file.")
    parser.add_argument("--report", required=True, type=str, help="Path to output JSON report file.")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s")

    input_path = Path(args.jsonl)
    output_path = Path(args.out)
    report_path = Path(args.report)

    if not input_path.is_file():
        logger.error("Input file not found: %s", input_path)
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        tracker = process_jsonl(input_path, output_path)
        report = {
            "classroom_summary": tracker.get_classroom_summary(),
            "student_profiles": tracker.get_student_profiles(),
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Successfully completed temporal analysis.")
        logger.info("Wrote Stage 4 JSONL to %s", output_path)
        logger.info("Wrote summary report to %s", report_path)
        return 0
    except Exception:
        logger.exception("Failed to run temporal analysis")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
