"""Unit tests for Stage 4: Temporal sequence analysis."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema.validators import Draft202012Validator

from backend.temporal import TemporalTracker, process_jsonl


def _node(node_id: int, engagement: str | None, eyes_closed: bool | None = None) -> dict:
    """Helper to build a node matching graph_schema.json."""
    return {
        "id": node_id,
        "person_id": node_id,
        "role": "student",
        "features": {
            "bbox": [10, 10, 50, 100],
            "gaze_label": "teacher",
            "posture": None,
            "expression": None,
            "behaviour": None,
            "engagement": engagement,
            "eyes_closed": eyes_closed,
            "rolling_engagement_pct": None,
            "is_sustained_distracted": None,
            "is_eyes_closed_sustained": None
        }
    }


def _edge(source: int, target: int, edge_type: str) -> dict:
    """Helper to build an edge matching graph_schema.json."""
    return {
        "type": edge_type,
        "source": source,
        "target": target,
        "features": {
            "distance_px": 100.0,
            "oriented_fraction": None,
            "shared_object_class": None,
            "is_sustained_interaction": None,
            "rolling_interaction_fraction": None
        }
    }


@pytest.fixture
def graph_validator() -> Draft202012Validator:
    """Fixture returning a JSON Schema validator for graph_schema.json."""
    schema_path = Path(__file__).resolve().parent.parent / "graph_schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


def test_sustained_distraction_and_calibration(graph_validator) -> None:
    """Sustained distraction fires only after the configured duration.

    Timings are taken from the config rather than written in, because they are
    a measured choice that has already moved once: the flag was set at 90
    seconds, far outside anything the classroom literature supports, and
    lowering it to the 20-second observation interval silently broke a test
    that had the old number baked into it.
    """
    from backend.config import CONFIG

    tracker = TemporalTracker()
    sustained_s = int(CONFIG.temporal.sustained_attention_seconds)
    calibration_s = 60

    # Off-task from the start, stopping just short of the sustained threshold.
    for t_s in range(min(sustained_s, calibration_s)):
        graph = {
            "frame_id": t_s,
            "timestamp_ms": t_s * 1000,
            "nodes": [_node(1, engagement="off")],
            "edges": []
        }
        res = tracker.update_frame(graph)
        graph_validator.validate(res)

        features = res["nodes"][0]["features"]
        assert features["rolling_engagement_pct"] == 0.0
        # Below the threshold this is an episode, not sustained distraction.
        assert features["is_sustained_distracted"] is False

    # Carry on to the calibration point and past the sustained threshold.
    for t_s in range(min(sustained_s, calibration_s), max(sustained_s, calibration_s) + 1):
        graph = {
            "frame_id": t_s,
            "timestamp_ms": t_s * 1000,
            "nodes": [_node(1, engagement="off")],
            "edges": []
        }
        res = tracker.update_frame(graph)
        features = res["nodes"][0]["features"]

    # Past the threshold the distraction is sustained, and the baseline has
    # calibrated (everything so far was off task, so the baseline is 0.0).
    assert features["is_sustained_distracted"] is True
    profile = tracker.get_student_profiles()[0]
    assert profile["sustained_distractions_count"] == 1
    assert profile["calibration_baseline"] == 0.0

    # Break the streak with on-task frames for longer than the rolling window.
    start = max(sustained_s, calibration_s) + 1
    for t_s in range(start, start + int(CONFIG.temporal.window_seconds) + 3):
        graph = {
            "frame_id": t_s,
            "timestamp_ms": t_s * 1000,
            "nodes": [_node(1, engagement="on")],
            "edges": []
        }
        res = tracker.update_frame(graph)
        features = res["nodes"][0]["features"]

    # Now the rolling window is majority on-task, so the sustained distraction flag must be False
    assert features["is_sustained_distracted"] is False


def test_sustained_interaction_threshold(graph_validator) -> None:
    tracker = TemporalTracker()

    # Feed 19 seconds of mutual orientation
    for t_s in range(20):  # t=0 to t=19
        graph = {
            "frame_id": t_s,
            "timestamp_ms": t_s * 1000,
            "nodes": [_node(1, None), _node(2, None)],
            "edges": [_edge(1, 2, "mutual_orientation")]
        }
        res = tracker.update_frame(graph)
        graph_validator.validate(res)

        edge = res["edges"][0]
        assert edge["features"]["rolling_interaction_fraction"] == 1.0
        assert edge["features"]["is_sustained_interaction"] is False

    # Feed at t=20s
    graph = {
        "frame_id": 20,
        "timestamp_ms": 20 * 1000,
        "nodes": [_node(1, None), _node(2, None)],
        "edges": [_edge(1, 2, "mutual_orientation")]
    }
    res = tracker.update_frame(graph)
    edge = res["edges"][0]
    assert edge["features"]["is_sustained_interaction"] is True

    # Check report counts
    summary = tracker.get_classroom_summary()
    assert summary["interacting_pairs_count"] == 1


def test_eyes_closed_sustained(graph_validator) -> None:
    tracker = TemporalTracker()
    # Feed eyes closed
    graph = {
        "frame_id": 0,
        "timestamp_ms": 0,
        "nodes": [_node(1, None, eyes_closed=True)],
        "edges": []
    }
    res = tracker.update_frame(graph)
    assert res["nodes"][0]["features"]["is_eyes_closed_sustained"] is True


def test_process_jsonl_temporal(tmp_path: Path, graph_validator) -> None:
    record = {
        "frame_id": 0,
        "timestamp_ms": 0,
        "nodes": [_node(1, engagement="on")],
        "edges": []
    }
    infile = tmp_path / "stage3.jsonl"
    outfile = tmp_path / "stage4.jsonl"
    infile.write_text(json.dumps(record) + "\n", encoding="utf-8")

    process_jsonl(infile, outfile)
    assert outfile.is_file()

    content = outfile.read_text(encoding="utf-8")
    graph = json.loads(content)
    graph_validator.validate(graph)

    # Nodes should have rolling features populated (not None)
    features = graph["nodes"][0]["features"]
    assert features["rolling_engagement_pct"] == 1.0
    assert features["is_sustained_distracted"] is False
