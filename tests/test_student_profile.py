"""Unit tests for :mod:`backend.student_profile`.

Builds tiny synthetic JSONL files by hand (schema-shaped dicts, not full
pipeline output) so these run without any ML model.

Test coverage:
    1. One student across multiple frames gets one profile with correct
       frames_seen, first/last timestamp, and duration.
    2. Two students in the same frames get two separate profiles.
    3. A negative person_id (never face-verified) is marked accordingly.
    4. Expression/behaviour tallies match backend.expression/behaviour's shape.
    5. A weak-reliability behaviour label is surfaced in weak_labels.
    6. Concentration is computed via backend.engagement, not reinvented.
    7. A person with track_id present but person_id null is skipped (no
       stable key to file them under).
    8. Missing file raises FileNotFoundError.
"""

from __future__ import annotations

import json

import pytest

from backend.student_profile import build_profiles


def _person(
    person_id,
    track_id=1,
    expression=None,
    behaviour=None,
    gaze_label=None,
):
    return {
        "track_id": track_id,
        "person_id": person_id,
        "bbox": [0, 0, 10, 10],
        "confidence": 0.9,
        "source": "yolo",
        "face": None,
        "head_pose": (
            {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "gaze_label": gaze_label}
            if gaze_label is not None
            else None
        ),
        "posture": None,
        "expression": (
            {"label": expression, "confidence": 0.8, "distribution": None}
            if expression is not None
            else None
        ),
        "behaviour": (
            {"label": behaviour[0], "confidence": 0.7, "reliability": behaviour[1]}
            if behaviour is not None
            else None
        ),
    }


def _frame(frame_id, ts_ms, persons):
    return {"frame_id": frame_id, "timestamp_ms": ts_ms, "persons": persons, "objects": []}


def _write_jsonl(path, records):
    path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )


def test_one_student_across_frames_gets_one_profile(tmp_path) -> None:
    p = tmp_path / "run.jsonl"
    _write_jsonl(p, [
        _frame(0, 0, [_person(1, gaze_label="teacher")]),
        _frame(1, 1000, [_person(1, gaze_label="teacher")]),
        _frame(2, 2000, [_person(1, gaze_label="left")]),
    ])
    profiles = build_profiles(p)
    assert set(profiles) == {1}
    prof = profiles[1]
    assert prof["frames_seen"] == 3
    assert prof["first_seen_ms"] == 0
    assert prof["last_seen_ms"] == 2000
    assert prof["duration_ms"] == 2000


def test_two_students_get_two_separate_profiles(tmp_path) -> None:
    p = tmp_path / "run.jsonl"
    _write_jsonl(p, [
        _frame(0, 0, [_person(1, track_id=1), _person(2, track_id=2)]),
    ])
    profiles = build_profiles(p)
    assert set(profiles) == {1, 2}


def test_negative_person_id_is_marked_not_face_verified(tmp_path) -> None:
    p = tmp_path / "run.jsonl"
    _write_jsonl(p, [_frame(0, 0, [_person(-1, track_id=5)])])
    profiles = build_profiles(p)
    assert profiles[-1]["face_verified"] is False


def test_positive_person_id_is_marked_face_verified(tmp_path) -> None:
    p = tmp_path / "run.jsonl"
    _write_jsonl(p, [_frame(0, 0, [_person(1)])])
    profiles = build_profiles(p)
    assert profiles[1]["face_verified"] is True


def test_expression_tally_matches_expected_shape(tmp_path) -> None:
    p = tmp_path / "run.jsonl"
    _write_jsonl(p, [
        _frame(0, 0, [_person(1, expression="happy")]),
        _frame(1, 1000, [_person(1, expression="happy")]),
        _frame(2, 2000, [_person(1, expression=None)]),
    ])
    profiles = build_profiles(p)
    expr = profiles[1]["expression"]
    assert expr["counts"] == {"happy": 2}
    assert expr["classified"] == 2
    assert expr["unavailable"] == 1


def test_weak_behaviour_label_is_surfaced(tmp_path) -> None:
    p = tmp_path / "run.jsonl"
    _write_jsonl(p, [
        _frame(0, 0, [_person(1, behaviour=("using_device", "weak"))]),
        _frame(1, 1000, [_person(1, behaviour=("write", "measured"))]),
    ])
    profiles = build_profiles(p)
    beh = profiles[1]["behaviour"]
    assert beh["counts"] == {"using_device": 1, "write": 1}
    assert beh["weak_labels"] == ["using_device"]


def test_concentration_uses_engagement_precedence(tmp_path) -> None:
    """Off-task behaviour must win over an attending gaze in the aggregate
    too -- proves student_profile delegates to backend.engagement rather than
    reimplementing its own (possibly inconsistent) rule."""
    p = tmp_path / "run.jsonl"
    _write_jsonl(p, [
        _frame(0, 0, [_person(1, gaze_label="teacher", behaviour=("using_device", "weak"))]),
    ])
    profiles = build_profiles(p)
    conc = profiles[1]["concentration"]
    assert conc["off"] == 1
    assert conc["on"] == 0
    assert conc["concentration_pct"] == pytest.approx(0.0)


def test_100pct_with_no_behaviour_readings_is_flagged_as_undetectable(tmp_path) -> None:
    """The real bug this guards against: 'off' is only reachable through a
    behaviour reading, so a student who never gets one reads 100%
    concentration purely from absence of evidence -- caught on real
    out-of-distribution footage where the behaviour model found zero
    detections for an entire video. Must be surfaced, not silently trusted."""
    p = tmp_path / "run.jsonl"
    _write_jsonl(p, [
        _frame(0, 0, [_person(1, gaze_label="teacher")]),
        _frame(1, 1000, [_person(1, gaze_label="teacher")]),
    ])
    profiles = build_profiles(p)
    conc = profiles[1]["concentration"]
    assert conc["concentration_pct"] == pytest.approx(100.0)
    assert conc["off_task_detectable"] is False
    assert "caveat" in conc


def test_100pct_with_behaviour_readings_present_is_not_flagged(tmp_path) -> None:
    """A student who genuinely never behaved off-task, WITH behaviour readings
    actually available, is a real finding and must not be second-guessed."""
    p = tmp_path / "run.jsonl"
    _write_jsonl(p, [
        _frame(0, 0, [_person(1, gaze_label="teacher", behaviour=("write", "measured"))]),
    ])
    profiles = build_profiles(p)
    conc = profiles[1]["concentration"]
    assert conc["concentration_pct"] == pytest.approx(100.0)
    assert conc["off_task_detectable"] is True
    assert "caveat" not in conc


def test_person_with_no_person_id_is_skipped(tmp_path) -> None:
    """track_id present but person_id null: no stable key to file them under."""
    p = tmp_path / "run.jsonl"
    record = _person(None, track_id=7)
    _write_jsonl(p, [_frame(0, 0, [record])])
    profiles = build_profiles(p)
    assert profiles == {}


def test_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        build_profiles("does/not/exist.jsonl")
