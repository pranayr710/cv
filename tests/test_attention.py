"""Unit tests for :mod:`backend.attention`.

Pure logic on schema-shaped dicts -- no ML dependency, runs everywhere. This
mirrors how the module itself works: it consumes finished JSONL output, not
live frames, so tests build that JSONL shape directly rather than mocking a
model.

Required coverage:
    1. classify_frame() covers every orientation category and the
       independent eyes_closed flag.
    2. iter_jsonl_signals() reads real JSONL and skips unconfirmed tracks.
    3. RollingAttentionTracker: window pruning, sustained-distraction timing
       (including that a broken streak resets it), calibration baseline,
       and the classroom-level summary never singling out one student.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.attention import (
    ALL_ORIENTATIONS,
    FrameSignal,
    RollingAttentionTracker,
    classify_frame,
    iter_jsonl_signals,
)
from backend.config import CONFIG, AttentionConfig

_CFG = CONFIG.attention


def _person(
    track_id: int | None = 1,
    bbox=(0, 0, 100, 200),
    gaze_label: str | None = "teacher",
    ear: float | None = 0.30,
    has_posture: bool = False,
) -> dict:
    """Build a schema-shaped person dict for one frame."""
    face = (
        None
        if gaze_label is None and ear is None
        else {"bbox": None, "landmarks": None, "ear": ear}
    )
    head_pose = (
        None
        if gaze_label is None
        else {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "gaze_label": gaze_label}
    )
    posture = (
        {
            "nose": [1.0, 1.0],
            "shoulder_mid": [1.0, 2.0],
            "hip_mid": None,
            "vertical_lean": 0.1,
        }
        if has_posture
        else None
    )
    return {
        "track_id": track_id,
        "bbox": list(bbox),
        "confidence": 0.9,
        "face": face,
        "head_pose": head_pose,
        "posture": posture,
    }


def _phone(bbox=(10, 10, 20, 20)) -> dict:
    return {"cls": "cell phone", "bbox": list(bbox), "confidence": 0.8}


def _book(bbox=(10, 10, 20, 20)) -> dict:
    return {"cls": "book", "bbox": list(bbox), "confidence": 0.5}


def _laptop(bbox=(10, 10, 20, 20)) -> dict:
    return {"cls": "laptop", "bbox": list(bbox), "confidence": 0.6}


# --------------------------------------------------------------------------- #
# The writing / on-task signal (head down over a book)
#
# Added after a reviewer asked why a student writing in a book with a pen was
# not identified as working. Before this, they landed in the same ambiguous
# bucket as a disengaged student.
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("down_label", ["down", "back"])
def test_head_down_over_a_book_is_writing_not_ambiguous(down_label: str) -> None:
    """The whole point of the category: a bowed head over a book is on-task."""
    p = _person(gaze_label=down_label, bbox=(0, 0, 100, 200))
    assert classify_frame(p, [_book()]).orientation == "head_down_writing"


def test_writing_is_never_classified_as_an_off_task_device_case() -> None:
    p = _person(gaze_label="down", bbox=(0, 0, 100, 200))
    result = classify_frame(p, [_book()])
    assert result.orientation not in (
        "head_down_with_device",
        "head_down_no_device",
    )


def test_phone_beats_book_when_both_are_near_the_same_student() -> None:
    """Contradictory evidence resolves to the more concerning reading.

    Crediting a student as "working" because a book happens to be open on the
    desk while they are on a phone is the easier error to make and the worse
    one, so the phone must win regardless of argument order.
    """
    p = _person(gaze_label="down", bbox=(0, 0, 100, 200))
    assert (
        classify_frame(p, [_book(), _phone()]).orientation
        == "head_down_with_device"
    )
    assert (
        classify_frame(p, [_phone(), _book()]).orientation
        == "head_down_with_device"
    )


def test_book_not_overlapping_the_student_does_not_count_as_writing() -> None:
    """Someone else's book across the room must not mark this student on-task."""
    p = _person(gaze_label="down", bbox=(0, 0, 100, 200))
    far_book = _book(bbox=(500, 500, 20, 20))
    assert classify_frame(p, [far_book]).orientation == "head_down_no_device"


def test_laptop_is_deliberately_not_a_writing_signal() -> None:
    """A laptop is genuinely ambiguous (computer lab vs ordinary classroom).

    Config comments in DetectionConfig record that img04 is a real computer lab
    and img01 is not, so a laptop cannot be read as either on- or off-task. It
    must leave the student in the ambiguous bucket rather than be guessed at.
    """
    p = _person(gaze_label="down", bbox=(0, 0, 100, 200))
    assert classify_frame(p, [_laptop()]).orientation == "head_down_no_device"


def test_book_does_not_override_attending_teacher() -> None:
    """A book on the desk is irrelevant while the student is looking up."""
    p = _person(gaze_label="teacher")
    assert classify_frame(p, [_book()]).orientation == "attending_teacher"


def test_writing_category_is_registered_in_all_orientations() -> None:
    """A category missing from ALL_ORIENTATIONS silently vanishes from reports.

    The window distribution and classroom summary both iterate
    ALL_ORIENTATIONS, so adding a Literal member without adding it here would
    classify students into a bucket that is never displayed.
    """
    from backend.attention import ALL_ORIENTATIONS

    assert "head_down_writing" in ALL_ORIENTATIONS


# --------------------------------------------------------------------------- #
# classify_frame
# --------------------------------------------------------------------------- #


def test_classify_frame_teacher_is_attending() -> None:
    p = _person(gaze_label="teacher")
    assert classify_frame(p, []).orientation == "attending_teacher"


@pytest.mark.parametrize("side", ["left", "right"])
def test_classify_frame_left_right_is_ambiguous_not_off_task(side: str) -> None:
    """gaze left/right must never be classified as an off-task category."""
    p = _person(gaze_label=side)
    result = classify_frame(p, [])
    assert result.orientation == "oriented_away"
    assert result.orientation not in ("head_down_with_device", "head_down_no_device")


@pytest.mark.parametrize("down_label", ["down", "back"])
def test_classify_frame_down_with_overlapping_phone_is_device(down_label: str) -> None:
    p = _person(gaze_label=down_label, bbox=(0, 0, 100, 200))
    objects = [_phone(bbox=(10, 10, 20, 20))]  # overlaps the person bbox
    assert classify_frame(p, objects).orientation == "head_down_with_device"


@pytest.mark.parametrize("down_label", ["down", "back"])
def test_classify_frame_down_without_phone_is_ambiguous(down_label: str) -> None:
    """A bowed head with no phone nearby is NOT assumed off-task (gaze
    aversion during concentration is a documented confound)."""
    p = _person(gaze_label=down_label)
    assert classify_frame(p, []).orientation == "head_down_no_device"


def test_classify_frame_down_with_nonoverlapping_phone_is_not_device() -> None:
    p = _person(gaze_label="down", bbox=(0, 0, 10, 10))
    objects = [_phone(bbox=(500, 500, 20, 20))]  # far away, no overlap
    assert classify_frame(p, objects).orientation == "head_down_no_device"


def test_classify_frame_no_face_but_posture_is_posture_only() -> None:
    p = _person(gaze_label=None, ear=None, has_posture=True)
    assert classify_frame(p, []).orientation == "posture_only"


def test_classify_frame_no_signal_at_all() -> None:
    p = _person(gaze_label=None, ear=None, has_posture=False)
    assert classify_frame(p, []).orientation == "no_signal"


def test_classify_frame_eyes_closed_independent_of_orientation() -> None:
    """eyes_closed is tracked alongside orientation, not instead of it."""
    p = _person(gaze_label="teacher", ear=0.05)  # below default 0.20 threshold
    result = classify_frame(p, [])
    assert result.orientation == "attending_teacher"
    assert result.eyes_closed is True


def test_classify_frame_eyes_closed_none_without_ear_data() -> None:
    p = _person(gaze_label="teacher", ear=None)
    assert classify_frame(p, []).eyes_closed is None


# --------------------------------------------------------------------------- #
# iter_jsonl_signals
# --------------------------------------------------------------------------- #


def test_iter_jsonl_signals_skips_unconfirmed_tracks(tmp_path: Path) -> None:
    records = [
        {
            "frame_id": 0,
            "timestamp_ms": 0,
            "persons": [_person(track_id=None), _person(track_id=5)],
            "objects": [],
        }
    ]
    path = tmp_path / "stage1.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")

    results = list(iter_jsonl_signals(path))
    assert len(results) == 1
    assert results[0][0] == 5


def test_iter_jsonl_signals_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        list(iter_jsonl_signals("does/not/exist.jsonl"))


# --------------------------------------------------------------------------- #
# RollingAttentionTracker
# --------------------------------------------------------------------------- #


def _fast_cfg(**overrides) -> AttentionConfig:
    """A config with short windows so tests run in milliseconds of sim time."""
    base = {
        "window_seconds": 10.0,
        "sustained_seconds": 20.0,
        "off_task_majority_fraction": 0.5,
        "calibration_seconds": 10.0,
    }
    base.update(overrides)
    return AttentionConfig(**base)


def test_window_prunes_old_frames() -> None:
    cfg = _fast_cfg(window_seconds=5.0)
    tracker = RollingAttentionTracker(cfg)
    tracker.update(1, 0, FrameSignal("attending_teacher", None))
    tracker.update(1, 4000, FrameSignal("attending_teacher", None))
    # This frame is 6s after t=0, which falls outside a 5s window -> pruned.
    tracker.update(1, 6000, FrameSignal("head_down_no_device", None))

    dist = tracker.window_distribution(1)
    assert sum(dist.values()) == pytest.approx(1.0)
    assert dist["attending_teacher"] == pytest.approx(
        0.5
    )  # only the t=4000 frame remains
    assert dist["head_down_no_device"] == pytest.approx(0.5)


def test_unknown_track_returns_zero_distribution() -> None:
    tracker = RollingAttentionTracker(_fast_cfg())
    dist = tracker.window_distribution(999)
    assert all(v == 0.0 for v in dist.values())
    assert set(dist) == set(ALL_ORIENTATIONS)


def test_sustained_device_distraction_requires_full_duration() -> None:
    cfg = _fast_cfg(sustained_seconds=20.0, window_seconds=5.0)
    tracker = RollingAttentionTracker(cfg)
    for t_s in range(19):  # 0..18 inclusive, 19s of continuous distraction
        tracker.update(1, t_s * 1000, FrameSignal("head_down_with_device", None))
    assert tracker.is_sustained_device_distraction(1) is False

    tracker.update(1, 20_000, FrameSignal("head_down_with_device", None))
    assert tracker.is_sustained_device_distraction(1) is True


def test_sustained_streak_resets_on_a_single_attending_frame() -> None:
    """A brief return to attending should reset the streak, not just pause it."""
    # window_seconds smaller than the 1s update spacing means each update's
    # window contains only itself -- the cleanest way to force the single
    # attending frame to flip that instant's majority on its own, rather than
    # being outvoted by neighbouring frames still inside a wider window.
    cfg = _fast_cfg(sustained_seconds=10.0, window_seconds=0.5)
    tracker = RollingAttentionTracker(cfg)
    for t_s in range(9):
        tracker.update(1, t_s * 1000, FrameSignal("head_down_with_device", None))
    tracker.update(1, 9000, FrameSignal("attending_teacher", None))
    for t_s in range(10, 20):
        tracker.update(1, t_s * 1000, FrameSignal("head_down_with_device", None))
    # Only ~10s of re-accumulated streak since the reset at t=9000 -> not yet sustained.
    assert tracker.is_sustained_device_distraction(1) is False


def test_calibration_baseline_none_until_period_elapses() -> None:
    cfg = _fast_cfg(calibration_seconds=10.0)
    tracker = RollingAttentionTracker(cfg)
    tracker.update(1, 0, FrameSignal("attending_teacher", None))
    tracker.update(1, 5000, FrameSignal("attending_teacher", None))
    assert tracker.personal_baseline(1) is None  # only 5s elapsed

    tracker.update(1, 11_000, FrameSignal("head_down_no_device", None))
    baseline = tracker.personal_baseline(1)
    assert baseline is not None
    assert 0.0 <= baseline <= 1.0


def test_deviation_from_baseline_none_before_calibrated() -> None:
    tracker = RollingAttentionTracker(_fast_cfg())
    tracker.update(1, 0, FrameSignal("attending_teacher", None))
    assert tracker.deviation_from_baseline(1) is None


def test_summarise_classroom_aggregates_without_naming_a_student() -> None:
    tracker = RollingAttentionTracker(_fast_cfg(window_seconds=100.0))
    tracker.update(1, 0, FrameSignal("attending_teacher", None))
    tracker.update(2, 0, FrameSignal("head_down_with_device", None))

    summary = tracker.summarise_classroom()
    assert summary["student_count"] == 2
    assert set(summary["distribution"]) == set(ALL_ORIENTATIONS)
    # Averaged 50/50 across the two students on these two categories.
    assert summary["distribution"]["attending_teacher"] == pytest.approx(0.5)
    assert summary["distribution"]["head_down_with_device"] == pytest.approx(0.5)
    # The summary is a count, not a list of which student -- no individual
    # identity should be recoverable from this return value alone.
    assert "track_id" not in json.dumps(summary)


def test_summarise_classroom_empty_tracker() -> None:
    tracker = RollingAttentionTracker(_fast_cfg())
    summary = tracker.summarise_classroom()
    assert summary["student_count"] == 0
    assert summary["sustained_device_distraction_count"] == 0


def test_eyes_closed_ratio_ignores_frames_without_ear_data() -> None:
    tracker = RollingAttentionTracker(_fast_cfg(window_seconds=100.0))
    tracker.update(1, 0, FrameSignal("attending_teacher", True))
    tracker.update(1, 1000, FrameSignal("posture_only", None))  # no EAR available
    tracker.update(1, 2000, FrameSignal("attending_teacher", False))

    ratio = tracker.window_eyes_closed_ratio(1)
    assert ratio == pytest.approx(0.5)  # 1 of the 2 EAR-bearing frames


def test_eyes_closed_ratio_none_with_no_ear_data_at_all() -> None:
    tracker = RollingAttentionTracker(_fast_cfg())
    tracker.update(1, 0, FrameSignal("posture_only", None))
    assert tracker.window_eyes_closed_ratio(1) is None
