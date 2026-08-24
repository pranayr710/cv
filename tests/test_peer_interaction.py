"""Unit tests for :mod:`backend.peer_interaction`.

Pure logic on schema-shaped dicts -- no ML dependency, runs everywhere. Same
architecture as :mod:`tests.test_attention`: this module consumes finished
JSONL output, so tests build that shape directly.

Required coverage:
    1. Geometric primitives (bbox proximity, undirected line angle,
       perpendicularity) tested in isolation with known synthetic geometry.
    2. classify_pair_frame() covers: too far apart, missing posture/shoulder
       data, oriented-toward-each-other (perpendicular shoulders), and
       not-oriented (parallel shoulders).
    3. iter_jsonl_pair_signals() reads real JSONL, skips untracked persons,
       and enumerates every pair.
    4. RollingPeerInteractionTracker: window pruning, sustained timing
       (including that a broken streak resets it), and the classroom summary
       returning a count, never naming which pair, by default.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.config import CONFIG, PeerInteractionConfig
from backend.peer_interaction import (
    RollingPeerInteractionTracker,
    _angular_distance_mod_180,
    _bbox_center,
    _deviation_from_perpendicular,
    _line_angle_degrees,
    _within_conversational_distance,
    classify_pair_frame,
    iter_jsonl_pair_signals,
)

_CFG = CONFIG.peer_interaction


def _person(
    track_id: int | None,
    bbox=(0, 0, 50, 100),
    left_shoulder=None,
    right_shoulder=None,
) -> dict:
    """Build a schema-shaped person dict carrying only what this module reads."""
    posture = None
    if left_shoulder is not None or right_shoulder is not None:
        posture = {
            "nose": None,
            "left_shoulder": list(left_shoulder) if left_shoulder else None,
            "right_shoulder": list(right_shoulder) if right_shoulder else None,
            "shoulder_mid": None,
            "hip_mid": None,
            "vertical_lean": None,
            "facing_direction": None,
        }
    return {
        "track_id": track_id,
        "bbox": list(bbox),
        "confidence": 0.9,
        "face": None,
        "head_pose": None,
        "posture": posture,
    }


# --------------------------------------------------------------------------- #
# Geometric primitives
# --------------------------------------------------------------------------- #


def test_bbox_center() -> None:
    assert _bbox_center((10, 20, 30, 40)) == (25.0, 40.0)


def test_within_conversational_distance_true_when_close() -> None:
    a = (0, 0, 50, 100)
    b = (100, 0, 50, 100)  # centres 100 apart, gap=50, threshold=min(50,50)*1.0=50
    assert _within_conversational_distance(a, b, _CFG) is True


def test_within_conversational_distance_false_when_far() -> None:
    a = (0, 0, 50, 100)
    b = (300, 0, 50, 100)  # gap=250, threshold=50
    assert _within_conversational_distance(a, b, _CFG) is False


def test_within_conversational_distance_boundary() -> None:
    a = (0, 0, 50, 100)
    b = (110, 0, 50, 100)  # gap=60, threshold=50 under default ratio 1.0
    assert _within_conversational_distance(a, b, _CFG) is False

    custom_cfg = PeerInteractionConfig(max_gap_to_width_ratio=1.5)
    assert _within_conversational_distance(a, b, custom_cfg) is True


def test_within_conversational_distance_true_when_overlapping() -> None:
    a = (0, 0, 50, 100)
    b = (10, 0, 50, 100)
    assert _within_conversational_distance(a, b, _CFG) is True


def test_line_angle_degrees_horizontal_is_zero() -> None:
    assert _line_angle_degrees((0, 0), (10, 0)) == pytest.approx(0.0)


def test_line_angle_degrees_vertical_is_90() -> None:
    assert _line_angle_degrees((0, 0), (0, 10)) == pytest.approx(90.0)


def test_line_angle_degrees_none_for_coincident_points() -> None:
    assert _line_angle_degrees((5, 5), (5, 5)) is None


def test_angular_distance_mod_180_wraps_correctly() -> None:
    assert _angular_distance_mod_180(10, 170) == pytest.approx(20.0)
    assert _angular_distance_mod_180(0, 90) == pytest.approx(90.0)
    assert _angular_distance_mod_180(45, 45) == pytest.approx(0.0)


def test_deviation_from_perpendicular_zero_when_perpendicular() -> None:
    assert _deviation_from_perpendicular(
        shoulder_angle=90.0, bearing_angle=0.0
    ) == pytest.approx(0.0)


def test_deviation_from_perpendicular_90_when_parallel() -> None:
    assert _deviation_from_perpendicular(
        shoulder_angle=0.0, bearing_angle=0.0
    ) == pytest.approx(90.0)


# --------------------------------------------------------------------------- #
# classify_pair_frame
# --------------------------------------------------------------------------- #


def test_classify_pair_false_when_too_far_apart() -> None:
    a = _person(1, bbox=(0, 0, 50, 100), left_shoulder=(0, 40), right_shoulder=(0, 60))
    b = _person(
        2, bbox=(1000, 0, 50, 100), left_shoulder=(1000, 40), right_shoulder=(1000, 60)
    )
    assert classify_pair_frame(a, b) is False


def test_classify_pair_false_when_posture_missing() -> None:
    a = _person(1, bbox=(0, 0, 50, 100))  # no posture at all
    b = _person(
        2, bbox=(60, 0, 50, 100), left_shoulder=(60, 40), right_shoulder=(60, 60)
    )
    assert classify_pair_frame(a, b) is False


def test_classify_pair_false_when_a_shoulder_missing() -> None:
    a = _person(1, bbox=(0, 0, 50, 100), left_shoulder=(0, 40), right_shoulder=None)
    b = _person(
        2, bbox=(60, 0, 50, 100), left_shoulder=(60, 40), right_shoulder=(60, 60)
    )
    assert classify_pair_frame(a, b) is False


def test_classify_pair_true_when_shoulders_perpendicular_to_bearing() -> None:
    """Two people at a horizontal bearing, each with a vertical shoulder
    line (perpendicular to the bearing) -- the geometric signature of both
    being oriented along the line connecting them."""
    a = _person(
        1, bbox=(0, 0, 50, 100), left_shoulder=(25, 40), right_shoulder=(25, 60)
    )
    b = _person(
        2, bbox=(60, 0, 50, 100), left_shoulder=(85, 40), right_shoulder=(85, 60)
    )
    assert classify_pair_frame(a, b) is True


def test_classify_pair_false_when_shoulders_parallel_to_bearing() -> None:
    """Shoulder lines running the same direction as the bearing (not
    crosswise to it) is NOT the F-formation signature -- both are oriented
    some direction unrelated to each other."""
    a = _person(
        1, bbox=(0, 0, 50, 100), left_shoulder=(15, 50), right_shoulder=(35, 50)
    )
    b = _person(
        2, bbox=(60, 0, 50, 100), left_shoulder=(75, 50), right_shoulder=(95, 50)
    )
    assert classify_pair_frame(a, b) is False


def test_classify_pair_is_symmetric() -> None:
    a = _person(
        1, bbox=(0, 0, 50, 100), left_shoulder=(25, 40), right_shoulder=(25, 60)
    )
    b = _person(
        2, bbox=(60, 0, 50, 100), left_shoulder=(85, 40), right_shoulder=(85, 60)
    )
    assert classify_pair_frame(a, b) == classify_pair_frame(b, a)


# --------------------------------------------------------------------------- #
# iter_jsonl_pair_signals
# --------------------------------------------------------------------------- #


def test_iter_jsonl_pair_signals_skips_untracked_and_pairs_the_rest(
    tmp_path: Path,
) -> None:
    record = {
        "frame_id": 0,
        "timestamp_ms": 0,
        "persons": [
            _person(None, bbox=(0, 0, 50, 100)),  # untracked -- excluded
            _person(
                1, bbox=(0, 0, 50, 100), left_shoulder=(25, 40), right_shoulder=(25, 60)
            ),
            _person(
                2,
                bbox=(60, 0, 50, 100),
                left_shoulder=(85, 40),
                right_shoulder=(85, 60),
            ),
            _person(3, bbox=(200, 0, 50, 100)),
        ],
        "objects": [],
    }
    path = tmp_path / "stage1.jsonl"
    path.write_text(json.dumps(record), encoding="utf-8")

    results = list(iter_jsonl_pair_signals(path))
    keys = {r[0] for r in results}
    # 3 tracked persons (1, 2, 3) -> exactly 3 unique pairs, none involving None.
    assert keys == {(1, 2), (1, 3), (2, 3)}


def test_iter_jsonl_pair_signals_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        list(iter_jsonl_pair_signals("does/not/exist.jsonl"))


# --------------------------------------------------------------------------- #
# RollingPeerInteractionTracker
# --------------------------------------------------------------------------- #


def _fast_cfg(**overrides) -> PeerInteractionConfig:
    base = {
        "window_seconds": 10.0,
        "majority_fraction": 0.5,
        "sustained_seconds": 20.0,
    }
    base.update(overrides)
    return PeerInteractionConfig(**base)


def test_window_prunes_old_frames() -> None:
    cfg = _fast_cfg(window_seconds=5.0)
    tracker = RollingPeerInteractionTracker(cfg)
    tracker.update((1, 2), 0, True)
    tracker.update((1, 2), 4000, True)
    tracker.update((1, 2), 6000, False)  # 6s after t=0 -> t=0 frame pruned

    assert tracker.window_fraction((1, 2)) == pytest.approx(
        0.5
    )  # 1 True, 1 False remain


def test_unknown_pair_returns_zero_fraction_and_not_sustained() -> None:
    tracker = RollingPeerInteractionTracker(_fast_cfg())
    assert tracker.window_fraction((9, 10)) == 0.0
    assert tracker.is_sustained((9, 10)) is False


def test_sustained_requires_full_duration() -> None:
    cfg = _fast_cfg(sustained_seconds=20.0, window_seconds=5.0)
    tracker = RollingPeerInteractionTracker(cfg)
    for t_s in range(19):
        tracker.update((1, 2), t_s * 1000, True)
    assert tracker.is_sustained((1, 2)) is False

    tracker.update((1, 2), 20_000, True)
    assert tracker.is_sustained((1, 2)) is True


def test_sustained_streak_resets_on_a_broken_run() -> None:
    cfg = _fast_cfg(sustained_seconds=10.0, window_seconds=0.5)
    tracker = RollingPeerInteractionTracker(cfg)
    for t_s in range(9):
        tracker.update((1, 2), t_s * 1000, True)
    tracker.update((1, 2), 9000, False)  # breaks the streak
    for t_s in range(10, 19):
        tracker.update((1, 2), t_s * 1000, True)
    assert tracker.is_sustained((1, 2)) is False  # only ~9s since the reset


def test_active_pairs_only_lists_sustained_ones() -> None:
    cfg = _fast_cfg(sustained_seconds=5.0, window_seconds=5.0)
    tracker = RollingPeerInteractionTracker(cfg)
    for t_s in range(6):
        tracker.update((1, 2), t_s * 1000, True)  # becomes sustained
    tracker.update((3, 4), 0, False)  # never sustained

    assert tracker.active_pairs() == [(1, 2)]
    assert set(tracker.known_pairs()) == {(1, 2), (3, 4)}


def test_summarise_classroom_returns_count_not_identity() -> None:
    cfg = _fast_cfg(sustained_seconds=5.0, window_seconds=5.0)
    tracker = RollingPeerInteractionTracker(cfg)
    for t_s in range(6):
        tracker.update((1, 2), t_s * 1000, True)
    tracker.update((3, 4), 0, False)

    summary = tracker.summarise_classroom()
    assert summary["pairs_considered"] == 2
    assert summary["active_pair_count"] == 1
    # The summary carries only counts -- no key exposing which pair(s), by
    # design (see summarise_classroom's docstring: use active_pairs() for
    # that drill-down explicitly instead).
    assert set(summary) == {"pairs_considered", "active_pair_count"}


def test_summarise_classroom_empty_tracker() -> None:
    tracker = RollingPeerInteractionTracker(_fast_cfg())
    summary = tracker.summarise_classroom()
    assert summary == {"pairs_considered": 0, "active_pair_count": 0}
