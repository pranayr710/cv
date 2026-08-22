"""Tests for the camera-relative yaw reference in :mod:`backend.headpose`.

Regression tests for a real bug that would have corrupted every engagement
figure computed from side- or corner-mounted camera footage.

The gaze buckets measure head rotation relative to the *camera* and treat yaw
~0 as "attending". That assumption only holds when the camera sits where the
teacher and board are. On a real corner-mounted classroom clip it produced
gaze_label "right" for 320 of 383 faces (84%), median yaw +37 deg -- the
head-pose model was correct (students really were rotated relative to that
camera, facing a board off-frame), but every attending student was labelled as
looking away, which feeds backend.attention's off-task bucket.

Applying the measured reference of +37 deg moved "teacher" from 5.2% to 37.9%
of faces on the same data.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from backend.config import CONFIG, HeadPoseConfig
from backend.headpose import classify_gaze, estimate_yaw_reference

_CFG = CONFIG.headpose


def _cfg(ref: float) -> HeadPoseConfig:
    return replace(HeadPoseConfig(), yaw_reference_deg=ref)


# --------------------------------------------------------------------------- #
# The offset changes what "attending" means
# --------------------------------------------------------------------------- #


def test_default_reference_is_zero_so_existing_behaviour_is_unchanged() -> None:
    """A front-mounted camera must behave exactly as before this feature."""
    assert HeadPoseConfig().yaw_reference_deg == 0.0
    assert classify_gaze(0.0, 0.0, HeadPoseConfig()) == "teacher"


def test_turned_head_is_attending_once_the_camera_offset_is_applied() -> None:
    """The actual bug: yaw +37 from a corner camera IS facing the front."""
    turned = 37.0
    assert classify_gaze(turned, 0.0, _cfg(0.0)) == "right"
    assert classify_gaze(turned, 0.0, _cfg(37.0)) == "teacher"


def test_offset_shifts_both_side_buckets_consistently() -> None:
    """Left/right must stay symmetric about the new reference, not the camera."""
    cfg = _cfg(37.0)
    # 37 + 25 is past the +20 side threshold measured from the reference.
    assert classify_gaze(37.0 + 25.0, 0.0, cfg) == "right"
    assert classify_gaze(37.0 - 25.0, 0.0, cfg) == "left"
    # Just inside the threshold on either side is still attending.
    assert classify_gaze(37.0 + 10.0, 0.0, cfg) == "teacher"
    assert classify_gaze(37.0 - 10.0, 0.0, cfg) == "teacher"


def test_facing_the_camera_becomes_looking_away_under_an_offset() -> None:
    """The converse, which is the honest cost of this correction.

    With a corner camera, a student looking straight *at the camera* is not
    attending — they are turned away from the board. Yaw 0 must therefore stop
    being "teacher" once a reference is set.
    """
    assert classify_gaze(0.0, 0.0, _cfg(37.0)) == "left"


def test_pitch_buckets_are_unaffected_by_the_yaw_reference() -> None:
    """The offset is yaw-only; a bowed head must still read as 'down'."""
    cfg = _cfg(37.0)
    assert classify_gaze(37.0, _CFG.pitch_down_threshold + 5, cfg) == "down"


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_yaw_still_rejected_with_an_offset(bad: float) -> None:
    with pytest.raises(ValueError):
        classify_gaze(bad, 0.0, _cfg(37.0))


# --------------------------------------------------------------------------- #
# Estimating the reference from data
# --------------------------------------------------------------------------- #


def test_reference_is_the_median_of_the_samples() -> None:
    """Median, not mean: it must ignore the genuinely-turned-away minority."""
    attending = [40.0] * 30
    turned_away = [-60.0] * 5      # a few students really are looking elsewhere
    assert estimate_yaw_reference(attending + turned_away) == pytest.approx(40.0)


def test_reference_refuses_to_estimate_from_too_few_samples() -> None:
    """A wrong reference is worse than none — it silently shifts every label."""
    assert estimate_yaw_reference([37.0] * 5) is None
    assert estimate_yaw_reference([]) is None


def test_min_samples_is_configurable() -> None:
    assert estimate_yaw_reference([37.0] * 5, min_samples=5) == pytest.approx(37.0)


def test_non_finite_samples_are_discarded_not_propagated() -> None:
    """One NaN must not poison the estimate into NaN."""
    yaws = [30.0] * 25 + [float("nan"), float("inf")]
    result = estimate_yaw_reference(yaws)
    assert result == pytest.approx(30.0)


def test_estimate_round_trips_through_classify_gaze() -> None:
    """End-to-end: estimate a reference, then attending students read attending.

    Mirrors the real measurement — a cluster of students at ~+37 deg from a
    corner camera — and asserts the pair of functions actually agree.
    """
    yaws = [35.0, 36.0, 37.0, 38.0, 39.0] * 6
    ref = estimate_yaw_reference(yaws)
    assert ref is not None
    cfg = _cfg(ref)
    assert all(classify_gaze(y, 0.0, cfg) == "teacher" for y in yaws)
