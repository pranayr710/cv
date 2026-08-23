"""Unit tests for :mod:`backend.engagement`.

Test coverage:
    1. Off-task behaviour wins over an attending gaze (precedence).
    2. On-task behaviour is reported even with no gaze reading.
    3. Attending gaze alone (no behaviour) counts as on-task.
    4. A bare non-attending gaze with no behaviour is unknown, not off-task.
    5. No evidence at all is unknown.
    6. Aggregation: concentration_pct excludes unknown frames from the
       denominator, and is None (not 0.0) when nothing was graded.
"""

from __future__ import annotations

import pytest

from backend.config import EngagementConfig
from backend.engagement import classify_engagement, summarise_engagement

_CFG = EngagementConfig()


# --------------------------------------------------------------------------- #
# classify_engagement
# --------------------------------------------------------------------------- #


def test_off_task_behaviour_wins_over_attending_gaze() -> None:
    """Contradictory evidence resolves to the more concerning reading, the
    same precedence backend.attention already uses for phone vs book."""
    assert classify_engagement("teacher", "using_device") == "off"
    assert classify_engagement("teacher", "sleep") == "off"


def test_on_task_behaviour_is_reported_without_a_gaze_reading() -> None:
    assert classify_engagement(None, "write") == "on"
    assert classify_engagement(None, "read") == "on"


def test_attending_gaze_alone_counts_as_on_task() -> None:
    assert classify_engagement("teacher", None) == "on"


@pytest.mark.parametrize("gaze", ["left", "right", "down", "back"])
def test_bare_non_attending_gaze_is_unknown_not_off_task(gaze: str) -> None:
    """The core honesty rule this module inherits from backend.attention:
    gaze aversion and peer-oriented turning are real, opposite-reading
    confounds, so a bare gaze label must never be guessed into "off"."""
    assert classify_engagement(gaze, None) is None


def test_no_evidence_at_all_is_unknown() -> None:
    assert classify_engagement(None, None) is None


def test_unrecognised_behaviour_label_falls_back_to_gaze() -> None:
    """A behaviour label outside both on/off sets (e.g. a future class) must
    not silently break the fallback to gaze."""
    assert classify_engagement("teacher", "some_new_class") == "on"
    assert classify_engagement("left", "some_new_class") is None


def test_config_is_respected_over_the_default() -> None:
    from dataclasses import replace

    custom = replace(_CFG, on_task_behaviours=("napping_but_ok",))
    assert classify_engagement(None, "napping_but_ok", custom) == "on"
    assert classify_engagement(None, "write", custom) is None


# --------------------------------------------------------------------------- #
# summarise_engagement
# --------------------------------------------------------------------------- #


def test_concentration_pct_excludes_unknown_frames() -> None:
    """3 on, 1 off, 2 unknown -> 75% of GRADED frames, not 50% of all frames."""
    verdicts = ["on", "on", "on", "off", None, None]
    summary = summarise_engagement(verdicts)
    assert summary["frames"] == 6
    assert summary["on"] == 3
    assert summary["off"] == 1
    assert summary["unknown"] == 2
    assert summary["concentration_pct"] == pytest.approx(75.0)


def test_concentration_pct_is_none_when_nothing_graded() -> None:
    """None, not 0.0 -- a report must not read 'no evidence' as 'off-task'."""
    summary = summarise_engagement([None, None, None])
    assert summary["concentration_pct"] is None
    assert summary["on"] == 0
    assert summary["off"] == 0


def test_summary_of_empty_sequence() -> None:
    summary = summarise_engagement([])
    assert summary["frames"] == 0
    assert summary["concentration_pct"] is None


def test_all_on_task_is_100_percent() -> None:
    summary = summarise_engagement(["on", "on", "on"])
    assert summary["concentration_pct"] == pytest.approx(100.0)


def test_all_off_task_is_zero_percent() -> None:
    summary = summarise_engagement(["off", "off"])
    assert summary["concentration_pct"] == pytest.approx(0.0)
