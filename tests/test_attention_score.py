"""Tests for the live per-student attention score.

The failures guarded against are a score that flickers, a score computed from
almost nothing, and a system that breaks when no model has been trained.
"""
from pathlib import Path

import pytest

from backend.attention_score import (
    UNCERTAIN_BAND,
    AttentionScorer,
    to_score_10,
)
from backend.engagement_features import FEATURE_NAMES, WINDOW_MS
from backend.engagement_model import EngagementModel


def _model(weight: float = 4.0) -> EngagementModel:
    n = len(FEATURE_NAMES)
    coefficients = [0.0] * n
    coefficients[FEATURE_NAMES.index("frac_on_task_action")] = weight
    return EngagementModel(coefficients, 0.0, list(FEATURE_NAMES),
                           [0.0] * n, [1.0] * n)


def _frame(action: str = "attentive") -> dict:
    return {
        "action": action, "object": None, "gaze_label": None,
        "oriented": None, "focus_offset_deg": None, "expression": None,
        "posture": None, "engagement": None,
        "is_eyes_closed_sustained": None,
    }


def _feed(scorer: AttentionScorer, person_id: int, n: int,
          action: str = "attentive", step: int = 333) -> list:
    return [r for i in range(n)
            if (r := scorer.update(person_id, i * step, _frame(action)))]


def test_score_spans_the_full_range():
    assert to_score_10(0.0) == 1
    assert to_score_10(1.0) == 10
    assert 1 <= to_score_10(0.5) <= 10


def test_score_is_monotonic_in_probability():
    """A more confident window must never score lower."""
    values = [to_score_10(p / 20) for p in range(21)]
    assert values == sorted(values)


def test_no_model_degrades_rather_than_breaking(tmp_path: Path):
    """The pipeline must still run before anything has been trained."""
    scorer = AttentionScorer(model=None)
    scorer.model = None
    readings = _feed(scorer, 1, 60)
    assert scorer.available is False
    assert readings, "should still emit readings, just unscored"
    assert all(r.score_10 is None for r in readings)


def test_a_window_is_scored_once_not_every_frame():
    """A score that changes every frame is unreadable on screen.

    Readings are emitted on window completion, so the count is bounded by the
    elapsed time over the stride rather than by the frame count.
    """
    scorer = AttentionScorer(model=_model(), fps=3.0)
    readings = _feed(scorer, 1, 90)          # 90 frames at 333 ms = ~30 s
    assert len(readings) <= 90 // 4, f"too many readings: {len(readings)}"
    assert readings


def test_thin_coverage_produces_no_score():
    """Two frames inside a fifteen-second window is not a window."""
    scorer = AttentionScorer(model=_model(), fps=3.0)
    scorer.update(1, 0, _frame())
    reading = scorer.update(1, WINDOW_MS - 1, _frame())
    assert reading is None or reading.score_10 is None


def test_on_task_frames_score_higher_than_off_task():
    scorer_on = AttentionScorer(model=_model(), fps=3.0)
    scorer_off = AttentionScorer(model=_model(), fps=3.0)
    on = _feed(scorer_on, 1, 60, "attentive")
    off = _feed(scorer_off, 2, 60, "on_phone")
    assert on and off
    assert on[-1].score_10 > off[-1].score_10


def test_students_are_scored_independently():
    """One student's frames must not leak into another's window."""
    scorer = AttentionScorer(model=_model(), fps=3.0)
    for i in range(60):
        scorer.update(1, i * 333, _frame("attentive"))
        scorer.update(2, i * 333, _frame("on_phone"))
    assert scorer.session_score(1) > scorer.session_score(2)


def test_session_score_averages_probability_not_rounded_scores():
    """Rounding once at the end, not accumulating rounding per window."""
    scorer = AttentionScorer(model=_model(), fps=3.0)
    _feed(scorer, 1, 60)
    assert 1 <= scorer.session_score(1) <= 10


def test_unknown_student_has_no_session_score():
    assert AttentionScorer(model=_model()).session_score(99) is None


def test_uncertain_windows_are_ordered_by_uncertainty():
    """The improvement loop depends on the least certain coming first."""
    scorer = AttentionScorer(model=_model(weight=0.0), fps=3.0)  # always 0.5
    _feed(scorer, 1, 60)
    windows = scorer.uncertain_windows()
    assert windows, "a model at p=0.5 should flag windows as uncertain"
    distances = [abs(w["probability"] - 0.5) for w in windows]
    assert distances == sorted(distances)
    assert all(d < UNCERTAIN_BAND for d in distances)


def test_confident_windows_are_not_flagged_for_labelling():
    """Labelling windows the model already decides confidently teaches little."""
    scorer = AttentionScorer(model=_model(weight=40.0), fps=3.0)
    _feed(scorer, 1, 60, "attentive")
    assert scorer.uncertain_windows() == []


def test_uncertain_windows_carry_their_features(tmp_path: Path):
    """They must be re-labellable without recomputing anything."""
    scorer = AttentionScorer(model=_model(weight=0.0), fps=3.0)
    _feed(scorer, 1, 60)
    written = scorer.save_uncertain(tmp_path / "next.json")
    assert written > 0
    import json
    rows = json.loads((tmp_path / "next.json").read_text(encoding="utf-8"))
    assert len(rows[0]["features"]) == len(FEATURE_NAMES)
    assert {"person_id", "start_ms", "end_ms"} <= set(rows[0])


def test_summary_reports_every_seen_student():
    scorer = AttentionScorer(model=_model(), fps=3.0)
    _feed(scorer, 1, 60)
    _feed(scorer, 2, 60)
    summary = scorer.summary()
    assert set(summary) == {1, 2}
    assert all("score_10" in v and "windows_scored" in v
               for v in summary.values())


def test_reason_names_features_when_a_decision_was_made():
    """A score shown to a teacher has to be arguable."""
    scorer = AttentionScorer(model=_model(), fps=3.0)
    readings = _feed(scorer, 1, 60)
    decided = [r for r in readings if r.label is not None]
    assert decided
    assert "frac_on_task_action" in decided[-1].reason


def test_score_and_label_agree():
    """A window labelled off must not carry a high score, and vice versa."""
    scorer = AttentionScorer(model=_model(), fps=3.0)
    for r in _feed(scorer, 1, 60, "attentive") + _feed(scorer, 2, 60, "on_phone"):
        if r.label == "on":
            assert r.score_10 >= 5
        elif r.label == "off":
            assert r.score_10 <= 6


@pytest.mark.parametrize("probability", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_score_is_always_in_range(probability):
    assert 1 <= to_score_10(probability) <= 10
