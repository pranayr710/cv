"""Tests for the window feature extractor.

Each test exists for a specific way this could be quietly wrong. A feature
matrix that looks plausible but encodes nothing is the failure mode here: it
trains, it reports a number, and the number means nothing.
"""
import json
from pathlib import Path

import pytest

from backend.engagement_features import (
    FEATURE_NAMES,
    MIN_COVERAGE,
    Window,
    iter_windows,
    window_features,
)


def _frame(**kw):
    """A node feature dict with everything absent unless named."""
    base = {
        "action": None, "object": None, "gaze_label": None, "oriented": None,
        "focus_offset_deg": None, "expression": None, "posture": None,
        "engagement": None, "is_eyes_closed_sustained": None,
    }
    base.update(kw)
    return base


def test_feature_vector_length_matches_the_names():
    """A mismatch here silently shifts every feature importance by one."""
    feats = window_features([_frame(action="attentive")], expected_frames=1)
    assert len(feats) == len(FEATURE_NAMES)


def test_empty_window_is_all_zero_not_a_crash():
    """A student absent for a whole window must produce a vector, not raise."""
    feats = window_features([], expected_frames=40)
    assert len(feats) == len(FEATURE_NAMES)
    assert all(v == 0.0 for v in feats)


def test_coverage_is_measured_against_expected_not_actual():
    """Ten frames in a forty-frame window is quarter coverage.

    Measuring against len(frames) would make every window fully covered, which
    would destroy the one feature that lets a learned scorer abstain.
    """
    feats = window_features([_frame(action="attentive")] * 10, expected_frames=40)
    assert feats[FEATURE_NAMES.index("coverage")] == pytest.approx(0.25)


def test_on_and_off_task_fractions_separate():
    """The two action fractions must not both fire on the same frame."""
    frames = [_frame(action="attentive")] * 3 + [_frame(action="on_phone")] * 1
    feats = window_features(frames, expected_frames=4)
    on = feats[FEATURE_NAMES.index("frac_on_task_action")]
    off = feats[FEATURE_NAMES.index("frac_off_task_action")]
    assert on == pytest.approx(0.75)
    assert off == pytest.approx(0.25)


def test_entropy_is_zero_for_a_constant_action():
    """A student doing one thing all window has no action variability."""
    feats = window_features([_frame(action="writing")] * 8, expected_frames=8)
    assert feats[FEATURE_NAMES.index("action_entropy")] == pytest.approx(0.0)


def test_entropy_rises_when_actions_vary():
    """Two equally-frequent actions is exactly one bit."""
    frames = [_frame(action="writing")] * 4 + [_frame(action="on_phone")] * 4
    feats = window_features(frames, expected_frames=8)
    assert feats[FEATURE_NAMES.index("action_entropy")] == pytest.approx(1.0)


def test_switch_rate_is_bounded_and_scale_free():
    """Alternating every frame is the maximum; it must not exceed 1.

    An earlier version reported switches per minute, which scaled with frame
    rate and exceeded 100 -- a feature whose magnitude depended on the camera.
    """
    frames = [_frame(action="writing" if i % 2 else "on_phone") for i in range(10)]
    feats = window_features(frames, expected_frames=10)
    rate = feats[FEATURE_NAMES.index("action_switch_rate")]
    assert rate == pytest.approx(1.0)
    assert 0.0 <= rate <= 1.0


def test_attending_gaze_uses_the_configured_label():
    """The attending label is 'teacher', not 'center'.

    The first version hard-coded ('center', 'front', 'forward') and produced a
    feature that was zero on every window in the real dataset -- a dead column
    that a model would silently ignore.
    """
    from backend.config import CONFIG

    label = CONFIG.engagement.attending_gaze_labels[0]
    feats = window_features([_frame(gaze_label=label)] * 4, expected_frames=4)
    assert feats[FEATURE_NAMES.index("frac_gaze_attending")] == pytest.approx(1.0)


def test_rule_verdict_is_carried_but_not_a_feature():
    """The rule verdict must stay out of the feature vector.

    If it leaked in, a classifier would learn to copy the rules and report
    near-perfect agreement with them, which measures nothing.
    """
    assert "engagement" not in FEATURE_NAMES
    assert "rule_verdict" not in FEATURE_NAMES
    assert not any("verdict" in n for n in FEATURE_NAMES)


def test_iter_windows_skips_windows_below_min_coverage(tmp_path: Path):
    """A window with almost no readable frames is not a training example."""
    rows = []
    for i in range(60):
        nodes = []
        # Present for only the first two frames of a long span.
        if i < 2:
            nodes.append({"person_id": 1, "features": _frame(action="attentive")})
        rows.append(json.dumps({"timestamp_ms": i * 500, "scene": 0,
                                "nodes": nodes}))
    path = tmp_path / "graph.jsonl"
    path.write_text("\n".join(rows), encoding="utf-8")
    windows = list(iter_windows(path))
    assert all(w.features[0] >= MIN_COVERAGE for w in windows)


def test_iter_windows_yields_typed_windows_with_identity(tmp_path: Path):
    """Every example must carry the student it belongs to, for splitting.

    Overlapping windows from one student are correlated; splitting them at
    random would leak the same student into train and test and inflate the
    score. That split is only possible if person_id survives extraction.
    """
    rows = []
    for i in range(80):
        rows.append(json.dumps({
            "timestamp_ms": i * 500, "scene": 0,
            "nodes": [{"person_id": 7,
                       "features": _frame(action="attentive",
                                          engagement="on")}],
        }))
    path = tmp_path / "graph.jsonl"
    path.write_text("\n".join(rows), encoding="utf-8")
    windows = list(iter_windows(path))
    assert windows, "expected at least one window from 40 seconds of frames"
    assert all(isinstance(w, Window) for w in windows)
    assert {w.person_id for w in windows} == {7}
    assert all(w.rule_verdict == "on" for w in windows)
