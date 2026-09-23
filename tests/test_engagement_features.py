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


def test_window_carries_the_frame_ids_it_covers(tmp_path: Path):
    """Frame ids must come from the log, not be derived from elapsed time.

    Deriving them assumed every source clip had been processed. Only 60 of 157
    were, so the estimated frame interval was 2.6x too small and crops resolved
    for fewer than half the windows.
    """
    rows = []
    for i in range(80):
        rows.append(json.dumps({
            "frame_id": i, "timestamp_ms": i * 333, "scene": 0,
            "nodes": [{"person_id": 3, "features": _frame(action="attentive")}],
        }))
    path = tmp_path / "graph.jsonl"
    path.write_text("\n".join(rows), encoding="utf-8")
    windows = list(iter_windows(path))
    assert windows
    for w in windows:
        assert w.frame_ids, "a window with frames must name them"
        assert all(isinstance(f, int) for f in w.frame_ids)
        # Coverage is how full the window is, so it must move with the frames
        # actually present. n_frames used to be asserted here and was removed
        # from the feature set: as a raw count it scaled with frame rate, and
        # a webcam running faster than the batch pipeline pushed it thirteen
        # standard deviations out and drove every live score to 1/10.
        coverage = w.features[FEATURE_NAMES.index("coverage")]
        # Slightly above 1.0 is legitimate: expected_frames is estimated from
        # the frame rate, so a window can hold a frame or two more than
        # predicted. What matters is that it stays near 1 rather than scaling
        # with the rate the way a raw count would.
        assert 0.0 < coverage <= 1.2


def test_no_feature_is_a_raw_count():
    """Every feature must be scale-free, or frame rate leaks into the model.

    A raw count is not comparable between a 3 fps batch run and a live webcam,
    and standardising one against the other's mean produces a value far outside
    anything the model saw in training.
    """
    frames = [_frame(action="attentive")] * 40
    few = window_features(frames[:10], expected_frames=10)
    many = window_features(frames, expected_frames=40)
    for i, name in enumerate(FEATURE_NAMES):
        assert -200.0 <= few[i] <= 200.0, f"{name} looks unbounded"
        assert -200.0 <= many[i] <= 200.0, f"{name} looks unbounded"
    assert "n_frames" not in FEATURE_NAMES


class TestPeerContext:
    """The room is context, and it was the largest gain found without labels."""

    def test_peer_columns_are_zero_when_nobody_else_is_there(self):
        """A student alone has no room to compare against.

        Zero rather than an omitted column, so the vector length never varies
        between a full classroom and a one-person webcam session.
        """
        from backend.engagement_features import PEER_FEATURE_NAMES, peer_features

        assert peer_features([]) == (0.0,) * len(PEER_FEATURE_NAMES)

    def test_peer_features_average_the_others(self):
        from backend.engagement_features import (
            BASE_FEATURE_NAMES,
            PEER_COUNT_SCALE,
            peer_features,
        )

        i_on = BASE_FEATURE_NAMES.index("frac_on_task_action")
        a = [0.0] * len(BASE_FEATURE_NAMES)
        b = [0.0] * len(BASE_FEATURE_NAMES)
        a[i_on], b[i_on] = 1.0, 0.0
        on, _gaze, count = peer_features([a, b])
        assert on == pytest.approx(0.5)
        assert count == pytest.approx(2 / PEER_COUNT_SCALE)

    def test_peer_count_is_scaled_not_raw(self):
        """A raw count would scale with class size.

        That is the same defect n_frames had: a magnitude that depends on the
        recording setup rather than on the student, which shifts far outside
        the training range when the setup changes.
        """
        from backend.engagement_features import BASE_FEATURE_NAMES, peer_features

        many = [[0.0] * len(BASE_FEATURE_NAMES) for _ in range(30)]
        assert peer_features(many)[2] <= 5.0

    def test_a_students_own_vector_is_not_in_its_peer_summary(self, tmp_path):
        """Including yourself would make the feature partly a copy of another.

        Two students in the same window would each carry a peer summary built
        from the other plus themselves, and the column would stop measuring
        the room.
        """
        from backend.engagement_features import FEATURE_NAMES, iter_windows

        rows = []
        for i in range(80):
            rows.append(json.dumps({
                "frame_id": i, "timestamp_ms": i * 333, "scene": 0,
                "nodes": [
                    {"person_id": 1, "features": _frame(action="attentive")},
                    {"person_id": 2, "features": _frame(action="on_phone")},
                ],
            }))
        path = tmp_path / "graph.jsonl"
        path.write_text("\n".join(rows), encoding="utf-8")
        windows = {w.person_id: w for w in iter_windows(path)}
        assert set(windows) == {1, 2}

        i_peer_on = FEATURE_NAMES.index("peer_frac_on_task")
        # Student 1 is on task, so student 2's peer summary must say so, and
        # vice versa. If each included itself the two would converge.
        assert windows[2].features[i_peer_on] > 0.9
        assert windows[1].features[i_peer_on] < 0.1
