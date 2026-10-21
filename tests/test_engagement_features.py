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
            PEER_COUNT_HALF,
            peer_features,
        )

        i_on = BASE_FEATURE_NAMES.index("frac_on_task_action")
        a = [0.0] * len(BASE_FEATURE_NAMES)
        b = [0.0] * len(BASE_FEATURE_NAMES)
        a[i_on], b[i_on] = 1.0, 0.0
        on, _gaze, count = peer_features([a, b])
        assert on == pytest.approx(0.5)
        assert count == pytest.approx(2 / (2 + PEER_COUNT_HALF))

    def test_peer_count_saturates_rather_than_scaling(self):
        """Dividing a raw count rescales it; it does not bound it.

        The first version divided by 10, so a class of 30 gave 2.9 against a
        training mean of 0.72 and a scale of 0.138 -- +16.5 standard
        deviations, and it would have appeared the first time this ran in a
        bigger room than the one it was built in. That is the defect n_frames
        had, with a different divisor.
        """
        from backend.engagement_features import BASE_FEATURE_NAMES, peer_features

        def count_for(n):
            return peer_features(
                [[0.0] * len(BASE_FEATURE_NAMES)] * n)[2] if n else 0.0

        for n in (1, 5, 11, 30, 60, 200):
            assert 0.0 <= count_for(n) < 1.0, f"{n} peers escaped [0, 1)"

        # Still ordered, so "a few peers" and "many peers" remain different.
        assert count_for(1) < count_for(5) < count_for(11) < count_for(30)

        # And the room sizes that differ meaningfully are still separated,
        # while the ones that do not have converged.
        assert count_for(5) - count_for(1) > 0.2
        assert count_for(60) - count_for(30) < 0.1

    def test_a_lone_student_has_no_peers(self):
        from backend.engagement_features import peer_features

        assert peer_features([])[2] == 0.0

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


class TestCoverageIsBounded:
    """Coverage gates abstention, so an inflated value disables the refusal."""

    def test_more_frames_than_expected_still_caps_at_one(self):
        """A 13 fps webcam against a 3 fps assumption gave coverage 4.33.

        Standardised against the shipped model that is +21 sigma -- larger
        than the 12.8 sigma that n_frames produced when it broke live scoring
        and pinned every student at 1/10.
        """
        feats = window_features([_frame(action="attentive")] * 195,
                                expected_frames=45)
        assert feats[FEATURE_NAMES.index("coverage")] == 1.0

    def test_a_half_covered_window_is_unaffected(self):
        """Clamping must not touch the range the feature actually uses."""
        feats = window_features([_frame(action="attentive")] * 20,
                                expected_frames=40)
        assert feats[FEATURE_NAMES.index("coverage")] == pytest.approx(0.5)


class TestPeersAreResolvedBySpan:
    """Peers used to be grouped by exact window-start equality.

    A student's windows begin at their own first observed frame, so two
    students present for the same fifteen seconds who entered at different
    moments never appeared in each other's cohort. On a 14-student recording
    that left 40 per cent of windows claiming no peers while six to eleven
    students were visible.
    """

    def test_a_student_who_arrived_later_is_still_a_peer(self, tmp_path):
        from backend.engagement_features import iter_windows

        rows = []
        for i in range(120):
            nodes = [{"person_id": 1,
                      "features": _frame(action="attentive")}]
            # Student 2 appears 10 seconds in, so their windows start on a
            # different boundary than student 1's.
            if i >= 30:
                nodes.append({"person_id": 2,
                              "features": _frame(action="on_phone")})
            rows.append(json.dumps({"frame_id": i, "timestamp_ms": i * 333,
                                    "scene": 0, "nodes": nodes}))
        path = tmp_path / "graph.jsonl"
        path.write_text("\n".join(rows), encoding="utf-8")

        windows = list(iter_windows(path))
        i_count = FEATURE_NAMES.index("peer_count")
        overlapping = [w for w in windows
                       if w.person_id == 1 and w.start_ms >= 10_000]
        assert overlapping, "expected student 1 windows after student 2 arrived"
        assert any(w.features[i_count] > 0 for w in overlapping), (
            "student 2 was visible in these windows but counted as no peer")

    def test_a_student_outside_the_span_is_not_a_peer(self, tmp_path):
        """The opposite failure: counting someone who was not there.

        The live path had this one, taking any student with buffered frames
        regardless of span.
        """
        from backend.engagement_features import iter_windows

        rows = []
        for i in range(120):
            # The two never overlap.
            pid = 1 if i < 45 else 2
            rows.append(json.dumps({
                "frame_id": i, "timestamp_ms": i * 333, "scene": 0,
                "nodes": [{"person_id": pid,
                           "features": _frame(action="attentive")}]}))
        path = tmp_path / "graph.jsonl"
        path.write_text("\n".join(rows), encoding="utf-8")

        i_count = FEATURE_NAMES.index("peer_count")
        for w in iter_windows(path):
            assert w.features[i_count] == 0.0, (
                f"student {w.person_id} at {w.start_ms}ms had a peer who was "
                "not in frame")

    def test_a_student_is_never_their_own_peer(self):
        from backend.engagement_features import peer_vectors_in_span

        frames = {1: [(0, _frame(action="attentive"))],
                  2: [(0, _frame(action="on_phone"))]}
        peers = peer_vectors_in_span(frames, 1, 0, 15_000, 1,
                                     min_coverage=0.0)
        assert len(peers) == 1


class TestBodyFeatures:
    """Posture geometry: the axis the original eighteen were blind to.

    Every quantity here is divided by a distance taken from the same body, so
    none of them carries how far the student sits from the camera. That is the
    property two earlier features lacked, and both broke the model when the
    recording setup changed.
    """

    @staticmethod
    def _posed(**kw):
        """A frame with a complete, upright pose."""
        posture = {
            "left_shoulder": [100.0, 200.0],
            "right_shoulder": [180.0, 200.0],
            "shoulder_mid": [140.0, 200.0],
            "hip_mid": [140.0, 300.0],
            "facing_direction": [0.0, 1.0],
            "left_wrist": [110.0, 280.0],
            "right_wrist": [170.0, 280.0],
        }
        posture.update(kw.pop("posture", {}))
        kw.setdefault("focus_offset_deg", 0.0)
        return _frame(posture=posture, **kw)

    def test_the_vector_has_one_value_per_name(self):
        from backend.engagement_features import BODY_FEATURE_NAMES, body_features

        assert len(body_features([self._posed()])) == len(BODY_FEATURE_NAMES)

    def test_distance_from_the_camera_does_not_change_them(self):
        """A student at the back must score the same as one at the front.

        Everything is a ratio of two distances on the same body, so scaling
        the whole pose must leave every value unchanged. A feature that fails
        this looks fine on one recording and drifts on the next.
        """
        from backend.engagement_features import body_features

        near = self._posed()
        far = self._posed(posture={
            k: [v[0] * 0.4, v[1] * 0.4]
            for k, v in self._posed()["posture"].items()
            if k != "facing_direction"})
        far["posture"]["facing_direction"] = [0.0, 1.0]

        for a, b in zip(body_features([near] * 5), body_features([far] * 5)):
            assert a == pytest.approx(b, abs=1e-6)

    def test_hands_visible_counts_only_frames_with_a_pose(self):
        from backend.engagement_features import BODY_FEATURE_NAMES, body_features

        i = BODY_FEATURE_NAMES.index("hands_visible")
        with_hands = self._posed()
        without = self._posed()
        without["posture"] = {k: v for k, v in without["posture"].items()
                              if "wrist" not in k}
        assert body_features([with_hands] * 4)[i] == pytest.approx(1.0)
        assert body_features([without] * 4)[i] == pytest.approx(0.0)
        assert body_features([with_hands, without])[i] == pytest.approx(0.5)

    def test_facing_variability_is_zero_when_the_body_does_not_turn(self):
        from backend.engagement_features import BODY_FEATURE_NAMES, body_features

        i = BODY_FEATURE_NAMES.index("facing_variability")
        assert body_features([self._posed()] * 6)[i] == pytest.approx(0.0)

    def test_facing_variability_rises_when_the_body_turns(self):
        from backend.engagement_features import BODY_FEATURE_NAMES, body_features

        i = BODY_FEATURE_NAMES.index("facing_variability")
        turning = []
        for direction in ([0.0, 1.0], [1.0, 0.0], [0.0, -1.0], [-1.0, 0.0]):
            frame = self._posed()
            frame["posture"]["facing_direction"] = direction
            turning.append(frame)
        assert body_features(turning)[i] > 0.2

    def test_misalignment_is_low_when_head_and_body_agree(self):
        """Square shoulders and a forward gaze is an aligned student."""
        from backend.engagement_features import BODY_FEATURE_NAMES, body_features

        i = BODY_FEATURE_NAMES.index("head_torso_misalign")
        assert body_features([self._posed()] * 4)[i] < 0.3

    def test_misalignment_rises_when_the_head_turns_but_the_body_does_not(self):
        """The case frac_oriented cannot see: compliance rather than attention."""
        from backend.engagement_features import BODY_FEATURE_NAMES, body_features

        i = BODY_FEATURE_NAMES.index("head_torso_misalign")
        square = body_features([self._posed()] * 4)[i]
        turned = body_features([self._posed(focus_offset_deg=80.0)] * 4)[i]
        assert turned > square

    def test_a_window_with_no_pose_is_all_zero(self):
        """Absent geometry is not symmetric geometry.

        Coverage already tells the model how much of the window was readable,
        so these must not invent a value for the part that was not.
        """
        from backend.engagement_features import body_features

        assert body_features([_frame(), _frame()]) == (0.0, 0.0, 0.0)
        assert body_features([]) == (0.0, 0.0, 0.0)

    def test_they_are_bounded(self):
        """Nothing here may grow with class size, frame rate or resolution."""
        from backend.engagement_features import body_features

        many = [self._posed(focus_offset_deg=90.0)] * 200
        for value in body_features(many):
            assert 0.0 <= value <= 2.0

    def test_the_window_vector_places_them_between_base_and_peer(self):
        """Order is the contract between the model's weights and the columns."""
        from backend.engagement_features import (
            BASE_FEATURE_NAMES,
            BODY_FEATURE_NAMES,
            FEATURE_NAMES,
        )

        start = len(BASE_FEATURE_NAMES)
        end = start + len(BODY_FEATURE_NAMES)
        assert FEATURE_NAMES[start:end] == BODY_FEATURE_NAMES
