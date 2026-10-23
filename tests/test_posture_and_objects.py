"""Two things the output was measuring and then throwing away.

**Lean.** ``vertical_lean`` was computed on 334 of 424 students in one
recording and reached the output on 2 of them, because it sat fifteenth of
eighteen in the action precedence and lost to every object rule, both gaze
rules and head-down. Posture is not a competitor to "on phone" -- a student
can be on their phone *and* slouching -- so it is now reported in its own
field on every frame.

**Object links.** Every detected object class reached the scene graph, with no
floor under how long it was attributed to anyone. Measured over the 424
person-object links in the sessions on disk, 219 rested on under 5% of that
student's frames and the weakest rested on a single frame -- one of which put
a cup in a results page for a room that had no cup in it, drawn exactly like a
laptop that was visible throughout.
"""
from collections import Counter

import pytest

from backend.actions import (
    FORWARD_LEAN,
    SLOUCH_LEAN,
    annotate_graph,
    classify,
    lean_state,
)

PERSON = (100, 100, 80, 200)


def _graph(gaze="teacher"):
    return {"frame_id": 0, "timestamp_ms": 0, "edges": [],
            "nodes": [{"id": 1, "person_id": 1, "role": "student",
                       "features": {"gaze_label": gaze}}]}


def _record(lean=None, objects=()):
    posture = None if lean is None else {"vertical_lean": lean,
                                         "shoulder_mid": (140, 180)}
    return {"frame_id": 0, "timestamp_ms": 0,
            "persons": [{"person_id": 1, "bbox": PERSON, "posture": posture,
                         "head_pose": {}, "face": {}}],
            "objects": list(objects)}


class TestLeanIsAlwaysReported:
    def test_it_is_reported_even_when_another_action_wins(self):
        """The whole defect in one assertion.

        A phone in hand wins the action, as it should. The lean was measured
        on the same frame and used to be discarded for losing that contest.
        """
        graph = annotate_graph(
            _graph(), _record(lean=0.1, objects=[
                {"cls": "cell phone", "bbox": (110, 150, 20, 30),
                 "confidence": 0.9}]))
        features = graph["nodes"][0]["features"]
        assert features["lean_state"] == "slouching"
        assert features["vertical_lean"] == pytest.approx(0.1)

    def test_an_unreadable_posture_is_null_not_upright(self):
        """Not having seen someone's shoulders is not evidence they sat up."""
        graph = annotate_graph(_graph(), _record(lean=None))
        assert graph["nodes"][0]["features"]["lean_state"] is None
        assert graph["nodes"][0]["features"]["vertical_lean"] is None

    @pytest.mark.parametrize("lean,expected", [
        (0.05, "slouching"), (-0.13, "upright"), (-0.30, "leaning_forward")])
    def test_the_bands(self, lean, expected):
        assert lean_state({"vertical_lean": lean})[0] == expected


class TestTheForwardThresholdIsReachable:
    def test_it_sits_inside_the_range_the_measurement_produces(self):
        """It did not. The threshold was -0.30 and the most forward lean in a
        63-frame recording was -0.279, so "leaning forward" was not rare --
        it was unreachable. Across 41,421 frames the signal has mean -0.131
        and standard deviation 0.081, and both thresholds are now one
        standard deviation out.
        """
        mean, sd = -0.131, 0.081
        assert FORWARD_LEAN > mean - 2 * sd
        assert SLOUCH_LEAN < mean + 2 * sd

    def test_the_two_bands_are_symmetric_about_the_population(self):
        mean = -0.131
        assert abs((SLOUCH_LEAN - mean) + (FORWARD_LEAN - mean)) < 0.02

    def test_a_forward_lean_is_actually_classified(self):
        a = classify(PERSON, [], "teacher", 0.0, 0.3,
                     posture={"vertical_lean": FORWARD_LEAN - 0.01})
        assert a.name == "leaning_forward"


class TestObjectConfidenceIsCarried:
    def test_the_action_carries_what_the_detector_was_sure_of(self):
        a = classify(PERSON, [
            {"cls": "laptop", "bbox": (110, 150, 40, 40), "confidence": 0.91},
        ], "teacher", 0.0, 0.3)
        assert a.name == "on_laptop"
        assert a.obj_confidence == pytest.approx(0.91)

    def test_it_reaches_the_graph(self):
        graph = annotate_graph(_graph(), _record(objects=[
            {"cls": "laptop", "bbox": (110, 150, 40, 40),
             "confidence": 0.33}]))
        assert graph["nodes"][0]["features"]["object_confidence"] == (
            pytest.approx(0.33))

    def test_a_detection_with_no_confidence_is_null_not_zero(self):
        """Null is "not recorded"; zero would read as "certainly not there"
        and would be drawn as the weakest possible link rather than as an
        unknown one."""
        a = classify(PERSON, [
            {"cls": "laptop", "bbox": (110, 150, 40, 40)},
        ], "teacher", 0.0, 0.3)
        assert a.obj_confidence is None


class TestWeakObjectLinksAreNotReported:
    """tools.report keeps only links that held for long enough."""

    def test_a_single_frame_link_is_dropped(self):
        from tools.report import _durable_objects

        kept = _durable_objects(Counter({(1, "cup"): 1}), Counter({1: 400}))
        assert kept == {}

    def test_a_sustained_link_is_kept(self):
        from tools.report import _durable_objects

        kept = _durable_objects(Counter({(1, "laptop"): 300}),
                                Counter({1: 400}))
        assert kept == {(1, "laptop"): 300}

    def test_a_short_clip_still_needs_an_absolute_floor(self):
        """On a five-frame clip, 10% is half a frame. Without the floor a
        single detection would clear the share test and be reported as
        something the student used."""
        from tools.report import _durable_objects

        kept = _durable_objects(Counter({(1, "book"): 1}), Counter({1: 5}))
        assert kept == {}

    def test_the_floor_is_per_student_not_per_session(self):
        """A student seen in 20 frames and one seen in 400 are judged against
        their own exposure. Otherwise someone who walked in at the end could
        never be credited with anything they were holding."""
        from tools.report import _durable_objects

        kept = _durable_objects(
            Counter({(1, "book"): 8, (2, "book"): 8}),
            Counter({1: 20, 2: 400}))
        assert kept == {(1, "book"): 8}
