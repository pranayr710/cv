"""What the dashboard is sent each frame.

Everything on the page is drawn from this one dict, so a field that is
measured but never put in it is invisible — and three were. The lean was
computed on 334 students in one recording and reached the page for 2 of them;
the object each student was judged to be holding was decided exclusively by
the pipeline and then not sent, so the graph drew every student joined to
every object; and the Mamba model's score existed only in the training
script.

These tests pin the payload rather than the drawing. A renderer can be
rewritten; a field that stops being sent takes a panel down silently.
"""
from collections import Counter, defaultdict, deque

import pytest

from tools.run.server import REPORTABLE_OBJECT_CONF, Session


@pytest.fixture
def session():
    s = object.__new__(Session)
    s.min_face_px = 60
    s.gaze = defaultdict(Counter)
    s.gaze[1]["teacher"] = 5
    s.expr = defaultdict(Counter)
    s.actions = defaultdict(Counter)
    s.recent = defaultdict(lambda: deque(maxlen=20))
    s.pairs = Counter()
    s.last_seen = {1: 10}
    s.fps = 5.0
    return s


def _record():
    return {
        "frame_id": 10, "timestamp_ms": 2000,
        "persons": [{"person_id": 1, "bbox": [100, 80, 60, 160],
                     "posture": {"facing_direction": [0.0, -1.0]},
                     "face": {"bbox": [115, 90, 30, 30]}}],
        "objects": [
            {"cls": "laptop", "bbox": [110, 150, 40, 40], "confidence": 0.83},
            {"cls": "bottle", "bbox": [90, 200, 10, 20], "confidence": 0.28},
        ],
    }


def _graph(**overrides):
    features = {
        "gaze_label": "teacher", "action": "on_laptop",
        "action_evidence": "laptop overlap", "object": "laptop",
        "object_confidence": 0.83, "lean_state": "slouching",
        "vertical_lean": 0.02, "oriented": True, "attention_score": 7,
        "attention_score_mamba": 6, "attention_label": "on",
        "attention_reason": "because", "attention_source": "model",
        "expression": "neutral",
    }
    features.update(overrides)
    return {"timestamp_ms": 2000,
            "nodes": [{"id": 1, "person_id": 1, "role": "student",
                       "features": features}]}


def _snap(session, graph=None):
    return session._snapshot(_record(), graph or _graph(), {1: "Asha"}, 2.0)


class TestBothModelScoresAreSent:
    def test_the_aggregate_score_is_sent(self, session):
        assert _snap(session)["students"][0]["attention_score"] == 7

    def test_the_sequence_model_score_is_sent_beside_it(self, session):
        """Side by side, never averaged. Out of fold on 179 labelled windows
        the two scored 0.609 and 0.592 -- too close to pick a winner from, so
        a disagreement between them is information, not noise to hide."""
        assert _snap(session)["students"][0]["attention_score_mamba"] == 6

    def test_a_missing_sequence_score_is_null_not_zero(self, session):
        """Zero would paint an empty bar, which reads as "this student scored
        nothing" rather than "no full window yet"."""
        snap = _snap(session, _graph(attention_score_mamba=None))
        assert snap["students"][0]["attention_score_mamba"] is None


class TestPostureReachesThePage:
    def test_the_lean_is_sent_even_though_another_action_won(self, session):
        """The action here is on_laptop. The lean used to be dropped for
        losing that contest, although it had been measured on the same
        frame."""
        snap = _snap(session)
        assert snap["students"][0]["lean"] == "slouching"
        assert snap["boxes"][0]["lean"] == "slouching"

    def test_the_raw_value_travels_with_the_band(self, session):
        assert _snap(session)["boxes"][0]["lean_value"] == pytest.approx(0.02)


class TestObjectsCarryTheirConfidence:
    def test_each_object_is_sent_with_a_number(self, session):
        """A bare class name drew a 0.28 bottle exactly like a 0.91 laptop."""
        objects = _snap(session)["objects"]
        assert {o["cls"] for o in objects} == {"laptop", "bottle"}
        assert all("conf" in o for o in objects)

    def test_a_weak_detection_is_sent_rather_than_dropped(self, session):
        """Dropping it in the backend would hide that the detector fired at
        all. The page dims it instead, so the operator can see what the
        threshold is excluding."""
        objects = {o["cls"]: o["conf"] for o in _snap(session)["objects"]}
        assert objects["bottle"] < REPORTABLE_OBJECT_CONF
        assert objects["laptop"] >= REPORTABLE_OBJECT_CONF

    def test_the_student_carries_only_their_own_object(self, session):
        """Exclusive ownership, decided by backend.actions.assign_objects.
        The graph used to join every student to every object in the frame,
        which drew a book for people who never touched one."""
        student = _snap(session)["students"][0]
        assert student["object"] == "laptop"
        assert student["object_conf"] == pytest.approx(0.83)


class TestPositionsAreMeasuredNotInvented:
    def test_the_student_is_sent_where_they_were_seen(self, session):
        """The graph drew students on a four-column grid in roster order,
        which is a picture of the array rather than of the room."""
        student = _snap(session)["students"][0]
        assert student["cx"] == 130
        assert student["cy"] == 160

    def test_orientation_comes_from_the_pipeline(self, session):
        assert _snap(session)["students"][0]["oriented"] is True

    def test_an_unmeasured_orientation_stays_null(self, session):
        """Not having read someone's shoulders is not evidence they faced
        away, and the graph colours those three states differently."""
        snap = _snap(session, _graph(oriented=None))
        assert snap["students"][0]["oriented"] is None
