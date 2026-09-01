"""Tests for action derivation (backend/actions.py).

These pin the decision rules and, more importantly, the two judgement calls the
module makes deliberately: that a missing face yields ``unknown`` rather than
``attentive``, and that reading and writing are not guessed apart.
"""

from __future__ import annotations

import pytest

from backend.actions import OFF_TASK, Action, annotate_graph, classify

PERSON = (100, 100, 80, 160)
#: A plain upright posture, so each test varies only the part it is about.
UPRIGHT = {"shoulder_mid": (140, 160), "nose": (140, 130)}


def _obj(cls, bbox):
    return {"cls": cls, "bbox": bbox}


class TestRules:
    def test_phone_overlap_wins(self):
        a = classify(PERSON, [_obj("cell phone", (120, 180, 20, 30))], "down", 25.0, 0.3)
        assert a.name == "on_phone"
        assert a.off_task is True

    def test_phone_beats_a_book(self):
        """A visible phone is stronger evidence than a book on the same desk."""
        objects = [_obj("book", (110, 200, 40, 30)), _obj("cell phone", (120, 180, 20, 30))]
        assert classify(PERSON, objects, "down", 25.0, 0.3).name == "on_phone"

    def test_book_reads_as_studying(self):
        a = classify(PERSON, [_obj("book", (110, 200, 40, 30))], "down", 25.0, 0.3)
        assert a.name == "studying"
        assert a.off_task is False

    def test_reading_and_writing_split_only_on_hand_evidence(self):
        """They differ only by what the hands are doing. With wrists visible
        the geometry is worth reporting -- marked inferred, because the closest
        published work reaches 57.8% on writing even with a strong temporal
        model. With no hands visible the honest answer is that it is one or the
        other."""
        book = [_obj("book", (110, 250, 50, 30))]

        writing = classify(PERSON, book, "down", 25.0, 0.3,
                           posture={**UPRIGHT, "left_wrist": (130, 220),
                                    "right_wrist": (150, 225)})
        assert writing.name == "writing"
        assert writing.confidence == "inferred"

        reading = classify(PERSON, book, "down", 25.0, 0.3, posture=UPRIGHT)
        assert reading.name == "reading"
        assert reading.confidence == "inferred"

        blind = classify(PERSON, book, "down", 25.0, 0.3, posture=None)
        assert blind.name == "studying"
        assert blind.confidence == "direct"

    def test_laptop(self):
        assert classify(PERSON, [_obj("laptop", (110, 200, 60, 40))], "teacher",
                        0.0, 0.3).name == "on_laptop"

    def test_closed_eyes_need_to_stay_closed(self):
        """A blink is not a closure.

        Human blinks run 100-400 ms. Scoring any low-EAR frame as eyes closed
        made 12.5% of a real session off task purely from blinking, so the
        closure has to persist past BLINK_MS.
        """
        from backend.actions import BLINK_MS

        shut = classify(PERSON, [], "teacher", 0.0, 0.05,
                        eyes_closed_ms=BLINK_MS + 100)
        assert shut.name == "eyes_closed"
        assert shut.off_task is True

        blink = classify(PERSON, [], "teacher", 0.0, 0.05, eyes_closed_ms=200)
        assert blink.name != "eyes_closed"

    def test_closed_eyes_without_a_duration_are_not_claimed(self):
        """With no duration we cannot tell a blink from a closure, and a blink
        wrongly scored off task is worse than a closure missed."""
        a = classify(PERSON, [], "teacher", 0.0, 0.05)
        assert a.name != "eyes_closed"

    @pytest.mark.parametrize("gaze", ["left", "right", "back"])
    def test_gaze_away_is_off_task(self, gaze):
        a = classify(PERSON, [], gaze, 0.0, 0.3)
        assert a.name == "looking_away"
        assert a.off_task is True

    def test_head_down_without_an_object(self):
        a = classify(PERSON, [], "down", 30.0, 0.3)
        assert a.name == "head_down"
        assert a.off_task is False

    def test_looking_at_the_front(self):
        assert classify(PERSON, [], "teacher", 0.0, 0.3).name == "attentive"

    def test_a_distant_object_is_not_theirs(self):
        """An object elsewhere in the room must not be attributed to a student
        just because it is in the same frame."""
        far = [_obj("cell phone", (900, 900, 20, 30))]
        assert classify(PERSON, far, "teacher", 0.0, 0.3).name == "attentive"

    def test_no_face_is_unknown_not_attentive(self):
        """The whole point: absence of evidence is not evidence of attention."""
        a = classify(PERSON, [], None, None, None)
        assert a.name == "unknown"
        assert a.off_task is False
        assert "unknown" not in OFF_TASK

    def test_action_carries_its_evidence(self):
        a = classify(PERSON, [_obj("cell phone", (120, 180, 20, 30))], "down", 25.0, 0.3)
        assert "cell phone" in a.evidence

    def test_label_is_human_readable(self):
        assert Action("on_phone", "x", True).label == "on phone"


class TestHands:
    """The three hand states are the ones most easily confused with each
    other, and the most opposite in meaning."""


    def test_raised_hand(self):
        a = classify(PERSON, [], "teacher", 0.0, 0.3,
                     posture={**UPRIGHT, "left_wrist": (150, 80)})
        assert a.name == "raising_hand"
        assert a.off_task is False

    def test_head_propped_on_hand_is_not_a_raised_hand(self):
        """A propped head also puts the wrist above the shoulders, so a
        shoulder-height test collapsed a bored student and a participating one
        into the same label."""
        a = classify(PERSON, [], "teacher", 0.0, 0.3,
                     posture={**UPRIGHT, "left_wrist": (145, 135)})
        assert a.name == "head_on_hand"

    def test_hand_on_the_desk_is_neither(self):
        a = classify(PERSON, [], "teacher", 0.0, 0.3,
                     posture={**UPRIGHT, "left_wrist": (130, 220)})
        assert a.name == "attentive"


class TestObjectActions:

    def test_bottle_at_head_height_is_drinking(self):
        a = classify(PERSON, [_obj("bottle", (130, 105, 18, 40))], "teacher",
                     0.0, 0.3, posture=UPRIGHT)
        assert a.name == "drinking"
        assert a.obj == "bottle"

    def test_bottle_on_the_desk_is_not_drinking(self):
        """Region matters: the same object low down means nothing."""
        a = classify(PERSON, [_obj("bottle", (130, 270, 18, 40))], "teacher",
                     0.0, 0.3, posture=UPRIGHT)
        assert a.name != "drinking"

    def test_food_at_head_height_is_eating(self):
        a = classify(PERSON, [_obj("apple", (130, 105, 18, 20))], "teacher",
                     0.0, 0.3, posture=UPRIGHT)
        assert a.name == "eating"

    def test_keyboard_is_typing(self):
        a = classify(PERSON, [_obj("keyboard", (110, 250, 60, 20))], "down",
                     25.0, 0.3, posture=UPRIGHT)
        assert a.name == "typing"


class TestPostureActions:

    def test_slouching(self):
        a = classify(PERSON, [], "teacher", 0.0, 0.3,
                     posture={**UPRIGHT, "vertical_lean": -0.02})
        assert a.name == "slouching"
        assert a.confidence == "inferred"

    def test_leaning_forward(self):
        a = classify(PERSON, [], "teacher", 0.0, 0.3,
                     posture={**UPRIGHT, "vertical_lean": -0.40})
        assert a.name == "leaning_forward"


class TestYawn:
    def test_wide_mouth_reads_as_a_yawn(self):
        from backend.actions import YAWN_RATIO

        marks = [(0.0, 0.0)] * 400
        marks[13] = (100.0, 100.0)
        marks[14] = (100.0, 100.0 + 40 * (YAWN_RATIO + 0.2))
        marks[78] = (80.0, 110.0)
        marks[308] = (120.0, 110.0)
        a = classify(PERSON, [], "teacher", 0.0, 0.3, landmarks=marks)
        assert a.name == "yawning"

    def test_speaking_is_not_a_yawn(self):
        """The threshold sits clear of speech on purpose -- reading talking as
        yawning would turn classroom discussion into disengagement."""
        marks = [(0.0, 0.0)] * 400
        marks[13] = (100.0, 100.0)
        marks[14] = (100.0, 112.0)
        marks[78] = (80.0, 110.0)
        marks[308] = (120.0, 110.0)
        a = classify(PERSON, [], "teacher", 0.0, 0.3, landmarks=marks)
        assert a.name != "yawning"

    def test_missing_landmarks_are_safe(self):
        from backend.actions import mouth_open_ratio

        assert mouth_open_ratio(None) is None
        assert mouth_open_ratio([(0.0, 0.0)] * 10) is None


class TestGraphAnnotation:
    @staticmethod
    def _record(objects=(), pitch=0.0, ear=0.3):
        return {
            "frame_id": 0,
            "timestamp_ms": 0,
            "objects": list(objects),
            "persons": [
                {"person_id": 1, "bbox": (100, 100, 80, 160),
                 "head_pose": {"pitch": pitch}, "face": {"ear": ear}},
                {"person_id": 2, "bbox": (300, 100, 80, 160),
                 "head_pose": {"pitch": pitch}, "face": {"ear": ear}},
            ],
        }

    @staticmethod
    def _graph(gaze="teacher"):
        return {
            "frame_id": 0, "timestamp_ms": 0, "edges": [],
            "nodes": [
                {"id": 1, "person_id": 1, "role": "student",
                 "features": {"gaze_label": gaze}},
                {"id": 2, "person_id": 2, "role": "student",
                 "features": {"gaze_label": gaze}},
            ],
        }

    def test_every_node_gets_an_action(self):
        graph = annotate_graph(self._graph(), self._record())
        for node in graph["nodes"]:
            assert node["features"]["action"]
            assert node["features"]["action_evidence"]

    def test_same_action_creates_a_shared_action_edge(self):
        """Two students doing the same thing is the relation worth drawing."""
        graph = annotate_graph(self._graph("left"), self._record())
        shared = [e for e in graph["edges"] if e["type"] == "shared_action"]
        assert len(shared) == 1
        assert shared[0]["features"]["action"] == "looking_away"

    def test_attentive_does_not_create_an_edge(self):
        """Everyone watching the front is the default state, not a relation --
        linking all of them would bury the edges that mean something."""
        graph = annotate_graph(self._graph("teacher"), self._record())
        assert not [e for e in graph["edges"] if e["type"] == "shared_action"]

    def test_unknown_does_not_create_an_edge(self):
        graph = annotate_graph(self._graph(None), self._record())
        assert not [e for e in graph["edges"] if e["type"] == "shared_action"]

    def test_different_actions_do_not_link(self):
        graph = self._graph("left")
        graph["nodes"][1]["features"]["gaze_label"] = "teacher"
        graph = annotate_graph(graph, self._record())
        assert not [e for e in graph["edges"] if e["type"] == "shared_action"]

    def test_a_node_without_a_person_is_skipped(self):
        graph = self._graph()
        graph["nodes"].append({"id": 9, "person_id": 99, "role": "student",
                               "features": {"gaze_label": "left"}})
        annotated = annotate_graph(graph, self._record())
        assert "action" not in annotated["nodes"][2]["features"]

    def test_existing_edges_are_preserved(self):
        graph = self._graph("left")
        graph["edges"].append({"type": "spatial_adjacency", "source": 1,
                               "target": 2, "features": {}})
        annotated = annotate_graph(graph, self._record())
        kinds = {e["type"] for e in annotated["edges"]}
        assert kinds == {"spatial_adjacency", "shared_action"}


class TestOnTaskPercentage:
    def test_off_task_actions_lower_the_score(self):
        from backend.student_profile import _on_task_pct

        assert _on_task_pct(["attentive"] * 8 + ["on_phone"] * 2) == 80.0

    def test_unknown_frames_are_excluded_not_counted_as_good(self):
        """Frames where nothing was read must not inflate the score."""
        from backend.student_profile import _on_task_pct

        assert _on_task_pct(["attentive", "on_phone", "unknown", None]) == 50.0

    def test_nothing_graded_is_none_not_zero(self):
        from backend.student_profile import _on_task_pct

        assert _on_task_pct(["unknown", None]) is None
