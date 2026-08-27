"""Tests for action derivation (backend/actions.py).

These pin the decision rules and, more importantly, the two judgement calls the
module makes deliberately: that a missing face yields ``unknown`` rather than
``attentive``, and that reading and writing are not guessed apart.
"""

from __future__ import annotations

import pytest

from backend.actions import OFF_TASK, Action, annotate_graph, classify

PERSON = (100, 100, 80, 160)


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

    def test_reading_and_writing_are_not_separated(self):
        """Both are a book plus a bowed head. The literature puts writing at
        57.8% with a strong temporal model, so guessing here would be inventing
        precision the evidence cannot support."""
        from backend.actions import LABELS

        assert "write" not in LABELS
        assert LABELS["studying"] == "reading / writing"

    def test_laptop(self):
        assert classify(PERSON, [_obj("laptop", (110, 200, 60, 40))], "teacher",
                        0.0, 0.3).name == "on_laptop"

    def test_closed_eyes(self):
        a = classify(PERSON, [], "teacher", 0.0, 0.05)
        assert a.name == "eyes_closed"
        assert a.off_task is True

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
