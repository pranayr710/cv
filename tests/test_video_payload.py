"""What a finished video analysis sends the page.

A batch run used to send ``students: []`` on every payload including the
final one. The results panel listed five students at the same moment the
scene graph said nobody was in frame, because the graph is drawn from
``students`` and the results panel is fetched separately. Everything had been
computed; none of it was being sent, so the two halves of the screen
disagreed and the half the eye goes to was the wrong one.

Caught by watching the real WebSocket through a real run rather than by
reading the code, which is the only way this kind of gap shows up: no error
is raised, nothing fails, a panel is simply empty.
"""
import json

import pytest

from tools.run.server import VideoJob


@pytest.fixture
def job(tmp_path):
    j = object.__new__(VideoJob)
    j.out = tmp_path
    j._preview_scale = 1.0
    return j


def _row():
    """One graph row as pass 2 writes it."""
    return {
        "timestamp_ms": 9000,
        "nodes": [
            {"id": 1, "person_id": 1, "role": "student", "features": {
                "bbox": [100, 80, 60, 160], "action": "writing",
                "action_evidence": "book with hands low at the desk",
                "object": "book", "object_confidence": 0.82,
                "lean_state": "slouching", "vertical_lean": 0.02,
                "oriented": True, "gaze_label": "teacher"}},
            {"id": 2, "person_id": 2, "role": "student", "features": {
                "bbox": [300, 90, 60, 150], "action": "on_laptop",
                "object": "laptop", "object_confidence": 0.91,
                "lean_state": "upright", "vertical_lean": -0.14,
                "oriented": False, "gaze_label": "left"}},
            # A track the resolver could not place. Must not reach the page.
            {"id": 3, "person_id": -1, "role": "student", "features": {
                "bbox": [500, 90, 50, 140], "action": "unknown"}},
        ],
    }


def _profiles(path):
    path.write_text(json.dumps([
        {"person_id": 1, "attention_score": 4, "attention_score_mamba": 7,
         "attention_windows": 45},
        {"person_id": 2, "attention_score": 9, "attention_score_mamba": 9,
         "attention_windows": 43},
    ]), encoding="utf-8")


class TestTheRosterIsSent:
    def test_students_are_not_empty(self, job, tmp_path):
        _profiles(tmp_path / "live_profiles.json")
        students, _ = job._resolved_students(_row())
        assert [s["id"] for s in students] == [1, 2]

    def test_both_model_scores_travel(self, job, tmp_path):
        _profiles(tmp_path / "live_profiles.json")
        students, _ = job._resolved_students(_row())
        assert students[0]["attention_score"] == 4
        assert students[0]["attention_score_mamba"] == 7

    def test_a_student_is_marked_present(self, job, tmp_path):
        """The graph filters on this. Without it the roster arrives full and
        the graph still draws nothing, which is the same bug one layer up."""
        _profiles(tmp_path / "live_profiles.json")
        students, _ = job._resolved_students(_row())
        assert all(s["present"] for s in students)

    def test_positions_come_from_the_frame(self, job, tmp_path):
        _profiles(tmp_path / "live_profiles.json")
        students, _ = job._resolved_students(_row())
        assert students[0]["cx"] == 130
        assert students[0]["cy"] == 160

    def test_an_unresolved_track_is_not_a_student(self, job, tmp_path):
        _profiles(tmp_path / "live_profiles.json")
        students, _ = job._resolved_students(_row())
        assert all(s["id"] > 0 for s in students)

    def test_every_key_the_page_reads_is_present(self, job, tmp_path):
        """A missing key throws in the handler before it repaints, leaving
        the previous run's picture on screen. Absent data has to be present
        and empty."""
        _profiles(tmp_path / "live_profiles.json")
        students, _ = job._resolved_students(_row())
        for key in ("id", "name", "present", "attention_score",
                    "attention_score_mamba", "action", "lean", "object",
                    "object_conf", "oriented", "cx", "cy", "gaze",
                    "expression", "actions", "recent"):
            assert key in students[0], key

    def test_missing_profiles_still_yield_a_roster(self, job):
        """Scores are absent; who was in frame is not. The graph should still
        draw the seating map."""
        students, _ = job._resolved_students(_row())
        assert len(students) == 2
        assert students[0]["attention_score"] is None


class TestObjectsAreSent:
    def test_each_attributed_object_is_reported(self, job, tmp_path):
        _profiles(tmp_path / "live_profiles.json")
        _, objects = job._resolved_students(_row())
        assert {o["cls"] for o in objects} == {"book", "laptop"}

    def test_confidence_travels_with_the_class(self, job, tmp_path):
        """A 0.26 book and a 0.91 laptop were drawn identically."""
        _profiles(tmp_path / "live_profiles.json")
        _, objects = job._resolved_students(_row())
        by_class = {o["cls"]: o["conf"] for o in objects}
        assert by_class["laptop"] == pytest.approx(0.91)

    def test_a_student_holding_nothing_adds_no_object(self, job, tmp_path):
        row = _row()
        row["nodes"][0]["features"]["object"] = None
        _, objects = job._resolved_students(row)
        assert [o["cls"] for o in objects] == ["laptop"]
