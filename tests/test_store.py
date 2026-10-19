"""Tests for the session database.

The tests that matter here are the ones protecting properties that fail
silently: a delete that leaves data behind, a non-student quietly entering the
class average, and a re-run overwriting an earlier measurement.
"""
import json

import pytest

from backend.store import (
    add_action_tally,
    add_windows,
    connect,
    forget_person,
    ingest_profiles,
    link_person,
    person_history,
    record_session,
    session_summary,
    upsert_student,
)


@pytest.fixture
def db(tmp_path):
    with connect(tmp_path / "t.db") as conn:
        yield conn


class TestForgetting:
    """A system that cannot delete someone should not be storing them."""

    def test_forgetting_a_person_removes_their_windows(self, db):
        """Foreign keys are off by default in SQLite.

        With them off this delete succeeds, reports success, and leaves every
        attention score behind -- the exact shape of a privacy failure nobody
        notices.
        """
        session = record_session(db, source="test")
        student = upsert_student(db, session, 1, attention_score=5)
        add_windows(db, student, [{"start_ms": 0, "end_ms": 15000,
                                   "score_10": 5}])
        add_action_tally(db, student, {"attentive": 10})
        person = link_person(db, student, "Asha")

        forget_person(db, person)

        assert db.execute("SELECT COUNT(*) FROM window").fetchone()[0] == 0
        assert db.execute(
            "SELECT COUNT(*) FROM action_tally").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM student").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM person").fetchone()[0] == 0

    def test_forgetting_one_person_leaves_the_others(self, db):
        session = record_session(db, source="test")
        a = link_person(db, upsert_student(db, session, 1), "Asha")
        link_person(db, upsert_student(db, session, 2), "Ben")

        forget_person(db, a)

        names = [r["display_name"] for r in
                 db.execute("SELECT display_name FROM person").fetchall()]
        assert names == ["Ben"]

    def test_the_session_survives_forgetting_a_student(self, db):
        """Deleting a person must not delete everyone else's recording."""
        session = record_session(db, source="test")
        person = link_person(db, upsert_student(db, session, 1), "Asha")
        forget_person(db, person)
        assert db.execute("SELECT COUNT(*) FROM session").fetchone()[0] == 1


class TestIngest:
    def _write(self, tmp_path, students):
        path = tmp_path / "live_profiles.json"
        path.write_text(json.dumps(students), encoding="utf-8")
        return path

    def test_unresolved_tracks_are_not_stored(self, db, tmp_path):
        """Negative ids are tracks the resolver could not turn into a person.

        Storing them would put noise into every class average computed
        afterwards, and they are not people to report on.
        """
        path = self._write(tmp_path, [
            {"person_id": 1, "on_task_pct": 50.0},
            {"person_id": -1, "on_task_pct": 100.0},
            {"person_id": -2, "on_task_pct": 100.0},
        ])
        session = record_session(db, source="test")
        assert ingest_profiles(db, session, path) == 1

    def test_the_teacher_is_not_counted_as_a_student(self, db, tmp_path):
        """One adult attending to nobody would skew a class mean."""
        path = self._write(tmp_path, [
            {"person_id": 1, "role": "student", "on_task_pct": 50.0},
            {"person_id": 2, "role": "teacher", "on_task_pct": 10.0},
        ])
        session = record_session(db, source="test")
        assert ingest_profiles(db, session, path) == 1

    def test_rejected_profiles_are_skipped(self, db, tmp_path):
        path = self._write(tmp_path, [
            {"person_id": 1, "on_task_pct": 50.0},
            {"person_id": 2, "rejected_reason": "face too small"},
        ])
        session = record_session(db, source="test")
        assert ingest_profiles(db, session, path) == 1

    def test_nested_action_counts_are_read(self, db, tmp_path):
        """The pipeline nests counts under a classified/unavailable summary."""
        path = self._write(tmp_path, [{
            "person_id": 1,
            "actions": {"classified": 30, "unavailable": 0,
                        "counts": {"on_phone": 20, "attentive": 10}},
        }])
        session = record_session(db, source="test")
        ingest_profiles(db, session, path)
        rows = dict(db.execute(
            "SELECT action, frames FROM action_tally").fetchall())
        assert rows == {"on_phone": 20, "attentive": 10}

    def test_reingesting_creates_a_second_session(self, db, tmp_path):
        """Two runs over the same footage are two measurements.

        Merging them would hide a change in the model between the runs, which
        is exactly what someone re-running the pipeline is usually checking.
        """
        path = self._write(tmp_path, [{"person_id": 1, "on_task_pct": 50.0}])
        for _ in range(2):
            ingest_profiles(db, record_session(db, source="test"), path)
        assert db.execute("SELECT COUNT(*) FROM session").fetchone()[0] == 2
        assert db.execute("SELECT COUNT(*) FROM student").fetchone()[0] == 2


class TestQueries:
    def test_history_spans_sessions(self, db):
        """The query the file-based output could not answer at all."""
        person = None
        for week, score in enumerate([3, 7, 9], start=1):
            session = record_session(db, source=f"week {week}",
                                     started_at=f"2026-01-0{week}T09:00:00")
            student = upsert_student(db, session, 1, attention_score=score)
            person = link_person(db, student, "Asha")

        history = person_history(db, person)
        assert [r["attention_score"] for r in history] == [3, 7, 9]

    def test_summary_puts_the_least_attentive_first(self, db):
        """A teacher opens this to find who needs attention, not who is fine."""
        session = record_session(db, source="test")
        for local, score in ((1, 9), (2, 2), (3, 6)):
            upsert_student(db, session, local, attention_score=score)
        assert [r["local_id"] for r in session_summary(db, session)] == [2, 3, 1]

    def test_students_without_a_score_sort_last(self, db):
        """Absent is not the same as low, and must not head the list."""
        session = record_session(db, source="test")
        upsert_student(db, session, 1, attention_score=None)
        upsert_student(db, session, 2, attention_score=4)
        assert [r["local_id"] for r in session_summary(db, session)] == [2, 1]

    def test_rewriting_a_window_does_not_duplicate_it(self, db):
        """Re-running the scorer over a session must not double its windows."""
        session = record_session(db, source="test")
        student = upsert_student(db, session, 1)
        window = {"start_ms": 0, "end_ms": 15000, "score_10": 4}
        add_windows(db, student, [window])
        add_windows(db, student, [{**window, "score_10": 7}])
        rows = db.execute("SELECT score_10 FROM window").fetchall()
        assert [r["score_10"] for r in rows] == [7]

    def test_updating_a_student_keeps_one_row(self, db):
        session = record_session(db, source="test")
        first = upsert_student(db, session, 1, attention_score=3)
        second = upsert_student(db, session, 1, attention_score=8)
        assert first == second
        assert db.execute(
            "SELECT attention_score FROM student").fetchone()[0] == 8
