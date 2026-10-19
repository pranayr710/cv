"""Tests for erasing one person from every store at once.

A deletion request is one request. The data lives in three places that knew
nothing about each other, so deleting through any one of them left the other
two intact and reported success. Each test below is one of those gaps.

The hardest property to get right is not the deletion -- it is that erasing
one student must not disturb anyone else's record. Dropping whole frames would
change the timeline every other student's windows are measured against, so
erasing one child would silently alter everybody else's scores.
"""
import json

import numpy as np
import pytest

from backend.enrollment import EnrolledGallery
from backend.erasure import erase
from backend.store import (
    add_windows,
    connect,
    link_person,
    record_session,
    upsert_student,
)


@pytest.fixture
def world(tmp_path):
    """Two enrolled students, present in all three stores."""
    gallery_path = tmp_path / "gallery.json"
    db_path = tmp_path / "cg.db"
    folder = tmp_path / "session"
    folder.mkdir()

    gallery = EnrolledGallery()
    rng = np.random.default_rng(0)
    ids = {}
    for name in ("Asha", "Ben"):
        vec = rng.normal(size=512).astype(np.float32)
        vec /= np.linalg.norm(vec)
        ids[name] = gallery.register(name, [vec, vec, vec]).person_id
    gallery.save(gallery_path)
    a, b = ids["Asha"], ids["Ben"]

    with (folder / "live_graph.jsonl").open("w", encoding="utf-8") as handle:
        for i in range(5):
            handle.write(json.dumps({
                "frame_id": i, "timestamp_ms": i * 333,
                "nodes": [
                    {"person_id": a, "features": {"action": "on_phone"}},
                    {"person_id": b, "features": {"action": "attentive"}},
                ],
                "edges": [{"source": a, "target": b, "kind": "near"}],
            }) + "\n")
    (folder / "live_profiles.json").write_text(json.dumps([
        {"person_id": a, "on_task_pct": 10.0},
        {"person_id": b, "on_task_pct": 90.0},
    ]), encoding="utf-8")
    (folder / "report.html").write_text("<h3>Asha</h3>", encoding="utf-8")

    with connect(db_path) as conn:
        session = record_session(conn, source="test")
        for local, name in ((a, "Asha"), (b, "Ben")):
            student = upsert_student(conn, session, local, attention_score=4)
            add_windows(conn, student,
                        [{"start_ms": 0, "end_ms": 15000, "score_10": 4}])
            link_person(conn, student, name)

    return {"gallery": gallery_path, "db": db_path, "folder": folder,
            "a": a, "b": b}


def _graph(folder):
    return [json.loads(line) for line
            in (folder / "live_graph.jsonl").read_text(
                encoding="utf-8").splitlines() if line.strip()]


def _ids_in_graph(folder):
    return {n["person_id"] for row in _graph(folder) for n in row["nodes"]}


class TestErasureReachesEveryStore:
    def test_the_face_record_is_destroyed(self, world):
        erase("Asha", gallery_path=world["gallery"], db_path=world["db"],
              output_dirs=[world["folder"]])
        names = [p.name for p in EnrolledGallery.load(world["gallery"]).people]
        assert names == ["Ben"]

    def test_the_database_rows_are_destroyed(self, world):
        erase("Asha", gallery_path=world["gallery"], db_path=world["db"],
              output_dirs=[world["folder"]])
        with connect(world["db"]) as conn:
            people = [r["display_name"] for r in
                      conn.execute("SELECT display_name FROM person")]
            windows = conn.execute(
                "SELECT COUNT(*) FROM window").fetchone()[0]
        assert people == ["Ben"]
        assert windows == 1

    def test_per_frame_records_are_destroyed(self, world):
        """A bounding box tied to a stable id is personal data without a name.

        This is the store the dashboard's delete never touched.
        """
        erase("Asha", gallery_path=world["gallery"], db_path=world["db"],
              output_dirs=[world["folder"]])
        assert world["a"] not in _ids_in_graph(world["folder"])

    def test_edges_naming_the_person_are_destroyed(self, world):
        """An edge is a record of who they sat near."""
        erase("Asha", gallery_path=world["gallery"], db_path=world["db"],
              output_dirs=[world["folder"]])
        edges = [e for row in _graph(world["folder"])
                 for e in row.get("edges", [])]
        assert not any(e["source"] == world["a"] or e["target"] == world["a"]
                       for e in edges)

    def test_the_derived_profile_is_destroyed(self, world):
        erase("Asha", gallery_path=world["gallery"], db_path=world["db"],
              output_dirs=[world["folder"]])
        profiles = json.loads((world["folder"] / "live_profiles.json")
                              .read_text(encoding="utf-8"))
        assert [p["person_id"] for p in profiles] == [world["b"]]

    def test_the_rendered_report_is_removed(self, world):
        """It has the name written into it and is rebuilt from the rest."""
        erase("Asha", gallery_path=world["gallery"], db_path=world["db"],
              output_dirs=[world["folder"]])
        assert not (world["folder"] / "report.html").exists()


class TestEveryoneElseIsUndisturbed:
    def test_the_other_student_survives_intact(self, world):
        erase("Asha", gallery_path=world["gallery"], db_path=world["db"],
              output_dirs=[world["folder"]])
        assert world["b"] in _ids_in_graph(world["folder"])

    def test_the_timeline_is_not_shortened(self, world):
        """Frames are kept even when they only held the erased student.

        Dropping them would change the span every other student's windows are
        measured against, so erasing one child would silently alter everyone
        else's scores -- a correctness bug produced by a privacy feature.
        """
        before = len(_graph(world["folder"]))
        erase("Asha", gallery_path=world["gallery"], db_path=world["db"],
              output_dirs=[world["folder"]])
        assert len(_graph(world["folder"])) == before

    def test_frame_timestamps_are_preserved(self, world):
        before = [r["timestamp_ms"] for r in _graph(world["folder"])]
        erase("Asha", gallery_path=world["gallery"], db_path=world["db"],
              output_dirs=[world["folder"]])
        assert [r["timestamp_ms"] for r in _graph(world["folder"])] == before


class TestReporting:
    def test_a_dry_run_changes_nothing(self, world):
        report = erase("Asha", gallery_path=world["gallery"],
                       db_path=world["db"], output_dirs=[world["folder"]],
                       dry_run=True)
        assert report.anything_found
        assert world["a"] in _ids_in_graph(world["folder"])
        names = [p.name for p in EnrolledGallery.load(world["gallery"]).people]
        assert "Asha" in names

    def test_an_unknown_name_reports_nothing_found(self, world):
        report = erase("Nobody", gallery_path=world["gallery"],
                       db_path=world["db"], output_dirs=[world["folder"]])
        assert not report.anything_found

    def test_copies_elsewhere_are_named_as_unreachable(self, world):
        """A deletion that silently missed a copy is worse than one that says
        where it stopped."""
        report = erase("Asha", gallery_path=world["gallery"],
                       db_path=world["db"], output_dirs=[world["folder"]])
        assert any("copy outside" in item for item in report.unreachable)

    def test_matching_is_case_insensitive_in_every_store(self, world):
        """Asserting only the gallery let a real bug through.

        The gallery matched case-insensitively and the database did not, so
        erase("asha") destroyed the face record, reported success, and left
        every attention score in place. A test that checks one store endorses
        that.
        """
        report = erase("asha", gallery_path=world["gallery"],
                       db_path=world["db"], output_dirs=[world["folder"]])
        assert report.gallery_removed
        assert report.db_sessions == 1
        with connect(world["db"]) as conn:
            remaining = [r["display_name"] for r in
                         conn.execute("SELECT display_name FROM person")]
        assert remaining == ["Ben"]
        assert world["a"] not in _ids_in_graph(world["folder"])

    def test_a_student_never_linked_to_a_person_is_still_erased(self, world,
                                                                tmp_path):
        """Ingest leaves person_id NULL; linking is a separate, optional step.

        forget_person only reaches students that have one, so a student who
        was ingested but never linked survived a deletion request entirely --
        which is the default state of every ingested session.
        """
        from backend.store import record_session, upsert_student

        with connect(world["db"]) as conn:
            session = record_session(conn, source="unlinked")
            student = upsert_student(conn, session, world["a"],
                                     attention_score=2)
            add_windows(conn, student,
                        [{"start_ms": 0, "end_ms": 15000, "score_10": 2}])
            before = conn.execute(
                "SELECT COUNT(*) FROM student WHERE person_id IS NULL"
            ).fetchone()[0]
        assert before == 1

        erase("Asha", gallery_path=world["gallery"], db_path=world["db"],
              output_dirs=[world["folder"]])

        with connect(world["db"]) as conn:
            after = conn.execute(
                "SELECT COUNT(*) FROM student WHERE person_id IS NULL"
            ).fetchone()[0]
            orphan_windows = conn.execute(
                "SELECT COUNT(*) FROM window w LEFT JOIN student s "
                "ON s.id = w.student_id WHERE s.id IS NULL").fetchone()[0]
        assert after == 0
        assert orphan_windows == 0
