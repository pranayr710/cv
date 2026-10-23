"""Uploaded recordings do not accumulate, and erasure admits it cannot
filter them.

Every video analysed through the dashboard was kept forever, beside its
results. Two days of testing left 20 copies of 14 recordings -- 219 MB, and
far more to the point, 219 MB of classroom footage of children with no
retention policy, in a project that has a retention sweep for the face
gallery and a single erasure path for everything else.

The second half is the sharper one. ``backend.erasure`` reached the graph,
the profiles, the database and the report, and never looked at the video
those were all derived from. A deletion request came back reporting success
having scrubbed the derived rows while leaving the child's face, in full, in
a file whose path the module could have printed.
"""
import json

import pytest

from backend.erasure import _recordings_in, erase
from tools.run.server import KEEP_RECENT_UPLOADS, prune_uploads


def _upload(uploads, name, when):
    """One analysed recording, staged the way VideoJob stages it."""
    folder = uploads / name
    folder.mkdir(parents=True)
    video = folder / f"{name}.mp4"
    video.write_bytes(b"\x00" * 64)
    import os
    os.utime(folder, (when, when))
    return folder


class TestRecordingsAreNotKeptForever:
    def test_only_the_most_recent_survive(self, tmp_path):
        uploads = tmp_path / "uploads"
        for i in range(8):
            _upload(uploads, f"run{i}", 1_000_000 + i * 60)
        removed = prune_uploads(uploads)
        assert removed == 8 - KEEP_RECENT_UPLOADS
        assert len(list(uploads.iterdir())) == KEEP_RECENT_UPLOADS

    def test_the_newest_is_the_one_kept(self, tmp_path):
        uploads = tmp_path / "uploads"
        for i in range(5):
            _upload(uploads, f"run{i}", 1_000_000 + i * 60)
        prune_uploads(uploads, keep=1)
        assert [p.name for p in uploads.iterdir()] == ["run4"]

    def test_nothing_is_removed_below_the_limit(self, tmp_path):
        uploads = tmp_path / "uploads"
        _upload(uploads, "only", 1_000_000)
        assert prune_uploads(uploads) == 0
        assert (uploads / "only").exists()

    def test_some_are_kept_on_purpose(self):
        """Not zero: tools/run/export_faces_to_enrol.py needs the source video to
        pull face crops, and that is the obvious next step after a run."""
        assert KEEP_RECENT_UPLOADS >= 1

    def test_a_bare_file_is_handled_as_well_as_a_folder(self, tmp_path):
        """An upload that failed before it was staged leaves a loose file."""
        import os

        uploads = tmp_path / "uploads"
        uploads.mkdir()
        stray = uploads / "stray.mp4"
        stray.write_bytes(b"\x00")
        # Explicitly aged, or it carries the current mtime, is genuinely the
        # most recent entry, and is correctly kept -- which is how this
        # failed the first time, for a reason other than the one it exists
        # to catch.
        os.utime(stray, (1_000_000, 1_000_000))
        _upload(uploads, "run", 2_000_000)
        prune_uploads(uploads, keep=1)
        assert not stray.exists()
        assert (uploads / "run").exists()

    def test_a_missing_directory_is_not_an_error(self, tmp_path):
        assert prune_uploads(tmp_path / "nothing-here") == 0

    def test_an_undeletable_entry_does_not_stop_the_analysis(self, tmp_path,
                                                             monkeypatch):
        """The next run matters more than the last one's housekeeping."""
        import shutil

        uploads = tmp_path / "uploads"
        for i in range(5):
            _upload(uploads, f"run{i}", 1_000_000 + i * 60)

        def refuse(path):
            raise OSError("in use by another process")

        monkeypatch.setattr(shutil, "rmtree", refuse)
        assert prune_uploads(uploads) == 0  # reported, not raised


class TestErasureNamesTheRecording:
    @pytest.fixture
    def folder(self, tmp_path):
        out = tmp_path / "session"
        (out / "uploads" / "1700000000").mkdir(parents=True)
        (out / "uploads" / "1700000000" / "1700000000.mp4").write_bytes(b"\x00")
        return out

    def test_the_source_video_is_found(self, folder):
        found = _recordings_in(folder)
        assert len(found) == 1
        assert "1700000000.mp4" in found[0]

    def test_a_folder_without_uploads_reports_nothing(self, tmp_path):
        assert _recordings_in(tmp_path) == []

    def test_a_non_video_is_not_reported(self, folder):
        (folder / "uploads" / "notes.txt").write_text("x", encoding="utf-8")
        assert all("notes.txt" not in item for item in _recordings_in(folder))

    def test_an_erasure_report_names_it(self, tmp_path, folder):
        """The whole point. This came back saying success while the child's
        face sat in a video file beside the results it had just scrubbed."""
        from backend.enrollment import EnrolledGallery
        import numpy as np

        gallery_path = tmp_path / "gallery.json"
        gallery = EnrolledGallery()
        rng = np.random.default_rng(0)
        vec = rng.normal(size=512).astype(np.float32)
        vec /= np.linalg.norm(vec)
        pid = gallery.register("Asha", [vec, vec, vec]).person_id
        gallery.save(gallery_path)

        (folder / "live_graph.jsonl").write_text(json.dumps({
            "frame_id": 0, "timestamp_ms": 0,
            "nodes": [{"person_id": pid, "features": {}}], "edges": [],
        }) + "\n", encoding="utf-8")

        report = erase("Asha", gallery_path=gallery_path,
                       db_path=tmp_path / "none.db", output_dirs=[folder])
        assert any("recording" in item for item in report.unreachable)

    def test_the_recording_is_not_deleted(self, tmp_path, folder):
        """Deleting it would destroy every other student's source. Naming it
        leaves that decision with the operator, which is whose it is."""
        from backend.enrollment import EnrolledGallery
        import numpy as np

        gallery_path = tmp_path / "gallery.json"
        gallery = EnrolledGallery()
        rng = np.random.default_rng(0)
        vec = rng.normal(size=512).astype(np.float32)
        vec /= np.linalg.norm(vec)
        gallery.register("Asha", [vec, vec, vec])
        gallery.save(gallery_path)

        video = folder / "uploads" / "1700000000" / "1700000000.mp4"
        erase("Asha", gallery_path=gallery_path,
              db_path=tmp_path / "none.db", output_dirs=[folder])
        assert video.exists()
