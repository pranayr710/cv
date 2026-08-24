"""Tests for registered-face identity (backend/enrollment.py).

Embeddings here are synthetic unit vectors, not real faces. That is deliberate:
these tests pin the *bookkeeping* -- id stability, persistence, the consent
affordances, and the cannot-link interaction -- none of which depend on how a
face is embedded. Recognition quality is a measurement question answered on
real footage, not something a unit test can assert.
"""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from backend.config import CONFIG
from backend.enrollment import (
    GALLERY_FORMAT_VERSION,
    EnrolledGallery,
    EnrolledIdentityResolver,
)

DIM = 512


def _unit(vec: np.ndarray) -> np.ndarray:
    return vec / np.linalg.norm(vec)


def _person(seed: int) -> np.ndarray:
    """A distinct synthetic identity."""
    return _unit(np.random.default_rng(seed).standard_normal(DIM).astype(np.float32))


def _shots(base: np.ndarray, n: int = 6, sigma: float = 0.0331, seed: int = 0):
    """``n`` noisy observations of one identity.

    sigma is calibrated to this project's measured same-person cosine
    similarity (median ~0.80 on real classroom faces).
    """
    rng = np.random.default_rng(seed)
    return [_unit(base + sigma * rng.standard_normal(DIM).astype(np.float32)) for _ in range(n)]


class TestGallery:
    def test_registering_returns_sequential_ids(self):
        gallery = EnrolledGallery()
        first = gallery.register("asha", _shots(_person(1)))
        second = gallery.register("bilal", _shots(_person(2)))
        assert (first.person_id, second.person_id) == (1, 2)

    def test_reregistering_keeps_the_same_id(self):
        """Adding better enrollment shots later must not renumber anyone --
        otherwise previously produced output silently changes meaning."""
        gallery = EnrolledGallery()
        original = gallery.register("asha", _shots(_person(1), n=2))
        updated = gallery.register("asha", _shots(_person(1), n=8, seed=7))
        assert updated.person_id == original.person_id
        assert updated.shots == 8
        assert len(gallery) == 1

    def test_name_matching_is_case_insensitive(self):
        gallery = EnrolledGallery()
        first = gallery.register("Asha", _shots(_person(1)))
        again = gallery.register("asha", _shots(_person(1), seed=3))
        assert again.person_id == first.person_id

    def test_registering_with_no_embeddings_raises(self):
        with pytest.raises(ValueError, match="no embeddings"):
            EnrolledGallery().register("asha", [])

    def test_reference_embedding_is_normalised(self):
        person = EnrolledGallery().register("asha", _shots(_person(1)))
        assert float(np.linalg.norm(person.embedding)) == pytest.approx(1.0, abs=1e-5)

    def test_identify_finds_the_right_person(self):
        gallery = EnrolledGallery()
        asha, bilal = _person(1), _person(2)
        gallery.register("asha", _shots(asha))
        gallery.register("bilal", _shots(bilal))
        hit = gallery.identify(_shots(asha, n=1, seed=42)[0])
        assert hit is not None and hit[0].name == "asha"

    def test_identify_returns_none_for_a_stranger(self):
        """An unregistered person is a real answer, not a failure -- they must
        not be forced onto the nearest registered id."""
        gallery = EnrolledGallery()
        gallery.register("asha", _shots(_person(1)))
        assert gallery.identify(_person(99)) is None

    def test_identify_on_an_empty_gallery_is_none(self):
        assert EnrolledGallery().identify(_person(1)) is None

    def test_identify_handles_a_missing_embedding(self):
        gallery = EnrolledGallery()
        gallery.register("asha", _shots(_person(1)))
        assert gallery.identify(None) is None


class TestConsentAffordances:
    def test_forget_removes_the_person(self):
        gallery = EnrolledGallery()
        gallery.register("asha", _shots(_person(1)))
        assert gallery.forget("Asha") is True
        assert len(gallery) == 0

    def test_forget_is_false_for_someone_not_registered(self):
        assert EnrolledGallery().forget("nobody") is False

    def test_a_forgotten_id_is_never_reused(self):
        """Reusing a retired id would silently turn one person's historical
        records into another person's."""
        gallery = EnrolledGallery()
        gallery.register("asha", _shots(_person(1)))
        gallery.register("bilal", _shots(_person(2)))
        gallery.forget("bilal")
        assert gallery.register("chandra", _shots(_person(3))).person_id == 3


class TestPersistence:
    def test_round_trip_preserves_people(self, tmp_path):
        gallery = EnrolledGallery()
        gallery.register("asha", _shots(_person(1)))
        gallery.register("bilal", _shots(_person(2)))
        path = gallery.save(tmp_path / "gallery.json")

        loaded = EnrolledGallery.load(path)
        assert [(p.person_id, p.name, p.shots) for p in loaded.people] == [
            (1, "asha", 6),
            (2, "bilal", 6),
        ]
        np.testing.assert_allclose(
            loaded.people[0].embedding, gallery.people[0].embedding, atol=1e-5
        )

    def test_a_reloaded_person_still_matches_themselves(self, tmp_path):
        """The whole point: identity survives the round trip to disk."""
        asha = _person(1)
        gallery = EnrolledGallery()
        gallery.register("asha", _shots(asha))
        path = gallery.save(tmp_path / "gallery.json")

        hit = EnrolledGallery.load(path).identify(_shots(asha, n=1, seed=42)[0])
        assert hit is not None and hit[0].name == "asha"

    def test_a_retired_id_stays_retired_across_a_reload(self, tmp_path):
        """Retirement must be a property of the gallery file, not of one
        process -- otherwise a restart quietly recycles the id."""
        gallery = EnrolledGallery()
        gallery.register("asha", _shots(_person(1)))
        gallery.register("bilal", _shots(_person(2)))
        gallery.forget("bilal")
        path = gallery.save(tmp_path / "gallery.json")

        reloaded = EnrolledGallery.load(path)
        assert reloaded.register("chandra", _shots(_person(3))).person_id == 3

    def test_loading_a_missing_file_gives_an_empty_gallery(self, tmp_path):
        """A first run before anyone is registered is not an error."""
        assert len(EnrolledGallery.load(tmp_path / "absent.json")) == 0

    def test_an_incompatible_format_version_raises(self, tmp_path):
        path = tmp_path / "gallery.json"
        path.write_text(
            f'{{"format_version": {GALLERY_FORMAT_VERSION + 1}, "people": []}}',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="format_version"):
            EnrolledGallery.load(path)

    def test_saving_creates_parent_directories(self, tmp_path):
        gallery = EnrolledGallery()
        gallery.register("asha", _shots(_person(1)))
        path = gallery.save(tmp_path / "nested" / "deeper" / "gallery.json")
        assert path.exists()


class TestEnrolledResolver:
    """The resolver must satisfy the same contract as TwoPassIdentityResolver,
    so backend.integrate can use either without knowing which it has."""

    @staticmethod
    def _feed(resolver, people, frames=20, seed=0):
        rng = np.random.default_rng(seed)
        for _ in range(frames):
            embeddings = [
                _unit(base + 0.0331 * rng.standard_normal(DIM).astype(np.float32))
                for base in people
            ]
            track_ids = list(range(1, len(people) + 1))
            scores = [0.9] * len(people)
            keys = resolver.keys_for(track_ids, embeddings, scores)
            resolver.observe(keys, embeddings, scores)
        return resolver.finalise()

    def test_registered_people_get_their_gallery_ids(self):
        people = [_person(1), _person(2), _person(3)]
        gallery = EnrolledGallery()
        for index, base in enumerate(people, start=1):
            gallery.register(f"student_{index}", _shots(base, seed=index))

        mapping = self._feed(EnrolledIdentityResolver(gallery), people)
        assert sorted({v for v in mapping.values() if v > 0}) == [1, 2, 3]

    def test_ids_are_identical_across_two_separate_videos(self):
        """The reason registration exists: person 2 on Tuesday is person 2 on
        Monday. Anonymous per-video ids cannot promise this."""
        people = [_person(1), _person(2), _person(3)]
        gallery = EnrolledGallery()
        for index, base in enumerate(people, start=1):
            gallery.register(f"student_{index}", _shots(base, seed=index))

        first = self._feed(EnrolledIdentityResolver(gallery), people, seed=1)
        second = self._feed(EnrolledIdentityResolver(gallery), people, seed=2)
        assert {v for v in first.values() if v > 0} == {v for v in second.values() if v > 0}

    def test_an_unregistered_person_is_reported_not_dropped(self):
        """Someone who never enrolled must still appear in the output, with an
        id outside the registered range rather than silently vanishing."""
        people = [_person(1), _person(2), _person(3)]
        gallery = EnrolledGallery()
        gallery.register("student_1", _shots(people[0], seed=1))

        ids = sorted({v for v in self._feed(EnrolledIdentityResolver(gallery), people).values() if v > 0})
        assert ids[0] == 1
        assert len(ids) == 3
        assert all(i > 1 for i in ids[1:])

    def test_co_occurrence_still_beats_a_gallery_match(self):
        """Two people alive in the same frame are provably different humans.
        Even if both match one registered reference, they must not share an id
        -- proof outranks similarity."""
        twin = _person(1)
        gallery = EnrolledGallery()
        gallery.register("student_1", _shots(twin, seed=1))

        # Two tracks that always co-occur, both looking like the same person.
        mapping = self._feed(EnrolledIdentityResolver(gallery), [twin, twin])
        positives = [v for v in mapping.values() if v > 0]
        assert len(set(positives)) == 2, "co-occurring tracks must not share an id"

    def test_an_empty_gallery_still_assigns_usable_ids(self):
        people = [_person(1), _person(2)]
        mapping = self._feed(EnrolledIdentityResolver(EnrolledGallery()), people)
        assert len({v for v in mapping.values() if v > 0}) == 2

    def test_faceless_tracks_keep_negative_ids(self):
        """Same convention as TwoPassIdentityResolver: no trustworthy face
        means no positive identity claim."""
        gallery = EnrolledGallery()
        gallery.register("student_1", _shots(_person(1), seed=1))
        resolver = EnrolledIdentityResolver(gallery)
        resolver.observe([7], [None], [None])
        assert resolver.finalise()[7] < 0


class TestPipelineDefault:
    def test_gallery_path_is_off_by_default(self):
        """Persistent identity must never switch itself on. The session-scoped
        property is the documented default."""
        assert CONFIG.identity.gallery_path is None


class TestPipelineWiring:
    """process_video must actually use the gallery when one is configured.

    Everything is faked except identity itself, so this pins the wiring -- does
    a configured gallery reach the resolver, and do its ids reach the output --
    without needing models or real footage.
    """

    @staticmethod
    def _run(tmp_path, gallery_path, person_embedding):
        import cv2

        from backend.detection import Person
        from backend.face import FaceResult
        from backend.headpose import HeadPoseResult
        from backend.integrate import process_video
        from backend.posture import PostureResult

        class _Detector:
            def detect(self, frame):
                return [Person(bbox=(20, 15, 60, 80), confidence=0.92)], []

        class _Faces:
            """Like the real analyzer, but every face is the same known person."""

            def analyze(self, frame, person_bboxes, detected_faces=None):
                return [
                    FaceResult(
                        face_bbox=(x + 5, y + 5, max(1, w - 10), max(1, h - 10)),
                        landmarks=[(float(x), float(y))] * 468,
                        ear=0.31,
                        embedding=person_embedding,
                        score=0.95,
                    )
                    for x, y, w, h in person_bboxes
                ]

        class _Pose:
            def estimate(self, frame, face_bboxes):
                return [
                    None
                    if bbox is None
                    else HeadPoseResult(yaw=2.0, pitch=-3.0, roll=1.0, gaze_label="teacher")
                    for bbox in face_bboxes
                ]

        class _Posture:
            def analyze(self, frame, person_bboxes):
                return [
                    PostureResult(
                        keypoints_detected=True,
                        nose=(float(x + w / 2), float(y + h * 0.1)),
                        left_shoulder=(float(x + w * 0.3), float(y + h * 0.3)),
                        right_shoulder=(float(x + w * 0.7), float(y + h * 0.3)),
                        shoulder_mid=(float(x + w / 2), float(y + h * 0.3)),
                        hip_mid=(float(x + w / 2), float(y + h * 0.7)),
                        vertical_lean=-0.2,
                        facing_direction=(0.0, -1.0),
                    )
                    for x, y, w, h in person_bboxes
                ]

        video = tmp_path / "clip.mp4"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 6.0, (160, 120))
        if not writer.isOpened():
            pytest.skip("VideoWriter codec unavailable.")
        try:
            for i in range(18):
                writer.write(np.full((120, 160, 3), i % 256, dtype=np.uint8))
        finally:
            writer.release()

        out = tmp_path / "stage1.jsonl"
        config = replace(CONFIG, identity=replace(CONFIG.identity, gallery_path=str(gallery_path)))
        process_video(
            video,
            out,
            config,
            detector=_Detector(),
            face_analyzer=_Faces(),
            headpose_estimator=_Pose(),
            posture_analyzer=_Posture(),
        )
        return [
            person["person_id"]
            for line in out.read_text(encoding="utf-8").strip().splitlines()
            for person in json.loads(line)["persons"]
        ]

    def test_a_registered_person_keeps_their_gallery_id_in_the_output(self, tmp_path):
        asha = _person(1)
        gallery = EnrolledGallery()
        gallery.register("filler", _shots(_person(50), seed=5))
        registered = gallery.register("asha", _shots(asha, seed=1))
        path = gallery.save(tmp_path / "gallery.json")

        ids = self._run(tmp_path, path, _shots(asha, n=1, seed=42)[0])
        assert ids, "pipeline produced no person records"
        assert set(ids) == {registered.person_id}

    def test_an_empty_gallery_falls_back_instead_of_failing(self, tmp_path):
        """A configured but empty gallery must not crash the pipeline -- it is
        what a first run looks like before anyone has been registered."""
        path = EnrolledGallery().save(tmp_path / "gallery.json")
        ids = self._run(tmp_path, path, _person(1))
        assert ids and all(isinstance(i, int) for i in ids)
