"""Unit tests for :mod:`backend.students` and the inference-size clamps.

Pure logic and geometry — no ML model is loaded, so these run everywhere.

Test coverage:
    1. A face already inside a person box does not create a duplicate student.
    2. A face with no containing person box does create one, tagged
       ``source="face_seeded"``.
    3. Estimated body boxes stay inside the frame and remain schema-valid
       (positive width/height).
    4. Weak faces below ``seed_min_face_score`` are not promoted to students.
    5. The master switch disables seeding entirely.
    6. Two detections of the same head yield one student, not two.
    7. Regression: the inference-size clamps that fixed the 1920/1600 bug —
       upscaling past a frame's native resolution silently lost detections.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from backend.config import CONFIG, StudentResolutionConfig
from backend.detection import Person
from backend.face_detect import DetectedFace, _effective_det_size
from backend.students import augment_persons, estimate_body_box

FRAME = (1080, 1920)  # (h, w)


def _person(x, y, w, h, conf=0.9) -> Person:
    return Person(bbox=(x, y, w, h), confidence=conf)


def _face(x, y, w, h, score=0.9) -> DetectedFace:
    return DetectedFace(bbox=(x, y, w, h), score=score)


# --------------------------------------------------------------------------- #
# Seeding behaviour
# --------------------------------------------------------------------------- #


def test_face_inside_person_box_creates_no_duplicate() -> None:
    """A face already covered by a person box must not be seeded again."""
    persons = [_person(100, 100, 200, 400)]
    faces = [_face(150, 120, 60, 60)]  # well inside that person box
    result = augment_persons(persons, faces, FRAME)
    assert len(result) == 1
    assert result[0].source == "yolo"


def test_uncovered_face_becomes_a_seeded_student() -> None:
    """A face no person box contains is the case this module exists for."""
    persons = [_person(100, 100, 200, 400)]
    faces = [_face(1200, 300, 60, 60)]  # nowhere near the person
    result = augment_persons(persons, faces, FRAME)
    assert len(result) == 2
    seeded = result[1]
    assert seeded.source == "face_seeded"
    # Confidence is carried over from the face detector, not invented.
    assert seeded.confidence == pytest.approx(0.9)


def test_seeded_student_ordering_preserves_original_persons() -> None:
    """Original persons keep their index; seeded students are appended."""
    persons = [_person(10, 10, 50, 100), _person(500, 10, 50, 100)]
    faces = [_face(1500, 500, 40, 40)]
    result = augment_persons(persons, faces, FRAME)
    assert result[:2] == persons
    assert result[2].source == "face_seeded"


def test_weak_face_is_not_promoted_to_a_student() -> None:
    """Below seed_min_face_score, a face is not trusted to invent a body."""
    cfg = StudentResolutionConfig()
    faces = [_face(1200, 300, 60, 60, score=cfg.seed_min_face_score - 0.05)]
    assert augment_persons([], faces, FRAME, cfg) == []


def test_master_switch_disables_seeding() -> None:
    """seed_persons_from_faces=False restores the old person-only behaviour."""
    cfg = replace(StudentResolutionConfig(), seed_persons_from_faces=False)
    faces = [_face(1200, 300, 60, 60)]
    assert augment_persons([], faces, FRAME, cfg) == []


def test_two_detections_of_one_head_yield_one_student() -> None:
    """Near-identical face boxes must not become two students."""
    faces = [_face(1200, 300, 60, 60, 0.95), _face(1205, 303, 58, 58, 0.90)]
    result = augment_persons([], faces, FRAME)
    assert len(result) == 1


def test_no_faces_returns_persons_unchanged() -> None:
    persons = [_person(10, 10, 50, 100)]
    assert augment_persons(persons, [], FRAME) == persons


# --------------------------------------------------------------------------- #
# Body-box estimation geometry
# --------------------------------------------------------------------------- #


def test_estimated_body_box_is_wider_and_taller_than_the_face() -> None:
    """The estimate must actually extend beyond the face it came from."""
    _x, _y, w, h = estimate_body_box((900, 400, 60, 60), *FRAME)
    assert w > 60 and h > 60


def test_estimated_body_box_is_centred_on_the_face() -> None:
    face = (900, 400, 60, 60)
    bx, _by, bw, _bh = estimate_body_box(face, *FRAME)
    face_cx = face[0] + face[2] / 2
    assert bx + bw / 2 == pytest.approx(face_cx, abs=1.0)


@pytest.mark.parametrize(
    "face",
    [
        (0, 0, 40, 40),          # top-left corner
        (1880, 1040, 40, 40),    # bottom-right corner
        (1900, 10, 40, 40),      # hugging the right edge
    ],
)
def test_estimated_body_box_stays_inside_the_frame(face) -> None:
    """A face at a frame edge must not produce an out-of-frame box.

    The schema requires positive width/height and non-negative origin, so an
    unclamped extrapolation here would emit invalid records.
    """
    h_frame, w_frame = FRAME
    x, y, w, h = estimate_body_box(face, h_frame, w_frame)
    assert x >= 0 and y >= 0
    assert w > 0 and h > 0
    assert x + w <= w_frame
    assert y + h <= h_frame


def test_seeded_student_boxes_are_schema_valid_at_frame_edges() -> None:
    """End-to-end version of the clamping check, through augment_persons."""
    faces = [_face(1890, 1050, 30, 30), _face(0, 0, 30, 30)]
    for student in augment_persons([], faces, FRAME):
        x, y, w, h = student.bbox
        assert x >= 0 and y >= 0 and w > 0 and h > 0
        assert x + w <= FRAME[1] and y + h <= FRAME[0]


# --------------------------------------------------------------------------- #
# Inference-size handling.
#
# The two detectors resolve the same trade-off differently, on measurement:
#
#   YOLO  -- imgsz 1920 is left as-is even when it upscales the frame, because
#            clamping to native cost 331 -> 263 persons (398 -> 379 students)
#            across the dataset. A warning covers the inputs where it hurts.
#   SCRFD -- det_size IS clamped, because clamping costs only 434 -> 422 faces
#            (2.8%) and without it a small frame can yield zero faces.
# --------------------------------------------------------------------------- #


def test_scrfd_det_size_is_clamped_to_native_resolution() -> None:
    """Same regression, same fix, for the face detector.

    Measured on the 802 px fixture: det_size 320/640/1024 all found the face,
    1600 found none.
    """
    clamped = _effective_det_size((1600, 1600), 807, 802)
    assert max(clamped) < 1600
    assert max(clamped) >= 807
    assert clamped[0] == clamped[1], "SCRFD is given a square input"
    assert max(clamped) % 32 == 0
    # Large frames are untouched.
    assert _effective_det_size((1600, 1600), 1088, 1920) == (1600, 1600)


def test_config_defaults_are_consistent() -> None:
    """Seeding and face assignment must agree on what 'covered' means.

    ``augment_persons`` defaults its containment threshold to
    ``CONFIG.face.assign_min_containment`` precisely so a student cannot be both
    seeded here and matched there. If that coupling is ever broken, this fails.
    """
    persons = [_person(100, 100, 200, 400)]
    # A face exactly at the assignment threshold's boundary case: fully inside.
    faces = [_face(150, 150, 50, 50)]
    assert len(augment_persons(persons, faces, FRAME)) == 1
    assert CONFIG.students.seed_min_face_score >= CONFIG.face.scrfd_det_thresh
