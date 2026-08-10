"""Recover students that person detection missed, using their detected face.

Why this module exists
----------------------

Measured across the 13 real classroom images in ``dataset/`` (see
``tools/bench_faces.py`` and ``tools/bench_persons.py``):

=========================================  ========
Signal                                     Count
=========================================  ========
Faces found by SCRFD                        434
Persons found by YOLOv11m @ imgsz 1280      264
Persons found by YOLOv11x @ imgsz 1920      374
=========================================  ========

Face detection finds **more students than person detection does**, at every
YOLO model size and inference size tried. The unmatched faces were rendered and
inspected by hand on ``img382.jpg`` (19 person boxes vs 56 faces): every single
unmatched face was a real student's head in a crowded back row. They are not
false positives.

The reason is geometric, not a tuning failure. In a classroom the camera sees a
student's **head** clearly and their **body** barely at all -- torsos are
occluded by desks, by the row in front, and by each other. COCO's "person"
class expects a mostly-visible human figure. A head is exactly what a face
detector is built for.

So the pipeline should not treat "person" as the primary unit and "face" as an
attribute of it. For classroom footage the face is the **more reliable anchor**.
This module inverts that dependency for the missed cases: any detected face
with no containing person box becomes a student in its own right, with an
**estimated** body box derived from the face geometry.

What this deliberately does NOT claim
-------------------------------------

The estimated body box is a geometric extrapolation from face size, not a
detection. It is fine for the things that only need a rough spatial anchor
(associating a nearby phone or book, seating position, tracking continuity) and
it is **not** suitable for anything needing true body extent. Every synthesised
student is therefore tagged ``source="face_seeded"`` in the output so downstream
consumers -- and anyone reading the JSONL -- can tell the difference. Students
found by YOLO keep ``source="yolo"``.

:mod:`backend.posture` will usually fail on a face-seeded student, which is
correct and expected: if YOLO could not see the body, MediaPipe Pose generally
cannot either. Such a student still yields face, gaze and expression signals.

Usage:
    from backend.students import augment_persons
    persons = augment_persons(persons, detected_faces, frame.shape[:2])
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from backend.config import CONFIG, StudentResolutionConfig
from backend.detection import Person
from backend.face_detect import DetectedFace

logger = logging.getLogger(__name__)

Bbox = tuple[int, int, int, int]


def _containment(inner: Bbox, outer: Bbox) -> float:
    """Fraction of ``inner``'s area that lies inside ``outer``.

    Args:
        inner: The box whose containment is measured (a face box).
        outer: The containing box (a person box).

    Returns:
        A value in ``[0.0, 1.0]``.
    """
    ix0, iy0, iw, ih = inner
    ox0, oy0, ow, oh = outer
    inter_w = max(0, min(ix0 + iw, ox0 + ow) - max(ix0, ox0))
    inter_h = max(0, min(iy0 + ih, oy0 + oh) - max(iy0, oy0))
    area = iw * ih
    if area <= 0:
        return 0.0
    return (inter_w * inter_h) / area


def estimate_body_box(
    face_bbox: Bbox,
    frame_h: int,
    frame_w: int,
    config: StudentResolutionConfig | None = None,
) -> Bbox:
    """Estimate a seated student's visible body box from their face box.

    A rough anthropometric extrapolation: shoulder span is a little over twice
    head width, and from an elevated classroom camera the visible extent of a
    seated student runs from just above the head to roughly the desk edge. Both
    multipliers are config-driven (:class:`StudentResolutionConfig`) and are
    **estimates, not measurements** -- see the module docstring.

    Args:
        face_bbox: The detected face box ``(x, y, w, h)`` in image pixels.
        frame_h: Frame height in pixels, used to clamp the result.
        frame_w: Frame width in pixels, used to clamp the result.
        config: Geometry settings. Defaults to ``CONFIG.students``.

    Returns:
        The estimated body box ``(x, y, w, h)``, clamped to the frame, with
        ``w``/``h`` at least 1 pixel so it satisfies the schema.
    """
    cfg = config if config is not None else CONFIG.students
    fx, fy, fw, fh = face_bbox

    body_w = fw * cfg.body_width_to_face_width
    body_h = fh * cfg.body_height_to_face_height
    face_cx = fx + fw / 2.0

    x0 = face_cx - body_w / 2.0
    y0 = fy - fh * cfg.body_top_above_face

    x0i = max(0, min(round(x0), frame_w - 1))
    y0i = max(0, min(round(y0), frame_h - 1))
    x1i = max(0, min(round(x0 + body_w), frame_w))
    y1i = max(0, min(round(y0 + body_h), frame_h))

    return (x0i, y0i, max(1, x1i - x0i), max(1, y1i - y0i))


def augment_persons(
    persons: Sequence[Person],
    detected_faces: Sequence[DetectedFace],
    frame_shape: tuple[int, int],
    config: StudentResolutionConfig | None = None,
    containment_threshold: float | None = None,
) -> list[Person]:
    """Add a student for every detected face that no person box contains.

    Args:
        persons: Persons found by :class:`~backend.detection.Detector`.
        detected_faces: Faces found by :class:`~backend.face_detect.FaceDetector`
            on the same frame.
        frame_shape: ``(height, width)`` of the frame, for clamping.
        config: Geometry settings. Defaults to ``CONFIG.students``.
        containment_threshold: Minimum fraction of a face box that must lie
            inside a person box to count as already covered. Defaults to
            ``CONFIG.face.assign_min_containment`` so this module and
            :mod:`backend.face`'s assignment agree on what "covered" means --
            if they disagreed, a face could be seeded here *and* matched to an
            existing person there, double-counting the student.

    Returns:
        A new list: the original persons (order preserved, unmodified) followed
        by one synthesised person per uncovered face, ordered by descending face
        confidence. Returns the original persons unchanged when
        ``config.seed_persons_from_faces`` is ``False``.
    """
    cfg = config if config is not None else CONFIG.students
    if not cfg.seed_persons_from_faces:
        return list(persons)

    threshold = (
        containment_threshold
        if containment_threshold is not None
        else CONFIG.face.assign_min_containment
    )
    frame_h, frame_w = frame_shape

    existing = [p.bbox for p in persons]
    seeded: list[Person] = []
    for face in sorted(detected_faces, key=lambda f: -f.score):
        if face.score < cfg.seed_min_face_score:
            continue
        if any(_containment(face.bbox, pb) >= threshold for pb in existing):
            continue
        body = estimate_body_box(face.bbox, frame_h, frame_w, cfg)
        # Guard against seeding two students from two detections of the same
        # head: an already-seeded body box that contains this face means the
        # student is accounted for.
        if any(_containment(face.bbox, s.bbox) >= threshold for s in seeded):
            continue
        # Confidence is the *face* detector's score, carried over honestly --
        # this student's evidence is their face, not a person detection.
        seeded.append(
            Person(bbox=body, confidence=face.score, source="face_seeded")
        )

    if seeded:
        logger.debug(
            "Seeded %d students from faces that person detection missed "
            "(%d YOLO persons -> %d total).",
            len(seeded),
            len(persons),
            len(persons) + len(seeded),
        )
    return list(persons) + seeded
