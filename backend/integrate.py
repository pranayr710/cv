"""Stage 1+2 integration — wire detection + face + head-pose + posture +
tracking into JSONL.

Runs the perception modules over a video and emits the Stage 1 contract, one
JSON object per processed frame:

    Detector       -> persons (bbox, confidence) + objects
    FaceAnalyzer   -> per-person face landmarks + EAR (index-aligned)
    HeadPoseEstimator -> per-face yaw/pitch/roll + gaze label (index-aligned)
    PostureAnalyzer   -> per-person raw pose geometry (index-aligned)
    PersonTracker  -> per-person track_id (Stage 2, index-aligned)

A person's ``track_id`` is ``null`` whenever ByteTrack has not (yet) confirmed
them as a track this frame — expected on a first sighting, not a dropped
detection; see :mod:`backend.tracking`. Output validates against
``schema.json``, which already types ``track_id`` as ``int | null`` for
exactly this reason.

``posture`` is run for every person regardless of whether a face was found —
that is the point of it: on real classroom footage a large fraction of persons
have no detectable face at all (bowed over a desk, turned away), and
:mod:`backend.posture` recovers a different, face-independent signal for that
population. It is raw geometry, not a posture classification — see that
module's docstring.

Usage (CLI):
    python -m backend.integrate --video in.mp4 --out out.jsonl --sample-rate 5

Usage (API):
    from backend.config import CONFIG
    from backend.integrate import process_video
    n = process_video("in.mp4", "out.jsonl", CONFIG)
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import numpy as np

from backend.config import CONFIG, Config

if TYPE_CHECKING:  # pragma: no cover - typing only
    from backend.detection import Detector, Obj, Person
    from backend.expression import ExpressionRecognizer
    from backend.face import FaceAnalyzer, FaceResult
    from backend.headpose import HeadPoseEstimator, HeadPoseResult
    from backend.posture import PostureAnalyzer, PostureResult
    from backend.tracking import PersonTracker

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Structural interfaces (duck-typed) so real modules and test fakes both fit.
# --------------------------------------------------------------------------- #


class DetectorLike(Protocol):
    """Anything exposing ``detect(frame) -> (persons, objects)``."""

    def detect(self, frame: np.ndarray) -> tuple[list[Person], list[Obj]]: ...


class FaceAnalyzerLike(Protocol):
    """Anything exposing ``analyze(frame, person_bboxes) -> list[FaceResult]``.

    An implementation may *optionally* also expose
    ``detect_faces(frame) -> list[DetectedFace]``. When it does, the pipeline
    reuses that whole-frame face list to recover students whose bodies person
    detection missed (see :mod:`backend.students`). When it does not — a test
    fake, or the ``"mediapipe"`` backend, which has no whole-frame step — that
    recovery is simply skipped.
    """

    def analyze(
        self, frame: np.ndarray, person_bboxes: Sequence[Sequence[float]]
    ) -> list[FaceResult]: ...


class HeadPoseLike(Protocol):
    """Anything exposing ``estimate(frame, face_bboxes) -> list[result|None]``."""

    def estimate(
        self, frame: np.ndarray, face_bboxes: Sequence[Sequence[float] | None]
    ) -> list[HeadPoseResult | None]: ...


class PostureAnalyzerLike(Protocol):
    """Anything exposing ``analyze(frame, person_bboxes) -> list[PostureResult]``."""

    def analyze(
        self, frame: np.ndarray, person_bboxes: Sequence[Sequence[float]]
    ) -> list[PostureResult]: ...


class ExpressionLike(Protocol):
    """Anything exposing ``classify(frame, face_bboxes) -> list[result|None]``."""

    def classify(
        self, frame: np.ndarray, face_bboxes: Sequence[Sequence[float] | None]
    ) -> list[object | None]: ...


class PersonTrackerLike(Protocol):
    """Anything exposing ``update(persons) -> list[track_id|None]``."""

    def update(self, persons: Sequence[Person]) -> list[int | None]: ...


def _face_to_json(face: FaceResult | None) -> dict | None:
    """Serialise a FaceResult into the frozen ``face`` object, or ``None``.

    A face is considered present only when it has a bounding box. Landmarks and
    EAR may still be ``None`` within a present face (e.g. degenerate eyes).

    Args:
        face: The per-person face result, or ``None``.

    Returns:
        A dict matching the schema's ``face`` object, or ``None`` when no face
        was matched to this person.
    """
    if face is None or face.face_bbox is None:
        return None
    landmarks = (
        [[float(x), float(y)] for x, y in face.landmarks]
        if face.landmarks is not None
        else None
    )
    return {
        "bbox": [int(v) for v in face.face_bbox],
        "landmarks": landmarks,
        "ear": None if face.ear is None else float(face.ear),
    }


def _headpose_to_json(hp: HeadPoseResult | None) -> dict | None:
    """Serialise a HeadPoseResult into the frozen ``head_pose`` object, or None.

    Args:
        hp: The per-person head-pose result, or ``None``.

    Returns:
        A dict matching the schema's ``head_pose`` object, or ``None``.
    """
    if hp is None:
        return None
    return {
        "yaw": float(hp.yaw),
        "pitch": float(hp.pitch),
        "roll": float(hp.roll),
        "gaze_label": hp.gaze_label,
    }


def _expression_to_json(expression) -> dict | None:
    """Serialise an ExpressionResult into the ``expression`` object, or ``None``.

    Args:
        expression: The per-person :class:`~backend.expression.ExpressionResult`,
            or ``None`` when there was no face or it was too small to classify.

    Returns:
        A dict matching the schema's ``expression`` object, or ``None``. The
        label describes the **visible expression**, never an inferred emotional
        state — see :mod:`backend.expression` for why that wording is
        load-bearing.
    """
    if expression is None:
        return None
    return {
        "label": expression.label,
        "confidence": float(expression.confidence),
        "distribution": (
            {k: float(v) for k, v in expression.distribution.items()}
            if expression.distribution
            else None
        ),
    }


def _point_to_json(point: tuple[float, float] | None) -> list[float] | None:
    """Serialise an (x, y) point, or ``None``.

    Args:
        point: A 2-tuple of coordinates, or ``None``.

    Returns:
        A two-element list, or ``None``.
    """
    return None if point is None else [float(point[0]), float(point[1])]


def _posture_to_json(posture: PostureResult | None) -> dict | None:
    """Serialise a PostureResult into the ``posture`` object, or ``None``.

    Args:
        posture: The per-person posture result, or ``None``.

    Returns:
        A dict matching the schema's ``posture`` object, or ``None`` when
        MediaPipe Pose found no body in this person's crop. This is raw
        geometry, not a posture classification — see backend/posture.py.
    """
    if posture is None or not posture.keypoints_detected:
        return None
    return {
        "nose": _point_to_json(posture.nose),
        "left_shoulder": _point_to_json(posture.left_shoulder),
        "right_shoulder": _point_to_json(posture.right_shoulder),
        "shoulder_mid": _point_to_json(posture.shoulder_mid),
        "hip_mid": _point_to_json(posture.hip_mid),
        "vertical_lean": (
            None if posture.vertical_lean is None else float(posture.vertical_lean)
        ),
        "facing_direction": _point_to_json(posture.facing_direction),
    }


def _assemble_frame(
    frame_id: int,
    timestamp_ms: int,
    persons: list[Person],
    faces: list[FaceResult],
    headposes: list[HeadPoseResult | None],
    postures: list[PostureResult],
    expressions: list[object | None],
    track_ids: list[int | None],
    objects: list[Obj],
) -> dict:
    """Build one JSONL record in the Stage 1 schema.

    Args:
        frame_id: Zero-indexed source frame number.
        timestamp_ms: Frame presentation time in milliseconds.
        persons: Detected persons for this frame.
        faces: Face results, index-aligned with ``persons``.
        headposes: Head-pose results, index-aligned with ``persons``.
        postures: Raw pose-geometry results, index-aligned with ``persons``.
            Computed independently of ``faces``/``headposes`` — see
            :mod:`backend.posture`.
        track_ids: Stage 2 track ids, index-aligned with ``persons``. An entry
            is ``None`` when ByteTrack has not (yet) confirmed that person as a
            track this frame (see :mod:`backend.tracking`) — expected, not an
            error.
        objects: Detected whitelisted objects.

    Returns:
        A JSON-serialisable dict matching ``schema.json``.

    Raises:
        ValueError: If ``faces``/``headposes``/``postures``/``track_ids`` are
            not aligned with ``persons``.
    """
    if not (
        len(persons)
        == len(faces)
        == len(headposes)
        == len(postures)
        == len(expressions)
        == len(track_ids)
    ):
        raise ValueError(
            "Misaligned per-person lists: "
            f"persons={len(persons)}, faces={len(faces)}, "
            f"headposes={len(headposes)}, postures={len(postures)}, "
            f"expressions={len(expressions)}, track_ids={len(track_ids)}."
        )

    person_records = []
    for person, face, hp, posture, expression, track_id in zip(
        persons, faces, headposes, postures, expressions, track_ids
    ):
        person_records.append(
            {
                "track_id": None if track_id is None else int(track_id),
                "bbox": [int(v) for v in person.bbox],
                "confidence": float(person.confidence),
                # "face_seeded" marks a student whose bbox is estimated from
                # their face because person detection missed the occluded body.
                # Kept in the output so no consumer mistakes an estimate for a
                # measurement — see backend/students.py.
                "source": getattr(person, "source", "yolo"),
                "face": _face_to_json(face),
                "head_pose": _headpose_to_json(hp),
                "posture": _posture_to_json(posture),
                "expression": _expression_to_json(expression),
            }
        )

    object_records = [
        {
            "cls": obj.cls,
            "bbox": [int(v) for v in obj.bbox],
            "confidence": float(obj.confidence),
        }
        for obj in objects
    ]

    return {
        "frame_id": int(frame_id),
        "timestamp_ms": int(timestamp_ms),
        "persons": person_records,
        "objects": object_records,
    }


def _build_detector(config: Config) -> Detector:
    """Construct the real :class:`~backend.detection.Detector` from config."""
    from backend.detection import Detector

    return Detector(config.detection)


def _build_face_analyzer(config: Config) -> FaceAnalyzer:
    """Construct the real :class:`~backend.face.FaceAnalyzer` from config."""
    from backend.face import FaceAnalyzer

    return FaceAnalyzer(config.face)


def _build_expression_recognizer(config: Config) -> ExpressionRecognizer:
    """Construct the real :class:`~backend.expression.ExpressionRecognizer`."""
    from backend.expression import ExpressionRecognizer

    return ExpressionRecognizer(config.expression)


def _build_headpose_estimator(config: Config) -> HeadPoseEstimator:
    """Construct the real :class:`~backend.headpose.HeadPoseEstimator` from config."""
    from backend.headpose import HeadPoseEstimator

    return HeadPoseEstimator(config.headpose)


def _build_posture_analyzer(config: Config) -> PostureAnalyzer:
    """Construct the real :class:`~backend.posture.PostureAnalyzer` from config."""
    from backend.posture import PostureAnalyzer

    return PostureAnalyzer(config.posture)


def _build_person_tracker(config: Config) -> PersonTracker:
    """Construct the real :class:`~backend.tracking.PersonTracker` from config."""
    from backend.tracking import PersonTracker

    return PersonTracker(config.tracking)


def process_video(
    video_path: str | Path,
    out_jsonl_path: str | Path,
    config: Config = CONFIG,
    *,
    detector: DetectorLike | None = None,
    face_analyzer: FaceAnalyzerLike | None = None,
    headpose_estimator: HeadPoseLike | None = None,
    posture_analyzer: PostureAnalyzerLike | None = None,
    expression_recognizer: ExpressionLike | None = None,
    person_tracker: PersonTrackerLike | None = None,
) -> int:
    """Run the full Stage 1+2 pipeline over a video and write JSONL output.

    Args:
        video_path: Path to the input video file.
        out_jsonl_path: Path to write the JSONL output to. Parent directories
            are created if missing.
        config: The full pipeline config. ``config.pipeline.sample_rate``
            controls frame subsampling; ``config.pipeline.log_every_frames``
            controls how often throughput is logged.
        detector: Optional detector to reuse (constructed from config if None).
        face_analyzer: Optional face analyzer (constructed from config if None).
        headpose_estimator: Optional head-pose estimator (built if None).
        posture_analyzer: Optional posture analyzer (built from config if
            None). Runs on every person independently of face/head-pose.
        person_tracker: Optional tracker to reuse (built from config if None).
            Must be fresh for this video — see :class:`backend.tracking.PersonTracker`.

    Returns:
        The number of frames processed and written.

    Raises:
        FileNotFoundError: If the input video does not exist.
        RuntimeError: If the video cannot be opened by OpenCV.
        ImportError: If a required ML package is missing and no estimator was
            injected.
    """
    import cv2

    src = Path(video_path)
    if not src.is_file():
        raise FileNotFoundError(f"Input video not found: {src}")

    # Each component builds independently, only if not injected, so a caller
    # supplying fakes for the heavy ML modules never pulls in their real
    # dependencies just because a different component (e.g. the tracker) was
    # left to build for real, and vice versa.
    detector = detector or _build_detector(config)
    face_analyzer = face_analyzer or _build_face_analyzer(config)
    headpose_estimator = headpose_estimator or _build_headpose_estimator(config)
    posture_analyzer = posture_analyzer or _build_posture_analyzer(config)
    expression_recognizer = expression_recognizer or _build_expression_recognizer(config)
    person_tracker = person_tracker or _build_person_tracker(config)

    sample_rate = max(int(config.pipeline.sample_rate), 1)
    log_every = max(int(config.pipeline.log_every_frames), 1)

    capture = cv2.VideoCapture(str(src))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"OpenCV could not open video: {src}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        logger.warning("Video reports invalid FPS (%s); timestamps use 0.", fps)
        fps = 0.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    out_path = Path(out_jsonl_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from tqdm import tqdm
    except ImportError as exc:  # pragma: no cover - tqdm is a hard dependency
        raise ImportError("tqdm is required. Install it via requirements.txt.") from exc

    frame_index = 0
    written = 0
    start = time.perf_counter()

    progress = tqdm(
        total=total_frames if total_frames > 0 else None,
        desc="frames",
        unit="frame",
    )
    try:
        with out_path.open("w", encoding="utf-8") as fh:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                progress.update(1)

                if frame_index % sample_rate != 0:
                    frame_index += 1
                    continue

                pos_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
                if pos_ms and pos_ms > 0:
                    timestamp_ms = round(pos_ms)
                elif fps > 0:
                    timestamp_ms = round(frame_index * 1000.0 / fps)
                else:
                    timestamp_ms = 0

                persons, objects = detector.detect(frame)

                # Face detection runs once here and its result is threaded into
                # both consumers. Two reasons: it is the most expensive stage in
                # the pipeline, and student seeding must see the same face list
                # the face-to-person assignment will see, or the two could
                # disagree about whether a student is already covered.
                # detect_faces is optional on the interface, so an analyzer
                # without it (test fake, mediapipe backend) just skips seeding.
                detect_faces = getattr(face_analyzer, "detect_faces", None)
                detected_faces = detect_faces(frame) if detect_faces else []
                if detected_faces:
                    from backend.students import augment_persons

                    persons = augment_persons(
                        persons,
                        detected_faces,
                        frame.shape[:2],
                        config.students,
                    )

                person_bboxes = [p.bbox for p in persons]
                faces = (
                    face_analyzer.analyze(frame, person_bboxes, detected_faces)
                    if detected_faces
                    else face_analyzer.analyze(frame, person_bboxes)
                )
                face_bboxes = [f.face_bbox for f in faces]
                headposes = headpose_estimator.estimate(frame, face_bboxes)
                # Posture runs on every person, not just faceless ones: it is
                # a face-independent signal by design (see backend/posture.py).
                postures = posture_analyzer.analyze(frame, person_bboxes)
                # Expression consumes the same face boxes as head pose; it needs
                # no landmarks, so it covers every student who has a face box.
                expressions = expression_recognizer.classify(frame, face_bboxes)
                # Tracking runs on every processed frame, in order: its motion
                # model assumes fixed spacing between consecutive updates, so
                # this must stay inside the sample_rate-filtered branch.
                track_ids = person_tracker.update(persons)

                record = _assemble_frame(
                    frame_index,
                    timestamp_ms,
                    persons,
                    faces,
                    headposes,
                    postures,
                    expressions,
                    track_ids,
                    objects,
                )
                fh.write(json.dumps(record) + "\n")
                written += 1

                if written % log_every == 0:
                    elapsed = time.perf_counter() - start
                    rate = written / elapsed if elapsed > 0 else 0.0
                    logger.info(
                        "Processed %d frames (%.1f FPS, last frame: %d persons, "
                        "%d objects).",
                        written,
                        rate,
                        len(persons),
                        len(objects),
                    )
                frame_index += 1
    finally:
        progress.close()
        capture.release()

    elapsed = time.perf_counter() - start
    rate = written / elapsed if elapsed > 0 else 0.0
    logger.info(
        "Done: %d frames written to %s (%.1f FPS avg).", written, out_path, rate
    )
    return written


def _positive_int(value: str) -> int:
    """Argparse type: parse a strictly-positive integer.

    Args:
        value: The raw CLI string.

    Returns:
        The parsed integer (>= 1).

    Raises:
        argparse.ArgumentTypeError: If ``value`` is not an integer >= 1.
    """
    try:
        parsed = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}.")
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {parsed}.")
    return parsed


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        A configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="python -m backend.integrate",
        description=(
            "Run the full ClassGraph Stage 1+2 pipeline (detection + face + "
            "head pose + tracking) over a video and write per-frame JSONL."
        ),
    )
    parser.add_argument("--video", required=True, type=str, help="Input video path.")
    parser.add_argument(
        "--out",
        type=str,
        default=CONFIG.pipeline.default_output,
        help=f"Output JSONL path (default: {CONFIG.pipeline.default_output}).",
    )
    parser.add_argument(
        "--sample-rate",
        type=_positive_int,
        default=CONFIG.pipeline.sample_rate,
        help="Process every Nth frame, integer >= 1 (default: %(default)s).",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default=None,
        help="Override the compute device for detection and head pose.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=CONFIG.log_level,
        help=f"Logging verbosity (default: {CONFIG.log_level}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]`` when ``None``.

    Returns:
        Process exit code (``0`` on success, ``1`` on a handled failure).
        Invalid arguments (e.g. ``--sample-rate 0``) exit ``2`` via argparse.
    """
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    config = replace(
        CONFIG, pipeline=replace(CONFIG.pipeline, sample_rate=args.sample_rate)
    )
    if args.device is not None:
        config = replace(
            config,
            detection=replace(config.detection, device=args.device),
            headpose=replace(config.headpose, device=args.device),
        )

    try:
        written = process_video(args.video, args.out, config)
    except (FileNotFoundError, RuntimeError, ImportError, ValueError) as exc:
        logger.error("Pipeline failed: %s", exc)
        return 1

    logger.info("Wrote %d frames to %s.", written, args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via CLI
    import sys

    sys.exit(main())
