"""Person + object detection for ClassGraph Stage 1 (Perception).

Wraps Ultralytics YOLOv11 (pretrained on COCO) to produce, per frame:

* ``Person`` records  — ``bbox`` + ``confidence`` for every detected person
  above :data:`DetectionConfig.person_conf`.
* ``Obj`` records     — ``cls`` + ``bbox`` + ``confidence`` for every object
  whose COCO class name is in :data:`DetectionConfig.object_whitelist` and
  whose confidence is above :data:`DetectionConfig.object_conf`.

This module deliberately does **not** compute face landmarks or head pose —
those fields are filled in later by Person B (`face.py`) and Person C
(`headpose.py`). The :func:`run_on_video` helper therefore writes the frozen
JSONL contract with ``face`` and ``head_pose`` set to ``null``.

All bounding boxes are ``[x, y, w, h]`` in image space with a top-left origin
and integer pixel values, matching ``schema.json``.

Usage:
    from backend.detection import Detector
    detector = Detector()
    persons, objects = detector.detect(frame)
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.config import CONFIG, DetectionConfig

logger = logging.getLogger(__name__)

# COCO class name for a person. Kept here (not in config) because it is a fixed
# property of the pretrained model, not a tunable knob.
_PERSON_CLASS_NAME: str = "person"


@dataclass(frozen=True)
class Person:
    """A single detected person.

    Attributes:
        bbox: Axis-aligned pixel box ``(x, y, w, h)``, top-left origin, ints.
        confidence: Detection confidence in ``[0.0, 1.0]``.
        source: How this student was found. ``"yolo"`` means YOLO detected the
            body directly and ``bbox`` is a real detection. ``"face_seeded"``
            means only their *face* was detected — YOLO missed the body, which
            happens constantly in crowded rows where desks and neighbours
            occlude torsos — and ``bbox`` is therefore **estimated from face
            geometry**, not measured. See :mod:`backend.students` for why this
            distinction is kept explicit rather than smoothed over.
    """

    bbox: tuple[int, int, int, int]
    confidence: float
    source: str = "yolo"


@dataclass(frozen=True)
class Obj:
    """A single detected whitelisted object.

    Attributes:
        cls: COCO class name (e.g. ``"cell phone"``, ``"laptop"``, ``"book"``).
        bbox: Axis-aligned pixel box ``(x, y, w, h)``, top-left origin, ints.
        confidence: YOLO detection confidence in ``[0.0, 1.0]``.
    """

    cls: str
    bbox: tuple[int, int, int, int]
    confidence: float


def _resolve_device(requested: str) -> str:
    """Resolve the requested compute device to a concrete torch device string.

    Falls back to CPU (with a warning) when CUDA is requested or auto-selected
    but no CUDA device is available.

    Args:
        requested: One of ``"cuda"``, ``"cpu"`` or ``"auto"`` from config.

    Returns:
        Either ``"cuda"`` or ``"cpu"``.

    Raises:
        ImportError: If PyTorch cannot be imported (ultralytics requires it).
        ValueError: If ``requested`` is not one of the accepted values.
    """
    if requested not in ("cuda", "cpu", "auto"):
        raise ValueError(
            f"Invalid device {requested!r}; expected 'cuda', 'cpu' or 'auto'."
        )

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "PyTorch is required for detection. Install it via requirements.txt."
        ) from exc

    cuda_available = torch.cuda.is_available()

    if requested == "cpu":
        return "cpu"

    if requested == "cuda":
        if not cuda_available:
            logger.warning(
                "device='cuda' requested but CUDA is unavailable; "
                "falling back to CPU. Inference will be significantly slower."
            )
            return "cpu"
        return "cuda"

    # requested == "auto"
    if cuda_available:
        logger.info("CUDA available; using GPU for detection.")
        return "cuda"
    logger.warning("CUDA unavailable; using CPU for detection.")
    return "cpu"


_UPSCALE_WARN_FACTOR: float = 2.0
"""Upscale factor above which :class:`Detector` warns about ``imgsz``.

Upscaling a frame well past its native resolution pushes apparent object size
outside the scale distribution COCO trained on, and can lose detections
entirely. Measured on ``tests/fixtures/frontal_face.jpg`` (802 px, one person
filling the frame):

==========  =======
imgsz       Persons
==========  =======
640-1440    1
1600        **0**
1920        **0**
==========  =======

It is deliberately a **warning, not a clamp**. Clamping ``imgsz`` to native
resolution was tried first and rejected on measurement: it costs far more than
it saves on the footage that matters. Across the 13 dataset images (many of
which are 1280x720, i.e. upscaled 1.5x to reach imgsz 1920):

============================  ==============  =================
                              clamp to native  imgsz 1920 as-is
============================  ==============  =================
Persons detected by YOLO      263              **331**
Students after face seeding   379              **398**
============================  ==============  =================

Upscaling *helps* crowded classroom shots, because there the objects are small
and upscaling brings them into range — the opposite of the portrait case.

What reconciles the two is not frame size but the size of the *upscale*. Every
classroom gain above was already reached at 1.5x or less (1280x720 x1.5 is
exactly 1920), while the portrait failures all sit above it. Capping the factor
at :data:`MAX_UPSCALE` therefore keeps the classroom setting untouched and
still refuses the upscale that erases a close-up. Measured both ways:

===========================  ==============  ==============
                             imgsz 1920      1.5x cap
===========================  ==============  ==============
63 classroom images          962 persons     **964**
640x480 webcam, one person   **0** persons   1
===========================  ==============  ==============

The warning below is kept for the residual cases the cap does not reach.
"""


MAX_UPSCALE: float = 1.5
"""Most a frame may be enlarged to reach ``imgsz``.

Detection is scale-sensitive: a person enlarged far past the sizes COCO trained
on stops looking like one. Above this factor the model begins finding *parts* of
a close-up person instead of the person -- on a 640x480 webcam frame the box
shrank from 460x270 at imgsz 640 to 286x260 at 1600 and vanished at 1920, which
is what puts a lone hand or a held-up sheet of paper in its own box.

1.5 is the largest factor with no measured cost, per the table in
:data:`_UPSCALE_WARN_FACTOR`.
"""


def effective_imgsz(frame_long_side: int, configured: int) -> int:
    """The inference size to actually use for a frame this size.

    Args:
        frame_long_side: The frame's longer dimension in pixels.
        configured: ``DetectionConfig.imgsz``, the size tuned for the target
            domain.

    Returns:
        ``configured``, reduced so the frame is enlarged by no more than
        :data:`MAX_UPSCALE`, rounded to the multiple of 32 the model strides on
        and never below 640 (small frames lose more from a tiny input than from
        a modest upscale).
    """
    if frame_long_side <= 0:
        return configured
    allowed = round(frame_long_side * MAX_UPSCALE / 32) * 32
    return max(640, min(configured, allowed))


def _xyxy_to_xywh(
    x1: float, y1: float, x2: float, y2: float
) -> tuple[int, int, int, int]:
    """Convert an ``(x1, y1, x2, y2)`` box to integer ``(x, y, w, h)``.

    Width and height are clamped to a minimum of 1 pixel so the box always
    satisfies the schema's ``exclusiveMinimum: 0`` constraint on ``w``/``h``.

    Args:
        x1: Left edge.
        y1: Top edge.
        x2: Right edge.
        y2: Bottom edge.

    Returns:
        The box as ``(x, y, w, h)`` with top-left origin and integer pixels.
    """
    x = round(x1)
    y = round(y1)
    w = round(x2 - x1)
    h = round(y2 - y1)
    return (max(x, 0), max(y, 0), max(w, 1), max(h, 1))


class Detector:
    """YOLOv11 wrapper that returns persons and whitelisted objects.

    The Ultralytics model is loaded once at construction time. COCO weights are
    auto-downloaded by Ultralytics on first use if not already cached.

    Attributes:
        config: The :class:`DetectionConfig` in effect for this detector.
        device: The resolved compute device (``"cuda"`` or ``"cpu"``).
    """

    def __init__(self, config: DetectionConfig | None = None) -> None:
        """Load the YOLO model and resolve the compute device.

        Args:
            config: Detection settings. Defaults to ``CONFIG.detection``.

        Raises:
            ImportError: If Ultralytics or PyTorch is not installed.
            FileNotFoundError: If a local weights path is given but missing.
            RuntimeError: If the model fails to load for any other reason.
        """
        self.config: DetectionConfig = (
            config if config is not None else CONFIG.detection
        )
        self.device: str = _resolve_device(self.config.device)

        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "Ultralytics is required for detection. "
                "Install it via requirements.txt (`pip install ultralytics`)."
            ) from exc

        weights = self.config.weights
        # A weights value ending in .pt that points at an existing path is a
        # local checkpoint; a bare name like "yolo11m.pt" is a model alias that
        # Ultralytics resolves/downloads itself. Only guard the local-path case.
        weights_path = Path(weights)
        is_local_checkpoint = (
            weights_path.suffix == ".pt" and weights_path.parent != Path(".")
        )
        if is_local_checkpoint and not weights_path.is_file():
            raise FileNotFoundError(
                f"YOLO weights not found at local path: {weights_path}"
            )

        try:
            self._model = YOLO(weights)
        except FileNotFoundError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load YOLO model from {weights!r}: {exc}"
            ) from exc

        # Map class-id -> class-name for the loaded model.
        self._names: dict[int, str] = dict(self._model.names)
        self._whitelist: frozenset[str] = frozenset(self.config.object_whitelist)
        # Per-class confidence overrides, flattened once at construction so the
        # hot loop in detect() does no lookup work beyond a dict get.
        self._class_conf: dict[str, float] = dict(self.config.object_conf_per_class)
        self._upscale_warned: bool = False

        logger.info(
            "Detector ready: weights=%s device=%s person_conf=%.2f objects=%s",
            weights,
            self.device,
            self.config.person_conf,
            tuple(self._whitelist),
        )

    def _warn_if_heavily_upscaled(self, frame: np.ndarray) -> None:
        """Warn once if ``imgsz`` upscales this frame far past its own size.

        See :data:`_UPSCALE_WARN_FACTOR` for the measurements behind the
        threshold and for why this warns instead of clamping. Logged once per
        :class:`Detector`, not per frame, so a video does not emit thousands of
        identical lines.

        Args:
            frame: The frame about to be passed to the model.
        """
        if self._upscale_warned:
            return
        native = max(frame.shape[:2])
        if native <= 0:
            return
        factor = self.config.imgsz / native
        if factor >= _UPSCALE_WARN_FACTOR:
            self._upscale_warned = True
            logger.warning(
                "imgsz=%d upscales this %dpx frame by %.1fx. That is tuned for "
                "wide classroom shots where students are small; on close-up or "
                "low-resolution input this can LOSE detections entirely "
                "(measured: a 802px single-person image drops from 1 person to "
                "0 between imgsz 1440 and 1600). Consider lowering "
                "CONFIG.detection.imgsz for this source.",
                self.config.imgsz,
                native,
                factor,
            )

    def detect(self, frame: np.ndarray) -> tuple[list[Person], list[Obj]]:
        """Run detection on a single frame.

        Args:
            frame: A ``(H, W, 3)`` BGR image as returned by OpenCV.

        Returns:
            A ``(persons, objects)`` tuple. Either list may be empty when
            nothing above threshold is detected.

        Raises:
            TypeError: If ``frame`` is not a NumPy array.
            ValueError: If ``frame`` is empty or not a 3-channel image.
        """
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"frame must be a numpy.ndarray, got {type(frame)!r}.")
        if frame.size == 0:
            raise ValueError("frame is empty (zero-size array).")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"frame must be an (H, W, 3) image, got shape {frame.shape!r}."
            )

        # Prefilter at the lowest threshold we care about so we never discard a
        # box the per-class threshold would have kept, then filter precisely.
        # Per-class overrides must be included here: a book override of 0.25
        # below the 0.35 default would otherwise be filtered out by YOLO before
        # this method ever saw it.
        min_conf = min(
            self.config.person_conf,
            self.config.object_conf,
            *(conf for _cls, conf in self.config.object_conf_per_class),
        )

        self._warn_if_heavily_upscaled(frame)
        results = self._model.predict(
            frame,
            imgsz=effective_imgsz(max(frame.shape[:2]), self.config.imgsz),
            conf=min_conf,
            iou=self.config.iou,
            device=self.device,
            verbose=False,
        )

        persons: list[Person] = []
        objects: list[Obj] = []

        if not results:
            return persons, objects

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return persons, objects

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)

        for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, clss):
            name = self._names.get(int(cls_id))
            if name is None:
                continue
            confidence = float(conf)
            bbox = _xyxy_to_xywh(float(x1), float(y1), float(x2), float(y2))

            if name == _PERSON_CLASS_NAME and confidence >= self.config.person_conf:
                persons.append(Person(bbox=bbox, confidence=confidence))
            elif name in self._whitelist:
                threshold = self._class_conf.get(name, self.config.object_conf)
                if confidence >= threshold:
                    objects.append(Obj(cls=name, bbox=bbox, confidence=confidence))
                elif confidence >= self.config.object_conf_near_person and _touches_any(
                    bbox, persons
                ):
                    # Held, and above the relaxed floor: an object on somebody's
                    # hands is a stronger proposition than the same box in empty
                    # space, and it is the only kind the action layer can use.
                    objects.append(Obj(cls=name, bbox=bbox, confidence=confidence))

        return persons, objects


def _touches_any(obj_bbox, persons) -> bool:
    """Whether an object box overlaps any detected person.

    Args:
        obj_bbox: ``(x, y, w, h)`` of the object.
        persons: This frame's detected people.

    Returns:
        ``True`` when any part of the object falls inside a person box. Judged
        against the object's own area, not IoU: a phone or a pen is tiny beside
        a person and IoU would never fire for either.
    """
    ox, oy, ow, oh = obj_bbox
    area = max(ow * oh, 1e-6)
    for person in persons:
        px, py, pw, ph = person.bbox
        ix = max(0.0, min(px + pw, ox + ow) - max(px, ox))
        iy = max(0.0, min(py + ph, oy + oh) - max(py, oy))
        if (ix * iy) / area > 0.0:
            return True
    return False


def _frame_record(
    frame_id: int,
    timestamp_ms: int,
    persons: list[Person],
    objects: list[Obj],
) -> dict:
    """Build one JSONL record in the frozen Stage 1 schema.

    Face, head-pose and posture fields are ``None`` here; Person B/C and
    ``integrate.py`` fill them in during integration.

    Args:
        frame_id: Zero-indexed frame number.
        timestamp_ms: Frame presentation time in milliseconds.
        persons: Detected persons for this frame.
        objects: Detected whitelisted objects for this frame.

    Returns:
        A JSON-serialisable dict matching ``schema.json``.
    """
    return {
        "frame_id": int(frame_id),
        "timestamp_ms": int(timestamp_ms),
        "persons": [
            {
                "track_id": None,  # filled by ByteTrack in Stage 2
                "person_id": None,  # filled by backend.identity in Stage 2
                "bbox": list(p.bbox),
                "confidence": p.confidence,
                "source": p.source,
                "face": None,  # Person B fills this in
                "head_pose": None,  # Person C fills this in
                "posture": None,  # integrate.py fills this in
                "expression": None,  # backend.expression fills this in
                "behaviour": None,  # backend.behaviour fills this in
            }
            for p in persons
        ],
        "objects": [
            {
                "cls": o.cls,
                "bbox": list(o.bbox),
                "confidence": o.confidence,
            }
            for o in objects
        ],
    }


def run_on_video(
    path: str | Path,
    out_json_path: str | Path,
    detector: Detector | None = None,
) -> int:
    """Detect on every (sampled) frame of a video and write JSONL output.

    One JSON object is written per processed frame. ``persons`` and ``objects``
    are populated; ``face`` and ``head_pose`` are ``null`` (filled in later).

    Args:
        path: Path to the input video file.
        out_json_path: Path to write the JSONL output to. Parent directories are
            created if missing.
        detector: An existing :class:`Detector` to reuse. A new one is built
            from ``CONFIG`` if omitted.

    Returns:
        The number of frames processed and written.

    Raises:
        FileNotFoundError: If the input video does not exist.
        RuntimeError: If the video cannot be opened by OpenCV.
    """
    import cv2

    src = Path(path)
    if not src.is_file():
        raise FileNotFoundError(f"Input video not found: {src}")

    if detector is None:
        detector = Detector()

    sample_rate = max(CONFIG.pipeline.sample_rate, 1)
    log_every = max(CONFIG.pipeline.log_every_frames, 1)

    capture = cv2.VideoCapture(str(src))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"OpenCV could not open video: {src}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        logger.warning("Video reports invalid FPS (%s); timestamps default to 0.", fps)
        fps = 0.0

    out_path = Path(out_json_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    frame_index = 0
    written = 0
    try:
        with out_path.open("w", encoding="utf-8") as fh:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                if frame_index % sample_rate != 0:
                    frame_index += 1
                    continue

                # Prefer the container's PTS; fall back to frame_index / fps.
                pos_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
                if pos_ms and pos_ms > 0:
                    timestamp_ms = round(pos_ms)
                elif fps > 0:
                    timestamp_ms = round(frame_index * 1000.0 / fps)
                else:
                    timestamp_ms = 0

                persons, objects = detector.detect(frame)
                record = _frame_record(frame_index, timestamp_ms, persons, objects)
                fh.write(json.dumps(record) + "\n")

                written += 1
                if written % log_every == 0:
                    logger.info(
                        "Processed %d frames (last: %d persons, %d objects).",
                        written,
                        len(persons),
                        len(objects),
                    )
                frame_index += 1
    finally:
        capture.release()

    logger.info("Wrote %d frame records to %s.", written, out_path)
    return written


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for standalone module invocation.

    Returns:
        A configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="python -m backend.detection",
        description=(
            "Run YOLOv11 person/object detection over a video and write "
            "per-frame JSONL in the frozen Stage 1 schema (face/head_pose "
            "left null)."
        ),
    )
    parser.add_argument(
        "--video",
        required=True,
        type=str,
        help="Path to the input video file.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=CONFIG.pipeline.default_output,
        help=(
            "Path to write JSONL output to "
            f"(default: {CONFIG.pipeline.default_output})."
        ),
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu", "auto"],
        default=None,
        help="Override config.detection.device for this run.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=CONFIG.log_level,
        help=f"Logging verbosity (default: {CONFIG.log_level}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: run detection on a video and write JSONL.

    Example:
        python -m backend.detection --video data/lecture.mp4 --out outputs/stage1.jsonl

    Args:
        argv: Argument list to parse. Defaults to ``sys.argv[1:]`` when ``None``.

    Returns:
        Process exit code (``0`` on success, ``1`` on a handled failure).
    """
    args = _build_arg_parser().parse_args(argv)

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    config = CONFIG.detection
    if args.device is not None:
        from dataclasses import replace

        config = replace(config, device=args.device)

    try:
        detector = Detector(config=config)
        written = run_on_video(args.video, args.out, detector=detector)
    except (FileNotFoundError, RuntimeError, ImportError, ValueError) as exc:
        logger.error("Detection run failed: %s", exc)
        return 1

    logger.info("Done: %d frames written to %s.", written, args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via CLI, not pytest
    import sys

    sys.exit(main())
