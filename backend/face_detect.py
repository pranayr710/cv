"""SCRFD face detection for ClassGraph Stage 1 (Perception).

Supplies face bounding boxes for :mod:`backend.face`, which then runs MediaPipe
Face Mesh on each box for landmarks and EAR. This module does **not** compute
landmarks, EAR, head pose or identity.

Why this module exists at all
----------------------------

Face detection was the pipeline's worst-performing stage and the hard ceiling
on everything that needs a face (gaze, EAR, and — once added — facial
expression). The original path used MediaPipe Face Mesh's own internal
detector on each person crop. Measured on ``dataset/img01.jpg`` (1920x1088
classroom CCTV, ~50 students visible, 30 persons found by YOLO):

===========================================  ============
Approach                                     Faces found
===========================================  ============
Face Mesh on whole frame (original bug)      0
Face Mesh per person crop (previous fix)     10
**SCRFD on whole frame (this module)**       **48**
===========================================  ============

The root cause was model fit, not tuning. MediaPipe's detector is BlazeFace,
built for short-range selfie-distance faces on a phone. SCRFD was designed and
trained for the WIDER FACE "hard" split, where ~79% of faces are smaller than
32x32 px and ~52% smaller than 16x16 px — which describes a classroom's back
rows precisely. No amount of threshold tuning closes a 10-vs-48 gap.

A consequence worth noting: SCRFD runs on the **whole frame**, so the
per-person-crop trick that rescued the MediaPipe path is not needed here and
is not used. Faces are bound to person boxes afterwards, by containment, in
:mod:`backend.face`.

A second consequence, which changes where the pipeline's bottleneck now sits:
SCRFD finds *more faces than YOLO finds persons* on the same frame (48 vs 30).
Person detection, not face detection, is now the weaker stage. Faces with no
containing person box are reported by :func:`FaceDetector.detect` and counted
by ``tools/bench_faces.py`` rather than silently dropped.

Usage:
    from backend.face_detect import FaceDetector
    detector = FaceDetector()
    boxes = detector.detect(frame)   # [(x, y, w, h), ...] + scores
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from backend.config import CONFIG, FaceConfig

logger = logging.getLogger(__name__)

# A pixel bounding box: (x, y, w, h), top-left origin, integer pixels.
Bbox = tuple[int, int, int, int]


@dataclass(frozen=True)
class DetectedFace:
    """One detected face, before any landmark or identity work.

    Attributes:
        bbox: Face box ``(x, y, w, h)`` in image pixels, top-left origin.
        score: SCRFD detection confidence in ``[0.0, 1.0]``.
        kps: SCRFD's 5 keypoints (both eyes, nose, both mouth corners) as an
            ``(5, 2)`` array in image pixels, or ``None`` if the model did not
            return them. Used to **align** the face before expression
            classification: AffectNet was trained on aligned faces, and feeding
            an unaligned box crop instead measurably costs confidence (median
            0.421 unaligned vs 0.481 aligned over 207 real classroom faces;
            share of predictions below 0.5 confidence 64% vs 55%).
        embedding: A 512-d L2-normalised ArcFace embedding (InsightFace's
            ``normed_embedding``), or ``None`` when
            :data:`FaceConfig.enable_recognition` is off. Consumed only by
            :mod:`backend.identity` for within-video re-identification — see
            that module for the scope and privacy boundary (embeddings are
            never written to output or kept past one video).
    """

    bbox: Bbox
    score: float
    kps: np.ndarray | None = None
    embedding: np.ndarray | None = None


def _clamped_xywh(
    x1: float, y1: float, x2: float, y2: float, img_w: int, img_h: int
) -> Bbox:
    """Convert an ``(x1, y1, x2, y2)`` box to integer ``(x, y, w, h)``, clamped.

    SCRFD can return boxes extending slightly past the frame edge for faces at
    the border, so the box is clamped to the image before use.

    Args:
        x1: Left edge.
        y1: Top edge.
        x2: Right edge.
        y2: Bottom edge.
        img_w: Image width in pixels.
        img_h: Image height in pixels.

    Returns:
        The box as ``(x, y, w, h)``, inside the image, with ``w``/``h`` at
        least 1 pixel to satisfy the schema's positive-size constraint.
    """
    cx1 = max(0, min(round(x1), img_w - 1))
    cy1 = max(0, min(round(y1), img_h - 1))
    cx2 = max(0, min(round(x2), img_w))
    cy2 = max(0, min(round(y2), img_h))
    return (cx1, cy1, max(1, cx2 - cx1), max(1, cy2 - cy1))


def _effective_det_size(
    configured: tuple[int, int], frame_h: int, frame_w: int
) -> tuple[int, int]:
    """Clamp the configured SCRFD input size to the frame's own resolution.

    Same failure mode as :func:`backend.detection._effective_imgsz`, and found
    by the same regression test: upscaling past native resolution costs
    detections rather than gaining them.

    ==========================  ====================  =====
    Image                       det_size              Faces
    ==========================  ====================  =====
    frontal_face.jpg (802 px)   320 / 640 / 1024      1
    frontal_face.jpg (802 px)   1600 (unclamped)      **0**
    ==========================  ====================  =====

    Args:
        configured: ``FaceConfig.scrfd_det_size`` as ``(w, h)``.
        frame_h: Frame height in pixels.
        frame_w: Frame width in pixels.

    Returns:
        A square ``(size, size)`` to hand SCRFD: the smaller of the configured
        size and the frame's long side, rounded up to a multiple of 32 and never
        below 32.
    """
    target = max(configured)
    native = max(frame_h, frame_w)
    if native >= target:
        return (target, target)
    rounded = ((native + 31) // 32) * 32
    size = max(32, min(target, rounded))
    return (size, size)


class FaceDetector:
    """InsightFace SCRFD wrapper returning face boxes for a whole frame.

    The ONNX model pack is loaded once at construction. InsightFace
    auto-downloads it to ``~/.insightface`` on first use, the same way
    Ultralytics fetches YOLO weights.

    Attributes:
        config: The :class:`FaceConfig` in effect for this detector.
    """

    def __init__(self, config: FaceConfig | None = None) -> None:
        """Load the SCRFD model.

        Args:
            config: Face settings. Defaults to ``CONFIG.face``.

        Raises:
            ImportError: If ``insightface`` / ``onnxruntime`` is not installed.
            RuntimeError: If the model pack fails to load.
        """
        self.config: FaceConfig = config if config is not None else CONFIG.face

        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "insightface (and onnxruntime) are required for SCRFD face "
                "detection. Install them via requirements.txt, or set "
                "CONFIG.face.detector='mediapipe' to use the older, much "
                "weaker detector."
            ) from exc

        try:
            # "recognition" additionally loads the pack's bundled ArcFace
            # embedding model (w600k_r50.onnx) when enabled — see
            # FaceConfig.enable_recognition for why this is no longer always
            # excluded, and backend/identity.py for the privacy boundary on
            # what happens to the embeddings it produces. landmark/genderage
            # sub-models remain excluded; nothing in this project uses them.
            modules = ["detection"]
            if self.config.enable_recognition:
                modules.append("recognition")
            self._app = FaceAnalysis(
                name=self.config.scrfd_model_pack,
                allowed_modules=modules,
            )
            # Prepared again on the first frame if that frame is smaller than
            # the configured det_size (see _effective_det_size). InsightFace
            # fixes det_size at prepare() time, not per call, so the size in
            # effect is tracked here and only re-prepared when it must change —
            # once per resolution, not per frame.
            self._prepared_size: tuple[int, int] | None = None
            self._prepare(tuple(self.config.scrfd_det_size))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load SCRFD model pack "
                f"{self.config.scrfd_model_pack!r}: {exc}"
            ) from exc

        logger.info(
            "FaceDetector ready: SCRFD pack=%s det_size=%s det_thresh=%.2f",
            self.config.scrfd_model_pack,
            tuple(self.config.scrfd_det_size),
            self.config.scrfd_det_thresh,
        )

    def _prepare(self, det_size: tuple[int, int]) -> None:
        """Configure SCRFD's input size, skipping a no-op re-prepare.

        Args:
            det_size: The ``(w, h)`` input size to run SCRFD at.
        """
        if self._prepared_size == det_size:
            return
        self._app.prepare(
            ctx_id=0,
            det_size=det_size,
            det_thresh=self.config.scrfd_det_thresh,
        )
        self._prepared_size = det_size

    def detect(self, frame: np.ndarray) -> list[DetectedFace]:
        """Detect every face in a whole frame.

        Args:
            frame: A ``(H, W, 3)`` BGR image as returned by OpenCV.

        Returns:
            Detected faces, sorted by descending confidence so downstream
            greedy assignment sees the strongest candidates first. Empty when
            no face clears ``scrfd_det_thresh``.

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

        img_h, img_w = frame.shape[:2]
        self._prepare(
            _effective_det_size(tuple(self.config.scrfd_det_size), img_h, img_w)
        )
        faces = self._app.get(frame)

        detected = [
            DetectedFace(
                bbox=_clamped_xywh(
                    float(f.bbox[0]),
                    float(f.bbox[1]),
                    float(f.bbox[2]),
                    float(f.bbox[3]),
                    img_w,
                    img_h,
                ),
                score=float(f.det_score),
                kps=(
                    np.asarray(f.kps, dtype=np.float32)
                    if getattr(f, "kps", None) is not None
                    else None
                ),
                embedding=(
                    np.asarray(f.normed_embedding, dtype=np.float32)
                    if self.config.enable_recognition
                    and getattr(f, "embedding", None) is not None
                    else None
                ),
            )
            for f in faces
        ]
        detected.sort(key=lambda d: -d.score)
        logger.debug("SCRFD found %d faces in a %dx%d frame.", len(detected), img_w, img_h)
        return detected
