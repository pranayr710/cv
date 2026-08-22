"""Head-pose estimation for ClassGraph Stage 1 (Perception).

Wraps SixDRepNet (the ``sixdrepnet`` PyPI package) to estimate head orientation
per face and map it to a coarse gaze label. For each face bounding box (from
Person B's face module) this produces yaw, pitch and roll in degrees plus a
``gaze_label`` in ``{"teacher", "left", "right", "down", "back"}``.

Design (Person C):

* SixDRepNet weights are loaded once at construction (auto-downloaded by the
  package, or from a local file under ``WEIGHTS_DIR`` if present).
* Inference runs on GPU when available, falling back to CPU with a warning.
* The result list is **aligned index-wise** with the input ``face_bboxes``: a
  ``None`` input (person with no detected face) yields a ``None`` output, and a
  face whose pose cannot be estimated also yields ``None`` — the slot is kept.

This module does not detect faces (Person B) — it consumes face boxes.

Usage:
    from backend.headpose import HeadPoseEstimator
    estimator = HeadPoseEstimator()
    results = estimator.estimate(frame, face_bboxes)
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from backend.config import CONFIG, WEIGHTS_DIR, HeadPoseConfig

logger = logging.getLogger(__name__)

# A pixel bounding box: (x, y, w, h), top-left origin, integer pixels.
Bbox = tuple[int, int, int, int]
GazeLabel = Literal["teacher", "left", "right", "down", "back"]

# The complete, frozen set of gaze labels (matches schema.json enum).
ALLOWED_GAZE_LABELS: tuple[GazeLabel, ...] = (
    "teacher",
    "left",
    "right",
    "down",
    "back",
)


@dataclass(frozen=True)
class HeadPoseResult:
    """Head orientation for a single face.

    Attributes:
        yaw: Rotation about the vertical axis, degrees. Negative = subject's
            head turned to their left; positive = to their right.
        pitch: Rotation about the horizontal axis, degrees. Positive = looking
            down; negative = looking up/back.

            Note this is the **negation** of what SixDRepNet returns. The
            package is up-positive: in its own ``draw_axis`` the face-direction
            axis is drawn at ``y = -cos(yaw) * sin(pitch)``, and because image
            ``y`` grows downward, a positive pitch points the nose upward.
            :meth:`HeadPoseEstimator.estimate` flips the sign so this
            down-positive contract holds for every consumer.
        roll: In-plane tilt, degrees.
        gaze_label: Coarse gaze bucket derived from yaw/pitch.
    """

    yaw: float
    pitch: float
    roll: float
    gaze_label: GazeLabel


def classify_gaze(yaw: float, pitch: float, config: HeadPoseConfig) -> GazeLabel:
    """Map a (yaw, pitch) pair to a coarse gaze label.

    Yaw is first re-expressed relative to :data:`HeadPoseConfig.yaw_reference_deg`
    so the buckets mean "relative to the front of the room" rather than
    "relative to the camera". Precedence then follows the Stage 1 spec:

    1. ``|yaw| < yaw_side`` and ``|pitch| < pitch_down`` -> ``"teacher"``
    2. ``yaw  >=  yaw_side``   -> ``"right"``
    3. ``yaw  <= -yaw_side``   -> ``"left"``
    4. ``pitch >= pitch_down`` -> ``"down"``
    5. ``pitch <= pitch_back`` -> ``"back"``

    A small backward tilt in the gap ``(pitch_back, -pitch_down]`` with a near-
    frontal yaw falls back to ``"teacher"`` (treated as effectively frontal).

    Args:
        yaw: Yaw angle in degrees.
        pitch: Pitch angle in degrees.
        config: Head-pose config supplying the threshold values.

    Returns:
        One of the five allowed :data:`GazeLabel` strings.

    Raises:
        ValueError: If ``yaw`` or ``pitch`` is not finite.
    """
    if not (math.isfinite(yaw) and math.isfinite(pitch)):
        raise ValueError(f"yaw/pitch must be finite, got yaw={yaw}, pitch={pitch}.")

    # Re-express yaw relative to where "attending" actually points for this
    # camera. With a corner-mounted camera, students facing the board sit at
    # yaw ~+37 deg, not ~0, and without this every attending student is
    # bucketed as looking away. See HeadPoseConfig.yaw_reference_deg for the
    # measurement that exposed this.
    yaw = yaw - config.yaw_reference_deg

    yaw_side = config.yaw_side_threshold
    pitch_down = config.pitch_down_threshold
    pitch_back = config.pitch_back_threshold

    if abs(yaw) < yaw_side and abs(pitch) < pitch_down:
        return "teacher"
    if yaw >= yaw_side:
        return "right"
    if yaw <= -yaw_side:
        return "left"
    if pitch >= pitch_down:
        return "down"
    if pitch <= pitch_back:
        return "back"
    return "teacher"


def _resolve_device(requested: str) -> str:
    """Resolve the requested compute device to a concrete torch device string.

    Falls back to CPU (with a warning) when CUDA is requested or auto-selected
    but unavailable.

    Args:
        requested: One of ``"cuda"``, ``"cpu"`` or ``"auto"``.

    Returns:
        Either ``"cuda"`` or ``"cpu"``.

    Raises:
        ImportError: If PyTorch cannot be imported.
        ValueError: If ``requested`` is not an accepted value.
    """
    if requested not in ("cuda", "cpu", "auto"):
        raise ValueError(
            f"Invalid device {requested!r}; expected 'cuda', 'cpu' or 'auto'."
        )
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "PyTorch is required for head-pose estimation. "
            "Install it via requirements.txt."
        ) from exc

    cuda_available = torch.cuda.is_available()
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not cuda_available:
            logger.warning(
                "device='cuda' requested but CUDA is unavailable; "
                "falling back to CPU. Head-pose inference will be slower."
            )
            return "cpu"
        return "cuda"
    # auto
    if cuda_available:
        logger.info("CUDA available; using GPU for head-pose estimation.")
        return "cuda"
    logger.warning("CUDA unavailable; using CPU for head-pose estimation.")
    return "cpu"


def _coerce_bbox(bbox: Sequence[float]) -> Bbox:
    """Validate and convert an input bbox to an integer ``(x, y, w, h)`` tuple.

    Args:
        bbox: A length-4 sequence ``(x, y, w, h)``.

    Returns:
        The bbox as a tuple of ints.

    Raises:
        ValueError: If ``bbox`` is not length 4 or ``w``/``h`` are non-positive.
        TypeError: If any element is not a real number.
    """
    if len(bbox) != 4:
        raise ValueError(f"bbox must have 4 elements (x, y, w, h), got {len(bbox)}.")
    try:
        x, y, w, h = (round(float(v)) for v in bbox)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"bbox elements must be numbers: {bbox!r}") from exc
    if w <= 0 or h <= 0:
        raise ValueError(f"bbox width/height must be positive, got w={w}, h={h}.")
    return (x, y, w, h)


def _to_scalar(value: object) -> float:
    """Coerce a scalar or size-1 array-like prediction to a Python float.

    Args:
        value: A number or array-like of size >= 1 (first element is used).

    Returns:
        The value as a float.

    Raises:
        ValueError: If ``value`` is empty.
    """
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError("Empty pose value from model prediction.")
    return float(arr[0])


class HeadPoseEstimator:
    """SixDRepNet wrapper returning per-face yaw/pitch/roll and a gaze label.

    The model is created once and reused. Inference is GPU-accelerated when a
    CUDA device is available, otherwise CPU (with a warning).

    Attributes:
        config: The :class:`HeadPoseConfig` in effect.
        device: The resolved compute device (``"cuda"`` or ``"cpu"``).
    """

    def __init__(
        self, config: HeadPoseConfig | None = None, model: object | None = None
    ) -> None:
        """Load SixDRepNet weights and resolve the compute device.

        Args:
            config: Head-pose settings. Defaults to ``CONFIG.headpose``.
            model: An optional pre-built pose model exposing
                ``predict(crop) -> (pitch, yaw, roll)``. When provided, the
                ``sixdrepnet`` package is not imported (used for testing and
                custom backends). When ``None``, a ``SixDRepNet`` is built.

        Raises:
            ImportError: If ``sixdrepnet`` (or PyTorch) is not installed.
            RuntimeError: If the model fails to load.
        """
        self.config: HeadPoseConfig = config if config is not None else CONFIG.headpose
        self.device: str = _resolve_device(self.config.device)

        if model is not None:
            self._model = model
            logger.info("HeadPoseEstimator using an injected model on %s.", self.device)
            return

        try:
            from sixdrepnet import SixDRepNet
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "sixdrepnet is required for head-pose estimation. "
                "Install it via requirements.txt (`pip install sixdrepnet`)."
            ) from exc

        # The sixdrepnet package selects the device via gpu_id: a non-negative
        # index selects that CUDA device, -1 selects CPU.
        gpu_id = 0 if self.device == "cuda" else -1

        # Use a local weights file if one exists under WEIGHTS_DIR; otherwise
        # pass an empty path so the package downloads its pretrained weights.
        weights_file = WEIGHTS_DIR / self.config.weights
        dict_path = str(weights_file) if weights_file.is_file() else ""
        if dict_path:
            logger.info("Loading SixDRepNet weights from %s.", weights_file)
        else:
            logger.info(
                "No local weights at %s; SixDRepNet will download pretrained "
                "weights on first use.",
                weights_file,
            )

        try:
            self._model = SixDRepNet(gpu_id=gpu_id, dict_path=dict_path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load SixDRepNet: {exc}") from exc

        logger.info(
            "HeadPoseEstimator ready: device=%s gpu_id=%d thresholds="
            "(yaw_side=%.1f, pitch_down=%.1f, pitch_back=%.1f)",
            self.device,
            gpu_id,
            self.config.yaw_side_threshold,
            self.config.pitch_down_threshold,
            self.config.pitch_back_threshold,
        )

    def _crop_face(self, frame: np.ndarray, bbox: Bbox) -> np.ndarray | None:
        """Crop a padded face region from the frame.

        Args:
            frame: The full ``(H, W, 3)`` BGR image.
            bbox: Face box ``(x, y, w, h)`` in image pixels.

        Returns:
            The cropped BGR region, or ``None`` if the padded box has no area
            inside the image.
        """
        img_h, img_w = frame.shape[:2]
        x, y, w, h = bbox
        pad_w = round(w * self.config.crop_padding)
        pad_h = round(h * self.config.crop_padding)
        x0 = max(0, x - pad_w)
        y0 = max(0, y - pad_h)
        x1 = min(img_w, x + w + pad_w)
        y1 = min(img_h, y + h + pad_h)
        if x1 <= x0 or y1 <= y0:
            return None
        crop = frame[y0:y1, x0:x1]
        if crop.size == 0:
            return None
        return crop

    def estimate(
        self,
        frame: np.ndarray,
        face_bboxes: Sequence[Sequence[float] | None],
    ) -> list[HeadPoseResult | None]:
        """Estimate head pose for each face box, aligned index-wise.

        Args:
            frame: A ``(H, W, 3)`` BGR image as returned by OpenCV.
            face_bboxes: Face boxes ``(x, y, w, h)`` from the face module, in
                image pixels. Use ``None`` for a person with no detected face.

        Returns:
            A list the same length as ``face_bboxes``. Each element is a
            :class:`HeadPoseResult`, or ``None`` when the input was ``None`` or
            the pose could not be estimated.

        Raises:
            TypeError: If ``frame`` is not a NumPy array or ``face_bboxes`` is
                not a sequence.
            ValueError: If ``frame`` is empty/not 3-channel, or a non-``None``
                bbox is malformed.
        """
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"frame must be a numpy.ndarray, got {type(frame)!r}.")
        if frame.size == 0:
            raise ValueError("frame is empty (zero-size array).")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"frame must be an (H, W, 3) image, got shape {frame.shape!r}."
            )
        if isinstance(face_bboxes, (str, bytes)) or not isinstance(
            face_bboxes, Sequence
        ):
            raise TypeError("face_bboxes must be a sequence of bbox|None entries.")

        results: list[HeadPoseResult | None] = []
        for entry in face_bboxes:
            if entry is None:
                results.append(None)
                continue

            box = _coerce_bbox(entry)
            crop = self._crop_face(frame, box)
            if crop is None:
                logger.warning("Degenerate face crop for bbox %s; result=None.", box)
                results.append(None)
                continue

            try:
                pitch, yaw, roll = self._model.predict(crop)
            except Exception:
                logger.exception("SixDRepNet.predict failed for bbox %s", box)
                results.append(None)
                continue

            try:
                yaw_f = _to_scalar(yaw)
                # SixDRepNet reports pitch as up-positive; this module's
                # contract (see HeadPoseResult) is down-positive. Negate here so
                # every consumer, including classify_gaze, sees one convention.
                pitch_f = -_to_scalar(pitch)
                roll_f = _to_scalar(roll)
            except ValueError as exc:
                logger.warning(
                    "Bad pose values for bbox %s: %s; result=None.", box, exc
                )
                results.append(None)
                continue

            if not all(math.isfinite(v) for v in (yaw_f, pitch_f, roll_f)):
                logger.warning("Non-finite pose for bbox %s; result=None.", box)
                results.append(None)
                continue

            label = classify_gaze(yaw_f, pitch_f, self.config)
            results.append(
                HeadPoseResult(yaw=yaw_f, pitch=pitch_f, roll=roll_f, gaze_label=label)
            )

        return results


def estimate_yaw_reference(
    yaws: Sequence[float], min_samples: int = 20
) -> float | None:
    """Estimate which yaw angle corresponds to "facing the front of the room".

    Solves a problem that is invisible until it corrupts the output: the gaze
    buckets in :func:`classify_gaze` measure rotation relative to the *camera*,
    and treat yaw ~0 as attending. That only holds if the camera sits where the
    teacher and board are. On a corner-mounted camera it is simply false --
    measured on one real clip, 320 of 383 faces (84%) were labelled ``"right"``
    with a median yaw of +37 deg, because the students were correctly detected
    as facing a board off-frame to the left. The angles were right and the
    labels were wrong.

    The estimate is the **median yaw** across many students and frames. That
    works because of an assumption worth stating plainly rather than hiding:
    **most students face the front most of the time.** In a normal lesson that
    holds. It would not hold for footage of a group-work session where the class
    is deliberately turned toward each other, and it would quietly produce a
    wrong reference there -- so a value derived this way should be sanity-checked
    against one rendered frame before being trusted, not applied blind.

    The median (not the mean) is deliberate: it is unaffected by the minority of
    students who genuinely are turned away, which is exactly the population that
    would drag a mean off-target.

    Args:
        yaws: Yaw angles in degrees, pooled across students and frames of one
            session filmed by one fixed camera.
        min_samples: Refuse to estimate from fewer than this many angles. A
            reference computed from a handful of faces is noise, and a wrong
            reference is worse than none because it silently shifts every label.

    Returns:
        The estimated reference yaw in degrees, to be set as
        :data:`HeadPoseConfig.yaw_reference_deg`, or ``None`` when there are too
        few samples to estimate honestly.

    Example:
        >>> ref = estimate_yaw_reference([36.0, 38.0, 40.0] * 10)
        >>> round(ref)
        38
    """
    finite = [float(y) for y in yaws if math.isfinite(y)]
    if len(finite) < min_samples:
        logger.warning(
            "Only %d finite yaw samples (need %d) -- not estimating a yaw "
            "reference. Leaving it at its configured value.",
            len(finite),
            min_samples,
        )
        return None
    finite.sort()
    n = len(finite)
    median = (
        finite[n // 2]
        if n % 2
        else (finite[n // 2 - 1] + finite[n // 2]) / 2.0
    )
    logger.info(
        "Estimated yaw reference %.1f deg from %d samples (range %.1f to %.1f).",
        median,
        n,
        finite[0],
        finite[-1],
    )
    return median
