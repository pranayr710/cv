"""Facial-expression classification for ClassGraph (happy / sad / neutral).

Wraps EmotiEffLib (formerly HSEmotion) to label the **expression visible on a
student's face**, per face box produced by :mod:`backend.face`. Like
:mod:`backend.headpose`, it consumes face boxes and does not detect faces.

What this module reports, and what it does not
----------------------------------------------

It reports an **expression label** — the configuration of a face at a moment.
It does **not** report what a student feels, and no output of this module should
ever be presented as such. This is not hedging; it is the documented state of
the science:

* Barrett, Adolphs, Marsella, Martinez & Pollak (2019) reviewed **over 1,000
  studies** and found no scientific support for reliably inferring emotion from
  facial movement. A smile can signal submission rather than happiness.
* The EU AI Act makes inferring **emotion** from biometric data in an education
  setting a flat prohibition, not merely "high-risk".
* This project's own slides cite a deployed classroom emotion-monitoring system
  in China as a **cautionary example** of measurable student harm.

So: "expression: happy" is a defensible statement about pixels. "This student is
happy" is not, and the field is named ``expression`` throughout for that reason.

Two guardrails follow from this, and both are deliberate:

1. **Report aggregates, not individuals.** :func:`summarise_expressions` returns
   a class-level distribution and is the intended consumer of this module. A
   per-student expression label exists in the JSONL because the schema is
   per-student, but showing a live per-student label on a screen is exactly the
   pattern the research above warns against. Same rule already applied to
   attention scoring ("never a bare individual verdict").
2. **Absent beats invented.** A face too small to classify honestly returns
   ``None`` rather than a guess — see :data:`ExpressionConfig.min_face_px`.
   Classroom back-row faces can be ~20 px; a 7x upscale into the model's 224 px
   input is not a measurement.

The 8-to-3 collapse
-------------------

The model emits AffectNet's 8 classes. The project reports three, per the
project owner's decision. The mapping lives in
:data:`ExpressionConfig.expression_map` and the **full 8-class distribution is
preserved** in :attr:`ExpressionResult.distribution`, so the collapse is
auditable and reversible rather than destroying information at the earliest
stage.

Anger, Contempt, Disgust, Fear and Surprise map to ``"neutral"``, not to
``"sad"``. Folding anger into sadness would assert something the model never
predicted; mapping to neutral says "not one of the three we report", which is
true. It does mean **"neutral" is a mixed bucket**, not a claim of calm — worth
knowing before reading any distribution this produces.

Known unvalidated for this population
-------------------------------------

EmotiEffLib's models are trained on AffectNet and benchmarked on ABAW, both
Western-skewed in-the-wild data. Accuracy on South Asian classroom faces is
**unknown**, exactly as documented for SixDRepNet in
:mod:`backend.fairness_audit`. The audit harness there should be pointed at this
model once labelled data exists. Until then, no accuracy claim is made here.

Usage:
    from backend.expression import ExpressionRecognizer
    recognizer = ExpressionRecognizer()
    results = recognizer.classify(frame, face_bboxes)
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from backend.config import CONFIG, ExpressionConfig

logger = logging.getLogger(__name__)

Bbox = tuple[int, int, int, int]


@dataclass(frozen=True)
class ExpressionResult:
    """One student's visible facial expression for one frame.

    Attributes:
        label: One of :data:`ExpressionConfig.reported_labels` — ``"happy"``,
            ``"sad"`` or ``"neutral"``. A statement about the face, not about
            the student's internal state; see the module docstring.
        confidence: Probability the model assigned to the winning *raw* class,
            in ``[0, 1]``. Note this is the raw AffectNet class score, not the
            summed probability of the mapped bucket, so a confidently-``Anger``
            face reports high confidence for label ``"neutral"``.
        distribution: The full raw 8-class AffectNet distribution, class name to
            probability. Kept so the 8-to-3 mapping stays auditable.
        uncertain: ``True`` when ``confidence`` fell below
            :data:`ExpressionConfig.min_confidence`, in which case ``label`` is
            ``"uncertain"``. Measured on real classroom faces, 55% of
            predictions land here — a system that always emitted one of three
            labels would be presenting a coin-flip as a finding.
    """

    label: str
    confidence: float
    distribution: dict[str, float]
    uncertain: bool = False


def _padded_crop(
    frame: np.ndarray, face_bbox: Bbox, padding: float
) -> np.ndarray | None:
    """Crop a padded region around a face box, clamped to the frame.

    Args:
        frame: The full ``(H, W, 3)`` BGR image.
        face_bbox: The face box ``(x, y, w, h)`` in image pixels.
        padding: Padding as a fraction of box size.

    Returns:
        The cropped BGR region, or ``None`` if the region is degenerate.
    """
    img_h, img_w = frame.shape[:2]
    x, y, w, h = face_bbox
    pad_w = round(w * padding)
    pad_h = round(h * padding)
    x0 = max(0, x - pad_w)
    y0 = max(0, y - pad_h)
    x1 = min(img_w, x + w + pad_w)
    y1 = min(img_h, y + h + pad_h)
    if x1 - x0 <= 0 or y1 - y0 <= 0:
        return None
    crop = frame[y0:y1, x0:x1]
    return None if crop.size == 0 else crop


def _aligned_crop(frame: np.ndarray, kps, size: int = 224) -> np.ndarray | None:
    """Similarity-align a face to the canonical AffectNet-style layout.

    AffectNet was trained on aligned faces, so a raw box crop is out of
    distribution. Measured over 207 real classroom faces, alignment lifted
    median confidence from 0.421 to 0.481 and cut sub-0.5 predictions from 64%
    to 55%.

    Args:
        frame: The full ``(H, W, 3)`` BGR image.
        kps: The detector's 5 keypoints as a ``(5, 2)`` array in image pixels.
        size: Output square size in pixels.

    Returns:
        The aligned BGR crop, or ``None`` if alignment is unavailable (missing
        keypoints, or ``insightface`` not installed).
    """
    if kps is None:
        return None
    try:
        from insightface.utils import face_align
    except ImportError:  # pragma: no cover - environment dependent
        return None
    try:
        return face_align.norm_crop(frame, landmark=np.asarray(kps), image_size=size)
    except (ValueError, IndexError, TypeError) as exc:
        logger.debug("Face alignment failed, falling back to box crop: %s", exc)
        return None


class ExpressionRecognizer:
    """EmotiEffLib wrapper returning a mapped expression label per face box.

    The ONNX model is downloaded on first use (to the package's own cache) and
    loaded once at construction.

    Attributes:
        config: The :class:`ExpressionConfig` in effect.
    """

    def __init__(self, config: ExpressionConfig | None = None) -> None:
        """Load the expression model.

        Args:
            config: Expression settings. Defaults to ``CONFIG.expression``.

        Raises:
            ImportError: If ``emotiefflib`` is not installed.
            RuntimeError: If the model fails to load.
            ValueError: If ``expression_map`` maps to a label not in
                ``reported_labels`` — a silent typo there would otherwise emit
                an unreportable category.
        """
        self.config: ExpressionConfig = (
            config if config is not None else CONFIG.expression
        )

        self._map: dict[str, str] = dict(self.config.expression_map)
        unknown = set(self._map.values()) - set(self.config.reported_labels)
        if unknown:
            raise ValueError(
                f"expression_map targets labels not in reported_labels: "
                f"{sorted(unknown)}"
            )

        try:
            from emotiefflib.facial_analysis import EmotiEffLibRecognizer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "emotiefflib is required for expression classification. "
                "Install it via requirements.txt."
            ) from exc

        try:
            self._model = EmotiEffLibRecognizer(
                engine=self.config.engine, model_name=self.config.model_name
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load expression model "
                f"{self.config.model_name!r} ({self.config.engine}): {exc}"
            ) from exc

        # Raw class names the model actually emits, so a mapping key that never
        # matches is caught here rather than silently defaulting everything.
        raw_classes = set(self._model.idx_to_emotion_class.values())
        unmapped = raw_classes - set(self._map)
        if unmapped:
            logger.warning(
                "Model emits classes with no mapping entry: %s. These will "
                "fall back to %r.",
                sorted(unmapped),
                self.config.reported_labels[-1],
            )

        logger.info(
            "ExpressionRecognizer ready: model=%s engine=%s labels=%s "
            "min_face_px=%d",
            self.config.model_name,
            self.config.engine,
            self.config.reported_labels,
            self.config.min_face_px,
        )

    def classify(
        self,
        frame: np.ndarray,
        face_bboxes: Sequence[Sequence[float] | None],
        face_kps: Sequence[object | None] | None = None,
    ) -> list[ExpressionResult | None]:
        """Classify the expression of every given face box.

        Args:
            frame: A ``(H, W, 3)`` BGR image as returned by OpenCV.
            face_bboxes: Face boxes ``(x, y, w, h)`` in image pixels, one per
                person, with ``None`` for a person who has no detected face.
            face_kps: Optional per-person 5-point keypoints (from
                :attr:`backend.face.FaceResult.kps`), index-aligned with
                ``face_bboxes``. When present and
                :data:`ExpressionConfig.align_faces` is set, the face is aligned
                before classification, which measurably improves confidence.
                Falls back to a padded box crop when absent.

        Returns:
            A list **aligned index-wise** with ``face_bboxes``. An entry is
            ``None`` when there was no face, the face was smaller than
            ``min_face_px``, or the crop was degenerate — absent rather than
            guessed.

        Raises:
            TypeError: If ``frame`` is not a NumPy array.
            ValueError: If ``frame`` is empty or not a 3-channel image.
        """
        import cv2

        if not isinstance(frame, np.ndarray):
            raise TypeError(f"frame must be a numpy.ndarray, got {type(frame)!r}.")
        if frame.size == 0:
            raise ValueError("frame is empty (zero-size array).")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"frame must be an (H, W, 3) image, got shape {frame.shape!r}."
            )

        results: list[ExpressionResult | None] = []
        too_small = 0
        kps_list = list(face_kps) if face_kps is not None else [None] * len(face_bboxes)
        if len(kps_list) != len(face_bboxes):
            raise ValueError(
                f"face_kps has {len(kps_list)} entries but face_bboxes has "
                f"{len(face_bboxes)}; they must be index-aligned."
            )
        for bbox, kps in zip(face_bboxes, kps_list):
            if bbox is None:
                results.append(None)
                continue
            x, y, w, h = (round(float(v)) for v in bbox)
            if min(w, h) < self.config.min_face_px:
                too_small += 1
                results.append(None)
                continue
            crop = None
            if self.config.align_faces:
                crop = _aligned_crop(frame, kps)
            if crop is None:
                crop = _padded_crop(frame, (x, y, w, h), self.config.crop_padding)
            if crop is None:
                results.append(None)
                continue
            # EmotiEffLib expects RGB; OpenCV frames are BGR.
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            try:
                _labels, scores = self._model.predict_emotions(rgb, logits=False)
            except (RuntimeError, ValueError, IndexError) as exc:
                # One unusual crop must not abort a whole frame of students.
                # Narrow to the failures ONNX Runtime and numpy actually raise on
                # a bad input, so a genuine bug still surfaces rather than being
                # swallowed as "no expression".
                logger.warning("Expression inference failed for one face: %s", exc)
                results.append(None)
                continue
            results.append(self._to_result(np.asarray(scores).reshape(-1)))

        if too_small:
            logger.debug(
                "%d of %d faces were smaller than min_face_px=%d and were left "
                "unclassified rather than upscaled.",
                too_small,
                sum(1 for b in face_bboxes if b is not None),
                self.config.min_face_px,
            )
        return results

    def _to_result(self, scores: np.ndarray) -> ExpressionResult:
        """Turn a raw score vector into a mapped :class:`ExpressionResult`.

        Args:
            scores: The model's per-class probabilities, in model class order.

        Returns:
            The mapped result, carrying the full raw distribution.
        """
        idx_to_class = self._model.idx_to_emotion_class
        distribution = {
            idx_to_class[i]: float(scores[i])
            for i in range(min(len(scores), len(idx_to_class)))
        }
        raw_label = max(distribution, key=distribution.get)
        confidence = distribution[raw_label]
        # Abstain rather than force a label the model is not confident about.
        # A wrong confident answer is worse than an admitted unknown, and this
        # is the mechanism that makes that concrete rather than aspirational.
        if confidence < self.config.min_confidence:
            return ExpressionResult(
                label=self.config.uncertain_label,
                confidence=confidence,
                distribution=distribution,
                uncertain=True,
            )
        # Unmapped classes fall back to the last reported label ("neutral"),
        # which the constructor warns about at load time.
        return ExpressionResult(
            label=self._map.get(raw_label, self.config.reported_labels[-1]),
            confidence=confidence,
            distribution=distribution,
        )


def summarise_expressions(
    results: Sequence[ExpressionResult | None],
    config: ExpressionConfig | None = None,
) -> dict[str, object]:
    """Aggregate per-student expressions into a class-level summary.

    **This is the intended way to consume this module.** A per-student
    expression label is available in the JSONL because the schema is
    per-student, but the reporting default is deliberately the class-level
    trend — the same guardrail :mod:`backend.attention` applies to attention
    ("never a bare individual verdict"), and the specific thing that separates
    this system from the deployed classroom-monitoring system this project's
    own slides criticise.

    Args:
        results: Per-student expression results, ``None`` where unavailable.
        config: Expression settings. Defaults to ``CONFIG.expression``.

    Returns:
        A dict with ``students`` (total slots), ``classified`` (how many yielded
        a label), ``unavailable`` (how many did not — usually a face too small
        or absent), ``counts`` per reported label, and ``shares`` per reported
        label as a fraction of ``classified``. ``shares`` is empty when nothing
        was classified, rather than reporting zeros that look like measurements.
    """
    cfg = config if config is not None else CONFIG.expression
    classified = [r for r in results if r is not None]
    counts = {label: 0 for label in cfg.reported_labels}
    for result in classified:
        counts[result.label] = counts.get(result.label, 0) + 1
    shares = (
        {label: counts[label] / len(classified) for label in counts}
        if classified
        else {}
    )
    return {
        "students": len(results),
        "classified": len(classified),
        "unavailable": len(results) - len(classified),
        "counts": counts,
        "shares": shares,
    }


class ExpressionWindow:
    """Rolling per-student aggregation of expression over recent frames.

    **This is the single largest accuracy lever available without new labels,
    and it costs no extra inference.** A single frame's expression is noise: on
    one real classroom clip, single-frame labels flipped between *consecutive*
    frames on 6.8% of steps — a student's expression does not genuinely change
    five times a second, so that flipping is measurement error. Averaging the
    raw class distribution over a 9-frame window cut it to 1.4%, a 5x reduction.

    Averaging the **distribution** rather than voting on labels is deliberate:
    it lets many weak-but-consistent frames outvote one confidently-wrong frame,
    which label voting cannot do.

    This is the same "never judge a single frame" principle
    :mod:`backend.attention` applies to gaze (15-second rolling window), and the
    mechanism RDFER — base paper 1 — uses for exactly this problem: separating a
    momentary facial movement from a sustained state.

    Requires a stable ``track_id`` per student, which Stage 2 (ByteTrack)
    supplies. Students with ``track_id`` ``None`` cannot be aggregated and
    should be reported from their single-frame result, or not at all.

    Usage:
        window = ExpressionWindow()
        window.update(track_id, result)          # once per student per frame
        smoothed = window.smoothed(track_id)     # aggregated result
    """

    def __init__(self, config: ExpressionConfig | None = None) -> None:
        """Create an empty window.

        Args:
            config: Expression settings. Defaults to ``CONFIG.expression``;
                ``window_frames`` sets how many frames are retained per student.
        """
        self.config: ExpressionConfig = (
            config if config is not None else CONFIG.expression
        )
        self._history: dict[int, list[dict[str, float]]] = {}

    def update(self, track_id: int, result: ExpressionResult | None) -> None:
        """Record one frame's raw distribution for one student.

        Frames where the expression was unavailable are skipped rather than
        recorded as zeros: a missing measurement is not evidence of a neutral
        face, and averaging zeros in would drag every student toward whichever
        label happens to sit at the distribution's centre.

        Args:
            track_id: The student's stable track id from Stage 2.
            result: This frame's result, or ``None`` if unavailable.
        """
        if result is None or not result.distribution:
            return
        history = self._history.setdefault(track_id, [])
        history.append(dict(result.distribution))
        excess = len(history) - max(1, int(self.config.window_frames))
        if excess > 0:
            del history[:excess]

    def smoothed(self, track_id: int) -> ExpressionResult | None:
        """Return the window-averaged expression for one student.

        Args:
            track_id: The student's stable track id.

        Returns:
            An :class:`ExpressionResult` whose ``distribution`` is the mean of
            the retained frames, with the abstention rule from
            :data:`ExpressionConfig.min_confidence` applied to that mean — so a
            student whose frames disagree stays ``"uncertain"`` instead of being
            resolved by whichever frame happened to be last. ``None`` if nothing
            has been recorded for this student.
        """
        history = self._history.get(track_id)
        if not history:
            return None

        classes = sorted({name for frame in history for name in frame})
        mean = {
            name: sum(frame.get(name, 0.0) for frame in history) / len(history)
            for name in classes
        }
        raw_label = max(mean, key=mean.get)
        confidence = mean[raw_label]
        if confidence < self.config.min_confidence:
            return ExpressionResult(
                label=self.config.uncertain_label,
                confidence=confidence,
                distribution=mean,
                uncertain=True,
            )
        return ExpressionResult(
            label=dict(self.config.expression_map).get(
                raw_label, self.config.reported_labels[-1]
            ),
            confidence=confidence,
            distribution=mean,
        )

    def frames_held(self, track_id: int) -> int:
        """How many frames are currently retained for one student.

        Useful for suppressing a reading until the window has filled: a
        "smoothed" label built from two frames is barely smoothed at all.

        Args:
            track_id: The student's stable track id.

        Returns:
            The number of retained frames, ``0`` if the student is unknown.
        """
        return len(self._history.get(track_id, ()))

    def known_track_ids(self) -> list[int]:
        """Track ids with at least one recorded frame, in insertion order."""
        return list(self._history)
