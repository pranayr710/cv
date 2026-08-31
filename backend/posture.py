"""Body-posture keypoints — exploratory, not part of the frozen Stage 1 contract.

Motivation: on real classroom footage, a large fraction of students have no
detectable face at all — not poorly detected, genuinely absent from the frame,
because a student bowed over a desk shows the camera the crown of their head.
No face model can recover a face that is not in view. Measured across the
project's real classroom sample set: **55% of detected persons have no face
either MediaPipe Face Mesh or MediaPipe's dedicated face_detection model can
find** (see backend/face.py's rejected-two-stage note).

This module checks whether a *different* signal — body-pose keypoints, which
do not require seeing a face — is recoverable for that faceless population.
MediaPipe Pose is the same kind of pretrained, off-the-shelf model as Face Mesh
and SixDRepNet; nothing here is trained.

Measured, faceless persons across 13 real classroom images (167 total):

    min_detection_confidence=0.2 -> 111/167 (66%) keypoints recovered
    min_detection_confidence=0.3 ->  94/167 (56%) keypoints recovered  <- used
    min_detection_confidence=0.5 (MediaPipe default) -> 46/167 (28%)
    min_detection_confidence=0.7 ->  18/167 (11%)

So: yes, keypoints are recoverable for the majority of faceless students.

What this module deliberately does NOT do: classify posture. An earlier pass
computed a single "vertical lean" feature (nose y-position relative to the
shoulder midpoint, in the person crop's normalised coordinates) and tried to
threshold it into "bowed" vs. "upright". Hand-checking a sample spanning the
full range of that feature's values showed it does not hold up — nearly every
faceless student, at every value of the feature, was visibly bowed over a
desk, because "faceless" itself already selects for "not facing the camera".
Comparing the feature's distribution between persons WITH a face (presumed
more often upright/facing camera) and WITHOUT one confirms this numerically:

    HAS a face  (n=67): mean=-0.083  median=-0.091  range=[-0.308, +0.225]
    NO face     (n=94): mean=-0.037  median=-0.044  range=[-0.303, +0.372]

The two distributions overlap almost completely (std ~0.10 for both, against a
mean difference of ~0.05). A threshold classifier here would misclassify a
large fraction of both populations and produce a confident-looking but wrong
"bowed"/"upright" label. Per the project's honesty rule ("report real metrics,
no inflation... don't fake it"), this module returns the raw geometry instead
and leaves classification to a properly trained/validated future model —
exactly the same posture as ``face.py``'s ``compute_ear``, which returns a raw
ratio and explicitly leaves the "drowsy" judgement to a downstream consumer.

Not wired into ``integrate.py``: ``schema.json`` has no posture field and every
level of it sets ``additionalProperties: false``, so adding this to the frozen
Stage 1 JSON output requires a deliberate schema change agreed with the team,
not a unilateral addition here.

Usage:
    from backend.posture import PostureAnalyzer
    with PostureAnalyzer() as analyzer:
        results = analyzer.analyze(frame, person_bboxes)
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from backend.config import CONFIG, PostureConfig

logger = logging.getLogger(__name__)

# A pixel bounding box: (x, y, w, h), top-left origin, integer pixels.
Bbox = tuple[int, int, int, int]
# A single keypoint in image pixel coordinates.
Point = tuple[float, float]


@dataclass(frozen=True)
class PostureResult:
    """Raw body-pose geometry for one person. Not a posture classification.

    All spatial values are in **image** coordinates. Every field is ``None``
    when MediaPipe Pose found no landmarks in the person's crop, or the
    relevant keypoint's visibility fell below
    :data:`PostureConfig.keypoint_min_visibility`.

    Attributes:
        keypoints_detected: Whether MediaPipe Pose found a body in the crop at
            all. ``False`` means every other field is ``None``.
        nose: The nose keypoint, or ``None``.
        left_shoulder: The subject's own left shoulder (MediaPipe landmark
            11), or ``None`` if its visibility is too low. For a person
            facing the camera this appears on the image's right side (a
            mirror effect), not the image's left.
        right_shoulder: The subject's own right shoulder (landmark 12), or
            ``None``. Appears on the image's left side when facing the
            camera.
        shoulder_mid: Midpoint of the two shoulder keypoints, or ``None`` if
            either shoulder's visibility is too low.
        hip_mid: Midpoint of the two hip keypoints, or ``None`` if either
            hip's visibility is too low.
        left_wrist: The subject's own left wrist (landmark 15), or ``None``.
        right_wrist: The subject's own right wrist (landmark 16), or ``None``.
        left_elbow: The subject's own left elbow (landmark 13), or ``None``.
        right_elbow: The subject's own right elbow (landmark 14), or ``None``.
        vertical_lean: ``nose.y - shoulder_mid.y``, normalised to the person
            crop's height (so roughly scale-invariant), or ``None`` if either
            input point is unavailable. **This is a raw, unvalidated feature,
            not a "bowed"/"upright" label** — see the module docstring for the
            measurement showing it does not cleanly separate the two. Positive
            means the nose sits below the shoulder line in the image (as seen
            from an elevated camera, this correlates loosely with leaning
            forward); do not threshold it into a semantic class without new
            validation.
        facing_direction: A unit vector `(dx, dy)` in the image plane,
            perpendicular to the shoulder line, intended to approximate which
            way the torso is turned as projected onto the image -- or
            ``None`` if either shoulder is unavailable. **The sign (which of
            the two perpendiculars) is an unconfirmed guess, not a validated
            convention.** The plan was to fix it by inspecting real images
            with people at known orientations, but the real classroom images
            available mostly show people from behind/above with heads down
            (the same camera-angle limitation documented throughout this
            project) -- the one clear, unoccluded candidate (a standing
            teacher) only had one shoulder pass the visibility threshold, so
            there was no reliable example to check the sign against. Also
            degenerate for a person facing straight at the camera regardless
            of sign (their true facing direction then runs along the depth
            axis, which has no projection onto the image plane). Because of
            this, :mod:`backend.peer_interaction` deliberately does NOT use
            this field for its core pairing logic -- it uses the shoulder
            line's undirected orientation instead, which has no front/back
            ambiguity to get wrong. Kept here as a documented, honestly
            unconfirmed feature for future validation, not a ready-to-use
            signal.
    """

    keypoints_detected: bool
    nose: Point | None
    left_shoulder: Point | None
    right_shoulder: Point | None
    shoulder_mid: Point | None
    hip_mid: Point | None
    vertical_lean: float | None
    facing_direction: Point | None
    # MediaPipe returns 33 landmarks; these were being computed on every frame
    # and discarded. Hands are what separate a raised hand from a resting one,
    # and writing from reading, so they are the difference between naming three
    # actions and naming a dozen.
    left_wrist: Point | None = None
    right_wrist: Point | None = None
    left_elbow: Point | None = None
    right_elbow: Point | None = None


def _facing_direction(
    left_shoulder: Point | None, right_shoulder: Point | None
) -> Point | None:
    """Approximate torso-facing direction, projected onto the image plane.

    Perpendicular to the shoulder line. The sign (which of the two
    perpendiculars) is an unconfirmed guess -- rendering it on real
    classroom images to pick the correct one turned out to be inconclusive
    (see :data:`PostureResult.facing_direction`'s docstring for why). Do not
    rely on the sign of this output; :mod:`backend.peer_interaction` uses the
    shoulder line's undirected orientation instead for exactly this reason.

    Args:
        left_shoulder: The subject's own left shoulder, or ``None``.
        right_shoulder: The subject's own right shoulder, or ``None``.

    Returns:
        A unit vector ``(dx, dy)``, or ``None`` if either input is ``None``
        or the shoulders are coincident (zero-length shoulder line).
    """
    if left_shoulder is None or right_shoulder is None:
        return None
    dx = right_shoulder[0] - left_shoulder[0]
    dy = right_shoulder[1] - left_shoulder[1]
    length = math.hypot(dx, dy)
    if length == 0.0:
        return None
    # Perpendicular to (dx, dy), picked as (dy, -dx) per the visual check
    # described above.
    return (dy / length, -dx / length)


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


class PostureAnalyzer:
    """MediaPipe Pose wrapper returning per-person raw keypoint geometry.

    The underlying Pose graph is created once and reused. CPU-bound, like
    :class:`backend.face.FaceAnalyzer`. Call :meth:`close` when done, or use as
    a context manager.

    Attributes:
        config: The :class:`PostureConfig` in effect for this analyzer.
    """

    def __init__(self, config: PostureConfig | None = None) -> None:
        """Create the Pose graph.

        Args:
            config: Posture settings. Defaults to ``CONFIG.posture``.

        Raises:
            ImportError: If MediaPipe is not installed, or this build lacks
                the Solutions Pose API.
            RuntimeError: If the Pose graph fails to initialise.
        """
        self.config: PostureConfig = config if config is not None else CONFIG.posture

        try:
            import mediapipe as mp
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "MediaPipe is required for posture analysis. "
                "Install it via requirements.txt (`pip install mediapipe`)."
            ) from exc

        try:
            self._mp_pose = mp.solutions.pose
        except AttributeError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "This MediaPipe build lacks the Solutions Pose API "
                "(mp.solutions.pose). Install the standard wheel pinned in "
                "requirements.txt (mediapipe>=0.10,<0.11)."
            ) from exc

        try:
            self._pose = self._mp_pose.Pose(
                static_image_mode=self.config.static_image_mode,
                model_complexity=self.config.model_complexity,
                min_detection_confidence=self.config.min_detection_confidence,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to initialise MediaPipe Pose: {exc}") from exc

        self._closed: bool = False
        logger.info(
            "PostureAnalyzer ready: model_complexity=%d min_detection_confidence=%.2f",
            self.config.model_complexity,
            self.config.min_detection_confidence,
        )

    def __enter__(self) -> PostureAnalyzer:  # noqa: PYI034 - typing.Self needs 3.11; this runs on 3.10
        """Enter the runtime context and return the analyzer."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Exit the runtime context, releasing the Pose graph."""
        self.close()

    def close(self) -> None:
        """Release the underlying MediaPipe Pose resources.

        Idempotent: calling it more than once is safe.
        """
        if not self._closed:
            self._pose.close()
            self._closed = True

    def _landmark_point(
        self, landmarks: Sequence[object], idx: int, region: Bbox
    ) -> tuple[Point | None, float]:
        """Return one landmark mapped to image coordinates, plus its visibility.

        Args:
            landmarks: MediaPipe's ``pose_landmarks.landmark`` sequence.
            idx: Index into ``landmarks``.
            region: The crop region ``(x, y, w, h)`` the landmarks are
                normalised against, in image pixels.

        Returns:
            A ``(point, visibility)`` tuple. ``point`` is ``None`` if
            visibility is below :data:`PostureConfig.keypoint_min_visibility`.
        """
        lm = landmarks[idx]
        rx, ry, rw, rh = region
        if lm.visibility < self.config.keypoint_min_visibility:
            return None, lm.visibility
        return (rx + lm.x * rw, ry + lm.y * rh), lm.visibility

    def analyze(
        self, frame: np.ndarray, person_bboxes: Sequence[Sequence[float]]
    ) -> list[PostureResult]:
        """Analyze one frame and return raw pose geometry per person box.

        Runs MediaPipe Pose on each person crop independently (mirrors
        :meth:`backend.face.FaceAnalyzer.analyze`'s per-person-crop design,
        for the same reason: a small person relative to the whole frame is
        otherwise lost before detection runs).

        Args:
            frame: A ``(H, W, 3)`` BGR image as returned by OpenCV.
            person_bboxes: Person boxes ``(x, y, w, h)``, in image pixels.

        Returns:
            A list of :class:`PostureResult`, one per entry in
            ``person_bboxes`` and in the same order.

        Raises:
            RuntimeError: If the analyzer has already been closed.
            TypeError: If ``frame`` is not a NumPy array.
            ValueError: If ``frame`` is empty/not 3-channel, or a person bbox
                is malformed.
        """
        if self._closed:
            raise RuntimeError("PostureAnalyzer is closed; create a new instance.")
        if not isinstance(frame, np.ndarray):
            raise TypeError(f"frame must be a numpy.ndarray, got {type(frame)!r}.")
        if frame.size == 0:
            raise ValueError("frame is empty (zero-size array).")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"frame must be an (H, W, 3) image, got shape {frame.shape!r}."
            )

        boxes: list[Bbox] = [_coerce_bbox(b) for b in person_bboxes]
        if not boxes:
            return []

        img_h, img_w = frame.shape[:2]
        empty = PostureResult(
            keypoints_detected=False,
            nose=None,
            left_shoulder=None,
            right_shoulder=None,
            shoulder_mid=None,
            hip_mid=None,
            vertical_lean=None,
            facing_direction=None,
        )

        results: list[PostureResult] = []
        for box in boxes:
            x, y, w, h = box
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(img_w, x + w), min(img_h, y + h)
            if x1 <= x0 or y1 <= y0:
                results.append(empty)
                continue
            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                results.append(empty)
                continue

            import cv2

            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            pose_result = self._pose.process(rgb)
            if pose_result.pose_landmarks is None:
                results.append(empty)
                continue

            region: Bbox = (x0, y0, x1 - x0, y1 - y0)
            landmarks = pose_result.pose_landmarks.landmark
            cfg = self.config

            nose, _ = self._landmark_point(landmarks, cfg.nose_idx, region)
            l_sh, _ = self._landmark_point(landmarks, cfg.left_shoulder_idx, region)
            r_sh, _ = self._landmark_point(landmarks, cfg.right_shoulder_idx, region)
            l_hip, _ = self._landmark_point(landmarks, cfg.left_hip_idx, region)
            r_hip, _ = self._landmark_point(landmarks, cfg.right_hip_idx, region)
            l_wr, _ = self._landmark_point(landmarks, cfg.left_wrist_idx, region)
            r_wr, _ = self._landmark_point(landmarks, cfg.right_wrist_idx, region)
            l_el, _ = self._landmark_point(landmarks, cfg.left_elbow_idx, region)
            r_el, _ = self._landmark_point(landmarks, cfg.right_elbow_idx, region)

            shoulder_mid = (
                ((l_sh[0] + r_sh[0]) / 2, (l_sh[1] + r_sh[1]) / 2)
                if l_sh is not None and r_sh is not None
                else None
            )
            hip_mid = (
                ((l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2)
                if l_hip is not None and r_hip is not None
                else None
            )

            vertical_lean: float | None = None
            if nose is not None and shoulder_mid is not None and region[3] > 0:
                vertical_lean = (nose[1] - shoulder_mid[1]) / region[3]

            facing_direction = _facing_direction(l_sh, r_sh)

            results.append(
                PostureResult(
                    keypoints_detected=True,
                    nose=nose,
                    left_shoulder=l_sh,
                    right_shoulder=r_sh,
                    shoulder_mid=shoulder_mid,
                    hip_mid=hip_mid,
                    vertical_lean=vertical_lean,
                    facing_direction=facing_direction,
                    left_wrist=l_wr,
                    right_wrist=r_wr,
                    left_elbow=l_el,
                    right_elbow=r_el,
                )
            )

        return results
