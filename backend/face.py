"""Face detection + landmark + EAR analysis for ClassGraph Stage 1.

Produces, per person, a face bounding box, the 468 canonical face-mesh
landmarks (in **image** coordinates, not crop coordinates), and an
eye-aspect-ratio (EAR) value used downstream for drowsiness/attention.

Two detector backends, selected by :data:`FaceConfig.detector`
------------------------------------------------------------

**``"scrfd"`` (default).** :mod:`backend.face_detect` runs SCRFD over the
**whole frame** to find face boxes; MediaPipe Face Mesh then runs on each face
box (padded by :data:`FaceConfig.scrfd_landmark_padding`) purely to fit
landmarks and compute EAR. Face Mesh's own detector is bypassed entirely.

**``"mediapipe"``.** The original path, kept only so the two can be compared
(``tools/bench_faces.py``): one Face Mesh inference per **person crop**, using
Face Mesh's internal BlazeFace detector.

Why the default changed, measured on ``dataset/img01.jpg`` (1920x1088 classroom
CCTV, ~50 students visible, 30 persons found by YOLO):

===========================================  ============
Approach                                     Faces found
===========================================  ============
Face Mesh on whole frame (original bug)      0
Face Mesh per person crop (previous fix)     10
SCRFD on whole frame (current default)       **48**
===========================================  ============

This supersedes the earlier conclusion recorded here that ~42-45% was the
ceiling for this camera angle. That figure was a property of **BlazeFace**, not
of the camera: BlazeFace is built for short-range, selfie-distance faces, while
SCRFD was trained for the WIDER FACE "hard" split where ~79% of faces are under
32x32 px. The previous rejection of BlazeFace-as-pre-detector (58 faces, worse
than 98) remains valid and is *why* a different detector family was needed —
the mistake was concluding the angle was at fault rather than the model.

A real remaining limit: a student bowed flat over a desk shows the camera the
crown of their head. SCRFD recovers many such cases (it detects heavily
downturned heads), but not all, and no face model can recover a face that is
not in frame. :mod:`backend.posture` exists for that population.

Assignment
----------

* Faces are bound to person boxes greedily by containment, highest detector
  score first, so an overlapping pair of person boxes cannot claim the same
  physical face twice (see :data:`FaceConfig.assign_min_containment` and
  :data:`FaceConfig.duplicate_face_iou`).
* The returned list is **aligned index-wise** with ``person_bboxes``: a person
  with no matching face keeps its slot with all fields ``None``.
* SCRFD now finds *more faces than YOLO finds persons* (48 vs 30 on img01), so
  some detected faces have no containing person box and are dropped by
  assignment. That is a **person-detection** shortfall, not a face one; it is
  counted and reported by ``tools/bench_faces.py`` rather than hidden.

This module does not compute head pose (Person C) or track identities
(Stage 2). Landmarks are the canonical 468 mesh points; the 10 iris points that
``refine_landmarks=True`` adds are dropped to match the frozen schema.

Usage:
    from backend.face import FaceAnalyzer
    with FaceAnalyzer() as analyzer:
        results = analyzer.analyze(frame, person_bboxes)
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from backend.config import CONFIG, FaceConfig

logger = logging.getLogger(__name__)

# A pixel bounding box: (x, y, w, h), top-left origin, integer pixels.
Bbox = tuple[int, int, int, int]
# A single landmark in image pixel coordinates.
Point = tuple[float, float]


@dataclass(frozen=True)
class FaceResult:
    """Per-person face analysis result.

    All spatial values are in **image** coordinates. Every field is ``None``
    when no face was matched to the corresponding person bounding box; the slot
    is still kept so the result list stays aligned with the input.

    Attributes:
        face_bbox: Face box ``(x, y, w, h)`` in image pixels, or ``None``.
        landmarks: List of ``num_landmarks`` ``(x, y)`` points in image pixels,
            or ``None``.
        ear: Mean eye-aspect-ratio over both eyes, or ``None``.
        kps: The detector's 5 keypoints for this face (SCRFD only), carried
            through so :mod:`backend.expression` can align the face before
            classifying it — alignment measurably improves confidence. ``None``
            under the mediapipe backend, which produces no such keypoints.

    Note:
        With the SCRFD backend, ``face_bbox`` can be present while
        ``landmarks`` and ``ear`` are ``None``: SCRFD found the face but Face
        Mesh could not fit a mesh to it. That combination is useful, not a
        failure — head pose only needs the box, so such a person still yields a
        gaze signal. Under the old MediaPipe-only path this case was
        unrepresentable, and every mesh failure silently became "no face".
    """

    face_bbox: Bbox | None
    landmarks: list[Point] | None
    ear: float | None
    kps: object | None = None


def _euclidean(a: Point, b: Point) -> float:
    """Return the Euclidean distance between two 2-D points.

    Args:
        a: First point ``(x, y)``.
        b: Second point ``(x, y)``.

    Returns:
        The straight-line distance between ``a`` and ``b``.
    """
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _single_eye_ear(landmarks: Sequence[Point], idx: Sequence[int]) -> float | None:
    """Compute the 6-point EAR for one eye.

    Uses the Soukupova & Cech formula::

        EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)

    where ``idx`` lists the landmark indices in ``(p1, p2, p3, p4, p5, p6)``
    order (p1/p4 the horizontal eye corners).

    Args:
        landmarks: All face landmarks as ``(x, y)`` points in image pixels.
        idx: The six landmark indices for this eye, in P1..P6 order.

    Returns:
        The EAR for this eye, or ``None`` if an index is out of range or the
        horizontal eye width is zero (degenerate — cannot normalise).

    Raises:
        ValueError: If ``idx`` does not contain exactly six indices.
    """
    if len(idx) != 6:
        raise ValueError(f"Expected 6 eye indices, got {len(idx)}.")
    if any(i < 0 or i >= len(landmarks) for i in idx):
        return None

    p1, p2, p3, p4, p5, p6 = (landmarks[i] for i in idx)
    horizontal = _euclidean(p1, p4)
    if horizontal == 0.0:
        return None
    vertical = _euclidean(p2, p6) + _euclidean(p3, p5)
    return vertical / (2.0 * horizontal)


def compute_ear(
    landmarks: Sequence[Point],
    left_idx: Sequence[int],
    right_idx: Sequence[int],
) -> float | None:
    """Compute the mean eye-aspect-ratio over both eyes.

    Args:
        landmarks: All face landmarks as ``(x, y)`` points in image pixels.
        left_idx: The six left-eye landmark indices in P1..P6 order.
        right_idx: The six right-eye landmark indices in P1..P6 order.

    Returns:
        The average EAR over whichever eyes could be computed, or ``None`` if
        neither eye yields a valid value.

    Raises:
        ValueError: If either index tuple does not contain exactly six indices.
    """
    left = _single_eye_ear(landmarks, left_idx)
    right = _single_eye_ear(landmarks, right_idx)
    valid = [e for e in (left, right) if e is not None]
    if not valid:
        return None
    return float(sum(valid) / len(valid))


def _bbox_from_points(points: Sequence[Point], img_w: int, img_h: int) -> Bbox:
    """Compute the tight, image-clamped bounding box of a set of points.

    Args:
        points: The landmark points as ``(x, y)`` image pixels.
        img_w: Image width in pixels.
        img_h: Image height in pixels.

    Returns:
        The bounding box ``(x, y, w, h)`` clamped to the image, with ``w`` and
        ``h`` at least 1 pixel.

    Raises:
        ValueError: If ``points`` is empty.
    """
    if not points:
        raise ValueError("Cannot compute a bbox from an empty point set.")
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0 = max(0, math.floor(min(xs)))
    y0 = max(0, math.floor(min(ys)))
    x1 = min(img_w, math.ceil(max(xs)))
    y1 = min(img_h, math.ceil(max(ys)))
    return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))


def _containment(inner: Bbox, outer: Bbox) -> float:
    """Fraction of ``inner``'s area that lies inside ``outer``.

    Args:
        inner: The box whose containment is measured (e.g. a face box).
        outer: The containing box (e.g. a person box).

    Returns:
        A value in ``[0.0, 1.0]``: intersection area divided by ``inner`` area.
    """
    ix0, iy0, iw, ih = inner
    ox0, oy0, ow, oh = outer
    ix1, iy1 = ix0 + iw, iy0 + ih
    ox1, oy1 = ox0 + ow, oy0 + oh

    inter_w = max(0, min(ix1, ox1) - max(ix0, ox0))
    inter_h = max(0, min(iy1, oy1) - max(iy0, oy0))
    inter = inter_w * inter_h

    inner_area = iw * ih
    if inner_area <= 0:
        return 0.0
    return inter / inner_area


def _iou(a: Bbox, b: Bbox) -> float:
    """Intersection-over-union of two boxes.

    Args:
        a: First box ``(x, y, w, h)``.
        b: Second box ``(x, y, w, h)``.

    Returns:
        A value in ``[0.0, 1.0]``; ``0.0`` when either box has no area.
    """
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    inter_w = max(0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
    inter_h = max(0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return inter / union


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


class FaceAnalyzer:
    """MediaPipe Face Mesh wrapper returning per-person landmarks and EAR.

    The underlying Face Mesh graph is created once and reused. It is CPU-bound,
    which is expected and fine (MediaPipe does not use the GPU here). Call
    :meth:`close` when done, or use the analyzer as a context manager.

    Attributes:
        config: The :class:`FaceConfig` in effect for this analyzer.
    """

    def __init__(self, config: FaceConfig | None = None) -> None:
        """Create the Face Mesh graph.

        Args:
            config: Face settings. Defaults to ``CONFIG.face``.

        Raises:
            ImportError: If MediaPipe is not installed.
            RuntimeError: If the Face Mesh graph fails to initialise.
        """
        self.config: FaceConfig = config if config is not None else CONFIG.face

        try:
            import mediapipe as mp
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "MediaPipe is required for face analysis. "
                "Install it via requirements.txt (`pip install mediapipe`)."
            ) from exc

        # The Face Mesh solution lives in the legacy Solutions API, which ships
        # with the standard mediapipe 0.10 wheel. Some stripped builds expose
        # only the newer Tasks API; fail loudly rather than silently degrade.
        try:
            self._mp_face_mesh = mp.solutions.face_mesh
        except AttributeError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "This MediaPipe build lacks the Solutions Face Mesh API "
                "(mp.solutions.face_mesh). Install the standard wheel pinned in "
                "requirements.txt (mediapipe>=0.10,<0.11)."
            ) from exc
        try:
            self._mesh = self._mp_face_mesh.FaceMesh(
                static_image_mode=self.config.static_image_mode,
                max_num_faces=self.config.max_num_faces,
                refine_landmarks=self.config.refine_landmarks,
                min_detection_confidence=self.config.min_detection_confidence,
                min_tracking_confidence=self.config.min_tracking_confidence,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialise MediaPipe Face Mesh: {exc}"
            ) from exc

        # SCRFD is built lazily-but-eagerly here (not per-frame) so a missing
        # insightface install fails at construction with a clear message rather
        # than mid-run on the first frame.
        self._face_detector = None
        if self.config.detector == "scrfd":
            from backend.face_detect import FaceDetector

            self._face_detector = FaceDetector(self.config)

        self._closed: bool = False
        logger.info(
            "FaceAnalyzer ready: detector=%s max_faces=%d refine=%s num_landmarks=%d",
            self.config.detector,
            self.config.max_num_faces,
            self.config.refine_landmarks,
            self.config.num_landmarks,
        )

    def __enter__(self) -> FaceAnalyzer:
        """Enter the runtime context and return the analyzer."""
        return self

    def __exit__(self, *exc: object) -> None:
        """Exit the runtime context, releasing the Face Mesh graph."""
        self.close()

    def close(self) -> None:
        """Release the underlying MediaPipe Face Mesh resources.

        Idempotent: calling it more than once is safe.
        """
        if not self._closed:
            self._mesh.close()
            self._closed = True

    def _padded_region(self, person_box: Bbox, img_w: int, img_h: int) -> Bbox:
        """Expand a person box by the configured padding, clamped to the image.

        Args:
            person_box: The person box ``(x, y, w, h)`` in image pixels.
            img_w: Image width in pixels.
            img_h: Image height in pixels.

        Returns:
            The padded region ``(x, y, w, h)``, clamped to the image bounds.
            Width/height may be ``0`` when the box lies entirely outside the
            image.
        """
        x, y, w, h = person_box
        pad_w = round(w * self.config.person_crop_padding)
        pad_h = round(h * self.config.person_crop_padding)
        x0 = max(0, x - pad_w)
        y0 = max(0, y - pad_h)
        x1 = min(img_w, x + w + pad_w)
        y1 = min(img_h, y + h + pad_h)
        return (x0, y0, max(0, x1 - x0), max(0, y1 - y0))

    def _detect_faces(
        self, frame: np.ndarray, region: Bbox
    ) -> list[tuple[Bbox, list[Point], float | None]]:
        """Run Face Mesh on one crop and return faces in **image** coordinates.

        Cropping before inference is the whole point: MediaPipe downscales its
        input for face detection, so a face that is small relative to the full
        frame is lost. Within a person crop the same face is large enough to
        survive that downscale.

        Args:
            frame: The full ``(H, W, 3)`` BGR image.
            region: The already-padded crop region ``(x, y, w, h)`` in image
                pixels.

        Returns:
            A list of ``(face_bbox, landmarks, ear)`` for every face detected in
            the crop, with all coordinates mapped back to image space. Empty if
            the crop is degenerate or no face is found.
        """
        import cv2

        img_h, img_w = frame.shape[:2]
        rx, ry, rw, rh = region
        if rw <= 0 or rh <= 0:
            return []
        crop = frame[ry : ry + rh, rx : rx + rw]
        if crop.size == 0:
            return []

        # MediaPipe expects RGB; OpenCV frames are BGR.
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        results = self._mesh.process(rgb)

        multi = getattr(results, "multi_face_landmarks", None)
        if not multi:
            return []

        n = self.config.num_landmarks
        faces: list[tuple[Bbox, list[Point], float | None]] = []
        for face_landmarks in multi:
            # Landmarks are normalised to the crop; map them back to the image.
            pts: list[Point] = [
                (rx + lm.x * rw, ry + lm.y * rh) for lm in face_landmarks.landmark[:n]
            ]
            if len(pts) < n:
                # Model returned fewer points than expected — skip defensively.
                logger.warning(
                    "Face has %d landmarks, expected %d; skipping.", len(pts), n
                )
                continue
            face_bbox = _bbox_from_points(pts, img_w, img_h)
            ear = compute_ear(
                pts, self.config.left_eye_ear_idx, self.config.right_eye_ear_idx
            )
            faces.append((face_bbox, pts, ear))
        return faces

    def _padded_face_region(self, face_box: Bbox, img_w: int, img_h: int) -> Bbox:
        """Expand a SCRFD face box for landmark fitting, clamped to the image.

        Unlike :meth:`_padded_region` (which pads a *person* box and was
        measured to be actively harmful there), padding is wanted here: SCRFD
        returns a tight face box, and Face Mesh fits more reliably with a little
        context around the face.

        Args:
            face_box: The SCRFD face box ``(x, y, w, h)`` in image pixels.
            img_w: Image width in pixels.
            img_h: Image height in pixels.

        Returns:
            The padded region ``(x, y, w, h)``, clamped to the image bounds.
        """
        x, y, w, h = face_box
        pad_w = round(w * self.config.scrfd_landmark_padding)
        pad_h = round(h * self.config.scrfd_landmark_padding)
        x0 = max(0, x - pad_w)
        y0 = max(0, y - pad_h)
        x1 = min(img_w, x + w + pad_w)
        y1 = min(img_h, y + h + pad_h)
        return (x0, y0, max(0, x1 - x0), max(0, y1 - y0))

    def _landmarks_for_face(
        self, frame: np.ndarray, face_box: Bbox
    ) -> tuple[list[Point] | None, float | None]:
        """Fit Face Mesh landmarks inside an already-detected face box.

        Args:
            frame: The full ``(H, W, 3)`` BGR image.
            face_box: A face box ``(x, y, w, h)`` from SCRFD, in image pixels.

        Returns:
            ``(landmarks, ear)`` in image coordinates, or ``(None, None)`` when
            Face Mesh cannot fit a mesh to this crop. The caller keeps the
            SCRFD box regardless — a box without landmarks is still usable for
            head pose.
        """
        img_h, img_w = frame.shape[:2]
        region = self._padded_face_region(face_box, img_w, img_h)
        found = self._detect_faces(frame, region)
        if not found:
            return (None, None)
        # A face-box crop contains exactly one face by construction; if Face
        # Mesh reports several, the largest is the intended one.
        _bbox, pts, ear = max(found, key=lambda f: f[0][2] * f[0][3])
        return (pts, ear)

    def detect_faces(self, frame: np.ndarray) -> list:
        """Run the whole-frame face detector and return its raw detections.

        Exposed so a caller that also needs the raw face list (e.g.
        :func:`backend.students.augment_persons`, which seeds students from
        faces person detection missed) can reuse one detection pass instead of
        paying for a second — SCRFD is the most expensive stage in the
        pipeline.

        Args:
            frame: A ``(H, W, 3)`` BGR image.

        Returns:
            A list of :class:`~backend.face_detect.DetectedFace`, sorted by
            descending confidence. Empty when the ``"mediapipe"`` backend is
            configured, which has no whole-frame detection step.
        """
        if self._face_detector is None:
            return []
        return self._face_detector.detect(frame)

    def _analyze_scrfd(
        self,
        frame: np.ndarray,
        boxes: list[Bbox],
        detections: list | None = None,
    ) -> list[FaceResult]:
        """Bind whole-frame SCRFD faces to person boxes and fit landmarks.

        Args:
            frame: A ``(H, W, 3)`` BGR image.
            boxes: Person boxes ``(x, y, w, h)`` in image pixels.
            detections: Pre-computed face detections for this frame. Detected
                here when ``None``.

        Returns:
            One :class:`FaceResult` per person box, in the same order.
        """
        assert self._face_detector is not None  # guaranteed by __init__
        if detections is None:
            detections = self._face_detector.detect(frame)

        # Greedy assignment, strongest detection first (SCRFD returns them
        # sorted). Each face goes to the person box that contains most of it,
        # and neither a face nor a person can be claimed twice.
        assigned: dict[int, object] = {}
        matched_faces = 0
        for det in detections:
            best_idx, best_score = -1, 0.0
            for person_idx, person_box in enumerate(boxes):
                if person_idx in assigned:
                    continue
                score = _containment(det.bbox, person_box)
                if score > best_score:
                    best_idx, best_score = person_idx, score
            if best_idx >= 0 and best_score >= self.config.assign_min_containment:
                assigned[best_idx] = det
                matched_faces += 1

        results: list[FaceResult] = []
        for person_idx in range(len(boxes)):
            det = assigned.get(person_idx)
            if det is None:
                results.append(FaceResult(face_bbox=None, landmarks=None, ear=None))
                continue
            pts, ear = self._landmarks_for_face(frame, det.bbox)
            results.append(
                FaceResult(
                    face_bbox=det.bbox, landmarks=pts, ear=ear, kps=det.kps
                )
            )

        # Unmatched faces mean SCRFD saw a student that YOLO's person detector
        # missed — a person-detection shortfall. Logged, never silently dropped.
        unmatched = len(detections) - matched_faces
        if unmatched > 0:
            logger.debug(
                "%d of %d SCRFD faces had no containing person box "
                "(person detection missed them).",
                unmatched,
                len(detections),
            )
        return results

    def _analyze_mediapipe(
        self, frame: np.ndarray, boxes: list[Bbox]
    ) -> list[FaceResult]:
        """Original path: one Face Mesh pass per person crop.

        Kept for benchmarking against the SCRFD backend only — it finds roughly
        a fifth as many faces (see the module docstring).

        Args:
            frame: A ``(H, W, 3)`` BGR image.
            boxes: Person boxes ``(x, y, w, h)`` in image pixels.

        Returns:
            One :class:`FaceResult` per person box, in the same order.
        """
        img_h, img_w = frame.shape[:2]

        candidates: list[tuple[float, int, tuple[Bbox, list[Point], float | None]]] = []
        detected = 0
        for person_idx, person_box in enumerate(boxes):
            region = self._padded_region(person_box, img_w, img_h)
            for face in self._detect_faces(frame, region):
                detected += 1
                score = _containment(face[0], person_box)
                if score >= self.config.assign_min_containment:
                    candidates.append((score, person_idx, face))

        # Greedy assignment, best containment first. Ties break on person index
        # so the result is deterministic. A face already claimed by another
        # person (overlapping boxes see the same head) is not reused.
        candidates.sort(key=lambda c: (-c[0], c[1]))
        assigned: dict[int, tuple[Bbox, list[Point], float | None]] = {}
        taken: list[Bbox] = []
        for _score, person_idx, face in candidates:
            if person_idx in assigned:
                continue
            if any(_iou(face[0], t) > self.config.duplicate_face_iou for t in taken):
                continue
            assigned[person_idx] = face
            taken.append(face[0])

        results: list[FaceResult] = []
        for person_idx in range(len(boxes)):
            face = assigned.get(person_idx)
            if face is None:
                results.append(FaceResult(face_bbox=None, landmarks=None, ear=None))
            else:
                face_bbox, pts, ear = face
                results.append(FaceResult(face_bbox=face_bbox, landmarks=pts, ear=ear))

        logger.debug(
            "analyze(mediapipe): %d persons, %d faces across crops, %d matched.",
            len(boxes),
            detected,
            len(assigned),
        )
        return results

    def analyze(
        self,
        frame: np.ndarray,
        person_bboxes: Sequence[Sequence[float]],
        detected_faces: Sequence | None = None,
    ) -> list[FaceResult]:
        """Analyze one frame and bind faces to the given person boxes.

        Args:
            frame: A ``(H, W, 3)`` BGR image as returned by OpenCV.
            person_bboxes: Person boxes ``(x, y, w, h)`` from ``detection.py``,
                in image pixels.
            detected_faces: Optional pre-computed output of
                :meth:`detect_faces` for this same frame, to avoid a second
                detection pass. Ignored by the ``"mediapipe"`` backend, which
                detects inside each person crop instead.

        Returns:
            A list of :class:`FaceResult`, one per entry in ``person_bboxes`` and
            in the same order. Persons with no matching face have all-``None``
            fields.

        Raises:
            RuntimeError: If the analyzer has already been closed.
            TypeError: If ``frame`` is not a NumPy array.
            ValueError: If ``frame`` is empty/not 3-channel, or a person bbox is
                malformed.
        """
        if self._closed:
            raise RuntimeError("FaceAnalyzer is closed; create a new instance.")
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

        if self._face_detector is not None:
            return self._analyze_scrfd(
                frame,
                boxes,
                None if detected_faces is None else list(detected_faces),
            )
        return self._analyze_mediapipe(frame, boxes)
