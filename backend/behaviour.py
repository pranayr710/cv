"""Per-student behaviour classification for ClassGraph (write / read / sleep).

Consumes the student boxes the pipeline already produces
(:mod:`backend.detection` + :mod:`backend.students`) and labels what each
student is *doing*. Like :mod:`backend.headpose` and :mod:`backend.expression`,
it is a layer over existing boxes and does not find students itself.

Why this exists
---------------

It replaces a proxy that was measured and failed. The previous "is this student
writing" signal was *a detected book near a bowed head*, which scored precision
31.9% / recall 20.7% / **F1 25.1%** against human labels. COCO's ``book`` class
was never trained on open notebooks at a classroom angle, and it does not fire
on loose exam paper at all, so no threshold rescued it. Training the behaviour
directly on 423 labelled frames lifted the same metric to **F1 ~62-68%** on
held-out clips.

A classifier, not a detector
----------------------------

This model can find students on its own, and it is deliberately not used that
way. On the same held-out data:

==============================  ==========  =======
Finding students                Precision   Recall
==============================  ==========  =======
SCRFD + YOLO pipeline             82.2%      90.6%
This model alone                  89.8%      70.5%
==============================  ==========  =======

It is more precise but finds ~20 points fewer students. A missed student is
invisible to *every* downstream signal, so detection recall wins and this model
is bound onto the existing boxes instead.

What it deliberately does not report
------------------------------------

* **``handrise`` and ``stand``** are dropped
  (:data:`BehaviourConfig.untrusted_classes`). They had 22 and 59 training boxes
  and scored F1 4.1% and 0.0% on the held-out split. A class measured at 4% F1
  is noise wearing a label.
* **``turn_head`` and ``look_forward``** are deferred to :mod:`backend.headpose`
  (:data:`BehaviourConfig.deferred_classes`). Head orientation is what a
  head-pose model exists to measure, and calibrated head pose scores F1 63.2%
  against this model's 25.0% on the same 371 labelled boxes. Reporting both and
  letting the weaker one win would be a regression disguised as a feature.
* **Phone use is an open gap, not a feature.** ``using_device`` reaches only
  ~20% recall, and a confidence sweep confirmed that is a model/data limit
  rather than a threshold: recall moves 20.0% to 26.7% while precision falls
  58.1% to 37.5%. It is surfaced with its weakness attached, never presented as
  working.

So the classes actually surfaced are ``read``, ``sleep``, ``using_device`` and
``write`` -- and of those, only ``write`` and ``read`` are strong enough to
build on today.

Usage:
    from backend.behaviour import BehaviourClassifier
    classifier = BehaviourClassifier()
    results = classifier.classify(frame, student_bboxes)
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.config import CONFIG, BehaviourConfig

logger = logging.getLogger(__name__)

Bbox = tuple[int, int, int, int]

#: Classes whose held-out recall is poor enough that the value must carry a
#: warning with it. ``using_device`` reaches only ~20% recall and ``sleep``
#: ~39-46%; see the module docstring.
_WEAK_CLASSES: frozenset[str] = frozenset({"using_device", "sleep"})


@dataclass(frozen=True)
class BehaviourResult:
    """One student's classified behaviour for one frame.

    Attributes:
        label: The behaviour class -- one of the surfaced classes (``read``,
            ``sleep``, ``using_device``, ``write``). Never an untrusted or
            deferred class; those are filtered before this is built.
        confidence: The model's detection confidence in ``[0, 1]``.
        reliability: How much this class can be trusted, carried *with* the
            value so a weak signal cannot be read as a strong one downstream.
            ``"measured"`` for ``write``/``read``, ``"weak"`` for classes whose
            held-out recall is poor.
    """

    label: str
    confidence: float
    reliability: str


def _centre_in(inner: Bbox, outer: Bbox) -> bool:
    """Whether ``inner``'s centre point lies inside ``outer``."""
    cx = inner[0] + inner[2] / 2.0
    cy = inner[1] + inner[3] / 2.0
    return (
        outer[0] <= cx <= outer[0] + outer[2]
        and outer[1] <= cy <= outer[1] + outer[3]
    )


def _iou(a: Bbox, b: Bbox) -> float:
    """Intersection-over-union of two ``(x, y, w, h)`` boxes."""
    inter_w = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    inter_h = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = inter_w * inter_h
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


class BehaviourClassifier:
    """Fine-tuned YOLOv11 wrapper binding behaviour labels to student boxes.

    Attributes:
        config: The :class:`BehaviourConfig` in effect.
    """

    def __init__(self, config: BehaviourConfig | None = None) -> None:
        """Load the fine-tuned behaviour model.

        Args:
            config: Behaviour settings. Defaults to ``CONFIG.behaviour``.

        Raises:
            ImportError: If Ultralytics is not installed.
            FileNotFoundError: If the fine-tuned weights are missing. Raised
                loudly rather than falling back to a COCO model, because a
                silent fallback would report generic object classes as student
                behaviour.
            RuntimeError: If the model fails to load.
            ValueError: If a configured class list names an unknown class.
        """
        self.config: BehaviourConfig = (
            config if config is not None else CONFIG.behaviour
        )

        known = set(self.config.class_names)
        unknown = (
            set(self.config.untrusted_classes) | set(self.config.deferred_classes)
        ) - known
        if unknown:
            raise ValueError(
                f"untrusted/deferred classes not in class_names: {sorted(unknown)}"
            )

        weights = Path(self.config.weights)
        if not weights.is_file():
            raise FileNotFoundError(
                f"Fine-tuned behaviour weights not found at {weights}. Train "
                f"them with `python -m tools.train_behaviour` (runs/ is "
                f"gitignored, so a fresh clone has none). Refusing to fall back "
                f"to a COCO model, which would report object classes as student "
                f"behaviour."
            )

        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "Ultralytics is required for behaviour classification."
            ) from exc

        try:
            self._model = YOLO(str(weights))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load behaviour model from {weights}: {exc}"
            ) from exc

        self._suppressed: frozenset[str] = frozenset(
            self.config.untrusted_classes
        ) | frozenset(self.config.deferred_classes)

        logger.info(
            "BehaviourClassifier ready: weights=%s imgsz=%d conf=%.2f surfacing=%s",
            weights,
            self.config.imgsz,
            self.config.conf,
            sorted(known - self._suppressed),
        )

    def classify(
        self, frame: np.ndarray, student_bboxes: Sequence[Sequence[float]]
    ) -> list[BehaviourResult | None]:
        """Classify the behaviour of each given student.

        Args:
            frame: A ``(H, W, 3)`` BGR image as returned by OpenCV.
            student_bboxes: Student boxes ``(x, y, w, h)`` in image pixels, from
                :mod:`backend.detection` plus :mod:`backend.students`.

        Returns:
            A list **aligned index-wise** with ``student_bboxes``. An entry is
            ``None`` when no behaviour box bound to that student, or when the
            only candidate was a suppressed class -- absent rather than guessed.

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

        students: list[Bbox] = [
            tuple(round(float(v)) for v in box) for box in student_bboxes
        ]
        if not students:
            return []

        detections = self._detect(frame)

        # Greedy one-to-one binding, strongest detection first. Mutual-centre
        # containment rather than IoU: this model was trained on tight
        # head+torso boxes while pipeline students are full-body or face-seeded,
        # so IoU under-matches for reasons unrelated to correctness (it rejected
        # 4 of 11 correct pairs on a real frame -- see tools/eval_detection.py).
        assigned: dict[int, tuple[str, float]] = {}
        for label, conf, box in sorted(detections, key=lambda d: -d[1]):
            best_idx, best_overlap = -1, -1.0
            for idx, student in enumerate(students):
                if idx in assigned:
                    continue
                if self.config.require_mutual_centre and not (
                    _centre_in(box, student) and _centre_in(student, box)
                ):
                    continue
                overlap = _iou(box, student)
                if overlap > best_overlap:
                    best_idx, best_overlap = idx, overlap
            if best_idx >= 0:
                assigned[best_idx] = (label, conf)

        results: list[BehaviourResult | None] = []
        for idx in range(len(students)):
            match = assigned.get(idx)
            if match is None:
                results.append(None)
                continue
            label, conf = match
            results.append(
                BehaviourResult(
                    label=label,
                    confidence=conf,
                    reliability="weak" if label in _WEAK_CLASSES else "measured",
                )
            )
        return results

    def _detect(self, frame: np.ndarray) -> list[tuple[str, float, Bbox]]:
        """Run the model and return surfaced ``(label, conf, bbox)`` detections.

        Suppressed classes -- untrusted, or deferred to a better-suited model --
        are filtered here so they never reach binding.

        Args:
            frame: A ``(H, W, 3)`` BGR image.

        Returns:
            Detections whose class this module surfaces.
        """
        result = self._model.predict(
            frame,
            imgsz=self.config.imgsz,
            conf=self.config.conf,
            iou=self.config.iou,
            verbose=False,
        )[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        names = self.config.class_names
        out: list[tuple[str, float, Bbox]] = []
        for (x1, y1, x2, y2), conf, cls_id in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.boxes.cls.cpu().numpy().astype(int),
        ):
            label = names[cls_id] if cls_id < len(names) else f"id{cls_id}"
            if label in self._suppressed:
                continue
            box = (
                round(float(x1)),
                round(float(y1)),
                max(1, round(float(x2 - x1))),
                max(1, round(float(y2 - y1))),
            )
            out.append((label, float(conf), box))
        return out


def summarise_behaviour(
    results: Sequence[BehaviourResult | None],
) -> dict[str, object]:
    """Aggregate per-student behaviours into a class-level summary.

    The intended consumer, for the same reason as
    :func:`backend.expression.summarise_expressions`: this project's reporting
    default is a class-level trend, never a live per-student verdict.

    Args:
        results: Per-student behaviour results, ``None`` where unavailable.

    Returns:
        A dict with ``students``, ``classified``, ``unavailable``, ``counts``
        per label, and ``weak_labels`` naming any reported class whose held-out
        accuracy is poor -- so counts cannot be read without also seeing which
        of them are unreliable.
    """
    classified = [r for r in results if r is not None]
    counts: dict[str, int] = {}
    for result in classified:
        counts[result.label] = counts.get(result.label, 0) + 1
    return {
        "students": len(results),
        "classified": len(classified),
        "unavailable": len(results) - len(classified),
        "counts": counts,
        "weak_labels": sorted(set(counts) & _WEAK_CLASSES),
    }
