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
directly reached **F1 77.9%** on held-out clips.

Trained on two independently-collected datasets
------------------------------------------------

877 images / 6091 boxes, merged from this project's own classroom footage plus a
second, unrelated classroom dataset (see ``tools/merge_behaviour_datasets.py``).
Merging was previously blocked by a label-density conflict and became possible
once ``look_forward`` was dropped -- see "what it does not report" below. The
payoff was large, and the out-of-distribution figure is the one that matters:

=================================  =========  =========
Metric                             1 dataset  2 datasets
=================================  =========  =========
mAP50                                 0.437      0.653
writing signal F1 (held out)          65.3%      77.9%
**F1 on an unseen classroom**       **7.3%**  **68.0%**
=================================  =========  =========

Per class on held-out clips, and note how the weak one MOVED:

==============  =========  =======  =====  ==========
class           precision  recall     F1   previously
==============  =========  =======  =====  ==========
using_device        79.7%    65.6%  72.0%      30.6%
write               73.8%    68.1%  70.9%      68.3%
sleep               75.0%    58.5%  65.8%      57.5%
read                45.9%    57.1%  50.9%      43.0%
==============  =========  =======  =====  ==========

A classifier, not a detector
----------------------------

This model can find students on its own, and it is deliberately not used that
way. On held-out data:

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

* **``look_forward`` and ``turn_head``** are not classes of this model at all.
  Head orientation is what a head-pose model exists to measure, and calibrated
  head pose scored F1 63.2% against this model's 25.0% on the same 371 labelled
  boxes. They were previously suppressed at inference time; they are now
  excluded from the training set entirely, which additionally dissolved the
  label conflict that had blocked merging the second dataset. ``look_forward``
  alone had been 2384 of 4603 boxes -- the model was dominated by a class the
  pipeline discarded.
* **``handrise`` and ``stand``** are likewise gone (22 and 59 boxes, F1 4.1%
  and 0.0%). Too little data to report honestly.
* **``read`` is the weak class now**, at F1 50.9% with sub-50% precision, and is
  confused with ``write`` in both directions -- an understandable confusion
  (both are head-down-at-a-desk) but one a consumer must not read as certain. It
  carries ``reliability="weak"``. ``using_device``, which used to be the
  headline weakness at ~20% recall, is now among the strongest at 72.0%.

So the four surfaced classes are ``read``, ``sleep``, ``using_device`` and
``write``, and ``read`` is the one to treat with suspicion.

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

#: Classes whose held-out accuracy is poor enough that the value must carry a
#: warning with it. Re-measured after the merged-dataset retrain, and the
#: membership CHANGED -- so this is kept in sync with measurement rather than
#: left as a stale assumption:
#:
#:   class          precision  recall    F1     was
#:   using_device      79.7%   65.6%   72.0%   30.6%  -> no longer weak
#:   write             73.8%   68.1%   70.9%   68.3%  -> fine
#:   sleep             75.0%   58.5%   65.8%   57.5%  -> borderline, acceptable
#:   read              45.9%   57.1%   50.9%   43.0%  -> WEAK: <half its
#:                                                       predictions are right
#:
#: ``using_device`` was the headline weakness before the retrain (~20% recall)
#: and is now one of the strongest classes. ``read`` is now the weak one, and is
#: confused with ``write`` in both directions -- an understandable confusion
#: (both are head-down-at-a-desk) but one a consumer must not read as certain.
_WEAK_CLASSES: frozenset[str] = frozenset({"read"})


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
