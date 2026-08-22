"""Unit tests for :mod:`backend.behaviour`.

Binding and filtering logic is pure Python and is tested with a fake model, so
these run without the fine-tuned weights (which live under gitignored ``runs/``
and do not exist on a fresh clone).

Test coverage:
    1. Untrusted classes (`handrise`, `stand`) never reach the output.
    2. Deferred classes (`turn_head`, `look_forward`) never reach the output,
       because head pose measures them better.
    3. Weak classes carry a "weak" reliability tag with the value.
    4. Results stay index-aligned with the input students.
    5. Binding is one-to-one and prefers the strongest detection.
    6. Missing weights fail loudly rather than falling back to a COCO model.
    7. `summarise_behaviour` surfaces which reported classes are unreliable.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from backend.behaviour import (
    BehaviourClassifier,
    BehaviourResult,
    summarise_behaviour,
)
from backend.config import CONFIG, BehaviourConfig

_NAMES = CONFIG.behaviour.class_names


class _FakeBoxes:
    """Minimal stand-in for an Ultralytics Boxes object."""

    def __init__(self, dets):
        self._xyxy = np.array(
            [[d[2][0], d[2][1], d[2][0] + d[2][2], d[2][1] + d[2][3]] for d in dets],
            dtype=np.float32,
        ).reshape(-1, 4)
        self._conf = np.array([d[1] for d in dets], dtype=np.float32)
        self._cls = np.array([_NAMES.index(d[0]) for d in dets], dtype=np.float32)

    def __len__(self):
        return len(self._conf)

    class _T:
        def __init__(self, a):
            self._a = a

        def cpu(self):
            return self

        def numpy(self):
            return self._a

    @property
    def xyxy(self):
        return self._T(self._xyxy)

    @property
    def conf(self):
        return self._T(self._conf)

    @property
    def cls(self):
        return self._T(self._cls)


class _FakeModel:
    """Returns a fixed detection list regardless of input."""

    def __init__(self, dets):
        self._dets = dets

    def predict(self, *_a, **_kw):
        class R:
            pass

        r = R()
        r.boxes = _FakeBoxes(self._dets) if self._dets else None
        return [r]


def _classifier(dets, config: BehaviourConfig | None = None):
    """Build a classifier with the model faked, so no weights are needed."""
    cfg = config if config is not None else CONFIG.behaviour
    clf = BehaviourClassifier.__new__(BehaviourClassifier)
    clf.config = cfg
    clf._model = _FakeModel(dets)
    clf._suppressed = frozenset(cfg.untrusted_classes) | frozenset(
        cfg.deferred_classes
    )
    return clf


def _frame(h: int = 1080, w: int = 1920) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


#: A student box and a behaviour box that mutually contain each other's centre.
_STUDENT = (100, 100, 200, 300)
_INSIDE = (120, 120, 160, 260)


# --------------------------------------------------------------------------- #
# Suppression: classes this module must never report
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("cls", ["handrise", "stand"])
def test_untrusted_classes_are_never_reported(cls: str) -> None:
    """These scored F1 4.1% and 0.0% held out — noise wearing a label."""
    clf = _classifier([(cls, 0.99, _INSIDE)])
    assert clf.classify(_frame(), [_STUDENT]) == [None]


@pytest.mark.parametrize("cls", ["turn_head", "look_forward"])
def test_deferred_classes_are_never_reported(cls: str) -> None:
    """Head orientation belongs to head pose (F1 63.2% vs this model's 25.0%).

    Reporting it here too would let the weaker answer win downstream.
    """
    clf = _classifier([(cls, 0.99, _INSIDE)])
    assert clf.classify(_frame(), [_STUDENT]) == [None]


def test_suppressed_detection_does_not_block_a_surfaced_one() -> None:
    """A filtered class must not consume the student's binding slot."""
    clf = _classifier(
        [("look_forward", 0.99, _INSIDE), ("write", 0.40, _INSIDE)]
    )
    result = clf.classify(_frame(), [_STUDENT])[0]
    assert result is not None
    assert result.label == "write"


# --------------------------------------------------------------------------- #
# Reliability travels with the value
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("cls", "expected"),
    [("write", "measured"), ("read", "measured"),
     ("using_device", "weak"), ("sleep", "weak")],
)
def test_reliability_tag_matches_measured_strength(cls: str, expected: str) -> None:
    """using_device reaches only ~20% recall; the tag stops it reading as solid."""
    clf = _classifier([(cls, 0.8, _INSIDE)])
    result = clf.classify(_frame(), [_STUDENT])[0]
    assert result is not None
    assert result.reliability == expected


# --------------------------------------------------------------------------- #
# Binding
# --------------------------------------------------------------------------- #


def test_results_stay_index_aligned_with_students() -> None:
    far_student = (1500, 700, 200, 300)
    clf = _classifier([("write", 0.9, _INSIDE)])
    results = clf.classify(_frame(), [far_student, _STUDENT, far_student])
    assert len(results) == 3
    assert results[0] is None
    assert results[1] is not None and results[1].label == "write"
    assert results[2] is None


def test_binding_prefers_the_strongest_detection() -> None:
    clf = _classifier([("read", 0.4, _INSIDE), ("write", 0.9, _INSIDE)])
    result = clf.classify(_frame(), [_STUDENT])[0]
    assert result is not None
    assert result.label == "write"


def test_binding_is_one_to_one() -> None:
    """Two detections on one student must not both bind to it."""
    clf = _classifier([("write", 0.9, _INSIDE), ("read", 0.8, _INSIDE)])
    results = clf.classify(_frame(), [_STUDENT])
    assert len([r for r in results if r is not None]) == 1


def test_detection_outside_every_student_is_dropped() -> None:
    clf = _classifier([("write", 0.9, (1500, 800, 100, 100))])
    assert clf.classify(_frame(), [_STUDENT]) == [None]


def test_no_detections_yields_all_none() -> None:
    clf = _classifier([])
    assert clf.classify(_frame(), [_STUDENT, _STUDENT]) == [None, None]


def test_empty_student_list_returns_empty() -> None:
    clf = _classifier([("write", 0.9, _INSIDE)])
    assert clf.classify(_frame(), []) == []


@pytest.mark.parametrize(
    "frame",
    [np.zeros((0, 0, 3), dtype=np.uint8), np.zeros((10, 10), dtype=np.uint8)],
)
def test_malformed_frame_is_rejected(frame) -> None:
    clf = _classifier([("write", 0.9, _INSIDE)])
    with pytest.raises(ValueError):
        clf.classify(frame, [_STUDENT])


# --------------------------------------------------------------------------- #
# Construction guards
# --------------------------------------------------------------------------- #


def test_missing_weights_fails_loudly() -> None:
    """A silent COCO fallback would report object classes as student behaviour."""
    cfg = replace(BehaviourConfig(), weights="runs/does/not/exist.pt")
    with pytest.raises(FileNotFoundError, match="tools.train_behaviour"):
        BehaviourClassifier(cfg)


def test_unknown_class_in_config_is_rejected() -> None:
    """A typo would silently suppress nothing, or suppress the wrong class."""
    cfg = replace(BehaviourConfig(), untrusted_classes=("hand_raise",))
    with pytest.raises(ValueError, match="not in class_names"):
        BehaviourClassifier(cfg)


def test_config_defaults_are_self_consistent() -> None:
    cfg = CONFIG.behaviour
    assert set(cfg.untrusted_classes) <= set(cfg.class_names)
    assert set(cfg.deferred_classes) <= set(cfg.class_names)
    surfaced = set(cfg.class_names) - set(cfg.untrusted_classes) - set(
        cfg.deferred_classes
    )
    # write/read are the classes the whole fine-tune existed to fix.
    assert {"write", "read"} <= surfaced


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def test_summary_counts_and_flags_weak_labels() -> None:
    results = [
        BehaviourResult("write", 0.9, "measured"),
        BehaviourResult("write", 0.8, "measured"),
        BehaviourResult("using_device", 0.5, "weak"),
        None,
    ]
    summary = summarise_behaviour(results)
    assert summary["students"] == 4
    assert summary["classified"] == 3
    assert summary["unavailable"] == 1
    assert summary["counts"] == {"write": 2, "using_device": 1}
    # The consumer cannot read the counts without seeing which are unreliable.
    assert summary["weak_labels"] == ["using_device"]


def test_summary_reports_no_weak_labels_when_none_present() -> None:
    summary = summarise_behaviour([BehaviourResult("write", 0.9, "measured")])
    assert summary["weak_labels"] == []


def test_summary_of_nothing_classified() -> None:
    summary = summarise_behaviour([None, None])
    assert summary["classified"] == 0
    assert summary["counts"] == {}
    assert summary["weak_labels"] == []
