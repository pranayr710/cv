"""Unit tests for :mod:`backend.behaviour`.

Binding and filtering logic is pure Python and is tested with a fake model, so
these run without the fine-tuned weights (which live under gitignored ``runs/``
and do not exist on a fresh clone).

Test coverage:
    1. Suppression defaults are now empty -- the classes that used to be
       filtered here (handrise/stand as untrusted, turn_head/look_forward as
       deferred to head pose) were excluded from the retrained model's dataset
       entirely, moving the decision from inference into the data.
    2. The suppression MECHANISM still works, so a future weak class can be
       filtered without new code.
    3. Weak classes carry a "weak" reliability tag with the value -- membership
       re-measured after the retrain (using_device is no longer weak; read is).
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


def test_suppression_defaults_are_empty_after_the_retrain() -> None:
    """The old suppression lists (handrise/stand as untrusted, turn_head/
    look_forward as deferred to head pose) are now empty, because those classes
    were excluded from the merged training set entirely -- the decision moved
    from inference-time filtering into the dataset itself. The mechanism is
    still exercised below; this pins that nothing is being silently filtered."""
    assert CONFIG.behaviour.untrusted_classes == ()
    assert CONFIG.behaviour.deferred_classes == ()
    assert set(CONFIG.behaviour.class_names) == {
        "read", "sleep", "using_device", "write"
    }


def test_configured_suppression_still_filters_a_class() -> None:
    """The suppression MECHANISM must keep working even though its default
    lists are now empty -- a future weak class should be suppressible without
    new code."""
    cfg = replace(BehaviourConfig(), untrusted_classes=("sleep",))
    clf = _classifier([("sleep", 0.99, _INSIDE)], cfg)
    assert clf.classify(_frame(), [_STUDENT]) == [None]


def test_suppressed_detection_does_not_block_a_surfaced_one() -> None:
    """A filtered class must not consume the student's binding slot."""
    cfg = replace(BehaviourConfig(), untrusted_classes=("sleep",))
    clf = _classifier(
        [("sleep", 0.99, _INSIDE), ("write", 0.40, _INSIDE)], cfg
    )
    result = clf.classify(_frame(), [_STUDENT])[0]
    assert result is not None
    assert result.label == "write"


# --------------------------------------------------------------------------- #
# Reliability travels with the value
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("cls", "expected"),
    [("write", "measured"), ("sleep", "measured"),
     ("using_device", "measured"), ("read", "weak")],
)
def test_reliability_tag_matches_measured_strength(cls: str, expected: str) -> None:
    """Membership CHANGED after the merged-dataset retrain, and these
    assertions were updated to match measurement rather than left stale:
    using_device went 30.6% -> 72.0% F1 (no longer weak) and read is now the
    weakest at 50.9% with sub-50% precision. See backend/behaviour.py's
    _WEAK_CLASSES table for the full before/after."""
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
        BehaviourResult("read", 0.5, "weak"),
        None,
    ]
    summary = summarise_behaviour(results)
    assert summary["students"] == 4
    assert summary["classified"] == 3
    assert summary["unavailable"] == 1
    assert summary["counts"] == {"write": 2, "read": 1}
    # The consumer cannot read the counts without seeing which are unreliable.
    assert summary["weak_labels"] == ["read"]


def test_summary_reports_no_weak_labels_when_none_present() -> None:
    summary = summarise_behaviour([BehaviourResult("write", 0.9, "measured")])
    assert summary["weak_labels"] == []


def test_summary_of_nothing_classified() -> None:
    summary = summarise_behaviour([None, None])
    assert summary["classified"] == 0
    assert summary["counts"] == {}
    assert summary["weak_labels"] == []
