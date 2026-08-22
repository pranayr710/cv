"""Unit tests for :mod:`backend.expression`.

The mapping, guardrail and aggregation logic is pure Python and is tested
without loading a model, via a fake recognizer. Only the two model-backed tests
need ``emotiefflib``, and they skip cleanly when it is absent rather than
faking a pass.

Test coverage:
    1. The 8-to-3 mapping collapses raw AffectNet classes correctly.
    2. The full raw distribution survives the collapse (auditable, reversible).
    3. Faces below ``min_face_px`` return ``None`` rather than a guess.
    4. ``None`` face boxes keep their slot, so results stay index-aligned.
    5. A mapping targeting an unreported label is rejected at construction.
    6. ``summarise_expressions`` reports class-level aggregates, and reports no
       shares at all when nothing was classified.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from backend.config import CONFIG, ExpressionConfig
from backend.expression import (
    ExpressionRecognizer,
    ExpressionResult,
    summarise_expressions,
)

_AFFECTNET = {
    0: "Anger",
    1: "Contempt",
    2: "Disgust",
    3: "Fear",
    4: "Happiness",
    5: "Neutral",
    6: "Sadness",
    7: "Surprise",
}


class _FakeModel:
    """Stands in for EmotiEffLib, returning a fixed score vector."""

    def __init__(self, scores: list[float]) -> None:
        self.idx_to_emotion_class = _AFFECTNET
        self._scores = scores

    def predict_emotions(self, _rgb, logits: bool = False):
        return (["ignored"], np.array([self._scores]))


def _recognizer(scores: list[float], config: ExpressionConfig | None = None):
    """Build a recognizer with the model swapped for a fake, no download."""
    rec = ExpressionRecognizer.__new__(ExpressionRecognizer)
    rec.config = config if config is not None else CONFIG.expression
    rec._map = dict(rec.config.expression_map)
    rec._model = _FakeModel(scores)
    return rec


def _frame(h: int = 400, w: int = 400) -> np.ndarray:
    return np.zeros((h, w, 3), dtype=np.uint8)


# --------------------------------------------------------------------------- #
# The 8 -> 3 mapping
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("winner", "expected"),
    [
        ("Happiness", "happy"),
        ("Sadness", "sad"),
        ("Neutral", "neutral"),
        # These map to neutral, not sad: folding anger into sadness would
        # assert something the model never predicted.
        ("Anger", "neutral"),
        ("Fear", "neutral"),
        ("Surprise", "neutral"),
        ("Disgust", "neutral"),
        ("Contempt", "neutral"),
    ],
)
def test_raw_class_maps_to_reported_label(winner: str, expected: str) -> None:
    idx = next(i for i, name in _AFFECTNET.items() if name == winner)
    scores = [0.01] * 8
    scores[idx] = 0.9
    rec = _recognizer(scores)
    result = rec.classify(_frame(), [(10, 10, 100, 100)])[0]
    assert result is not None
    assert result.label == expected


def test_full_raw_distribution_is_preserved() -> None:
    """The collapse must stay auditable — all 8 raw classes are retained."""
    scores = [0.05, 0.05, 0.05, 0.05, 0.5, 0.15, 0.1, 0.05]
    rec = _recognizer(scores)
    result = rec.classify(_frame(), [(10, 10, 100, 100)])[0]
    assert result is not None
    assert set(result.distribution) == set(_AFFECTNET.values())
    assert result.distribution["Happiness"] == pytest.approx(0.5)
    assert sum(result.distribution.values()) == pytest.approx(1.0)


def test_confidence_is_the_raw_winning_class_probability() -> None:
    """Documented behaviour: confidence is the raw class score, not the bucket.

    A confidently-Anger face reports high confidence for label "neutral". This
    is deliberate and documented on ExpressionResult; pinning it here so the
    meaning cannot drift silently.
    """
    scores = [0.8, 0.0, 0.0, 0.0, 0.05, 0.1, 0.05, 0.0]
    rec = _recognizer(scores)
    result = rec.classify(_frame(), [(10, 10, 100, 100)])[0]
    assert result is not None
    assert result.label == "neutral"
    assert result.confidence == pytest.approx(0.8)


# --------------------------------------------------------------------------- #
# Guardrails: absent beats invented
# --------------------------------------------------------------------------- #


def test_face_below_min_size_is_unclassified_not_guessed() -> None:
    """A back-row face is ~20px; upscaling it 7x into 224px is not a measurement."""
    cfg = CONFIG.expression
    rec = _recognizer([0.9] + [0.01] * 7)
    small = cfg.min_face_px - 1
    assert rec.classify(_frame(), [(10, 10, small, small)]) == [None]


def test_face_at_exactly_min_size_is_classified() -> None:
    """The threshold is inclusive — an off-by-one here silently drops faces."""
    cfg = CONFIG.expression
    rec = _recognizer([0.01] * 4 + [0.9] + [0.01] * 3)
    px = cfg.min_face_px
    assert rec.classify(_frame(), [(10, 10, px, px)])[0] is not None


def test_none_face_keeps_its_slot() -> None:
    """Results must stay index-aligned with persons, like headpose/posture."""
    rec = _recognizer([0.01] * 4 + [0.9] + [0.01] * 3)
    results = rec.classify(
        _frame(), [None, (10, 10, 100, 100), None]
    )
    assert len(results) == 3
    assert results[0] is None
    assert results[1] is not None
    assert results[2] is None


def test_empty_face_list_returns_empty() -> None:
    rec = _recognizer([0.9] + [0.01] * 7)
    assert rec.classify(_frame(), []) == []


@pytest.mark.parametrize(
    "frame",
    [
        np.zeros((0, 0, 3), dtype=np.uint8),
        np.zeros((10, 10), dtype=np.uint8),
    ],
)
def test_malformed_frame_is_rejected(frame) -> None:
    rec = _recognizer([0.9] + [0.01] * 7)
    with pytest.raises(ValueError):
        rec.classify(frame, [(1, 1, 50, 50)])


def test_bad_mapping_is_rejected_at_construction() -> None:
    """A typo'd mapping target would emit a category nothing can report."""
    bad = replace(
        ExpressionConfig(),
        expression_map=(("Happiness", "delighted"), ("Sadness", "sad")),
    )
    with pytest.raises(ValueError, match="not in reported_labels"):
        ExpressionRecognizer(bad)


# --------------------------------------------------------------------------- #
# Class-level aggregation — the intended way to consume this module
# --------------------------------------------------------------------------- #


def _result(label: str) -> ExpressionResult:
    return ExpressionResult(label=label, confidence=0.9, distribution={})


def test_summary_counts_and_shares() -> None:
    results = [
        _result("happy"), _result("happy"), _result("neutral"), _result("sad"),
    ]
    summary = summarise_expressions(results)
    assert summary["students"] == 4
    assert summary["classified"] == 4
    assert summary["unavailable"] == 0
    assert summary["counts"] == {"happy": 2, "sad": 1, "neutral": 1}
    assert summary["shares"]["happy"] == pytest.approx(0.5)


def test_summary_counts_unavailable_students() -> None:
    """Shares are of *classified* students, so coverage stays visible."""
    summary = summarise_expressions([_result("happy"), None, None])
    assert summary["students"] == 3
    assert summary["classified"] == 1
    assert summary["unavailable"] == 2
    assert summary["shares"]["happy"] == pytest.approx(1.0)


def test_summary_reports_no_shares_when_nothing_classified() -> None:
    """Zeros would read as a measurement of a calm classroom. They are not."""
    summary = summarise_expressions([None, None])
    assert summary["classified"] == 0
    assert summary["shares"] == {}
    assert summary["counts"] == {"happy": 0, "sad": 0, "neutral": 0}


def test_every_reported_label_is_present_in_counts() -> None:
    """A label with zero occurrences must still appear, or charts lose a bar."""
    summary = summarise_expressions([_result("neutral")])
    assert set(summary["counts"]) == set(CONFIG.expression.reported_labels)


# --------------------------------------------------------------------------- #
# Model-backed (skipped when emotiefflib is unavailable)
# --------------------------------------------------------------------------- #


def test_real_model_loads_and_emits_a_reported_label() -> None:
    pytest.importorskip("emotiefflib")
    cv2 = pytest.importorskip("cv2")
    recognizer = ExpressionRecognizer()
    frame = np.full((300, 300, 3), 128, dtype=np.uint8)
    cv2.circle(frame, (150, 150), 80, (200, 180, 170), -1)
    results = recognizer.classify(frame, [(70, 70, 160, 160)])
    assert len(results) == 1
    # "uncertain" is a valid outcome, not a failure: a synthetic blob is exactly
    # the kind of input the model should decline to label confidently.
    allowed = (*CONFIG.expression.reported_labels, CONFIG.expression.uncertain_label)
    assert results[0] is None or results[0].label in allowed


def test_real_model_class_set_matches_the_mapping() -> None:
    """Guards against an upstream model with a different taxonomy.

    If EmotiEffLib ships a model whose class names differ, every prediction
    would silently fall back to "neutral". This fails loudly instead.
    """
    pytest.importorskip("emotiefflib")
    recognizer = ExpressionRecognizer()
    raw_classes = set(recognizer._model.idx_to_emotion_class.values())
    mapped = set(dict(CONFIG.expression.expression_map))
    assert raw_classes <= mapped, (
        f"Model emits unmapped classes: {sorted(raw_classes - mapped)}"
    )
