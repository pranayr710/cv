"""Tests for the learned window-level verdict.

The failure this guards against is a model that runs, returns plausible
verdicts, and has its weights attached to the wrong columns.
"""
import json
from pathlib import Path

import pytest

from backend.engagement_features import FEATURE_NAMES
from backend.engagement_model import (
    ABSTAIN_BAND,
    MIN_COVERAGE_TO_DECIDE,
    EngagementModel,
    load_or_none,
)


def _model(**over) -> EngagementModel:
    """A model with one strong weight on frac_on_task_action."""
    n = len(FEATURE_NAMES)
    coefficients = [0.0] * n
    coefficients[FEATURE_NAMES.index("frac_on_task_action")] = 4.0
    kw = {
        "coefficients": coefficients,
        "intercept": 0.0,
        "feature_names": list(FEATURE_NAMES),
        "mean": [0.0] * n,
        "scale": [1.0] * n,
    }
    kw.update(over)
    return EngagementModel(**kw)


def _features(**named) -> list[float]:
    values = [0.0] * len(FEATURE_NAMES)
    for name, value in named.items():
        values[FEATURE_NAMES.index(name)] = value
    return values


def test_mismatched_lengths_are_rejected_at_construction():
    """Silently misaligned weights are the failure worth guarding hardest."""
    with pytest.raises(ValueError, match="inconsistent"):
        EngagementModel([1.0, 2.0], 0.0, list(FEATURE_NAMES), [0.0], [1.0])


def test_wrong_feature_count_is_rejected_at_call():
    with pytest.raises(ValueError, match="expected"):
        _model().probability([0.1, 0.2, 0.3])


def test_probability_moves_with_the_weighted_feature():
    model = _model()
    low = model.probability(_features(coverage=1.0, frac_on_task_action=0.0))
    high = model.probability(_features(coverage=1.0, frac_on_task_action=1.0))
    assert high > low
    assert 0.0 <= low <= 1.0 and 0.0 <= high <= 1.0


def test_extreme_input_does_not_overflow():
    """A huge negative logit must saturate, not raise."""
    model = _model(coefficients=[1e6] * len(FEATURE_NAMES))
    p = model.probability(_features(coverage=1.0, frac_off_task_action=-1.0))
    assert 0.0 <= p <= 1.0


def test_thin_coverage_refuses_regardless_of_probability():
    """A verdict from almost no readable frames is not a verdict."""
    verdict = _model().classify(
        _features(coverage=MIN_COVERAGE_TO_DECIDE / 2,
                  frac_on_task_action=1.0))
    assert verdict.label is None
    assert verdict.abstained
    assert "coverage" in verdict.reason


def test_probability_near_the_boundary_abstains():
    """The band exists so a coin flip is reported as no answer."""
    model = _model(coefficients=[0.0] * len(FEATURE_NAMES))  # always p = 0.5
    verdict = model.classify(_features(coverage=1.0))
    assert verdict.label is None
    assert verdict.abstained
    assert abs(verdict.probability - 0.5) < ABSTAIN_BAND


def test_confident_windows_are_decided():
    model = _model()
    on = model.classify(_features(coverage=1.0, frac_on_task_action=1.0))
    off = model.classify(_features(coverage=1.0, frac_on_task_action=-1.0))
    assert on.label == "on" and not on.abstained
    assert off.label == "off" and not off.abstained


def test_explain_names_the_feature_that_moved_it():
    """A verdict that cannot be interrogated is worse than a rule that can."""
    text = _model().explain(_features(coverage=1.0, frac_on_task_action=1.0))
    assert "frac_on_task_action" in text


def test_round_trip_through_json_preserves_behaviour(tmp_path: Path):
    """The artifact is coefficients, so it must reload bit-for-bit."""
    model = _model()
    features = _features(coverage=1.0, frac_on_task_action=0.7)
    before = model.probability(features)
    path = model.save(tmp_path / "m.json")
    after = EngagementModel.load(path).probability(features)
    assert after == pytest.approx(before)


def test_saved_model_is_readable_json_not_a_pickle(tmp_path: Path):
    """The artifact must carry no code and stay diffable."""
    path = _model().save(tmp_path / "m.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert set(payload) == {"coefficients", "intercept", "feature_names",
                            "mean", "scale"}
    assert payload["feature_names"] == list(FEATURE_NAMES)


def test_missing_model_degrades_to_none_not_an_exception(tmp_path: Path):
    """A missing model must fall back to the rules, never break the pipeline."""
    assert load_or_none(tmp_path / "absent.json") is None


def test_corrupt_model_degrades_to_none(tmp_path: Path):
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"coefficients": [1.0]}), encoding="utf-8")
    assert load_or_none(path) is None


def test_zero_scale_does_not_divide_by_zero():
    """A constant feature has zero variance; standardising it must not blow up."""
    n = len(FEATURE_NAMES)
    model = _model(scale=[0.0] * n)
    p = model.probability(_features(coverage=1.0, frac_on_task_action=1.0))
    assert 0.0 <= p <= 1.0
