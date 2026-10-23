"""The score must not claim a precision the model does not have.

A student came out 1/10 from the aggregate model and 10/10 from the Mamba
sequence model, on the same window. Neither was disagreeing. Both were
saturated, in opposite directions.

Measured out of fold on 557 hand-labelled windows, grouped by student: the
aggregate model claimed more than 90% confidence on 65% of them and was
right on 59.9% of those. Windows it scored 0.98 were on task 62% of the
time. Because to_score_10 is a linear map of the probability, 58% of every
score came out as exactly 1 or exactly 10.

So the probability is corrected before it becomes a score. The verdict is
not: its ordering is what every quoted accuracy measured, and calibration is
monotonic, so changing it would move the decision boundary for no gain.
"""
import json
import math

import pytest

from backend.engagement_model import EngagementModel


def _model(calibration=None, n=2):
    m = EngagementModel(
        coefficients=[1.0] * n, intercept=0.0,
        feature_names=["coverage", "a"][:n],
        mean=[0.0] * n, scale=[1.0] * n)
    m.calibration = calibration
    return m


class TestWithoutACalibration:
    def test_the_probability_is_returned_unchanged(self):
        m = _model()
        v = [0.9, 2.0]
        assert m.calibrated_probability(v) == m.probability(v)

    def test_a_nan_stays_a_nan(self):
        m = _model(calibration=(0.04, 0.12))
        got = m.calibrated_probability([0.9, float("nan")])
        assert got != got


class TestWithACalibration:
    def test_it_pulls_an_overconfident_probability_in(self):
        """The whole point: 0.98 from a model right 62% of the time should
        not be reported as 0.98."""
        m = _model(calibration=(0.0431, 0.1212))
        raw = m.probability([0.9, 8.0])
        assert raw > 0.95
        assert m.calibrated_probability([0.9, 8.0]) < 0.7

    def test_it_is_monotonic(self):
        """Ordering has to survive, or the verdict and the score would
        contradict each other."""
        m = _model(calibration=(0.0431, 0.1212))
        values = [-8.0, -2.0, 0.0, 2.0, 8.0]
        cal = [m.calibrated_probability([0.9, v]) for v in values]
        assert cal == sorted(cal)

    def test_it_stays_a_probability(self):
        m = _model(calibration=(0.0431, 0.1212))
        for v in (-50.0, 0.0, 50.0):
            p = m.calibrated_probability([0.9, v])
            assert 0.0 <= p <= 1.0

    def test_an_extreme_logit_does_not_overflow(self):
        m = _model(calibration=(5.0, 0.0))
        assert not math.isnan(m.calibrated_probability([0.9, 1e6]))

    def test_the_verdict_still_uses_the_raw_probability(self):
        """Calibration is for the score, not the decision. If the verdict
        moved too, every accuracy quoted for this model would be stale."""
        m = _model(calibration=(0.0431, 0.1212))
        v = [0.9, 8.0]
        assert m.classify(v).probability == pytest.approx(m.probability(v))


class TestItSurvivesASaveAndLoad:
    def test_it_round_trips(self, tmp_path):
        path = tmp_path / "m.json"
        _model(calibration=(0.0431, 0.1212)).save(path)
        back = EngagementModel.load(path)
        assert back.calibration == pytest.approx((0.0431, 0.1212))

    def test_a_model_saved_without_one_still_loads(self, tmp_path):
        path = tmp_path / "old.json"
        path.write_text(json.dumps({
            "coefficients": [1.0, 1.0], "intercept": 0.0,
            "feature_names": ["coverage", "a"],
            "mean": [0.0, 0.0], "scale": [1.0, 1.0],
        }), encoding="utf-8")
        assert EngagementModel.load(path).calibration is None


class TestTheShippingModel:
    def test_it_is_calibrated(self):
        from backend.engagement_model import load_or_none

        m = load_or_none()
        if m is None:
            pytest.skip("no trained model present")
        assert m.calibration is not None, (
            "the shipping model is uncalibrated, so its scores saturate at 1 "
            "and 10 -- run tools/train/fit_calibration.py")

    def test_scores_no_longer_pin_to_the_ends(self):
        """The symptom, on the footage that produced it."""
        from pathlib import Path

        from backend.attention_score import to_score_10
        from backend.engagement_features import iter_windows
        from backend.engagement_model import load_or_none

        graph = Path("outputs/final2/live_graph.jsonl")
        m = load_or_none()
        if m is None or not graph.exists() or m.calibration is None:
            pytest.skip("trained calibrated model or training graph absent")

        scores = []
        for w in iter_windows(graph):
            v = m.classify(tuple(w.features))
            if v.abstained or v.probability != v.probability:
                continue
            scores.append(to_score_10(m.calibrated_probability(
                tuple(w.features))))

        pinned = sum(1 for s in scores if s in (1, 10)) / max(len(scores), 1)
        assert pinned < 0.05, (
            f"{pinned:.0%} of scores sit at 1 or 10; before calibration it "
            "was 58%, which is what made one model read 1/10 against the "
            "other's 10/10")
