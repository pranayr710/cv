"""A model asked about footage it was not fitted for must say so.

The symptom was two scores on one screen that could not both be right: the
aggregate model reporting 1/10 and the Mamba sequence model 10/10 for the
same student, on the same window.

Neither was disagreeing. Both had saturated. On a one-person webcam session,
six of the twenty-one window features are outside the range the model was
fitted on -- there are no peers in a one-person session, and a room focus
needs three students before it can be measured -- and a sigmoid extrapolates
to 0 or to 1 without any sign that it is extrapolating. Measured on that
run: probabilities of 0.023, 0.014 and 0.001 from the aggregate model, every
window 1/10, against 203 windows of 10/10 from the sequence model.

The fix is not a better extrapolation. It is for the model to carry the
range it was fitted on and refuse outside it, which is what this system does
everywhere else when it cannot answer.
"""
import json

import pytest

from backend.engagement_model import (
    MAX_FEATURES_OUT_OF_RANGE,
    OUT_OF_RANGE_TOLERANCE,
    EngagementModel,
)


def _model(lo=None, hi=None, n=4):
    return EngagementModel(
        coefficients=[1.0] * n, intercept=0.0,
        feature_names=["coverage", "a", "b", "c"][:n],
        mean=[0.0] * n, scale=[1.0] * n, lo=lo, hi=hi)


class TestTheModelKnowsWhatItSaw:
    def test_a_value_inside_the_range_is_not_flagged(self):
        m = _model(lo=[0, 0, 0, 0], hi=[1, 1, 1, 1])
        assert m.out_of_range([0.5, 0.5, 0.5, 0.5]) == []

    def test_a_value_outside_the_range_is_named(self):
        m = _model(lo=[0, 0, 0, 0], hi=[1, 1, 1, 1])
        assert m.out_of_range([0.5, 9.0, 0.5, 0.5]) == ["a"]

    def test_a_model_without_a_range_reports_nothing(self):
        """Not 'nothing is out of range' but 'this model cannot tell'. A
        model saved before the range existed must still load and run."""
        assert _model().out_of_range([99.0, 99.0, 99.0, 99.0]) == []

    def test_a_hair_past_the_maximum_is_tolerated(self):
        """A student doing something slightly unusual is not different
        footage."""
        m = _model(lo=[0, 0, 0, 0], hi=[1, 1, 1, 1])
        assert m.out_of_range([0.5, 1.0 + OUT_OF_RANGE_TOLERANCE / 2,
                               0.5, 0.5]) == []

    def test_a_nan_is_not_an_out_of_range_value(self):
        """NaN means a feature could not be computed, which classify()
        already refuses separately and for a different reason."""
        m = _model(lo=[0, 0, 0, 0], hi=[1, 1, 1, 1])
        assert m.out_of_range([0.5, float("nan"), 0.5, 0.5]) == []


class TestItRefusesRatherThanExtrapolating:
    def test_enough_strange_features_abstain(self):
        m = _model(lo=[0, 0, 0, 0], hi=[1, 1, 1, 1])
        v = m.classify([0.9, 9.0, 9.0, 9.0])
        assert v.abstained
        assert v.label is None
        assert "outside the range" in v.reason

    def test_the_reason_names_the_features(self):
        """A refusal nobody can act on is the same as a wrong answer."""
        m = _model(lo=[0, 0, 0, 0], hi=[1, 1, 1, 1])
        v = m.classify([0.9, 9.0, 9.0, 9.0])
        assert "a" in v.reason and "b" in v.reason

    def test_one_strange_feature_still_decides(self):
        """One feature out of range is a student doing something unusual."""
        m = _model(lo=[0, 0, 0, 0], hi=[1, 1, 1, 1])
        v = m.classify([0.9, 9.0, 0.5, 0.5])
        assert "outside the range" not in (v.reason or "")

    def test_the_threshold_is_what_it_says(self):
        m = _model(lo=[0, 0, 0, 0], hi=[1, 1, 1, 1])
        strange = m.out_of_range([0.9, 9.0, 9.0, 9.0])
        assert len(strange) >= MAX_FEATURES_OUT_OF_RANGE

    def test_coverage_is_still_checked_first(self):
        """A thin window is refused for being thin, which is the more
        specific reason and the one the operator can act on."""
        m = _model(lo=[0, 0, 0, 0], hi=[1, 1, 1, 1])
        v = m.classify([0.05, 9.0, 9.0, 9.0])
        assert "coverage" in v.reason


class TestARoomFeatureDisqualifiesOnItsOwn:
    """Why one of two students was scored and the other was not.

    peer_count and the two peer fractions describe the room, not the
    student: if one is outside the fitted range it is outside it for
    everybody in the frame. Counting them against each student individually
    meant that on a two-person session both began every window with two of
    the three strikes already spent, and whichever student happened to have
    a third borderline feature tip over got no score while the other did --
    on footage equally unfamiliar for both.
    """

    def _model(self):
        names = ["coverage", "a", "peer_count", "peer_frac_on_task"]
        return EngagementModel([0.0] * 4, 0.0, names, [0.0] * 4, [1.0] * 4,
                               lo=[0, 0, 0.333, 0.136],
                               hi=[1, 1, 0.600, 0.681])

    def test_peer_count_alone_refuses(self):
        v = self._model().classify([0.9, 0.5, 0.111, 0.4])
        assert v.abstained
        assert "peer_count" in v.reason

    def test_the_reason_says_what_is_wrong_with_the_footage(self):
        """'Three features out of range' tells an operator nothing. 'Fitted
        on rooms of five or more' tells them to use different footage."""
        v = self._model().classify([0.9, 0.5, 0.111, 0.4])
        assert "five or more" in v.reason

    def test_two_students_in_one_frame_are_treated_alike(self):
        """The actual symptom. Same room, so the same verdict."""
        m = self._model()
        a = m.classify([0.9, 0.1, 0.111, 0.4])   # a third feature borderline
        b = m.classify([0.9, 0.5, 0.111, 0.4])   # it is not
        assert a.abstained and b.abstained
        assert ("peer_count" in a.reason) == ("peer_count" in b.reason)

    def test_a_full_room_is_unaffected(self):
        v = self._model().classify([0.9, 0.5, 0.45, 0.4])
        assert "outside the range" not in (v.reason or "")


class TestTheRangeSurvivesASaveAndLoad:
    def test_it_round_trips(self, tmp_path):
        path = tmp_path / "m.json"
        _model(lo=[0, 1, 2, 3], hi=[4, 5, 6, 7]).save(path)
        back = EngagementModel.load(path)
        assert back.lo == [0, 1, 2, 3]
        assert back.hi == [4, 5, 6, 7]

    def test_a_model_saved_without_one_still_loads(self, tmp_path):
        path = tmp_path / "old.json"
        payload = {
            "coefficients": [1.0, 1.0], "intercept": 0.0,
            "feature_names": ["coverage", "a"], "mean": [0.0, 0.0],
            "scale": [1.0, 1.0],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        back = EngagementModel.load(path)
        assert back.lo is None
        assert back.classify([0.9, 0.5]).reason


class TestTheShippingModel:
    """The model in models/ and the footage that exposed this."""

    def test_it_carries_its_fitted_range(self):
        from backend.engagement_model import load_or_none

        m = load_or_none()
        if m is None:
            pytest.skip("no trained model present")
        assert m.lo is not None, (
            "the shipping model has no fitted range, so it cannot tell when "
            "it is extrapolating -- run tools/train/stamp_model_range.py")
        assert len(m.lo) == len(m.feature_names)

    def test_classroom_footage_is_still_scored(self):
        """The gate must not fire on the footage the model is for. If it
        does, every score in the system goes blank."""
        from pathlib import Path

        from backend.engagement_features import iter_windows
        from backend.engagement_model import load_or_none

        graph = Path("outputs/final2/live_graph.jsonl")
        m = load_or_none()
        if m is None or not graph.exists():
            pytest.skip("trained model or training graph not present")

        verdicts = [m.classify(tuple(w.features))
                    for w in iter_windows(graph)]
        refused = sum(1 for v in verdicts
                      if "outside the range" in (v.reason or ""))
        assert refused == 0, (
            f"{refused} of {len(verdicts)} windows of the model's own "
            "training footage were refused as out of range")
