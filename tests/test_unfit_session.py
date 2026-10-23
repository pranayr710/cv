"""A refusal has to hold between windows, not only at them.

The aggregate model correctly refused a one-person webcam session -- three
of its twenty-one features are peer and room measurements that do not exist
with one student -- and the dashboard showed a dash for it. The Mamba score
beside it still read 10/10.

The cause was in the glue rather than in either model. ``update()`` returns
None on every frame between completed windows, so a caller that read the
refusal off the *current* reading saw nothing to act on for most frames and
went on publishing the other model's session score. The refusal is a
property of the session, not of the one window that happened to close.

Leaving one of the two on screen is worse than leaving both: a lone
saturated number reads as a verdict, where two contradictory ones at least
look wrong.
"""
import pytest

from backend.attention_score import AttentionScorer, annotate_attention
from backend.engagement_model import EngagementModel


class RefusingModel(EngagementModel):
    """Stands in for a model asked about footage it was not fitted on."""

    def __init__(self, names):
        super().__init__([0.0] * len(names), 0.0, names,
                         [0.0] * len(names), [1.0] * len(names),
                         lo=[0.0] * len(names), hi=[0.0] * len(names))


class StubMamba:
    """Always has a score, which is the point -- it must be suppressed."""

    available = True

    def update(self, person_id, timestamp_ms, features):
        return 10

    def session_score(self, person_id):
        return 10


def _frame(ms, **extra):
    f = {"action": "attentive", "gaze_label": "teacher", "oriented": True,
         "expression": "neutral", "posture": {"vertical_lean": -0.13}}
    f.update(extra)
    return {"timestamp_ms": ms,
            "nodes": [{"id": 1, "person_id": 1, "role": "student",
                       "features": f}]}


@pytest.fixture
def scorer():
    from backend.engagement_features import (
        BASE_FEATURE_NAMES, BODY_FEATURE_NAMES, PEER_FEATURE_NAMES)

    names = (list(BASE_FEATURE_NAMES) + list(BODY_FEATURE_NAMES)
             + list(PEER_FEATURE_NAMES))
    return AttentionScorer(model=RefusingModel(names))


class TestTheRefusalSticks:
    def test_it_is_remembered_after_the_window_that_caused_it(self, scorer):
        for ms in range(0, 20_000, 400):
            annotate_attention(_frame(ms), scorer, StubMamba())
        assert scorer.is_unfit(1)

    def test_an_unseen_student_is_not_unfit(self, scorer):
        assert not scorer.is_unfit(99)

    def test_the_second_score_is_suppressed_on_every_frame(self, scorer):
        """The bug exactly: it was suppressed only on the frames where a
        window closed, which is a handful of the hundreds in a session."""
        published = []
        for ms in range(0, 30_000, 400):
            g = annotate_attention(_frame(ms), scorer, StubMamba())
            f = g["nodes"][0]["features"]
            if scorer.is_unfit(1):
                published.append(f.get("attention_score_mamba"))
        assert published, "the model never refused; the fixture is wrong"
        assert set(published) == {None}, (
            f"the sequence model published {len(published) - published.count(None)}"
            " scores on frames where the window was refused")

    def test_neither_score_survives(self, scorer):
        for ms in range(0, 30_000, 400):
            g = annotate_attention(_frame(ms), scorer, StubMamba())
        f = g["nodes"][0]["features"]
        assert f.get("attention_score") is None
        assert f.get("attention_score_mamba") is None


class TestAFittedSessionIsUnaffected:
    def test_a_model_that_does_not_refuse_leaves_both_scores(self):
        """The gate must not fire on footage the models are for."""
        from backend.engagement_features import (
            BASE_FEATURE_NAMES, BODY_FEATURE_NAMES, PEER_FEATURE_NAMES)

        names = (list(BASE_FEATURE_NAMES) + list(BODY_FEATURE_NAMES)
                 + list(PEER_FEATURE_NAMES))
        # A wide range refuses nothing.
        model = EngagementModel([0.0] * len(names), 0.0, names,
                                [0.0] * len(names), [1.0] * len(names),
                                lo=[-1e6] * len(names), hi=[1e6] * len(names))
        scorer = AttentionScorer(model=model)
        for ms in range(0, 30_000, 400):
            g = annotate_attention(_frame(ms), scorer, StubMamba())
        assert not scorer.is_unfit(1)
        assert g["nodes"][0]["features"].get("attention_score_mamba") == 10
