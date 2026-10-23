"""The Mamba sequence model, scored live beside the aggregate model.

The point of showing two numbers is that they are comparable. That is the
property these tests defend, and it is easy to lose without anything on screen
looking wrong: two scores over different spans, two different 1-10 mappings,
or a model fed columns in an order it was not trained on all produce a
plausible number from an incomparable measurement.
"""
import pytest

from backend.engagement_features import MIN_COVERAGE, STRIDE_MS, WINDOW_MS
from backend.engagement_sequence import FRAME_FEATURE_NAMES
from backend.mamba_score import MIN_REAL_STEPS_FRACTION, MambaScorer


def _frame(action="attentive", gaze="teacher"):
    return {"action": action, "gaze_label": gaze, "oriented": True,
            "expression": "neutral", "posture": {"vertical_lean": -0.13}}


class TestItDegradesWithoutAModel:
    def test_a_missing_checkpoint_is_not_an_error(self, tmp_path):
        """A missing optional model must leave one score on screen, not take
        the working one down with it."""
        scorer = MambaScorer(model_path=tmp_path / "nothing.pt")
        assert not scorer.available
        assert scorer.update(1, 0, _frame()) is None
        assert scorer.session_score(1) is None

    def test_a_corrupt_checkpoint_is_not_an_error(self, tmp_path):
        path = tmp_path / "model.pt"
        path.write_bytes(b"not a torch file")
        assert not MambaScorer(model_path=path).available

    def test_a_checkpoint_from_other_features_is_refused(self, tmp_path):
        """The one failure a score on screen cannot show.

        If the per-frame vector built here and the one the trainer built ever
        drift apart, the model is fed columns that mean something else and
        returns a confident number from nonsense. There is nothing in the
        output that looks wrong, so refusing to load is the only safe answer.
        """
        torch = pytest.importorskip("torch")
        from backend.mamba_model import MambaAttentionScorer, MambaConfig

        path = tmp_path / "model.pt"
        model = MambaAttentionScorer(MambaConfig(n_features=4))
        torch.save({"state_dict": model.state_dict(),
                    "config": {"n_features": 4},
                    "feature_names": ["a", "b", "c", "d"],
                    "seq_len": 48}, path)
        assert not MambaScorer(model_path=path).available

    def test_a_matching_checkpoint_loads(self, tmp_path):
        torch = pytest.importorskip("torch")
        from backend.mamba_model import MambaAttentionScorer, MambaConfig

        path = tmp_path / "model.pt"
        model = MambaAttentionScorer(
            MambaConfig(n_features=len(FRAME_FEATURE_NAMES)))
        torch.save({"state_dict": model.state_dict(),
                    "config": {"n_features": len(FRAME_FEATURE_NAMES)},
                    "feature_names": list(FRAME_FEATURE_NAMES),
                    "seq_len": 48}, path)
        assert MambaScorer(model_path=path).available


class TestItDescribesTheSameSpanAsTheOtherModel:
    """Two scores side by side are only readable if they measure the same
    thing. Any of these drifting makes the pair misleading rather than
    broken, which is worse."""

    def test_the_window_and_stride_match(self):
        scorer = MambaScorer(model_path=None)
        assert scorer.window_ms == WINDOW_MS
        assert scorer.stride_ms == STRIDE_MS

    def test_the_coverage_gate_matches(self):
        """At 0.5 this abstained on windows the aggregate model scored: 20
        against 51 on a 10.5-second clip, which cannot fill a 15-second
        window, so half its grid slots are padding by construction. A
        student's two numbers were then averages over different spans."""
        assert MIN_REAL_STEPS_FRACTION == MIN_COVERAGE

    def test_the_score_mapping_matches(self):
        from backend.attention_score import to_score_10 as aggregate
        from backend.mamba_score import to_score_10

        for i in range(101):
            assert to_score_10(i / 100.0) == aggregate(i / 100.0)


class TestScoringAWindow:
    @pytest.fixture
    def scorer(self, tmp_path):
        torch = pytest.importorskip("torch")
        from backend.mamba_model import MambaAttentionScorer, MambaConfig

        path = tmp_path / "model.pt"
        model = MambaAttentionScorer(
            MambaConfig(n_features=len(FRAME_FEATURE_NAMES)))
        torch.save({"state_dict": model.state_dict(),
                    "config": {"n_features": len(FRAME_FEATURE_NAMES)},
                    "feature_names": list(FRAME_FEATURE_NAMES),
                    "seq_len": 48}, path)
        return MambaScorer(model_path=path)

    def test_it_abstains_until_a_window_is_covered(self, scorer):
        assert scorer.update(1, 0, _frame()) is None

    def test_a_covered_window_scores_in_range(self, scorer):
        last = None
        for ms in range(0, WINDOW_MS + 1, 250):
            got = scorer.update(1, ms, _frame())
            if got is not None:
                last = got
        assert last is not None
        assert 1 <= last <= 10

    def test_it_scores_at_most_once_per_stride(self, scorer):
        scores = [scorer.update(1, ms, _frame())
                  for ms in range(0, 4 * WINDOW_MS, 250)]
        scored = [s for s in scores if s is not None]
        # One window per stride over the span, give or take the first.
        assert len(scored) <= (4 * WINDOW_MS) // STRIDE_MS + 1

    def test_students_are_scored_independently(self, scorer):
        for ms in range(0, WINDOW_MS + 1, 250):
            scorer.update(1, ms, _frame())
            scorer.update(2, ms, _frame(action="on_phone", gaze="down"))
        assert scorer.session_score(1) is not None
        assert scorer.session_score(2) is not None
        assert set(scorer.summary()) == {1, 2}

    def test_an_unseen_student_has_no_score(self, scorer):
        assert scorer.session_score(99) is None
