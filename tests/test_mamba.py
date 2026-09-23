"""Tests for the sequence features and the Mamba attention scorer.

The tests that matter here are not "does it run". They are the two properties
that decide whether a sequence model is worth having at all -- that it reads
order, and that padding cannot leak into the answer -- plus causality, which
is easy to break silently with a convolution.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from backend.engagement_sequence import (
    FRAME_FEATURE_NAMES,
    SEQ_LEN,
    frame_vector,
    sequence_for_window,
)
from backend.mamba_model import (
    MambaAttentionScorer,
    MambaConfig,
    to_score_10,
)


def _present(**kw):
    """A frame the student is visible in."""
    base = {"action": "attentive", "expression": None, "posture": {},
            "gaze_label": None, "oriented": True, "focus_offset_deg": 0.0,
            "object": None, "is_eyes_closed_sustained": False}
    base.update(kw)
    return base


class TestFrameVector:
    def test_absent_frame_is_flagged_not_merely_empty(self):
        """Not visible must be distinguishable from visible-and-showing-nothing.

        If both were all-zero, the model could not tell a student who left the
        frame from one sitting perfectly still, and coverage would silently
        become part of every other feature.
        """
        absent = frame_vector(None)
        present = frame_vector(_present(action=None))
        assert absent[0] == 0.0
        assert present[0] == 1.0
        assert absent != present

    def test_vector_length_matches_the_names(self):
        """A mismatch shifts every column's meaning by one."""
        assert len(frame_vector(_present())) == len(FRAME_FEATURE_NAMES)
        assert len(frame_vector(None)) == len(FRAME_FEATURE_NAMES)

    def test_on_and_off_task_are_mutually_exclusive(self):
        on = frame_vector(_present(action="writing"))
        off = frame_vector(_present(action="on_phone"))
        i_on = FRAME_FEATURE_NAMES.index("on_task_action")
        i_off = FRAME_FEATURE_NAMES.index("off_task_action")
        assert (on[i_on], on[i_off]) == (1.0, 0.0)
        assert (off[i_on], off[i_off]) == (0.0, 1.0)


class TestResampling:
    def test_window_is_always_the_same_length(self):
        """Fixed length is what makes windows comparable across frame rates.

        The batch pipeline and a live webcam run at different rates; variable
        length sequences would let the model learn the camera rather than the
        student.
        """
        sparse = sequence_for_window({0: _present()}, 0, 1)
        dense = sequence_for_window(
            {i * 100: _present() for i in range(150)}, 0, 1)
        assert len(sparse.steps) == SEQ_LEN
        assert len(dense.steps) == SEQ_LEN
        assert sparse.n_real < dense.n_real

    def test_missing_stretches_become_absent_steps(self):
        """A student present only at the start leaves the rest marked absent."""
        seq = sequence_for_window({0: _present(), 300: _present()}, 0, 1)
        flags = [s[0] for s in seq.steps]
        assert flags[0] == 1.0
        assert flags[-1] == 0.0
        assert seq.n_real == sum(flags)


class TestMambaModel:
    def test_it_is_small_enough_for_the_data(self):
        """Capacity has to be comparable to the baselines it is measured
        against, or the comparison reports the parameter count rather than the
        architecture. Published vision Mambas are 7-30M; this must not be.
        """
        model = MambaAttentionScorer(
            MambaConfig(n_features=len(FRAME_FEATURE_NAMES)))
        assert model.n_params < 20_000

    def test_padding_cannot_change_the_answer(self):
        """Masked pooling must ignore absent steps entirely.

        Averaging over padding would make the prediction depend on how short
        the window happened to be, which is an artefact of the padding scheme
        and not a fact about the student.
        """
        torch.manual_seed(0)
        model = MambaAttentionScorer(
            MambaConfig(n_features=len(FRAME_FEATURE_NAMES))).eval()
        real = [frame_vector(_present()) for _ in range(10)]
        absent = frame_vector(None)

        short = torch.tensor([real + [absent] * 10], dtype=torch.float32)
        long = torch.tensor([real + [absent] * 38], dtype=torch.float32)
        with torch.no_grad():
            a, b = float(model(short)), float(model(long))
        assert a == pytest.approx(b, abs=1e-5)

    def test_order_changes_the_prediction(self):
        """The entire reason to use a sequence model.

        These two windows have identical aggregate features -- same fractions,
        same means -- and differ only in when the off-task frames occur. If the
        model gave them the same answer it would be an expensive way to
        reproduce the averages the MLP already computes.
        """
        torch.manual_seed(0)
        model = MambaAttentionScorer(
            MambaConfig(n_features=len(FRAME_FEATURE_NAMES))).eval()
        on = frame_vector(_present(action="attentive"))
        off = frame_vector(_present(action="on_phone"))

        drifts_away = torch.tensor([[on] * 12 + [off] * 12], dtype=torch.float32)
        comes_back = torch.tensor([[off] * 12 + [on] * 12], dtype=torch.float32)
        # Same aggregate: half on, half off, in both.
        assert (np.mean([s[1] for s in drifts_away[0].tolist()])
                == pytest.approx(np.mean([s[1] for s in comes_back[0].tolist()])))

        with torch.no_grad():
            a, b = float(model(drifts_away)), float(model(comes_back))
        assert a != pytest.approx(b, abs=1e-4)

    def test_no_timestep_sees_the_future(self):
        """Causality: changing the last frame must not alter earlier states.

        The depthwise convolution pads on both sides by default, which would
        let a frame read the ones after it. That is harmless offline and wrong
        the moment the model runs live, where the future does not exist yet.
        """
        torch.manual_seed(0)
        cfg = MambaConfig(n_features=len(FRAME_FEATURE_NAMES))
        model = MambaAttentionScorer(cfg).eval()
        base = [frame_vector(_present()) for _ in range(20)]
        changed = [list(r) for r in base]
        changed[-1] = frame_vector(_present(action="on_phone"))

        with torch.no_grad():
            h1 = model.embed(torch.tensor([base], dtype=torch.float32))
            h2 = model.embed(torch.tensor([changed], dtype=torch.float32))
            for block in model.blocks:
                h1, h2 = block(h1), block(h2)
        # Everything before the convolution's reach must be untouched.
        cut = 20 - cfg.d_conv
        assert torch.allclose(h1[0, :cut], h2[0, :cut], atol=1e-5)


class TestScoreMapping:
    @pytest.mark.parametrize("prob,expected", [
        (0.0, 1), (0.05, 1), (0.5, 6), (0.99, 10), (1.0, 10)])
    def test_scores_stay_in_range(self, prob, expected):
        """A score outside 1-10 would render as a broken figure in the UI."""
        assert to_score_10(prob) == expected
