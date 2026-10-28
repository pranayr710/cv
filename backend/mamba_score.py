"""Score the same windows with the Mamba sequence model, alongside the MLP.

The shipping score comes from a logistic model over 21 aggregate features.
Aggregates are averages, and averaging destroys order: a student who looks
away and comes back and a student who drifts away and stays away produce
nearly the same aggregate vector. ``backend/mamba_model.py`` reads the window
as a sequence instead, which is the one thing the aggregates cannot represent.

Measured out-of-fold on 179 hand-labelled windows, grouped by student, the
two land close together -- 0.609 for the MLP against 0.592 for Mamba. That
gap is well inside the noise of 179 labels, so neither replaces the other and
**both are reported**. Two numbers that agree are evidence; one number with a
second kept out of sight is not.

What this module adds over calling the model directly: the model wants a fixed
48-step grid covering a 15-second window, and the running pipeline delivers
frames one at a time at a rate that varies between a webcam and a batch run.
This keeps the per-student buffer and resamples onto that grid, using the
same window and stride boundaries :class:`backend.attention_score.AttentionScorer`
uses, so the two scores on screen always describe the same span. They would be
incomparable otherwise, and being incomparable is not visible on screen.
"""
from __future__ import annotations

import pickle
from collections import deque
from pathlib import Path
from typing import Any

from backend.engagement_features import MIN_COVERAGE, STRIDE_MS, WINDOW_MS
from backend.engagement_sequence import (
    FRAME_FEATURE_NAMES,
    sequence_for_window,
)

#: Where ``tools/train/train_mamba_scorer.py`` leaves its trained weights.
DEFAULT_MODEL = Path("models/mamba_attention_model.pt")

#: Below this share of the 48 grid slots filled with real frames, the window
#: is mostly padding and the masked mean is taken over too little to mean
#: anything.
#:
#: This is MIN_COVERAGE from backend.engagement_features, not a number chosen
#: here, and that matters more than its value. The two scorers sit beside each
#: other on screen, so they have to abstain on the same windows -- otherwise a
#: student's two numbers are averages over different spans and the viewer
#: reads a difference in the data as a difference in judgement. At 0.5 the
#: sequence model scored 20 windows where the aggregate model scored 51 on the
#: same recording, because a 10.5-second clip cannot fill a 15-second window
#: and half its grid slots are padding by construction.
MIN_REAL_STEPS_FRACTION: float = MIN_COVERAGE


class MambaScorer:
    """Rolling per-student buffer feeding the trained sequence model.

    Falls back silently when there is no checkpoint or no torch: every
    :meth:`update` returns ``None`` and :attr:`available` is ``False``, so the
    pipeline runs unchanged and the page simply has one score to show instead
    of two. A missing optional model must never take the working one down.
    """

    def __init__(self, model_path: Path | None = None,
                 window_ms: int = WINDOW_MS,
                 stride_ms: int = STRIDE_MS) -> None:
        self.window_ms = window_ms
        self.stride_ms = stride_ms
        self.model = None
        self._torch = None
        self._students: dict[int, deque] = {}
        self._last_window: dict[int, int] = {}
        self._scores: dict[int, list[float]] = {}
        self._load(Path(model_path) if model_path else DEFAULT_MODEL)

    def _load(self, path: Path) -> None:
        if not path.exists():
            return
        try:
            import torch

            from backend.mamba_model import MambaAttentionScorer, MambaConfig
        except ImportError:
            return
        try:
            blob = torch.load(path, map_location="cpu", weights_only=False)
            names = tuple(blob.get("feature_names") or ())
            # The per-frame vector is built here and was built by the trainer.
            # If the two ever drift apart the model is being fed columns that
            # mean something else, which produces a plausible-looking number
            # from nonsense -- the one failure mode a score on screen cannot
            # show. Refusing is the only safe answer.
            if names and names != FRAME_FEATURE_NAMES:
                return
            cfg = MambaConfig(**(blob.get("config") or {}))
            model = MambaAttentionScorer(cfg)
            model.load_state_dict(blob["state_dict"])
            model.eval()
        except (KeyError, RuntimeError, TypeError, ValueError, EOFError,
                OSError, pickle.UnpicklingError):
            # Broad on purpose, and only here. Everything this can raise means
            # the same thing -- there is no usable second model -- and the
            # correct response to all of it is to run with one score instead
            # of taking the whole pipeline down over an optional extra. A
            # truncated file raises UnpicklingError rather than any of the
            # tidier errors, which a test of this function found.
            return
        self.model = model
        # Platt coefficients, if the checkpoint carries them. Without one
        # this model saturates: measured on its own training footage it put
        # 23% of windows at exactly 10/10, and on a two-person webcam
        # session it returned 10/10 for every single window -- the same
        # number for a student on their phone and a student attentive
        # throughout, which is not a measurement.
        calib = blob.get("calibration")
        self.calibration = (
            (float(calib[0]), float(calib[1])) if calib else None)
        self._torch = torch

    @property
    def available(self) -> bool:
        """Whether a usable checkpoint was found."""
        return self.model is not None

    def update(self, person_id: int, timestamp_ms: int,
               features: dict[str, Any]) -> int | None:
        """Record one frame; return a 1-10 score when a window completes.

        Returns ``None`` between windows, matching
        :meth:`backend.attention_score.AttentionScorer.update`, so a caller
        can hold the previous value rather than blanking the display every
        frame.
        """
        if self.model is None:
            return None

        buffer = self._students.setdefault(person_id, deque())
        buffer.append((timestamp_ms, features))
        cutoff = timestamp_ms - self.window_ms
        while buffer and buffer[0][0] < cutoff:
            buffer.popleft()

        last = self._last_window.get(person_id, -1)
        if last >= 0 and timestamp_ms - last < self.stride_ms:
            return None
        if not buffer:
            return None

        start = buffer[0][0]
        sequence = sequence_for_window({t: f for t, f in buffer}, start,
                                       person_id)
        if sequence.n_real < len(sequence.steps) * MIN_REAL_STEPS_FRACTION:
            return None

        self._last_window[person_id] = timestamp_ms
        probability = self._probability(sequence.steps)
        self._scores.setdefault(person_id, []).append(probability)
        return to_score_10(probability)

    def _probability(self, steps: list[list[float]]) -> float:
        """The calibrated probability, or the raw one if never calibrated."""
        import math

        torch = self._torch
        with torch.no_grad():
            x = torch.tensor([steps], dtype=torch.float32)
            p = float(torch.sigmoid(self.model(x))[0])
        if self.calibration is None:
            return p
        a, b = self.calibration
        p = min(max(p, 1e-6), 1.0 - 1e-6)
        z = a * math.log(p / (1.0 - p)) + b
        return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, z))))

    def raw_probability(self, steps: list[list[float]]) -> float:
        """Before calibration. Used by the fitter, which needs the input."""
        torch = self._torch
        with torch.no_grad():
            x = torch.tensor([steps], dtype=torch.float32)
            return float(torch.sigmoid(self.model(x))[0])

    def session_score(self, person_id: int) -> int | None:
        """The student's Mamba score over the session so far, 1-10.

        The mean of the probabilities, not of the rounded scores, so the
        rounding happens once instead of accumulating -- the same rule the
        aggregate scorer follows, because the two sit next to each other and a
        difference in arithmetic would read as a difference in judgement.
        """
        scores = self._scores.get(person_id)
        if not scores:
            return None
        return to_score_10(sum(scores) / len(scores))

    def summary(self) -> dict[int, dict[str, Any]]:
        """Every student's Mamba session score and how much it rests on."""
        return {pid: {"score_10": self.session_score(pid),
                      "windows_scored": len(scores)}
                for pid, scores in self._scores.items()}


def to_score_10(probability: float) -> int:
    """A 1-10 score from a probability.

    Deliberately re-exported from :mod:`backend.attention_score` rather than
    reimplemented. The two mappings had drifted apart -- one rounded
    ``p * 9 + 1`` and the other floored ``p * 10 + 1`` -- which at p=0.5 gives
    5 and 6. Shown side by side those two numbers read as the models
    disagreeing when the models agreed exactly and the arithmetic did not.
    """
    from backend.attention_score import to_score_10 as canonical

    return canonical(probability)
