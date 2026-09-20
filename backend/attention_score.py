"""Per-student attention score, computed live and improved by new labels.

This is the piece that turns the trained window model into something the
running system uses. It maintains, per student, a rolling buffer of the frames
seen so far; once a student has been visible long enough to fill a window, it
summarises that window into the same feature vector the model was trained on
and asks for a verdict and a probability. The probability becomes a 1-10
attention score.

Why the score is a probability and not a count of on-task frames. The rule
pipeline already reports an on-task percentage, and on 179 hand-labelled
windows it agreed with a human 45% of the time -- worse than chance, because it
answers "off" for 82% of windows. Counting frames a broken rule labelled does
not produce a better number by averaging. The learned model reaches 60% on the
same windows, so the score is built from it instead.

Improving itself, concretely. There is no online weight update here, and that
is deliberate: a model that adjusts to its own predictions drifts toward
whatever it already believed, and nothing in the running system provides a
correction signal. What it does instead is decide which windows would be worth
a human's time, and record them. Windows the model is least certain about
teach it the most per label, so :meth:`AttentionScorer.uncertain_windows`
returns them ready to be fed back through the labelling tool. Label those,
retrain, drop in the new coefficients: the loop closes through a person, which
is the only place new information can enter.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.engagement_features import (
    MIN_COVERAGE,
    STRIDE_MS,
    WINDOW_MS,
    window_features,
)
from backend.engagement_model import EngagementModel, load_or_none

#: Frames-per-second assumed when converting the window length into a frame
#: count. Overridden per session by the caller when the real rate is known.
DEFAULT_FPS: float = 3.0

#: Probability distance from 0.5 below which a window is worth labelling.
#: Windows the model already decides confidently teach it little; the ones near
#: the boundary are where a human's judgement adds the most.
UNCERTAIN_BAND: float = 0.15


@dataclass
class _StudentState:
    """Rolling per-student buffer and the scores derived from it."""

    frames: deque = field(default_factory=deque)
    last_window_ms: int = -1
    scores: list[tuple[int, float]] = field(default_factory=list)
    uncertain: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class AttentionReading:
    """One student's attention at one moment.

    Attributes:
        person_id: Who this is about.
        score_10: Attention on a 1-10 scale, or ``None`` when the model
            declined to score this window.
        probability: The model's fitted probability of on-task.
        label: ``"on"``, ``"off"`` or ``None``.
        reason: Why, in words, including the features that moved it.
        window_start_ms: Start of the window this describes.
    """

    person_id: int
    score_10: int | None
    probability: float | None
    label: str | None
    reason: str
    window_start_ms: int


def to_score_10(probability: float) -> int:
    """Map an on-task probability onto 1-10.

    Linear and stated plainly: this is a rescaling of the model's confidence,
    not an independently calibrated intensity. Calling 0.9 a nine asserts only
    that it is more confident than a five, which is what the number is for.
    """
    return max(1, min(10, round(probability * 9) + 1))


class AttentionScorer:
    """Turns a stream of per-frame records into per-student attention scores.

    Falls back cleanly when no model has been trained: :attr:`available` is
    ``False`` and every reading comes back unscored, so a caller can run the
    pipeline unchanged and the rule verdict remains the only answer.
    """

    def __init__(self, model: EngagementModel | None = None,
                 fps: float = DEFAULT_FPS,
                 window_ms: int = WINDOW_MS,
                 stride_ms: int = STRIDE_MS) -> None:
        self.model = model if model is not None else load_or_none()
        self.fps = fps
        self.window_ms = window_ms
        self.stride_ms = stride_ms
        self.expected_frames = max(round(fps * window_ms / 1000.0), 1)
        self._students: dict[int, _StudentState] = {}

    @property
    def available(self) -> bool:
        """Whether a trained model was found."""
        return self.model is not None

    def update(self, person_id: int, timestamp_ms: int,
               features: dict[str, Any]) -> AttentionReading | None:
        """Record one frame for one student; score when a window completes.

        Args:
            person_id: Stable identity.
            timestamp_ms: Frame time.
            features: The node ``features`` dict from the scene graph.

        Returns:
            A reading when this frame completed a window, otherwise ``None``.
            Returning only on completion keeps the score stable between
            windows rather than flickering every frame.
        """
        state = self._students.setdefault(person_id, _StudentState())
        state.frames.append((timestamp_ms, features))

        cutoff = timestamp_ms - self.window_ms
        while state.frames and state.frames[0][0] < cutoff:
            state.frames.popleft()

        if state.last_window_ms >= 0 and (
                timestamp_ms - state.last_window_ms < self.stride_ms):
            return None
        if not state.frames:
            return None

        start = state.frames[0][0]
        vector = window_features([f for _, f in state.frames],
                                 self.expected_frames)
        if vector[0] < MIN_COVERAGE:
            return None

        state.last_window_ms = timestamp_ms
        if self.model is None:
            return AttentionReading(person_id, None, None, None,
                                    "no trained model; rules apply", start)

        verdict = self.model.classify(vector)
        probability = None if verdict.probability != verdict.probability \
            else verdict.probability
        score = None if probability is None else to_score_10(probability)
        if score is not None:
            state.scores.append((start, float(probability)))

        if probability is not None and abs(probability - 0.5) < UNCERTAIN_BAND:
            state.uncertain.append({
                "person_id": person_id,
                "start_ms": start,
                "end_ms": start + self.window_ms,
                "probability": round(float(probability), 4),
                "features": list(vector),
            })

        reason = verdict.reason
        if not verdict.abstained:
            reason = f"{verdict.reason} ({self.model.explain(vector)})"
        return AttentionReading(person_id, score, probability, verdict.label,
                                reason, start)

    def session_score(self, person_id: int) -> int | None:
        """A student's attention over the whole session so far, 1-10.

        The mean of the window probabilities rather than of the 1-10 values,
        so the rounding happens once at the end instead of accumulating.
        """
        state = self._students.get(person_id)
        if not state or not state.scores:
            return None
        mean = sum(p for _, p in state.scores) / len(state.scores)
        return to_score_10(mean)

    def summary(self) -> dict[int, dict[str, Any]]:
        """Every student's session score and how much it rests on."""
        out: dict[int, dict[str, Any]] = {}
        for person_id, state in self._students.items():
            out[person_id] = {
                "score_10": self.session_score(person_id),
                "windows_scored": len(state.scores),
                "windows_uncertain": len(state.uncertain),
            }
        return out

    def uncertain_windows(self, limit: int = 100) -> list[dict[str, Any]]:
        """The windows a human should label next, least certain first.

        This is the improvement loop. A model retrained on windows it was
        already confident about learns nothing; the ones near its decision
        boundary are where a label carries the most information. Feed these
        back through the labelling tool, retrain, and replace the coefficients.
        """
        pooled = [w for s in self._students.values() for w in s.uncertain]
        pooled.sort(key=lambda w: abs(w["probability"] - 0.5))
        return pooled[:limit]

    def save_uncertain(self, path: Path, limit: int = 100) -> int:
        """Write the next windows to label as JSON. Returns how many."""
        import json

        windows = self.uncertain_windows(limit)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(windows, indent=1), encoding="utf-8")
        return len(windows)


def annotate_attention(graph: dict, scorer: AttentionScorer) -> dict:
    """Attach the learned attention score to every student in one frame.

    This is the pipeline's entry point. It runs after the scene graph, the
    layout and the action rules, because the window features it needs are
    built from what those produce.

    Each node gains:

    * ``attention_score`` -- 1-10, or ``None`` when the model has not yet seen
      enough of this student to judge. A student who just entered frame has no
      score for the first window, and inventing one would be the opposite of
      what the rest of this system does.
    * ``attention_label`` -- the model's on/off verdict.
    * ``attention_reason`` -- which features moved it, so a teacher shown a low
      score can be told why.
    * ``attention_source`` -- ``"model"`` or ``"rule"``, so the two are never
      confused in the output.

    The rule verdict is left untouched in ``engagement``. It is no longer what
    the score is built from, but removing it would throw away the comparison
    that shows why the model replaced it -- on 179 hand-labelled windows the
    rule agreed with a human 45% of the time against the model's 60%.

    Args:
        graph: One frame's scene graph, already annotated.
        scorer: A live :class:`AttentionScorer`, carried across frames.

    Returns:
        The same graph, mutated in place and returned for chaining.
    """
    timestamp = int(graph.get("timestamp_ms") or 0)
    for node in graph.get("nodes", []):
        person_id = node.get("person_id")
        features = node.get("features")
        if not person_id or person_id <= 0 or not isinstance(features, dict):
            continue

        reading = scorer.update(int(person_id), timestamp, features)
        if reading is None:
            # Mid-window: hold the previous score rather than blanking it, so
            # the number on screen is steady instead of flickering off between
            # windows.
            previous = scorer.session_score(int(person_id))
            features.setdefault("attention_score", previous)
            features.setdefault("attention_label", None)
            features.setdefault("attention_reason", "within the current window")
            features.setdefault("attention_source",
                                "model" if scorer.available else "rule")
            continue

        features["attention_score"] = reading.score_10
        features["attention_label"] = reading.label
        features["attention_reason"] = reading.reason
        features["attention_source"] = "model" if scorer.available else "rule"
        features["attention_session_score"] = scorer.session_score(
            int(person_id))
    return graph
