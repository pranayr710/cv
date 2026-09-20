"""A learned window-level engagement verdict, and the rule it competes with.

:func:`backend.engagement.classify_engagement` decides on/off by precedence:
off-task routes are checked, then on-task, first match wins. The ordering comes
from BOSS and is defensible, but the thresholds behind it were set by hand and
have never been fitted to anything.

This is the learned alternative. It consumes the same window feature vector
:mod:`backend.engagement_features` already produces and returns the same
verdict, so the two are interchangeable at the call site and can be measured
against each other on identical windows.

Three properties are kept deliberately, because losing them would make the
learned version worse than the rules it replaces even at higher accuracy:

* **It can abstain.** Coverage is a feature, and a window whose probability
  sits inside a band around the decision boundary returns ``None`` rather than
  a coin flip. The rule pipeline refuses on thin evidence; a replacement that
  always answers would be trading honesty for a metric.
* **It explains itself.** :meth:`EngagementModel.explain` reports which named
  features moved this particular decision, so a wrong verdict can be argued
  with rather than merely disbelieved.
* **It is a linear model over interpretable features.** With a few hundred
  labelled windows, anything heavier fits the raters rather than the task.

The model is trained by ``tools/train_window_model.py`` and loaded from a JSON
file -- coefficients, not a pickle, so the artifact is readable, diffable, and
carries no import-time code.
"""
from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from backend.engagement_features import FEATURE_NAMES

#: Default location of the trained coefficients.
DEFAULT_MODEL = Path("outputs/window_engagement_model.json")

#: Half-width of the abstention band around p = 0.5. A window whose probability
#: falls inside it is reported as unknown rather than decided.
#:
#: 0.10 is not tuned -- it is chosen to match the behaviour the rules already
#: have, where a student with no usable evidence gets no verdict. Tuning it on
#: the same few hundred windows that trained the model would be fitting the
#: abstention rate to noise.
ABSTAIN_BAND: float = 0.10

#: Below this coverage the window is refused outright, whatever the model says.
#: Mirrors backend.engagement_features.MIN_COVERAGE: a verdict computed from
#: almost no readable frames is not a verdict.
MIN_COVERAGE_TO_DECIDE: float = 0.30


@dataclass(frozen=True)
class Verdict:
    """One window's learned engagement decision.

    Attributes:
        label: ``"on"``, ``"off"``, or ``None`` when the model abstained.
        probability: Fitted probability that the window is on-task.
        reason: Why this came out as it did, in words.
        abstained: True when the band or the coverage floor refused a decision.
    """

    label: str | None
    probability: float
    reason: str
    abstained: bool


class EngagementModel:
    """Logistic regression over the named window features.

    Attributes:
        coefficients: One weight per entry in ``feature_names``.
        intercept: The bias term.
        feature_names: The features the weights correspond to, stored with the
            model so a reordering of FEATURE_NAMES cannot silently misalign
            them.
        mean, scale: Standardisation applied before the linear term.
    """

    def __init__(self, coefficients: Sequence[float], intercept: float,
                 feature_names: Sequence[str], mean: Sequence[float],
                 scale: Sequence[float]) -> None:
        if not (len(coefficients) == len(feature_names) == len(mean)
                == len(scale)):
            raise ValueError(
                f"model is inconsistent: {len(coefficients)} coefficients, "
                f"{len(feature_names)} names, {len(mean)} means, "
                f"{len(scale)} scales")
        self.coefficients = list(coefficients)
        self.intercept = float(intercept)
        self.feature_names = list(feature_names)
        self.mean = list(mean)
        self.scale = [s if s else 1.0 for s in scale]

    @classmethod
    def load(cls, path: Path | None = None) -> EngagementModel:
        """Read a trained model from JSON.

        Raises:
            FileNotFoundError: If no model has been trained yet. The caller is
                expected to fall back to the rules rather than crash --
                shipping without the learned model must stay possible.
        """
        path = path or DEFAULT_MODEL
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("kind") == "mlp":
            return MLPEngagementModel(
                payload["w1"], payload["b1"], payload["w2"], payload["b2"],
                payload["feature_names"], payload["mean"], payload["scale"])
        return cls(payload["coefficients"], payload["intercept"],
                   payload["feature_names"], payload["mean"], payload["scale"])

    def save(self, path: Path | None = None) -> Path:
        path = Path(path or DEFAULT_MODEL)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "coefficients": self.coefficients,
            "intercept": self.intercept,
            "feature_names": self.feature_names,
            "mean": self.mean,
            "scale": self.scale,
        }, indent=1), encoding="utf-8")
        return path

    def _standardise(self, features: Sequence[float]) -> list[float]:
        return [(v - m) / s for v, m, s
                in zip(features, self.mean, self.scale)]

    def probability(self, features: Sequence[float]) -> float:
        """Fitted probability that this window is on-task."""
        if len(features) != len(self.feature_names):
            raise ValueError(
                f"expected {len(self.feature_names)} features, "
                f"got {len(features)}")
        z = self.intercept + sum(
            c * v for c, v in zip(self.coefficients,
                                  self._standardise(features)))
        # Clamped before exp so a large negative z cannot overflow.
        return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, z))))

    def classify(self, features: Sequence[float]) -> Verdict:
        """Decide, or decline to.

        Args:
            features: Values in :data:`FEATURE_NAMES` order, as produced by
                :func:`backend.engagement_features.window_features`.

        Returns:
            A :class:`Verdict`. ``label`` is ``None`` whenever the evidence is
            too thin or the probability sits too near the boundary -- the same
            refusal the rule pipeline makes, kept deliberately.
        """
        coverage = features[self.feature_names.index("coverage")]
        if coverage < MIN_COVERAGE_TO_DECIDE:
            return Verdict(None, float("nan"),
                           f"coverage {coverage:.0%} below the "
                           f"{MIN_COVERAGE_TO_DECIDE:.0%} floor", True)

        p = self.probability(features)
        if abs(p - 0.5) < ABSTAIN_BAND:
            return Verdict(None, p,
                           f"probability {p:.2f} inside the abstention band",
                           True)
        label = "on" if p >= 0.5 else "off"
        return Verdict(label, p, f"probability {p:.2f}", False)

    def explain(self, features: Sequence[float], top: int = 4) -> str:
        """Which features moved this decision, largest contribution first.

        A learned verdict that cannot be interrogated is worse than a rule that
        can, however much more accurate it is -- the output of this system is
        meant to be shown to a teacher.
        """
        contributions = [
            (name, c * v) for name, c, v
            in zip(self.feature_names, self.coefficients,
                   self._standardise(features))
        ]
        contributions.sort(key=lambda kv: -abs(kv[1]))
        parts = [f"{name} {value:+.2f}" for name, value in contributions[:top]]
        return ", ".join(parts)


class MLPEngagementModel(EngagementModel):
    """One hidden layer over the same features, when it measurably wins.

    Benchmarked against ten alternatives under the shipped protocol -- folds
    split by student, out-of-fold -- a sixteen-unit hidden layer reached 0.632
    against logistic regression's 0.588, and did so with roughly half the
    spread across folds (0.202 against 0.364). The tighter spread is the more
    persuasive half: a higher mean with wider variance usually means a split
    the model happened to suit, and this is the opposite.

    The gain is real but small, and 179 labels cannot separate close candidates
    with confidence. Re-run tools/compare_classifiers.py after each round of
    labelling; if logistic regression catches up, prefer it, because a linear
    model's explanation is exact rather than approximate.

    Weights stay JSON arrays, and the forward pass is written out in numpy, so
    the artifact remains readable and carries no import-time code.
    """

    def __init__(self, w1, b1, w2, b2, feature_names, mean, scale) -> None:
        # Coefficients are the first-layer weights collapsed by their output
        # weights: a linear summary used only for ordering the explanation.
        import numpy as np

        self.w1 = np.asarray(w1, dtype=float)
        self.b1 = np.asarray(b1, dtype=float)
        self.w2 = np.asarray(w2, dtype=float).reshape(-1)
        self.b2 = float(np.asarray(b2).reshape(-1)[0])
        super().__init__(
            (self.w1 * self.w2[None, :]).sum(axis=1).tolist(),
            self.b2, feature_names, mean, scale)

    def probability(self, features: Sequence[float]) -> float:
        import numpy as np

        if len(features) != len(self.feature_names):
            raise ValueError(
                f"expected {len(self.feature_names)} features, "
                f"got {len(features)}")
        x = np.asarray(self._standardise(features), dtype=float)
        hidden = np.maximum(0.0, x @ self.w1 + self.b1)   # ReLU, as sklearn
        z = float(hidden @ self.w2 + self.b2)
        return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, z))))

    def explain(self, features: Sequence[float], top: int = 4) -> str:
        """Local attribution: how much each feature moved THIS prediction.

        A linear model's explanation is the same everywhere; a network's is
        not, so the weights alone would be misleading. This perturbs each
        feature to its mean and reports how far the probability moves, which
        answers the question actually being asked -- why this student, now.
        """
        base = self.probability(features)
        contributions = []
        for i, name in enumerate(self.feature_names):
            muted = list(features)
            muted[i] = self.mean[i]
            contributions.append((name, base - self.probability(muted)))
        contributions.sort(key=lambda kv: -abs(kv[1]))
        return ", ".join(f"{n} {v:+.2f}" for n, v in contributions[:top])

    def save(self, path: Path | None = None) -> Path:
        path = Path(path or DEFAULT_MODEL)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "kind": "mlp",
            "w1": self.w1.tolist(),
            "b1": self.b1.tolist(),
            "w2": self.w2.tolist(),
            "b2": self.b2,
            "feature_names": self.feature_names,
            "mean": self.mean,
            "scale": self.scale,
        }, indent=1), encoding="utf-8")
        return path


def load_or_none(path: Path | None = None) -> EngagementModel | None:
    """The trained model, or ``None`` if it has not been trained.

    Callers fall back to :func:`backend.engagement.classify_engagement`. A
    missing model must degrade to the rules rather than break the pipeline,
    because the rules are what ships today.
    """
    try:
        return EngagementModel.load(path)
    except (FileNotFoundError, KeyError, ValueError):
        return None


def feature_names_match() -> bool:
    """Whether a saved model was trained on the current feature set."""
    model = load_or_none()
    return model is not None and model.feature_names == list(FEATURE_NAMES)
