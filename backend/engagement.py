"""Per-frame and per-student concentration scoring for ClassGraph.

Combines the two signals a live frame actually carries — ``head_pose``'s
``gaze_label`` and ``behaviour``'s ``label`` — into a single on-task / off-task
/ unknown verdict, then aggregates that into a concentration percentage per
student. See :class:`~backend.config.EngagementConfig` for the precedence
rules and why they are not new inventions but the same honesty and precedence
principles already established in :mod:`backend.attention`, applied
consistently rather than re-derived per module.

This module reads the already-serialised JSONL person dict shape directly
(``person["head_pose"]``, ``person["behaviour"]``), not the internal
dataclasses, so it composes with a finished pipeline run without re-importing
model-heavy modules.

Usage:
    from backend.engagement import classify_engagement, summarise_engagement
    verdict = classify_engagement(gaze_label, behaviour_label)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from backend.config import CONFIG, EngagementConfig

Verdict = Literal["on", "off"]


def classify_engagement(
    gaze_label: str | None,
    behaviour_label: str | None,
    config: EngagementConfig | None = None,
) -> Verdict | None:
    """Classify one frame's engagement from its gaze and behaviour readings.

    Args:
        gaze_label: ``person["head_pose"]["gaze_label"]``, or ``None`` if no
            head pose was available this frame.
        behaviour_label: ``person["behaviour"]["label"]``, or ``None`` if no
            behaviour reading was available this frame.
        config: Engagement settings. Defaults to ``CONFIG.engagement``.

    Returns:
        ``"off"`` if an off-task behaviour was read (checked first — a
        contradictory attentive-looking gaze does not override it, matching
        :mod:`backend.attention`'s own precedence for the same reason: the
        more concerning reading is the safer default). ``"on"`` if an on-task
        behaviour was read, or no behaviour reading contradicted an attending
        gaze. ``None`` (unknown) when neither behaviour nor gaze gives usable
        evidence — a bare gaze of "left"/"right"/"down"/"back" is deliberately
        NOT treated as off-task on its own, for the same reason
        :mod:`backend.attention` already documents: gaze aversion and
        peer-oriented turning are real, opposite-reading confounds.
    """
    cfg = config if config is not None else CONFIG.engagement

    if behaviour_label in cfg.off_task_behaviours:
        return "off"
    if behaviour_label in cfg.on_task_behaviours:
        return "on"
    if gaze_label in cfg.attending_gaze_labels:
        return "on"
    return None


def summarise_engagement(
    verdicts: Sequence[Verdict | None],
) -> dict[str, object]:
    """Aggregate a sequence of per-frame verdicts into one student's summary.

    Args:
        verdicts: One :func:`classify_engagement` result per frame that
            student was seen in, in any order.

    Returns:
        A dict with ``frames`` (total verdicts given), ``on`` / ``off`` /
        ``unknown`` counts, and ``concentration_pct`` — the on-task share of
        *graded* frames (``on`` / (``on`` + ``off``)), excluding unknown frames
        from the denominator so an unresolvable gaze does not silently count
        against a student. ``concentration_pct`` is ``None`` (not ``0.0``) when
        zero frames were graded, so a report cannot read "no evidence" as
        "confirmed off-task".
    """
    total = len(verdicts)
    on = sum(1 for v in verdicts if v == "on")
    off = sum(1 for v in verdicts if v == "off")
    unknown = total - on - off
    graded = on + off
    return {
        "frames": total,
        "on": on,
        "off": off,
        "unknown": unknown,
        "concentration_pct": (on / graded * 100.0) if graded > 0 else None,
    }
