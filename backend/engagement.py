"""Per-frame and per-student observed on-task scoring for ClassGraph.

Produces a *behavioral proxy score* — an "observed on-task indicator," never a
measurement of a student's internal concentration or attention. See
:data:`BEHAVIORAL_PROXY_CAVEAT` below for the standing caveat attached to every
summary this module emits.

Theoretical basis — BOSS, not ad hoc. The precedence rule at this module's
core (off-task behaviour overrides on-task gaze) is not an invention of this
project: it is a documented adaptation of the structure of **BOSS (Behavioral
Observation of Students in Schools)**, the momentary time-sampling instrument
used in school psychology (Shapiro, 2004, *School Psychology Review*; see also
Volpe et al. on its use in classroom observation). BOSS codes on-task vs.
off-task behaviour in categories — motor, verbal, passive — that are defined by
*what the student is doing*, explicitly independently of where their head or
eyes happen to point. That structural choice is exactly why our rule lets a
phone/sleep behaviour reading override an attentive-looking gaze: BOSS's
convergent-validity evidence (it discriminates ADHD from typical classrooms in
clinical research) was earned by coding behaviour directly rather than
inferring state from orientation. Citing this converts the rule from ad hoc to
a documented adaptation of an established instrument — with the honest limit,
stated in :mod:`docs.LITERATURE_REVIEW` section 5, that we have NOT reproduced
BOSS's validation: no external check against attention, comprehension, or
outcome data exists for our score yet. :mod:`tools.boss_agreement` plus
:mod:`docs.BOSS_VALIDATION` define the cheapest such check.

Combines the two signals a live frame actually carries — ``head_pose``'s
``gaze_label`` and ``behaviour``'s ``label`` — into a single on-task / off-task
/ unknown verdict, then aggregates that into an observed on-task percentage per
student. See :class:`~backend.config.EngagementConfig` for the precedence
rules and why they are not new inventions but the same honesty and precedence
principles already established in :mod:`backend.attention`, applied
consistently rather than re-derived per module.

This module reads the already-serialised JSONL person dict shape directly
(``person["head_pose"]``, ``person["behaviour"]``), not the internal
dataclasses, so it composes with a finished pipeline run without re-importing
model-heavy modules.

Naming note (deliberate): the serialised summary key remains
``concentration_pct`` so downstream consumers (``student_profile.py``,
``render_video.py``, tests) do not break; it carries the alias
``behavioral_proxy_pct`` set to the identical value, and every report built on
this output should prefer the latter name in prose. New consumers must use the
alias. The word "concentration" survives only as a wire-compat legacy name.

Usage:
    from backend.engagement import classify_engagement, summarise_engagement
    verdict = classify_engagement(gaze_label, behaviour_label)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from backend.config import CONFIG, EngagementConfig

Verdict = Literal["on", "off"]

# Standing caveat mandated by docs/LITERATURE_REVIEW.md section 5,
# recommendation (1): the metric must be renamed/qualified as a "behavioral
# proxy score" or "observed on-task indicator," never "concentration," with a
# caveat sentence stating it derives entirely from a hand-authored precedence
# rule and has not been validated against attention, comprehension, or outcome
# data. This exact text is embedded in every summary dict emitted by
# :func:`summarise_engagement`, so any report built on this module inherits
# the caveat automatically instead of relying on each author remembering it.
BEHAVIORAL_PROXY_CAVEAT = (
    "Behavioral proxy score, not a concentration measurement: this figure "
    "derives entirely from a hand-authored precedence rule (off-task behaviour "
    "overrides on-task gaze) over two machine labels -- head-pose gaze class "
    "and behaviour classification. It has not been validated against "
    "attention, comprehension, or outcome data."
)


def classify_engagement(
    gaze_label: str | None,
    behaviour_label: str | None,
    config: EngagementConfig | None = None,
    *,
    phone_nearby: bool = False,
    eyes_closed: bool | None = None,
) -> Verdict | None:
    """Classify one frame's engagement from every signal actually available.

    Args:
        gaze_label: ``person["head_pose"]["gaze_label"]``, or ``None`` if no
            head pose was available this frame.
        behaviour_label: ``person["behaviour"]["label"]``, or ``None`` if no
            behaviour reading was available this frame.
        config: Engagement settings. Defaults to ``CONFIG.engagement``.
        phone_nearby: Whether a phone-class object from the frame's ``objects``
            list overlaps this student's box. Independent of the behaviour
            model, so it still works when that model produces nothing — see
            :class:`~backend.config.EngagementConfig` for the measured reason
            this fallback exists.
        eyes_closed: Whether ``face.ear`` was below
            :data:`~backend.config.FaceConfig.ear_closed_threshold`, or
            ``None`` if no EAR was available. Only counts as off-task in
            combination with a head-down gaze, which is what separates dozing
            from an ordinary blink.

    Returns:
        ``"off"``, ``"on"``, or ``None`` (unknown).

        Precedence is deliberate and matches :mod:`backend.attention`: every
        off-task route is checked before any on-task route, because a
        contradictory attentive-looking gaze should not override concrete
        evidence of disengagement — the more concerning reading is the safer
        default when signals disagree.

        ``None`` is returned when nothing gives usable evidence. A bare gaze of
        ``"left"``/``"right"``/``"down"``/``"back"`` on its own is deliberately
        NOT off-task, for the reason :mod:`backend.attention` documents at
        length: gaze aversion during effortful thinking and turning toward a
        peer are real, opposite-reading confounds.
    """
    cfg = config if config is not None else CONFIG.engagement

    # --- off-task routes, all checked first --------------------------------- #
    if behaviour_label in cfg.off_task_behaviours:
        return "off"
    if cfg.use_object_fallback and phone_nearby:
        return "off"
    if (
        cfg.use_eye_closure_fallback
        and eyes_closed
        and gaze_label in cfg.eye_closure_gaze_labels
    ):
        return "off"

    # --- on-task routes ----------------------------------------------------- #
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
        ``unknown`` counts, ``behavioral_proxy_pct`` — the on-task share of
        *graded* frames (``on`` / (``on`` + ``off``)), excluding unknown frames
        from the denominator so an unresolvable gaze does not silently count
        against a student — plus the standing caveat string under ``caveat``
        (:data:`BEHAVIORAL_PROXY_CAVEAT`), so downstream reports carry the
        honesty label even if their author never read this docstring.

        ``behavioral_proxy_pct`` is ``None`` (not ``0.0``) when zero frames
        were graded, so a report cannot read "no evidence" as "confirmed
        off-task".

        Wire-compat legacy key: ``concentration_pct`` is emitted alongside
        with the identical value, for existing consumers only (see the module
        docstring's naming note). Do not use it in new code or new prose.
    """
    total = len(verdicts)
    on = sum(1 for v in verdicts if v == "on")
    off = sum(1 for v in verdicts if v == "off")
    unknown = total - on - off
    graded = on + off
    pct = (on / graded * 100.0) if graded > 0 else None
    return {
        "frames": total,
        "on": on,
        "off": off,
        "unknown": unknown,
        # Canonical, honest name. New consumers read this key.
        "behavioral_proxy_pct": pct,
        # Legacy wire-compat alias -- same value, kept so student_profile.py,
        # render_video.py and existing tests do not break. See module
        # docstring naming note.
        "concentration_pct": pct,
        # Mandated by LITERATURE_REVIEW section 5; see BEHAVIORAL_PROXY_CAVEAT.
        "caveat": BEHAVIORAL_PROXY_CAVEAT,
    }
