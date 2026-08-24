# BOSS as the Theoretical Basis for Engagement Scoring (+ Validation Protocol)

*Owner: Person C. Task source: docs/WORK_PERSON_C.md month-3 task 5 — "Cite
the BOSS observation protocol as the design's theoretical basis" in
backend/engagement.py and in the report, with an optional cheapest-real-
validation study. This doc holds the report paragraph and that protocol.*

---

## Report-ready paragraph

The engagement metric at the core of this system is not a novel construct.
Its structure is a documented adaptation of the **Behavioral Observation of
Students in Schools (BOSS)** momentary time-sampling instrument (Shapiro,
2004), long used in school psychology to quantify on-task behaviour. Two
design choices are inherited from BOSS rather than invented here. First,
on-task versus off-task is coded from *observable behaviour categories* —
in BOSS's case motoric, verbal, and passive off-task behaviour; in ours, a
behaviour classifier's read/write/phone/sleep-style verdicts combined under a
fixed precedence rule. Second — the load-bearing inheritance — behavioural
coding is defined independently of head or eye orientation, so a student who
looks toward the board while using a phone is scored off-task, exactly as
BOSS's convergent-validity studies would code them. Our precedence rule
(off-task behaviour overrides on-task gaze) operationalises that principle
with two machine labels standing in for human coders. What we have *not*
inherited is BOSS's validation: published instruments earned their status
through inter-rater reliability studies against clinically meaningful
outcomes; our score has neither yet. It is therefore reported as a behavioral
proxy score with a standing caveat (`backend.engagement.BEHAVIORAL_PROXY_
CAVEAT`), and the protocol below exists to begin closing precisely that gap.

*(backend/engagement.py's module docstring carries the same attribution in
shorter form; this paragraph is the report-length version.)*

## Why BOSS specifically (and what "adaptation" honestly means)

* BOSS codes **momentary states at intervals**, not cumulative impressions —
  matching our per-frame verdicts aggregated per student.
* BOSS separates **active vs. passive engagement** and **off-task subtypes**
  instead of one scalar — our on/off/unknown verdict keeps the same
  refusal-to-collapse-uncertainty (unknown stays unknown).
* We do NOT claim BOSS's psychometrics transfer automatically: our "coders"
  are two ML models with their own measured error rates (behaviour F1 ~51% on
  read/write confusion — see CHALLENGES_AND_SOLUTIONS.md), so any reliability
  figure must be measured end-to-end, human-vs-system, not assumed.

## Simplified BOSS coding protocol (half-day, optional-but-recommended)

Goal: measure whether the precedence rule's verdicts track coarse human
coding of the same intervals — the cheapest real validation available before
any OUC-CGE work.

1. **Material.** One 10-minute segment of existing test footage (no new
   recording needed). Divide into 20-second intervals → 30 intervals.
2. **Raters.** Two people, coding blind to each other and to system output:
   each interval × visible student gets `on` / `off` / `unknown`, using the
   one-page rule sheet below.
3. **Training gate.** Raters first co-code 5 practice intervals, then code
   the real 30 separately. Compute rater1-vs-rater2 agreement with
   `tools/boss_agreement.py --rater2`. Gate: Cohen's kappa ≥ 0.6, else
   reconcile the rule sheet and repeat. (Published BOSS studies typically
   report higher; 0.6 is this project's stated floor.)
4. **System extraction.** Run the pipeline over the same segment; pull each
   student's per-interval majority verdict into the system CSV.
   Coverage below ~90% must be reported alongside results — the tool prints it.
5. **Scoring.** `tools/boss_agreement.py --human rater1.csv --system system.csv`
   → percent agreement + Cohen's kappa + confusion table.

### One-page coding rule sheet (for raters)

| Code | Use when |
|---|---|
| `on` | Attending to task materials/instruction: writing, reading, looking at board/screen/teacher while otherwise settled |
| `off` | Motoric (out of seat without task reason), object play/phone, talking off-topic, sleeping/head down |
| `unknown` | Student not clearly visible, ambiguous, or occluded ≥ half the interval |

Tie-breaker mirroring the system's precedence rule: if gaze says attending
but behaviour says off-task, code `off`.

## Interpreting outcomes (decided now, before data exists)

* **kappa ≥ 0.6:** the rule tracks coarse human judgement at interval grain.
  Report next to the caveat; proceed to OUC-CGE external validation.
* **kappa 0.3–0.6:** weak tracking. Diagnose via the confusion table whether
  disagreement concentrates in a specific transition (e.g. passive off-task);
  consider demoting the offending machine label's weight in the precedence
  rule — as a documented change, re-run protocol.
* **kappa < 0.3:** the precedence rule does not track even coarse human
  coding. Per docs/WORK_PERSON_C.md: it then gets **demoted from "principled"
  to documented guess** — the score keeps its caveat, loses any "grounded in
  classroom observation literature" phrasing in all materials, and reports
  become descriptive counts only until redesigned.

*Numbers standard:* kappa/agreement figures produced by this protocol enter
reports only after both CSVs exist; until then every mention says "protocol
defined, not yet run."
