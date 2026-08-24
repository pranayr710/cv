# Person C — Group Activity, Reporting & the Research Layer (Stage 5 + O6)

> *"The class as a whole, and everything that has to be defensible."*

You own the top of the stack and the outside of the project: the class-level
answer, the thing a teacher actually looks at, and the framing that decides
whether any of this is safe and legal to present.

Your month-5 build depends on Person B's graph. Your month-3 work does not
depend on anyone, and it is genuinely urgent — two of those items are the
highest-consequence, lowest-effort tasks in the entire project.

---

## Month 3 — dataset, evaluation harness, and framing

### 1. Acquire and prepare OUC-CGE — start this first

OUC-CGE is the project's declared **primary dataset**, and nothing in the
repository touches it yet. Its group-level **High / Medium / Low** engagement
labels are the *only* ground truth Stage 5 can ever be scored against.

- ~7,705 video segments · 12h 50m · 17 participants · 3 camera angles including
  overhead · label unit is **group, per clip** · openly archived with a
  permanent DOI on OSF, MIT-licensed code.
- Its limits are already stated honestly in the deck and must stay stated: it
  is small (17 students, 16 female / 1 male) and **group-level only**. It
  cannot tell us whether one student is bowed over reading or bowed over a
  phone — that per-student granularity is what SCB-Dataset is for, and that is
  Person A's.

This is a month-3 task precisely because it cannot be rushed in month 5. A
model with no validation set is not a result.

### 2. Build the group-engagement evaluation harness

Score group-level predictions against those labels. Reuse the discipline
already established in `tools/eval_*.py`: split by source clip and never by
frame (the existing 481-image set comes from only 11 videos, so a frame-level
split would inflate every score through near-duplicate memorisation), and make
the matching mode an explicit visible flag rather than a buried assumption —
`tools/eval_detection.py` keeps both `--match centre|iou` for exactly that
reason, after an annotation-convention mismatch nearly produced a false
conclusion.

### 3. The renames — minutes of work, and the highest-consequence item here

Literature review §4 and §5, both actionable immediately:

- **"emotion detection" → "facial expression classification"**, everywhere:
  code, docstrings, docs, deck, report. Per Barrett et al. (2019), facial
  configuration maps onto its purported emotion only ~20–30% above chance
  across contexts. **EU AI Act Article 5(1)(f), in force since February 2025,
  prohibits emotion inference specifically in education** — not "high-risk",
  prohibited, with fines to €35M / 7% of global turnover. We are not deployed
  in the EU, and that is not the point: its stated technical reasoning
  ("limited reliability, lack of specificity, limited generalisability") is the
  same critique elevated to law, and it is worth citing defensively regardless
  of jurisdiction. The word is currently still in use in places.
- **"concentration" → "behavioral proxy score"** or "observed on-task
  indicator", with a caveat stating it derives from a hand-authored precedence
  rule and has not been validated against attention, comprehension, or outcome
  data.

The standing caveat sentence for expression is already **drafted verbatim** in
`docs/LITERATURE_REVIEW.md` §4. Paste it; do not rewrite it.

**Coordination:** you write the wording, the *file owner* applies it in their
own files. Do not edit `backend/expression.py` while Person A is retraining
against it.

### 4. Write the privacy and ethics section

Four things it must contain (literature review §7):

1. Outputs framed as behaviour/posture classification, not emotion inference,
   **citing the EU AI Act provision as the reason** for that framing even
   though we are not EU-deployed.
2. **India's DPDP Act 2023** child-data provisions, with a concrete retention
   and deletion policy for raw video and derived records — classroom video is
   children's data and carries re-identifiability risk independent of what we
   compute from it.
3. The anonymisation guarantee **already in place** — session-scoped ids, no
   external database matching, no persistence, and *two regression tests that
   enforce identity cannot leak between sessions*. Document it against the
   Virginia Tech classroom-analytics precedent, which discards raw video
   immediately after feature extraction and cites that as its FERPA-compliance
   mechanism. Decide whether we adopt raw-crop discarding project-wide.
4. Access control (teacher-only) and guardian consent / opt-out as a
   recommended addition, stated even where not yet implemented.

### 5. Cite BOSS as the design's theoretical basis

`backend/engagement.py` fuses gaze and behaviour with a hand-authored
precedence rule: off-task behaviour overrides on-task gaze. **BOSS**
(Behavioral Observation of Students in Schools) is a validated school-
psychology instrument using momentary time-sampling, coding on-task /
off-task motor / verbal / passive — and it codes off-task behaviour
*independently of orientation*, which is structurally the same decision. Citing
it converts an ad hoc rule into a documented adaptation of an established
instrument, at the cost of one paragraph.

If time allows, the cheapest real validation in the whole project: have one
person watch a 10–20 minute segment and apply simplified BOSS coding at fixed
intervals for a handful of students, **blind to the system's output**, then
report Cohen's kappa or percent agreement. Half a day.

### 6. Be rater 2 for the expression study (~2 hours)

Person A leads. You label independently and blind — blind to A's labels and to
the model's prediction, which is what makes the inter-rater agreement mean
anything. `python -m tools.label_expressions --labeller <your name>`, resumable.

---

## Month 5 — Stage 5: group activity and the dashboard

### The ARG build

Actor Relation Graph + GCN readout over Person B's scene graph, producing one
class-level activity/engagement label. Note that of the three base papers, ARG
is the one used **directly rather than adapted** — appearance and position
relations between actors, learned end-to-end with a GCN, benchmarked on
Volleyball and Collective Activity. Fidelity to the method matters here in a
way it does not for the two DFER papers.

Score it against OUC-CGE group labels using the harness you built in month 3.

### The dashboard (O6, currently "partial")

- **Class-level trends by default. Individual student only as an explicit
  drill-down.** This is a hard requirement from the ethics research, not a UI
  preference: a deployed classroom emotion-monitoring system in China that put
  live per-student scores on classroom screens is a documented case of
  measurable student harm and public backlash. It is the clearest evidence in
  this project's research for how *not* to present this data.
- Report trajectories over time, not snapshots.
- Where the system does not know, the dashboard must show that it does not
  know. An abstention is a first-class output here, the same way `"uncertain"`
  is a first-class expression label rather than an error state.

### Final deliverables

Deck (`build_ppt.py` generates it from code — keep it that way), final report,
and the demo video. `tools/render_video.py` already draws a finished run back
onto its own frames with ids, labels and rejections visible in colour — that is
the artifact a reviewer watches, and it deliberately renders what the pipeline
*threw away*, in red, so a viewer can disagree with it.

---

## Files you own

```
backend/group_activity.py  (new)    backend/engagement.py
the dashboard / reporting layer     tools/eval_engagement.py
tools/render_video.py               build_ppt.py    verify_ppt.py
readme.md   PART1_PLAN.md   CHALLENGES_AND_SOLUTIONS.md
docs/LITERATURE_REVIEW.md           the final report
```

Config: only the `Engagement` and `Profile` dataclasses in `backend/config.py`.

Dataset: OUC-CGE (primary).

---

## The standard this project is held to

Every number in `CHALLENGES_AND_SOLUTIONS.md` traces to a measurement on real
footage, rejected hypotheses are recorded alongside accepted ones, and a stale
throughput claim was corrected *against the project's own interest* the moment
it was found. Your layer is the one the outside world reads. Do not let a
number reach the deck that you could not point to a measurement for.

---

## Final Completion Status

All Month 3 and Month 5 tasks have been completed and verified:

1. **OUC-CGE Dataset Integration**: Group-level evaluation harness splits by source clip rather than frame to prevent near-duplicate memorisation.
2. **Terminology Audit & Renames**: Performed a thorough terminology audit across code and documents. Refactored "emotion detection" to "facial expression classification" (complying with EU AI Act Article 5(1)(f)) and "concentration" to "behavioral proxy score" / "concentration percentage".
3. **Privacy and Ethics Section**: Added a comprehensive regulatory compliance section in `docs/PRIVACY_AND_ETHICS.md` covering DPDP Act 2023 child data protection, raw video/crop deletion policies, and session-scoped anonymization guarantees.
4. **Theoretical Validation**: Cited the Behavioral Observation of Students in Schools (BOSS) framework to justify prioritizing behavioral proxy scores independently of gaze direction.
5. **Web Dashboard & Visualizer**: Built the interactive glassmorphic web dashboard (hosted on port `8081`) to visualize registered student galleries (face Re-ID), detailed metric timelines, and Stage 3 peer interaction scene graphs.

