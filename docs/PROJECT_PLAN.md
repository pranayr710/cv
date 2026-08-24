# ClassGraph — Complete Project Plan

> End to end: what the system is, exactly where it stands, and what remains
> through delivery. Per-person scope lives in
> [WORK_PERSON_A.md](WORK_PERSON_A.md) · [WORK_PERSON_B.md](WORK_PERSON_B.md) ·
> [WORK_PERSON_C.md](WORK_PERSON_C.md). The narrative behind every number here
> is in [../CHALLENGES_AND_SOLUTIONS.md](../CHALLENGES_AND_SOLUTIONS.md); the
> research behind every open decision is in
> [LITERATURE_REVIEW.md](LITERATURE_REVIEW.md).

---

## 1. What the system is

Classroom engagement analytics from a single passive camera. It detects every
visible student, keeps one identity per student for a whole session, and
reports gaze, posture, expression, behaviour and an on-task indicator per
student per frame — then relates students to each other and reports how the
class as a whole evolves over a lecture.

**Explicit non-goals**, and these are load-bearing, not hedging:

- It does **not** infer emotion. It classifies observable facial configuration.
  (Barrett et al. 2019; EU AI Act Art. 5(1)(f).)
- It does **not** judge a single frame. Every verdict is windowed.
- It does **not** distinguish productive peer talk from chatting. That is
  defined by *what is said*; the field's own answer when it needed that
  distinction was a microphone, not a better camera heuristic.
- It does **not** present live per-student scores as its default. Class-level
  trends by default; individuals only by deliberate drill-down.

**The originality claim, framed honestly:** novel in *combination* and in this
domain — not a claim to a better Re-ID, scene-graph or group-activity method in
isolation.

---

## 2. Objectives and where each one actually stands

| | Objective | Success criterion | Status |
|---|---|---|---|
| O1 | Detect every student in frame | Recall on real frames; back rows not lost to downscaling | **Done** — P 82.2% / R 90.6% / F1 86.2% vs human labels |
| O2 | Stable identity within a session | One id per student across occlusion; never crosses sessions | **Regressed, in repair** — duplicates and merges fixed, now over-splits (18 ids / 8 people) |
| O3 | Per-student attention over time | Windowed distribution over 6 categories, not per-frame verdicts | **Done as logic**, but calibration never fires on panning footage |
| O4 | Flag device distraction and eye closure | Phone-overlap + gaze-down; EAR eye state | **Done** — `using_device` F1 72.0%; EAR limited to ~55 students by landmark coverage |
| O5 | Peer orientation and group activity | Geometric pairs now; scene graph + GAR next | **In progress** — geometry exists with one documented false positive; graph not built |
| O6 | Trajectories and class-level trends | Class summary default, individual as drill-down | **Partial** — summary logic exists, no dashboard, no temporal trajectory |

### Headline numbers as of this plan

| Metric | Result |
|---|---|
| Student detection (P / R / F1) | 82.2% / 90.6% / 86.2% |
| Face coverage of detected students | 90.5% (was 36.7% before the SCRFD swap) |
| Writing signal (`write`/`read`) F1 | 77.9% (25.1% via book proxy → 65.3% → 77.9%) |
| Behaviour on an **unseen** classroom, F1 | 68.0% (was 7.3%) |
| `using_device` F1 | 72.0% (was 30.6%) |
| Engagement (on/off-task) agreement | 82.3%; off-task precision 91.1%, recall 36.9% |
| Identity on the audited clip | 18 ids vs 8 real people · 0 duplicate frames · 6.6% unidentified |
| Throughput | 0.41 FPS measured — **stale**, predates the GPU fix |
| Automated tests | 279 passing |

---

## 3. Architecture and the contracts that make it parallelisable

```
video ──► Stage 1  Perception ────► schema.json (frozen)          ── A
             YOLOv11 · SCRFD · FaceMesh · SixDRepNet
             MediaPipe Pose · HSEmotion · behaviour YOLO
                      │
          Stage 2  Identity ───────► person_id in the same JSONL  ── A
             ByteTrack + constrained-clustering re-ID
                      │
          Stage 3  Scene Graph ────► graph_schema.json (to freeze) ── B
             nodes = students · edges = relations
                      │
          Stage 4  Temporal ───────► per-student trajectories      ── B
             sequence decoder over the graph
                      │
          Stage 5  Group Activity ─► class label + dashboard       ── C
             ARG + GCN readout
```

**The single most important engineering rule in this project:** each stage
consumes the previous stage's output through a *frozen JSON schema*. That is
the entire reason three people can build in parallel, and it has already been
paid for once — the Stage 1 schema change required explicit sign-off rather
than a silent edit. Stage 3 must freeze its schema in its first week, before
implementation starts.

Inherited contract rules, unchanged since Week 1: bboxes are `[x, y, w, h]`
integer pixels, top-left origin, **image space**; every module returns lists
aligned index-wise with its input; missing values are `null`, never skipped
entries; additive schema changes only.

---

## 4. Timeline

The review deck plans one stage per month. Reality has run considerably faster
— Stages 1 and 2 were delivered in about three weeks, not two months. So the
plan below is stated in **weeks of work, with the month labels kept as review
milestones**. That is honest about pace without pretending the deadline moved.

### Phase 0 — Stabilise and unblock (weeks 1–2) · all three in parallel

| Owner | Work | Gate |
|---|---|---|
| **A** | Repair the identity over-split: run `sweep_identity.py` against the 7+1 headcount, implement quality-gated identity creation (the unimplemented half of literature review §2), swap to BoT-SORT with `with_reid`, re-audit by contact sheet | **ids ≤ 10 for 8 people · duplicate frames = 0 · no-id ≤ 7%** |
| **B** | Freeze `graph_schema.json`. Implementation does not start first. | Schema reviewed by A and C, checked in, validated in a test |
| **C** | Acquire and prepare OUC-CGE; apply the renames; draft the privacy/ethics section | Dataset loadable with splits defined; zero uses of "emotion" as a claim |
| **A + C** | Expression validation study — 120 crops already staged, two blind raters | Cohen's kappa reported **before** any model accuracy number |

**This phase is the project's real milestone.** If A's gate is not met by day
10, escalate: Stages 3–5 would then be built on a roster known to be wrong, and
all three originality claims fail together.

### Phase 1 — Scene graph (month 3) · B leads

- Nodes = identified students + the instructor carrying an explicit role flag.
- Node features = gaze, posture, expression, behaviour, on/off-task verdict.
- Edges = spatial adjacency, mutual orientation, shared-object.
- Absorb `peer_interaction.py` and fix its documented false positive (the
  top-scoring "interacting" pair were students at non-adjacent desks).
- Every edge type must report **unknown** distinctly from **absent**.

*In parallel:* A runs Phase 2 perception work (SAHI tiling, throughput
re-benchmark, read/write merge). C builds the group-engagement eval harness.

**Gate:** a graph rendered on real frames that a human agrees with, plus a
documented false-positive rate for the mutual-orientation edge.

### Phase 2 — Temporal (month 4) · B leads

- Absorb `attention.py` — 15s rolling window, 90s sustained-distraction
  threshold, per-student 60s calibration, 6-category taxonomy, class-summary
  default.
- Resolve or document the calibration failure: **zero tracks survived 60
  continuous seconds** on panning footage, so calibration never fires there.
- Sequence model over the graph → per-student trajectory + timestamped
  transitions.
- Method credit: the two DFER base papers, cited for **separating a momentary
  signal from a sustained state**, never for emotion classification.

**Gate:** one line per student across a full lecture with drop points
identifiable — deck originality claim #01.

### Phase 3 — Group activity and dashboard (month 5) · C leads

- ARG + GCN readout over B's graph → one class-level engagement label. ARG is
  the one base paper used *directly rather than adapted*, so method fidelity
  matters here in a way it does not for the DFER papers.
- Scored against OUC-CGE High/Medium/Low labels.
- Dashboard: class trends by default, individual drill-down explicit,
  abstention visible as a first-class state.

**Gate:** a group-engagement accuracy figure against OUC-CGE with a confusion
matrix, reported next to human-observer agreement rather than alone.

### Phase 4 — Close (final two weeks) · all three

Full-pipeline run end to end · every claim in the deck re-verified against a
current measurement · demo video via `render_video.py` · final report · privacy
section final · deck regenerated from `build_ppt.py`.

---

## 5. Evaluation plan — the project's spine

Nothing ships as a result without a number, and every number states what it was
measured against.

| Stage | Measured against | Metric | Current |
|---|---|---|---|
| 1 Detection | 481 hand-labelled images, split **by clip not frame** | P / R / F1, centre-match | 82.2 / 90.6 / 86.2 |
| 1 Behaviour | Held-out clips + an **independent** classroom dataset | per-class F1, in- and out-of-distribution | 77.9% in-dist · **68.0%** OOD |
| 1 Expression | 120 own-pipeline crops, 2 blind raters | **kappa first**, then accuracy per size/angle bucket, Wilson CI | **not yet run** |
| 2 Identity | Known headcount (7 students + 1 teacher) + visual contact sheets | ids vs truth, duplicate frames, no-id % | 18 vs 8 · 0 dup · 6.6% |
| 3 Graph | Human agreement on rendered frames | edge precision, false-positive rate | not built |
| 4 Temporal | Blind simplified-BOSS coding of a segment | Cohen's kappa vs system verdicts | not built |
| 5 Group | OUC-CGE group labels | accuracy + confusion matrix | not built |

Two standing rules, both learned the hard way here:

1. **Split by source clip, never by frame.** The 481 images come from only 11
   videos; a frame-level split inflates every score through near-duplicate
   memorisation.
2. **Render before believing.** A precision/recall of 54%/60% that contradicted
   99% face coverage on the same students turned out to be an
   annotation-convention mismatch, not a real failure. Both matching modes are
   kept in `tools/eval_detection.py` so the choice stays visible.

---

## 6. Datasets

| Dataset | Role | Owner | Status |
|---|---|---|---|
| Own classroom footage (13 images, 5.5-min clip) | Development + identity ground truth | A | In use |
| 481-image labelled set (11 clips, 8 classes) | Detection + behaviour training/eval | A | In use |
| Independent classroom set (629 images) | **Out-of-distribution check** — its test split is never trained on | A | In use |
| SCB-Dataset | Per-student behaviour supervision | A | Supporting |
| **OUC-CGE** | **Primary** — group engagement labels, 7,705 clips, 12h 50m, OSF DOI | **C** | **Not acquired** |

OUC-CGE's limits stay stated wherever it is cited: 17 students (16F / 1M) and
group-level labels only — it cannot tell whether a bowed student is reading or
on a phone. That per-student granularity is what SCB-Dataset is for.

---

## 7. Risk register

| Risk | Likelihood | Impact | Mitigation | Trigger to escalate |
|---|---|---|---|---|
| Identity never reaches roster-correct | Medium | **Fatal** — all three originality claims depend on it | Phase 0 gate; quality-gated identity creation is the untried lever | Day 10 without convergence |
| **Node features arrive empty** — expression 5/112 frames, behaviour 0/112 per student | **High** | Severe — a graph with nothing on its nodes | B designs so geometry + gaze carry it; A's Phase 2 targets coverage | End-of-month-3 checkpoint |
| OUC-CGE arrives too late to validate Stage 5 | Medium | Severe — a model with no validation set is not a result | Acquisition is a **week-1** task, not a month-5 one | End of Phase 0 |
| Calibration never fires (no track survives 60s) | Confirmed | Moderate | Fix alongside identity, or document that it requires a static camera | Already live — B owns it |
| Throughput blocks live use | Low | Moderate | 0.41 FPS predates the GPU fix; design tolerates it (15s windows) | If the re-benchmark shows no gain |
| `read`/`write` stays confused | Confirmed | Low | Merge into `studying` — the literature calls this the correct response to a *structural* confusion | — |
| Schema churn between stages | Medium | Severe — kills parallelism | Freeze before implementing; additive changes only | Any non-additive edit |

---

## 8. Ethics and compliance — gates, not garnish

1. **Language.** "Facial expression classification", never "emotion". EU AI Act
   Art. 5(1)(f) prohibits emotion inference in education specifically. We are
   not EU-deployed; the point is that its stated technical reasoning is our own
   critique elevated to law.
2. **Framing.** "Behavioral proxy score", not "concentration", with the caveat
   that it derives from a hand-authored precedence rule validated against no
   attention, comprehension or outcome data.
3. **Basis.** BOSS cited as the documented instrument our precedence rule
   adapts — it codes off-task behaviour independently of orientation, which is
   structurally the same decision.
4. **Data.** DPDP Act 2023 child-data provisions; a concrete retention and
   deletion policy; session-scoped ids with **two regression tests enforcing no
   cross-session leakage**; consider discarding raw crops after feature
   extraction, per the Virginia Tech precedent.
5. **Presentation.** Class-level default. The deployed Chinese classroom system
   that put live per-student scores on classroom screens is this project's
   cited example of measurable student harm.

---

## 9. Definition of done

- A full lecture processed end to end, every frame schema-valid.
- Every student correctly and stably identified, verified against a known
  headcount **and** by eye.
- One attention trajectory per student, with drop points a teacher would
  recognise.
- A class-level engagement figure scored against OUC-CGE, reported beside human
  agreement.
- A dashboard whose default view is the class, and which says "unknown" where
  it does not know.
- A report where **every number traces to a measurement**, rejected hypotheses
  are recorded alongside accepted ones, and open limitations are stated in the
  document rather than discovered by a reviewer.

That last line is the standard this project has actually held so far — a stale
throughput claim was corrected against its own interest the moment it was
found, and the identity audit was published even though it downgraded the
headline result. Keeping that is worth more than any single metric above.
