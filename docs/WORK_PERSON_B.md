# Person B — Relations & Time (Stages 3–4)

> *"What relates to what, and how it changes."*

You own the two middle stages, which are the two the review deck marks as
**"in progress"** (O5, peer orientation & group activity) and **"partial"**
(O6's temporal half). Stages 1 and 2 answer *who is here and what are they
doing*. Nothing yet answers *how they relate to each other* or *how any of it
evolves across a lecture* — that is the entire gap you close.

This is the largest greenfield build in the project.

---

## Month 3 — Stage 3, the scene graph

### Week 1: freeze `graph_schema.json` before writing anything else

Do not skip this. Stage 1's frozen `schema.json` is the single reason three
people were able to build a perception pipeline in parallel against stable
interfaces, and the project has already paid for one schema change with an
explicit sign-off process. Your graph output is Person C's input for Stage 5;
they should be able to start building against it in week 2 without waiting for
your implementation.

Rules inherited from `schema.json`, keep them: additive changes only, missing
values are `null` rather than skipped entries, all coordinates in image space,
and validate every run against the schema in a test.

### The graph

- **Nodes** = identified students, plus the instructor carrying an explicit
  role flag. `ProfileConfig.instructor_ids` declares the instructor manually,
  and literature review §6 confirms that is the field's own practical answer,
  not a stopgap — no published system separates teacher from student using
  passive single-camera RGB geometry alone, and all four geometric signals were
  measured on our own footage and all four failed.
- **Node features** = gaze label (calibrated), posture geometry, expression,
  behaviour class, and the on/off-task verdict from `backend/engagement.py`.
- **Edges** = spatial adjacency, mutual orientation, and shared-object
  relations (a book or phone lying between two students).

### Absorb `backend/peer_interaction.py`, and fix its known bug

The geometry for detecting two students jointly oriented toward each other
already exists — and on its first real check, the top-scoring "interacting"
pair turned out to be two students at completely different, non-adjacent desks
with no sign of interacting at all. That false positive is documented in the
code rather than buried, along with a measurement showing the distance
threshold is looser than this classroom's real desk spacing. **That is your
first real bug, already found and waiting for you.**

Note the dependency it carries: `PostureConfig.only_when_faceless` is a 19%
speed saving that is deliberately **off by default**, because peer interaction
needs shoulder keypoints for *both* students of a pair and turning it on would
silently disable the feature. If you change that trade, change it knowingly.

### Design for absence — this is a requirement, not a nicety

Deck originality claim #03 is *"say we don't know instead of guessing"*, and it
lives in your layer. Every edge type must be able to report **unknown**
distinctly from **absent**. Two consequences:

- A student whose face is hidden must not silently become a node with default
  values. The pipeline already treats `face_bbox` present with `landmarks=None`
  as a representable state rather than "no face at all"; carry that discipline
  upward.
- Two students facing each other may be discussing the lesson or chatting, and
  the CSCL literature is explicit that vision alone cannot distinguish them —
  the field's own answer when it needed that distinction was to add a
  microphone, not a smarter camera heuristic. `turn_head` is already excluded
  from the on/off-task binary for exactly this reason. Your peer-oriented edges
  inherit that: they are a *category*, never a verdict.

**Be aware of what you are actually receiving.** On the audited footage a
typical student has expression classified in 5 frames of 112 and behaviour in
**0 of 112**. Person A is working on this in month 3. Design the graph so that
geometry and gaze — which *are* well-populated — carry it, and richer node
features improve it rather than being load-bearing. Raise it at the end-of-
month-3 checkpoint if it has not moved.

---

## Month 4 — Stage 4, temporal

### Absorb `backend/attention.py`

A significant part of Stage 4 already exists in a Stage 1-era file, and it is
the most research-grounded module in the project. It reads a finished JSONL and
adds:

- a **rolling 15-second window** before judging anything as off-task, matching
  the ~12-second window validated in published lecture-gaze research — never a
  per-frame verdict;
- a **90-second sustained-distraction threshold**, so a single missed glance
  back never triggers a flag;
- **per-student calibration** from each student's own first 60 seconds, the one
  accuracy lever the research found actually measured (+0.084 AUC in a real
  classroom study);
- a deliberately honest **6-category taxonomy** where gaze toward a neighbour
  is its own *ambiguous* bucket, never counted as distraction;
- **class-level summary as the default output**, with individual data reachable
  only by deliberately asking for it.

### The known problem you inherit

On genuinely continuous panning footage, **zero tracks survived 60 continuous
seconds** (28 track ids for a maximum of 9 concurrent people; 2 tracks spanned
90s). Per-student calibration therefore never fires on that footage at all.
Two confounds were stated rather than glossed over: the camera pans, and 1 fps
sampling was forced by compute cost rather than chosen. Either fix this with
Person A's identity work, or document plainly that calibration requires a
static camera — do not quietly ship a feature that never runs.

### The build

A sequence model over the graph producing, per student, a trajectory across the
whole lecture, and transitions (attentive → off-task) with timestamps.

Method credit belongs to the two DFER base papers, and note carefully *why*
they are cited: **not** for emotion classification, which this project
deliberately excludes. We take the principle of **separating a momentary signal
from a sustained state**, and the technique for handling ambiguous, noisy signal
over time. Cite the technique, not the task.

### Gate

**One line per student across an entire lecture, with the moments attention
dropped identifiable on it** — deck originality claim #01. If you can point at
a timestamp and say "here, and here is why", Stage 4 is done.

---

## Files you own

```
backend/scene_graph.py    (new)     backend/temporal.py      (new)
backend/peer_interaction.py         backend/posture.py
backend/attention.py                graph_schema.json        (new)
tests/test_scene_graph.py (new)     tests/test_temporal.py   (new)
tests/test_peer_interaction.py      tests/test_posture.py    tests/test_attention.py
```

Config: only the `Posture`, `PeerInteraction` and `Attention` dataclasses in
`backend/config.py`, plus whatever new dataclasses your stages need.

---

## How to not be blocked

Develop against the checked-in sample run, **as-is, wrong ids and all**. The
`person_id` *values* will change when Person A's repair lands; the *schema*
will not. Waiting for correct identity costs you a month; re-running your
pipeline once against a corrected JSONL costs you an afternoon.
