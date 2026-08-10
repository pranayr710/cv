# Part 1 (Perception) — Revalidated Plan, with the Model Named for Every Step

> Written before any new code, per sir's instruction: get Part 1 fully right —
> person detection, face detection, the writing signal, and now expression —
> before touching Stage 2+ (identity, scene graph, group activity), which is
> already built and deliberately paused.
>
> **Revalidation note:** adding expression recognition (item 5) *changes the
> order of the plan*, and that is the single most important finding in this
> revision. Expression can only be computed for a student whose **face** is
> found. On `dataset/img01.jpg` we currently match **10 faces out of 30
> detected persons (33%)**. So an expression classifier bolted on today would
> silently cover one student in three. **Face recall is now the gate on the
> whole feature**, so it moves ahead of everything else.

---

## 0. Decisions now settled (no longer open questions)

| Question | Decision |
|---|---|
| Emotion taxonomy | **happy / sad / neutral** — 3 classes (sir's call) |
| Book/pen | **Keep**, reclassified as an on-task *writing* signal, not dropped |
| Bigger dataset | User will supply; requirements spelled out in §5 |
| Framing | Reported as **facial expression**, never as verified internal emotion (§6) |

**Why 3 classes is the right call, not a compromise:** the 7-class basic-emotion
taxonomy that DFEW/FERV39K use includes categories (disgust, fear, contempt)
that are both the least reliably classified *and* essentially absent from
classroom footage. Collapsing to happy/sad/neutral removes the classes that
would generate the most false positives. The real risk it introduces is
**class imbalance** — in a classroom, "neutral" will be the overwhelming
majority, likely 85–95% of all frames. That means **overall accuracy is a
useless metric here** (a model that always answers "neutral" would score
~90%), and we must report **per-class recall for happy and sad separately**
from day one. This is planned for in §4 step 5, not discovered later.

---

## 1. Model choice per step — the summary table

| # | Step | Model / library | Status | Why this one |
|---|---|---|---|---|
| 1 | Person detection | **YOLOv11m** (`yolo11m.pt`, Ultralytics) | already shipped | COCO-pretrained, no training needed; already tuned (`imgsz=1280`, `person_conf=0.30`) |
| 2 | Person recall push | **YOLOv11l / YOLOv11x** + `imgsz` sweep | to measure | same family, bigger backbone — an offline-only accuracy/latency trade |
| 3 | **Face detection (the gate)** | **SCRFD** (InsightFace) — replacing MediaPipe's internal detector | **to build, highest priority** | see §2 — MediaPipe is the wrong tool for distant classroom faces |
| 3b | Face landmarks / EAR | **MediaPipe Face Mesh** (keep) | already shipped | still fine *once given a face box*; only its detector is the weak link |
| 4 | Head pose / gaze | **SixDRepNet** | already shipped | unchanged; pitch-sign bug already fixed |
| 5 | Writing signal | **YOLOv11 (`book`) + MediaPipe Pose (wrist keypoints)** | to build | fusion rule, not a new model — see §3 |
| 6 | **Facial expression** | **EmotiEffLib / HSEmotion** (EfficientNet-B0, AffectNet-pretrained) | to build | see §4 — pip-installable, SOTA on AffectNet, ONNX, already does engagement too |
| 6b | Temporal smoothing of expression | **RDFER method** (our base paper), reimplemented as windowing | to build | no public code found for the paper; we take the *method* — see §4 |
| 7 | Posture fallback | **MediaPipe Pose** | already shipped | unchanged; recovers 56% of faceless students |

---

## 2. Step 3 — Face detection is the bottleneck; SCRFD is the fix

**The problem, measured:** 10/30 faces on img01 (33%). Earlier measurement
across 12 images: 98 faces from 236 persons (42%), and MediaPipe's own
BlazeFace tested as a pre-detector did *worse* (25%) — already documented as
a rejected approach in `backend/face.py`.

**Why MediaPipe keeps losing, root-caused:** BlazeFace is explicitly a
**short-range, selfie-distance** detector — it is built to run at 200+ FPS on
a phone looking at one nearby face, and trades accuracy on small and
awkwardly-angled faces to get there
([model comparison](https://learnopencv.com/what-is-face-detection-the-ultimate-guide/)).
A classroom shot is the opposite problem: many faces, all small, many
partially turned. We were not using it wrong — it is the wrong tool.

**Why SCRFD is the right replacement, with numbers:** SCRFD was built
specifically for the WIDER FACE **hard** subset, where **78.9% of faces are
under 32×32 px, 51.9% under 16×16 px** — i.e. exactly our back-row students.
It reaches RetinaFace-level accuracy at **3–10× the speed**, and it does so
by deliberately redistributing training samples into the network's early
layers to improve small-face recall
([SCRFD paper, ICLR 2022](https://arxiv.org/pdf/2105.04714),
[InsightFace repo](https://github.com/deepinsight/insightface/tree/master/detection/scrfd)).

**Plan:** keep the existing architecture exactly as-is — the per-person-crop
strategy stays, since that fix was independently proven (0 faces → 8/20).
Swap **only** the detector: SCRFD finds the face box, MediaPipe Face Mesh
still provides the 468 landmarks and EAR on that box. This is a contained
change to `backend/face.py`, and it must be **measured against the current
33%/42% baseline on the same images** before it's accepted — same
before/after discipline used for `imgsz` and the pitch-sign bug.

**Honest caveat to keep in mind:** the earlier finding that ~45–55% of
students have *no recoverable face at all* from an overhead angle is a
**camera-geometry** limit, not a model limit. SCRFD will raise recall
toward that ceiling; it cannot break through it. A student bowed over a desk
shows the camera the crown of their head. That number should be stated to
sir up front rather than discovered by him.

---

## 3. Step 5 — Book/pen as a *writing* signal (answering sir's question directly)

Sir's objection was correct and dropping `book` was the wrong call. A student
with a book and pen is **on-task**; a student with a phone is **off-task**.
These are opposite signals and the pipeline currently cannot tell them apart,
because `backend/attention.py` only wires up `cell phone`
(`device_object_classes`) — `book` is detected and then ignored.

**Model: no new model. A fusion rule over two we already run.**

1. **YOLOv11** already detects the `book` class (it's in `object_whitelist`).
2. **MediaPipe Pose** already gives wrist/hand keypoints per person (built for
   the posture fallback).
3. New rule, symmetric to the existing phone rule: `book` box near a person's
   box **AND** a wrist keypoint near that book → **`writing` / on-task**
   bucket, reported as its own category — never collapsed into a generic
   "engaged" score. Same honesty pattern already in the code, where a bowed
   head only becomes meaningful when paired with a nearby object.

**The real weakness, stated plainly:** COCO's `book` class was trained largely
on bookshelves and stacked books, not an open notebook viewed at a downward
classroom angle. This is a **domain mismatch**, so tuning confidence alone
may not rescue it. Two escalation options, in order:

- **First**, give `book` its own confidence threshold, independent of
  `cell phone` (they currently share `object_conf=0.35`), and sweep it on real
  images. Cheap, no training.
- **If that hits a ceiling**, fine-tune YOLOv11 on **SCB-Dataset**, which has
  real classroom behaviour labels including *write*, *read*, and *using the
  phone* ([SCB-Dataset](https://arxiv.org/html/2304.02488v7)). This trains the
  behaviour directly instead of inferring it from a mismatched object class.
  This is the one place in Part 1 where training (not just pretrained
  inference) is genuinely justified.

Precedent for pose+object fusion over a single class: classroom-behaviour
systems in the literature routinely combine skeleton keypoints with an object
or action classifier to separate *writing* from other desk postures
([skeleton-feature pose recognition](https://link.springer.com/chapter/10.1007/978-3-032-18138-1_17),
[hand-raising detection with spatial context](https://www.sciencedirect.com/science/article/abs/pii/S0097849322002047)).

---

## 4. Step 6 — Facial expression: happy / sad / neutral

**Model: EmotiEffLib (formerly HSEmotion), by Savchenko —
EfficientNet-B0 pretrained on AffectNet.**

Why this specific library, and not training our own:

- It is **pip-installable** (`emotiefflib`), runs on **PyTorch or ONNX**, and
  is built for exactly our deployment shape: efficient inference over many
  face crops per frame
  ([EmotiEffLib](https://github.com/sb-ai-lab/EmotiEffLib),
  [PyPI](https://pypi.org/project/emotiefflib/)).
- Its models hold **state-of-the-art results on AffectNet**, and the team
  placed in multiple **ABAW** (Affective Behavior Analysis in-the-wild)
  competition tracks — 2nd in Compound Expression Recognition, 3rd in Action
  Unit detection ([HSEmotion at ABAW-6](https://arxiv.org/pdf/2403.11590)).
  This is a benchmarked model, not a random GitHub checkpoint.
- Critically, **it already ships engagement recognition from video**, not just
  per-frame expression — which overlaps with what ClassGraph's attention
  module does independently. That makes it a genuine cross-check on our own
  engagement output, which we currently have no second opinion on.
- It plugs into **face crops we already produce** in `backend/face.py`. No new
  detection stage; it consumes the output of step 3.

**Mapping to 3 classes:** the model outputs the 7/8-class AffectNet taxonomy.
We aggregate: `happy → happy`; `sad → sad`; `neutral → neutral`; and
`anger / disgust / fear / surprise / contempt → neutral`, **with the raw
7-class distribution retained in the output record** so the mapping is
auditable and reversible, and so we never destroy information at the earliest
stage. The mapping must be a config constant, not hardcoded — consistent with
this project's no-magic-numbers rule.

**Temporal smoothing — where our base paper actually earns its citation:**
a single frame's expression is noise. **RDFER** (Liu, Wang & Shen, 2025 — base
paper #1) proposes exactly the right method: disentangle **short-term facial
movement** from **longer-term state**, and identify genuinely hard samples by
checking whether predictions **agree across differently-sampled clips** of the
same video ([arXiv:2502.16129](https://www.arxiv.org/abs/2502.16129)).
We reuse that as our windowing/agreement rule over the existing **15-second
rolling window** already implemented in `backend/attention.py`.

> **Note for honesty in the write-up:** I searched for RDFER's code release and
> **did not find a public repository**. So we cite it as *method*, and
> implement the agreement-over-resampled-windows idea ourselves on top of
> EmotiEffLib. We must not imply we ran the authors' code. This is the same
> "method, not task" framing already used in the deck, and it is now
> *stronger*, because with expression as a real requirement the paper's task
> and ours finally match.

**Metrics, decided now, not later:** report **per-class recall and a confusion
matrix** for happy/sad/neutral. Do **not** report overall accuracy as the
headline number — with neutral at ~90% prevalence it would be meaningless.
`sad` will be the hardest and rarest class; expect it to be weak and say so.

---

## 5. Step 4 — The bigger dataset: exactly what to provide

Current basis is 12–13 hand-picked photos plus one 321-frame video. Enough to
tune directionally; **not** enough to state precision/recall credibly, which
is the main thing blocking a "Part 1 is done" claim.

In priority order:

1. **Footage from the real deployment camera angle** (overhead / rear-corner).
   Angle matters more than volume — 10 frames from the actual camera beat 100
   generic ones, because our two hardest failures (face recall, bowed
   students) are both *geometry* problems.
2. **The hard conditions, on purpose:** back rows, students leaning and bowed
   over desks, backlit windows. These are where detection currently fails, so
   they carry the most information. Easy frontal shots teach us nothing new.
3. **A hand-labelled ground-truth subset — 20-30 images is enough.** Person
   boxes, face boxes, and book/phone boxes marked by hand. This is the single
   thing that converts every "counted by eye" figure in our deck into a real,
   defensible number.
4. **For expression specifically:** a few dozen face crops labelled
   happy/sad/neutral **by more than one person**, so we can check the humans
   agree with each other before asking whether the model agrees with them. If
   two annotators can't agree on "sad," the model's score is meaningless — and
   this is exactly the disagreement the research in §6 predicts.

**Video is more valuable than stills** for expression and attention, because
both are defined over time windows, not frames.

---

## 6. The framing guardrail on expression — non-negotiable, and cheap

This project earlier decided *not* to infer emotion, for a researched reason,
and that reason has not changed: Barrett, Adolphs, Marsella, Martinez & Pollak
(2019) reviewed **over 1,000 studies** and found no scientific support for
reliably reading emotion from facial movement — a smile can signal submission
rather than happiness
([Psychological Science in the Public Interest](https://journals.sagepub.com/doi/10.1177/1529100619832930),
[ACLU summary](https://www.aclu.org/news/privacy-technology/experts-say-emotion-recognition-lacks-scientific)).
Our own deck also cites a real deployed Chinese classroom emotion-monitoring
system as a **cautionary example**.

None of that blocks sir's requirement. It only fixes the **wording and the
reporting**, at essentially zero engineering cost:

- Call the output a **facial expression label** ("expression: happy"), never
  "the student is happy" or "emotion." One word, and it's the difference
  between a defensible feature and an overclaim a reviewer can attack.
- **Report class-level aggregates and trends by default** — never a live
  per-student expression label on a screen. This is already the rule
  implemented in `backend/attention.py` ("never a bare individual verdict"),
  and it is precisely what separates us from the system our own slides
  criticise.
- Put the Barrett limitation **in the module docstring**, same as the existing
  honest caveats on EAR and posture.

This turns a potential reviewer attack into a point in our favour: we
implemented the feature *and* we can explain its scientific limits.

---

## 7. Pin-to-pin execution order

Reordered from the previous draft, because face recall gates expression:

| Order | Task | Model | Depends on |
|---|---|---|---|
| **1** | Swap face detector to **SCRFD**, keep Face Mesh for landmarks; measure vs the 33%/42% baseline on all 13 images | SCRFD + MediaPipe Face Mesh | — |
| **2** | Re-run `tools/render_faces.py` across every `dataset/*.jpg`; tabulate per-image face-match rate (not img01 alone) | existing | 1 |
| **3** | Person-recall sweep: `imgsz` 1536/1920, and YOLOv11l/x vs 11m; report recall *and* latency | YOLOv11 l/x | — (parallel) |
| **4** | Writing signal: independent `book` threshold + wrist-proximity fusion rule in `attention.py` | YOLOv11 + MediaPipe Pose | — (parallel) |
| **5** | Expression: integrate EmotiEffLib on existing face crops; 7→3 class mapping in config; retain raw distribution | EmotiEffLib EfficientNet-B0 | 1 |
| **6** | Temporal smoothing of expression over the existing 15s window, using RDFER's agreement-across-resampled-clips method | our implementation | 5 |
| **7** | Fine-tune YOLOv11 on SCB-Dataset for `write`/`read`/`phone` — **only if** step 4 hits a ceiling | YOLOv11 fine-tune | 4 |
| **8** | Validation pass on the new dataset: hand-label a subset; real precision/recall for person, face, book/phone, expression (per-class) | — | dataset arrives |
| **9** | Replace every "counted by eye" number in the deck with a validated one | — | 8 |
| **10** | Only then: resume Stage 2+ (identity, scene graph, group activity) | existing | 1–9 |

Steps 1, 3 and 4 have no dependency on the incoming dataset or on any further
decision, so they can start immediately. Step 5 waits on step 1 — putting an
expression classifier behind a detector that finds one face in three would
waste the effort and produce a misleading demo.

---

## 8. What is still unverified, and should be treated as such

- **SCRFD's real gain on *our* footage** is unmeasured. WIDER-FACE-hard
  numbers are a strong reason to expect improvement, not proof of it. If it
  doesn't beat 42% on our images, we say so and keep MediaPipe.
- **EmotiEffLib's accuracy on South Asian classroom faces** is unknown.
  AffectNet/ABAW results are on in-the-wild Western-skewed data. This is the
  same fairness gap already documented for SixDRepNet in
  `backend/fairness_audit.py` — the existing audit harness should be pointed
  at the expression model too, once labelled data exists.
- **`book` recall on real classroom desks** has never been measured at all.
  It needs a number before any claim is made about the writing signal.
- **Human agreement on happy/sad/neutral** in classroom footage is untested
  and may itself be low. That is a finding worth reporting either way, not an
  obstacle to hide.
