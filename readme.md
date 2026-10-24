# ClassGraph

**Per-student engagement from a single fixed classroom camera.**

One frame goes in. For every student visible, what comes out is: who they
are, what they are doing, which way they are facing, and an attention score
on a 1–10 scale aggregated over 15-second windows — plus a scene graph
relating students to each other and to the objects they handle.

Nine models, of which three are trained here. Every number below is measured
against hand-labelled ground truth and reproducible from `results/`.

---

## Start here

**[docs/SETUP.md](docs/SETUP.md)** — clone to working dashboard: install,
what to download, the three ways to run it, and the errors people actually
hit.

### 1. Install

```bash
git clone https://github.com/pranayr710/cv.git classgraph && cd classgraph
python -m venv .venv && .venv\Scripts\Activate.ps1   # or: source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
classgraph doctor                      # what this machine is missing
```

A clone is **6 MB** and needs no data to start — the public weights
(YOLO11m, InsightFace, SixDRepNet, the expression model, ~410 MB) download
themselves on first use.

### 2. Run the web interface

```bash
python -m tools.run.server
```

Then open **<http://127.0.0.1:8000>** and paste the access token it printed
in the terminal. That is the whole user interface: register students, start a
camera session, upload a video and watch it being analysed, see the live
overlay and the scene graph, and open the finished report.

```
ClassGraph dashboard -> http://127.0.0.1:8000
  access token: 7f3a9c2b1e...        <- paste this into the page
```

To fix the token instead of reading a new one each run:

```powershell
$env:CLASSGRAPH_TOKEN = "classgraph"   # PowerShell
export CLASSGRAPH_TOKEN=classgraph     # bash
```

> `classgraph serve` does the same thing. `uvicorn tools.run.server:app
> --factory` also works, but cannot print the token.

### 3. Everything else

```bash
classgraph video --dir <folder of clips> --out outputs/run1   # recorded footage, no browser
classgraph live                                               # webcam, no browser
classgraph train                                              # refit the attention model
pytest                                                        # 729 tests
```

---

## Three rules that explain most of the output

* **It abstains.** When a model cannot answer, the system says so rather than
  substituting its best guess. An unrecognised student is reported as
  unidentified, never as the nearest match; a window too sparse to judge is
  left unscored.
* **Nothing rests on one frame.** Every verdict is windowed over 15 seconds.
* **It does not infer emotion.** It classifies observable facial
  configuration. (Barrett et al. 2019; EU AI Act Art. 5(1)(f).)

---

## Where it stands

All figures out-of-fold and grouped by student, so no student appears on both
sides of a split.

| | Measured |
|---|---|
| **Attention score** | **0.717 accuracy, 0.708 macro F1, 0.830 off-task recall** on 82 twice-confirmed labels, against a majority baseline of 0.671 / 0.402 / **0.000** |
| **Label ceiling** | **0.770** — the most a perfect model could score, given that re-judging 127 windows blind gave κ = 0.261 and ~23% label error. 0.717 is **93%** of it |
| Student detection | P 82.2% / R 90.6% / F1 86.2% |
| Behaviour detector | mAP@50 0.607 overall; `write` **0.767**, `read` 0.419 |
| Image engagement scorer | **0.664** held out by near-duplicate group — *not* the 0.968 a random split reports |
| Face recognition, close range | 0.978 against a held-out photo, 0.032 against a stranger |
| Headcount | **over-counts**: 13–18 ids for 8 real people. Per-student figures are sound per id; the roster is not |

### The ceiling is the number that matters

0.717 is not "72% of the way to a solved problem". Re-judging the same
windows blind produced a different answer a third of the time, so roughly a
quarter of the labels are wrong, and **no model scored against them can
exceed 0.770**. The remaining gap is bought with better annotation, not a
better model.

### What was tried and did not win

Kept in the record rather than dropped, because a result that did not hold is
the only evidence that the ones that did were not luck:

| | |
|---|---|
| Mamba sequence scorer | 0.592 against the MLP's 0.609 — a tie on 179 labels |
| Vision Mamba on 12,582 images | 0.618 against ResNet18's 0.664 |
| Self-training, label spreading, 9 of 12 literature features | no gain that survived fold permutation |

The one change that helped was **peer context** — what the rest of the room
is doing — which was worth more than any architecture tried.

The Mamba scorer now **runs alongside** the shipping model rather than being
filed away. Both score every window and both numbers reach the dashboard,
because "statistically tied" is a reason to show two answers, not to pick
one. They share a window, a stride, a coverage gate and a 1–10 mapping, so a
difference between them is a difference in judgement and never in arithmetic.

### Two things the models refuse to do

* **Score footage they were not fitted on.** The attention model carries the
  per-feature range of its training set. On a one-person webcam session,
  three of its twenty-one features do not exist — there are no peers, and a
  room focus needs three students — so it declines instead of extrapolating.
* **Claim a confidence they have not earned.** Out of fold the raw model
  claimed >90% certainty on 65% of windows and was right on 60% of those, so
  the probability is Platt-calibrated before it becomes a score. Calibration
  error 0.339 → 0.035.

---

## Layout

```
backend/      the pipeline: detection, identity, actions, attention scoring
tools/        78 scripts, grouped by what you would open one for — tools/README.md
  run/          what you run to use the system (server, batch, report, enrol)
  train/        fitting the models and the improvement loop
  evaluate/     measuring against ground truth
  diagnose/     one-off investigations; each answers a question
  data/         building and maintaining the label and image sets
  assets/       images and charts for the decks
classgraph/   the installed CLI (classgraph serve / video / live / train)
models/       the two trained scorers the system loads at runtime (tracked)
results/      the measurement behind every number above (tracked)
outputs/      what a run produces — disposable, gitignored
tests/        729 tests; model-backed ones skip cleanly rather than faking a pass
docs/         setup, plans, literature review, privacy, per-person work split
ppt/          deck builders; the decks are written to the root
schema.json   the frozen per-frame output contract
graph_schema.json   the scene-graph contract
```

---

## Where to look

| | |
|---|---|
| **[docs/SETUP.md](docs/SETUP.md)** | install, run, troubleshoot. Start here |
| **[docs/PART1_PLAN.md](docs/PART1_PLAN.md)** | what is delivered, the decisions and why, limitations stated plainly |
| **[docs/CHALLENGES_AND_SOLUTIONS.md](docs/CHALLENGES_AND_SOLUTIONS.md)** | every bug found, how it was verified, the measurement behind every number |
| **[docs/PRIVACY_AND_ETHICS.md](docs/PRIVACY_AND_ETHICS.md)** | EU AI Act Art. 5(1)(f), India DPDP, retention, consent, access control |
| **[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)** | installing somewhere that is not this machine; Docker |
| **[docs/LITERATURE_REVIEW.md](docs/LITERATURE_REVIEW.md)** | the research behind each open decision |
| **[results/README.md](results/README.md)** | which tool rewrites each report, and which number it holds |
| **Work split** | [Person A](docs/WORK_PERSON_A.md) · [Person B](docs/WORK_PERSON_B.md) · [Person C](docs/WORK_PERSON_C.md) |

Base papers: ARG (Wu et al., CVPR 2019) contributes an architecture;
RDFER (Liu, Wang & Shen, T-BIOM 2025) and Dynamic Objectives Learning
(Wen et al., TMM 2020) contribute methods. The level of adoption is stated
per paper in the review deck rather than implied.

---

## Known limits

Stated here rather than discovered later:

* **The roster over-counts.** 13–18 ids for 8 real people, narrowed to
  re-identification across temporal gaps. Per-student figures are sound per
  id; the headcount is not.
* **The attention model is classroom-footage only.** One-person webcam
  sessions are refused, by design — see above.
* **`read` is the weak behaviour class** (0.419). Reading and writing differ
  by what is in the hand, not by posture; the literature reports the same
  confusion with far stronger temporal models.
* **The group-activity network is implemented and untrained.** It needs a
  labelled group-engagement corpus that is not publicly available.
* **Annotation reliability caps everything at 0.770.** Raising it is the
  single highest-value next step; nothing else can be shown to help until it
  moves.
