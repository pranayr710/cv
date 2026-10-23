# ClassGraph

Classroom engagement analytics from video: detects every visible student,
tracks them across frames, and reports per-student gaze, posture, facial
expression, and behaviour (reading / writing / sleeping / phone use) — each
signal validated against independently hand-labelled classroom images, not
self-reported.

**Current phase:** Part 1 (Perception) — student detection through per-student
signal extraction. Stage 2+ (identity persistence, scene graph, group
activity) is built but deliberately paused until Part 1 is solid.


## Install and run

```bash
pip install -r requirements.txt
pip install -e ".[learned]"
classgraph doctor          # check this machine first
```

```bash
classgraph serve                               # dashboard on :8000
classgraph video --dir <folder of clips>       # process recorded footage
classgraph train                               # refit the attention model
```

**[DEPLOYMENT.md](docs/DEPLOYMENT.md)** covers installing on a machine that is not
this one: model weights, Docker, privacy obligations, the known limits, and
the failures worth recognising on sight.

## Where it stands

| | Measured |
|---|---|
| Attention score | **0.650 accuracy, 0.648 macro F1** at 91% coverage, against a rule baseline of **0.447 / 0.371**. Out-of-fold, split by student. |
| Student detection | P 82.2% / R 90.6% / F1 86.2% |
| Behaviour model (fine-tuned YOLO11m) | mAP50 0.607; `write` 0.769 |
| Face recognition, close range | 0.978 against a held-out photo, 0.032 against a stranger |
| Image engagement model | 0.664 held out by near-duplicate group — **not** the 0.968 a random split reports |
| Headcount | **over-counts**: 13-18 ids for 8 real people. Per-student figures are sound per id; the roster is not. |

Architectures that were tried and did **not** beat the shipping model, all
measured the same way and kept in the record rather than dropped: a Mamba
sequence scorer (0.574 against 0.575, statistically tied), Vision Mamba on
12,582 images (0.618 against ResNet18's 0.664), and three semi-supervised
methods over the 405 unlabelled windows (no gain that survived fold
permutation). The one change that did help was giving the model peer context
-- what the rest of the room was doing -- which is worth more than any of
them.

The Mamba scorer now **runs alongside** the shipping model rather than being
kept in the record only. Both score every window and both numbers reach the
dashboard and the report, because "statistically tied" is a reason to show
two answers, not to pick one and present it as the answer. Where they
disagree about a student, that is on screen. The two share a window, a
stride, a coverage gate and a 1-10 mapping, so a difference between them is
always a difference in judgement and never in arithmetic.

## Layout

```
backend/      the pipeline: detection, identity, actions, attention scoring
tools/        78 scripts, grouped by what you would open one for -- see tools/README.md
  run/          what you run to use the system (server, batch, report, enrol)
  train/        fitting the models and the improvement loop
  evaluate/     measuring against ground truth
  diagnose/     one-off investigations; each answers a question
  data/         building and maintaining the label and image sets
  assets/       images and charts for the decks
classgraph/   the installed CLI (`classgraph serve`, `classgraph video`, ...)
tests/        699 tests
docs/         plans, literature review, privacy, per-person work split
ppt/          deck builders; the decks themselves are written to the root
schema.json   the frozen per-frame output contract
graph_schema.json  the scene-graph contract
```

## Where to look

- **[PART1_PLAN.md](docs/PART1_PLAN.md)** — current status: what's delivered, the
  headline numbers, the decisions made and why, and the limitations stated
  plainly rather than hidden. Read this first.
- **[CHALLENGES_AND_SOLUTIONS.md](docs/CHALLENGES_AND_SOLUTIONS.md)** — the full
  narrative log: every bug found, how it was verified (not assumed), and the
  measurement behind every number in this project.
- **[schema.json](schema.json)** — the frozen per-frame output contract.
- **Work split across the five stages, one file per person** —
  [Person A](docs/WORK_PERSON_A.md) (Perception & Identity, stages 1–2),
  [Person B](docs/WORK_PERSON_B.md) (Relations & Time, stages 3–4),
  [Person C](docs/WORK_PERSON_C.md) (Group Activity & Reporting, stage 5).
- **[HANDOFF.md](docs/HANDOFF.md)** — a dated (2026-08-07) point-in-time snapshot
  for bringing a new session up to speed on Week 1 state. Historical; see
  PART1_PLAN.md for current state instead.

## Stage 5 (Person C) — group activity & reporting

Scaffolding is in place ahead of its inputs (B's scene graph, the OUC-CGE
download); every piece states its own honest status:

- `backend/group_activity.py` — ARG-style relation graph + GCN readout for
  clip-level engagement (high/medium/low), abstention-first. Untrained and
  unvalidated by design until its inputs exist.
- `backend/reporting.py` + `tools/run/dashboard.py` — O6 reporting: class-level
  trends by default, per-student drill-down only behind an explicit
  `--per-student` flag, gaps rendered rather than smoothed over.
- `tools/data/prepare_ouccge.py` / `tools/evaluate/eval_group_activity.py` — external
  evaluation set preparation + harness (splits grouped by source recording;
  see [docs/OUC_CGE_PREP.md](docs/OUC_CGE_PREP.md)).
- `tools/evaluate/boss_agreement.py` + [docs/BOSS_VALIDATION.md](docs/BOSS_VALIDATION.md)
  — Cohen's kappa between blind human BOSS coding and system verdicts.
- [docs/PRIVACY_ETHICS.md](docs/PRIVACY_ETHICS.md) — framing vs EU AI Act
  Art 5(1)(f), India DPDP retention policy, anonymisation evidence, access
  control.
- [docs/RENAME_HANDOFF.md](docs/RENAME_HANDOFF.md) — wording patches pending
  file owners ("concentration" → behavioral proxy score; "emotion detection"
  → facial expression classification).

## Running it

```
pip install -r requirements.txt
python -m backend.integrate --video path/to/video.mp4 --out outputs/stage1.jsonl
```

Output is one JSON object per processed frame, matching `schema.json`.
Per-student behaviour classification additionally requires a fine-tuned model
(`python -m tools.train.train_behaviour`) — the pipeline runs without it (that field
is simply `null`) if the weights aren't present.

## Tests

```
pytest
```

658 tests, no ML weights required for most of them (model-backed tests skip
cleanly when a dependency is absent rather than faking a pass).
