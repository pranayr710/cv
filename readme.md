# ClassGraph

Classroom engagement analytics from video: detects every visible student,
tracks them across frames, and reports per-student gaze, posture, facial
expression, and behaviour (reading / writing / sleeping / phone use) — each
signal validated against independently hand-labelled classroom images, not
self-reported.

**Current phase:** Part 1 (Perception) — student detection through per-student
signal extraction. Stage 2+ (identity persistence, scene graph, group
activity) is built but deliberately paused until Part 1 is solid.

## Where to look

- **[PART1_PLAN.md](PART1_PLAN.md)** — current status: what's delivered, the
  headline numbers, the decisions made and why, and the limitations stated
  plainly rather than hidden. Read this first.
- **[CHALLENGES_AND_SOLUTIONS.md](CHALLENGES_AND_SOLUTIONS.md)** — the full
  narrative log: every bug found, how it was verified (not assumed), and the
  measurement behind every number in this project.
- **[schema.json](schema.json)** — the frozen per-frame output contract.
- **Work split across the five stages, one file per person** —
  [Person A](docs/WORK_PERSON_A.md) (Perception & Identity, stages 1–2),
  [Person B](docs/WORK_PERSON_B.md) (Relations & Time, stages 3–4),
  [Person C](docs/WORK_PERSON_C.md) (Group Activity & Reporting, stage 5).
- **[HANDOFF.md](HANDOFF.md)** — a dated (2026-08-07) point-in-time snapshot
  for bringing a new session up to speed on Week 1 state. Historical; see
  PART1_PLAN.md for current state instead.

## Running it

```
pip install -r requirements.txt
python -m backend.integrate --video path/to/video.mp4 --out outputs/stage1.jsonl
```

Output is one JSON object per processed frame, matching `schema.json`.
Per-student behaviour classification additionally requires a fine-tuned model
(`python -m tools.train_behaviour`) — the pipeline runs without it (that field
is simply `null`) if the weights aren't present.

## Tests

```
pytest
```

194 tests, no ML weights required for most of them (model-backed tests skip
cleanly when a dependency is absent rather than faking a pass).
