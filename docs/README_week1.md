# ClassGraph — Week 1 (Stage 1: Perception)

Backend-only perception layer. Input: a classroom video. Output: JSONL where
each line is one frame's detections in the frozen schema. No UI yet.

## Team split (3–4 days)

| Person | File | Responsibility |
|--------|------|----------------|
| A | `backend/detection.py` | YOLOv11 → person + object bboxes |
| B | `backend/face.py`      | MediaPipe Face Mesh → landmarks + EAR |
| C | `backend/headpose.py`  | SixDRepNet → yaw/pitch/roll + gaze label |
| All | `backend/integrate.py` | Day 4: wire modules → frozen JSONL |

## Source of truth (do NOT reinvent these)

- **`backend/config.py`** — every threshold, path, and model choice.
  Import `CONFIG` from it. Never hardcode a magic number in a module.
- **`schema.json`** — the frozen output contract. Validate your module's
  output against this in your tests. If the schema needs a change,
  raise it with the team first, don't silently edit.

## Setup (once, on each machine)

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Coding standards (enforced in CI)

- Python 3.11, `black` formatted, `ruff` clean.
- Full type hints on every function signature.
- Google-style docstrings with Args/Returns/Raises.
- No `print()` — use `logging.getLogger(__name__)`.
- pytest for every public function.
- Handle: missing files, empty frames, no detections, CUDA-unavailable.
- No swallowed exceptions.

## Day-by-day plan

| Day | A (Detection) | B (Face) | C (Head pose) |
|-----|--------------|----------|---------------|
| 1 | Repo skeleton + YOLO loads + runs on video | MediaPipe on full frames | SixDRepNet weights load + runs on a sample face |
| 2 | Person + object detection with whitelist | Landmarks + EAR extraction | Yaw/pitch/roll extraction |
| 3 | Wrap to contract, write tests | Wrap to contract, write tests | Map angles → gaze label, write tests |
| 4 | **All three:** run `integrate.py` end-to-end on a 5-second fixture video, validate against `schema.json` |

## Integration rules

- Every module returns lists **aligned index-wise** with its input list.
  Missing values → `None`, not skipped entries.
- All pixel coordinates in **image space**, not crop space.
- All bboxes are `[x, y, w, h]` top-left origin, integer pixels.
- Face and head-pose fields on a Person are `None` in raw detection.py output —
  Person B/C fill them in during integration.

## Testing

```bash
pytest tests/ -v                      # all
pytest tests/test_detection.py -v     # per-module
```

Every PR must:
1. Pass its module's tests.
2. Not break the schema (there's a schema-validation test in `test_integrate.py`).

## Honest metrics

Report real numbers. No inflation. If a fixture video has 3 people and your
detector finds 2, log the miss — don't hide it.
