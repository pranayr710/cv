# ClassGraph — Session Handoff (Stage 1 / Week 1)

> Paste this whole file into a new chat to bring an assistant fully up to speed.
> Written 2026-08-07. Everything marked ✅ was verified by running it; everything
> marked ⚠️ is explicitly NOT verified. Do not upgrade a ⚠️ to a ✅ without running it.

---

## 1. Project

**ClassGraph** — classroom engagement analytics via computer vision.

- **Stage:** Week 1, Stage 1 (**Perception**). Backend only, no UI.
- **Project root:** `C:\Users\himes\pranay\pranay\sem5\projects\classgraph`
  (this is a standalone folder, NOT inside the `lumina` repo)
- **Git:** own repo, branch has 2 commits (`Initial commit`, `Update gitignore`), working tree clean.
- **Stack:** Python 3.11 (target), PyTorch, OpenCV, Ultralytics YOLOv11m, MediaPipe Face Mesh, SixDRepNet. Runs on a single NVIDIA GPU.

### Team split (the code is written to this division)
| Person | File | Responsibility |
|---|---|---|
| A | `backend/detection.py` | YOLOv11 → person + object bboxes |
| B | `backend/face.py` | MediaPipe Face Mesh → 468 landmarks + EAR |
| C | `backend/headpose.py` | SixDRepNet → yaw/pitch/roll + gaze label |
| All | `backend/integrate.py` | Wire modules → frozen JSONL |

### Non-negotiable code standards
Full type hints; Google-style docstrings with Args/Returns/Raises; **no `print()`** (use `logging`);
all config in `config.py` dataclasses (no inline magic numbers); pytest test per public function;
explicit handling of missing files / empty frames / no-detection; **never silently swallow exceptions**;
`black` formatted; `ruff` clean.

### Honesty rules (the user cares about this a lot)
Report real metrics, no inflation. If something isn't implemented, mark it TODO — don't fake it.
Tests must **skip visibly** when a dependency is missing, never fake a green pass.

---

## 2. Frozen JSON contract

`schema.json` (124 lines, at project root) is the **frozen source of truth** — do not edit it
without raising it with the team. One JSON object per processed frame (JSONL):

```json
{
  "frame_id": 0,
  "timestamp_ms": 0,
  "persons": [
    {
      "track_id": null,
      "bbox": [412, 88, 96, 210],
      "confidence": 0.91,
      "face": { "bbox": [430,95,60,70], "landmarks": [[431.2,96.0], "...468 pts..."], "ear": 0.28 },
      "head_pose": { "yaw": 3.1, "pitch": -2.0, "roll": 1.4, "gaze_label": "teacher" }
    }
  ],
  "objects": [ { "cls": "laptop", "bbox": [380,300,120,80], "confidence": 0.74 } ]
}
```

Rules: bboxes are `[x,y,w,h]` **int pixels, top-left origin, image space** (not crop space).
`track_id` is **always `null` in Stage 1** (ByteTrack fills it in Stage 2).
`face` / `head_pose` are the object-or-`null`. `gaze_label` ∈ `{teacher, left, right, down, back}`.
Every module returns lists **aligned index-wise** with its input — missing values are `None`,
never skipped entries.

---

## 3. What is built (all complete)

```
classgraph/
├── backend/
│   ├── __init__.py
│   ├── config.py       143 lines  — all tunables, frozen dataclasses, CONFIG global
│   ├── detection.py    498 lines  — Person A + standalone CLI
│   ├── face.py         403 lines  — Person B
│   ├── headpose.py     385 lines  — Person C
│   └── integrate.py    439 lines  — full pipeline + CLI
├── tests/
│   ├── fixtures/README.md
│   ├── test_detection.py  180
│   ├── test_face.py       229
│   ├── test_headpose.py   207
│   └── test_integrate.py  172
├── conftest.py         — puts repo root on sys.path
├── schema.json         — FROZEN contract
├── requirements.txt
└── README_week1.md
```

### `config.py`
Frozen dataclasses: `DetectionConfig`, `FaceConfig`, `HeadPoseConfig`, `PipelineConfig`, composed
into `Config`, exposed as the global `CONFIG`. Key values:
- Detection: `weights="yolo11m.pt"` (stock COCO, auto-downloads), `person_conf=0.40`,
  `object_conf=0.35`, `iou=0.50`, `object_whitelist=("cell phone","laptop","book")`, `imgsz=960`.
- Face: `max_num_faces=40`, `refine_landmarks=True`, `num_landmarks=468`,
  `static_image_mode=True`, `assign_min_containment=0.50`, 6-point EAR indices per eye.
- HeadPose: `weights="sixdrepnet_300w_lp_alpha1.pth"`, `yaw_side_threshold=20.0`,
  `pitch_down_threshold=20.0`, `pitch_back_threshold=-25.0`, `crop_padding=0.20`.
- Pipeline: `sample_rate=1`, `log_every_frames=30`, `default_output="outputs/stage1.jsonl"`.

### `detection.py` (Person A)
`Detector` wrapping Ultralytics YOLOv11m. `detect(frame) -> (list[Person], list[Obj])`, frozen
dataclasses `Person{bbox,confidence}` / `Obj{cls,bbox,confidence}`. Prefilters at
`min(person_conf, object_conf)` then filters per-class. xyxy→xywh with 1px min clamp (satisfies
schema `exclusiveMinimum: 0`). Device auto/cuda/cpu with **CPU fallback + warning**.
`run_on_video(path, out_json_path)` writes JSONL with `face`/`head_pose` = `null`.
Also has its own CLI: `python -m backend.detection --video X --out Y [--device] [--log-level]`.

### `face.py` (Person B)
`FaceAnalyzer` wrapping **MediaPipe Solutions Face Mesh** (`mp.solutions.face_mesh`).
`analyze(frame, person_bboxes) -> list[FaceResult]` where
`FaceResult{face_bbox, landmarks, ear}`. Runs **one** Face Mesh pass over the whole frame
(`max_num_faces=40`), then binds each detected face to the person bbox that best **contains** it
(greedy, threshold `assign_min_containment`). Result list is index-aligned with input; unmatched
persons keep their slot with all-`None` fields. Landmarks are the canonical **468** in **image
coords** (the 10 iris points from `refine_landmarks` are dropped to match schema).
`compute_ear()` = Soukupova & Cech 6-point formula, averaged over both eyes, `None` if degenerate.
Context manager + idempotent `close()`. CPU-bound (expected/fine).

### `headpose.py` (Person C)
`HeadPoseEstimator` wrapping SixDRepNet. `estimate(frame, face_bboxes) -> list[HeadPoseResult|None]`,
index-aligned: **`None` in → `None` out**, and unestimable poses → `None` with a logged reason.
`classify_gaze(yaw, pitch, config)` implements the spec precedence:
`|yaw|<20 and |pitch|<20 → teacher`; `yaw>=20 → right`; `yaw<=-20 → left`; `pitch>=20 → down`;
`pitch<=-25 → back`; the `(-25,-20]` pitch dead-zone with frontal yaw falls back to `teacher`.
Raises on non-finite angles. GPU with CPU fallback + warning (`gpu_id=0` vs `-1` per the
`sixdrepnet` package convention). Uses local weights under `WEIGHTS_DIR` if present, else the
package auto-downloads.
**Accepts an injected model** (`HeadPoseEstimator(model=...)` needing only
`predict(crop) -> (pitch, yaw, roll)`) — this is what makes it testable without the heavy package.

### `integrate.py`
`process_video(video_path, out_jsonl_path, config, *, detector=None, face_analyzer=None,
headpose_estimator=None)` — opens video with OpenCV, per (sampled) frame runs
**Detector → FaceAnalyzer → HeadPoseEstimator**, keeps all three index-aligned, assembles the
frozen contract, writes one JSON line per frame, **logs FPS every 30 frames**, shows a **tqdm**
progress bar. Estimators are built lazily from config OR injected (dependency injection).
CLI: `python -m backend.integrate --video X --out Y --sample-rate N [--device] [--log-level]`.
`--sample-rate` validated by argparse (`--sample-rate 0` → clean error, exit 2); missing video /
load failure → logged, exit 1.

---

## 4. Test status — REAL, current numbers

```
python -m pytest tests/ -q     →     10 passed, 9 skipped
```

✅ **The 10 passes are genuine logic verification** on this laptop:
- EAR math (open ≈0.30 vs closed ≈0.05, degenerate → `None`)
- Gaze-label thresholds + precedence, non-finite rejection, all labels valid
- Head-pose index alignment (`None`→`None`, only non-`None` boxes trigger inference)
- Input validation (TypeError/ValueError) and degenerate-crop → `None`
- **Full integration end-to-end**: builds a 5-second (6fps×5s=30-frame) synthetic video →
  30 JSONL lines → each **validated against `schema.json`** with `Draft202012Validator`
- **Sample-rate**: rate 5 on 30 frames → exactly 6 lines, frame_ids `[0,5,10,15,20,25]`

⚠️ **The 9 skips are ALL "heavy ML dep missing on this dev laptop"**, each with a visible reason:
- 1 — whole `test_detection.py`: `ultralytics` not installed
- 6 — `test_face.py` model tests: this laptop's mediapipe is a **Tasks-only build** lacking `mp.solutions`
- 2 — `test_headpose.py` model tests: `sixdrepnet` not installed

These flip to real runs on the GPU PC after `pip install -r requirements.txt`.
Expected there: **16 passed, 2 skipped** without fixture images, **18 passed** with them.

✅ `black --check` clean, `ruff check` clean across all files.

### Dev-laptop environment (why things skip here)
Python **3.10.10** (not 3.11), torch 2.4.1+**cpu**, no CUDA, cv2 5.0.0, numpy **2.2.6**,
pytest 9.0.3, jsonschema 4.26, tqdm 4.67.3. No `ultralytics`, no `sixdrepnet`,
mediapipe 0.10.35 (Tasks-only, unusable for this project).

---

## 5. Key decisions & findings (don't re-derive these)

### The mediapipe pin — root-caused and fixed ✅
`face.py` needs the **legacy Solutions API** (`mp.solutions.face_mesh`). MediaPipe **removed the
Solutions API from the wheels at 0.10.30+** (those are Tasks-API-only and want numpy≥2).
Verified by inspecting actual wheel contents:

| version | `solutions/face_mesh.py` present |
|---|---|
| 0.10.14 | ✅ |
| **0.10.21** | ✅ ← pinned |
| 0.10.30 | ❌ |
| 0.10.35 | ❌ (what this laptop has) |

`requirements.txt` now pins **`mediapipe==0.10.21`** (exact pin chosen by the user over a range).
✅ Empirically confirmed in an isolated venv:
`PROBE_RESULT {"version":"0.10.21","has_face_mesh":true,"constructed":true,"closed":true}`
This is also consistent with the pre-existing `numpy>=1.26,<2.0` pin.
`face.py` raises a clear `ImportError` (not a cryptic crash) if it ever meets a Tasks-only build.

### Dependency injection is deliberate
`HeadPoseEstimator(model=...)` and `process_video(..., detector=, face_analyzer=,
headpose_estimator=)` exist so the pipeline logic is testable without GPU/weights. Test fakes
return the **real** result dataclasses, so wiring/alignment/serialization are genuinely exercised.

### Model downloads
- **YOLOv11m** — Ultralytics auto-downloads `yolo11m.pt` on first use (needs internet once, then cached).
- **MediaPipe** — models are **bundled in the wheel**, no download, works offline.
- **SixDRepNet** — package auto-downloads pretrained weights on first construction (internet once).
  ⚠️ Not verified by me (package not installed here) — this is documented package behavior.
  Offline fallback: drop the `.pth` into `weights/` matching `CONFIG.headpose.weights`.

---

## 6. Test fixtures (optional, turns skips into passes)

`tests/fixtures/` currently holds only a `README.md`. Two optional images:

| File | Enables | Status |
|---|---|---|
| any classroom/person `.jpg` | face landmark + EAR tests | **Available** — see dataset below |
| `frontal_face.jpg` (straight-on face) | frontal head-pose test (`\|yaw\|<15`) | **Not yet sourced** |

`test_detects_person_on_fixture` auto-falls back to Ultralytics' bundled `bus.jpg`/`zidane.jpg`,
so it runs even with no fixture. The **frontal** test deliberately has **no** fallback (a
non-frontal sample would make the strict assertion flaky) — it skips instead.

### Dataset the user already has
`C:\Users\himes\Downloads\dataset` — Roboflow export, **481 images + YOLO labels**, 8 classes:
`handrise, look_forward, read, sleep, stand, turn_head, using_device, write`. Real
classroom/seminar frames (Pakistani university lecture rooms, multiple students).
- ✅ **Great for the classroom fixture** and as real input for running the pipeline.
- ❌ **Not suitable for `frontal_face.jpg`** — "look_forward" means facing the *instructor/board*,
  not the camera. I inspected the largest-bbox candidates; nobody looks at the lens. Best weak
  candidate: `ammad-seminar-stand_mp4-0000_jpg.rf.YEBK1APvLAES9QhtRuCX.jpg` (single standing
  person, but face is ~5% of frame and slightly turned).
- The `labels/*.txt` + `data.yaml` are **irrelevant to Stage 1** (they annotate 8 engagement
  behaviors for training a *custom* YOLO; Stage 1 uses stock COCO for person/phone/laptop/book).
  They become relevant in a later behavior-classification stage.

Suggested sources for a frontal face (searched and surfaced, not yet downloaded):
[Biwi Kinect Head Pose DB](https://www.kaggle.com/datasets/kmader/biwi-kinect-head-pose-database)
(has ground-truth yaw — filter `|yaw|<10`), [Head Pose Estimation Data (Pitch,Roll,Yaw)](https://www.kaggle.com/datasets/khaledashrafm3wad/head-pose-estimation-data-pitch-roll-yaw),
[Head Pose Image Database](http://crowley-coutaz.fr/Head%20Pose%20Image%20Database.html)
(has a `Front/` folder of 30 frontal images). Simplest alternative: a phone selfie.

---

## 7. ⚠️ IMPORTANT: the bogus CHANGELOG.md

The user shared `C:\Users\himes\Downloads\CHANGELOG.md` claiming a lot of completed work
(fine-tuned `weights/yolo_classroom.pt` at **mAP@0.5 ≈ 0.71**, `train_classroom_yolo.py`,
`train_headpose_upna.py`, `generate_sample_video.py`, `run_pipeline.py`, `sample_video.mp4`,
`outputs/stage1_sample.jsonl`, config switched to `yolo_classroom.pt`, `huggingface_hub` added).

**I verified against the filesystem: NONE of it exists in this project.** No `weights/` dir,
no `outputs/`, none of those scripts, `config.py` still says `yolo11m.pt`, no `huggingface_hub`
in requirements, and `git log` shows no such commits. It also references a **different machine**
(`C:\Users\srisu\OneDrive\Desktop\classgraph`) and is signed *"Generated by Antigravity"* — a
different tool. The mAP number has no logs or artifacts behind it.

**Treat that file as not applicable to this project.** Do not assume any fine-tuning happened.
The project is still on **stock COCO `yolo11m.pt`** (which is correct per the Stage 1 spec).

---

## 8. How to run

```bash
# from the project root
cd C:\Users\himes\pranay\pranay\sem5\projects\classgraph

# tests
python -m pytest tests/ -v          # verbose
python -m pytest tests/ -rs         # show skip reasons
python -m pytest tests/ -q          # quick

# full pipeline (needs the ML deps installed)
python -m backend.integrate --video path\to\classroom.mp4 --out outputs\stage1.jsonl --sample-rate 1
python -m backend.integrate --video X.mp4 --out Y.jsonl --sample-rate 5 --device cpu

# detection only
python -m backend.detection --video X.mp4 --out Y.jsonl
```

GPU-PC first-time setup:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m pytest tests/ -v
```

---

## 9. Where we are / what's next

**Stage 1 (Perception) code is COMPLETE** — all four modules implemented to spec, formatted,
linted, tested. Contract frozen and validated.

**Not yet done (the honest gap):**
1. ⚠️ **No real end-to-end run on real weights has ever happened.** Every model-backed path is
   unverified. This is the #1 thing to do on the GPU machine.
2. ⚠️ SixDRepNet weight auto-download unconfirmed.
3. Optional fixtures not added (2 tests skip).
4. No real-classroom-video output has been eyeballed for quality.

**Recommended next steps, in order:**
1. On the GPU PC: `pip install -r requirements.txt`, then `pytest tests/ -v` → expect
   16 passed / 2 skipped. Fix anything that surfaces (most likely spot: the `sixdrepnet`
   `gpu_id`/`predict` API signature, or the mediapipe pin vs. Python version).
2. Run `backend.integrate` on one real clip from the Roboflow dataset's source videos;
   eyeball the JSONL for sane bboxes / EAR / gaze labels. **Record real metrics, no inflation.**
3. Add the two fixture images to reach 18/18.
4. Then **Stage 2: ByteTrack** to fill `track_id` (the schema already reserves it as
   `int | null`, so no contract change needed).

**Do not** trust the Downloads CHANGELOG.md as evidence any of this is already done.
