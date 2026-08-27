# Part 1 (Perception) — Status at Close

> This file was originally written as a forward-looking plan before the work
> below existed. It is superseded by the results here. For the full narrative
> of what broke, how each finding was verified, and every number's origin, see
> **[CHALLENGES_AND_SOLUTIONS.md](CHALLENGES_AND_SOLUTIONS.md)** — this file is
> the short version for a reviewer who wants the current state, not the story.

## What Part 1 delivers

A student-detection and per-student-signal pipeline for classroom video:
finds every visible student (body or face), tracks them across frames, and
reports gaze, posture, facial expression, and behaviour (reading / writing /
sleeping / phone use) per student per frame — all validated against
independently hand-labelled classroom images, not self-reported.

## Headline numbers, all measured against human labels

| Metric | Result |
|---|---|
| Student detection (precision / recall) | **82.2% / 90.6%** |
| Face coverage of detected students | **90.5%** (was 36.7% before the SCRFD swap) |
| Writing signal (`write`/`read`), F1 | **77.9%** (was 25.1% via the book-proximity proxy, then 65.3% before the merged-dataset retrain) |
| **Behaviour on an unseen classroom, F1** | **68.0%** (was 7.3% — the single biggest fix in the project) |
| `using_device` F1 | **72.0%** (was 30.6% — went from worst class to one of the strongest) |
| Identity: stable IDs per video | **45 raw tracks → 10 person IDs** on real 5.5-min footage |
| Engagement (on-task vs off-task) agreement | **82.3%** |
| Off-task detection precision | **91.1%** (recall 36.9% — deliberately the safer failure direction) |
| Expression coverage / accuracy | **~66%** coverage at a measured **68%** accuracy for 28px faces — **not a classroom-validated claim**, see below |
| Pipeline throughput | **2.5 FPS** over sampled frames, **1.3 FPS** on frames actually containing students; **43% CPU-bound** (`tools/bench_pipeline.py`, RTX 4050, 640x360 clip, imgsz 1920). Supersedes a stale 0.41 FPS / 83% figure measured before SCRFD and ArcFace moved to the GPU. |

## Decisions this project made, and why (final)

| Question | Decision | Why |
|---|---|---|
| Facial expression taxonomy | happy / sad / neutral | 3-class request; framed as *facial expression classification*, never inferred emotion — see `backend/expression.py` docstring for the research and legal reasons that wording is load-bearing |
| Book/pen | Kept, retrained as a real behaviour class | The proxy (book detected near a bowed head) scored F1 25.1%; a fine-tuned classifier on human-labelled `write`/`read` reached **77.9%** |
| `handrise` / `stand` | Removed from the model entirely | Too little training data (22, 59 boxes) to report honestly |
| `turn_head` / `look_forward` | Head pose owns orientation; both excluded from the behaviour model | Calibrated head pose scores F1 63.2% vs the behaviour model's 25.0% on the same boxes. Excluding them also dissolved the label conflict blocking a second dataset, enabling the 4.3x data increase behind the 68.0% figure above |
| Camera framing | `HeadPoseConfig.yaw_reference_deg`, calibrated per camera via `tools/calibrate_gaze.py` | An uncalibrated off-centre camera silently mislabels most attending students as "looking away" — a real bug found and fixed mid-project |

## Known, open limitations — stated, not hidden

- **`read` is the weak class** — F1 50.9% with sub-50% precision, confused with `write` in both directions (both are head-down-at-a-desk). Carries `reliability="weak"` in the output so it cannot be read as certain. Note this replaced the previous weak class: `using_device` was ~20% recall and is now 72.0%.
- **Classroom-specific expression accuracy is unvalidated.** The 68% figure is measured, but on public faces downscaled to simulate classroom size — not on this project's actual students. Real validation needs labelled crops from this footage, which is the one outstanding item only the project owner can supply.
- **Tracking under camera motion is fragile, though much improved.** Raw motion tracking fragmented badly (45 tracks for ~10 people); two-pass face re-identification cuts that to 10 stable IDs. But whether 10 is exactly right is **not yet verified against ground truth**, and on panning footage no single track survived 60 continuous seconds, which is what per-student calibration needs.
- **Generalization is now measured, not assumed** — F1 68.0% on a classroom the model never trained on, up from 7.3%. Fixed by merging a second independent dataset once dropping `look_forward` dissolved the label-density conflict. Still an out-of-distribution figure below the 77.9% in-distribution one, so "works in general" remains a qualified claim, not an unqualified one.
- **Throughput is 2.5 FPS over sampled frames and 1.3 FPS on frames that actually contain students**, re-measured with `tools/bench_pipeline.py` after commit `787fe92` moved SCRFD and ArcFace onto the GPU. The previously published 0.41 FPS / 83% CPU-bound predated that change and is withdrawn.

  Both numbers are quoted because only one of them is about the pipeline. 22 of 41 sampled frames in the audited clip contain nobody — the camera pans off the class — and every per-person stage returns immediately on an empty frame, so an all-frames average partly measures the footage. The per-person marginal cost is **120 ms**.

  The bottleneck moved. It is no longer face work: **MediaPipe posture is now the single largest stage at 43% of frame time**, because MediaPipe has no GPU build on Windows and runs on CPU whatever the hardware. Per stage, per frame containing people: posture 371 ms, detect 92 ms, face 142 ms, head pose 84 ms, expression 74 ms, tracking 2 ms.

  Still not real-time at the frame level, and the design continues to tolerate that (attention judges 15-second windows, not single frames). The live dashboard reaches **12-19 FPS** on a webcam by running MediaPipe in tracking rather than static mode, dropping the detector to the frame's own resolution, and running expression, head pose and posture every third frame — see `tools/server.py`.

- **BoT-SORT was evaluated and not adopted.** ByteTrack associates almost entirely by IoU, and 12.6% of boxes fall below its floor between consecutive processed frames on panning footage, so the swap to BoT-SORT (global motion compensation plus appearance ReID) was expected to help. Replaying one fixed set of 735 detections through both: ByteTrack 42 track ids / 20.3% untracked / 0.7 ms per frame; BoT-SORT 44 ids / 14.7% untracked / 10.8 ms. It leaves fewer detections untracked, which is real, but produces slightly more ids at 15x the cost — and ids are what the identity gate is scored on. ReID made no difference whatever, which fits the rest of this footage's story: at 27-37px faces, different students reach 0.6-0.8 cosine similarity, so an appearance embedding has nothing to separate. Available behind `TrackingConfig.tracker="botsort"`; worth revisiting on higher-resolution footage.

- **The behaviour model is trained, and it does not transfer to this footage.** Retrained on the merged 4-class set (877 train / 58 val images, 6091 boxes), early-stopped at epoch 27 of 60 in 23 minutes on an RTX 4050. Validation mAP50 0.607: `write` 0.769, `using_device` 0.720, `sleep` 0.520, `read` 0.420.

  That settles the read/write merge experiment. The rule was to collapse them into one `studying` class unless `write` was healthy standalone — it is the strongest class in the set, so they stay separate, and `read` carries its weakness in the `reliability` field rather than being hidden inside a merge. `read` being the weak one matches the literature rather than contradicting it.

  But on the audited clip the model fires on **3 of 801 person detections**, and `write` never fires at all — even at conf 0.04, which is far below anything usable. This is not a broken model: on 20 images from its own validation distribution it produces 117 detections at conf 0.30 (write 50, using_device 38, read 28). The training images are near-full-frame classrooms at 1920x1080; the audited clip is 640x360 with roughly eight students, so each student occupies a ninth of the pixels the model was trained to see. Section 20 of `CHALLENGES_AND_SOLUTIONS.md` argued that near-zero behaviour is partly the correct answer for this footage; this measurement separates that from the resolution limit, and the resolution limit dominates.

  Consequence for the product: `concentration` can now reach "off task" in principle, but not on 640x360 input. The rule-based action layer (`backend/actions.py`) covers the gap on this footage — 15 distinct actions against the behaviour model's 3 — because objects and pose survive downscaling in a way a whole-behaviour classifier does not.

## Stage 2+ (identity persistence, scene graph, group activity)

Built in an earlier phase of this project, deliberately paused per direct
instruction to get Part 1 right first. Not touched during this phase.

### Stage 5 scaffolding (Person C, added 2026-08-25)

Engineering-ahead-of-inputs, each piece stating its own status: the
ARG+GCN group module (`backend/group_activity.py`) is untrained/unvalidated
until B's scene graph and the OUC-CGE download exist; the O6 reporting layer
(`backend/reporting.py`, `tools/dashboard.py`) renders class-level trends by
default with drill-down only behind an explicit flag; external-eval tooling
(`tools/prepare_ouccge.py`, `tools/eval_group_activity.py`) enforces
split-by-source-recording; and `docs/PRIVACY_ETHICS.md`,
`docs/BOSS_VALIDATION.md` (+ `tools/boss_agreement.py`),
`docs/OUC_CGE_PREP.md`, and `docs/RENAME_HANDOFF.md` land the month-3
ethics/validation/rename deliverables. No measured numbers yet from any of
it — none may be quoted until the inputs exist.
