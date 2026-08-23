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
| Writing signal (`write`/`read`), F1 | **65.3%** (was 25.1% via the book-proximity proxy it replaced) |
| Engagement (on-task vs off-task) agreement | **82.3%** |
| Off-task detection precision | **91.1%** (recall 36.9% — deliberately the safer failure direction) |
| Expression sanity-check accuracy | 56.7% (FER2013, low-res) to 80.5% (higher-res photos) — **not a classroom-accuracy claim**, see below |
| Pipeline throughput | **0.41 FPS**, 83% CPU-bound (corrects an earlier stale 7.8 FPS claim) |

## Decisions this project made, and why (final)

| Question | Decision | Why |
|---|---|---|
| Emotion taxonomy | happy / sad / neutral | 3-class request; framed as *expression*, never inferred emotion — see `backend/expression.py` docstring for the research and legal reasons that wording is load-bearing |
| Book/pen | Kept, retrained as a real behaviour class | The proxy (book detected near a bowed head) scored F1 25.1%; a fine-tuned classifier on human-labelled `write`/`read` reached 65.3% |
| `handrise` / `stand` | Dropped from output | Too little training data (22, 59 boxes) to report honestly |
| `turn_head` / `look_forward` | Deferred to head pose, not the behaviour model | Calibrated head pose scores F1 63.2% on this vs the behaviour model's 25.0% on the same boxes |
| Camera framing | `HeadPoseConfig.yaw_reference_deg`, calibrated per camera via `tools/calibrate_gaze.py` | An uncalibrated off-centre camera silently mislabels most attending students as "looking away" — a real bug found and fixed mid-project |

## Known, open limitations — stated, not hidden

- **Phone/`using_device` detection does not work** (~20% recall). Measured to be a genuine visual ambiguity (a phone under a bowed head looks like a notebook to this camera), not a tuning gap — needs a different signal, not more training.
- **Classroom-specific expression accuracy is unmeasured.** The 56.7–80.5% range above is from public, non-classroom, non-South-Asian datasets. Real validation needs labelled crops from this project's own footage.
- **Tracking under camera motion is fragile.** On the one genuinely continuous real clip available (204s, a panning camera), zero tracks survived a continuous 60 seconds — below what per-student calibration needs. Confounded by camera motion and sparse (1 fps) sampling; not yet tested on a static camera at a realistic frame rate.
- **Generalization to a visually different classroom is poor** (F1 7.3% vs 65.3% in-distribution), traced to a labelling-density mismatch between datasets, not merely a domain-shift problem. Documented as a real ceiling on any "works in general" claim.
- **Throughput is 0.41 FPS**, not real-time at the frame level. The design tolerates this (attention judges 15-second windows, not single frames), but "real-time" should not be claimed without `onnxruntime-gpu` and the other identified speed-ups.

## Stage 2+ (identity persistence, scene graph, group activity)

Built in an earlier phase of this project, deliberately paused per direct
instruction to get Part 1 right first. Not touched during this phase.
