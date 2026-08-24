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
| Pipeline throughput | **0.41 FPS**, 83% CPU-bound (corrects an earlier stale 7.8 FPS claim) |

## Decisions this project made, and why (final)

| Question | Decision | Why |
|---|---|---|
| Emotion taxonomy | happy / sad / neutral | 3-class request; framed as *expression*, never inferred emotion — see `backend/expression.py` docstring for the research and legal reasons that wording is load-bearing |
| Book/pen | Kept, retrained as a real behaviour class | The proxy (book detected near a bowed head) scored F1 25.1%; a fine-tuned classifier on human-labelled `write`/`read` reached **77.9%** |
| `handrise` / `stand` | Removed from the model entirely | Too little training data (22, 59 boxes) to report honestly |
| `turn_head` / `look_forward` | Head pose owns orientation; both excluded from the behaviour model | Calibrated head pose scores F1 63.2% vs the behaviour model's 25.0% on the same boxes. Excluding them also dissolved the label conflict blocking a second dataset, enabling the 4.3x data increase behind the 68.0% figure above |
| Camera framing | `HeadPoseConfig.yaw_reference_deg`, calibrated per camera via `tools/calibrate_gaze.py` | An uncalibrated off-centre camera silently mislabels most attending students as "looking away" — a real bug found and fixed mid-project |

## Known, open limitations — stated, not hidden

- **`read` is the weak class** — F1 50.9% with sub-50% precision, confused with `write` in both directions (both are head-down-at-a-desk). Carries `reliability="weak"` in the output so it cannot be read as certain. Note this replaced the previous weak class: `using_device` was ~20% recall and is now 72.0%.
- **Classroom-specific expression accuracy is unvalidated.** The 68% figure is measured, but on public faces downscaled to simulate classroom size — not on this project's actual students. Real validation needs labelled crops from this footage, which is the one outstanding item only the project owner can supply.
- **Tracking under camera motion is fragile, though much improved.** Raw motion tracking fragmented badly (45 tracks for ~10 people); two-pass face re-identification cuts that to 10 stable IDs. But whether 10 is exactly right is **not yet verified against ground truth**, and on panning footage no single track survived 60 continuous seconds, which is what per-student calibration needs.
- **Generalization is now measured, not assumed** — F1 68.0% on a classroom the model never trained on, up from 7.3%. Fixed by merging a second independent dataset once dropping `look_forward` dissolved the label-density conflict. Still an out-of-distribution figure below the 77.9% in-distribution one, so "works in general" remains a qualified claim, not an unqualified one.
- **Throughput is 0.41 FPS**, not real-time at the frame level. The design tolerates this (attention judges 15-second windows, not single frames), but "real-time" should not be claimed without `onnxruntime-gpu` and the other identified speed-ups.

## Stage 2+ (identity persistence, scene graph, group activity)

Built in an earlier phase of this project, deliberately paused per direct
instruction to get Part 1 right first. Not touched during this phase.
