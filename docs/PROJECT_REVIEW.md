# Project review — what is solid, what is not, and what to do next

An honest assessment of ClassGraph as it stands. Written to be read by someone
deciding what to trust in a presentation and what to build next, so it leads
with the weaknesses rather than the feature list.

Scope: 63 modules, ~22,000 lines, 413 tests.

---

## 1. What is genuinely solid

These have been measured on real footage, not asserted.

| Capability | Evidence |
|---|---|
| **Room-layout detection** | Focus ratios: round tables 0.76/0.97/1.26, classrooms 2.36/4.95/5.62. Threshold 1.6 sits in the gap. 7/7 on real scenes. |
| **Scene segmentation** | Histogram correlation: ordinary motion median 1.000 (p10 0.999), real changes 0.75–0.90. |
| **Object → action** | 17 actions from detections the pipeline already made. All rules unit-tested. |
| **Exclusive object ownership** | 35% of phones had overlapped 2+ students before the fix. |
| **Face recognition, close range** | Held-out webcam photo 0.978, stranger 0.032. |
| **Behaviour model** | mAP50 0.607; `write` 0.769 — the strongest class, which settled the read/write merge question. |
| **Identity clustering** | Duplicate-id frames 0 at every threshold. |

The architectural decision that matters most: **`facing_direction` comes from
shoulder geometry in image space**, so orientation never passes through
camera-relative yaw. That is what removed the per-camera calibration constant
rather than tuning it.

---

## 2. Where the project falls short

### 2.1 Identity over-counts, and it is the top of the stack

The Phase 1 gate is **≤10 ids for 8 people**. Best achieved by threshold alone
was 11, and the audited clip still reported 12–18. Everything downstream — every
per-student number, every graph node — inherits that error.

Scene splitting removed the largest contributor (a roster that was the sum of
all scenes). A visual audit of all 36 ids from a 10-minute session has now been
done, and it splits the remaining error into two different problems:

* **Genuinely-not-a-person ids.** One of 36 was 108 blurred crops with a best
  face of 33px. Raising the founding size floor to 35px removes it and keeps
  9 of 9 audited students, whose best faces were all 72px or better.
* **Contaminated ids — the larger share, and not fixable this way.** Two ids
  were a real student's track that had also absorbed furniture, hands and
  paper. Their best face is a genuine face (81px, 130px), so no founding
  threshold can reach them. This is an *association* failure, not a quality
  one: the tracker or the face-to-person binding drifted onto background.

Two candidate discriminators were measured and **rejected**, both overlapping:
blur (real ids 42–279, false 32–63) and median face size (real 56–63, false
26–69). Neither can separate a contaminated id from a real one, because a
contaminated id mostly *is* a real student.

**Per-student figures remain usable per id; the headcount does not.** The next
step is association, not another threshold.

### 2.2 Three of the five headline signals are weakly evidenced

| Signal | State |
|---|---|
| action (object-derived) | **solid** — rests on detection |
| orientation / engagement | **solid** — rests on measured geometry |
| behaviour model | **conditional** — works at 1080p, fires 3/801 at 640×360 |
| expression | **weak** — `min_confidence = 0.40` is uncalibrated |
| gaze label | **weak** — camera-relative; now bypassed for `looking_away`, still used elsewhere |

`concentration` remains in the profile alongside `on_task_pct` and
`engagement_pct` and disagrees with both — one student measures 93.0%, 23.8%
and 6.25% on the three. It can only reach "on task" through a gaze label
matched against a global reference, which cannot be right for every seat.

It is now marked `superseded_by: [on_task_pct, engagement_pct]` rather than
deleted. Deleting it is a **cross-owner change**: `backend/reporting.py` and
`tools/dashboard.py` read it, and both belong to Person C. That needs their
agreement, not a unilateral removal.

### 2.3 Constants that are still asserted

`temporal.window_seconds = 15.0` and `sustained_interaction_seconds = 20.0`
have no measurement behind them. `expression.min_confidence = 0.40` is
documented in-repo as "explicitly a starting point".

### 2.4 Coverage is uneven

418 tests, but concentrated in `backend/`. **Not one of the 27 tools has a test
file** — including `server.py`, `batch_session.py` and `report.py`, which are
what a demo actually runs. Three real bugs shipped through that gap: the
FastAPI 403 (annotations resolved against module globals), a `pgrep` liveness
check that cannot see Windows processes, and two tools whose `--help` crashed on
any Windows console because their docstrings contained a character cp1252 cannot
encode.

`tests/test_output_contract.py` now closes the narrower gap that let four schema
violations ship silently — it validates what the pipeline really writes rather
than hand-built records — but the tools themselves remain untested.

### 2.5 Performance is CPU-bound on a stage that need not be

MediaPipe posture is ~43% of frame time and has no Windows GPU build.
YOLO11-pose was benchmarked at **24.6 ms vs 371 ms**, on GPU, with shoulder
midpoints agreeing within 2.4% of box width. Not yet adopted.

### 2.6 Long runs are fragile

`batch_session` resolves identity only after seeing everything, so a run that
dies loses all of pass 1. This happened twice: once at 42/157 clips when a
session ended, once through my own faulty liveness check.

---

## 3. What is out of reach without people or better data

Not weaknesses in the code — limits of what evidence exists.

- **Expression validation** needs two human raters labelling 120 crops blind
  (Cohen's kappa). No substitute.
- **SAHI tiling** requires a size-bucketed eval set built from this footage
  first, deliberately, so an aggregate recall number cannot hide the back-row
  failure it is meant to fix.
- **Social vs academic talk** cannot be separated from vision alone. The
  difference is in what is said. Available proxies — shared object, duration,
  whether they return to task — are correlational and should be labelled as such.
- **Behaviour below 1080p.** The model works in its own domain and does not
  transfer down. That is a footage-resolution problem, not a training one.

---

## 4. What to do next, in order

1. **Fix track contamination.** The visual audit is done and points here: ids
   absorb background crops mid-track. Look at the face-to-person binding
   (`FaceConfig.assign_min_containment`) and at whether a track should reject a
   face whose embedding is far from its own running mean. Two quality
   thresholds have already been measured and rejected, so do not add a third.
2. **Agree the removal of `concentration` with Person C**, whose reporting
   layer reads it. It is marked superseded; the deletion itself is theirs to
   approve.
3. **Swap posture to YOLO11-pose** behind a config flag, with the agreement
   study across a few hundred frames rather than the single frame tested so far.
4. **Test the tools.** `server.py`, `batch_session.py`, `report.py` at minimum.
5. **Checkpoint `batch_session`** so a long run can resume.
6. **Calibrate `expression.min_confidence`** — needs the rater study, so it
   waits on item 1 of section 3.

---

## 5. What to say, and not say, in a presentation

**Defensible:**
- The system detects the room's layout from geometry and adapts what
  "attention" means to it, with no per-video configuration.
- Actions come from detected objects and body pose, with the evidence recorded
  per frame.
- Two scores are reported separately because they rest on different evidence.
- Thresholds are grounded in published work and their sensitivity is measured.

**Say with the caveat attached:**
- Per-student percentages — identity is not yet audited at scene level.
- Expression — the confidence threshold is uncalibrated.

**Do not claim:**
- That the system measures attention. It measures orientation and observable
  action. Neither is attention, and the report says so on its face.
- That it distinguishes on-topic from off-topic discussion.
- Any accuracy figure for reading vs writing beyond the model's own 0.607 mAP50
  in its own domain.
