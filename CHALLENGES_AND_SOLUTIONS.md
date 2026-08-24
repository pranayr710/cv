# ClassGraph — Challenges & Solutions Log

> Written for slide-building. Each entry: **what broke / what was wrong → how we found out → what we did → the number that proves it.**
> Every number below was measured on this project's real footage, not assumed. Session covers: GPU environment setup, two real pipeline bugs, detection tuning, a new posture-fallback module, repo hygiene, and a research phase feeding Stage 3 design.

---

## 1. Environment: from bare Python to a working GPU pipeline

**Problem:** The GPU machine had nothing installed — no PyTorch, no CUDA, no ML packages. The handoff's plan assumed a clean `pip install -r requirements.txt` would work.

**What went wrong along the way:**

| Issue | How it was found | Fix |
|---|---|---|
| `requirements.txt` pinned `sixdrepnet>=0.1.7`, which **has never existed on PyPI** (latest is 0.1.6) | The pin blocked the *entire* install — one bad line stopped numpy, opencv, ultralytics, mediapipe from installing too | Repinned to `>=0.1.6,<0.2` |
| MediaPipe failed with a cryptic DLL load error | Root-caused to a **missing Microsoft Visual C++ Redistributable** — the machine had zero MSVC runtime installed | Downloaded and installed the official redistributable (signature-verified before running) |
| Plain `pip install torch` installs a **CPU-only** build silently | Would have made the whole GPU purchase pointless — caught before it happened | Installed explicitly from PyTorch's `cu124` index |

**Result:** `torch.cuda.is_available() == True` on the RTX 4050 Laptop GPU, confirmed — not assumed. Every dependency verified importable, including `mediapipe`'s legacy Solutions API (`mp.solutions.face_mesh`), which the handoff had only verified on a *different* machine.

---

## 2. Bug #1 — Face detection found zero faces on real footage

**The problem:** Before any fix, running the pipeline on real classroom photos and a real video produced **0 faces, on every single frame** — despite people clearly having visible faces.

**Root cause:** `face.py` ran MediaPipe Face Mesh once on the **whole video frame**. Face Mesh's internal detector downscales its input to a small fixed size before looking for a face — so a face that's small *relative to the whole frame* gets destroyed before detection even runs. In a 3840×2160 frame, a perfectly visible face was invisible to the model.

**How it was proven, not guessed:**

| Test image | Whole-frame approach | Per-person-crop approach |
|---|---|---|
| 3840×2160 video, 1 student | 0 faces | **1 / 1** |
| 1920×1088 classroom CCTV, 20 students | 0 faces | **8 / 20** |

**The fix:** Crop to each detected person's bounding box *first*, then run Face Mesh on that crop. This restores the face's size relative to its input.

**A second-order finding while fixing it:** padding the crop for "safety margin" actually made things *worse* — it re-enlarges the frame relative to the face, undoing the fix. Measured: `pad=0.15` → 1/5 video frames; `pad=0.0` → 5/5. Shipped with **zero padding**, against initial intuition.

**Why it mattered:** this bug meant the entire face + gaze half of the engagement pipeline produced nothing on any real footage — it only ever "passed" because the test suite used synthetic frames and fake injected models.

---

## 3. Bug #2 — The gaze direction was backwards

**The problem:** After fixing face detection, gaze labels looked wrong: students visibly bowed over desks writing were being labeled `"back"` (looking backward/up) — and `"down"` never appeared at all, across 12 real classroom images.

**Root cause:** SixDRepNet (the head-pose model) reports pitch as **up-positive**. Our code's contract — and the `classify_gaze` logic — assumed **down-positive**. The two conventions were exactly inverted. Confirmed directly from the model's own source code (`draw_axis` function), not inferred from behavior.

**Measured effect of the fix, same 12 images:**

| Label | Before fix | After fix |
|---|---|---|
| `down` | 0 | **6** |
| `back` | 2 | **0** |
| `teacher` | 17 | 13 |

**Why it mattered:** "looking down at a phone" is arguably the single most important signal for an engagement system. The bug was silently inverting it — a student on their phone would have been unlabelable as `"down"` at all.

**Process note:** no existing test caught this, because the test's fake model fed values straight through and only checked that labels were *valid*, never that they were *correct*. A regression test pinning the sign convention was added.

---

## 4. Detection tuning — recovering the students the model was resizing away

**Problem:** Person detection was missing a large fraction of students, especially in back rows of wide classroom shots.

**Root cause:** YOLO resizes every frame to a fixed inference size (`imgsz`) before detecting anything. At the shipped default of 960px, a back-row student who was ~60px tall in a 1920px-wide frame shrank to ~30px — too small to detect.

**Measured sweep across 12 real classroom images:**

| `imgsz` | Persons found | Speed (ms/image) |
|---|---|---|
| 960 (old default) | 175 | 34 |
| **1280 (chosen)** | **236** | 50 |
| 1536 | 271 | 72 |
| 1920 | 301 | 86 |

Also lowered the person-confidence threshold (0.40 → 0.30) — distant students score lower, and for engagement statistics a missed student is worse than an extra box.

**Net effect:** 139 → 236 persons found across the same 12 images; faces recoverable went from 0 → 95 (combined with the crop fix above).

**A mistake caught along the way:** initially concluded that raising `object_conf` would remove "phantom laptop" false positives seen in one classroom image. Checking a *second* image revealed it was an actual **computer lab** — those laptops were real. The threshold was left alone rather than "fixing" a problem that was actually a labeling assumption error.

---

## 5. Building a face-independent fallback signal — `PostureAnalyzer`

**Problem underneath everything above:** even after both fixes and tuning, only ~40–45% of detected students ever have a usable face — an inherent limit of an overhead/rear-corner camera, not something more tuning fixes. A student bowed over a desk shows the camera the crown of their head; no face algorithm can recover a face that isn't in the frame.

**The idea tested:** if a face isn't visible, is *body posture* (head/shoulder/hip position) still detectable? MediaPipe Pose is another off-the-shelf, pretrained model — no training required, consistent with the rest of the pipeline.

**Measured, 167 faceless persons across 13 real classroom images:** 94 (56%) still yielded usable pose keypoints. Visually confirmed by rendering skeletons — the one standing teacher in a room full of seated writing students was the single clean "upright" case, correctly distinguished from every seated "bowed" student.

**A hypothesis that was tested and rejected, honestly:** initially tried classifying posture as "bowed" vs. "upright" using a simple geometric rule (nose position relative to shoulder line). Hand-checking a spread of 40 real crops showed this **did not hold up** — comparing the feature's distribution between students who had a visible face (presumably more upright/facing camera) and those who didn't showed the two populations almost completely overlap. **No fake classifier was shipped.** Instead, `PostureAnalyzer` returns raw geometry only (nose/shoulder/hip coordinates), explicitly documented as *not* a validated posture label — the same honesty pattern the eye-openness (EAR) module already used.

**A second idea tested and rejected:** chaining MediaPipe's dedicated face-detection model before Face Mesh, hoping it would recover more faces at distance. Measured across all 13 images: it performed *worse* (25% vs. the existing 42%) — documented as a rejected approach so it isn't re-attempted.

**Result after wiring it into the full pipeline** (real run, 321-frame video):

| | Before posture fallback | After |
|---|---|---|
| Persons with **some** signal (face or posture) | 265 / 321 (face only) | **321 / 321 (100%)** |
| Persons with **no** signal at all | 56 | **0** |

**Cost:** processing speed dropped from 11.0 → 7.8 FPS on the RTX 4050 (a fourth model now runs per person). Documented plainly, not hidden.

---

## 6. Schema change — done with the project owner's direct sign-off

**Problem:** the output contract (`schema.json`) was explicitly "frozen — do not edit without raising it with the team." Adding posture output meant changing it.

**Resolution:** the project owner authorized the change directly. Added a new, additive `posture` field to each person record (nullable, matching the existing `face`/`head_pose` pattern) rather than overwriting anything. Verified: 321/321 real output records still validate against the updated schema.

---

## 7. Repository hygiene issues

| Issue | What happened | Fix |
|---|---|---|
| Stale local `main` branch caused a rejected push | An earlier, separate history-rewrite (squashing commits, stripping AI co-authorship) left a duplicate local branch that a plain `git push origin main` grabbed by name instead of the actual work | Diagnosed via `git merge-base`, confirmed no data loss, deleted the stale branch, renamed the correct one to `main` |
| AI co-author trailer reappeared in a commit after being explicitly removed earlier | A default habit (auto-appending a `Co-Authored-By: Claude` line) conflicted with the project owner's explicit, repeated request to keep it out of GitHub's contributor list | Caught before/immediately after push in two separate instances; amended and force-pushed the correction each time; the policy was adopted going forward for this repo |
| Ruff (linter) flagged 80 issues codebase-wide | Version drift — the handoff's "ruff clean" claim was true for an older ruff release, not the one now installed | Fixed 78 by hand/auto-fix (redundant casts, stale suppressions, minor logging style); 2 left as a documented, deliberate exception (would require a new dependency for a stub-only nitpick) |

---

## 8. Research phase — grounding Stage 3 design decisions before building them

**The trigger:** two open design questions with no obvious answer — (a) how to avoid punishing normal, brief attention lapses, and (b) how to avoid mistaking a student asking a neighbor for help as "distracted."

**What was done:** four parallel deep-research passes across cognitive science, learning sciences (CSCL), computer-vision prior art, and AI ethics / sensor-fusion design — roughly 130 real, cited sources — synthesized into a shared reference document (published separately) with every claim tagged by evidence strength (well-supported / contested / unverified).

**Headline findings that will shape Stage 3:**

- A single ~2-second mental break was shown to **eliminate** vigilance decline entirely across a 50-minute task — brief lapses aren't attention failing, they're the mechanism that protects it.
- Real gaze-based lecture-attention systems get their best results aggregating over **~12-second windows**, never single frames — direct precedent for how ClassGraph should score attention.
- "Productive peer talk" is, by the learning-sciences field's own definition, a property of *what's said*, not of gaze or posture — no vision-only system in the literature claims to make that distinction, and the field's own answer when it needed to was to add a microphone, not a smarter camera heuristic.
- A **real, deployed** classroom emotion-monitoring system in China (live per-student scores on classroom screens) is a documented case of measurable student harm and public backlash — the clearest evidence for how *not* to present this kind of data.
- The EU AI Act makes inferring **emotion** from biometric data in an education setting a flat legal prohibition, not just "high-risk" — regardless of where ClassGraph is ultimately used, this sets the tone for how outputs should be framed.

**Nine concrete, source-traceable decisions came out of this** (rolling attention windows, a distinct "peer-oriented" category, equal engineering investment in the posture branch, per-student calibration, grounding the label taxonomy in an established construct, never using the word "emotion," defaulting reports to class-level trends rather than individual live scores, resetting tracking identity every session, and auditing face/gaze accuracy across skin tone) — full detail and sourcing in the published research artifact.

---

## 9. Turning research into code — a windowed attention tracker

**Problem:** the research phase (§8) produced nine concrete design decisions, but decisions on paper aren't the same as working software. The two cheapest, highest-value ones — never judging a single frame, and giving each student a personal baseline — needed to actually exist as code.

**What was built:** `backend/attention.py`, a new module that reads a finished Stage 1+2 JSONL file (does not touch the frozen schema or the live capture loop) and adds:

- **A rolling 15-second window** before judging anything as off-task — directly matching the ~12-second window validated in real published lecture-gaze research, instead of scoring frame-by-frame.
- **A 90-second sustained-distraction threshold** — a single missed glance-back never triggers a flag; only a continuous pattern does.
- **Per-student calibration** — a personal baseline built from each student's own first 60 seconds, the one concrete accuracy lever the research found actually measured (+0.084 AUC in a real classroom study).
- **An honestly-scoped 6-category taxonomy**, not an invented "engaged/disengaged" scale. Gaze toward a neighbor is its own "ambiguous" bucket — never counted as distraction, because the research showed that's exactly the mistake to avoid. A bowed head only counts as a meaningful signal when a phone is detected nearby — a bowed head alone is equally consistent with reading or writing.
- **Class-level summary as the default output** — a single student's data is only reachable by deliberately asking for it, matching the "never a bare individual verdict" guardrail from the ethics research.

**Validated on real data, not just synthetic tests:** run against a fresh real capture. Two honest, un-smoothed-over observations came out of it:
- The video was too short (13 seconds) to ever trigger calibration (needs 60) — expected behavior, not a bug, but a reminder that this needs real multi-minute footage to prove out.
- "Eyes closed" came back at 91% for this video — consistent with EAR values already measured earlier in the session, but it raises a real open question about whether the eye-closure threshold (borrowed as-is from the face module) needs its own tuning for this population. Flagged, not silently fixed.

**Result:** 24 new tests (all synthetic, no ML dependency — this is pure logic on the JSONL output shape), one of which initially failed because of a wrong assumption *in the test itself* about how many frames a rolling window retains — caught and fixed before committing, not shipped broken.

---

## 10. Closing four flagged gaps — session identity, peer interaction, taxonomy, fairness

After the attention tracker shipped, four follow-up items were flagged as unfinished. All four are now closed.

**Session-reset identity — verified, not assumed.** Grepped every module for face embeddings or re-identification logic: none exist anywhere in the codebase. Found the default usage path is already safe (a fresh tracker is built per video automatically). Added two regression tests to make this a *tested* contract instead of a comment someone could miss — including one that caught a wrong assumption of my own: I expected a "leaked" session to show up as continued ID numbering, but it actually manifests as the new session's person coming back completely unconfirmed. The test failed, told me why, and got fixed before being trusted.

**Peer-interaction detection — built, and it caught a real problem on the first real check.** Added the geometry needed to detect two students jointly oriented toward each other (extending the posture module with shoulder-orientation data), then built the detector itself. Validated it against a real classroom photo — and the top-scoring "interacting" pair turned out, on inspection, to be two students at completely different, non-adjacent desks with no visible sign of interacting at all. That false positive is now documented prominently in the code rather than buried, along with a specific measurement showing the distance threshold is looser than this classroom's real desk spacing.

**Label taxonomy — researched a real match, left the decision to the team.** Found BOSS (Behavioral Observation of Students in Schools), a validated school-psychology instrument that closely matches the categories already built — and independently confirmed the 15-second measurement window chosen earlier sits inside the range that instrument's own methodology research supports, for completely unrelated reasons. The mapping is documented as a recommendation; no category names were changed, since that's a call for the team to make.

**Skin-tone fairness audit — researched first, built what the evidence supports.** Before building an audit with no data to run it on, checked whether the two AI models in use already have documented bias findings. Found real answers: MediaPipe (the face-detection model) has an official Google fairness report that already tested a "Southern Asia" population bucket — the closest match to this project's real data — and passed, though not with the best score of any region tested. SixDRepNet (the head-pose model), by contrast, has **zero** published fairness testing anywhere. Built two things: (1) a ready-to-run stratified audit that only needs labeled data to produce real numbers, and (2) a data-free diagnostic that checks whether image quality (resolution, brightness) affects detection accuracy — and actually ran it on all 13 real classroom photos. Result: detection accuracy dropped to 7% on the lowest-resolution image, which is the same number found by hand much earlier in this project, now explained by a measurable cause instead of attributed to camera angle alone.

---

## 11. Review 1 feedback — fixing detection coverage properly

**The feedback:** not all students were boxed, only some books were boxed with no explanation of why the rest weren't, and boxes should be on faces rather than whole bodies. Part 1 to be finished before anything downstream continues.

### 11.1 The face detector was the wrong model, not badly tuned

Earlier in this project (§5) we concluded that ~42–45% face coverage was the **camera angle's** ceiling and that improving it was "a camera-placement problem, not a code problem." **That conclusion was wrong.** It was a property of the *model*: MediaPipe Face Mesh's internal detector is BlazeFace, built for short-range selfie-distance faces on a phone.

Replaced it with **SCRFD** (InsightFace), which is trained for the WIDER FACE "hard" split where ~79% of faces are under 32×32 px and ~52% under 16×16 px — a description of a classroom's back rows. MediaPipe Face Mesh is retained, but only to fit landmarks *inside* the boxes SCRFD supplies.

Measured across all 13 real classroom images:

| | MediaPipe (before) | SCRFD (now) |
|---|---|---|
| Faces bound to a student | 97 | **360** |
| Face coverage of students | 36.7% | **90.5%** |

The rejection of BlazeFace-as-pre-detector in §5 was still correct — it just led to the wrong conclusion. The lesson is that "we tried harder on the same model family and it didn't help" is not evidence that a task is impossible.

### 11.2 The bottleneck moved — and person detection turned out to be worse than face detection

SCRFD found **more faces than YOLO found persons** (421 vs 331). Rendering the unmatched faces on `img382.jpg` (19 person boxes vs 56 faces) showed every one was a **real student** in a crowded back row, not a false positive.

The cause is geometric: a classroom camera sees heads clearly and bodies barely at all — torsos are occluded by desks, by the row in front, and by each other. COCO's "person" class expects a mostly-visible human figure.

So `backend/students.py` now treats a detected face with no person box as a student in its own right, with a body box **estimated** from face geometry. Every such student is tagged `source="face_seeded"` in the output so an estimate can never be mistaken for a measurement.

| | Before | Now |
|---|---|---|
| Students found | 264 | **398** |
| — by YOLO body detection | 264 | 331 |
| — recovered from their face | 0 | +67 |

**A real regression caught by our own tests, and the fix that was rejected.** Raising `imgsz` from 1280 to 1920 (+67 persons on classroom footage) silently broke detection on smaller images — `tests/fixtures/frontal_face.jpg` (802 px) went from 1 person to **0**. The obvious fix, clamping `imgsz` to native resolution, was implemented and then **rejected on measurement**: it cost 331 → 263 persons (398 → 379 students), because upscaling genuinely *helps* crowded shots where students are small. The two cases cannot be reconciled by one rule on frame size — an 800×450 classroom shot *gains* 20 → 30 persons from the same 2.4× upscale that breaks the 802 px portrait. Shipped the setting that serves the target domain plus a loud warning for inputs where it is likely wrong, and pinned the limitation in a test so it cannot later be mistaken for a code defect.

### 11.3 Books: kept and turned into a positive signal, not dropped

The initial instinct was to drop the `book` class since nothing downstream used it. That was wrong, and the review question exposed why: **a student with a book and pen is on-task**, which is the opposite signal from a phone. Previously both a studying student and a disengaged one landed in the same ambiguous `head_down_no_device` bucket, so the system could not credit anyone for working.

Added a `head_down_writing` category (`backend/attention.py`) — bowed head + book nearby — and gave `book` its own confidence threshold, since COCO's `book` class was trained on bookshelves and closed books rather than open notebooks at a downward angle. Books detected: **21 → 35**. A phone still wins when both are near the same student: the more concerning reading is the safer default on contradictory evidence.

Threshold chosen by *looking* at the boxes, not by the count — at 0.15 false positives appear, including a box drawn around a student's head. This project made the opposite mistake once already (§4's "phantom laptop" that turned out to be a real computer lab).

**Two limitations recorded rather than smoothed over:**
- Several students visibly writing on **loose exam paper** get no box at any threshold — COCO's `book` class does not fire on loose sheets. The writing signal will under-report in exam and worksheet settings specifically. Fine-tuning on SCB-Dataset's `write`/`read` labels detects the behaviour instead of a proxy object, and is the real fix.
- Pairing the book with a **wrist keypoint** would separate "writing" from "a book is open on the desk." `backend/posture.py` extracts only nose/shoulder/hip landmarks, so this is deferred, not overlooked.

### 11.4 What did *not* improve, stated plainly

Of the 360 faces now bound to students, only **55 also yield Face Mesh landmarks**. SCRFD finds the face box; Face Mesh still cannot fit a 468-point mesh to most small classroom faces. Consequences:

- **Head pose and gaze** need only the box → these now work for 360 students instead of 97.
- **Facial expression** (the new requirement) needs only the box → unblocked at the same scale.
- **EAR / eye-closure** needs landmarks → still limited to ~55 students. Not fixed, and not claimed to be.

A `face_bbox` present with `landmarks=None` is now a representable, useful state. Under the old path every mesh failure silently became "no face at all."

---

## 12. First evaluation against human labels — and the writing proxy failing it

A 481-image labelled dataset arrived (Roboflow YOLO export, 4,603 hand-drawn student boxes, 8 behaviour classes: `look_forward`, `turn_head`, `using_device`, `read`, `write`, `sleep`, `stand`, `handrise`; all 1920×1080). Every detection number in this project up to here was **self-measured** — counted by eye, or compared against another model's output. This is the first evaluation against **human labels**.

**Student detection, 98 images stratified across all 11 source clips:**

| Metric | Value |
|---|---|
| Ground-truth students | 819 |
| Precision | **82.2%** |
| Recall | **90.6%** |
| F1 | **86.2%** |

**Recall and face coverage by ground-truth behaviour:**

| Behaviour | GT boxes | Recall | Has a face |
|---|---|---|---|
| look_forward | 413 | 86.9% | 99.0% |
| turn_head | 121 | 90.9% | 95.9% |
| using_device | 103 | 98.1% | 91.3% |
| write | 60 | 100.0% | 98.3% |
| read | 53 | 96.2% | 90.6% |
| sleep | 46 | 84.8% | **73.9%** |
| stand | 20 | 95.0% | 80.0% |

`sleep` has the worst face coverage, which is the expected direction — a sleeping student's face is against the desk or hidden in their arms — but it is also the behaviour we would most want a face for.

**A measurement trap that would have produced a false conclusion.** The first run scored precision 54% / recall 60% at IoU ≥ 0.5. That contradicted face coverage of 99% on the same students, so it got checked rather than reported. Rendering ground-truth and detected boxes on one frame showed **11 GT students and 11 detected students — the same 11 people** — with only 7 pairs clearing IoU 0.5. The cause is an annotation-convention mismatch: this dataset's boxes are tight head+torso regions, YOLO's are full-body. Scoring "did we find this student" by mutual centre containment instead gives the 90.6% above. Both modes are kept in `tools/eval_detection.py` (`--match centre|iou`) so the choice is visible rather than buried.

**The writing signal measured, and it fails.** The book-proximity proxy from §11.3 was tested against the ground-truth `write`/`read` labels:

| Metric | Value |
|---|---|
| Precision | **31.9%** |
| Recall | **20.7%** |
| F1 | **25.1%** |

This confirms with real numbers what §11.3 flagged as a suspicion: COCO's `book` class is not a usable proxy for "this student is writing." It was the honest stopgap available without labels; labels now exist, so it should be replaced by fine-tuning on the `write`/`read` classes directly (664 labelled boxes). Reported rather than quietly retained — a 25% F1 signal must not be presented as a working feature.

**Assessment of the dataset itself — usable, with one caveat that could invalidate results.** 481 frames come from only **11 source videos**, so consecutive frames are near-duplicates and the largest single clip is 24.5% of all images. Any train/validation split must be **by clip, not by frame**, or scores will be inflated by memorised near-identical frames. `tools/analyse_labelled.py` reports this, and the evaluation sampler stratifies by clip for the same reason. Two further limits: severe class imbalance (`handrise` 26 boxes, `stand` 60 — too few to train), and a median of 10 students per image versus up to ~56 in the existing 13-image set, so it does **not** stress the small-face/crowding problem that motivated SCRFD. The two sets are complementary, not substitutes.

---

## 13. Making the expression signal reliable, and verifying Part 1 end to end

### 13.1 Three measured improvements to expression

Expression shipped working but weak: median confidence 0.42, with 67% of predictions below 0.50. Three levers were tested on real classroom faces.

| Lever | Result | Kept? |
|---|---|---|
| **Face alignment** using SCRFD's 5 keypoints | median confidence 0.421 → **0.481**; sub-0.5 predictions 64% → **55%** | Yes |
| **Temporal averaging** over 9 frames | consecutive-frame label flips 6.8% → **1.4%** (5x fewer) | Yes |
| **Larger model** (`enet_b2_8`) | median confidence **0.271**, 98% under 0.5 — much worse | **Rejected** |

Alignment matters because AffectNet was trained on *aligned* faces, so a raw box crop was out of distribution. Temporal averaging matters because a student's expression does not genuinely change five times a second — that flipping was measurement error, and it is removed for free. Distributions are averaged rather than labels voted on, so many weak-but-consistent frames can outvote one confidently-wrong frame.

Verified in the shipped pipeline, not just in the experiment (20 held-out images, 270 students classified):

| | median confidence | abstentions |
|---|---|---|
| box crop (old) | 0.419 | 125 / 270 (46%) |
| aligned (shipped) | **0.470** | **92 / 270 (34%)** |

Alignment turns 33 abstentions into usable labels. Worth noting honestly: `sad` predictions dropped from 9 to 3 under alignment, which suggests some of the box-crop `sad` labels were artifacts of a misaligned crop rather than real signal.

### 13.2 Abstention — the answer to "we cannot be right all the time"

Since 55% of predictions fall below 0.50 confidence, a system that always emitted one of happy/sad/neutral would be presenting a coin flip as a finding. Below `min_confidence` the output is now **`"uncertain"`** — a first-class label in the schema, not an error state.

`min_confidence = 0.40` is explicitly **a starting point, not a calibrated threshold**. Calibrating it requires labelled expression crops that do not exist yet, and the trade of coverage against reliability should be made against labels rather than by taste.

### 13.3 Part 1 verified end to end

A 40-frame 1920x1080 clip was rebuilt from consecutive dataset frames and run through the complete Stage 1+2 pipeline. **All 40 frame records validate against `schema.json`.**

| Signal | Coverage of student-frames |
|---|---|
| Face box | 88.9% |
| Face-mesh landmarks | 78.2% |
| Head pose / gaze | 88.9% |
| Posture geometry | 90.7% |
| **Expression** | **86.5%** |
| Track id assigned | 98.6% |

431 student-frames, 14 distinct track ids, 176 objects. Expression broke down as 235 neutral, 19 sad, 3 happy, **116 uncertain (31%)** — the abstention rule doing visible work.

Landmark coverage is 78.2% here versus ~15% on the crowded exam frame measured earlier. The difference is face size, not code: this footage has ~10 students at medium distance, that one had ~50 at distance. Both numbers are real; neither generalises to the other.

**One anomaly worth flagging rather than burying:** gaze came back `right` for 320 of 383 faces (84%). That is plausible as camera geometry — a side-mounted camera makes students facing the teacher all appear turned one way — but a signal that returns one value 84% of the time is not discriminating, and it needs checking against the head-pose sign convention before any gaze-derived claim is trusted on this footage. Not investigated yet; recorded so it is not mistaken for a working signal.

---

## 14. The gaze labels were meaningless on an off-centre camera

**Found by following up an anomaly rather than shipping it.** The end-to-end run in §13.3 produced `gaze_label` `"right"` for 320 of 383 faces (84%). A signal that returns one value 84% of the time is not measuring anything, so it got investigated instead of reported.

**The head-pose model was correct.** Rendering yaw per face showed the students really are rotated +26° to +77° relative to that camera, median +37°, because the camera is corner-mounted and the board is off-frame to the left.

**What was wrong was the label's meaning.** `classify_gaze` treats yaw ≈ 0 as "attending", which silently assumes *the camera sits where the teacher does*. It does not. So every attending student was bucketed as looking away — and `backend/attention.py` feeds `oriented_away` into its off-task reasoning, so this would have corrupted **every engagement figure** derived from this footage. This is the same class of bug as the pitch-sign inversion in §3: the model was fine, the interpretation was not.

**Fix:** `HeadPoseConfig.yaw_reference_deg`, subtracted from yaw before bucketing, so labels mean "relative to the front of the room" rather than "relative to the lens". Default 0.0, so front-mounted cameras are unaffected.

| Label | Before | After (ref +37°) |
|---|---|---|
| teacher | 5.2% | **37.9%** |
| left | 1.6% | 14.1% |
| right | **83.6%** | 9.9% |
| down | 9.7% | **38.1%** |

The calibrated split matches what the frame shows: roughly a third attending, a third heads-down writing.

`estimate_yaw_reference()` derives the value from pooled yaw samples, and `tools/calibrate_gaze.py` runs it over a finished JSONL and prints the before/after split. It takes the **median** deliberately — a mean would be dragged off-target by the genuinely-turned-away minority, which is exactly the population the reference must ignore — and refuses to estimate from under 20 samples, since a wrong reference is worse than none.

**The assumption, stated rather than buried:** this rests on *most students facing the front most of the time*. True for an ordinary lesson; **false for group-work footage**, where it would confidently return the wrong reference. A derived value must be checked against one rendered frame before adoption.

**Honest cost:** with a reference set, a student looking straight at the camera now reads as `left`, not `teacher`. That is correct for a corner camera — facing the lens means facing away from the board — and it is pinned by a test so it is not later mistaken for a regression.

---

## 15. Assigning each signal to the model that actually measures it best

The fine-tuned behaviour model (§16) scored F1 25.0% on `turn_head`. But "facing forward versus turned" is precisely what a head-pose model exists to measure, so both were tested on the same ground truth (371 labelled boxes with a usable face) instead of assuming the newer model should own it.

| Detecting `turn_head` | Precision | Recall | F1 |
|---|---|---|---|
| Fine-tuned behaviour model | 79.2% | 14.8% | 25.0% |
| Head pose, uncalibrated | 34.9% | 84.1% | 49.3% |
| **Head pose, calibrated** | 51.8% | 81.0% | **63.2%** |

Calibrated head pose is **2.5× better** than the behaviour model here. This also independently confirms §14 on a *different camera and dataset*: calibration alone moved head pose from 49.3% to 63.2%. The reference estimated for this second camera was +35.5°, close to the +37.4° of the first — both are off-centre mounts.

**Resulting division of labour, each assignment made by measurement:**

| Signal | Owner | Evidence |
|---|---|---|
| Finding students | SCRFD + YOLO | recall 90.6% vs 70.5% for the behaviour model |
| `look_forward` / `turn_head` | Calibrated head pose | F1 63.2% vs 25.0% |
| `write` / `read` | Fine-tuned behaviour model | F1 67.9% vs 25.1% for the book proxy |
| `sleep` | Fine-tuned behaviour model | F1 59.4% |
| Phone / `using_device` | **No good owner yet** | 14.4% recall — a real, open gap |

The last row is stated as a gap rather than filled with the least-bad option. Phone use is one of the most important off-task signals, and neither COCO's `cell phone` class nor the fine-tuned model detects it reliably on this footage.

---

## 16. The behaviour model: first-generation numbers

> **Superseded by section 20.** Every figure below is from the single-dataset model. It is kept because the analysis (especially 16.1's error-source split and 16.2's engagement framing) is what led to the fix — but for current numbers read section 20.

Fine-tuned YOLOv11m on 423 labelled frames from 8 clips, validated on 2 held-out
clips (58 images, 650 labelled students). Best **mAP50 0.437 at epoch 45**, early
stopped at 58. Config that fits a 6.4 GB card: batch 4, imgsz 960, 5.03 GB used,
~40 s/epoch.

**The question it was built to answer:**

| Writing signal (`write`/`read`) | Precision | Recall | F1 |
|---|---|---|---|
| Book proximity (replaced) | 31.9% | 20.7% | **25.1%** |
| Fine-tuned model | 55.8% | 78.6% | **65.3%** |

**Per class**, held out:

| Class | GT | Precision | Recall | F1 |
|---|---|---|---|---|
| look_forward | 246 | 62.2% | 85.0% | 71.8% |
| write | 91 | 62.2% | 75.8% | **68.3%** |
| sleep | 41 | 65.6% | 51.2% | 57.5% |
| turn_head | 128 | 78.3% | 36.7% | 50.0% |
| read | 49 | 33.7% | 59.2% | 43.0% |
| using_device | 90 | 55.9% | 21.1% | **30.6%** |
| handrise / stand | 4 / 1 | - | - | unmeasurable |

**Resolution: 960 beats 1920, the opposite of the person detector.** Evaluated at
the pipeline's 1920 the writing signal *drops* from F1 65.3% to 55.6%, and
student recall from 70.5% to 67.1%. Matching train and test resolution matters
more than absolute resolution here. Measured rather than assumed, since the
reverse holds for YOLO person detection (section 11.2).

### 16.1 Splitting detection failures from classification failures

Scores said `using_device` was weak; they did not say why. Reporting where each
error goes did:

| `using_device` outcome | Share |
|---|---|
| Labelled `look_forward` | 27% |
| Labelled `read` | 17% |
| Labelled `write` | 16% |
| Never found at all | 16% |
| **Correct** | **21%** |

Phone use is not being *missed*, it is being read as **ordinary desk activity**.
A student head-down over a phone and one head-down over a notebook look alike
from a classroom camera. That is a genuine visual ambiguity, and it explains why
the earlier confidence sweep could not fix it: there was no threshold to find.
Solving it needs a different signal (reliable phone object detection, which COCO
does not provide at this resolution), not more training on these labels.

The same view independently justified deferring `turn_head` to head pose: 24% of
it is labelled `look_forward` and a further 27% is never found, so the behaviour
model loses roughly half that class before classification even begins.

### 16.2 The metric that actually matters

Per-class F1 of 43-72% reads as weak, but no teacher asks whether a student's
pose class is `read` or `write` -- they ask whether the student is working.
`read` mistaken for `write` is a wrong class but a **correct engagement
reading**, and only the binary scores that honestly.

| Engagement (on-task vs off-task) | Value |
|---|---|
| Overall agreement, given detection | **82.3%** |
| Overall agreement, strict | 73.7% |
| Detecting off-task -- precision | **91.1%** |
| Detecting off-task -- recall | 36.9% |

**That precision/recall shape is right for this application, not merely
flattering.** When the system says a student is off-task it is right 91% of the
time, and it misses roughly two thirds of off-task students. Wrongly flagging an
attentive student to a teacher is far worse than staying quiet about a distracted
one -- and this project already cites a deployed system that harmed students by
doing the former. The errors fall on the safer side.

`turn_head` is excluded from the binary entirely (133 ground-truth boxes, left
ungraded and reported as ungraded). The CSCL literature is explicit that a
student turned toward a neighbour cannot be called on- or off-task from vision
alone; the field's own answer when it needed that distinction was to add a
microphone.

---

## 17. Throughput: the existing FPS claim is stale by ~19x

Measured per frame, GPU idle, on 1920x1080 frames with ~14 students each:

| Stage | ms | Share | Device |
|---|---|---|---|
| SCRFD faces | 959 | **39%** | CPU |
| Posture | 662 | **27%** | CPU |
| Expression | 263 | 11% | CPU |
| SixDRepNet pose | 250 | 10% | GPU |
| FaceMesh landmarks | 137 | 6% | CPU |
| YOLO detect | 130 | 5% | GPU |
| Behaviour | 45 | 2% | GPU |
| **TOTAL** | **2446** | | **0.41 FPS** |

**Section 5 of this document claims 7.8-11 FPS. That figure predates SCRFD,
expression and behaviour, and must not be presented.** The honest current number
is **0.41 FPS**, with **83% of frame time CPU-bound**.

Three identified recoveries, none yet applied:

1. **`onnxruntime-gpu`.** SCRFD and expression are 50% of frame time running on
   CPU while the GPU idles. The single largest lever. Not attempted yet because
   it needs a CUDA/cuDNN version match and could destabilise a working
   environment -- worth doing deliberately, not casually.
2. **`CONFIG.posture.only_when_faceless`.** Posture is 27% of frame time and
   exists as a face-independent *fallback*, yet runs for everyone. That was right
   when face coverage was 42%; at 89% most of it is redundant. Implemented, but
   **off by default**, because `backend.peer_interaction` needs shoulder
   keypoints for *both* students of a pair and would be effectively disabled.
   Saving 19% by silently breaking a feature is a bad trade.
3. **Frame sampling.** The design does not need per-frame throughput:
   `backend.attention` judges 15-second windows, never single frames. At 0.41 FPS
   a 15-second window still collects ~6 samples, so the system is usable as
   designed even now. But "real-time" should not be claimed.

---

## 18. Generalization test on an independent classroom dataset -- fails hard

> **Resolved in section 20.** The 7.3% recorded here was real, and the label-density diagnosis below is exactly what made the fix possible: F1 is now 68.0% on this same dataset.

A second, wholly independent labelled dataset arrived (629 images, different classroom, different country/camera style, CC BY 4.0, Roboflow: `classroom-na2vo/classroom-student-dataset`), plus an 11-minute continuous classroom video. Unlike the 13-image and 481-image sets already used, this one was collected by someone else entirely -- the right test of whether the fine-tuned behaviour model learned classroom behaviour or memorised its own 8 training clips.

**It memorised the 8 clips.** Running the model (unmodified, never trained on this data) against its `test` split, with classes mapped `Using Phone -> using_device`, `Reading -> read`, `Sleeping -> sleep`, `Writing -> write`, `Hand Rising -> handrise`:

| | In-distribution (own held-out clips) | This independent dataset |
|---|---|---|
| Overall F1 | 65.3% (writing) | **7.3%** |
| `using_device` recall | 21% | 5.4% |
| `write` recall | 75.8% | **0.0%** |
| `handrise` recall | unmeasurable (22 boxes) | **0.0%** |

**Verified by rendering, not trusted as a number alone** -- a mismatch this large could be a box-convention artifact rather than a real failure, so a frame was rendered with both ground truth and predictions before drawing any conclusion. It confirmed a genuine failure: in a classroom with 10+ students visibly reading, writing, raising hands and using phones, the model produced only 2 boxes total, and both were wrong.

**Root cause, visible in the same render: a labelling-density mismatch, not only a visual domain shift.** This new dataset labels *only* the notable behaviours -- phone use, sleeping, hand-raising -- and leaves attentive students completely unannotated. Our own training data labels every visible student, including plain `look_forward`. Naively merging the two would teach the model that a normal attentive student (boxed as `look_forward` in one dataset) is background/nothing (unboxed in the other), which is a direct label conflict, not just more data. A retrain was deliberately NOT attempted this session for that reason -- it needs curation (re-labelling attentive students in the new set, or restricting its use to eval-only) before it can safely join training, not a quick merge.

**What this dataset is used for instead, right now:** an honest, permanent generalization check, kept separate from the training pool. The 7.3% F1 is the real ceiling on any claim of "this works in a general classroom" until either more diverse training data is curated or the model is explicitly scoped to "classrooms similar to the training footage."

---

## 19. First tracking test on genuinely continuous footage -- calibration never fires

Two "continuous classroom videos" handed over turned out to be edited research/documentary clips (Young Lives / CLASS observation series) with title cards and hard cuts -- confirmed by rendering frames and checking, not assumed from file length. A histogram-correlation cut detector (robust to camera motion, unlike a raw pixel-diff check that flagged ordinary movement as false cuts) found one genuinely continuous 204-second stretch inside one of them and it was verified by eye (a smooth camera pan with coherent subtitled dialogue, not a cut) before being used.

**Result, processed at 1 fps (204 frames):**

| | Value |
|---|---|
| Distinct track IDs assigned | 28 |
| Max concurrent people in any frame | 9 |
| ID-per-real-person ratio | 3.11 (1.0 = stable) |
| Tracks surviving a continuous 60s | **0** |
| Tracks spanning 90s | 2 |

Zero tracks reach the 60 continuous seconds `backend/attention.py`'s per-student calibration requires. On this footage, calibration would never fire at all.

**Two confounds this test cannot separate, both stated rather than glossed over:**

1. **The camera pans.** This is handheld footage swinging between teacher and students, not the fixed classroom camera the system is designed around. A person crossing the frame between two 1-second samples can look like a different person to an IoU-based tracker (ByteTrack) -- expected under that motion, not necessarily a defect.
2. **1 fps sampling was forced by compute cost**, not chosen to match a target rate. Processing every frame of 204s at the measured clean 0.41 FPS would take ~4 hours; this run sampled to get a result in-session. 1 fps is a large gap for any IoU tracker regardless of camera motion.

So this is a real, useful measurement of what happens under camera motion + sparse sampling -- a **genuine deployment risk** if the real camera is not perfectly rigid or the live system cannot sample densely enough -- but it is **not** proof that ByteTrack is weak on a static camera at a reasonable sample rate. That test still does not exist and needs footage from a camera that does not move, ideally processed at a much higher sample rate than 1 fps once throughput (section 17) is improved.

---

## 20. Fixing the generalization failure: two datasets, one 4-class model

Section 18 recorded the behaviour model collapsing to **F1 7.3%** on an
independent classroom dataset, and section 19's video work then hit the same
wall from the other side: **zero** detections across an entire 640x360 video,
still zero at conf 0.05 and still zero with the frame upscaled 2x and 3x. So it
was neither a threshold nor a resolution problem — the model had memorised its
own eight training clips.

### What actually unblocked it

Section 18 identified a **label-density conflict** as the reason the second
dataset could not simply be merged: it labels only notable behaviours and leaves
attentive students unannotated, while ours labels every student including plain
`look_forward`. Merging directly would teach the model that an attentive student
is background.

Dropping `look_forward` dissolves that conflict — and dropping it was correct
independently, because `backend/behaviour.py` **already suppressed** it (section
15: calibrated head pose measures orientation at F1 63.2% vs the behaviour
model's 25.0%). The model had been carrying, and being numerically dominated by,
a class the pipeline discarded: `look_forward` alone was **2384 of 4603 boxes**.

### The merge

| class | ours | second dataset | merged |
|---|---|---|---|
| read | 344 | 1063 | 1407 |
| write | 320 | 975 | 1295 |
| sleep | 232 | 1474 | 1706 |
| using_device | 522 | 1975 | **2497** |
| **total** | 1418 | 5487 | **6905** (4.9x) |

877 training images after dropping frames with no remaining label. Split **by
source**, never by frame. The second dataset's own `test` split was held
entirely out of training so it stays a true out-of-distribution check. Trained
with `scale=0.6` plus lighting augmentation — deliberately, because the previous
model had only ever seen clean 1920x1080 input.

### Results, against three gates declared before training

| | before | after |
|---|---|---|
| mAP50 | 0.437 | **0.653** |
| writing signal F1 (held out) | 65.3% | **77.9%** |
| **F1 on an unseen classroom** | **7.3%** | **68.0%** (9.3x) |
| `using_device` F1 | 30.6% | **72.0%** |
| `write` F1 | 68.3% | 70.9% |
| `sleep` F1 | 57.5% | 65.8% |
| `read` F1 | 43.0% | 50.9% |

### A gate I designed badly, and the correction

Gate 2 asserted "non-zero detections on the 640x360 video". The new model still
returned zero, which looked like failure. Investigated instead of reported:
downscaling second-dataset images to that **exact** resolution still yielded 65
detections (vs 95 at native), so the model handles 640x360 fine. Rendering the
sampled video frames showed the real reason — they contain a teacher at a
blackboard and students listening, i.e. **no reading, writing, sleeping or phone
use at all**. Zero detections is the *correct* answer for footage containing none
of the four classes. The gate conflated "detects nothing" with "is broken", and
that is a flaw in the test, not the model.

### The weak class moved, and the tags moved with it

`using_device` was the headline weakness for most of this project (~20% recall,
section 16.1) and is now among the strongest at 72.0%. `read` is now weakest at
F1 50.9% with **sub-50% precision**, confused with `write` in both directions —
understandable (both are head-down-at-a-desk) but not something a consumer
should read as certain. `_WEAK_CLASSES` was re-measured and its membership
changed accordingly, rather than being left as a stale assumption; nine tests
that pinned the old classes were updated to match measurement, with the
suppression *mechanism* tests rewritten against a configured class rather than
deleted.

---

## Where things stand, in numbers

| Metric | Session start | Now |
|---|---|---|
| Automated tests passing | 10 (9 skipped) | **134 (0 skipped, 0 failed)** |
| CUDA confirmed working | No | **Yes — RTX 4050** |
| Faces bound to a student (13 images) | 0 | **360 (90.5% of students)** |
| Students found (13-image sample) | 139 | **398** |
| Student detection vs human labels | Never measured | **P 82.2% / R 90.6% / F1 86.2%** |
| Writing signal vs human labels | Never measured | **F1 77.9%** (25.1% via book proxy, 65.3% pre-retrain) |
| Behaviour on an UNSEEN classroom | Never measured | **F1 68.0%** (was 7.3%) |
| Engagement (on/off-task) agreement | Never measured | **82.3%**, off-task precision 91.1% |
| Pipeline throughput | claimed 7.8 FPS | **0.41 FPS measured**, 83% CPU-bound |
| Gaze `"down"` label reachable | No (bug) | Yes |
| Real end-to-end run completed | Never | Yes — 321 frames, schema-valid |
| Stage 2 (tracking) | Not started | Done (ByteTrack) |
| Stage 3 (engagement scoring) design | No research basis | Grounded in ~130 sourced findings |
| Stage 3 (engagement scoring) code | None | Attention tracker + peer-interaction detector, both real modules |
| Fairness/bias evidence | None gathered | Two models researched to their source; one real diagnostic run on real data |
