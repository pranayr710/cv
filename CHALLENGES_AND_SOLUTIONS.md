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

## Where things stand, in numbers

| Metric | Session start | Now |
|---|---|---|
| Automated tests passing | 10 (9 skipped) | **134 (0 skipped, 0 failed)** |
| CUDA confirmed working | No | **Yes — RTX 4050** |
| Faces bound to a student (13 images) | 0 | **360 (90.5% of students)** |
| Students found (13-image sample) | 139 | **398** |
| Gaze `"down"` label reachable | No (bug) | Yes |
| Real end-to-end run completed | Never | Yes — 321 frames, schema-valid |
| Stage 2 (tracking) | Not started | Done (ByteTrack) |
| Stage 3 (engagement scoring) design | No research basis | Grounded in ~130 sourced findings |
| Stage 3 (engagement scoring) code | None | Attention tracker + peer-interaction detector, both real modules |
| Fairness/bias evidence | None gathered | Two models researched to their source; one real diagnostic run on real data |
