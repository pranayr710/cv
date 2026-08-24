# Person A — Perception & Identity (Stages 1–2)

> *"Who is in the room, and is it the same person a minute later."*

You own the bottom of the stack: every module that turns pixels into a
per-student record, and the identity layer that keeps that record attached to
one human for a whole lecture.

**You are the critical path.** All three originality claims in the review deck
— a per-student attention trend, distraction spreading between neighbours, and
an honest "we don't know" — require `person_id` to be correct. Stages 3–5
cannot be validated on a roster that is wrong, no matter how well they are
built.

---

## Phase 1 — Repair identity (weeks 1–2, hard gate)

The identity layer is mid-repair. `backend/identity.py` was switched from
sequential greedy matching to **constrained agglomerative clustering** (commit
`1b1d404`), following `docs/LITERATURE_REVIEW.md` section 2: the audited
3-person merge was a **transitivity error**, structural to pairwise matching,
which no better `match_threshold` value can fix.

That change fixed two failures and introduced a third. Measured on the audited
5.5-minute clip:

| | before | after | |
|---|---|---|---|
| frames containing a duplicate id | 56 / 331 (17%) | **0** | fixed |
| detections with no id at all | 20.5% | **6.6%** | fixed |
| positive `person_id`s | 12 | **18** | **regressed** |

Ground truth for that clip is **7 students + 1 teacher = 8 people**
(`docs/IDENTITY_GROUND_TRUTH.md`). So duplicates and merges are structurally
gone, but the clustering now **over-splits**: one person is being spread across
several ids.

### Tasks, in order

1. **Run the sweep.** `tools/sweep_identity.py` grades `match_threshold`
   against the known headcount from a single expensive video pass. It is
   written and has never been run. `match_threshold = 0.35` has been an
   uncalibrated ballpark since the day it was typed, and there was no ground
   truth to calibrate it against until now.

   ```
   python -m tools.sweep_identity --students 7 --video <the audited clip>
   ```

2. **Diagnose the over-split.** Two prime suspects, both cheap to test:
   - **Surrogate keys.** `IdentityConfig.identify_untracked` enters each
     untracked detection as its own single-frame surrogate key. Check whether
     these are each founding their own cluster instead of joining an existing
     one.
   - **Quality-gated identity creation — the unimplemented half of literature
     review §2.** The recommendation was two changes, and only one shipped.
     The other: gate *identity creation* by a quality score (face size, blur,
     pose angle) so a low-quality face can **join** an existing cluster but can
     never **found** a new one. MagFace/AdaFace establish that low-quality
     embeddings cluster with higher variance near the origin, so no single
     threshold is simultaneously safe for high- and low-quality faces in the
     same video.

3. **Swap the tracker config** — `bytetrack.yaml` → `botsort.yaml` with
   `with_reid: True` (literature review §3). Hours, not days. Measure against
   the recorded baseline: 12.6% of boxes fall below ByteTrack's IoU association
   floor between consecutive processed frames.

4. **Re-audit visually.** `tools/audit_identity.py` renders one contact sheet
   per id. Judge every surviving id by eye and rewrite the verdict table in
   `docs/IDENTITY_GROUND_TRUTH.md`. Every automated metric this project has
   ever collected was blind to duplicates, merges, posters and the teacher —
   because all four are perfectly consistent detections of the wrong thing.

### Gate

**ids ≤ 10 for 8 real people · duplicate frames stays 0 · no-id ≤ 7% ·
confirmed by contact sheets, not by the count alone.**

Escalate on day 10 if this is not converging. Do not let Stages 3–5 spend a
month building on an identity layer that is known to be wrong.

---

## Phase 2 — Make the per-student signals non-empty (weeks 3–4)

**The blocker nobody has named yet.** In the per-student profile output, a
typical student over 112 frames has:

- expression classified in **5** frames, unavailable in 107
- behaviour classified in **0** frames, unavailable in 112

Person B is about to build a scene graph whose node features are expression and
behaviour. On this footage those features are almost entirely absent. Part of
that is correct (§20 of `CHALLENGES_AND_SOLUTIONS.md` established that zero
behaviour detections is the *right* answer for footage containing no reading,
writing, sleeping or phone use) and part of it is a 640×360 face-size limit.
Separating those two causes is your job, and it is a checkpoint for the whole
project at the end of month 3.

### Tasks

1. **SAHI-style tiling** (literature review §1) — slice the frame, detect per
   slice at native resolution, merge with NMS. Pure inference-time wrapper, no
   retraining, and Ultralytics ships tiled inference natively. Apply it **only
   to the back-of-room region**, found by comparing YOLO person-box sizes
   across the frame.

   **Build a size-bucketed eval set from our own footage before changing
   anything** — a few dozen frames, faces bucketed by pixel size. No published
   benchmark validates on our fixed-camera geometry, and an aggregate recall
   number would hide exactly the back-row failure you are trying to fix.

2. **Re-benchmark throughput.** `PART1_PLAN.md` publishes **0.41 FPS, 83%
   CPU-bound**. That predates commit `787fe92`, which moved SCRFD and ArcFace
   onto the GPU — roughly half of frame time. The published number is now wrong
   *in our own favour*, which is the worst direction for a project that sells
   itself on honest measurement. Re-measure the per-stage table and correct the
   docs.

3. **Lead the expression validation study** (literature review §4). 120
   stratified crops and a manifest are already staged under
   `outputs/expression_labels/`; `tools/label_expressions.py` (blind,
   resumable) and `tools/score_expression_labels.py` are written. **Zero labels
   have been collected.**

   You are rater 1, **Person C is rater 2**, labelling independently and blind
   to each other and to the model's prediction. Per Whitehill et al. (2014),
   report **Cohen's kappa first** — human agreement is the ceiling; if two
   people cannot agree on a crop, the model cannot be faulted for missing a
   signal that is not reliably readable at that resolution. Then model accuracy
   against the agreed subset only, per bucket, with Wilson intervals.

   Then calibrate `ExpressionConfig.min_confidence`, currently 0.40 and
   documented in `backend/config.py` as "explicitly a starting point, not a
   calibrated threshold".

4. **read/write merge experiment** (literature review §7). `read` is the weak
   class at F1 50.9% with sub-50% precision, confused with `write` in both
   directions. This is a *named, documented phenomenon*, not bad data — the
   closest precedent (SCB-ST-Dataset4) gets writing to only 57.8% with a strong
   temporal model. Retrain with the classes merged into one `studying` class,
   **and** report the pre-merge confusion matrix so the share of `read`'s loss
   that is specifically write-confusion is visible. Keep both classes only if
   `write` is healthy standalone.

---

## Phase 3 — Sustaining (months 4–5)

Own the JSONL contract. Re-run the pipeline whenever B or C need fresh data.
Keep the evaluation harness honest as models change.

---

## Files you own

```
backend/detection.py      backend/face.py           backend/face_detect.py
backend/students.py       backend/tracking.py       backend/identity.py
backend/headpose.py       backend/expression.py     backend/behaviour.py
schema.json               tools/eval_*.py           tools/bench_*.py
tools/sweep_identity.py   tools/audit_identity.py   tools/calibrate_gaze.py
tools/*_expression_labels.py                        tools/train_behaviour.py
docs/IDENTITY_AUDIT.md    docs/IDENTITY_GROUND_TRUTH.md
```

Config: only the `Detection`, `Face`, `HeadPose`, `Expression`, `Behaviour`,
`Identity` and `Tracking` dataclasses in `backend/config.py`.

Dataset: SCB-Dataset (per-student behaviour labels).

---

## Things already learned here — do not rediscover them

- Padding a face crop for "safety margin" makes detection **worse**
  (`pad=0.15` → 1/5 frames; `pad=0.0` → 5/5). Shipped with zero padding.
- MediaPipe's own face detector as a pre-detector performs **worse** than the
  current path (25% vs 42%). Rejected, measured, do not re-attempt.
- A bigger expression model (`enet_b2_8`) was **worse**: median confidence
  0.271, 98% under 0.5. Rejected.
- Clamping `imgsz` to native resolution was implemented and then **rejected on
  measurement** — it cost 331 → 263 persons. The 800px-portrait regression is
  pinned by a test as a known limitation, not a defect.
- Positional variance cannot separate wall posters from students, because this
  camera pans. **Appearance invariance** does (posters 0.906/0.909 vs students
  0.311–0.817).
- Gaze labels are meaningless on an off-centre camera until
  `HeadPoseConfig.yaw_reference_deg` is calibrated per camera. Two cameras
  measured so far: +37.4° and +35.5°.
