# Reconciling the output against a known headcount

The clip `dataset/23-08/vedio/WhatsApp Video 2026-08-23 at 10.16.09.mp4` contains,
confirmed by the person who recorded it:

> **7 students and 1 teacher — 8 people total.**

This is the first identity ground truth this project has had. Every earlier
identity claim ("45 raw tracks → 10 person ids") was an *improvement* number
with nothing to check it against. With a headcount, the output becomes gradeable.

## What the pipeline reported

18 `person_id`s, of which 11 survived the poster and transient filters and were
reported as students. **Against 7 actual students, that is a 57% over-count.**

## Mapping every id onto a real person

Two objective structural facts do most of the work, neither needing ground truth:

* **Co-occurrence.** Two ids seen in the *same frame* are certainly different
  people. This is proof, not similarity.
* **Duplication.** An id appearing *twice in one frame* is certainly an error —
  one person cannot occupy two boxes.

| id | sightings | verdict against ground truth |
|---|---|---|
| 1 | 112 | ✅ real student — but **duplicated in 28 frames** |
| 2 | 70 | ❌ **merge of 3 people** (boy in cyan + a woman + a girl) |
| 3 | 75 | ✅ real student |
| 4 | 75 | ✅ real student |
| 5 | 84 | ✅ real student — **duplicated in 7 frames** |
| 6 | 18 | ⚠️ back-of-head; **candidate split with id 7** (they never co-occur) |
| 7 | 6 | ⚠️ back-of-head; same person as id 6 on visual evidence |
| 8 | 125 | ❌ **the teacher**, profiled as a student — **duplicated in 21 frames** |
| 10 | 27 | ❌ wall poster (now rejected automatically) |
| 11 | 20 | ❌ wall poster (now rejected automatically) |
| 9, 12, −1…−6 | 1–8 | ❌ transient noise (now rejected by the frame minimum) |

## The four distinct failures, in order of severity

### 1. Duplicate ids — 56 of 331 frames (17%)

`id 1` in 28 frames, `id 8` in 21, `id 5` in 7. This directly violates the
stated requirement that no two people ever share an id.

**Root cause, confirmed in code, not guessed:** `TwoPassIdentityResolver.finalise()`
matched each track against the gallery *independently*. Nothing anywhere
enforced that two tracks alive in the same frame must be different people. Two
students who look similar at 640×360 both cleared the 0.35 threshold against one
gallery entry, so both got that id.

**Fix:** the tracker already knows which tracks co-occur. `finalise()` now
forbids a track from taking any id already held by a track it co-occurs with.
The constraint is *hard* — skipped entirely, not penalised — because
co-occurrence is proof. This makes duplicates impossible by construction rather
than unlikely by tuning: resolution is sequential, so of any two co-occurring
tracks one is always already assigned and therefore excluded.

### 2. A 3-person merge — id 2

Worse in kind than a split: id 2's profile blends three people's expression and
concentration into one record that describes nobody. The co-occurrence
constraint should split some of this apart automatically. Whatever remains is a
`match_threshold` problem at this resolution.

### 3. 20.5% of detections get no id at all

164 of 801 person detections have `person_id: null`, and `build_profiles` skips
anyone without an id — so those people are silently absent from the output.
This is the faculty's original complaint ("not all students detected") in its
real form: they *are* detected, they just never get identified, and the count
then quietly omits them.

### 4. The teacher is counted as a student

`id 8` has the most sightings of any identity (125) and its contact sheet is
correctly one consistent person — the tracking is right, the *role* is wrong.
An earlier audit note called id 8 "one girl, consistent": that judged internal
consistency without asking *who* it was. Nothing in the pipeline distinguishes
instructor from student, so the teacher receives a concentration score and
skews every class-level aggregate.

## Why no metric caught any of this

Every number collected before this point measured *detection* (how many faces,
how many boxes) or *self-consistency* (how many tracks collapsed into how many
ids). None of them could see a duplicate, a merge, a poster, or a teacher,
because all four are perfectly consistent detections of the wrong thing. A
known headcount and a look at the actual frames found in one pass what months of
metrics did not.

---

# Re-audit after quality-gated identity creation (2026-08-25)

The verdict table above described the pre-clustering 18-id state. This section
replaces it with a fresh audit of the current pipeline, judged by eye from
contact sheets rather than from counts, as the Phase 1 gate requires.

## What was run, in order

1. `tools/sweep_identity.py` — first execution ever. **No `match_threshold`
   value reaches the ground truth**: best is 11 ids at 0.20, 13 at the shipped
   0.35, 17 at 0.60. Duplicate frames are 0 at every value and no-id is a flat
   6.3%, so the threshold is not the lever.
2. Duplicate-box hypothesis — **measured and rejected**. Over 1321 same-frame
   box pairs, no overlapping pair had matching faces (similarity median 0.041,
   max 0.307, all under the 0.35 floor). Overlapping boxes on this footage are
   adjacent students, not one person detected twice.
3. Quality-gated identity creation (`IdentityConfig.quality_gated_creation`) —
   the untried half of `docs/LITERATURE_REVIEW.md` §2. This is the lever that
   moved the number.
4. Poster rejection (`tools/reject_static_faces.py`) and the visual audit.

## Verdict table — 10 surviving ids, judged by eye

Ground truth: **7 students + 1 teacher = 8 people.**

| id | frames | verdict from the contact sheet |
|---|---|---|
| 1 | 80 | ✅ pure — one girl, consistent across all 12 crops |
| 2 | 86 | ✅ pure — one girl |
| 3 | 74 | ❌ **merge** — 9 crops of a boy in cyan, then 3 of a girl |
| 4 | 77 | ✅ pure — one girl |
| 5 | 106 | ❌ **merge** — one girl, plus 2 crops of a woman in red |
| 6 | 68 | ✅ pure — one girl |
| 7 | 59 | ⚠️ pure but **likely the teacher**, not a student |
| 8 | 46 | ⚠️ consistent, but back-of-head crops — identity unjudgeable by eye |
| 9 | 55 | ✅ pure — one girl |
| 10 | 10 | ⚠️ thin — 10 crops, plausibly a split of another id |

Rejected automatically and confirmed correct by eye:

* Two **wall posters** — a bridal portrait (appearance invariance 0.928) and a
  man's portrait (0.935), each identical in every crop. One of them was also
  splitting across a second id.
* Eight **transient** ids of 1–2 frames.

## Gate status

| condition | result |
|---|---|
| ids ≤ 10 for 8 real people | ✅ **10** |
| duplicate frames stays 0 | ✅ **0** |
| no-id ≤ 7% | ❌ **17.2%** (7.2% before gating) |
| confirmed by contact sheets | ✅ done — this table |

**Two of the surviving errors are merges, not splits.** That matters: the
count being close to right partly reflects merges cancelling splits, so 10 is
not 10 correct people. A merge blends two students' expression and attention
into one record describing nobody, and is the worse failure of the two.

## The tension the gate cannot resolve by itself

The three conditions pull against each other under this lever. The extra
identities *are* low-quality faces. Either they found an id — and the roster
over-counts — or they do not, and no-id rises. Measured:

| founding floor | ids | no-id % |
|---|---|---|
| none (gate off) | 13 | 7.1 |
| 14 px | 12 | 8.2 |
| 20 px | 11 | 13.9 |
| **24 px** | **10** | 14.4 |
| 32 px | 10 | 14.6 |

No setting satisfies both `ids ≤ 10` and `no-id ≤ 7%`. Meeting both requires
those faces to be *better*, not classified differently — which is Phase 2's
SAHI tiling for the back-of-room region, not a Phase 1 threshold. Phase 1 has
taken identity as far as this lever goes.

## Gaze was uncalibrated, and it mattered more than expected

`HeadPoseConfig.yaw_reference_deg` was still 0.0. `tools/calibrate_gaze.py`
measured **+44.1°** for this camera — in line with the +37.4° and +35.5°
measured on the two others, so an off-centre mount is the norm here, not the
exception. Before calibration 73.9% of faces read "right" and *every* student's
concentration was 100% `unknown`. After, the split is teacher 48.7% / left
28.5% / right 14.2% / down 8.6%, and concentration becomes a real number for
all 10 students — one of them at 12%, from 27 frames looking down.

This did not change identity, but it means every attention figure recorded
before this date was measuring the camera angle rather than the students.
