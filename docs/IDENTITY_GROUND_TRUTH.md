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
