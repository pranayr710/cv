# Identity audit — visual verification of `person_id` correctness

Method: `tools/audit_identity.py` renders one contact sheet per `person_id`,
sampling face crops evenly across that id's whole lifetime, then each sheet is
judged by eye. Source: `outputs/wa_full_video.jsonl` (full 5.5-min WhatsApp
video, 331 frames at 1 fps).

This is a manual small-N audit, not an automated metric. It exists because
"45 raw tracks → 10 stable person ids" is an improvement claim with no evidence
of *correctness* behind it — 10 could be right, or it could be merges and
splits cancelling out. The audit distinguishes those.

## Verdict per id

| person_id | sightings | verdict | note |
|---|---|---|---|
| 1 | 109 | ✅ **pure** | one girl, distinctive earring, consistent across all crops |
| 2 | 69 | ❌ **MERGE** | **three different people**: boy in cyan shirt, a woman, a girl with earrings |
| 3 | 65 | ✅ pure | one girl, consistent |
| 4 | 66 | ✅ pure | one girl, consistent |
| 5 | 79 | ✅ pure | one girl, consistent |
| 6 | 18 | ⚠️ partial | mostly one student's head from behind/side; low-information crops, hard to fully confirm |
| 7 | 6 | — | too few crops to judge |
| 8 | 82 | ✅ pure | one girl, consistent |
| 9 | 1 | ❌ **noise** | single frame, span 0.0s |
| 10 | 27 | ❌ **NOT A PERSON** | a **printed face on a wall poster** — static, identical in every crop |
| 11 | 20 | ❌ **NOT A PERSON** | a second **wall poster / portrait** |
| 12 | 4 | — | too few crops to judge |
| −3, −4, −6 | 3, 1, 1 | ❌ noise | 1–3 frames each, never face-verified |

## Summary

- **Genuinely pure student ids: 5** (ids 1, 3, 4, 5, 8) — plus id 6 probably, on
  weak evidence.
- **1 merge**: id 2 fuses three different people into one identity.
- **2 non-people**: ids 10 and 11 are faces printed on wall posters, tracked and
  profiled as if they were students for 27 and 20 sightings respectively.
- **5 noise ids** of 1–3 frames.

So of the 10 "stable ids" previously reported, roughly **5–6 are real, correct
students**, 1 is a merge, 2 are posters, and the rest are transient noise.

## Why this matters more than the headline number

The 45 → 10 reduction was real, but reporting it alone overstated the result.
Two distinct problems were invisible to every metric collected so far:

1. **Posters are indistinguishable from students** to a face detector. Nothing
   in the pipeline asks "did this face ever move?" — a printed face is a
   perfectly good face. This inflates student counts and pollutes class-level
   aggregates with a permanently "attentive, neutral" phantom.
2. **A merge silently attributes several students' behaviour to one profile.**
   This is the worse failure mode: a split under-counts continuity but never
   mixes people's data, whereas id 2's profile is a blend of three people and is
   simply wrong.

## Fixes these findings justify

- **Static-face rejection**: a face whose box barely moves across its entire
  lifetime is a poster, not a student. Cheap to test (positional variance over
  the id's sightings) and directly targets ids 10 and 11.
- **Transient-id filtering** (already planned as Step 2): drops the 1–3 frame
  noise ids.
- **Merge reduction**: id 2's merge suggests `match_threshold` (0.35) is too
  permissive for low-resolution faces. Worth a sweep now that there is a
  concrete failing case to measure against.
