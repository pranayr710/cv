# Rename Handoff — wording patches for file owners to apply

*From Person C, per the coordination rule in docs/WORK_PERSON_C.md month-3
task 3: "you write the wording, the file owner applies it in their own
files." Everything below is copy-pasteable. C has already applied the renames
in every C-owned file (backend/engagement.py, backend/config.py Engagement
section, backend/reporting.py, tools/render_video.py, tools/dashboard.py,
PART1_PLAN.md, readme.md).*

## Ground rules (so owners don't over- or under-apply)

* **Prose and docstrings**: replace "concentration" with *behavioral proxy
  score* / *observed on-task indicator*; replace any claim-style "emotion"
  with *facial expression*.
* **Serialized keys** (`concentration_pct`, `"concentration"` dict key,
  `idx_to_emotion_class`): **keep for now** — additive-only discipline. New
  canonical names exist alongside (`behavioral_proxy_pct`; see
  backend/engagement.py module docstring). A one-time key migration is a
  separate coordinated change, NOT part of this handoff.
* **Library/API names** (`HSEmotion`, `EmotiEffLib`), literature references
  ("Barrett et al.", EU AI Act emotion-inference discussion), and mentions of
  OTHER systems' emotion monitoring (build_ppt.py's China cases) are
  **correct as-is — do not change**.

## Patch list

### `backend/student_profile.py` (owner: pipeline owner)

| Line | Before | After |
|---|---|---|
| 7 | `breakdown, and a single concentration percentage.` | `breakdown, and a single observed on-task percentage (behavioral proxy score).` |
| 13 | `expression labels, behaviour labels, gaze-derived concentration. It does not` | `expression labels, behaviour labels, a gaze-derived behavioral proxy score. It does not` |
| 32 | ``will show 100% concentration`` | ``will show a 100% on-task proxy score`` |
| 245 | `# 100% concentration_pct is an absence-of-evidence artifact, not a` | `# A 100% behavioral_proxy_pct is an absence-of-evidence artifact, not a` |
| 261 | `"detected at all -- concentration_pct reflects gaze only and "` | `"detected at all -- the score reflects gaze only and "` |
| 326 | `"(concentration) from a finished Stage 1+2 JSONL run."` | `"(behavioral proxy summary) from a finished Stage 1+2 JSONL run."` |

Optional additive improvement (recommended, non-breaking): where line 238
assigns `concentration = summarise_engagement(...)`, also emit the same dict
under profile key `"behavioral_proxy"` alongside the existing
`"concentration"` key at line ~313, then migrate readers (render_video.py,
backend/reporting.py already prefer the new name when present).

**⚠ Applied by C during this handoff (owner review requested):** line ~245's
conditional `caveat` assignment previously OVERWROTE the standing caveat that
`summarise_engagement` now always attaches. It now PREPENDS its
absence-of-evidence warning onto the standing text instead (both honesty
layers survive). One pinned test was updated to match:
`tests/test_student_profile.py::test_100pct_with_behaviour_readings_present_
is_not_flagged` now asserts the caveat is exactly the standing one (nothing
prepended) plus `off_task_detectable is True`. If you own this file and
prefer a different composition order, change it — but keep both texts.

### `tests/test_engagement.py`

* Docstring line 9: `6. Aggregation: concentration_pct excludes unknown frames from the` →
  `6. Aggregation: behavioral_proxy_pct excludes unknown frames from the`
* Rename tests when next touched: `test_concentration_pct_excludes_unknown_frames`
  → `test_behavioral_proxy_pct_excludes_unknown_frames`;
  `test_concentration_pct_is_none_when_nothing_graded` →
  `test_behavioral_proxy_pct_is_none_when_nothing_graded`.
* Existing asserts on the legacy key may stay; please ADD:
  `assert summary["behavioral_proxy_pct"] == summary["concentration_pct"]`
  in one test, and `assert "caveat" in summary` in another (both keys are
  now emitted).

### `tests/test_student_profile.py`

* Line 13: `6. Concentration is computed via backend.engagement, not reinvented.` →
  `6. The behavioral proxy score comes via backend.engagement, not reinvented.`
* Test names containing `concentration` → rename to `behavioral_proxy_*`
  opportunistically; asserts on legacy keys may stay (add alias asserts as above).

### `docs/IDENTITY_GROUND_TRUTH.md`

* Line 63: `concentration into one record that describes nobody.` →
  `on-task scores into one record that describes nobody.`
* Line 81: `instructor from student, so the teacher receives a concentration score and` →
  `instructor from student, so the teacher receives an on-task proxy score and`

### `tools/apply_gaze_calibration.py`

* Line 12: `applied, and the "concentration" numbers in that profile were consequently` →
  `applied, and the on-task scores in that profile were consequently`

## Explicitly NO-ACTION files (checked, uses are legitimate)

* `backend/expression.py`, `tests/test_expression.py`,
  `tools/eval_expression_sanity.py`, `tools/prepare_expression_labels.py`,
  `requirements.txt` — library names (`HSEmotion`/`EmotiEffLib`) and
  defensive citations only. Also: do NOT touch expression.py during Person
  A's retrain window.
* `schema.json` — frozen contract; its only "emotion" mention is the
  defensive description ("not an inferred emotional state"). Correct as-is;
  renaming descriptions inside a frozen schema needs its own coordinated change.
* `docs/LITERATURE_REVIEW.md` — source document that mandated the renames;
  historical record, left verbatim (its §4 caveat sentence is now quoted in
  docs/PRIVACY_ETHICS.md).
* `docs/PROJECT_PLAN.md`, `CHALLENGES_AND_SOLUTIONS.md` — defensive/discussion
  mentions of why the words are banned.
* `tests/test_attention.py:194` — "gaze aversion during concentration"
  describes the psychological phenomenon in cited literature; referential,
  correct as-is.
* `build_ppt.py` — all remaining "emotion"/"concentration"-adjacent lines
  describe other systems' surveillance or the ban itself.
