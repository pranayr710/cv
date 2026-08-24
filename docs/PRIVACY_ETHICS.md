# Privacy & Ethics

*Owner: Person C (per docs/WORK_PERSON_C.md, month-3 task 4). This section is
written to be lifted into the final project report nearly verbatim; it cites
repo mechanisms and tests by name so every claim here is checkable rather
than aspirational.*

---

## 1. What this system classifies — and why the framing is load-bearing, not cosmetic

ClassGraph's outputs are, in order of the pipeline: tracked person boxes with
session-scoped ids; facial **expression** categories (happy / sad / neutral);
body-posture behaviour categories; head-pose gaze direction; and a derived
**behavioral proxy score** ("observed on-task indicator") computed from a
hand-authored precedence rule over those machine labels.

None of these outputs is an emotion measurement or an inference of internal
state, and the distinction is enforced at three levels:

* **Terminology.** The expression feature is named "facial expression
  classification" throughout code, docs, deck and report — never "emotion
  detection." The standing caveat, quoted verbatim from
  `docs/LITERATURE_REVIEW.md` §4 and attached wherever the feature is
  described:

  > "This system classifies observable facial configurations into
  > happy/sad/neutral categories using a model trained on posed, frontal
  > datasets (AffectNet). Per Barrett et al. (2019), facial configuration is
  > not a reliable indicator of internal emotional state; this output is a
  > low-confidence behavioral proxy, not an emotion measurement, and has not
  > yet been validated against classroom footage at the resolutions/angles
  > this system operates on."

* **The engagement metric carries its own caveat in its own output.**
  `backend.engagement.BEHAVIORAL_PROXY_CAVEAT` is embedded in every summary
  dict `summarise_engagement()` emits, so any report built on it inherits the
  label automatically:

  > "Behavioral proxy score, not a concentration measurement: this figure
  > derives entirely from a hand-authored precedence rule (off-task behaviour
  > overrides on-task gaze) over two machine labels -- head-pose gaze class
  > and behaviour classification. It has not been validated against attention,
  > comprehension, or outcome data."

* **Regulatory framing.** EU AI Act **Article 5(1)(f)** (in force Feb 2025)
  prohibits *emotion inference* in education, citing "limited reliability,
  lack of specificity, limited generalisability" (`docs/LITERATURE_REVIEW.md`
  §4, §7). This project is not deployed in the EU and the Act is not binding
  on us; we adopt its framing anyway as a design floor, because its stated
  technical reasoning is the same critique our own literature review reached
  independently. Concretely: no module infers emotion, and none may — a
  future PR adding inferred-state output would contradict this document and
  should be rejected in review.

## 2. Legal context for deployment (India): DPDP Act 2023 child-data provisions

Classroom video in India falls under the **Digital Personal Data Protection
Act, 2023**, which treats children's data as high-scrutiny: verifiable
guardian consent for processing a child's personal data, and purpose
limitation — data collected for one stated purpose may not be repurposed
(`docs/LITERATURE_REVIEW.md` §7).

Concrete policy adopted for this project:

| Data | Retention | Deletion |
|---|---|---|
| Raw video | Processed locally; retained only as long as the recording device/user keeps it. The system writes no copy of raw video anywhere. | User-controlled; nothing to delete server-side because nothing is uploaded. |
| Raw face crops | **Not persisted.** Expression labels + confidence + session-scoped id only, per the data-minimisation recommendation in `LITERATURE_REVIEW.md` §4 recommendation (3). | Never stored ⇒ never to delete. |
| Derived JSONL records | Live only in the operator's working directory (`outputs/`). No cloud, no database, no sync. | Deleting the file deletes the data. Recommended practice stated to users: discard within the term of the consent given. |
| Rendered demo videos | Contain re-identifiable faces; treated with the same sensitivity as raw video. | Same as raw video. |

Purpose limitation is explicit: outputs exist to give the teacher same-day,
aggregate feedback on classroom activity. They must not be repurposed for
individual evaluation, discipline, ranking, or shared outside the classroom
staff — see access control below.

## 3. Anonymisation guarantees — what holds today, checked how

Three guarantees are claimed, each mapped to the mechanism and evidence that
backs it:

1. **Session-scoped identities.** Track ids restart per run and carry no
   link to any person registry. Evidence: `tests/test_tracking.py::
   test_reset_restarts_id_numbering`, and
   `tests/test_integrate.py::test_two_videos_never_share_track_identity`
   (two independent sessions both confirm their first person as id 1, and a
   fresh tracker is built per `process_video` call).
2. **No external matching.** Nothing leaves the machine: no embeddings are
   compared against external galleries, no names are ever inputs, and there
   is no network egress in the pipeline.
3. **No persistence across sessions.** Each `process_video()` run starts
   from an empty tracker state.

**Honest boundary (stated because an ethics section that overclaims is worse
than one that admits limits):** reuse of a single `PersonTracker` across two
videos *without calling `.reset()`* leaks identity state across recordings.
This exact failure mode is documented and pinned by
`tests/test_integrate.py::test_reusing_one_tracker_without_reset_does_leak_identity`
— the contract is that callers injecting a shared tracker must reset it. It
is a caller-responsibility guardrail, not an architectural impossibility.

**Against precedent:** the Virginia Tech classroom-analytics pilot
(arXiv:2604.03401) made FERPA compliance hinge on discarding raw video
immediately after feature extraction, keeping only derived pose/JSON data
(`LITERATURE_REVIEW.md` §7). We already match the stronger half of that postu­re
for crops (never persisted) but not yet the whole of it for video itself:
raw video currently survives as long as the operator's input file does.

**Decision on project-wide raw-crop/video discarding:** face crops are never
persisted (adopted). Full raw-video auto-deletion after processing is
recommended as the default mode for the final build (a `--discard-source`
flag on `backend/integrate.process_video`) but is **not implemented yet**;
it is listed as open work, not claimed as done.

## 4. Access control & consent

* **Access control now:** reports are generated to local files; there is no
  multi-user surface to harden. The intended consumer model is teacher-only:
  class-level dashboards first (`tools/dashboard.py` defaults to aggregated,
  class-level trends), individual drill-down only behind an explicit
  `--per-student <id>` request — a deliberate act, not a browsing default,
  so individual views are countable and attributable.
* **Guardian consent / opt-out:** **not yet implemented** — recommended
  addition before any real-classroom use: a plain-language notice to
  guardians describing what is computed (posture, gaze direction, expression
  category, observed on-task indicator), what is never done (no identification,
  no emotion inference, no records beyond the local session), and a simple
  opt-out whose implementation is seating/exclusion from capture, since the
  pipeline cannot retroactively exclude a student from a group recording.
* **Abstention as an ethical property:** where evidence is insufficient the
  system says "unknown" (gaze verdicts, behavioural proxy percentages,
  group-level decisions in `backend/group_activity.py`) rather than guessing.
  Reports show gaps instead of smoothing over them
  (`backend/reporting.class_trends` retains empty buckets). A system that
  visibly abstains cannot quietly manufacture confidence about a child.

---

### Status checklist (kept current by Person C)

| Item | Status |
|---|---|
| Expression feature renamed + standing caveat | ✅ wording live in owned files; pending-owner patches in `docs/RENAME_HANDOFF.md` |
| Behavioral-proxy caveat embedded in output | ✅ `BEHAVIORAL_PROXY_CAVEAT` flows through profiles/reports |
| DPDP retention/deletion policy stated | ✅ table above |
| Face crops non-persistence | ✅ recommended & followed in tooling; enforce-by-default still open |
| Raw-video auto-discard flag | ❌ open work (final-build item) |
| Guardian consent/opt-out flow | ❌ recommended, not built |
| Teacher-only access model | ✅ structural (class-level default; drill-down explicit) |

*Numbers standard note:* this document contains no measured performance
numbers, so `CHALLENGES_AND_SOLUTIONS.md` numbering rules are not triggered;
any future measured claim added here must follow that standard.
