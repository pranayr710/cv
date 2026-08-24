# ClassGraph Phase 1 — Literature Review

Seven scoped research passes, each aimed at one concrete open problem in this
project, not the field in general. Every question was framed with our actual
measured numbers (640×360 panning camera, 12.6% of boxes below ByteTrack's
association floor, id 2's 3-person merge, teacher id 8 landing inside the
student range on every geometric signal, read F1 50.9%) so the answers are
directly actionable rather than a generic reading list.

Format per problem: **what we do now → what the literature establishes →
concrete recommendation → cost to implement**.

---

## 1. Small/low-resolution face detection

**Now:** SCRFD (buffalo_l), whole-frame, 1600×1600 input.

**Literature:** SCRFD (Guo et al., ICLR 2022) is still the right choice for
this compute/accuracy point — it specifically wins the tiny-face/low-compute
corner of the tradeoff curve against RetinaFace and DSFD, which is where a
laptop GPU on a 640×360 feed sits. No clean drop-in replacement beats it
without giving up the landmark pipeline we already depend on. But WIDER
FACE — the benchmark SCRFD is tuned against — is unconstrained handheld
photos; our fixed wide-shot classroom geometry is closer to crowd/aerial
detection (CrowdHuman, SAHI's own VisDrone validation) than to WIDER FACE's
shot composition, so its published hard-AP numbers don't transfer 1:1 to our
footage. **SAHI (Slicing Aided Hyper Inference)** — slice the frame,
detect per-slice at native resolution, merge with NMS — is a pure
inference-time wrapper (no retraining) reporting +5–15% AP on exactly this
"small object in a wide low-res frame" failure mode, and Ultralytics ships
native tiled-inference support already.

**Recommendation:** Keep SCRFD. Add SAHI-style tiling, applied only to the
back-of-room region (found by comparing YOLO person-box sizes across the
frame — no need to tile the whole image). Skip pixel-level super-resolution
pre-processing: the literature flags a domain-gap risk (a detector trained on
real pixels performing worse on SR artifacts), and the methods with real
evidence (EfficientSRFace) require joint retraining we don't have time for.
**Critically:** build a small, size-bucketed eval set from our own footage
(a few dozen frames, faces bucketed by pixel size) before/after any change —
no benchmark here validates on our specific fixed-camera geometry, and an
aggregate recall number would hide exactly the back-row failure we're trying
to fix.

**Cost:** Days. Tiling is a wrapper around an existing call.

---

## 2. Face re-identification threshold (id 2's 3-person merge)

**Now:** A single fixed cosine threshold (0.35) against ArcFace embeddings,
greedy match against a growing gallery, borrowed as a generic ballpark — not
calibrated to our population, because there's no labelled genuine/impostor
pair set to calibrate against.

**Literature — this is the most consequential finding of the whole review.**
A fixed global threshold is a *pairwise verification* rule; our task is
*clustering within one closed population* (assign each track to one of N
students in a video), and these are different problems with different
failure modes. The specific mechanism behind our merge is **transitivity
error**: if A~B clears threshold and B~C clears threshold, A gets merged with
C even though A and C never would have matched directly — a structural flaw
no better threshold number fixes. Wu et al. (CVPR 2013), *"Constrained
Clustering and Its Application to Face Clustering in Videos,"* formalizes
exactly our co-occurrence signal as a **cannot-link constraint**: two faces
in the same frame can never be merged, enforced as a hard constraint inside
clustering rather than a pairwise threshold check. An EDM 2024 paper doing
classroom-specific person re-id independently confirms clustering
outperforms verification-style matching in exactly our regime: one video, a
small closed population, high pose/quality variance. Separately, MagFace/
AdaFace show low-quality (small/off-angle) embeddings cluster with higher
variance near the origin — no single threshold is simultaneously safe for
high- and low-quality faces in the same video, which is a second, independent
reason a fixed number can't work here. On sample size: responsible EER/ROC
threshold calibration wants hundreds of labelled pairs; we have a handful of
manually-audited identities from one video, nowhere near enough to
responsibly tune a number even if the transitivity problem didn't exist.

**Recommendation:** This directly obsoletes the "sweep match_threshold"
step from my prior plan. Instead: **switch matching from greedy
threshold-and-merge to constrained agglomerative clustering** over track-level
mean/EMA embeddings, with the co-occurrence table we already compute
(`TwoPassIdentityResolver._conflicts`) enforced as a hard cannot-link
constraint during clustering, not just at final assignment as we do today.
Additionally: gate *identity creation* (not just detection) by a quality
score — face size, blur, pose angle — so a low-quality face can only join an
existing cluster, never found a new one. This is implementable in days
(scikit-learn `AgglomerativeClustering`, average linkage, cosine distance;
the cannot-link data already exists in our code) and requires no new labelled
data.

**Cost:** Days. Reuses existing co-occurrence data; swaps the matching
algorithm inside `identity.py`, not the surrounding pipeline.

---

## 3. Tracking under camera motion (the 20.5% missing-ID problem)

**Now:** ByteTrack, 1 fps sampling, panning camera. 12.6% of boxes fall below
ByteTrack's IoU association floor between consecutive processed frames and
never get a `track_id`. We built a face-embedding surrogate-key fallback to
recover them (this session, tested only in unit tests so far).

**Literature confirms this is not a tuning bug** — it's IoU-based association
operating outside its designed envelope. Every tracker built specifically for
low-frame-rate conditions (APPTracker+, ACM MM 2022/IJCV 2024; UCMCTrack, AAAI
2024) converges on the same answer: **as frame rate drops, weight appearance
over motion, don't try to fix motion association alone.** APPTracker+'s own
ablation on MOT17 downsampled to 1/10 frame rate shows appearance-forward
fusion winning; UCMCTrack is built explicitly for "low-frame-rate scenarios
where many trackers struggle due to large object motion." Neither is
Ultralytics-integrated, so porting either is a real integration cost. BoT-SORT
with Global Motion Compensation (GMC) directly targets our panning-camera
symptom by estimating camera ego-motion and warping predicted boxes before
IoU matching — but it was validated at normal frame rates on small
inter-frame gaps, not the 1-second, multi-meter gaps we have, so it should
help, not fully close the gap. **No classroom/lecture-capture-specific MOT
paper exists** — the closest hits are all commercial PTZ auto-framing
products, not published identity-tracking research.

**Recommendation:** Two actions, not one. (1) **Swap `bytetrack.yaml` →
`botsort.yaml`** in the existing Ultralytics tracker config, with
`with_reid: True` — this is a config change, hours not days, and should
recover some fraction of the 12.6% before it ever reaches our fallback. (2)
**Keep the face-embedding surrogate-key fallback as the primary mechanism, not
a patch** — the literature's own conclusion is that appearance-first identity
is structurally necessary at this frame rate, so this fallback isn't a
workaround, it's the field-recommended answer. Don't attempt to port
UCMCTrack/APPTracker+ — non-trivial integration for marginal expected gain
this close to the deadline.

**Cost:** Hours for the BoT-SORT config swap; the fallback is already
written, just needs its real-footage confirmation run (still pending from
before this research pass).

---

## 4. Facial expression / "emotion" detection

**Now:** HSEmotion/EmotiEffLib (EfficientNet-B0, AffectNet 8-class),
collapsed to happy/sad/neutral, applied to small (≥25px) off-angle classroom
crops. Never validated against real classroom faces with human ground truth.
Faculty explicitly asked for this feature.

**Literature gives a precise, actionable split between the scientific
critique and the engineering task.** Barrett et al. (2019, meta-analytic
review) found facial configuration maps onto its purported emotion only
~20-30% above chance across contexts — a smile signals submission, politeness
or discomfort as often as happiness — and its own recommended fix is
terminology: use "facial configuration" or "expression," never "emotion,"
since the latter claims an inference the data doesn't support. Separately,
this is not just a philosophical point: HSEmotion's own team (ABAW 2024 paper)
found their larger, higher-AffectNet-accuracy models generalize *worse*
cross-dataset — training accuracy is a poor proxy for reliability on footage
this different from the training distribution. Occlusion/off-angle studies
report 57-61% relative accuracy drops under conditions similar to ours. The
EU AI Act (Article 5(1)(f), in force since Feb 2025) prohibits emotion
inference specifically in education, citing "limited reliability, lack of
specificity, limited generalisability" — not binding here, but its stated
technical reasoning is the same critique, elevated to law, and worth citing
defensively regardless of jurisdiction. Whitehill et al. (2014) supplies the
right validation template: report human inter-rater agreement *first*
(it sets the ceiling — the model can't beat what two humans agree on), then
model accuracy against that.

**Recommendation:** (1) **Rename the feature** from "emotion detection" to
"facial expression classification" everywhere — code, docs, faculty-facing
report — with a standing caveat sentence (drafted, ready to use): *"This
system classifies observable facial configurations into happy/sad/neutral
categories using a model trained on posed, frontal datasets (AffectNet). Per
Barrett et al. (2019), facial configuration is not a reliable indicator of
internal emotional state; this output is a low-confidence behavioral proxy,
not an emotion measurement, and has not yet been validated against classroom
footage at the resolutions/angles this system operates on."* (2) **Run the
minimal validation protocol already scoped by the research**: ~90-150 face
crops actually produced by our pipeline, stratified across resolution/angle
bins, labelled independently by two people (happy/sad/neutral/unlabelable),
Cohen's kappa computed *before* touching the model (if humans don't agree, the
3-class signal isn't cleanly readable at this resolution regardless of the
classifier), then model accuracy against the agreed labels, reported with a
confidence interval given the small N — 2-3 days total. (3) Don't persist raw
face crops after the label is extracted — label + confidence + anonymous id
only, per the data-minimization point in problem 7 below.

**Cost:** Renaming and caveat text: minutes. Validation protocol: 2-3 days,
needs a second labeller (you + one other person).

---

## 5. Concentration / engagement scoring

**Now:** A hand-authored precedence rule (off-task behaviour overrides
on-task gaze) fusing head-pose gaze label and behaviour classifier label into
a per-frame on/off/unknown verdict, then a per-student percentage. No
external validation of any kind.

**Literature is unanimous on one point:** engagement/concentration is a
multi-dimensional psychological construct, not something directly observable
in a frame; every rigorous paper treats a CV-derived score as a *proxy
requiring external validation*, never as ground truth standing alone. The
closest usable ground-truth instrument is **BOSS (Behavioral Observation of
Students in Schools)** — momentary time-sampling coding on-task / off-task
motor / off-task verbal / off-task passive, with real convergent-validity
evidence (used to discriminate ADHD in clinical research). Its structure —
off-task behaviour coded independently of orientation — is a close match to
our own precedence rule (behaviour overrides gaze), which gives our design a
citable rationale rather than looking ad hoc. Kendon's F-formations support
treating gaze-toward-teacher as *a* joint-attention signal, but the gaze
literature is explicit that gaze is socially ambiguous (peer-gaze, notebook
-gaze, normal aversion during thinking) — no paper treats it as sufficient
alone. Public benchmarks (DAiSEE, EmotiW/EngageWild) report modest accuracy
(51-78% on 4-level engagement) using human-annotated video as their own
"ground truth" — even the benchmark literature doesn't have an
independently-validated outcome measure, just observer judgment, same as
ours would be. One sharper warning: Bosch et al. found students "pretending
to engage" ~23% of class time were rated as engaged by outside human
observers — a CV system reading gaze+behaviour alone inherits this blind
spot, likely worse.

**Recommendation:** (1) **Rename/qualify** the metric as a "behavioral proxy
score" or "observed on-task indicator," not "concentration," with a caveat
sentence stating it derives entirely from a hand-authored precedence rule and
has not been validated against attention, comprehension, or outcome data. (2)
**Cite BOSS explicitly** as the design's theoretical basis — this converts an
ad hoc rule into a documented adaptation of an established instrument. (3) If
any time exists: the cheapest real validation is having one other person
watch a 10-20 minute segment and apply simplified BOSS coding at fixed
intervals for a handful of students, blind to the system's output, then
report agreement (Cohen's kappa or percent agreement) — half a day. Even a
single teacher's post-hoc holistic rating per student, correlated informally
against our aggregate score, is a real, citable minimal validation (under an
hour).

**Cost:** Wording changes: minutes. Minimal validation: half a day if a
second observer is available.

---

## 6. Teacher vs. student role separation

**Now:** Four geometric signals measured against the known teacher and known
students — all four failed (teacher fell inside the student range on every
one). Currently: `ProfileConfig.instructor_ids`, declared manually per video.

**Literature confirms this was the right call, not a shortcut.** This is a
genuinely underexplored problem — no published system solves teacher/student
discrimination from passive, uninstrumented, single-camera RGB geometry
alone. The two approaches that actually work in the field are: (a) a
**calibrated teaching-zone ROI**, computed once per room/camera install (the
"teacher set" concept — front-of-room region, reused across sessions from the
same fixed camera), or (b) **audio** — one paper matching our exact question
("Multimodal Classroom Diarization: Teacher or Student?") reports 99.3%
accuracy on this binary role question using speaking-time/diarization, far
stronger than any geometric proxy. Every commercial lecture-capture/PTZ
product solves this via a worn mic/RF tag or a hard-coded zone, never via
passive body-geometry inference. Papers that attempt fully automatic,
calibration-free, audio-free role separation from generic footage explicitly
acknowledge the same robustness problems our four-signal experiment found
(panning cameras, non-stereotyped teacher movement, low resolution) — this
isn't a gap in our engineering, it's an accepted open problem in the field.

**Recommendation:** Keep manual declaration — it is the literature's own
practical answer here, not a workaround pending a better fix. Two upgrades
worth considering only if applicable: if this camera/room setup is fixed
across multiple videos, a one-time calibrated "teaching zone" ROI could
auto-nominate the teacher per video instead of a per-video manual declaration
(setup cost once, not per video). If audio ever becomes available, speaking-
time/diarization is the one signal the literature shows reliably solving this
— worth flagging as a genuine future-work item, not aspirational hand-waving.

**Cost:** Zero — this confirms the existing design. No code change needed.

---

## 7. Behaviour classification (read/write merge) and privacy framework

**Behaviour — read/write confusability.** Our measured weakness (read F1
50.9%, confused with write-like postures) is a **named, documented phenomenon
in the literature, not a symptom of our data being bad.** The closest
precedent (SCB-ST-Dataset4, arXiv:2310.16267) using a strong temporal model
(SlowFast) still gets writing to only 57.8% — nearly identical to our gap —
and explicitly measures high read/write similarity via a "Behavior Similarity
Index." The broader action-recognition literature attributes this to a
structural cause, not a fixable modeling choice: reading and writing differ
mainly in *what object is in the hand* (book/pen), not in gross body pose or
motion — so pose-based or single-frame classifiers are close to their ceiling
on this distinction regardless of how much more data or temporal context is
added. No paper in this search claims either single-frame or temporal
modeling fully resolves it; the ceiling for the weaker class sits around
55-60% even with strong video models.

**Recommendation:** Merge read+write into a single "studying" class as the
primary reported metric — the literature supports this as the correct
response to a known structural confusion, not a workaround. Before
committing, validate empirically: retrain/evaluate with the merge, and
separately report a pre-merge confusion matrix to confirm how much of read's
F1 loss is specifically write-confusion (using the same BSI-style approach as
the cited precedent) versus confusion with other classes. If write's own F1
turns out to be healthy standalone, keeping both is only defensible with an
added hand-object cue (a phone/pen/book detector or hand-region crop) — not
by tuning posture-only features further.

**Privacy and regulatory framework.** Two provisions matter beyond what the
expression-detection research (problem 4) already surfaced: **EU AI Act
Article 5(1)(f)** prohibits emotion inference in education outright (fines up
to €35M/7% global turnover) — directly relevant to how we frame and disclose
the expression feature, regardless of deployment jurisdiction. **India's DPDP
Act 2023** treats children's data (which classroom video is) as high-scrutiny,
requiring purpose limitation and guardian consent considerations even though
our IDs are anonymous and non-persistent — the video itself carries
re-identifiability risk independent of what we compute from it. A directly
comparable precedent (Virginia Tech classroom-analytics pilot,
arXiv:2604.03401) goes further than we currently do: it discards raw video
immediately after feature extraction and keeps only derived pose/JSON data,
citing this explicitly as its FERPA-compliance mechanism.

**Recommendation:** Add an explicit privacy/ethics section to the project
write-up: (1) frame outputs as behaviour/posture classification, not emotion
inference, citing the EU AI Act provision as the reason for that framing even
though not EU-deployed; (2) cite DPDP Act 2023 child-data provisions and state
a concrete retention/deletion policy for raw video and derived records; (3)
document the anonymization guarantee already in place (session-scoped IDs,
no external database matching, no persistence) explicitly against the
Virginia Tech precedent, and consider whether discarding raw face crops after
feature extraction (already recommended in problem 4) is worth adopting
project-wide, not just for expression; (4) state access control (teacher-only)
and note guardian consent/opt-out as a recommended addition even if not yet
implemented.

**Cost:** Behaviour merge: a day or two (retrain + evaluate). Privacy
write-up: half a day, no code changes required.

---

## What changes because of this review

Three things I was about to do differently, corrected by this pass:

1. **Id 2's merge** — I was going to sweep `match_threshold` for a better
   number. The literature shows this can't work (transitivity error is
   structural, not threshold-sensitive) and that the field's actual answer is
   constrained clustering with our co-occurrence data as a hard cannot-link
   rule. This is a different, larger change than I'd planned, but it's the
   correct one.
2. **Teacher/student separation** — I was treating manual declaration as a
   stopgap pending a better signal. The literature confirms there isn't one
   for this footage (no audio, no fixed room), so manual declaration is the
   destination, not a placeholder.
3. **Tracking under camera motion** — the face-embedding fallback I built
   this session, which I was treating as a workaround, turns out to be what
   the low-frame-rate MOT literature itself converges on as the correct
   primary mechanism. Confidence in keeping it is now higher; it's also worth
   the cheap addition of a BoT-SORT config swap alongside it.

Everything else (small-face tiling, expression renaming + validation,
concentration renaming + BOSS citation, read/write merge, privacy section) is
new work this review surfaced, not a correction to something already planned.
