"""Central configuration for ClassGraph Stage 1 (Perception).

Every tunable knob lives here. No magic numbers in module code. Modify values
here and every module picks them up — this is the contract that lets three
teammates work in parallel without stepping on each other.

Usage:
    from backend.config import CONFIG
    detector = Detector(weights=CONFIG.detection.weights)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = REPO_ROOT / "data"
WEIGHTS_DIR: Path = REPO_ROOT / "weights"
OUTPUTS_DIR: Path = REPO_ROOT / "outputs"


# --------------------------------------------------------------------------- #
# Per-module configs
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DetectionConfig:
    """Person A — YOLOv11 detector settings."""

    weights: str = "yolo11m.pt"  # ultralytics auto-downloads if missing
    device: Literal["cuda", "cpu", "auto"] = "auto"

    # Confidence thresholds.
    # person_conf lowered from 0.40: distant back-row students score low, and
    # for engagement statistics a missed student costs more than a stray box.
    person_conf: float = 0.30
    # object_conf left at 0.35 deliberately. Raising it to 0.50 cuts "laptop"
    # detections from 19 to 6 across the sample set, but that is not a pure
    # false-positive win: img04 is a computer lab where the laptops are real,
    # while img01 is an ordinary classroom where they are not. One global
    # threshold cannot separate those cases — tune this only against labelled
    # ground truth.
    object_conf: float = 0.35

    # Per-class overrides of object_conf, as (class_name, threshold) pairs.
    # A tuple of pairs rather than a dict because this config is frozen and a
    # dict default would be mutable shared state.
    #
    # "book" is overridden because it is the weakest class we depend on and the
    # one a reviewer asked about directly. COCO's `book` class was trained
    # largely on bookshelves and stacked/closed books, not an open notebook seen
    # at a downward classroom angle -- a domain mismatch, so the default 0.35 is
    # miscalibrated for our footage rather than merely strict. Books detected
    # across the 13 dataset images (yolo11m, imgsz 1920):
    #    0.35 (shared default) ->  21
    #    0.30                  ->  28
    #    0.25 (chosen)         ->  40
    #    0.20                  ->  51
    #    0.15                  ->  77
    # 0.25 was chosen by *looking* at the boxes on img01, not by taking the
    # count: at 0.25 the detections are genuine open books and notebooks, while
    # below it false positives appear -- including one box drawn around a
    # student's head. This project has made the opposite mistake before (the
    # "phantom laptop" that turned out to be a real computer lab), so a
    # threshold is not moved here on a count alone.
    #
    # This buys recall for the "writing / on-task" signal in backend.attention,
    # where a missed book wrongly leaves a studying student in the ambiguous
    # bucket.
    #
    # A limitation the same inspection exposed, and the reason this is a
    # stopgap: several students visibly writing on **loose sheets of paper**
    # (img01 is an exam, not a textbook lesson) get no box at any threshold --
    # COCO's `book` class does not fire on loose paper. So the writing signal
    # will systematically under-report exactly in exam and worksheet settings.
    # Fine-tuning on SCB-Dataset's `write`/`read` labels detects the *behaviour*
    # rather than a proxy object, and is the real fix.
    object_conf_per_class: tuple[tuple[str, float], ...] = (("book", 0.25),)

    # NMS IoU
    iou: float = 0.50

    # COCO class names we care about (person auto-included).
    # "book" is kept deliberately: a student with a book and pen is *on-task*,
    # which is the opposite signal from a phone. Dropping it would leave the
    # pipeline unable to tell studying from distraction -- see
    # backend/attention.py's writing_object_classes.
    #
    # COCO classes worth naming in a classroom. YOLO detects all 80 either way;
    # the whitelist only decides which reach the output. Widened from three
    # because "what is this student doing" is largely answered by what they are
    # holding -- a bottle means drinking, a keyboard means typing, food means
    # eating, and none of those were expressible before.
    object_whitelist: tuple[str, ...] = (
        "cell phone", "laptop", "book", "bottle", "cup", "keyboard", "mouse",
        "backpack", "handbag", "sandwich", "apple", "banana", "donut", "pizza",
        "scissors", "remote", "tv",
    )

    # Inference resolution. YOLO resizes the frame to this before inference, so
    # it directly controls whether distant students survive. Raised from 960:
    # at 960 a back-row student ~60 px tall in a 1920-wide frame shrinks to
    # ~30 px and is lost.
    #
    # Re-swept across all 13 dataset images against model size, with SCRFD's
    # 434 detected faces as an independent reference for how many students are
    # actually present (persons found / ms per image on an RTX 4050):
    #
    #   imgsz   yolo11m      yolo11l      yolo11x
    #    1280   264 /  ~50   227 /   94   234 / 146
    #    1536   302 /   77   282 /   88   296 / 154
    #    1920   331 /  106   351 /  132   374 / 240
    #
    # Two findings: resolution matters far more than model capacity (11m@1920
    # beats 11x@1280 by 97 persons while running faster), and no configuration
    # reaches the 434 students the face detector finds -- which is why
    # backend/students.py exists. 1920 chosen: +67 persons over 1280 for +56 ms.
    # yolo11x@1920 finds 43 more still, at 2.3x the latency -- worth switching
    # to for offline batch runs, not for the live path.
    imgsz: int = 1920

    # Batch size when running on video frames
    batch_size: int = 1


@dataclass(frozen=True)
class FaceConfig:
    """Person B — face detection + MediaPipe Face Mesh landmark settings.

    Two detector backends, selected by :attr:`detector`:

    * ``"scrfd"`` (default) — InsightFace SCRFD finds face boxes on the
      **whole frame**; Face Mesh then supplies landmarks/EAR per box.
    * ``"mediapipe"`` — the original path: Face Mesh's own internal detector,
      run on each person crop. Kept so the two can be benchmarked against each
      other (``tools/bench_faces.py``), not because it is competitive.

    Measured on ``dataset/img01.jpg`` (1920x1088 classroom CCTV, ~50 students,
    30 persons found by YOLO):

    ==================  ==============
    Backend             Faces found
    ==================  ==============
    mediapipe (old)     10
    scrfd (default)     48
    ==================  ==============

    The gap is a model-fit problem, not a tuning one. MediaPipe's detector
    (BlazeFace) is built for short-range, selfie-distance faces. SCRFD was
    trained for the WIDER FACE "hard" split, where ~79% of faces are under
    32x32 px and ~52% under 16x16 px — i.e. exactly a classroom's back rows.
    """

    # Which face detector supplies the face boxes. See the class docstring for
    # the measurement behind this default.
    detector: Literal["scrfd", "mediapipe"] = "scrfd"

    # --- SCRFD (InsightFace) settings; ignored when detector="mediapipe" --- #

    # InsightFace model pack. "buffalo_l" bundles det_10g.onnx (SCRFD-10GF).
    # Auto-downloaded to ~/.insightface on first use, like YOLO's weights.
    scrfd_model_pack: str = "buffalo_l"

    # Inference resolution for SCRFD, same trade-off as DetectionConfig.imgsz:
    # small faces die when the frame is downscaled too far. Measured on img01:
    #    640  -> 37 faces
    #   1024  -> 42 faces
    #   1600  -> 48 faces   <- chosen
    #   2048  -> 48 faces   (no further gain, more compute)
    scrfd_det_size: tuple[int, int] = (1600, 1600)

    # Detection confidence floor. Kept at 0.30 to match DetectionConfig's
    # person_conf for the same reason: a missed student costs more than a
    # stray box in engagement statistics.
    scrfd_det_thresh: float = 0.30

    # Also load buffalo_l's bundled ArcFace recognition sub-model
    # (w600k_r50.onnx) and compute a 512-d embedding per face. Previously left
    # off deliberately -- "persistent face embeddings are exactly what this
    # project's identity design avoids" -- because the earlier identity design
    # only needed motion-based tracking to survive within a single video.
    #
    # That assumption changed: a student who is briefly fully occluded or turns
    # away and back needs to keep the SAME id, which position-based tracking
    # alone cannot guarantee (measured: 28 track ids for at most 9 concurrent
    # people on one real continuous clip -- see backend/identity.py). Enabling
    # this lets a reappearing face be matched back to its earlier id.
    #
    # The privacy property that motivated the original "off" default is
    # unchanged and still enforced: embeddings live only in
    # backend.identity.IdentityGallery, which is built fresh per video and
    # discarded after (see that module's docstring and its regression tests).
    # No embedding is ever written to output or persisted across videos.
    enable_recognition: bool = True

    # SCRFD returns a tight face box. Face Mesh needs a little context around
    # it to fit the mesh reliably, so the crop handed to Face Mesh is padded by
    # this fraction of the box size. Distinct from person_crop_padding below,
    # which pads a *person* box and was measured to be harmful.
    scrfd_landmark_padding: float = 0.25

    max_num_faces: int = 40  # cap per Face Mesh pass (a crop normally has 1)
    refine_landmarks: bool = True
    min_detection_confidence: float = 0.50
    min_tracking_confidence: float = 0.50

    # EAR (Eye Aspect Ratio) thresholds — used downstream for sleep detection.
    # Here for documentation; the face module only computes the raw value.
    ear_closed_threshold: float = 0.20
    ear_open_typical_range: tuple[float, float] = (0.20, 0.40)

    # MediaPipe eye landmark indices (refined 478-landmark map). These are
    # the 6-point EAR landmarks per eye (P1..P6 in the Soukupova & Cech paper).
    left_eye_ear_idx: tuple[int, ...] = (33, 160, 158, 133, 153, 144)
    right_eye_ear_idx: tuple[int, ...] = (362, 385, 387, 263, 373, 380)

    # Face Mesh runs each frame independently (no cross-frame tracking) since
    # analyze() receives one frame at a time. Set False only for a continuous
    # single-face stream where temporal tracking helps.
    static_image_mode: bool = True

    # Landmarks kept per face. MediaPipe returns 478 with refine_landmarks=True
    # (468 mesh + 10 iris); we keep the canonical 468 to match the frozen
    # schema. Iris points (indices 468-477) are dropped. Refinement still
    # improves the precision of the eye landmarks used for EAR.
    num_landmarks: int = 468

    # A detected face is bound to a person bbox only if at least this fraction
    # of the face's bounding box lies inside that person's bounding box.
    assign_min_containment: float = 0.50

    # Face Mesh runs on a per-person crop, not the whole frame. MediaPipe's
    # face detector downscales its input to a small fixed size, so a face that
    # is small *relative to the frame* is destroyed before detection runs.
    # Measured on real footage with the whole-frame approach:
    #   3840x2160 clip, 1 student  -> 0 faces  (per-person crops: 1/1)
    #   1920x1088 classroom CCTV   -> 0 faces  (per-person crops: 8/20)
    # Padding added around each person box before cropping, as a fraction of
    # the box size. Measured to be actively harmful — padding enlarges the crop,
    # which shrinks the face relative to it and reverses the benefit of
    # cropping at all. Faces found (5 video frames / 20-person CCTV frame):
    #   pad 0.15, full body -> 1/5  and  7/20
    #   pad 0.00, full body -> 5/5  and  8/20   <- default
    # Cropping only the upper part of the person box was also tried and is
    # worse on the classroom frame (top 50% -> 5/20, top 30% -> 1/20), because
    # students bent over desks have their head low in the box. Kept
    # configurable, but raise it only with measurements to justify it.
    person_crop_padding: float = 0.0

    # Overlapping person boxes can both see the same physical face. A candidate
    # whose IoU with an already-assigned face exceeds this is treated as a
    # duplicate and not assigned twice.
    duplicate_face_iou: float = 0.50


@dataclass(frozen=True)
class StudentResolutionConfig:
    """Recovering students that person detection missed — see backend/students.py.

    Face detection finds more students than person detection does, at every YOLO
    model/resolution combination measured across the 13 dataset images:

    =========================================  ========
    Signal                                     Count
    =========================================  ========
    Faces (SCRFD)                               434
    Persons (YOLOv11m @ 1280, old default)      264
    Persons (YOLOv11m @ 1920, current)          331
    Persons (YOLOv11x @ 1920, best tried)       374
    =========================================  ========

    Unmatched faces were rendered and hand-checked on ``img382.jpg``: all 30
    were real students in back rows, not false positives. Bodies are occluded by
    desks and neighbours; heads are not. So a face with no person box is treated
    as a student whose body box is *estimated*.
    """

    # Master switch. Off = the old behaviour (a student exists only if YOLO
    # found their body), which under-counts crowded rows by roughly a third.
    seed_persons_from_faces: bool = True

    # Body-box geometry, extrapolated from the face box. These are rough
    # anthropometric ratios, NOT measurements from this footage:
    #   * shoulder span is a little over twice head width
    #   * from an elevated camera, a seated student's visible extent runs from
    #     just above the crown to about the desk edge
    # They exist to give a face-seeded student a usable spatial anchor for
    # object association (a phone/book near them) and tracking continuity —
    # never for true body extent. See backend/students.py.
    body_width_to_face_width: float = 2.6
    body_height_to_face_height: float = 4.0
    body_top_above_face: float = 0.3

    # A face this weak is not trusted to invent a whole student. Set above
    # FaceConfig.scrfd_det_thresh on purpose: a marginal face is still worth
    # reporting when it sits inside a confirmed person box, but not worth
    # fabricating a body box for on its own.
    seed_min_face_score: float = 0.40


@dataclass(frozen=True)
class HeadPoseConfig:
    """Person C — SixDRepNet head-pose settings."""

    weights: str = "sixdrepnet_300w_lp_alpha1.pth"
    device: Literal["cuda", "cpu", "auto"] = "auto"

    # Where "attending" actually points, in degrees of yaw, for THIS camera.
    #
    # The gaze buckets below measure head rotation relative to the *camera*, and
    # the "teacher" bucket assumes the camera sits roughly where the teacher and
    # board are. A corner- or side-mounted camera breaks that assumption
    # completely, and it does so silently.
    #
    # Found on real footage: a 40-frame clip from a corner-mounted classroom
    # camera produced gaze_label "right" for 320 of 383 faces (84%), median yaw
    # +37 deg, with only 6 faces negative. Rendering yaw on the frame showed the
    # head-pose model was CORRECT -- the students really are rotated ~+26 to +77
    # deg relative to that camera, because they are facing a board that is
    # off-frame to the left. The angles were right; the *label* was wrong. Every
    # attending student was being classified as looking away, which in turn
    # feeds backend.attention's "oriented_away" bucket and would have corrupted
    # every engagement figure computed from this footage.
    #
    # This offset is subtracted from yaw before bucketing, so 0.0 means "camera
    # is at the front, co-located with the teacher". Estimate it per deployment
    # with backend.headpose.estimate_yaw_reference() rather than guessing.
    yaw_reference_deg: float = 0.0

    # Gaze bucket thresholds in degrees. Yaw = left/right, Pitch = up/down.
    # Ordered evaluation: "teacher" (frontal) > "down" > "back" > "left"/"right".
    yaw_side_threshold: float = 20.0  # |yaw| >= this -> left or right
    pitch_down_threshold: float = 20.0  # pitch >= this -> looking down
    pitch_back_threshold: float = -25.0  # pitch <= this -> looking backward/up

    # Padding around face bbox before feeding to SixDRepNet (relative to bbox size)
    crop_padding: float = 0.20


@dataclass(frozen=True)
class BehaviourConfig:
    """Fine-tuned per-student behaviour classification -- see backend/behaviour.py.

    Replaces the book-proximity proxy, which failed against human labels
    (precision 31.9%, recall 20.7%, F1 25.1% for "is this student writing").
    Training the behaviour directly rather than inferring it from a proxy object
    lifted that to F1 ~62-68% on held-out clips.

    This model is a **classifier layer on top of** the existing student
    detection, NOT a replacement for it. Measured on the same held-out data:

    ======================================  ==========  =======
    Finding students                        Precision   Recall
    ======================================  ==========  =======
    SCRFD + YOLO pipeline                     82.2%      90.6%
    This behaviour model alone                89.8%      70.5%
    ======================================  ==========  =======

    It is more precise but finds ~20 points fewer students, and a missed student
    is invisible to every downstream signal, so recall wins for detection.
    """

    # Produced by tools/train_behaviour.py. Under runs/ which is gitignored, so
    # a fresh clone must retrain (or be pointed at a copied checkpoint) rather
    # than silently running an untrained model.
    weights: str = "runs/behaviour/merged4_aug/weights/best.pt"
    device: Literal["cuda", "cpu", "auto"] = "auto"

    # Trained at 960 for VRAM reasons (batch 4 on a 6.4 GB card); inference uses
    # the same size, since matching train/test resolution is the safer default.
    imgsz: int = 960

    # Detection confidence. Swept on held-out clips, and lowering it does NOT
    # help -- for the weakest class, `using_device`, recall moves 20.0% -> 24.4%
    # -> 26.7% at conf 0.30 -> 0.15 -> 0.08 while precision collapses 58.1% ->
    # 45.8% -> 37.5%. That is a model/data limitation, not a threshold to tune,
    # so 0.30 stays.
    conf: float = 0.30
    iou: float = 0.50

    # Class order must match the training data.yaml exactly. A mismatch here
    # silently relabels every prediction.
    class_names: tuple[str, ...] = ("read", "sleep", "using_device", "write")

    # Previously dropped `handrise`/`stand` (22 and 59 training boxes, F1 4.1%
    # and 0.0%). The retrained model does not have those classes at all -- they
    # were excluded from the merged dataset -- so there is nothing left to
    # suppress here. Kept as an empty tuple rather than removed so the
    # suppression mechanism stays available if a future class needs it.
    untrusted_classes: tuple[str, ...] = ()

    # `turn_head`/`look_forward` were deferred to backend.headpose, which
    # measures head orientation far better (F1 63.2% vs 25.0% on the same
    # boxes). That decision is now baked into the model itself: both classes
    # were excluded from the merged training set entirely, which also dissolved
    # the label-density conflict that had blocked merging the second dataset
    # (see tools/merge_behaviour_datasets.py). Nothing left to defer at
    # inference time; head pose still owns orientation.
    deferred_classes: tuple[str, ...] = ()

    # Minimum fraction of a behaviour box that must fall inside a student's box
    # (or vice versa) to bind them. Mutual-centre containment is used rather
    # than IoU because the two box conventions differ: this model was trained on
    # tight head+torso annotations while the pipeline's students are full-body
    # or face-seeded boxes. IoU 0.5 rejected 4 of 11 correct pairs on a real
    # frame -- see tools/eval_detection.py.
    require_mutual_centre: bool = True


@dataclass(frozen=True)
class ExpressionConfig:
    """Facial-expression classification -- see backend/expression.py.

    Added as a requirement after Review 1. Note carefully what it is called:
    this reports an **expressed facial signal**, not an inferred internal
    emotional state. That distinction is not pedantry -- Barrett, Adolphs,
    Marsella, Martinez & Pollak (2019) reviewed over 1,000 studies and found no
    scientific support for reading emotion reliably from facial movement (a
    smile can signal submission rather than happiness). The EU AI Act also makes
    inferring *emotion* from biometric data in education a flat prohibition, not
    merely high-risk. Calling the output an expression label keeps the feature
    defensible; calling it emotion would not.
    """

    # EmotiEffLib (formerly HSEmotion) model. enet_b0_8_best_vgaf is the
    # EfficientNet-B0 8-class AffectNet model: smallest and fastest of the pack,
    # which matters because it runs once per student per frame.
    model_name: str = "enet_b0_8_best_vgaf"

    # "onnx" or "torch". ONNX runs on CPU here without competing with YOLO and
    # SixDRepNet for the 6.4 GB of VRAM those already occupy.
    engine: Literal["onnx", "torch"] = "onnx"

    # The model emits AffectNet's 8 classes; the project reports 3. Mapping is
    # config-driven (not hardcoded) and the **full 8-class distribution is kept**
    # in the output, so this collapse is auditable and reversible.
    #
    # Anger/Contempt/Disgust/Fear/Surprise map to "neutral" rather than to
    # "sad": they are distinct states, and folding anger into sadness would be a
    # claim the model never made. Mapping them to neutral says "not one of the
    # three we report", which is true.
    expression_map: tuple[tuple[str, str], ...] = (
        ("Happiness", "happy"),
        ("Sadness", "sad"),
        ("Neutral", "neutral"),
        ("Anger", "neutral"),
        ("Contempt", "neutral"),
        ("Disgust", "neutral"),
        ("Fear", "neutral"),
        ("Surprise", "neutral"),
    )

    # The three labels actually reported.
    reported_labels: tuple[str, ...] = ("happy", "sad", "neutral")

    # Align the face using the detector's 5 keypoints before classifying, rather
    # than feeding a raw box crop. AffectNet was trained on aligned faces, so an
    # unaligned crop is out of distribution. Measured over 207 real classroom
    # faces (enet_b0_8_best_vgaf):
    #                    median conf   share below 0.5 conf
    #   box crop            0.421            64%
    #   aligned (chosen)    0.481            55%
    # Requires SCRFD keypoints; falls back to the padded box crop when absent
    # (the mediapipe backend supplies none).
    align_faces: bool = True

    # Below this confidence the prediction is reported as "uncertain" instead of
    # being forced into happy/sad/neutral. This matters more than it looks:
    # measured on real classroom faces, 55% of aligned predictions fall under
    # 0.50, so a system that always emits one of three labels would be
    # presenting a coin-flip as a finding. Abstention is the honest alternative
    # to a confident wrong answer, and it is the reason "uncertain" is a
    # first-class output rather than an error.
    #
    # 0.40 is a starting point, NOT a calibrated threshold -- calibrating it
    # needs labelled expression crops this project does not have yet. Raising it
    # trades coverage for reliability; that trade should be made against labels,
    # not by taste.
    min_confidence: float = 0.40
    uncertain_label: str = "uncertain"

    # Temporal aggregation window, in frames, for ExpressionWindow. A single
    # frame's expression is noise: on one real clip, single-frame labels flipped
    # between consecutive frames on 6.8% of steps, while averaging the
    # distribution over 9 frames cut that to 1.4% -- a 5x reduction, for no
    # extra model cost. Nine frames is ~1 second at the pipeline's ~8 FPS.
    # This is the same "never judge a single frame" principle backend.attention
    # already applies to gaze, and the mechanism RDFER (base paper 1) uses.
    window_frames: int = 9

    # A face box smaller than this (shorter side, pixels) is skipped rather than
    # upscaled to the model's 224x224 input, on the rule that an unusable signal
    # is reported as absent rather than invented.
    #
    # Lowered from a guessed 40 to a MEASURED 25. The original 40 was set from
    # the intuition that "classifying a 7x upscale is guessing"; that intuition
    # was never tested and turned out to be far too conservative. Calibrated by
    # downscaling 120 labelled faces (dataset/kaggle_emotion_sample) to each
    # size, upscaling back to the model's 224px input -- exactly what the
    # pipeline does to a small face -- and measuring real 3-class accuracy
    # (random chance = 33%):
    #
    #   face size   accuracy
    #   197px        89.3%   (original, undownscaled)
    #    60px        84.0%
    #    45px        78.8%
    #    35px        70.3%
    #    28px        68.2%
    #    20px        54.7%
    #    15px        35.1%   <- indistinguishable from chance, genuinely useless
    #
    # 25 sits just below the 28px point where accuracy is still 68% (2x chance)
    # and above the 20px region where it collapses toward chance. The cost of
    # the old 40 was severe and measured: on 640x360 classroom video (median
    # face 28px) it rejected 91% of detected faces, giving 9.1% expression
    # coverage where 25 gives ~66%.
    #
    # Re-measure this curve before moving the gate again -- it is model- and
    # resolution-specific, not a universal constant.
    min_face_px: int = 25

    # Padding around the face box before the crop, as a fraction of box size.
    # AffectNet training images include hair, jawline and some background;
    # a tight box alone is out of distribution.
    crop_padding: float = 0.15


@dataclass(frozen=True)
class PostureConfig:
    """Exploratory — body-pose keypoints as a signal independent of a face.

    Not part of the frozen Stage 1 contract (schema.json has no posture field)
    and not wired into integrate.py's output. See backend/posture.py's module
    docstring for why this exists and what it deliberately does NOT claim.
    """

    model_complexity: int = 1
    static_image_mode: bool = True

    # Run posture ONLY for students with no detected face, instead of for
    # everyone. This is an opt-in performance trade, off by default, because it
    # is not free and the cost is easy to take by accident.
    #
    # The case for turning it on: posture is 21% of pipeline latency (1162 ms of
    # 5456 ms per 1920x1080 frame with ~14 students), and it exists as a
    # face-INDEPENDENT fallback. When that rationale was written, face coverage
    # was ~42%, so running it on everyone was nearly all useful. After the SCRFD
    # swap face coverage is ~89%, so most of that 21% now recomputes a fallback
    # for students who already have a better signal.
    #
    # The case against, and why the default stays False:
    # backend.peer_interaction requires BOTH students' shoulder keypoints to
    # detect a pair oriented toward each other. With this enabled, posture is
    # present only for faceless students, so peer detection would only ever fire
    # between two faceless students -- effectively disabling it. Saving 19% of
    # runtime by silently breaking a whole feature is a bad trade, and a worse
    # one to make invisibly.
    #
    # Turn it on when peer-interaction detection is not needed for a given run
    # (e.g. a latency-sensitive live demo), with that consequence understood.
    only_when_faceless: bool = False

    # Recovery of a faceless person's pose keypoints, measured across 167
    # faceless persons in 13 real classroom images:
    #   0.2 -> 111/167 (66%)
    #   0.3 -> 94/167  (56%)   <- chosen: the value actually hand-checked
    #   0.5 (MediaPipe's own default) -> 46/167 (28%)
    #   0.7 -> 18/167  (11%)
    # 0.2 recovers more but was not hand-verified against the real images the
    # way 0.3 was (see the module docstring's montage review) — raise it only
    # after doing the same check.
    min_detection_confidence: float = 0.3

    # MediaPipe Pose's 33-point landmark indices (BlazePose topology) used
    # here. Config-driven per the project's no-magic-numbers rule, though
    # these are fixed by the model, not tunable.
    nose_idx: int = 0
    left_shoulder_idx: int = 11
    right_shoulder_idx: int = 12
    # MediaPipe Pose indices. Wrists and elbows come free with the same
    # inference and were previously ignored; they are what make a raised hand,
    # writing, and a head resting on a hand distinguishable from each other.
    left_wrist_idx: int = 15
    right_wrist_idx: int = 16
    left_elbow_idx: int = 13
    right_elbow_idx: int = 14

    left_hip_idx: int = 23
    right_hip_idx: int = 24

    # A landmark is reported only if its MediaPipe visibility meets this.
    keypoint_min_visibility: float = 0.5


@dataclass(frozen=True)
class AttentionConfig:
    """Exploratory — Stage 3 first slice: windowed, per-student attention signal.

    Consumes Stage 1+2 JSONL output (gaze_label, EAR, posture presence,
    objects, track_id) after the fact; nothing here is wired into
    schema.json or the live capture loop. See backend/attention.py's module
    docstring for the research this operationalises and what it deliberately
    does not claim.

    Every timing default below is an engineering interpolation across several
    adjacent findings in cognitive-science and gaze-based mind-wandering
    research, not a number lifted from a single study that measured this
    exact system. Treat these as tunable starting points to validate against
    real footage, not settled constants.
    """

    # Rolling window for the per-frame category distribution. Gaze-based
    # mind-wandering detectors built for real lecture footage get their best
    # results aggregating over roughly 12 seconds, not single frames (Faber,
    # Bixler & D'Mello). 15s rounds that up with margin for this pipeline's
    # lower frame rate under posture fallback (~8 FPS on an RTX 4050).
    window_seconds: float = 15.0

    # How long a rolling window must stay majority "head_down_with_device"
    # before it is flagged as sustained rather than a normal brief lapse.
    # A single ~2-second break was shown to prevent vigilance decline
    # entirely (Ariga & Lleras 2011); most self-reported classroom lapses
    # last under a minute (Bunce, Flens & Neiles 2010). 90s sits at the
    # midpoint of the literature's 60-120s "this is no longer a blip" range
    # — deliberately not the lower bound, so a single missed glance-back
    # doesn't trip it.
    sustained_seconds: float = 90.0

    # Fraction of the rolling window that must be "head_down_with_device"
    # for the window to count as currently off-task, when accumulating
    # toward sustained_seconds.
    off_task_majority_fraction: float = 0.5

    # Per-student calibration baseline period. A real classroom deployment
    # (Sumer et al. 2021) measured a personal calibration baseline built
    # from the student's own first ~60s of data improving AUC by +0.084 --
    # the one concrete, literature-measured accuracy lever available here.
    calibration_seconds: float = 60.0

    # gaze_label "left"/"right" is deliberately NOT treated as off-task.
    # Kendon's F-formation research gives a real geometric definition of
    # joint peer interaction (reciprocal, sustained body orientation between
    # two tracked people), but detecting it needs pairing across tracked
    # students, which this first slice does not implement -- see the module
    # docstring. Collapsing "turned toward a neighbour" into "distracted"
    # would be exactly the mistake that research warned against, so it is
    # reported as its own "oriented_away" bucket instead: known-ambiguous,
    # not guessed at.
    #
    # gaze_label "down"/"back" is similarly not assumed to be off-task on its
    # own -- gaze aversion while concentrating on a hard problem is a
    # documented, opposite-reading confound (Doherty-Sneddon et al.). It is
    # only treated as a meaningful signal when combined with a nearby
    # "cell phone" detection, which is the one case with a defensible
    # behavioural reading in the existing schema (a phone under a bowed
    # head is a stronger proxy than a bowed head alone, which is equally
    # consistent with reading or writing).
    device_gaze_labels: tuple[str, ...] = ("down", "back")
    device_object_classes: tuple[str, ...] = ("cell phone",)

    # A bowed head over a *book* is the opposite signal from a bowed head over a
    # phone: it is the posture of a student working. Splitting these was raised
    # directly in review -- previously both collapsed into the ambiguous
    # "head_down_no_device" bucket, so a studying student and a disengaged one
    # were indistinguishable, and the system could not credit anyone for
    # working. A phone still wins when both are detected (see
    # backend.attention.classify_frame): the more concerning reading is the
    # safer default when the evidence is contradictory.
    #
    # "laptop" is deliberately NOT here. It is genuinely ambiguous -- note in
    # DetectionConfig that img04 is a computer lab where laptops are the work
    # and img01 an ordinary classroom where they are not -- and this project's
    # rule is that an ambiguous signal gets its own bucket rather than a guess.
    writing_object_classes: tuple[str, ...] = ("book",)

    # A "cell phone" detection counts as near a person if its box overlaps
    # theirs at all, in image space.
    device_proximity_iou: float = 0.0


@dataclass(frozen=True)
class PeerInteractionConfig:
    """Exploratory -- pairwise "peer-oriented" detection between students.

    Not part of the frozen Stage 1 contract; reads finished Stage 1+2 JSONL
    like backend.attention does. See backend/peer_interaction.py's module
    docstring for the F-formation research this operationalises, and what
    it deliberately does not claim (it detects joint physical orientation
    between two tracked students, never whether their interaction is
    academically productive or off-task -- that distinction is not
    recoverable from vision alone per the CSCL literature this implements a
    decision from).

    Every threshold below is an engineering default, not one measured for
    this exact system -- there is no labelled peer-interaction ground truth
    to calibrate against yet. Validate before trusting the output.
    """

    # Two people count as "at conversational distance" if the shorter of
    # their two bbox widths, scaled by this factor, exceeds the gap between
    # their bboxes. Scale-relative (not a fixed pixel count) so it holds
    # across near/far students in the same frame.
    max_gap_to_width_ratio: float = 1.0

    # How close each person's shoulder-line orientation must be to
    # "perpendicular to the line connecting them" to count as oriented
    # toward each other this frame, in degrees. Wide on purpose: Kendon's
    # F-formations include both vis-a-vis (face-to-face) and L-shaped
    # (cooperative, common in classroom side-by-side seating) arrangements,
    # and the shoulder-line-orientation test used here (see the module
    # docstring for why it is undirected, sidestepping the front/back
    # ambiguity in a single shoulder line) is a coarse proxy for either.
    orientation_tolerance_degrees: float = 35.0

    # Rolling window before judging a pair, and majority fraction of that
    # window required to count the pair as currently oriented toward each
    # other. Same rationale as backend.attention's windowing: Kendon's own
    # turn-taking research shows real conversation has intermittent gaze,
    # so a momentary break must not reset a genuine pairing.
    window_seconds: float = 15.0
    majority_fraction: float = 0.5

    # How long a pair must stay majority-oriented before being reported at
    # all. Deliberately not tied to backend.attention's sustained_seconds:
    # this is reporting a detected joint orientation, not a sustained
    # concern, so it can surface sooner.
    sustained_seconds: float = 20.0


@dataclass(frozen=True)
class SceneGraphConfig:
    """Stage 3 -- scene graph generation parameters."""

    # Two nodes are spatially adjacent if their scale-relative gap is within
    # this ratio times the narrower of their widths.
    adjacency_gap_ratio: float = 2.0

    # Max distance perpendicular to the connecting line segment between two
    # students for a shared object to be considered "between" them, in pixels.
    max_shared_object_distance_px: float = 150.0


@dataclass(frozen=True)
class TemporalConfig:
    """Stage 4 -- temporal windowing analysis parameters."""

    # Rolling window sizes and thresholds (consolidated from defaults).
    window_seconds: float = 15.0
    sustained_attention_seconds: float = 90.0
    sustained_interaction_seconds: float = 20.0



@dataclass(frozen=True)
class FairnessAuditConfig:
    """Exploratory -- tooling for a demographic accuracy audit, and a
    confound diagnostic that needs no labels at all.

    See backend/fairness_audit.py's module docstring for the research this
    is grounded in: a real, primary-source check found MediaPipe Face Mesh
    has a published Google fairness model card (tested across Fitzpatrick
    skin tone AND a "Southern Asia" geographic bucket -- this project's
    actual population), while SixDRepNet has zero published fairness
    evaluation of any kind. Neither has been tested by anyone, anywhere,
    against South Asian faces specifically. This module cannot fill that gap
    without labelled data this project does not yet have -- it makes running
    that audit mechanical once such data exists, and runs the cheaper
    confound diagnostic the research recommends doing first.
    """

    # Fitzpatrick I-VI, matching Google's own MediaPipe Face Mesh fairness
    # card exactly, so any future ClassGraph audit is directly comparable to
    # published numbers rather than using an incompatible scale. Labels
    # should come from trained human annotation, not an automated classifier
    # -- the one academic study found using automated race/skin-tone labels
    # (WFLW's "Indian" subgroup, via a FairFace+CLIP ensemble) is flagged in
    # that same research as a source of label noise, not a shortcut to trust.
    skin_tone_scale: tuple[str, ...] = ("I", "II", "III", "IV", "V", "VI")

    # Resolution buckets (shorter image side, in pixels) for the label-free
    # confound diagnostic. A 2026 academic audit of a different landmark
    # model found image resolution alone explained 29.3% of landmark-error
    # variance -- the single largest factor found, ahead of any demographic
    # one -- which is why this is checked before any skin-tone-labelled
    # audit is attempted.
    resolution_bucket_edges: tuple[int, ...] = (480, 720, 1080)


@dataclass(frozen=True)
class TrackingConfig:
    """Stage 2 — ByteTrack settings, filling the ``track_id`` field Stage 1 always leaves ``null``.

    Wraps ultralytics' own ``BYTETracker``; these six fields are exactly what
    it reads from its ``args`` object (verified against
    ``ultralytics/trackers/byte_tracker.py`` and its default
    ``bytetrack.yaml``), so no local tracking logic is implemented here.
    """

    # First-stage association only matches detections scoring at or above
    # this; second stage recovers weaker ones down to track_low_thresh so an
    # occluded person is not immediately dropped.
    track_high_thresh: float = 0.25
    track_low_thresh: float = 0.10

    # A detection with no match starts a new track only if its score is at
    # least this — keeps one-off false-positive detections from spawning IDs.
    new_track_thresh: float = 0.25

    # How many frames a track survives with no matching detection before it is
    # dropped for good (handles brief occlusion / a face turned away).
    track_buffer: int = 30

    # IoU distance threshold for the Hungarian assignment between existing
    # tracks and this frame's detections.
    match_thresh: float = 0.80

    # Blend detection confidence into the assignment cost, not just IoU.
    fuse_score: bool = True


@dataclass(frozen=True)
class IdentityConfig:
    """Within-video face-recognition re-identification -- see backend/identity.py.

    Fills the gap ByteTrack alone cannot: a student who is briefly fully
    occluded, turns away, or leaves and re-enters frame gets a NEW track_id
    from motion tracking, because there is nothing to associate across the
    gap. Matching the reappearing face against faces already seen in this
    video recovers the original identity instead.

    Scope, stated plainly: this is re-identification WITHIN one video only.
    The gallery is built fresh per video (per
    :class:`~backend.identity.IdentityGallery` instance) and discarded when
    that instance goes out of scope -- no embedding is written to output or
    reused across videos. This preserves the project's existing
    session-reset identity property; it only makes identity more robust
    *inside* one session, which is what was asked for.
    """

    # Minimum cosine similarity between L2-normalised ArcFace embeddings to
    # count as the same person. 0.35 is a commonly-cited ballpark for this
    # embedding family (buffalo_l / w600k_r50), NOT a threshold calibrated
    # against this project's own population -- there is no labelled
    # same/different-identity pair data to calibrate it against yet. Treat as
    # a starting point: raise it if two different students get merged into
    # one id, lower it if the same student keeps splitting into new ids.
    match_threshold: float = 0.35

    # A face below this SCRFD detection score is not trusted to register or
    # match an identity -- a low-confidence detection can carry a distorted
    # embedding, and a wrong merge (two different people sharing one id) is a
    # worse failure than a missed re-identification (one person split across
    # two ids, which is what already happens today without this module).
    min_face_score_for_identity: float = 0.50

    # A person's stored embedding is updated toward each new sighting by this
    # weight (exponential moving average), rather than replaced outright, so
    # one poor-quality frame does not overwrite a good representative
    # embedding built from many earlier sightings.
    embedding_update_rate: float = 0.3

    # --- rejecting things that are not students ---------------------------- #
    #
    # A visual identity audit (tools/audit_identity.py) found the pipeline was
    # profiling two WALL POSTERS as students -- printed faces, tracked for 27
    # and 20 sightings, contributing a permanently "attentive, neutral" phantom
    # to every class-level aggregate. Nothing in the pipeline asked whether a
    # face ever changed: to a face detector, a printed face is a perfectly good
    # face.
    #
    # A first attempt used positional variance (a poster should not move) and
    # was REJECTED on measurement: this camera pans, so the posters appeared to
    # move as much as some students (1.38-1.74 vs a barely-moving student at
    # 0.39 face-widths). Position cannot separate them.
    #
    # What does separate them cleanly is APPEARANCE INVARIANCE. A real face
    # blinks, turns and changes expression; a printed one does not. Mean
    # pairwise similarity between a person's own face crops (lighting-normalised
    # grayscale), measured on the audited video:
    #
    #   known posters   0.906, 0.909
    #   known students  0.311, 0.580, 0.583, 0.728, 0.817
    #
    # 0.86 sits in that gap, closer to the students' worst case than to the
    # posters, so a genuinely still student is less likely to be discarded than
    # a poster is to be kept. Deliberately conservative in that direction:
    # dropping a real student loses data, while keeping a poster corrupts
    # aggregates for every student.
    reject_static_faces: bool = True
    static_face_similarity: float = 0.86

    # Below this many sightings there is not enough evidence to judge whether an
    # identity is static, so it is never rejected as a poster on that basis --
    # a student who appears in 4 frames would trivially look "invariant".
    static_face_min_sightings: int = 8

    # --- identifying people the tracker never picked up -------------------- #
    #
    # Measured on the real video: 164 of 801 person detections (20.5%) carried
    # NO track_id, and identity was keyed on track_id alone, so every one of
    # them was silently dropped from the student roster. Those people were
    # detected -- they just never got identified, and then vanished from the
    # count. This is the "not all students detected" complaint in its real form.
    #
    # The cause is not detection confidence (mean 0.527 for the unidentified,
    # well above the 0.25 tracker threshold). It is that ByteTrack associates by
    # box overlap and needs two CONSECUTIVE matches before issuing an id, while
    # this pipeline samples 1 frame per second from a PANNING camera. Measured
    # best IoU from each box to the next processed frame:
    #
    #   p10 0.113   p25 0.419   p50 0.721   p75 0.874
    #
    # 12.6% of boxes fall below IoU 0.20, ByteTrack's association floor, so they
    # cannot be matched at all. ByteTrack assumes adjacent 30fps frames where
    # boxes barely move; a one-second gap with camera motion destroys exactly
    # the signal it depends on.
    #
    # So do not rely on it for these. A detection with a trustworthy face can be
    # matched into the gallery on APPEARANCE, which across a one-second gap on a
    # moving camera is far more reliable than box overlap -- and the embedding is
    # already computed, so this costs nothing extra. Each such detection enters
    # as its own single-frame surrogate key; the co-occurrence constraint in
    # TwoPassIdentityResolver still applies, so it cannot be folded into someone
    # who is visible alongside it.
    identify_untracked: bool = True

    # Surrogate keys start here to stay clear of real ByteTrack ids, which are
    # small positive ints. Kept out of the output: a surrogate is an identity
    # bookkeeping key, not a claim that the tracker tracked anything, so
    # track_id in the JSONL stays null for these.
    surrogate_key_base: int = 1_000_000

    # --- quality-gated identity creation ---------------------------------- #
    #
    # The second half of docs/LITERATURE_REVIEW.md section 2. Only one half
    # shipped (constrained clustering, commit 1b1d404); this is the other.
    #
    # A face too small or too weak to trust may still JOIN an identity -- it is
    # matched against a cluster anchored by better observations, so the
    # comparison means something. It may not FOUND one, because founding is
    # checked against nothing at all. MagFace/AdaFace establish why the
    # asymmetry is necessary: low-quality embeddings sit closer to the origin
    # with higher angular variance, so their similarities are unreliable in
    # BOTH directions -- spuriously high (merging two students) and spuriously
    # low (splitting one). No single match_threshold is simultaneously safe for
    # high- and low-quality faces in the same video, which is exactly what the
    # threshold sweep measured: over 0.20-0.60 the id count never reached the
    # 8-person ground truth, so no value of it can be the fix.
    #
    # Off by default: it changes how many people the pipeline reports, so it is
    # a deliberate choice with a measured effect, not a silent default.
    quality_gated_creation: bool = False

    # Floors a cluster's BEST observation must clear to found an identity.
    # Best, not median: one clear look at someone is enough to establish they
    # exist, and demanding sustained quality would delete back-row students who
    # are only ever seen well once.
    min_face_score_to_found: float = 0.60
    min_face_px_to_found: int = 20

    # Path to a registered-face gallery written by tools/register_faces.py.
    #
    # None (the default) keeps this project's session-scoped identity property
    # exactly as it has always been: ids are anonymous, invented per video and
    # discarded, and no face data outlives processing.
    #
    # Setting it opts into the opposite: person ids come from a PERSISTENT
    # gallery of named people, so the same human keeps the same id across every
    # video they appear in. That is useful -- it is the only way an id means
    # anything beyond one session -- but it is a different privacy regime, and
    # backend/enrollment.py documents the consent and legal obligations that
    # come with it (DPDP Act 2023 child-data provisions; the school-facial-
    # recognition bans cited in backend/tracking.py). It is a deliberate choice,
    # never a default.
    gallery_path: str | None = None


@dataclass(frozen=True)
class PipelineConfig:
    """Shared integration (Day 4) settings."""

    # Process every Nth frame (1 = every frame). Higher = faster, less temporal
    # detail. Downstream temporal module needs consistent spacing so keep this
    # fixed for a given output.
    sample_rate: int = 1

    # How often to log throughput (in processed frames).
    log_every_frames: int = 30

    # Where to write JSONL output when using the CLI.
    default_output: str = "outputs/stage1.jsonl"

    # Fail fast if a video can't be opened.
    strict_io: bool = True


@dataclass(frozen=True)
class ProfileConfig:
    """Per-student profile reporting -- see backend/student_profile.py.

    Exists because a visual identity audit (tools/audit_identity.py) showed the
    raw profile list was not a student roster. On the audited 5.5-minute video
    it contained 18 entries, of which only ~5-6 were real, correctly-identified
    students; the rest were 1-3 frame detection ghosts and two wall posters.
    Reporting that list unfiltered overstates the result and pollutes every
    class-level aggregate.
    """

    # An identity seen in fewer frames than this is not reported as a student.
    # Measured on the audited video: 5 of 18 profiles had 1-2 sightings with
    # span 0.0s -- transient detection noise, not people. 3 is deliberately low
    # (a genuinely brief appearance is still a student) but enough to remove the
    # single-frame ghosts.
    min_frames_for_profile: int = 3

    # Keep filtered entries in the output, marked, rather than deleting them.
    # A reviewer should be able to see what was rejected and why -- silently
    # dropping detections is how a pipeline starts lying about its own recall.
    report_rejected: bool = True

    # --- who is the instructor -------------------------------------------- #
    #
    # The audit found the TEACHER being reported as a student, with the highest
    # sighting count of any identity (125). The tracking was right; the ROLE was
    # wrong. She receives an on-task proxy score and skews every class aggregate.
    #
    # Four ways to tell teacher from student were measured on the audited video,
    # all from signals the pipeline already computes, against the known teacher
    # (id 8) and known students (ids 1-5). ALL FOUR FAILED -- the teacher sits
    # inside the student range on every one:
    #
    #   signal            teacher    students      separates?
    #   median height       219       137-230          no
    #   aspect ratio       1.88      1.48-2.48         no
    #   centroid travel    3.03      1.87-4.50         no
    #   hips visible %       75%       18-93%          no
    #
    # A standing-vs-seated or walks-around heuristic therefore cannot be
    # justified on this footage: at 640x360, with a panning camera and a teacher
    # who is often at the front rather than pacing, the geometry simply does not
    # distinguish the roles. Rather than ship a guess that is wrong in an
    # invisible way, the role is stated explicitly by whoever ran the video --
    # one id, named once per recording. Listed ids are reported with
    # role="instructor" and excluded from student aggregates, never deleted.
    #
    # If a labelled multi-classroom set ever exists, revisit: this is a
    # measurement gap, not a claim that the problem is unsolvable.
    instructor_ids: tuple[int, ...] = ()

    # Whether an identity that was never verified by face may enter the student
    # roster. A negative person_id means identity could not confirm who this is
    # -- either no usable face was ever read, or (with
    # IdentityConfig.quality_gated_creation on) every face was too poor to
    # found an identity.
    #
    # Counting those as students inflates the roster with people the system
    # cannot actually identify, which is the failure this project's "honest
    # we-don't-know" claim exists to avoid: an unidentified detection should be
    # reported as unidentified, not silently promoted to a named student whose
    # attention trend then gets plotted.
    #
    # On the audited clip before quality gating this rule changes nothing --
    # all 4 unverified ids were already rejected as transient -- so it is a
    # tightening that costs no existing student.
    require_face_verified: bool = True


@dataclass(frozen=True)
class EngagementConfig:
    """Per-student behavioral proxy scoring -- see backend/engagement.py.

    Combines the two REAL signals a live frame actually carries --
    ``head_pose.gaze_label`` and ``behaviour.label`` -- into a single on-task /
    off-task / unknown verdict per frame, then an observed on-task percentage
    per student. The output is a *behavioral proxy score* (an "observed
    on-task indicator"), never a measurement of internal concentration: it is
    a hand-authored precedence rule with no external validation yet, and its
    structure is a documented adaptation of BOSS (Behavioral Observation of
    Students in Schools), which codes off-task behaviour independently of head
    orientation -- the same choice this config encodes. See
    ``backend/engagement.py`` (``BEHAVIORAL_PROXY_CAVEAT``) and
    ``docs/LITERATURE_REVIEW.md`` section 5. This is deliberately not a new
    classifier: it is the same precedence and honesty rules already
    established elsewhere in this project, applied consistently rather than
    re-invented per module.

    * A phone/sleep behaviour reading wins over an attentive-looking gaze, the
      same precedence :mod:`backend.attention` already uses (a contradictory
      reading resolves to the more concerning one, since crediting a student
      as working on the strength of a stray "attending" gaze while they hold a
      phone is the worse error).
    * Gaze alone (`left`/`right`/`down`/`back` with no behaviour reading) is
      NOT treated as off-task -- :mod:`backend.attention`'s own documented
      reasoning: gaze aversion and peer-oriented turning are both real,
      opposite-reading confounds for a bare gaze label. Verdict is left
      ``None`` (unknown) rather than guessed at.
    """

    # behaviour.label values that count as on-task regardless of gaze.
    on_task_behaviours: tuple[str, ...] = ("write", "read")

    # behaviour.label values that count as off-task -- checked BEFORE
    # on_task_behaviours and before gaze, so a phone/sleep reading always wins
    # a contradiction.
    off_task_behaviours: tuple[str, ...] = ("using_device", "sleep")

    # gaze_label values that count as on-task when no behaviour reading is
    # available or none of the above applied.
    attending_gaze_labels: tuple[str, ...] = ("teacher",)

    # --- Fallbacks for when the behaviour model produces nothing ------------ #
    #
    # Measured need, not speculation: on 640x360 out-of-distribution classroom
    # video the fine-tuned behaviour model returned ZERO detections across all
    # 801 person-observations (and still zero at conf 0.05, and still zero when
    # the frame was upscaled 2x and 3x -- so it is a domain gap, not a
    # resolution or threshold problem). Because "off" was only reachable via a
    # behaviour reading, every student scored 100% on-task purely from
    # absence of evidence.
    #
    # These fallbacks use signals the pipeline already computes and was
    # discarding. Both are the SAME evidence backend.attention already treats
    # as meaningful, so this is reusing a validated rule rather than inventing
    # a new one:
    #
    # 1. A phone detected overlapping the student's box (COCO object detection,
    #    independent of the behaviour model). backend.attention already uses
    #    exactly this for its head_down_with_device category.
    # 2. Eyes closed (EAR below FaceConfig.ear_closed_threshold) while the head
    #    is down -- the drowsiness signal EAR was computed for in the first
    #    place. EAR was available for 486/801 observations on that same video
    #    and was being thrown away entirely.
    #
    # Deliberately NOT a fallback: a bare non-attending gaze. That stays
    # "unknown" for the reason backend.attention documents at length -- gaze
    # aversion during hard thinking and turning toward a peer are real,
    # opposite-reading confounds.
    use_object_fallback: bool = True
    fallback_off_task_objects: tuple[str, ...] = ("cell phone",)

    # Eyes closed alone is not enough (a blink, or a bad EAR reading on a small
    # face). It counts as off-task only together with a head-down gaze, which is
    # the posture that distinguishes dozing from blinking.
    use_eye_closure_fallback: bool = True
    eye_closure_gaze_labels: tuple[str, ...] = ("down",)


@dataclass(frozen=True)
class Config:
    """Top-level config — compose all modules."""

    detection: DetectionConfig = field(default_factory=DetectionConfig)
    face: FaceConfig = field(default_factory=FaceConfig)
    students: StudentResolutionConfig = field(
        default_factory=StudentResolutionConfig
    )
    headpose: HeadPoseConfig = field(default_factory=HeadPoseConfig)
    identity: IdentityConfig = field(default_factory=IdentityConfig)
    expression: ExpressionConfig = field(default_factory=ExpressionConfig)
    behaviour: BehaviourConfig = field(default_factory=BehaviourConfig)
    engagement: EngagementConfig = field(default_factory=EngagementConfig)
    profile: ProfileConfig = field(default_factory=ProfileConfig)
    posture: PostureConfig = field(default_factory=PostureConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    peer_interaction: PeerInteractionConfig = field(
        default_factory=PeerInteractionConfig
    )
    scene_graph: SceneGraphConfig = field(default_factory=SceneGraphConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    fairness_audit: FairnessAuditConfig = field(default_factory=FairnessAuditConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"


# Global instance imported everywhere. Immutable (frozen dataclass) so no
# module can accidentally mutate it at runtime.
CONFIG: Config = Config()
