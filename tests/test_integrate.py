"""Integration tests for :mod:`backend.integrate`.

The full pipeline is driven with lightweight fakes injected into
``process_video`` for the four heavy modules (the real
Detector/FaceAnalyzer/HeadPoseEstimator/PostureAnalyzer need ML packages +
weights). The fakes return the *real* result dataclasses
(``Person``/``Obj``/``FaceResult``/``HeadPoseResult``/``PostureResult``), so
the wiring, index-alignment, contract assembly, JSONL writing and sample-rate
behaviour are all exercised for real and validated against ``schema.json``.

Person tracking (Stage 2) is deliberately NOT faked here: ``PersonTracker``
needs only ``ultralytics`` + ``lap`` (no weights, no GPU), the same real
dependency ``test_detection.py`` already requires, so ``process_video`` builds
a genuine ``PersonTracker`` and these tests exercise real ByteTrack end to end.

Required coverage:
    1. End-to-end on a 5-second fixture video -> valid JSONL matching schema.
    2. --sample-rate 5 produces ~1/5 the lines of --sample-rate 1.
    3. Stage 2: a continuously-visible person keeps one stable track_id.
    4. Two separate process_video() calls never share track identity when no
       tracker is injected -- this is the code-level boundary between
       "attention analytics" and persistent facial recognition (see
       backend.tracking's module docstring); it needs a real test, not just
       a comment, given the regulatory precedent behind it (Sweden's first
       GDPR fine targeted exactly a school system that persisted identity
       across sessions).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
jsonschema = pytest.importorskip("jsonschema")
pytest.importorskip("tqdm")
# PersonTracker (Stage 2) is not faked below, so its real dependency is needed.
pytest.importorskip("ultralytics")

from backend.config import CONFIG
from backend.detection import Obj, Person
from backend.face import FaceResult
from backend.headpose import HeadPoseResult
from backend.integrate import process_video
from backend.posture import PostureResult

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCHEMA_PATH = _REPO_ROOT / "schema.json"

_FIXTURE_FPS = 6  # keep the fixture small: 6 fps * 5 s = 30 frames
_FIXTURE_SECONDS = 5
_FIXTURE_W, _FIXTURE_H = 160, 120


class _FakeDetector:
    """Returns one person and one object on every frame, at a fixed bbox."""

    def __init__(self, bbox: tuple[int, int, int, int] = (20, 15, 60, 80)) -> None:
        self._bbox = bbox

    def detect(self, frame: np.ndarray) -> tuple[list[Person], list[Obj]]:
        persons = [Person(bbox=self._bbox, confidence=0.92)]
        objects = [Obj(cls="laptop", bbox=(5, 5, 30, 20), confidence=0.77)]
        return persons, objects


class _FakeFaceAnalyzer:
    """Returns a full 468-landmark FaceResult for each person bbox."""

    def analyze(self, frame: np.ndarray, person_bboxes) -> list[FaceResult]:
        results: list[FaceResult] = []
        for x, y, w, h in person_bboxes:
            landmarks = [(float(x + i % w), float(y + i % h)) for i in range(468)]
            results.append(
                FaceResult(
                    face_bbox=(x + 5, y + 5, max(1, w - 10), max(1, h - 10)),
                    landmarks=landmarks,
                    ear=0.31,
                )
            )
        return results


class _FakeHeadPose:
    """Returns a frontal 'teacher' pose for each non-None face bbox."""

    def estimate(self, frame: np.ndarray, face_bboxes) -> list[HeadPoseResult | None]:
        out: list[HeadPoseResult | None] = []
        for bbox in face_bboxes:
            if bbox is None:
                out.append(None)
            else:
                out.append(
                    HeadPoseResult(yaw=2.0, pitch=-3.0, roll=1.0, gaze_label="teacher")
                )
        return out


class _FakePostureAnalyzer:
    """Returns a fixed PostureResult (keypoints found) for each person bbox."""

    def analyze(self, frame: np.ndarray, person_bboxes) -> list[PostureResult]:
        results: list[PostureResult] = []
        for x, y, w, h in person_bboxes:
            l_sh = (float(x + w * 0.3), float(y + h * 0.3))
            r_sh = (float(x + w * 0.7), float(y + h * 0.3))
            results.append(
                PostureResult(
                    keypoints_detected=True,
                    nose=(float(x + w / 2), float(y + h * 0.1)),
                    left_shoulder=l_sh,
                    right_shoulder=r_sh,
                    shoulder_mid=(float(x + w / 2), float(y + h * 0.3)),
                    hip_mid=(float(x + w / 2), float(y + h * 0.7)),
                    vertical_lean=-0.2,
                    facing_direction=(0.0, -1.0),
                )
            )
        return results


def _make_fixture_video(path: Path, n_frames: int) -> int:
    """Write a small synthetic video and return the number of frames written."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(path), fourcc, float(_FIXTURE_FPS), (_FIXTURE_W, _FIXTURE_H)
    )
    assert writer.isOpened(), "VideoWriter failed to open (codec unavailable?)."
    try:
        for i in range(n_frames):
            # Vary pixels a little per frame so it's not a degenerate stream.
            frame = np.full((_FIXTURE_H, _FIXTURE_W, 3), i % 256, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()
    return n_frames


class _FakeExpression:
    """Returns a fake ExpressionResult for each face bbox."""

    def classify(self, frame: np.ndarray, face_bboxes) -> list:
        from backend.expression import ExpressionResult
        out = []
        for bbox in face_bboxes:
            if bbox is None:
                out.append(None)
            else:
                out.append(
                    ExpressionResult(
                        label="neutral",
                        confidence=0.8,
                        distribution={"Happiness": 0.1, "Sadness": 0.1, "Neutral": 0.8},
                    )
                )
        return out


class _FakeBehaviourClassifier:
    """Returns a fake BehaviourResult for each student bbox."""

    def classify(self, frame: np.ndarray, student_bboxes) -> list:
        from backend.behaviour import BehaviourResult
        out = []
        for bbox in student_bboxes:
            out.append(
                BehaviourResult(
                    label="write",
                    confidence=0.85,
                    reliability="measured",
                )
            )
        return out


def _fakes(bbox: tuple[int, int, int, int] = (20, 15, 60, 80)) -> dict:
    """Injected-estimator kwargs for process_video.

    ``person_tracker`` is deliberately absent: process_video builds a real
    ``PersonTracker`` (see the module docstring for why that's safe here).

    Args:
        bbox: The fake detector's fixed person bbox. Overridden by the
            session-identity tests so two "videos" place their person at
            different positions, which is what makes tracker reuse without
            ``.reset()`` observably different from a fresh tracker.
    """
    return {
        "detector": _FakeDetector(bbox=bbox),
        "face_analyzer": _FakeFaceAnalyzer(),
        "headpose_estimator": _FakeHeadPose(),
        "posture_analyzer": _FakePostureAnalyzer(),
        "expression_recognizer": _FakeExpression(),
        "behaviour_classifier": _FakeBehaviourClassifier(),
    }


@pytest.fixture(scope="module")
def schema() -> dict:
    """The frozen Stage 1 JSON schema, loaded once."""
    with _SCHEMA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def test_end_to_end_valid_jsonl(schema: dict, tmp_path: Path) -> None:
    """A 5-second fixture video yields schema-valid JSONL, one line per frame."""
    n_frames = _FIXTURE_FPS * _FIXTURE_SECONDS  # 30
    video = tmp_path / "clip.mp4"
    _make_fixture_video(video, n_frames)

    out = tmp_path / "stage1.jsonl"
    written = process_video(video, out, CONFIG, **_fakes())

    assert written == n_frames
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == written

    validator = jsonschema.Draft202012Validator(schema)
    prev_frame_id = -1
    for line in lines:
        record = json.loads(line)
        validator.validate(record)  # raises on any contract violation

        assert record["frame_id"] > prev_frame_id  # strictly increasing
        prev_frame_id = record["frame_id"]

        assert len(record["persons"]) == 1
        person = record["persons"][0]
        # Stage 2: the fake detector reports the same bbox every frame, so
        # ByteTrack should hold one stable id across the whole clip. Frame 0
        # is ByteTrack's own frame_id 1, which activates a new track
        # immediately (see backend.tracking's module docstring), so the id is
        # present from the very first record here, not just from frame 1 on.
        assert person["track_id"] == 1
        assert len(person["face"]["landmarks"]) == 468
        assert person["head_pose"]["gaze_label"] == "teacher"
        assert person["posture"]["nose"] is not None
        assert person["posture"]["vertical_lean"] == pytest.approx(-0.2)
        assert record["objects"][0]["cls"] == "laptop"


def test_two_videos_never_share_track_identity(tmp_path: Path) -> None:
    """Two separate process_video() calls, no tracker injected, never leak
    track identity between them -- the default/safe path.

    Uses a different bbox for each "video" so a leak would be observable
    (see test_reusing_one_tracker_without_reset_does_leak_identity for why
    an identical bbox cannot distinguish a fresh tracker from a reused one:
    ByteTrack would match the still-alive old track by position and land on
    id 1 either way, coincidentally, for the wrong reason).

    This is not just a Stage 2 numbering detail: it is the code-level
    boundary between "attention analytics" (identity is scoped to one
    session) and persistent facial recognition (identity survives across
    sessions), which several jurisdictions regulate or ban outright in
    schools -- see CHALLENGES_AND_SOLUTIONS.md and the "Reading the Room"
    research this implements a guardrail from.
    """
    n_frames = _FIXTURE_FPS * _FIXTURE_SECONDS
    video_a = tmp_path / "clip_a.mp4"
    video_b = tmp_path / "clip_b.mp4"
    _make_fixture_video(video_a, n_frames)
    _make_fixture_video(video_b, n_frames)

    out_a = tmp_path / "a.jsonl"
    out_b = tmp_path / "b.jsonl"
    # No person_tracker= passed to either call: process_video must build a
    # fresh one each time (see backend.integrate._build_person_tracker).
    process_video(video_a, out_a, CONFIG, **_fakes(bbox=(20, 15, 60, 80)))
    process_video(video_b, out_b, CONFIG, **_fakes(bbox=(400, 300, 60, 80)))

    first_id_a = json.loads(out_a.read_text(encoding="utf-8").splitlines()[0])[
        "persons"
    ][0]["track_id"]
    first_id_b = json.loads(out_b.read_text(encoding="utf-8").splitlines()[0])[
        "persons"
    ][0]["track_id"]

    # A properly fresh tracker gets the frame-1 instant-activation bonus
    # (see backend.tracking's module docstring) regardless of bbox position,
    # so both independent sessions confirm their first person immediately,
    # both numbered starting from 1.
    assert first_id_a == 1
    assert first_id_b == 1


def test_reusing_one_tracker_without_reset_does_leak_identity(tmp_path: Path) -> None:
    """The one real gap: deliberately reusing a PersonTracker across videos
    without calling .reset() is NOT guarded against, and does leak state.

    Video B's person appears at a bbox far from video A's, so the leak is
    observable: a genuinely fresh tracker would instant-activate it as id 1
    on its first frame (the frame-1 bonus applies to whatever bbox shows up
    first). A tracker that was never reset is no longer on its own frame 1
    -- it is several videos'-worth of frames into a single continuous
    ByteTrack sequence -- so video B's person, appearing at an unrelated
    position, gets no such bonus and starts unconfirmed like any new
    detection mid-sequence, exactly as if it were just another person
    walking into frame partway through video A.

    This test exists to make that contract concrete and testable, not just a
    warning in a docstring. Anyone injecting person_tracker= across more than
    one process_video() call must call .reset() themselves in between.
    """
    from backend.tracking import PersonTracker

    n_frames = _FIXTURE_FPS * _FIXTURE_SECONDS
    video_a = tmp_path / "clip_a.mp4"
    video_b = tmp_path / "clip_b.mp4"
    _make_fixture_video(video_a, n_frames)
    _make_fixture_video(video_b, n_frames)

    shared_tracker = PersonTracker()
    out_a = tmp_path / "a.jsonl"
    out_b = tmp_path / "b.jsonl"
    process_video(
        video_a,
        out_a,
        CONFIG,
        person_tracker=shared_tracker,
        **_fakes(bbox=(20, 15, 60, 80)),
    )
    # Deliberately no shared_tracker.reset() here -- this is the misuse case.
    process_video(
        video_b,
        out_b,
        CONFIG,
        person_tracker=shared_tracker,
        **_fakes(bbox=(400, 300, 60, 80)),
    )

    first_id_b = json.loads(out_b.read_text(encoding="utf-8").splitlines()[0])[
        "persons"
    ][0]["track_id"]
    # No frame-1 bonus available (the shared tracker's internal frame counter
    # is already well past 1) -- video B's person is treated as an ordinary
    # mid-sequence sighting and comes back unconfirmed, not instantly id 1.
    # That is the leak: the session boundary this student's "arrival" should
    # have marked was invisible to the tracker.
    assert first_id_b is None


def test_sample_rate_reduces_line_count(tmp_path: Path) -> None:
    """--sample-rate 5 produces about one fifth of the full-rate lines."""
    n_frames = _FIXTURE_FPS * _FIXTURE_SECONDS  # 30
    video = tmp_path / "clip.mp4"
    _make_fixture_video(video, n_frames)

    cfg_full = replace(CONFIG, pipeline=replace(CONFIG.pipeline, sample_rate=1))
    cfg_sampled = replace(CONFIG, pipeline=replace(CONFIG.pipeline, sample_rate=5))

    out_full = tmp_path / "full.jsonl"
    out_sampled = tmp_path / "sampled.jsonl"
    n_full = process_video(video, out_full, cfg_full, **_fakes())
    n_sampled = process_video(video, out_sampled, cfg_sampled, **_fakes())

    assert n_full == n_frames
    # Every 5th frame of N frames -> ceil(N / 5).
    expected = -(-n_frames // 5)
    assert n_sampled == expected
    # "~1/5" sanity band around the exact ceil value.
    assert abs(n_sampled - n_full / 5) <= 1

    # Sampled frame_ids are the multiples of 5 actually present in the source.
    sampled_ids = [
        json.loads(line)["frame_id"]
        for line in out_sampled.read_text(encoding="utf-8").strip().splitlines()
    ]
    assert sampled_ids == list(range(0, n_frames, 5))
