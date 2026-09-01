"""Unit tests for :mod:`backend.detection`.

These tests exercise the real YOLOv11 model. They require ``ultralytics`` +
``torch`` to be installed (and, on first run, network access for Ultralytics to
download the COCO weights). When those deps are unavailable the model-backed
tests skip cleanly rather than fail — we never fake a green result.

Test coverage (as specified for Person A):
    1. The model loads without error.
    2. A person is detected on a supplied fixture image.
    3. A black (all-zero) frame yields no persons and no objects.
    4. JSONL output written by ``run_on_video`` matches ``schema.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

# Skip the entire module if the detection stack is not installed. This keeps CI
# honest: absent deps => skipped (visibly), never a false pass.
pytest.importorskip("ultralytics")
pytest.importorskip("torch")
cv2 = pytest.importorskip("cv2")
jsonschema = pytest.importorskip("jsonschema")

from backend.config import CONFIG
from backend.detection import (
    Detector,
    Obj,
    Person,
    run_on_video,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCHEMA_PATH = _REPO_ROOT / "schema.json"
_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="module")
def detector() -> Detector:
    """A single shared detector for the module (model load is expensive)."""
    return Detector()


@pytest.fixture(scope="module")
def schema() -> dict:
    """The frozen Stage 1 JSON schema loaded once."""
    with _SCHEMA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _find_fixture_image() -> Path | None:
    """Return a person-containing fixture image, or ``None`` if none is found.

    Resolution order:
        1. Any image a teammate drops into ``tests/fixtures/`` (preferred — use a
           real classroom frame here for a representative test).
        2. Ultralytics' bundled sample assets (``bus.jpg`` has four people,
           ``zidane.jpg`` has two), which ship with the package. This lets the
           test run for real out of the box on the target GPU machine without
           anyone having to supply an image manually.

    Returns:
        A path to a usable image, or ``None`` if neither source is available.
    """
    if _FIXTURE_DIR.is_dir():
        for pattern in ("*.jpg", "*.jpeg", "*.png"):
            for candidate in sorted(_FIXTURE_DIR.glob(pattern)):
                return candidate

    try:
        from ultralytics.utils import ASSETS
    except Exception:  # noqa: BLE001 - any import/attr failure => no fallback
        return None

    for name in ("bus.jpg", "zidane.jpg"):
        candidate = Path(ASSETS) / name
        if candidate.is_file():
            return candidate
    return None


def test_model_loads_without_error(detector: Detector) -> None:
    """The Ultralytics model loads and exposes a class-name map."""
    assert detector.device in ("cuda", "cpu")
    # `.names` is populated from the loaded model; person must be a known class.
    assert "person" in detector._names.values()


#: Known wide classroom frames in ``dataset/``, in preference order. Named
#: explicitly rather than picked by a heuristic: the folder also holds close-up
#: stock photos (``8FURIilo_2x.jpg`` is three people holding phones, added to
#: exercise phone detection), and "widest file" or "first alphabetically" both
#: select one of those. Choosing by detected-person count would make the test
#: circular — it would pass by construction.
_CLASSROOM_IMAGES: tuple[str, ...] = ("img01.jpg", "img12.jpg", "img382.jpg")


def _find_classroom_image() -> Path | None:
    """Return a known wide classroom frame from ``dataset/``, or ``None``.

    ``dataset/`` is gitignored, so this returns ``None`` on a fresh clone and
    the caller skips rather than fails.

    Returns:
        A path to a known classroom frame, or ``None`` if none is present.
    """
    dataset = _REPO_ROOT / "dataset"
    if not dataset.is_dir():
        return None
    for name in _CLASSROOM_IMAGES:
        candidate = dataset / name
        if candidate.is_file():
            return candidate
    return None


def test_detects_many_students_on_a_real_classroom_frame(detector: Detector) -> None:
    """The target domain: a wide classroom shot must yield many students.

    A far stronger assertion than ">= 1 person in some image" — a wide
    classroom frame contains dozens of students, so a configuration that
    silently halves recall fails here instead of passing.
    """
    image = _find_classroom_image()
    if image is None:
        pytest.skip("No classroom images in dataset/ (gitignored).")

    frame = cv2.imread(str(image))
    assert frame is not None, f"Failed to read classroom image: {image}"

    persons, objects = detector.detect(frame)

    assert isinstance(objects, list)
    assert len(persons) >= 10, (
        f"Expected many students in the classroom frame {image.name}, "
        f"found {len(persons)}. Detection recall has regressed badly."
    )
    for person in persons:
        assert isinstance(person, Person)
        _x, _y, w, h = person.bbox
        assert w > 0 and h > 0
        assert 0.0 <= person.confidence <= 1.0
        assert person.confidence >= detector.config.person_conf
        assert person.source == "yolo"
    for obj in objects:
        assert isinstance(obj, Obj)
        assert obj.cls in detector.config.object_whitelist


def test_detects_person_on_fixture() -> None:
    """A portrait-style image yields a person at a portrait-appropriate imgsz.

    Documents a real, measured limitation rather than asserting the shipped
    config works everywhere. ``CONFIG.detection.imgsz`` is 1920 because that is
    right for wide classroom shots, but on ``frontal_face.jpg`` (802 px, one
    person filling the frame) that upscale loses the detection entirely:

        imgsz 640-1440 -> 1 person
        imgsz 1600     -> 0 persons
        imgsz 1920     -> 0 persons

    Clamping imgsz to native resolution was tried as a fix and rejected: it
    cost 331 -> 263 persons across the dataset (see backend/detection.py's
    ``_UPSCALE_WARN_FACTOR``). So the model and code are fine — this is purely
    scale tuning, and this test pins that diagnosis so the limitation cannot be
    mistaken for a code defect later.
    """
    fixture = _find_fixture_image()
    if fixture is None:
        pytest.skip("No fixture image in tests/fixtures/.")

    frame = cv2.imread(str(fixture))
    assert frame is not None, f"Failed to read fixture image: {fixture}"

    from dataclasses import replace

    native = max(frame.shape[:2])
    portrait_detector = Detector(replace(CONFIG.detection, imgsz=1280))
    persons, _objects = portrait_detector.detect(frame)

    assert len(persons) >= 1, (
        f"Expected >=1 person in {fixture.name} at imgsz=1280 "
        f"(native {native}px), found 0."
    )
    for person in persons:
        assert isinstance(person, Person)
        _x, _y, w, h = person.bbox
        assert w > 0 and h > 0
        assert 0.0 <= person.confidence <= 1.0


def test_heavy_upscaling_is_warned_about(caplog) -> None:
    """The warning that replaced the rejected clamp must actually fire.

    Without it, running this classroom-tuned config on a low-resolution source
    would silently return nothing — the exact failure mode measured above.
    """
    import logging
    from dataclasses import replace

    detector = Detector(replace(CONFIG.detection, imgsz=1920))
    small = np.zeros((300, 400, 3), dtype=np.uint8)  # 1920/400 = 4.8x upscale
    with caplog.at_level(logging.WARNING, logger="backend.detection"):
        detector.detect(small)
    assert any("upscales" in r.message for r in caplog.records), (
        "Expected an imgsz upscale warning for a 400px frame at imgsz=1920."
    )


def test_black_frame_returns_empty(detector: Detector) -> None:
    """An all-zero frame produces no persons and no objects."""
    black = np.zeros((720, 1280, 3), dtype=np.uint8)
    persons, objects = detector.detect(black)
    assert persons == []
    assert objects == []


def test_jsonl_output_matches_schema(
    detector: Detector, schema: dict, tmp_path: Path
) -> None:
    """`run_on_video` output is valid JSONL and conforms to schema.json."""
    # Build a short synthetic video (10 black frames). It will contain no
    # detections, but the per-frame records must still validate — empty
    # `persons`/`objects` arrays are legal under the schema.
    video_path = tmp_path / "clip.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    width, height, fps, n_frames = 320, 240, 10.0, 10
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    assert writer.isOpened(), "Could not open VideoWriter (codec unavailable?)."
    try:
        for _ in range(n_frames):
            writer.write(np.zeros((height, width, 3), dtype=np.uint8))
    finally:
        writer.release()

    out_path = tmp_path / "stage1.jsonl"
    written = run_on_video(video_path, out_path, detector=detector)

    assert written >= 1
    assert out_path.is_file()

    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == written

    validator = jsonschema.Draft202012Validator(schema)
    seen_ids: list[int] = []
    for line in lines:
        record = json.loads(line)
        # Raises jsonschema.ValidationError on any contract violation.
        validator.validate(record)
        seen_ids.append(record["frame_id"])
        for person in record["persons"]:
            assert person["track_id"] is None
            assert person["face"] is None
            assert person["head_pose"] is None

    # frame_ids are unique and monotonically increasing.
    assert seen_ids == sorted(seen_ids)
    assert len(set(seen_ids)) == len(seen_ids)


def test_jsonl_schema_holds_with_real_persons_present(
    detector: Detector, schema: dict, tmp_path: Path
) -> None:
    """Regression: the black-frame schema test above cannot catch a missing
    per-person field, because black frames produce an empty ``persons`` list
    and the per-person required fields are never exercised. Found for real:
    ``_frame_record`` was missing ``expression``/``behaviour`` (added to the
    schema's required list by later stages) for months, undetected, because
    every existing schema test happened to use empty-detection frames. This
    test uses a real classroom frame specifically so ``persons`` is non-empty.
    """
    image_path = _find_classroom_image()
    if image_path is None:
        pytest.skip("No classroom images in dataset/ (gitignored).")
    frame = cv2.imread(str(image_path))
    assert frame is not None

    video_path = tmp_path / "real_clip.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w = frame.shape[:2]
    writer = cv2.VideoWriter(str(video_path), fourcc, 5.0, (w, h))
    assert writer.isOpened()
    try:
        for _ in range(3):
            writer.write(frame)
    finally:
        writer.release()

    out_path = tmp_path / "real.jsonl"
    run_on_video(video_path, out_path, detector=detector)

    validator = jsonschema.Draft202012Validator(schema)
    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    any_persons = False
    for line in lines:
        record = json.loads(line)
        validator.validate(record)  # every required field must be present
        if record["persons"]:
            any_persons = True
    assert any_persons, (
        f"Expected at least one detected person in {image_path.name} to "
        f"actually exercise per-person schema validation."
    )


def test_detect_rejects_bad_input(detector: Detector) -> None:
    """Non-array and malformed frames raise explicit, typed errors."""
    with pytest.raises(TypeError):
        detector.detect("not-a-frame")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        detector.detect(np.zeros((0, 0, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        detector.detect(np.zeros((10, 10), dtype=np.uint8))  # missing channels


def test_effective_imgsz_never_upscales_past_the_measured_limit():
    """A close-up frame must not be enlarged into the range that erases it.

    On a 640x480 webcam frame, imgsz 1920 found zero people while one was
    plainly in shot -- the failure that put a bare hand and a held-up sheet of
    paper in their own person boxes.
    """
    from backend.detection import MAX_UPSCALE, effective_imgsz

    for native in (240, 480, 640, 720, 802, 1080):
        chosen = effective_imgsz(native, 1920)
        assert chosen <= max(640, native * MAX_UPSCALE) + 32


def test_effective_imgsz_leaves_the_classroom_setting_alone():
    """The cap must cost nothing on the footage imgsz 1920 was tuned for.

    The dataset images are 1280x720 and 1920-wide; both already sit at or below
    a 1.5x upscale, so the tuned value has to survive unchanged.
    """
    from backend.detection import effective_imgsz

    assert effective_imgsz(1280, 1920) == 1920
    assert effective_imgsz(1920, 1920) == 1920
