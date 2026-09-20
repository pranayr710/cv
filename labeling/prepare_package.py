"""Build the standalone labelling package: manifest.json + images/.

Run this ONCE, from the repository root, with the pipeline's environment
active. It needs everything the labeller should NOT need: the full repo, the
1 GB of source clips, and outputs/final2's raw.jsonl and live_graph.jsonl.

    python labeling/prepare_package.py

Everything it produces lives under labeling/ and is self-contained from there:
label_gui.py only needs manifest.json and images/, so labeling/ can be zipped
and handed to a teammate with no dependency on the rest of this repository.

Frames are subsampled and resized before being written, because saving every
frame at full resolution (605 windows x ~45 frames each) would be roughly half
a gigabyte of small files -- unnecessary weight for a folder meant to be
emailed or dropped in shared storage. FRAME_STRIDE and MAX_LONG_SIDE below are
the two knobs that trade package size against playback smoothness.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.engagement_features import FEATURE_NAMES, iter_windows

ROOT = Path(__file__).resolve().parent
CLIPS = Path("C:/Users/prana/Downloads/high-2636(2191-2347)/high-2636(2191-2347)")
GRAPH = Path("outputs/final2/live_graph.jsonl")
RAW = Path("outputs/final2/raw.jsonl")

IMAGES_DIR = ROOT / "images"
MANIFEST = ROOT / "manifest.json"

#: The pipeline sampled every Nth source frame; frame_id must be scaled back up
#: to find the matching frame in the original clips.
SAMPLE = 10

#: Keep roughly every 3rd saved frame. A 15-second window at ~3 fps carries
#: about 45 frames; a person's posture and activity do not change fast enough
#: to need all of them for a labelling judgement, and a third of the frames
#: keeps the sequence visibly continuous rather than a slideshow.
FRAME_STRIDE = 3

#: Resize each crop so its longer side is at most this many pixels. Chosen to
#: stay legible for judging posture/phone/book while keeping each JPEG small.
MAX_LONG_SIDE = 320

JPEG_QUALITY = 78


def clip_index() -> tuple[list[Path], list[int]]:
    clips = sorted(CLIPS.glob("*.mp4"),
                   key=lambda p: int(re.findall(r"\d+", p.name)[-1]))
    counts = []
    for c in clips:
        cap = cv2.VideoCapture(str(c))
        counts.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        cap.release()
    return clips, counts


def locate(frame_id: int, counts: list[int]) -> tuple[int, int] | tuple[None, None]:
    src = frame_id * SAMPLE
    for i, n in enumerate(counts):
        if src < n:
            return i, src
        src -= n
    return None, None


def load_boxes() -> dict[tuple[int, int], tuple[int, int, int, int]]:
    boxes: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for line in RAW.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        fid = int(rec.get("frame_id", -1))
        for person in rec.get("persons", []):
            pid = person.get("person_id")
            if pid and pid > 0 and person.get("bbox"):
                boxes[(fid, int(pid))] = tuple(int(v) for v in person["bbox"])
    return boxes


def save_crop(frame, bbox, dest: Path) -> bool:
    x, y, w, h = bbox
    pad = int(max(w, h) * 0.22)
    H, W = frame.shape[:2]
    crop = frame[max(0, y - pad):min(H, y + h + pad),
                 max(0, x - pad):min(W, x + w + pad)]
    if crop.size == 0:
        return False
    scale = MAX_LONG_SIDE / max(crop.shape[:2])
    if scale < 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale)
    dest.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(dest), crop,
                            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]))


def main() -> int:
    if not CLIPS.is_dir():
        print(f"source clips not found at {CLIPS}", file=sys.stderr)
        return 1
    if not GRAPH.exists() or not RAW.exists():
        print(f"pipeline outputs not found ({GRAPH}, {RAW})", file=sys.stderr)
        return 1

    windows = list(iter_windows(GRAPH))
    clips, counts = clip_index()
    boxes = load_boxes()
    print(f"{len(windows)} windows, {len(clips)} source clips, "
          f"{len(boxes)} boxes loaded")

    cap_cache: dict[int, cv2.VideoCapture] = {}
    manifest = []
    total_frames_written = 0

    for w in windows:
        window_id = f"p{w.person_id}_{w.start_ms}"
        frame_ids = [f for f in w.frame_ids if (f, w.person_id) in boxes]
        sampled = frame_ids[::FRAME_STRIDE] or frame_ids[:1]

        saved: list[str] = []
        for fid in sampled:
            ci, local = locate(fid, counts)
            if ci is None:
                continue
            cap = cap_cache.get(ci)
            if cap is None:
                cap = cap_cache[ci] = cv2.VideoCapture(str(clips[ci]))
            cap.set(cv2.CAP_PROP_POS_FRAMES, local)
            ok, frame = cap.read()
            if not ok:
                continue
            rel = f"{window_id}/{len(saved):03d}.jpg"
            if save_crop(frame, boxes[(fid, w.person_id)], IMAGES_DIR / rel):
                saved.append(rel)

        if not saved:
            continue
        total_frames_written += len(saved)
        manifest.append({
            "window_id": window_id,
            "person_id": w.person_id,
            "start_ms": w.start_ms,
            "end_ms": w.end_ms,
            "scene": w.scene,
            "rule_verdict": w.rule_verdict,
            "features": list(w.features),
            "feature_names": list(FEATURE_NAMES),
            "frames": saved,
        })

    for cap in cap_cache.values():
        cap.release()

    MANIFEST.write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    size_mb = sum(f.stat().st_size for f in IMAGES_DIR.rglob("*.jpg")) / 1e6
    print(f"\n{len(manifest)} windows packaged, {total_frames_written} frames "
          f"written ({size_mb:.0f} MB in {IMAGES_DIR})")
    print(f"manifest: {MANIFEST}")
    print("\nThe labeling/ folder is now self-contained. Zip it and share it, "
          "or hand it to a teammate as-is -- label_gui.py needs nothing else "
          "from this repository.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
