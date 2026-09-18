"""Hand-label engagement windows, for training a learned score.

The rule pipeline's verdict cannot be the training target -- a model fitted to
it would reproduce the rules and report near-perfect agreement while measuring
nothing. So the labels have to come from a person watching the footage.

The single most important property of this tool is that it shows you the
STUDENT, not the system's opinion of the student. The rule verdict, the action
labels and the model's confidence are all deliberately withheld while you
decide; the verdict is recorded alongside your label afterwards, so the two can
be compared, but it never appears on screen. Seeing it first would anchor the
label to the thing being evaluated.

Controls
    o / left      on task
    f / right     off task
    u             unclear -- excluded from training, counted in the report
    space         replay the window
    b             go back one window
    s             skip without labelling (revisit later)
    q             save and quit

Progress is written after every label, so quitting and resuming loses nothing.

    python tools/label_windows.py
    python tools/label_windows.py --limit 100     # a shorter sitting
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import cv2

sys.path.insert(0, ".")

from backend.engagement_features import FEATURE_NAMES, iter_windows

CLIPS = Path("C:/Users/prana/Downloads/high-2636(2191-2347)/high-2636(2191-2347)")
GRAPH = Path("outputs/final2/live_graph.jsonl")
RAW = Path("outputs/final2/raw.jsonl")
LABELS = Path("outputs/engagement_labels.json")

#: The pipeline sampled every Nth source frame; frame_id must be multiplied
#: back up to find the original frame in the clips.
SAMPLE = 10

#: Playback size for the student crop.
CROP_PX = 460

#: Milliseconds between frames during playback. Slower than real time on
#: purpose: the judgement is about sustained behaviour, and at true speed a
#: 15-second window is over before it can be read.
FRAME_DELAY_MS = 90

KEY_ON = {ord("o"), 81, 2424832}
KEY_OFF = {ord("f"), 83, 2555904}
KEY_UNCLEAR = {ord("u")}
KEY_REPLAY = {ord(" ")}
KEY_BACK = {ord("b")}
KEY_SKIP = {ord("s")}
KEY_QUIT = {ord("q"), 27}


def clip_index() -> tuple[list[Path], list[int]]:
    """The clips in playback order, with each one's frame count."""
    clips = sorted(CLIPS.glob("*.mp4"),
                   key=lambda p: int(re.findall(r"\d+", p.name)[-1]))
    counts = []
    for c in clips:
        cap = cv2.VideoCapture(str(c))
        counts.append(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        cap.release()
    return clips, counts


def locate(frame_id: int, counts: list[int]) -> tuple[int, int] | tuple[None, None]:
    """Map a pipeline frame_id to (clip index, frame within that clip)."""
    src = frame_id * SAMPLE
    for i, n in enumerate(counts):
        if src < n:
            return i, src
        src -= n
    return None, None


def load_boxes() -> dict[tuple[int, int], tuple[int, int, int, int]]:
    """Every (frame_id, person_id) -> bbox, so a crop can be cut per frame."""
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


def frames_for(window, boxes) -> list[int]:
    """The frame ids in this window that have a box for this student.

    Taken from the window itself rather than derived from elapsed time. An
    earlier version divided milliseconds by an estimated frame interval, which
    assumed all 157 source clips had been processed when only 60 were, and
    resolved crops for fewer than half the windows.
    """
    return [f for f in window.frame_ids if (f, window.person_id) in boxes]


def render(cap_cache, clips, counts, boxes, window, frame_ids, idx, total,
           labelled):
    """Play the window's crops once. Returns the key pressed, or None."""
    for fid in frame_ids:
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
        x, y, w, h = boxes[(fid, window.person_id)]
        pad = int(max(w, h) * 0.22)
        H, W = frame.shape[:2]
        crop = frame[max(0, y - pad):min(H, y + h + pad),
                     max(0, x - pad):min(W, x + w + pad)]
        if crop.size == 0:
            continue
        scale = CROP_PX / max(crop.shape[:2])
        crop = cv2.resize(crop, None, fx=scale, fy=scale)

        panel = cv2.copyMakeBorder(crop, 108, 62, 20, 20,
                                   cv2.BORDER_CONSTANT, value=(28, 22, 18))
        # Deliberately no action label, no rule verdict, no confidence.
        cv2.putText(panel, f"student {window.person_id}", (20, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.putText(panel, f"window {idx + 1} of {total}   ({labelled} labelled)",
                    (20, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (150, 190, 210), 1,
                    cv2.LINE_AA)
        cv2.putText(panel, f"{window.start_ms / 1000:.0f}s - "
                           f"{window.end_ms / 1000:.0f}s", (20, 92),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 190, 210), 1,
                    cv2.LINE_AA)
        foot = panel.shape[0] - 24
        cv2.putText(panel, "o on-task    f off-task    u unclear",
                    (20, foot - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(panel, "space replay   b back   s skip   q save+quit",
                    (20, foot + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                    (150, 190, 210), 1, cv2.LINE_AA)

        cv2.imshow("label engagement", panel)
        key = cv2.waitKey(FRAME_DELAY_MS) & 0xFFFFFF
        if key != 0xFFFFFF and key != 255:
            return key
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after this many windows (0 = all)")
    ap.add_argument("--graph", type=Path, default=GRAPH)
    ap.add_argument("--out", type=Path, default=LABELS)
    args = ap.parse_args()

    if not CLIPS.is_dir():
        print(f"source clips not found at {CLIPS}", file=sys.stderr)
        print("Extract high-2636(2191-2347).zip there first.", file=sys.stderr)
        return 1

    windows = list(iter_windows(args.graph))
    if args.limit:
        # Take an even spread rather than the first N, so a short sitting still
        # covers the whole session instead of only its opening minutes.
        step = max(1, len(windows) // args.limit)
        windows = windows[::step][:args.limit]
    print(f"{len(windows)} windows to label")

    store = {}
    if args.out.exists():
        store = json.loads(args.out.read_text(encoding="utf-8"))
        print(f"resuming: {len(store)} already labelled")

    clips, counts = clip_index()
    boxes = load_boxes()
    resolvable = sum(1 for w in windows if frames_for(w, boxes))
    print(f"{resolvable} of them have crops available")

    cap_cache: dict[int, cv2.VideoCapture] = {}
    i = 0
    try:
        while 0 <= i < len(windows):
            w = windows[i]
            key_id = f"{w.person_id}:{w.start_ms}"
            if key_id in store and args.limit == 0:
                i += 1
                continue
            frame_ids = frames_for(w, boxes)
            if not frame_ids:
                i += 1
                continue

            key = None
            while key is None:
                key = render(cap_cache, clips, counts, boxes, w, frame_ids,
                             i, len(windows), len(store))
                if key in KEY_REPLAY:
                    key = None

            if key in KEY_QUIT:
                break
            if key in KEY_BACK:
                i = max(0, i - 1)
                continue
            if key in KEY_SKIP:
                i += 1
                continue
            label = ("on" if key in KEY_ON else
                     "off" if key in KEY_OFF else
                     "unclear" if key in KEY_UNCLEAR else None)
            if label is None:
                continue

            store[key_id] = {
                "person_id": w.person_id,
                "start_ms": w.start_ms,
                "end_ms": w.end_ms,
                "scene": w.scene,
                "label": label,
                # Recorded for comparison, never shown before the decision.
                "rule_verdict": w.rule_verdict,
                "features": list(w.features),
                "feature_names": list(FEATURE_NAMES),
            }
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(store, indent=1), encoding="utf-8")
            i += 1
    finally:
        for cap in cap_cache.values():
            cap.release()
        cv2.destroyAllWindows()

    counts_by = {}
    for v in store.values():
        counts_by[v["label"]] = counts_by.get(v["label"], 0) + 1
    print(f"\nsaved {len(store)} labels to {args.out}")
    for k in sorted(counts_by):
        print(f"  {k:<8} {counts_by[k]}")
    usable = sum(n for k, n in counts_by.items() if k != "unclear")
    print(f"  usable for training: {usable}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
