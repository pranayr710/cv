"""Hand-label engagement windows from pre-extracted crops.

Standalone: needs only this folder (manifest.json + images/) and two pip
packages (opencv-python, numpy). No connection to the rest of the ClassGraph
repository, no source video, no GPU.

The single most important property of this tool is what it does NOT show you.
The rule verdict, the action labels and the model's confidence are all
withheld while you decide; the verdict is recorded alongside your label
afterwards so the two can be compared, but it never appears on screen before
your decision. Seeing it first would anchor the label to the thing being
evaluated, and every agreement number computed later would be meaningless.

Each window is shown as a contact sheet: every frame at once, left to right and
top to bottom, with the student under judgement boxed in green. There is no
timer and no animation, so a window can be looked at for as long as it takes.
Pass --animate to play the frames in sequence instead.

Controls
    o / left      on task
    f / right     off task
    u             unclear -- excluded from training, still counted
    space         redraw (or replay, under --animate)
    b             go back one window
    s             skip without labelling (revisit later)
    q             save and quit

Progress is written to labels.json after every keypress, so a session can be
closed and resumed at any point with nothing lost. If several people are
labelling the same package, give each person their own copy of the folder and
merge the labels.json files afterwards (merge_labels.py does that).

    python label_gui.py
    python label_gui.py --limit 100
    python label_gui.py --who alex        # writes to labels_alex.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.json"
IMAGES = HERE / "images"

#: Frames across the contact sheet. Six columns over ~15 frames gives three
#: rows, which fits a laptop screen without shrinking each frame past the point
#: where a phone or a bowed head is visible.
SHEET_COLUMNS = 6

#: Size of each tile in the sheet.
TILE_PX = 190

#: Only used by --animate. Kept well above the old 110 ms, which cycled a whole
#: window in under two seconds and gave no still look at anything.
FRAME_DELAY_MS = 320

KEY_ON = {ord("o"), 81, 2424832}
KEY_OFF = {ord("f"), 83, 2555904}
KEY_UNCLEAR = {ord("u")}
KEY_REPLAY = {ord(" ")}
KEY_BACK = {ord("b")}
KEY_SKIP = {ord("s")}
KEY_QUIT = {ord("q"), 27}


def contact_sheet(window, idx, total, labelled):
    """Lay a window's frames out as one still image.

    The first version animated them at 110 ms a frame and looped, which meant
    the whole window flashed past in under two seconds and repeated. There was
    no moment at which a rater could actually look at anything, which is a poor
    way to ask for a considered judgement.

    A sheet shows every frame at once. The eye can move back and forth, compare
    the start of the window against the end, and take as long as it needs --
    which is what the question deserves, since the judgement is about what the
    student did across the whole fifteen seconds.
    """
    tiles = []
    for rel in window["frames"]:
        img = cv2.imread(str(IMAGES / rel))
        if img is None:
            continue
        scale = TILE_PX / max(img.shape[:2])
        img = cv2.resize(img, None, fx=scale, fy=scale)
        pad = np.full((TILE_PX, TILE_PX, 3), 24, np.uint8)
        y0 = (TILE_PX - img.shape[0]) // 2
        x0 = (TILE_PX - img.shape[1]) // 2
        pad[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
        tiles.append(pad)
    if not tiles:
        return None

    rows = []
    for i in range(0, len(tiles), SHEET_COLUMNS):
        row = tiles[i:i + SHEET_COLUMNS]
        while len(row) < SHEET_COLUMNS:
            row.append(np.full((TILE_PX, TILE_PX, 3), 24, np.uint8))
        rows.append(cv2.hconcat(row))
    sheet = cv2.vconcat(rows)

    panel = cv2.copyMakeBorder(sheet, 104, 60, 20, 20, cv2.BORDER_CONSTANT,
                               value=(28, 22, 18))
    cv2.putText(panel, f"student {window['person_id']}   "
                       f"{window['start_ms'] / 1000:.0f}s -"
                       f" {window['end_ms'] / 1000:.0f}s",
                (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2,
                cv2.LINE_AA)
    cv2.putText(panel, f"window {idx + 1} of {total}   ({labelled} labelled)"
                       f"   -- read left to right, top to bottom",
                (20, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 190, 210), 1,
                cv2.LINE_AA)
    cv2.putText(panel, "the green box is the student being judged",
                (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (120, 220, 140), 1,
                cv2.LINE_AA)
    foot = panel.shape[0] - 22
    cv2.putText(panel, "o on-task    f off-task    u unclear",
                (20, foot - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(panel, "b back    s skip    q save + quit",
                (20, foot + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                (150, 190, 210), 1, cv2.LINE_AA)
    return panel


def render(window, idx, total, labelled) -> int | None:
    """Play one window's saved frames. Returns the key pressed, or None."""
    frames = window["frames"]
    for rel in frames:
        img = cv2.imread(str(IMAGES / rel))
        if img is None:
            continue
        panel = cv2.copyMakeBorder(img, 108, 62, 20, 20,
                                   cv2.BORDER_CONSTANT, value=(28, 22, 18))
        # Deliberately no action label, no rule verdict, no confidence here.
        cv2.putText(panel, f"student {window['person_id']}", (20, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2,
                    cv2.LINE_AA)
        cv2.putText(panel, f"window {idx + 1} of {total}   "
                           f"({labelled} labelled)",
                    (20, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (150, 190, 210), 1, cv2.LINE_AA)
        cv2.putText(panel, f"{window['start_ms'] / 1000:.0f}s - "
                           f"{window['end_ms'] / 1000:.0f}s",
                    (20, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (150, 190, 210), 1, cv2.LINE_AA)
        foot = panel.shape[0] - 24
        cv2.putText(panel, "o on-task    f off-task    u unclear",
                    (20, foot - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (230, 230, 230), 1, cv2.LINE_AA)
        cv2.putText(panel, "space replay   b back   s skip   q save+quit",
                    (20, foot + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                    (150, 190, 210), 1, cv2.LINE_AA)

        cv2.imshow("label engagement", panel)
        key = cv2.waitKey(FRAME_DELAY_MS) & 0xFFFFFF
        if key not in (0xFFFFFF, 255):
            return key
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0,
                    help="stop after this many windows (0 = all)")
    ap.add_argument("--animate", action="store_true",
                    help="play the frames in sequence instead of showing them "
                         "as one sheet; slower to judge, occasionally useful "
                         "when motion is the question")
    ap.add_argument("--who", default=None,
                    help="labeller name; writes to labels_<who>.json instead "
                         "of labels.json, so several people can label the "
                         "same package without clobbering each other")
    args = ap.parse_args()

    if not MANIFEST.exists():
        print(f"{MANIFEST} not found. Run prepare_package.py first, or copy "
              "manifest.json and images/ into this folder.")
        return 1

    out = HERE / (f"labels_{args.who}.json" if args.who else "labels.json")
    windows = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if args.limit:
        step = max(1, len(windows) // args.limit)
        windows = windows[::step][:args.limit]
    print(f"{len(windows)} windows to label -> {out.name}")

    store = {}
    if out.exists():
        store = json.loads(out.read_text(encoding="utf-8"))
        print(f"resuming: {len(store)} already labelled")

    i = 0
    try:
        while 0 <= i < len(windows):
            w = windows[i]
            if w["window_id"] in store and args.limit == 0:
                i += 1
                continue

            if args.animate:
                key = None
                while key is None:
                    key = render(w, i, len(windows), len(store))
                    if key in KEY_REPLAY:
                        key = None
            else:
                sheet = contact_sheet(w, i, len(windows), len(store))
                if sheet is None:
                    i += 1
                    continue
                cv2.imshow("label engagement", sheet)
                # Blocks until a key: no timer, no loop, no time pressure.
                key = cv2.waitKey(0) & 0xFFFFFF
                if key in KEY_REPLAY:
                    continue

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

            store[w["window_id"]] = {
                "person_id": w["person_id"],
                "start_ms": w["start_ms"],
                "end_ms": w["end_ms"],
                "scene": w["scene"],
                "label": label,
                # Recorded for comparison, never shown before the decision.
                "rule_verdict": w["rule_verdict"],
                "features": w["features"],
                "feature_names": w["feature_names"],
            }
            out.write_text(json.dumps(store, indent=1), encoding="utf-8")
            i += 1
    finally:
        cv2.destroyAllWindows()

    counts = {}
    for v in store.values():
        counts[v["label"]] = counts.get(v["label"], 0) + 1
    print(f"\nsaved {len(store)} labels to {out}")
    for k in sorted(counts):
        print(f"  {k:<8} {counts[k]}")
    usable = sum(n for k, n in counts.items() if k != "unclear")
    print(f"  usable for training: {usable}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
