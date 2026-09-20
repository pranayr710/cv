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

Controls
    o / left      on task
    f / right     off task
    u             unclear -- excluded from training, still counted
    space         replay the window
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

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.json"
IMAGES = HERE / "images"

FRAME_DELAY_MS = 110  # slower than real time: judging sustained behaviour

KEY_ON = {ord("o"), 81, 2424832}
KEY_OFF = {ord("f"), 83, 2555904}
KEY_UNCLEAR = {ord("u")}
KEY_REPLAY = {ord(" ")}
KEY_BACK = {ord("b")}
KEY_SKIP = {ord("s")}
KEY_QUIT = {ord("q"), 27}


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

            key = None
            while key is None:
                key = render(w, i, len(windows), len(store))
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
