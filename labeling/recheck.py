"""Re-judge windows you have already labelled, without being shown your answer.

Why this exists. The first 179 labels train a model to 0.640 accuracy. The 378
added in the second sitting train one to 0.492 -- chance. Both cover the same
27 students, so it is not that the later windows came from harder people. A
model that cannot beat a coin flip on a label set is saying the features carry
no information about those labels, and there are only two ways that happens:
the windows are genuinely unreadable, or the labels do not reflect a
consistent judgement.

This separates them. It replays windows you have already judged, in random
order, showing nothing about what you said before, and reports how often you
agree with yourself.

That number is worth more than any accuracy figure in the project, because it
is the ceiling on all of them. A model cannot be more consistent than the
labels it learns from. If self-agreement is near 0.9 the labels are sound and
the model is the problem; near 0.6 and the opposite is true.

    python recheck.py                 # 40 windows, drawn from both sittings
    python recheck.py --n 30 --new    # only the ones added recently
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "manifest.json"
LABELS = HERE / "labels.json"
OUT = HERE / "recheck.json"

KEY_ON = {ord("o"), 81, 2424832}
KEY_OFF = {ord("f"), 83, 2555904}
KEY_UNCLEAR = {ord("u")}
KEY_SKIP = {ord("s")}
KEY_QUIT = {ord("q"), 27}

SHEET_COLUMNS = 6
TILE_PX = 190


def contact_sheet(window: dict, i: int, total: int, done: int):
    """The same grid the labelling tool shows, with no prior answer on it."""
    frames = []
    for name in window.get("frames", [])[:SHEET_COLUMNS * 3]:
        img = cv2.imread(str(HERE / "images" / name))
        if img is not None:
            h = TILE_PX
            w = int(img.shape[1] * h / img.shape[0])
            frames.append(cv2.resize(img, (w, h)))
    if not frames:
        return None

    width = max(f.shape[1] for f in frames)
    frames = [cv2.copyMakeBorder(f, 0, 0, 0, width - f.shape[1],
                                 cv2.BORDER_CONSTANT, value=(18, 18, 18))
              for f in frames]
    rows = []
    for start in range(0, len(frames), SHEET_COLUMNS):
        row = frames[start:start + SHEET_COLUMNS]
        while len(row) < SHEET_COLUMNS:
            row.append(row[0] * 0)
        rows.append(cv2.hconcat(row))
    sheet = cv2.vconcat(rows)

    panel = cv2.copyMakeBorder(sheet, 96, 56, 20, 20, cv2.BORDER_CONSTANT,
                               value=(18, 18, 18))
    cv2.putText(panel, f"RE-CHECK  {i + 1} / {total}   recorded {done}",
                (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 235, 235), 1,
                cv2.LINE_AA)
    cv2.putText(panel, "Judge from the frames alone. Your earlier answer is "
                       "deliberately not shown.",
                (24, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (150, 190, 210), 1,
                cv2.LINE_AA)
    cv2.putText(panel, "o on task    f off task    u unclear    s skip    "
                       "q save + quit",
                (24, panel.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                (150, 190, 210), 1, cv2.LINE_AA)
    return panel


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=40,
                    help="how many to re-judge (40 is enough to tell 0.9 "
                         "from 0.6)")
    ap.add_argument("--new", action="store_true",
                    help="draw only from windows whose labels train to chance")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not LABELS.exists():
        print(f"{LABELS} not found -- nothing to re-check.")
        return 1

    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    windows = {w["window_id"]: w
               for w in json.loads(MANIFEST.read_text(encoding="utf-8"))}

    pool = [wid for wid, e in labels.items()
            if e["label"] in ("on", "off") and wid in windows]
    if args.new:
        # Ordered by when they were added: dict order in labels.json follows
        # the sitting they were written in.
        pool = pool[179:] or pool
    random.Random(args.seed).shuffle(pool)
    pool = pool[:args.n]

    store = {}
    if OUT.exists():
        store = json.loads(OUT.read_text(encoding="utf-8"))
        print(f"resuming: {len(store)} already re-checked")
    print(f"{len(pool)} windows to re-judge -> {OUT.name}\n")

    i = 0
    try:
        while 0 <= i < len(pool):
            wid = pool[i]
            if wid in store:
                i += 1
                continue
            sheet = contact_sheet(windows[wid], i, len(pool), len(store))
            if sheet is None:
                i += 1
                continue
            cv2.imshow("re-check", sheet)
            key = cv2.waitKey(0) & 0xFFFFFF

            if key in KEY_QUIT:
                break
            if key in KEY_SKIP:
                i += 1
                continue
            if key in KEY_ON:
                store[wid] = "on"
            elif key in KEY_OFF:
                store[wid] = "off"
            elif key in KEY_UNCLEAR:
                store[wid] = "unclear"
            else:
                continue
            OUT.write_text(json.dumps(store, indent=1), encoding="utf-8")
            i += 1
    finally:
        cv2.destroyAllWindows()

    OUT.write_text(json.dumps(store, indent=1), encoding="utf-8")
    print(f"\nsaved {len(store)} re-checks to {OUT}\n")

    both = [(labels[w]["label"], store[w]) for w in store
            if labels.get(w, {}).get("label") in ("on", "off")
            and store[w] in ("on", "off")]
    if len(both) < 10:
        print("  Too few decided both times to report agreement.")
        return 0

    agree = sum(1 for a, b in both if a == b)
    rate = agree / len(both)
    # Cohen's kappa: agreement above what the two label rates would produce
    # by chance alone, which matters when one answer dominates.
    n = len(both)
    p_on_a = sum(1 for a, _ in both if a == "on") / n
    p_on_b = sum(1 for _, b in both if b == "on") / n
    expected = p_on_a * p_on_b + (1 - p_on_a) * (1 - p_on_b)
    kappa = (rate - expected) / (1 - expected) if expected < 1 else float("nan")

    print(f"  agreed with yourself on {agree} of {n}  ({rate:.1%})")
    print(f"  Cohen's kappa: {kappa:.3f}")
    print()
    if rate >= 0.85:
        print("  The labels are consistent. The ceiling is high, so a model")
        print("  scoring 0.56 is the model's limitation, not the labels'.")
    elif rate >= 0.70:
        print("  Moderate. A model cannot exceed this, so an accuracy figure")
        print("  near it is close to the ceiling rather than underperforming.")
    else:
        print("  Low. These labels do not encode a repeatable judgement, so")
        print("  they cannot train a model and no accuracy measured against")
        print("  them means much. Re-labelling carefully is worth more than")
        print("  any change to the model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
