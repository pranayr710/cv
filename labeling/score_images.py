"""Score engagement images on an ordinal scale, for calibration.

Standalone: needs this folder, ``images/calibration/`` and two pip packages
(opencv-python, numpy). Nothing else from the repository.

Why five points and not ten. The goal is a 1-10 score, but people are not
reliable at ten levels -- asked twice, the same rater routinely says 6 then 7,
and that disagreement is noise presented as resolution. Every published dataset
in this area uses fewer: OUC-CGE three, DAiSEE four, DIPSER five. So the
judgement is made at five, where a person can actually hold the distinction,
and the 1-10 score comes from the model's calibrated probability afterwards.
Scoring at five and mapping up is more defensible than pretending to ten.

What is deliberately hidden: the image's folder label. Every one of these came
from an Engagement or Disengagement directory, and the point of scoring them is
to check whether the model's confidence tracks human judgement. Seeing the
label first would anchor the score to the very thing being validated.

Controls
    1  not engaged at all        4  engaged
    2  mostly disengaged         5  clearly, fully engaged
    3  borderline / cannot tell
    b  back one image            s  skip
    q  save and quit

Progress saves after every keypress, so quitting loses nothing.

    python score_images.py
    python score_images.py --who priya
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
IMAGES = HERE / "images" / "calibration"
MANIFEST = IMAGES / "manifest.json"

#: Display size for the image being scored.
VIEW_PX = 520

SCALE = {
    ord("1"): 1, ord("2"): 2, ord("3"): 3, ord("4"): 4, ord("5"): 5,
}
KEY_BACK = {ord("b")}
KEY_SKIP = {ord("s")}
KEY_QUIT = {ord("q"), 27}

ANCHORS = {
    1: "not engaged at all",
    2: "mostly disengaged",
    3: "borderline / cannot tell",
    4: "engaged",
    5: "clearly, fully engaged",
}



def write_outputs(store: dict, json_path: Path) -> None:
    """Save scores as JSON and as a spreadsheet-friendly CSV.

    JSON is what the calibration step reads and what makes a session
    resumable. The CSV exists so the scores can be opened in Excel and eyeballed
    without a parser -- checking your own distribution part-way through is how
    you notice you have drifted into scoring everything a 3.
    """
    json_path.write_text(json.dumps(store, indent=1), encoding="utf-8")

    csv_path = json_path.with_suffix(".csv")
    columns = ["file", "score_5", "folder_label", "age_band", "gender", "source"]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        for name, entry in sorted(store.items()):
            writer.writerow({"file": name, **{c: entry.get(c, "")
                                              for c in columns[1:]}})


def draw(entry, idx: int, total: int, scored: int):
    """Render one image with the scale visible and the label hidden."""
    img = cv2.imread(str(IMAGES / entry["file"]))
    if img is None:
        return None
    scale = VIEW_PX / max(img.shape[:2])
    img = cv2.resize(img, None, fx=scale, fy=scale,
                     interpolation=cv2.INTER_CUBIC if scale > 1
                     else cv2.INTER_AREA)
    panel = cv2.copyMakeBorder(img, 78, 148, 24, 24, cv2.BORDER_CONSTANT,
                               value=(28, 22, 18))
    cv2.putText(panel, "How engaged does this person look?", (24, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.64, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, f"{idx + 1} of {total}   ({scored} scored)", (24, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 190, 210), 1, cv2.LINE_AA)

    top = panel.shape[0] - 132
    for value in range(1, 6):
        y = top + (value - 1) * 22
        cv2.putText(panel, f"{value}", (28, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, (120, 220, 140), 2, cv2.LINE_AA)
        cv2.putText(panel, ANCHORS[value], (56, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.46, (225, 225, 225), 1, cv2.LINE_AA)
    cv2.putText(panel, "b back    s skip    q save + quit",
                (28, panel.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                (150, 190, 210), 1, cv2.LINE_AA)
    return panel


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--who", default=None,
                    help="scorer name; writes scores_<who>.json so several "
                         "people can score the same set independently")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if not MANIFEST.exists():
        print(f"{MANIFEST} not found. Run tools/sample_calibration_set.py "
              "first.")
        return 1

    out = HERE / (f"scores_{args.who}.json" if args.who else "scores.json")
    entries = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if args.limit:
        entries = entries[:args.limit]

    store = {}
    if out.exists():
        store = json.loads(out.read_text(encoding="utf-8"))
        print(f"resuming: {len(store)} already scored")
    print(f"{len(entries)} images to score -> {out.name}")

    i = 0
    try:
        while 0 <= i < len(entries):
            entry = entries[i]
            if entry["file"] in store:
                i += 1
                continue
            panel = draw(entry, i, len(entries), len(store))
            if panel is None:
                i += 1
                continue
            cv2.imshow("score engagement", panel)
            key = cv2.waitKey(0) & 0xFFFFFF

            if key in KEY_QUIT:
                break
            if key in KEY_BACK:
                i = max(0, i - 1)
                continue
            if key in KEY_SKIP:
                i += 1
                continue
            if key not in SCALE:
                continue

            store[entry["file"]] = {
                "score_5": SCALE[key],
                # Carried for the calibration step, never shown before scoring.
                "folder_label": entry["folder_label"],
                "age_band": entry["age_band"],
                "gender": entry["gender"],
                "source": entry["source"],
            }
            write_outputs(store, out)
            i += 1
    finally:
        cv2.destroyAllWindows()

    counts: dict[int, int] = {}
    agree = disagree = 0
    for v in store.values():
        counts[v["score_5"]] = counts.get(v["score_5"], 0) + 1
        if v["score_5"] == 3:
            continue
        human_engaged = v["score_5"] >= 4
        folder_engaged = v["folder_label"] == "Engagement"
        if human_engaged == folder_engaged:
            agree += 1
        else:
            disagree += 1

    print(f"\nsaved {len(store)} scores to {out}")
    for value in sorted(counts):
        print(f"  {value}  {ANCHORS[value]:<26} {counts[value]}")
    decided = agree + disagree
    if decided:
        print(f"\n  agreement with the folder label: {agree}/{decided} "
              f"({100 * agree / decided:.0f}%)")
        print("  (3s excluded -- they are the rater declining to decide)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
