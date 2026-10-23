"""Write a labelled export of the whole image set, with provenance per row.

The 350 hand-scored images give an ordinal scale -- how engaged, not merely
whether -- and this applies that scale to the remaining 12,232 so the dataset
carries one number per image rather than a binary folder name.

**What each label is, and is not.** The Engagement/Disengagement folder name
is a human judgement made by the dataset's authors. The 1-5 score on 350
images is a human judgement made here. The 1-5 score on the other 12,232 is a
model's output, calibrated against those 350. The export says which is which
on every row, because a reader who cannot tell them apart will eventually
treat a prediction as a label.

**Why the scores are not refit on.** Training on them would teach the model
what it already believes, sharpen whatever it has wrong, and return the higher
confidence as progress. They are a deliverable, not training data.

    python tools/data/export_image_labels.py
    python tools/data/export_image_labels.py --out outputs/labelled_images.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

SCORES = Path("outputs/image_engagement_scores.csv")
HUMAN = Path("labeling/scores.csv")
DEFAULT_OUT = Path("outputs/labelled_images.csv")

#: Written beside the export so the file is readable without this script.
README = """Image engagement labels
=======================

One row per image in the deduplicated set (12,582 images).

Columns
-------
file            image filename
folder_label    Engagement or Disengagement. A HUMAN label, made by the
                dataset's authors (Chate & Patil, Zenodo 16731734, CC BY 4.0).
                Present on every row.
age_band        5-10, 11-15, 16-21 or 22-Above Years, from the dataset's own
                directory structure.
gender          Boy or Girl, likewise.
score_5         Engagement on a 1-5 scale.
score_10        The same, rescaled to 1-10 for reporting.
label_source    "human" for the 350 scored by hand here; "model" for the
                12,232 the calibrated model scored.
path            full path within the repository.

How the scores were produced
----------------------------
A classifier was trained on the folder labels, then the map from its
probability to a human 1-5 judgement was fitted on 350 images scored by hand.
Held out over near-duplicate groups, that route reaches Spearman 0.515 against
the human scores. Fitting a regressor directly on the 350 instead reaches
0.353, so the indirect route is the one used.

A limitation to read before using the scores
--------------------------------------------
76% of the images land on just two values (5 and 8 out of 10). The 350 hand
scores are spread evenly across all five levels by comparison.

The cause is that the underlying classifier was trained on a binary target,
so its probabilities pile up near 0 and 1, and the calibration can only
stretch them so far. In practice these scores are closer to a confident
binary judgement than to a true 1-10 ranking.

Use them to separate engaged from disengaged. Do not read the difference
between a 7 and an 8 as meaningful -- that distinction is not supported by
how they were produced.


What these numbers are worth
----------------------------
The folder labels are human but not ours; agreement between them and the 350
hand scores is 84%, so roughly one image in six is filed differently from how
it was judged here. No model trained on this set can be expected to exceed
that.

The model scores are predictions. They are reported, never trained on:
refitting a model on its own output sharpens its existing mistakes and returns
the resulting confidence as though it were progress.
"""


def parse_path(path: str) -> tuple[str, str]:
    """Age band and gender, read from the dataset's directory structure."""
    parts = Path(path).parts
    age = next((p for p in parts if "Years" in p), "")
    gender = next((p for p in parts if p in ("Boy", "Girl")), "")
    return age, gender


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--json", action="store_true",
                    help="also write the same rows as JSON")
    args = ap.parse_args()

    if not SCORES.exists():
        print(f"{SCORES} not found. Run tools/data/score_all_images.py first.")
        return 1

    scored = list(csv.DictReader(SCORES.open(encoding="utf-8")))

    # The hand scores are the authority wherever they exist, so they are read
    # from their own file rather than trusted to have survived a round trip.
    human: dict[str, float] = {}
    if HUMAN.exists():
        for row in csv.DictReader(HUMAN.open(encoding="utf-8")):
            if row.get("score_5"):
                human[Path(row["source"]).name] = float(row["score_5"])

    rows = []
    for entry in scored:
        name = Path(entry["path"]).name
        age, gender = parse_path(entry["path"])
        by_hand = name in human
        score_5 = human[name] if by_hand else float(entry["score_5"])
        rows.append({
            "file": name,
            "folder_label": entry["folder_label"],
            "age_band": age,
            "gender": gender,
            "score_5": round(score_5, 3),
            "score_10": max(1, min(10, round(score_5 * 2))),
            "label_source": "human" if by_hand else "model",
            "path": entry["path"],
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    readme = args.out.with_name(args.out.stem + "_README.txt")
    readme.write_text(README, encoding="utf-8")

    if args.json:
        args.out.with_suffix(".json").write_text(
            json.dumps(rows, indent=1), encoding="utf-8")

    source = Counter(r["label_source"] for r in rows)
    folder = Counter(r["folder_label"] for r in rows)
    spread = Counter(r["score_10"] for r in rows)

    print(f"{len(rows):,} images -> {args.out}")
    print(f"  {source['human']:,} scored by hand, "
          f"{source['model']:,} scored by the model")
    print(f"  folder labels: {dict(folder)}\n")
    print("  score_10 distribution")
    widest = max(spread.values())
    for value in range(1, 11):
        count = spread.get(value, 0)
        bar = "#" * int(40 * count / widest) if count else ""
        print(f"    {value:>2}  {count:>6}  {bar}")

    agree = sum(1 for r in rows
                if (r["score_10"] >= 6) == (r["folder_label"] == "Engagement"))
    print(f"\n  agrees with the folder label: {agree:,}/{len(rows):,} "
          f"({agree / len(rows):.1%})")
    print(f"  notes written to {readme.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
