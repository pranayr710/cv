"""Local, keypress-driven labelling tool for the expression-validation study.

Two people run this independently, BLIND to each other's labels and to the
model's own prediction (the manifest's model_label column is never shown here)
-- that blinding is what makes the inter-rater agreement computed afterwards
mean anything. See docs/LITERATURE_REVIEW.md section 4 for why the protocol
needs two independent raters and Cohen's kappa before ever comparing to the
model: if two humans can't agree on a crop, the model can't be faulted for
missing a signal that isn't reliably there in the first place (Whitehill et
al. 2014's validation template).

Runs entirely locally -- an OpenCV window on your own screen. Nothing here
uploads a face crop anywhere.

Controls:
    h  happy      s  sad      n  neutral      u  unclear/unlabelable
    b  back (redo the previous crop)
    q  save and quit (resumable -- rerun with the same --labeller to continue)

Run:
    python -m tools.label_expressions --labeller you
    python -m tools.label_expressions --labeller rater2
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

KEYS = {ord("h"): "happy", ord("s"): "sad", ord("n"): "neutral", ord("u"): "unclear"}


def _load_manifest(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _load_existing(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return {row["crop_id"]: row["label"] for row in csv.DictReader(fh)}


def label(out_dir: Path, labeller: str) -> None:
    import cv2

    manifest = _load_manifest(out_dir / "manifest.csv")
    if not manifest:
        raise SystemExit(f"Empty manifest at {out_dir / 'manifest.csv'}")

    labels_path = out_dir / f"labels_{labeller}.csv"
    done = _load_existing(labels_path)
    todo = [row for row in manifest if row["crop_id"] not in done]

    if not todo:
        print(f"{labeller} has already labelled all {len(manifest)} crops.")
        return

    print(f"{len(done)}/{len(manifest)} already labelled by {labeller!r}. "
          f"{len(todo)} remaining.")
    print("h=happy  s=sad  n=neutral  u=unclear  b=back  q=save&quit")

    history: list[str] = []
    i = 0
    window = "expression labelling"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    while i < len(todo):
        row = todo[i]
        img_path = out_dir / "crops" / f"{row['crop_id']}.png"
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"missing crop file {img_path}, skipping")
            i += 1
            continue

        remaining = len(todo) - i
        cv2.putText(
            img, f"{remaining} left  [h/s/n/u, b=back, q=quit]",
            (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 0), 1,
        )
        cv2.imshow(window, img)
        key = cv2.waitKey(0) & 0xFF

        if key == ord("q"):
            break
        if key == ord("b"):
            if history:
                prev_id = history.pop()
                done.pop(prev_id, None)
                i = max(0, i - 1)
            continue
        if key in KEYS:
            done[row["crop_id"]] = KEYS[key]
            history.append(row["crop_id"])
            i += 1
            continue
        # Any other key: ignore and redraw the same crop.

    cv2.destroyWindow(window)

    with labels_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["crop_id", "label"])
        for row in manifest:
            if row["crop_id"] in done:
                writer.writerow([row["crop_id"], done[row["crop_id"]]])

    print(f"saved {len(done)}/{len(manifest)} labels to {labels_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", default="outputs/expression_labels")
    parser.add_argument("--labeller", required=True)
    args = parser.parse_args()
    label(Path(args.dir), args.labeller)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
