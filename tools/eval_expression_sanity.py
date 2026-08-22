"""Sanity-check ExpressionRecognizer against FER2013's own happy/sad/neutral labels.

**What this proves, and what it explicitly does not.**

Before this, expression accuracy was completely unmeasured -- coverage and
confidence were reported, correctness never was, because this project has no
labelled classroom expression data. FER2013 (28,709 labelled faces, downloaded
from https://huggingface.co/datasets/SmolVEncoder/fer2013, Apache-2.0) provides
real happy/sad/neutral ground truth, so this is the first time the model's
*correctness* is measured against any label at all rather than only its
coverage and confidence.

It is a sanity check, not a validation, and the difference matters:

* FER2013 faces are 48x48 grayscale web photos, mostly Western-skewed, posed or
  candid stock/movie stills -- not classroom footage, not this project's real
  population, and not the resolution or lighting this pipeline actually sees.
* If the model performs badly HERE, it should not be trusted on classroom
  footage either -- this is a necessary-but-not-sufficient bar.
* If it performs well here, that says only "the model works on its own
  training distribution restated in a fresh split" -- it says NOTHING about
  accuracy on South Asian classroom faces at 20-70px, which remains completely
  unmeasured. See backend/expression.py's "Known unvalidated for this
  population" section; this script does not close that gap.

Run:
    python -m tools.eval_expression_sanity
    python -m tools.eval_expression_sanity --root dataset/fer2013_sample
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


def evaluate(root: Path) -> None:
    import cv2

    from backend.expression import ExpressionRecognizer

    if not root.is_dir():
        raise FileNotFoundError(
            f"{root} not found. Extract a sample first, e.g. from FER2013 "
            f"(https://huggingface.co/datasets/SmolVEncoder/fer2013)."
        )

    labels = sorted(d.name for d in root.iterdir() if d.is_dir())
    recognizer = ExpressionRecognizer()

    confusion: Counter[tuple[str, str]] = Counter()
    total = correct = uncertain = 0

    for true_label in labels:
        for img_path in sorted((root / true_label).glob("*.jpg")):
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            h, w = frame.shape[:2]
            # These crops ARE the face already (FER2013 is pre-cropped), so the
            # whole frame is passed as the face box rather than re-detecting.
            result = recognizer.classify(frame, [(0, 0, w, h)])[0]
            total += 1
            if result is None:
                continue
            if result.uncertain:
                uncertain += 1
            confusion[(true_label, result.label)] += 1
            if result.label == true_label:
                correct += 1

    print(f"\n{root}: {total} labelled crops (FER2013, NOT this project's "
          f"population -- see module docstring)\n")
    print(f"overall accuracy   : {correct / total * 100:.1f}% ({correct}/{total})")
    print(f"abstained          : {uncertain / total * 100:.1f}% ({uncertain}/{total})")

    header = "true/predicted"
    print(f"\n{header:<14}", end="")
    for pred in (*labels, "uncertain"):
        print(f"{pred:>11}", end="")
    print()
    for true_label in labels:
        row_total = sum(v for (t, _p), v in confusion.items() if t == true_label)
        print(f"{true_label:<14}", end="")
        for pred in (*labels, "uncertain"):
            count = confusion.get((true_label, pred), 0)
            share = count / row_total * 100 if row_total else 0.0
            print(f"{count:>4} ({share:>3.0f}%)", end="")
        print()

    print("\nPer-class recall:")
    for true_label in labels:
        row_total = sum(v for (t, _p), v in confusion.items() if t == true_label)
        hits = confusion.get((true_label, true_label), 0)
        print(f"  {true_label:<10}{hits}/{row_total}  "
              f"({hits / row_total * 100:.1f}%)" if row_total else "  n/a")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="dataset/fer2013_sample")
    args = parser.parse_args()
    evaluate(Path(args.root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
