"""Face-only detection overlay -- Stage 1 visual check for sir's review feedback.

Sir's feedback on the first review: (1) the demo image marked some
persons/objects and not others with no explanation, and (2) he wants boxes
drawn ONLY on faces, not the whole body, and wants Part 1 (perception)
polished before anything downstream is touched.

This script does exactly that: run the real Stage 1 detector + face
analyzer on a classroom image and draw ONLY the matched face boxes -- no
person boxes, no object boxes. It also prints the honest counts so the
"why weren't all of them marked" question has a real number attached
instead of a guess.

Usage:
    python -m tools.render_faces --image dataset/img01.jpg
    python -m tools.render_faces --image dataset/img01.jpg --out outputs/img01_faces.jpg
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def render(image_path: str | Path, out_path: str | Path) -> None:
    import cv2

    from backend.detection import Detector
    from backend.face import FaceAnalyzer

    src = Path(image_path)
    if not src.is_file():
        raise FileNotFoundError(f"Image not found: {src}")

    frame = cv2.imread(str(src))
    if frame is None:
        raise RuntimeError(f"OpenCV could not read image: {src}")

    detector = Detector()
    persons, _objects = detector.detect(frame)

    with FaceAnalyzer() as analyzer:
        faces = analyzer.analyze(frame, [p.bbox for p in persons])

    out = frame.copy()
    n_faces = 0
    for face in faces:
        if face.face_bbox is None:
            continue
        n_faces += 1
        x, y, w, h = face.face_bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 220, 0), 2)

    n_persons = len(persons)
    pct = (n_faces / n_persons * 100.0) if n_persons else 0.0
    summary = (
        f"persons detected: {n_persons}  |  faces matched: {n_faces} "
        f"({pct:.0f}% of detected persons)"
    )
    cv2.putText(
        out, summary, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 0), 2,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out)

    print(summary)
    print(f"Saved: {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="dataset/img01.jpg")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    out = args.out or f"outputs/{Path(args.image).stem}_faces.jpg"
    render(args.image, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
