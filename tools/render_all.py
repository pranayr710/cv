"""Render every Part 1 signal onto real frames, for eyeball verification.

Numbers hide things that a picture does not. Two of the worst bugs in this
project were caught by rendering and looking, not by a metric: the gaze labels
that were 84% "right" because the camera is corner-mounted (section 14), and
book detections that at a lower threshold boxed a student's head (section 11.3).

So every signal gets drawn on real frames before it is trusted:

* green box   -- student found by YOLO
* amber box   -- student recovered from their face (estimated body box)
* cyan box    -- face
* text        -- gaze / expression / behaviour per student

Ground truth is drawn in red when a matching label file exists, so predictions
can be compared against human labels rather than judged on plausibility.

Run:
    python -m tools.render_all --images dataset/behaviour/val/images --limit 6
    python -m tools.render_all --images dataset --limit 4 --out outputs/render
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

GREEN = (0, 220, 0)
AMBER = (0, 170, 255)
CYAN = (255, 220, 0)
RED = (0, 0, 235)
WHITE = (255, 255, 255)


def render(images_dir: Path, out_dir: Path, limit: int) -> None:
    import cv2
    import yaml

    from backend.detection import Detector
    from backend.expression import ExpressionRecognizer
    from backend.face import FaceAnalyzer
    from backend.headpose import HeadPoseEstimator
    from backend.students import augment_persons

    try:
        from backend.behaviour import BehaviourClassifier

        behaviour = BehaviourClassifier()
    except (FileNotFoundError, ImportError) as exc:
        print(f"behaviour disabled: {exc}")
        behaviour = None

    # Ground truth, if this directory is a labelled split.
    label_dir = images_dir.parent / "labels"
    names: list[str] = []
    data_yaml = images_dir.parent.parent / "data.yaml"
    if data_yaml.is_file():
        with data_yaml.open(encoding="utf-8") as fh:
            names = list(yaml.safe_load(fh).get("names", []))

    detector = Detector()
    headpose = HeadPoseEstimator()
    expression = ExpressionRecognizer()

    paths = sorted(images_dir.glob("*.jpg"))[:limit]
    if not paths:
        raise FileNotFoundError(f"No .jpg under {images_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    with FaceAnalyzer() as analyzer:
        for path in paths:
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            img_h, img_w = frame.shape[:2]
            canvas = frame.copy()

            persons, objects = detector.detect(frame)
            faces_raw = analyzer.detect_faces(frame)
            students = augment_persons(persons, faces_raw, frame.shape[:2])
            boxes = [s.bbox for s in students]
            faces = analyzer.analyze(frame, boxes, faces_raw)
            poses = headpose.estimate(frame, [f.face_bbox for f in faces])
            exprs = expression.classify(
                frame, [f.face_bbox for f in faces], [f.kps for f in faces]
            )
            behs = (
                behaviour.classify(frame, boxes)
                if behaviour is not None
                else [None] * len(students)
            )

            # Ground truth first, so predictions draw over it.
            gt_count = 0
            label_file = label_dir / f"{path.stem}.txt"
            if label_file.is_file() and names:
                for line in label_file.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    cid = int(float(parts[0]))
                    cx, cy, bw, bh = (float(v) for v in parts[1:5])
                    x = int((cx - bw / 2) * img_w)
                    y = int((cy - bh / 2) * img_h)
                    w = int(bw * img_w)
                    h = int(bh * img_h)
                    gt_count += 1
                    cv2.rectangle(canvas, (x, y), (x + w, y + h), RED, 2)
                    label = names[cid] if cid < len(names) else str(cid)
                    cv2.putText(canvas, f"GT:{label}", (x, y + h + 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.42, RED, 1)

            for student, face, pose, expr, beh in zip(
                students, faces, poses, exprs, behs
            ):
                x, y, w, h = student.bbox
                colour = GREEN if student.source == "yolo" else AMBER
                cv2.rectangle(canvas, (x, y), (x + w, y + h), colour, 2)

                if face.face_bbox is not None:
                    fx, fy, fw, fh = face.face_bbox
                    cv2.rectangle(canvas, (fx, fy), (fx + fw, fy + fh), CYAN, 1)

                # Stack the per-student readings above their box.
                lines = []
                if pose is not None:
                    lines.append(f"gaze:{pose.gaze_label}")
                if expr is not None:
                    lines.append(f"expr:{expr.label} {expr.confidence:.2f}")
                if beh is not None:
                    tag = "!" if beh.reliability == "weak" else ""
                    lines.append(f"beh:{beh.label}{tag} {beh.confidence:.2f}")
                for i, text in enumerate(lines):
                    cv2.putText(canvas, text, (x, max(12, y - 4 - i * 13)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.40, WHITE, 1)

            for obj in objects:
                ox, oy, ow, oh = obj.bbox
                cv2.rectangle(canvas, (ox, oy), (ox + ow, oy + oh), (200, 200, 0), 1)
                cv2.putText(canvas, obj.cls, (ox, oy - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 0), 1)

            seeded = sum(1 for s in students if s.source == "face_seeded")
            header = (
                f"{path.name}  students {len(students)} (seeded {seeded})  "
                f"faces {sum(1 for f in faces if f.face_bbox)}  "
                f"expr {sum(1 for e in exprs if e)}  "
                f"beh {sum(1 for b in behs if b)}"
                + (f"  GT {gt_count}" if gt_count else "")
            )
            cv2.rectangle(canvas, (0, 0), (img_w, 34), (0, 0, 0), -1)
            cv2.putText(canvas, header, (10, 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)

            dest = out_dir / f"{path.stem}_all.jpg"
            cv2.imwrite(str(dest), canvas)
            print(header)
            print(f"  -> {dest}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", default="dataset/behaviour/val/images")
    parser.add_argument("--out", default="outputs/render")
    parser.add_argument("--limit", type=int, default=6)
    args = parser.parse_args()
    render(Path(args.images), ROOT / args.out, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
