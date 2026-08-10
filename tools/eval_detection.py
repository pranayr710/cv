"""Measure student-detection recall against hand-labelled ground truth.

Until now every detection number in this project was self-measured -- counted by
eye on one image, or compared against another model's output. This is the first
evaluation against **human labels**, so it is the first one that can state
precision and recall rather than "more boxes than before".

Reports, over a labelled YOLO-format dataset:

1. **Student detection** -- precision/recall/F1 of the pipeline's students
   (YOLO bodies + faces seeded by :mod:`backend.students`) against the ground
   truth boxes, matched greedily by IoU. Broken out by ``source`` so the
   contribution of face seeding is visible, and by ground-truth behaviour class
   so it is clear which behaviours get missed.
2. **Face coverage** -- what fraction of *ground-truth* students SCRFD finds a
   face for. Distinct from (1): a student can be detected but faceless.
3. **Writing signal** -- how well the book-proximity proxy in
   :mod:`backend.attention` agrees with the ground-truth ``write``/``read``
   labels. This is the question raised in review: does a detected book actually
   predict a writing student?

Sampling is **stratified by source clip** with ``--per-clip``. The dataset's 481
frames come from only 11 videos, so consecutive frames are near-duplicates;
taking the first N images would sample one classroom, not the dataset.

Run:
    python -m tools.eval_detection --root dataset/dataset --per-clip 12
    python -m tools.eval_detection --root dataset/dataset --all
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter, defaultdict
from pathlib import Path

from tools.analyse_labelled import _source_clip

logger = logging.getLogger(__name__)

Bbox = tuple[float, float, float, float]


def _iou(a: Bbox, b: Bbox) -> float:
    """Intersection-over-union of two ``(x, y, w, h)`` boxes."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    inter_w = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    inter_h = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _centre_in(inner: Bbox, outer: Bbox) -> bool:
    """Whether ``inner``'s centre point lies inside ``outer``."""
    cx = inner[0] + inner[2] / 2
    cy = inner[1] + inner[3] / 2
    return outer[0] <= cx <= outer[0] + outer[2] and outer[1] <= cy <= outer[1] + outer[3]


def _match_score(det: Bbox, gt: Bbox, mode: str) -> float:
    """Score a (detection, ground-truth) pair for greedy matching.

    Args:
        det: The pipeline's student box.
        gt: The ground-truth student box.
        mode: ``"centre"`` (default) requires each box's centre to fall inside
            the other, which is robust to the two using different box
            conventions; ``"iou"`` is plain IoU.

    Returns:
        A score in ``[0, 1]``; ``0.0`` means "not a match at all". Under
        ``"centre"``, a mutual-centre pair scores ``0.5 + IoU / 2`` so genuine
        pairs always outrank non-pairs while better-overlapping pairs are still
        preferred.
    """
    iou = _iou(det, gt)
    if mode == "iou":
        return iou
    if _centre_in(det, gt) and _centre_in(gt, det):
        return 0.5 + iou / 2
    return 0.0


def _containment(inner: Bbox, outer: Bbox) -> float:
    """Fraction of ``inner``'s area inside ``outer``."""
    ix, iy, iw, ih = inner
    ox, oy, ow, oh = outer
    inter_w = max(0.0, min(ix + iw, ox + ow) - max(ix, ox))
    inter_h = max(0.0, min(iy + ih, oy + oh) - max(iy, oy))
    area = iw * ih
    return (inter_w * inter_h / area) if area > 0 else 0.0


def _load_labels(path: Path, img_w: int, img_h: int, names: list[str]):
    """Read one YOLO label file into ``[(cls_name, (x, y, w, h)), ...]`` pixels.

    Args:
        path: The ``.txt`` label file.
        img_w: Image width in pixels.
        img_h: Image height in pixels.
        names: Class-id -> name mapping from ``data.yaml``.

    Returns:
        Ground-truth boxes in absolute pixel ``(x, y, w, h)`` form.
    """
    out = []
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls_id = int(float(parts[0]))
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        x = (cx - bw / 2) * img_w
        y = (cy - bh / 2) * img_h
        name = names[cls_id] if cls_id < len(names) else f"id{cls_id}"
        out.append((name, (x, y, bw * img_w, bh * img_h)))
    return out


def _sample(root: Path, per_clip: int | None, take_all: bool) -> list[Path]:
    """Pick images, stratified by source clip unless ``take_all``."""
    images = sorted((root / "images").glob("*.jpg"))
    if take_all or per_clip is None:
        return images
    by_clip: defaultdict[str, list[Path]] = defaultdict(list)
    for path in images:
        by_clip[_source_clip(path.stem)].append(path)
    picked: list[Path] = []
    for clip_images in by_clip.values():
        # Evenly spaced through the clip, not the first N, so the sample spans
        # the whole video rather than one moment in it.
        step = max(1, len(clip_images) // per_clip)
        picked.extend(clip_images[::step][:per_clip])
    return sorted(picked)


def evaluate(root: Path, per_clip: int | None, take_all: bool, iou_thr: float,
             match: str = "centre") -> None:
    import cv2
    import yaml

    from backend.attention import classify_frame
    from backend.detection import Detector
    from backend.face import FaceAnalyzer
    from backend.students import augment_persons

    with (root / "data.yaml").open(encoding="utf-8") as fh:
        names = list(yaml.safe_load(fh).get("names", []))

    images = _sample(root, per_clip, take_all)
    print(f"evaluating {len(images)} images from "
          f"{len({_source_clip(p.stem) for p in images})} clips, "
          f"match={match} threshold={iou_thr}")

    detector = Detector()

    tp = fp = fn = 0
    tp_by_source: Counter[str] = Counter()
    gt_by_class: Counter[str] = Counter()
    hit_by_class: Counter[str] = Counter()
    face_by_class: Counter[str] = Counter()
    # writing-signal confusion, over GT students the pipeline detected
    writing_tp = writing_fp = writing_fn = writing_tn = 0

    with FaceAnalyzer() as analyzer:
        for idx, img_path in enumerate(images, 1):
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            img_h, img_w = frame.shape[:2]
            gt = _load_labels(
                root / "labels" / f"{img_path.stem}.txt", img_w, img_h, names
            )

            persons, objects = detector.detect(frame)
            detected_faces = analyzer.detect_faces(frame)
            students = augment_persons(
                persons, detected_faces, frame.shape[:2]
            )
            faces = analyzer.analyze(
                frame, [s.bbox for s in students], detected_faces
            )

            # Greedy matching, best pair first, one-to-one.
            #
            # IoU alone understates recall badly here, and the reason is an
            # annotation-convention mismatch rather than a detection failure:
            # this dataset's boxes are tight head+torso regions, while YOLO
            # emits full-body boxes. Rendering both on one frame showed 11 GT
            # students and 11 YOLO students -- the same 11 people -- yet only
            # 7 pairs cleared IoU 0.5.
            #
            # So "did we find this student" is scored by mutual centre
            # containment (each box's centre inside the other) with IoU as a
            # weaker tiebreak. --match iou restores pure IoU scoring for
            # comparison against published numbers, which use that convention.
            pairs = sorted(
                (
                    (_match_score(tuple(s.bbox), g_box, match), si, gi)
                    for si, s in enumerate(students)
                    for gi, (_g_cls, g_box) in enumerate(gt)
                ),
                reverse=True,
            )
            used_s: set[int] = set()
            used_g: set[int] = set()
            matched: dict[int, int] = {}  # gt index -> student index
            for score, si, gi in pairs:
                if score <= 0.0 or si in used_s or gi in used_g:
                    continue
                if match == "iou" and score < iou_thr:
                    continue
                used_s.add(si)
                used_g.add(gi)
                matched[gi] = si

            tp += len(matched)
            fp += len(students) - len(matched)
            fn += len(gt) - len(matched)

            for gi, (g_cls, g_box) in enumerate(gt):
                gt_by_class[g_cls] += 1
                if gi in matched:
                    hit_by_class[g_cls] += 1
                    tp_by_source[students[matched[gi]].source] += 1
                # Face coverage is measured against the GT box directly, so a
                # student the body detector missed is still counted fairly.
                if any(
                    _containment(d.bbox, g_box) >= 0.5 for d in detected_faces
                ):
                    face_by_class[g_cls] += 1

                # Writing signal: only judged where the pipeline produced a
                # student with a head-pose reading, since the proxy needs one.
                if gi not in matched:
                    continue
                si = matched[gi]
                person_dict = {
                    "bbox": list(students[si].bbox),
                    "face": None,
                    "head_pose": None,
                    "posture": None,
                }
                face = faces[si]
                if face.face_bbox is not None:
                    person_dict["face"] = {
                        "bbox": list(face.face_bbox),
                        "landmarks": None,
                        "ear": face.ear,
                    }
                    # Assume head-down so the object branch is reached: this
                    # isolates whether the *book* evidence is right, rather
                    # than re-testing head-pose accuracy here.
                    person_dict["head_pose"] = {
                        "yaw": 0.0, "pitch": 30.0, "roll": 0.0,
                        "gaze_label": "down",
                    }
                obj_dicts = [
                    {"cls": o.cls, "bbox": list(o.bbox), "confidence": o.confidence}
                    for o in objects
                ]
                predicted_writing = (
                    classify_frame(person_dict, obj_dicts).orientation
                    == "head_down_writing"
                )
                actually_writing = g_cls in ("write", "read")
                if predicted_writing and actually_writing:
                    writing_tp += 1
                elif predicted_writing and not actually_writing:
                    writing_fp += 1
                elif not predicted_writing and actually_writing:
                    writing_fn += 1
                else:
                    writing_tn += 1

            if idx % 25 == 0:
                print(f"  ... {idx}/{len(images)}")

    def prf(tp_: int, fp_: int, fn_: int) -> tuple[float, float, float]:
        p = tp_ / (tp_ + fp_) if tp_ + fp_ else 0.0
        r = tp_ / (tp_ + fn_) if tp_ + fn_ else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return p, r, f

    print("\n=== STUDENT DETECTION vs human labels ===")
    p, r, f1 = prf(tp, fp, fn)
    print(f"  ground-truth students : {tp + fn}")
    print(f"  detected students     : {tp + fp}")
    print(f"  matched (TP)          : {tp}")
    print(f"  missed  (FN)          : {fn}")
    print(f"  spurious(FP)          : {fp}")
    print(f"  precision {p * 100:.1f}%   recall {r * 100:.1f}%   F1 {f1 * 100:.1f}%")
    print("\n  matched students by detection source:")
    for source, count in tp_by_source.most_common():
        print(f"    {source:<14}{count:>6}  ({count / max(tp, 1) * 100:.1f}% of TP)")

    print("\n=== RECALL AND FACE COVERAGE BY GROUND-TRUTH BEHAVIOUR ===")
    print(f"  {'behaviour':<15}{'GT':>6}{'detected':>10}{'recall':>9}{'has face':>10}")
    for cls_name, total in gt_by_class.most_common():
        det = hit_by_class[cls_name]
        fac = face_by_class[cls_name]
        print(f"  {cls_name:<15}{total:>6}{det:>10}{det / total * 100:>8.1f}%"
              f"{fac / total * 100:>9.1f}%")

    print("\n=== WRITING SIGNAL (book proximity) vs GT write/read ===")
    wp, wr, wf = prf(writing_tp, writing_fp, writing_fn)
    print(f"  judged students       : {writing_tp + writing_fp + writing_fn + writing_tn}")
    print(f"  GT writing/reading    : {writing_tp + writing_fn}")
    print(f"  predicted writing     : {writing_tp + writing_fp}")
    print(f"  precision {wp * 100:.1f}%   recall {wr * 100:.1f}%   F1 {wf * 100:.1f}%")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="dataset/dataset")
    parser.add_argument("--per-clip", type=int, default=12)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--match", choices=["centre", "iou"], default="centre")
    parser.add_argument("--log-level", default="ERROR")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level)
    evaluate(Path(args.root), args.per_clip, args.all, args.iou, args.match)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
