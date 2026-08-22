"""Evaluate the fine-tuned behaviour model on held-out clips.

Judges the model against the thing it replaces. The book-proximity proxy scored
precision 31.9% / recall 20.7% / F1 25.1% for "is this student writing" against
human labels; this reports the same writing metric for the fine-tuned model, so
the comparison is like-for-like rather than a fresh set of numbers with no
baseline.

Also reported, because a behaviour model that cannot find the student is
useless regardless of its classification accuracy:

* **student detection** (any class counts as "a student is here"), comparable to
  the 82.2% / 90.6% the COCO pipeline scored on this data;
* **per-class precision/recall**, so the classes with too little training data
  (`stand`, `handrise`) are visibly unmeasured rather than quietly averaged in;
* **resolution sensitivity** -- the model trains at 960 for VRAM reasons while
  the pipeline detects at 1920, so it is evaluated at both.

Matching uses the same mutual-centre rule as ``tools/eval_detection.py``: this
dataset annotates tight head+torso boxes, so IoU 0.5 understates recall for
reasons unrelated to model quality.

Run:
    python -m tools.eval_behaviour
    python -m tools.eval_behaviour --imgsz 960 1280 1920
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from tools.eval_detection import _load_labels, _match_score

ROOT = Path(__file__).resolve().parent.parent
WRITING_CLASSES = ("write", "read")


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def evaluate(weights: Path, data_root: Path, sizes: list[int], conf: float) -> None:
    import cv2
    import yaml
    from ultralytics import YOLO

    with (data_root / "data.yaml").open(encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    names = list(cfg["names"])

    val_images = sorted((data_root / "val" / "images").glob("*.jpg"))
    if not val_images:
        raise FileNotFoundError(f"No val images under {data_root / 'val'}")

    model = YOLO(str(weights))

    for imgsz in sizes:
        det_tp = det_fp = det_fn = 0
        cls_tp: Counter[str] = Counter()
        cls_fp: Counter[str] = Counter()
        cls_fn: Counter[str] = Counter()
        gt_total: Counter[str] = Counter()
        # (ground_truth_class, predicted_class) -> count, for students that were
        # found but labelled wrongly. Scores alone say a class is weak; this
        # says what it is being mistaken for, which is what points at a fix.
        confusion: Counter[tuple[str, str]] = Counter()
        # Ground-truth students no predicted box bound to at all -- a detection
        # miss rather than a classification error, and a different problem.
        missed: Counter[str] = Counter()
        w_tp = w_fp = w_fn = 0

        for img_path in val_images:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            img_h, img_w = frame.shape[:2]
            gt = _load_labels(
                data_root / "val" / "labels" / f"{img_path.stem}.txt",
                img_w, img_h, names,
            )
            for g_cls, _g_box in gt:
                gt_total[g_cls] += 1

            result = model.predict(
                frame, imgsz=imgsz, conf=conf, verbose=False
            )[0]
            preds = []
            if result.boxes is not None:
                for (x1, y1, x2, y2), cls_id in zip(
                    result.boxes.xyxy.cpu().numpy(),
                    result.boxes.cls.cpu().numpy().astype(int),
                ):
                    preds.append(
                        (
                            names[cls_id] if cls_id < len(names) else f"id{cls_id}",
                            (float(x1), float(y1), float(x2 - x1), float(y2 - y1)),
                        )
                    )

            # One-to-one greedy matching on position only. Class agreement is
            # scored *after* matching, so a student found but mislabelled counts
            # as a detection hit and a classification error -- not as both a
            # miss and a spurious box, which would double-punish it.
            pairs = sorted(
                (
                    (_match_score(p_box, g_box, "centre"), pi, gi)
                    for pi, (_p_cls, p_box) in enumerate(preds)
                    for gi, (_g_cls, g_box) in enumerate(gt)
                ),
                reverse=True,
            )
            used_p: set[int] = set()
            used_g: set[int] = set()
            matched: dict[int, int] = {}
            for score, pi, gi in pairs:
                if score <= 0.0 or pi in used_p or gi in used_g:
                    continue
                used_p.add(pi)
                used_g.add(gi)
                matched[gi] = pi

            det_tp += len(matched)
            det_fp += len(preds) - len(matched)
            det_fn += len(gt) - len(matched)

            for gi, (g_cls, _g_box) in enumerate(gt):
                if gi not in matched:
                    cls_fn[g_cls] += 1
                    missed[g_cls] += 1
                    if g_cls in WRITING_CLASSES:
                        w_fn += 1
                    continue
                p_cls = preds[matched[gi]][0]
                confusion[(g_cls, p_cls)] += 1
                if p_cls == g_cls:
                    cls_tp[g_cls] += 1
                else:
                    cls_fn[g_cls] += 1
                    cls_fp[p_cls] += 1
                gt_writing = g_cls in WRITING_CLASSES
                pred_writing = p_cls in WRITING_CLASSES
                if gt_writing and pred_writing:
                    w_tp += 1
                elif pred_writing and not gt_writing:
                    w_fp += 1
                elif gt_writing and not pred_writing:
                    w_fn += 1
            # Unmatched predictions are spurious students; their class is a
            # false positive for that class too.
            for pi, (p_cls, _p_box) in enumerate(preds):
                if pi not in used_p:
                    cls_fp[p_cls] += 1
                    if p_cls in WRITING_CLASSES:
                        w_fp += 1

        print(f"\n{'=' * 62}\n=== imgsz {imgsz}, conf {conf} "
              f"({len(val_images)} held-out images) ===")
        p, r, f1 = _prf(det_tp, det_fp, det_fn)
        print("\nSTUDENT DETECTION (any class)")
        print(f"  GT {det_tp + det_fn}   TP {det_tp}   FP {det_fp}   FN {det_fn}")
        print(f"  precision {p * 100:.1f}%   recall {r * 100:.1f}%   F1 {f1 * 100:.1f}%")
        print("  (COCO pipeline on this data: precision 82.2%  recall 90.6%)")

        print("\nPER-CLASS (position matched, then class compared)")
        print(f"  {'class':<15}{'GT':>5}{'prec':>8}{'recall':>8}{'F1':>8}")
        for name in names:
            cp, cr, cf = _prf(cls_tp[name], cls_fp[name], cls_fn[name])
            flag = "  <- too little data" if gt_total[name] < 10 else ""
            print(f"  {name:<15}{gt_total[name]:>5}{cp * 100:>7.1f}%"
                  f"{cr * 100:>7.1f}%{cf * 100:>7.1f}%{flag}")

        # Split the two failure modes apart. A class can score badly because
        # the student was never found (detection) or because they were found and
        # mislabelled (classification), and those need different fixes.
        print()
        print("WHERE THE ERRORS GO (found, but labelled wrongly)")
        wrong = [(pair, c) for pair, c in confusion.items() if pair[0] != pair[1]]
        if not wrong:
            print("  no class confusions")
        for (g_cls, p_cls), count in sorted(wrong, key=lambda kv: -kv[1])[:8]:
            share = count / gt_total[g_cls] * 100 if gt_total[g_cls] else 0.0
            print(f"  {g_cls:<14} -> {p_cls:<14}{count:>5}"
                  f"  ({share:.0f}% of all {g_cls})")

        print()
        print("NOT FOUND AT ALL (no predicted box bound to the student)")
        if not missed:
            print("  every ground-truth student was found")
        for g_cls, count in missed.most_common(6):
            share = count / gt_total[g_cls] * 100 if gt_total[g_cls] else 0.0
            print(f"  {g_cls:<14}{count:>5}  ({share:.0f}% of all {g_cls})")

        wp, wr, wf = _prf(w_tp, w_fp, w_fn)
        print("\nWRITING SIGNAL (write/read), head-to-head")
        print(f"  fine-tuned model : precision {wp * 100:.1f}%   "
              f"recall {wr * 100:.1f}%   F1 {wf * 100:.1f}%")
        print("  book proximity   : precision  31.9%   recall  20.7%   F1  25.1%")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights",
        default="runs/behaviour/yolo11m_960/weights/best.pt",
    )
    parser.add_argument("--data-root", default="dataset/behaviour")
    parser.add_argument("--imgsz", nargs="+", type=int, default=[960, 1920])
    parser.add_argument("--conf", type=float, default=0.30)
    args = parser.parse_args()

    evaluate(
        ROOT / args.weights if not Path(args.weights).is_absolute()
        else Path(args.weights),
        Path(args.data_root),
        args.imgsz,
        args.conf,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
