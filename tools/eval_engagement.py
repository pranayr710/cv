"""Measure the question the system actually exists to answer.

The per-class behaviour scores are modest (F1 43-72%), but no teacher asks
"is this student's pose class `read` or `write`". They ask whether a student is
**working or not**. Those are different questions with different error costs,
and the second is much easier: `read` mistaken for `write` is a wrong class but
a *correct* engagement reading, and the per-class metric punishes it while the
product-level metric correctly does not.

This scores the binary the deployment cares about:

    on-task   = write, read, look_forward
    off-task  = using_device, sleep
    excluded  = turn_head, handrise, stand

``turn_head`` is excluded rather than assigned. Kendon's F-formation work and
the CSCL literature this project already cites both say a student turned toward
a neighbour cannot be called on- or off-task from vision alone -- a productive
academic discussion and idle chat look identical, and the field's own answer
when it needed that distinction was to add a microphone. Forcing it into either
bucket here would be exactly the mistake ``backend.attention`` was written to
avoid. ``handrise`` and ``stand`` are excluded for the mundane reason that they
have too little data to measure at all (22 and 59 training boxes).

Reported both ways deliberately:

* **strict** -- a student the model never found counts as an engagement error,
  which is the honest end-to-end number;
* **given detection** -- scored only over students that were found, which
  isolates how good the engagement *reading* is from how good detection is.

Run:
    python -m tools.eval_engagement
    python -m tools.eval_engagement --weights runs/behaviour/yolo11m_b4/weights/best.pt
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from tools.eval_detection import _load_labels, _match_score

ROOT = Path(__file__).resolve().parent.parent

ON_TASK = frozenset({"write", "read", "look_forward"})
OFF_TASK = frozenset({"using_device", "sleep"})
EXCLUDED = frozenset({"turn_head", "handrise", "stand"})


def _engagement(label: str) -> str | None:
    """Map a behaviour class to ``"on"``, ``"off"``, or ``None`` if excluded."""
    if label in ON_TASK:
        return "on"
    if label in OFF_TASK:
        return "off"
    return None


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def evaluate(weights: Path, data_root: Path, imgsz: int, conf: float) -> None:
    import cv2
    import yaml
    from ultralytics import YOLO

    with (data_root / "data.yaml").open(encoding="utf-8") as fh:
        names = list(yaml.safe_load(fh)["names"])

    val_images = sorted((data_root / "val" / "images").glob("*.jpg"))
    if not val_images:
        raise FileNotFoundError(f"No val images under {data_root / 'val'}")

    model = YOLO(str(weights))

    # Confusion over the binary, plus the students never found.
    cell: Counter[tuple[str, str]] = Counter()
    not_found: Counter[str] = Counter()
    excluded_gt = 0

    for img_path in val_images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        img_h, img_w = frame.shape[:2]
        gt = _load_labels(
            data_root / "val" / "labels" / f"{img_path.stem}.txt",
            img_w, img_h, names,
        )
        result = model.predict(frame, imgsz=imgsz, conf=conf, verbose=False)[0]
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

        pairs = sorted(
            (
                (_match_score(p_box, g_box, "centre"), pi, gi)
                for pi, (_pc, p_box) in enumerate(preds)
                for gi, (_gc, g_box) in enumerate(gt)
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

        for gi, (g_cls, _g_box) in enumerate(gt):
            truth = _engagement(g_cls)
            if truth is None:
                excluded_gt += 1
                continue
            if gi not in matched:
                not_found[truth] += 1
                continue
            predicted = _engagement(preds[matched[gi]][0])
            # A found student whose predicted class is itself excluded has no
            # engagement reading; counted separately, not silently as correct.
            cell[(truth, predicted or "none")] += 1

    print(f"\n{'=' * 60}")
    print(f"ENGAGEMENT (on-task vs off-task), imgsz {imgsz}, conf {conf}")
    print(f"{len(val_images)} held-out images")
    print(f"\n  on-task  = {sorted(ON_TASK)}")
    print(f"  off-task = {sorted(OFF_TASK)}")
    print(f"  excluded = {sorted(EXCLUDED)}  ({excluded_gt} ground-truth boxes)")

    print(f"\n{'':<12}{'pred on':>10}{'pred off':>10}{'no reading':>12}"
          f"{'not found':>11}")
    for truth in ("on", "off"):
        print(f"  true {truth:<7}{cell[(truth, 'on')]:>10}"
              f"{cell[(truth, 'off')]:>10}{cell[(truth, 'none')]:>12}"
              f"{not_found[truth]:>11}")

    # "off-task" is the actionable class -- it is what would surface to a
    # teacher -- so it is the positive class for precision/recall.
    for mode, include_missed in (("given detection", False), ("strict", True)):
        tp = cell[("off", "off")]
        fp = cell[("on", "off")]
        fn = cell[("off", "on")] + cell[("off", "none")]
        if include_missed:
            fn += not_found["off"]
        p, r, f1 = _prf(tp, fp, fn)
        graded = tp + fp + fn + cell[("on", "on")] + cell[("on", "none")]
        if include_missed:
            graded += not_found["on"]
        correct = tp + cell[("on", "on")]
        print(f"\n  [{mode}]  detecting OFF-TASK")
        print(f"    precision {p * 100:.1f}%   recall {r * 100:.1f}%   "
              f"F1 {f1 * 100:.1f}%")
        print(f"    overall agreement {correct / graded * 100:.1f}% "
              f"of {graded} students")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights", default="runs/behaviour/yolo11m_b4/weights/best.pt"
    )
    parser.add_argument("--data-root", default="dataset/behaviour")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.30)
    args = parser.parse_args()

    weights = Path(args.weights)
    evaluate(
        weights if weights.is_absolute() else ROOT / weights,
        Path(args.data_root),
        args.imgsz,
        args.conf,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
