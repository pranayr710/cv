"""Judge the merged 4-class behaviour model against three acceptance gates.

The retrain exists to fix one specific failure: the previous model returns ZERO
detections on 640x360 classroom video (and still zero at conf 0.05, and still
zero when the frame is upscaled 2x or 3x -- so a domain gap, not a threshold or
resolution problem). A new model is only worth adopting if it actually fixes
that *and* does not regress elsewhere, so all three gates are checked together
rather than cherry-picking whichever number looks best.

Gate 1 -- held-out clips from the original dataset. Must not regress against
         the previous model's writing-signal F1 of 65.3%.
Gate 2 -- the real video. Must produce NON-ZERO detections. This is the actual
         bug; a model that still returns nothing here has failed regardless of
         how good its other numbers are.
Gate 3 -- the second dataset's own `test` split, deliberately excluded from
         training (see tools/merge_behaviour_datasets.py). Must beat the
         previous model's 7.3% F1 to count as genuinely better-generalizing
         rather than just fitted to more of the same.

Run:
    python -m tools.eval_merged_model
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent

#: Class order of the merged model, mirroring merge_behaviour_datasets.CANONICAL.
MERGED_CLASSES: tuple[str, ...] = ("read", "sleep", "using_device", "write")
WRITING_CLASSES = frozenset({"read", "write"})

#: Previous model's numbers, for a like-for-like verdict rather than a bare score.
BASELINE_WRITING_F1 = 65.3
BASELINE_OOD_F1 = 7.3


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def _centre_in(inner, outer) -> bool:
    cx, cy = inner[0] + inner[2] / 2, inner[1] + inner[3] / 2
    return (
        outer[0] <= cx <= outer[0] + outer[2]
        and outer[1] <= cy <= outer[1] + outer[3]
    )


def _load_gt(label_file: Path, w: int, h: int, names: list[str], mapping: dict):
    """Load ground truth, remapped to merged class names; unmapped are dropped."""
    out = []
    if not label_file.is_file():
        return out
    for line in label_file.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        canon = mapping.get(names[int(float(parts[0]))])
        if canon is None:
            continue
        cx, cy, bw, bh = (float(v) for v in parts[1:5])
        out.append((canon, ((cx - bw / 2) * w, (cy - bh / 2) * h, bw * w, bh * h)))
    return out


def _score_split(model, images, labels_dir, names, mapping, imgsz, conf):
    """Position-match predictions to GT, then compare class. Returns metrics."""
    import cv2

    cls_tp: Counter[str] = Counter()
    cls_fp: Counter[str] = Counter()
    cls_fn: Counter[str] = Counter()
    w_tp = w_fp = w_fn = 0
    total_dets = 0

    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        gt = _load_gt(labels_dir / f"{img_path.stem}.txt", w, h, names, mapping)

        result = model.predict(frame, imgsz=imgsz, conf=conf, verbose=False)[0]
        preds = []
        if result.boxes is not None:
            for (x1, y1, x2, y2), cid in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.cls.cpu().numpy().astype(int),
            ):
                label = MERGED_CLASSES[cid] if cid < len(MERGED_CLASSES) else f"id{cid}"
                preds.append((label, (float(x1), float(y1), float(x2 - x1), float(y2 - y1))))
        total_dets += len(preds)

        used_p: set[int] = set()
        used_g: set[int] = set()
        for pi, (p_cls, p_box) in enumerate(preds):
            for gi, (g_cls, g_box) in enumerate(gt):
                if gi in used_g:
                    continue
                if _centre_in(p_box, g_box) or _centre_in(g_box, p_box):
                    used_p.add(pi)
                    used_g.add(gi)
                    if p_cls == g_cls:
                        cls_tp[g_cls] += 1
                    else:
                        cls_fn[g_cls] += 1
                        cls_fp[p_cls] += 1
                    gw, pw = g_cls in WRITING_CLASSES, p_cls in WRITING_CLASSES
                    if gw and pw:
                        w_tp += 1
                    elif pw and not gw:
                        w_fp += 1
                    elif gw and not pw:
                        w_fn += 1
                    break
        for gi, (g_cls, _b) in enumerate(gt):
            if gi not in used_g:
                cls_fn[g_cls] += 1
                if g_cls in WRITING_CLASSES:
                    w_fn += 1
        for pi, (p_cls, _b) in enumerate(preds):
            if pi not in used_p:
                cls_fp[p_cls] += 1
                if p_cls in WRITING_CLASSES:
                    w_fp += 1

    return cls_tp, cls_fp, cls_fn, _prf(w_tp, w_fp, w_fn), total_dets


def evaluate(weights: Path, imgsz: int, conf: float) -> None:
    import cv2
    from ultralytics import YOLO

    if not weights.is_file():
        raise FileNotFoundError(f"weights not found: {weights}")
    model = YOLO(str(weights))
    print(f"model: {weights}\nimgsz={imgsz} conf={conf}\n")

    # ---------------- Gate 1: held-out clips, original dataset -------------- #
    merged = ROOT / "dataset" / "behaviour_merged"
    val_images = sorted((merged / "val" / "images").glob("*.jpg"))
    cls_tp, cls_fp, cls_fn, (wp, wr, wf), _ = _score_split(
        model, val_images, merged / "val" / "labels",
        MERGED_CLASSES, {c: c for c in MERGED_CLASSES}, imgsz, conf,
    )
    print("=" * 64)
    print(f"GATE 1 -- held-out clips ({len(val_images)} images)")
    print(f"  {'class':<14}{'prec':>8}{'recall':>8}{'F1':>8}")
    for name in MERGED_CLASSES:
        p, r, f = _prf(cls_tp[name], cls_fp[name], cls_fn[name])
        print(f"  {name:<14}{p*100:>7.1f}%{r*100:>7.1f}%{f*100:>7.1f}%")
    print(f"  writing signal: precision {wp*100:.1f}%  recall {wr*100:.1f}%  "
          f"F1 {wf*100:.1f}%   (baseline {BASELINE_WRITING_F1}%)")
    gate1 = wf * 100 >= BASELINE_WRITING_F1
    print(f"  -> {'PASS' if gate1 else 'FAIL'}")

    # ---------------- Gate 2: the real video, non-zero detections ---------- #
    video = ROOT / "dataset" / "23-08" / "vedio" / "WhatsApp Video 2026-08-23 at 10.16.09.mp4"
    print("\n" + "=" * 64)
    print("GATE 2 -- real 640x360 video: are there ANY detections at all?")
    gate2 = False
    if not video.is_file():
        print(f"  video not found: {video}  -> SKIPPED")
    else:
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        found = 0
        per_class: Counter[str] = Counter()
        for sec in range(30, 220, 10):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(sec * fps))
            ok, frame = cap.read()
            if not ok:
                continue
            res = model.predict(frame, imgsz=imgsz, conf=conf, verbose=False)[0]
            if res.boxes is not None:
                found += len(res.boxes)
                for cid in res.boxes.cls.cpu().numpy().astype(int):
                    per_class[MERGED_CLASSES[cid]] += 1
        cap.release()
        print(f"  detections across 19 sampled frames: {found}  (previous model: 0)")
        print(f"  by class: {dict(per_class) or 'none'}")
        gate2 = found > 0
        print(f"  -> {'PASS' if gate2 else 'FAIL'}")

    # ---------------- Gate 3: independent dataset, unseen in training ------ #
    new = ROOT / "dataset" / "23-08"
    new_names = yaml.safe_load((new / "data.yaml").read_text(encoding="utf-8"))["names"]
    ood_map = {
        "Reading": "read", "Sleeping": "sleep",
        "Using Phone": "using_device", "Writing": "write",
    }
    ood_images = sorted((new / "test" / "images").glob("*.jpg"))
    o_tp, o_fp, o_fn, (_op, _orr, of), _ = _score_split(
        model, ood_images, new / "test" / "labels", new_names, ood_map, imgsz, conf
    )
    print("\n" + "=" * 64)
    print(f"GATE 3 -- independent dataset test split ({len(ood_images)} images, "
          f"never trained on)")
    all_tp = sum(o_tp.values()); all_fp = sum(o_fp.values()); all_fn = sum(o_fn.values())
    gp, gr, gf = _prf(all_tp, all_fp, all_fn)
    print(f"  overall: precision {gp*100:.1f}%  recall {gr*100:.1f}%  F1 {gf*100:.1f}%"
          f"   (baseline {BASELINE_OOD_F1}%)")
    print(f"  writing signal F1: {of*100:.1f}%")
    gate3 = gf * 100 > BASELINE_OOD_F1
    print(f"  -> {'PASS' if gate3 else 'FAIL'}")

    print("\n" + "=" * 64)
    passed = sum([gate1, gate2, gate3])
    print(f"VERDICT: {passed}/3 gates passed")
    if passed == 3:
        print("  ADOPT: set CONFIG.behaviour.weights + class_names to the merged model.")
    else:
        print("  DO NOT ADOPT as-is. The previous model stays; report which gate failed.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights", default="runs/behaviour/merged4_aug/weights/best.pt"
    )
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.30)
    args = parser.parse_args()
    w = Path(args.weights)
    evaluate(w if w.is_absolute() else ROOT / w, args.imgsz, args.conf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
