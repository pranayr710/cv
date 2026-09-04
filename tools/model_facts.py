"""Print every model number the decks quote, read off the models themselves.

A parameter count copied from a paper is a claim about a model somebody else
loaded. This loads ours: the graphs actually shipped in this environment, the
weights actually on disk, and the fine-tuned checkpoint re-validated rather than
read out of its training log.

Run before a presentation, and diff against the tables in build_models_ppt.py.

    python tools/model_facts.py            # sizes and parameter counts
    python tools/model_facts.py --validate # also re-run behaviour validation
"""
import argparse
import csv
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")

RUN = Path("runs/behaviour/merged4_aug")
BEHAVIOUR_DATA = Path("dataset/behaviour_merged")


def _mb(path: Path) -> str:
    return f"{path.stat().st_size / 1e6:.1f}" if path.exists() else "-"


def torch_models(rows: list) -> None:
    """YOLO checkpoints, counted from the loaded module."""
    from ultralytics import YOLO

    for label, path in (
        ("YOLO11m (COCO)", Path("yolo11m.pt")),
        ("YOLO11m fine-tuned", RUN / "weights/best.pt"),
        ("YOLO11m-pose (benchmarked)", Path("yolo11m-pose.pt")),
    ):
        if not path.exists():
            rows.append((label, "-", "not present", ""))
            continue
        model = YOLO(str(path))
        n = sum(p.numel() for p in model.model.parameters())
        classes = len(model.names)
        rows.append((label, _mb(path), f"{n:,}", f"{classes} classes"))


def onnx_models(rows: list) -> None:
    """InsightFace and the expression model, counted from graph initialisers."""
    import numpy as np
    import onnx

    targets = [
        ("SCRFD det_10g", Path.home() / ".insightface/models/buffalo_l/det_10g.onnx"),
        ("ArcFace w600k_r50",
         Path.home() / ".insightface/models/buffalo_l/w600k_r50.onnx"),
        ("EfficientNet-B0 emotion",
         Path.home() / ".emotiefflib/enet_b0_8_best_vgaf.onnx"),
    ]
    for label, path in targets:
        if not path.exists():
            rows.append((label, "-", "not downloaded", ""))
            continue
        graph = onnx.load(str(path)).graph
        n = sum(int(np.prod(t.dims)) for t in graph.initializer)
        out = graph.output[0]
        shape = "x".join(str(d.dim_value) for d in out.type.tensor_type.shape.dim)
        rows.append((label, _mb(path), f"{n:,}", f"output {shape}"))


def headpose(rows: list) -> None:
    """SixDRepNet, whose checkpoint filename also records its training set."""
    try:
        from sixdrepnet import SixDRepNet

        net = SixDRepNet().model
        n = sum(p.numel() for p in net.parameters())
    except Exception as exc:  # noqa: BLE001 - reporting beats failing a fact dump
        rows.append(("SixDRepNet", "-", f"unavailable: {exc}", ""))
        return
    found = list(Path.home().glob(".cache/**/6DRepNet*.pth"))
    size = _mb(found[0]) if found else "-"
    note = found[0].stem if found else ""
    rows.append(("SixDRepNet", size, f"{n:,}", note))


def mediapipe(rows: list) -> None:
    """MediaPipe ships TFLite graphs; parameter counts are not exposed."""
    import mediapipe

    base = Path(mediapipe.__file__).parent / "modules"
    for label, rel in (
        ("MediaPipe Pose (full)", "pose_landmark/pose_landmark_full.tflite"),
        ("MediaPipe Face Mesh", "face_landmark/face_landmark.tflite"),
    ):
        rows.append((label, _mb(base / rel), "not exposed", "TFLite"))


def finetune_facts() -> None:
    """The fine-tuning run: configuration as used, metrics as recorded."""
    args = RUN / "args.yaml"
    if args.exists():
        keep = {"model", "data", "epochs", "patience", "batch", "imgsz",
                "freeze", "lr0", "optimizer", "pretrained"}
        print("\nfine-tuning configuration (runs/.../args.yaml)")
        for line in args.read_text(encoding="utf-8").splitlines():
            key = line.split(":", 1)[0].strip()
            if key in keep:
                print(f"  {line.strip()}")

    results = RUN / "results.csv"
    if results.exists():
        rows = list(csv.DictReader(results.open()))
        best = max(rows, key=lambda r: float(r["metrics/mAP50-95(B)"]))
        print(f"\nepochs completed : {len(rows)}")
        print(f"best epoch       : {best['epoch']}")
        print(f"first-epoch mAP50: {float(rows[0]['metrics/mAP50(B)']):.4f}")
        print(f"best  mAP50      : {float(best['metrics/mAP50(B)']):.4f}")
        print(f"best  mAP50-95   : {float(best['metrics/mAP50-95(B)']):.4f}")
        print(f"train wall-clock : {float(rows[-1]['time']) / 60:.0f} min")

    for split in ("train", "val"):
        labels = BEHAVIOUR_DATA / split / "labels"
        if not labels.is_dir():
            continue
        counts: dict[str, int] = {}
        for f in labels.glob("*.txt"):
            for line in f.read_text(encoding="utf-8").split("\n"):
                if line.strip():
                    counts[line.split()[0]] = counts.get(line.split()[0], 0) + 1
        images = len(list((BEHAVIOUR_DATA / split / "images").glob("*")))
        print(f"\n{split}: {images} images, {sum(counts.values())} boxes")
        for cls in sorted(counts):
            print(f"  class {cls}: {counts[cls]}")


def validate() -> None:
    """Re-run validation so per-class numbers come from the weights, not a log."""
    from ultralytics import YOLO

    weights = RUN / "weights/best.pt"
    if not weights.exists():
        print("\nno fine-tuned weights to validate")
        return
    # workers=0: Windows re-imports the entry module per worker, which fails
    # whenever this is run from anything but a plain script path.
    r = YOLO(str(weights)).val(data=str(BEHAVIOUR_DATA / "data.yaml"), imgsz=640,
                               batch=4, workers=0, plots=False, verbose=False)
    print(f"\nvalidation  mAP50={r.box.map50:.4f}  mAP50-95={r.box.map:.4f}  "
          f"P={r.box.mp:.4f}  R={r.box.mr:.4f}")
    for i, c in enumerate(r.box.ap_class_index):
        print(f"  {r.names[c]:<14} P={r.box.p[i]:.3f} R={r.box.r[i]:.3f} "
              f"mAP50={r.box.ap50[i]:.3f} mAP50-95={r.box.ap[i]:.3f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--validate", action="store_true",
                    help="also re-run validation of the fine-tuned model")
    args = ap.parse_args()

    rows: list[tuple[str, str, str, str]] = []
    torch_models(rows)
    onnx_models(rows)
    headpose(rows)
    mediapipe(rows)

    print(f"{'model':<28} {'MB':>8} {'parameters':>15}  note")
    print("-" * 74)
    total = 0
    for label, mb, params, note in rows:
        print(f"{label:<28} {mb:>8} {params:>15}  {note}")
        if params.replace(",", "").isdigit():
            total += int(params.replace(",", ""))
    print("-" * 74)
    print(f"{'countable total':<28} {'':>8} {total:>15,}")

    finetune_facts()
    if args.validate:
        validate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
