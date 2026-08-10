"""Benchmark the two face-detector backends across every dataset image.

Answers the question sir asked at Review 1 -- "why were only some marked?" --
with a per-image number instead of one cherry-picked example. Reports, for each
image and for both backends:

* persons  -- YOLOv11 person detections
* faces    -- faces bound to a person box (what the pipeline actually uses)
* raw      -- faces the detector found before person-binding (SCRFD only)
* orphan   -- raw faces with NO containing person box. These are students the
              *person* detector missed, not a face failure -- the diagnostic
              that shows where the remaining bottleneck actually is.

Run:
    python -m tools.bench_faces
    python -m tools.bench_faces --backends scrfd
    python -m tools.bench_faces --render        # also write overlay images
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import replace
from pathlib import Path

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"


def _containment(inner, outer) -> float:
    """Fraction of ``inner``'s area inside ``outer`` (both ``(x, y, w, h)``)."""
    ix0, iy0, iw, ih = inner
    ox0, oy0, ow, oh = outer
    inter_w = max(0, min(ix0 + iw, ox0 + ow) - max(ix0, ox0))
    inter_h = max(0, min(iy0 + ih, oy0 + oh) - max(iy0, oy0))
    area = iw * ih
    return (inter_w * inter_h / area) if area > 0 else 0.0


def bench(backends: list[str], render: bool) -> None:
    import cv2

    from backend.config import CONFIG
    from backend.detection import Detector
    from backend.face import FaceAnalyzer

    images = sorted(p for p in DATASET.glob("*.jpg"))
    if not images:
        raise FileNotFoundError(f"No .jpg images found in {DATASET}")

    detector = Detector()

    # Person detection is backend-independent, so run it once per image and
    # reuse it for every face backend -- otherwise the comparison would also be
    # measuring YOLO's (deterministic, but wasted) re-run.
    frames: dict[Path, tuple] = {}
    for path in images:
        frame = cv2.imread(str(path))
        if frame is None:
            logger.warning("Skipping unreadable image: %s", path)
            continue
        persons, _objects = detector.detect(frame)
        frames[path] = (frame, [p.bbox for p in persons])

    results: dict[str, dict] = {}
    for backend in backends:
        cfg = replace(CONFIG.face, detector=backend)
        totals = {"persons": 0, "faces": 0, "raw": 0, "orphan": 0, "secs": 0.0}
        rows = []
        with FaceAnalyzer(cfg) as analyzer:
            for path, (frame, person_boxes) in frames.items():
                t0 = time.perf_counter()
                faces = analyzer.analyze(frame, person_boxes)
                elapsed = time.perf_counter() - t0

                n_faces = sum(1 for f in faces if f.face_bbox is not None)
                n_persons = len(person_boxes)

                # Raw/orphan counts are only meaningful for a whole-frame
                # detector; the mediapipe path never sees a face outside a
                # person crop by construction.
                n_raw = n_orphan = 0
                if backend == "scrfd":
                    raw = analyzer.detect_faces(frame)
                    n_raw = len(raw)
                    n_orphan = sum(
                        1
                        for d in raw
                        if not any(
                            _containment(d.bbox, pb) >= cfg.assign_min_containment
                            for pb in person_boxes
                        )
                    )

                rows.append((path.name, n_persons, n_faces, n_raw, n_orphan, elapsed))
                totals["persons"] += n_persons
                totals["faces"] += n_faces
                totals["raw"] += n_raw
                totals["orphan"] += n_orphan
                totals["secs"] += elapsed

                if render:
                    out = frame.copy()
                    for f in faces:
                        if f.face_bbox is None:
                            continue
                        x, y, w, h = f.face_bbox
                        # Green = landmarks fitted, amber = box only (still
                        # usable for head pose).
                        colour = (0, 220, 0) if f.landmarks else (0, 170, 255)
                        cv2.rectangle(out, (x, y), (x + w, y + h), colour, 2)
                    cv2.putText(
                        out,
                        f"{backend}: {n_faces} faces / {n_persons} persons",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 0), 2,
                    )
                    dest = ROOT / "outputs" / f"{path.stem}_{backend}.jpg"
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(dest), out)

        results[backend] = {"rows": rows, "totals": totals}

    # --- report ---
    for backend, data in results.items():
        print(f"\n=== backend: {backend} ===")
        print(f"{'image':<20} {'persons':>8} {'faces':>7} {'rate':>7} "
              f"{'raw':>5} {'orphan':>7} {'ms':>7}")
        for name, np_, nf, nr, no, secs in data["rows"]:
            rate = f"{nf / np_ * 100:.0f}%" if np_ else "-"
            print(f"{name:<20} {np_:>8} {nf:>7} {rate:>7} "
                  f"{nr:>5} {no:>7} {secs * 1000:>7.0f}")
        t = data["totals"]
        rate = f"{t['faces'] / t['persons'] * 100:.1f}%" if t["persons"] else "-"
        print(f"{'TOTAL':<20} {t['persons']:>8} {t['faces']:>7} {rate:>7} "
              f"{t['raw']:>5} {t['orphan']:>7} {t['secs'] * 1000:>7.0f}")

    if len(results) > 1:
        print("\n=== comparison ===")
        base = results.get("mediapipe")
        new = results.get("scrfd")
        if base and new:
            b, n = base["totals"], new["totals"]
            gain = (n["faces"] / b["faces"]) if b["faces"] else float("inf")
            print(f"faces bound to a person:  mediapipe {b['faces']} "
                  f"-> scrfd {n['faces']}  ({gain:.1f}x)")
            print(f"face-match rate:          "
                  f"{b['faces'] / b['persons'] * 100:.1f}% "
                  f"-> {n['faces'] / n['persons'] * 100:.1f}%")
            print(f"scrfd raw faces found:    {n['raw']} "
                  f"({n['orphan']} had no person box -> person detection "
                  f"missed those students)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backends", nargs="+", default=["mediapipe", "scrfd"],
        choices=["mediapipe", "scrfd"],
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--log-level", default="WARNING")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")
    bench(args.backends, args.render)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
