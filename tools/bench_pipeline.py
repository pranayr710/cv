"""Time every pipeline stage on real footage, so the published figure is true.

``PART1_PLAN.md`` publishes **0.41 FPS, 83% CPU-bound**. That predates commit
787fe92, which moved SCRFD and ArcFace onto the GPU -- roughly half of frame
time. A stale number that flatters the project is worse than no number at all,
which is why this exists as a tool rather than a one-off script: the figure has
to be re-measurable whenever the models change.

What is measured
----------------

Wall-clock per stage, over real frames from the target video, after a warm-up
pass so model loading and CUDA context creation are not counted as throughput.
Stages are timed individually rather than by difference, so the numbers add up
to something a reader can act on.

Run:
    python -m tools.bench_pipeline --video <clip> --frames 40
"""

from __future__ import annotations

import argparse
import statistics
import time


def _percent(part: float, whole: float) -> str:
    """Format a share of total frame time."""
    return f"{100 * part / whole:5.1f}%" if whole else "    - "


def main() -> int:
    """Benchmark the pipeline and print a per-stage table.

    Returns:
        A process exit code.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        default="dataset/23-08/vedio/WhatsApp Video 2026-08-23 at 10.16.09.mp4",
    )
    parser.add_argument("--frames", type=int, default=40,
                        help="Frames to time, after warm-up.")
    parser.add_argument("--sample-rate", type=int, default=30,
                        help="Take every Nth frame, matching how the pipeline runs.")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="Override the detector size (default: config).")
    args = parser.parse_args()

    import logging

    logging.disable(logging.CRITICAL)

    import cv2

    from backend.config import CONFIG
    from backend.integrate import (
        _build_detector,
        _build_expression_recognizer,
        _build_face_analyzer,
        _build_headpose_estimator,
        _build_person_tracker,
        _build_posture_analyzer,
    )

    config = CONFIG
    if args.imgsz:
        from dataclasses import replace

        config = replace(config, detection=replace(config.detection, imgsz=args.imgsz))

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"Could not open {args.video}")

    frames = []
    index = 0
    while len(frames) < args.frames + 2:
        ok, frame = capture.read()
        if not ok:
            break
        if index % args.sample_rate == 0:
            frames.append(frame)
        index += 1
    capture.release()
    if len(frames) < 3:
        raise SystemExit("Not enough frames to benchmark.")
    print(f"{len(frames)} frames at {frames[0].shape[1]}x{frames[0].shape[0]}, "
          f"detector imgsz={config.detection.imgsz}\n")

    detector = _build_detector(config)
    headpose = _build_headpose_estimator(config)
    posture = _build_posture_analyzer(config)
    expression = _build_expression_recognizer(config)
    tracker = _build_person_tracker(config)

    timings: dict[str, list[float]] = {k: [] for k in
                                       ("detect", "face", "posture", "headpose",
                                        "expression", "track")}
    people = []

    with _build_face_analyzer(config) as faces:
        # Warm-up: first-call CUDA kernels and lazy model loads are startup
        # cost, not throughput, and counting them understates the pipeline.
        warm = frames[0]
        persons, _ = detector.detect(warm)
        boxes = [p.bbox for p in persons]
        results = faces.analyze(warm, boxes)
        posture.analyze(warm, boxes)
        headpose.estimate(warm, [r.face_bbox for r in results])
        expression.classify(warm, [r.face_bbox for r in results],
                            [r.kps for r in results])

        for frame in frames[1:]:
            t = time.perf_counter()
            persons, _objects = detector.detect(frame)
            timings["detect"].append(time.perf_counter() - t)
            boxes = [p.bbox for p in persons]
            people.append(len(persons))

            t = time.perf_counter()
            results = faces.analyze(frame, boxes)
            timings["face"].append(time.perf_counter() - t)
            face_boxes = [r.face_bbox for r in results]

            t = time.perf_counter()
            posture.analyze(frame, boxes)
            timings["posture"].append(time.perf_counter() - t)

            t = time.perf_counter()
            headpose.estimate(frame, face_boxes)
            timings["headpose"].append(time.perf_counter() - t)

            t = time.perf_counter()
            expression.classify(frame, face_boxes, [r.kps for r in results])
            timings["expression"].append(time.perf_counter() - t)

            t = time.perf_counter()
            tracker.update(persons)
            timings["track"].append(time.perf_counter() - t)

    # Frames containing nobody cost almost nothing -- every per-person stage
    # returns immediately -- so averaging over all frames reports a number that
    # describes the footage rather than the pipeline. On a clip where the camera
    # pans off the class for stretches, that is the difference between a
    # throughput figure and a sampling artefact. Both are printed.
    busy = [i for i, n in enumerate(people) if n > 0]
    total = sum(statistics.mean(v) for v in timings.values())
    total_busy = (
        sum(statistics.mean([v[i] for i in busy]) for v in timings.values())
        if busy
        else 0.0
    )

    print(f"{'stage':<14}{'all ms':>9}{'busy ms':>10}{'per-person':>12}"
          f"{'share':>8}  device")
    devices = {
        "detect": "GPU (torch)",
        "face": "GPU (onnxruntime CUDA)",
        "posture": "CPU (MediaPipe, no Windows GPU build)",
        "headpose": "GPU (torch)",
        "expression": "GPU (onnxruntime CUDA)",
        "track": "CPU (pure Python)",
    }
    heads = sum(people) or 1
    for name, values in sorted(timings.items(),
                               key=lambda kv: -statistics.mean(kv[1])):
        mean = statistics.mean(values)
        busy_mean = statistics.mean([values[i] for i in busy]) if busy else 0.0
        per_head = sum(values) / heads
        print(f"{name:<14}{mean * 1000:>9.1f}{busy_mean * 1000:>10.1f}"
              f"{per_head * 1000:>12.1f}{_percent(mean, total):>8}  {devices[name]}")

    cpu = statistics.mean(timings["posture"]) + statistics.mean(timings["track"])
    heads_total = sum(sum(v) for v in timings.values()) / heads
    print(f"\n{'TOTAL':<14}{total * 1000:>9.1f}{total_busy * 1000:>10.1f}"
          f"{heads_total * 1000:>12.1f}{'100.0%':>8}")
    print(f"\nthroughput, all sampled frames  {1 / total:.2f} FPS")
    if total_busy:
        print(f"throughput, frames with people  {1 / total_busy:.2f} FPS")
    print(f"CPU-bound share                 {_percent(cpu, total).strip()}")
    print(f"frames with at least one person {len(busy)}/{len(people)}")
    print(f"persons per frame               mean {statistics.mean(people):.1f}, "
          f"max {max(people)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
