"""Run a live classroom session from a webcam, keyed on registered students.

This is the whole system in one loop, live instead of offline:

    webcam frame
      -> detect people
      -> read each face
      -> MATCH against the registered gallery  (fixed person_id)
      -> head pose / gaze, posture, expression
      -> per-frame scene graph, nodes keyed on person_id
      -> temporal analysis across frames
      -> one profile per student when the session ends

Why this can be single-pass, when the offline pipeline cannot
--------------------------------------------------------------

:class:`backend.identity.TwoPassIdentityResolver` must see the whole video
before it can name anyone, because it *discovers* identities by clustering and
a cluster is not final until the last frame is in. Registration removes that
dependency entirely: identity becomes a lookup against known references, which
is decidable on the first frame a face appears. Live operation is not a
compromised version of the offline path -- it is what registration makes
possible.

It also lands on the right side of this project's hardest measured limit. On
640x360 classroom footage, faces are 27-37px and different students reach
0.6-0.8 cosine similarity, so recognition is unreliable no matter what the code
does. A webcam sees faces several times that size, which is the regime face
recognition actually works in.

Privacy
-------

A gallery is biometric data about named people -- read the privacy section of
:mod:`backend.enrollment` before running this. Frames are never written to
disk; only the derived per-frame records and the end-of-session profiles are.

Run:
    # 1. register each student once (or use --images with photo folders)
    python -m tools.register_faces --webcam --name asha --shots 12

    # 2. run the session; q or Ctrl-C ends it
    python -m tools.live_session

    # headless, with a time limit
    python -m tools.live_session --no-window --seconds 120
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

#: Sentinel person_id for a face the gallery does not recognise. Negative, in
#: the same convention identity.py already uses for "not verified as anyone".
UNKNOWN_PERSON_ID = -1


def _draw(frame, record, graph, names, fps):
    """Overlay the live reading onto the frame, in place.

    Args:
        frame: The BGR frame to draw on.
        record: This frame's Stage 1 record.
        graph: This frame's scene graph, for the per-node features.
        names: ``{person_id: name}`` from the gallery.
        fps: Measured frames per second, for the header.

    Returns:
        The same frame, annotated.
    """
    import cv2

    features = {
        node.get("person_id"): (node.get("features") or {})
        for node in graph.get("nodes", [])
    }
    for person in record["persons"]:
        pid = person.get("person_id")
        x, y, w, h = (int(v) for v in person["bbox"])
        known = pid is not None and pid > 0
        colour = (80, 220, 80) if known else (80, 80, 220)
        cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)

        feat = features.get(pid, {})
        label = names.get(pid, "unknown") if known else "unknown"
        lines = [
            f"{label}" + (f"  #{pid}" if known else ""),
            f"gaze: {feat.get('gaze_label') or '-'}",
            f"expr: {feat.get('expression') or '-'}",
            f"engage: {feat.get('engagement') or '-'}",
        ]
        for i, text in enumerate(lines):
            cv2.putText(frame, text, (x, max(12, y - 6 - 14 * (len(lines) - 1 - i))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour, 1, cv2.LINE_AA)

    cv2.putText(frame, f"{len(record['persons'])} people | {fps:4.1f} fps | q to end",
                (8, frame.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (240, 240, 240), 1, cv2.LINE_AA)
    return frame


def run(args) -> int:
    """Run the live session until the user stops it, then write the outputs.

    Args:
        args: Parsed CLI arguments.

    Returns:
        A process exit code.
    """
    import cv2

    from backend.config import CONFIG
    from backend.enrollment import EnrolledGallery
    from backend.integrate import (
        _assemble_frame,
        _build_detector,
        _build_expression_recognizer,
        _build_face_analyzer,
        _build_headpose_estimator,
        _build_posture_analyzer,
    )
    from backend.scene_graph import generate_scene_graph
    from backend.student_profile import build_profiles
    from backend.temporal import TemporalTracker

    config = CONFIG
    if args.yaw_reference is not None:
        from dataclasses import replace

        config = replace(
            config, headpose=replace(config.headpose, yaw_reference_deg=args.yaw_reference)
        )

    gallery = EnrolledGallery.load(args.gallery, config.identity)
    if len(gallery) == 0:
        print(
            f"Nobody is registered in {args.gallery}.\n"
            f"Register students first:\n"
            f"    python -m tools.register_faces --webcam --name <student>\n"
            f"Without a gallery every face would read as 'unknown' and nothing "
            f"could be stored under a student id."
        )
        return 1
    names = {p.person_id: p.name for p in gallery.people}
    print(f"{len(gallery)} registered: " + ", ".join(f"{p.name} (#{p.person_id})" for p in gallery.people))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_path = out_dir / "live_frames.jsonl"
    graph_path = out_dir / "live_graph.jsonl"
    profiles_path = out_dir / "live_profiles.json"

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        print(f"Could not open camera {args.camera}.")
        return 1

    # Read one frame before building the detector, because the right imgsz
    # depends on how big the frames actually are.
    #
    # DetectionConfig.imgsz is 1920, tuned for wide classroom shots where
    # students are small. A webcam is the opposite case: a 640x480 frame with
    # one person filling it. Upscaling that 3x does not just waste time, it
    # DESTROYS the detection -- measured on a real webcam-sized frame, YOLO
    # finds the person at imgsz 416 through 1600 and finds NOTHING at 1920.
    # Person detection failing takes everything with it, because faces are
    # bound to person boxes, so the whole session reports zero people while the
    # picture on screen is perfectly clear.
    #
    # Capping the upscale at 2x keeps the small-subject benefit for large
    # inputs and stays inside the range that works for webcam frames.
    # Webcams open dark and take a second or two to settle their exposure.
    # Measured on this machine: the first 40 frames came back at brightness
    # 9-11 out of 255 while the room was lit. Starting immediately means the
    # opening seconds of every session are unusable black frames.
    ok = False
    for _ in range(args.warmup_frames):
        ok, probe = capture.read()
        if not ok:
            break
    if not ok:
        print(f"Camera {args.camera} opened but returned no frame.")
        capture.release()
        return 1
    brightness = float(probe.mean())
    if brightness < 20:
        print(
            f"  warning: the camera is returning a near-black image "
            f"(brightness {brightness:.0f}/255). Check for a privacy shutter, "
            f"Windows camera permissions, or another app holding the camera -- "
            f"nothing can be detected in a black frame."
        )
    height, width = probe.shape[:2]
    imgsz = args.imgsz or min(config.detection.imgsz, 2 * max(width, height))
    if imgsz != config.detection.imgsz:
        from dataclasses import replace as _replace

        config = _replace(config, detection=_replace(config.detection, imgsz=imgsz))
        logger.info(
            "Frames are %dx%d; using imgsz=%d instead of %d.",
            width, height, imgsz, CONFIG.detection.imgsz,
        )
    print(f"Camera {args.camera}: {width}x{height}, detecting at imgsz={imgsz}")

    detector = _build_detector(config)
    headpose = _build_headpose_estimator(config)
    posture = _build_posture_analyzer(config)
    expression = _build_expression_recognizer(config)
    temporal = TemporalTracker(config)

    print("Session running. Press q in the window (or Ctrl-C) to end it.\n")
    started = time.time()
    frame_id = 0
    fps = 0.0
    seen: set[int] = set()

    with (
        _build_face_analyzer(config) as faces,
        frames_path.open("w", encoding="utf-8") as frames_out,
        graph_path.open("w", encoding="utf-8") as graph_out,
    ):
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                tick = time.time()
                elapsed = tick - started
                if args.seconds and elapsed >= args.seconds:
                    break

                persons, objects = detector.detect(frame)
                boxes = [p.bbox for p in persons]
                face_results = faces.analyze(frame, boxes)

                # Identity: a direct gallery lookup, decidable this frame.
                person_ids: list[int | None] = []
                for result in face_results:
                    hit = (
                        gallery.identify(result.embedding)
                        if result.embedding is not None
                        and (result.score or 0.0) >= config.identity.min_face_score_for_identity
                        else None
                    )
                    if hit is None:
                        person_ids.append(UNKNOWN_PERSON_ID if result.face_bbox else None)
                    else:
                        person_ids.append(hit[0].person_id)
                        seen.add(hit[0].person_id)

                face_boxes = [r.face_bbox for r in face_results]
                record = _assemble_frame(
                    frame_id,
                    int(elapsed * 1000),
                    persons,
                    face_results,
                    headpose.estimate(frame, face_boxes),
                    posture.analyze(frame, boxes),
                    expression.classify(frame, face_boxes, [r.kps for r in face_results]),
                    [None] * len(persons),
                    [None] * len(persons),
                    person_ids,
                    objects,
                )
                graph = temporal.update_frame(generate_scene_graph(record, config))

                frames_out.write(json.dumps(record) + "\n")
                graph_out.write(json.dumps(graph) + "\n")

                frame_id += 1
                fps = 0.9 * fps + 0.1 * (1.0 / max(time.time() - tick, 1e-6)) if fps else 1.0 / max(time.time() - tick, 1e-6)
                if not args.no_window:
                    cv2.imshow("ClassGraph live", _draw(frame, record, graph, names, fps))
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                elif frame_id % 30 == 0:
                    print(f"  {frame_id} frames, {len(seen)} registered students seen, {fps:.1f} fps")
        except KeyboardInterrupt:
            print("\nstopped.")
        finally:
            capture.release()
            if not args.no_window:
                cv2.destroyAllWindows()

    if frame_id == 0:
        print("No frames captured; nothing written.")
        return 1

    profiles = build_profiles(graph_path, config)
    profiles_path.write_text(
        json.dumps(list(profiles.values()), indent=2), encoding="utf-8"
    )

    students = [p for p in profiles.values() if p.get("is_student")]
    print(f"\n{frame_id} frames in {time.time() - started:.0f}s")
    print(f"per-frame records : {frames_path}")
    print(f"scene graph       : {graph_path}")
    print(f"student profiles  : {profiles_path}\n")

    if not students:
        print("No student was recognised for long enough to profile.")
        return 0

    print(f"{'id':>4} {'name':<14} {'frames':>7} {'attention':<26} {'concentration':>13} {'emotion':<18}")
    for profile in sorted(students, key=lambda p: p["person_id"]):
        pid = profile["person_id"]
        pct = profile["concentration"]["behavioral_proxy_pct"]
        top_gaze = ", ".join(
            f"{k} {v}" for k, v in sorted(
                profile["attention"]["counts"].items(), key=lambda kv: -kv[1]
            )[:3]
        )
        top_expr = ", ".join(
            f"{k} {v}" for k, v in sorted(
                profile["expression"]["counts"].items(), key=lambda kv: -kv[1]
            )[:2]
        )
        print(f"{pid:>4} {names.get(pid, 'unknown'):<14} {profile['frames_seen']:>7} "
              f"{top_gaze:<26} {('n/a' if pct is None else f'{pct:.0f}%'):>13} {top_expr:<18}")
    return 0


def main() -> int:
    """Parse arguments and run a live session.

    Returns:
        A process exit code.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument(
        "--gallery",
        default="outputs/enrollment/gallery.json",
        help="Registered-student gallery from tools/register_faces.py.",
    )
    parser.add_argument("--out", default="outputs/live", help="Output directory.")
    parser.add_argument(
        "--seconds", type=float, default=None, help="Stop after this many seconds."
    )
    parser.add_argument(
        "--no-window", action="store_true", help="Run headless, without a preview window."
    )
    parser.add_argument(
        "--yaw-reference",
        type=float,
        default=None,
        help="Per-camera gaze reference in degrees, from tools/calibrate_gaze.py. "
             "A webcam facing the student head-on needs 0 (the default); a "
             "camera mounted off to one side does not, and leaving it wrong "
             "makes every attention figure measure the camera angle instead.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=None,
        help="Detector input size. Default caps the upscale at 2x the frame, "
             "because the config's 1920 loses the person entirely on a 640px "
             "webcam frame.",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=20,
        help="Frames to discard while the camera settles its exposure.",
    )
    parser.add_argument("--verbose", action="store_true", help="Show library logging.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
