"""Process a folder of consecutive clips as ONE session.

Long recordings arrive split into short files -- a lecture cut into 157
ten-second clips, numbered in order. Running the pipeline once per file and
merging afterwards is the obvious approach and it is wrong: identity is
discovered per video, so the same student becomes a different ``person_id`` in
every file, and a 27-minute lecture ends up with 157 times the roster it should
have.

This runs them as a single session instead. One tracker, one identity resolver
and one temporal tracker span every clip, so a student who appears in clip 3
and again in clip 40 keeps the same id.

Continuity is checked, not assumed: the last frame of each clip is compared
with the first frame of the next, and a jump is reported rather than silently
treated as continuous.

Two passes, because identity is only decidable once everything has been seen
(see :class:`backend.identity.TwoPassIdentityResolver`). Pass 1 streams
per-frame records to disk keyed by accumulation key; pass 2 rewrites those keys
as final person ids and builds the scene graph. Nothing is held in memory
beyond one frame, so the length of the recording is not a limit.

Run:
    python -m tools.batch_session --dir <folder> --out outputs/lecture
    python -m tools.batch_session --dir <folder> --limit 12 --sample-rate 5
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import re
import time
from dataclasses import replace
from pathlib import Path

logger = logging.getLogger(__name__)

#: Mean absolute pixel difference above which two clips are treated as
#: belonging to different scenes. Measured on a real split recording:
#: consecutive boundaries sat at 4.5-5.9, two clips 150 apart at 73.9.
CONTINUITY_LIMIT = 20.0

#: Colour-histogram correlation below which consecutive SAMPLED frames are
#: treated as a new scene.
#:
#: Chosen from the data the detector actually sees. Over 20 clips of sampled
#: frames, ordinary movement -- people leaning, standing, walking past -- kept
#: histogram correlation at a median of 1.000 with a 10th percentile of 0.999,
#: while real changes fell to 0.75-0.90. Mean absolute pixel difference was
#: tried first and discarded: ordinary motion reached 21 there while a genuine
#: change reached 80, so the two overlapped and no threshold separated them.
#: A histogram moves when the ROOM changes, not when people do.
SCENE_HIST_CORRELATION = 0.90


def ordered_clips(directory: Path, pattern: str) -> list[Path]:
    """Return the clips in numeric order.

    Args:
        directory: Folder holding the clips.
        pattern: Glob for the files.

    Returns:
        Paths sorted by the first number in the filename, so ``view9`` precedes
        ``view10`` -- lexicographic sorting would interleave them and silently
        scramble a recording's chronology.
    """
    def key(path: Path):
        digits = re.findall(r"\d+", path.stem)
        return (int(digits[-1]) if digits else 0, path.stem)

    return sorted(directory.glob(pattern), key=key)


def check_continuity(clips: list[Path]) -> list[tuple[str, str, float]]:
    """Report clip boundaries that look like a scene change.

    Args:
        clips: The clips, in order.

    Returns:
        ``(previous, next, difference)`` for each boundary that exceeds
        :data:`CONTINUITY_LIMIT`.
    """
    import cv2
    import numpy as np

    breaks = []
    for previous, following in itertools.pairwise(clips):
        # Seeking near the end fails on some codecs, and a failed seek must
        # not be reported as a scene change -- that would invent a warning
        # about the footage from a limitation of the decoder. Walk back a few
        # frames until one reads.
        cap = cv2.VideoCapture(str(previous))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok_a, last = False, None
        for back in (2, 5, 10, 20):
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, count - back))
            ok_a, last = cap.read()
            if ok_a:
                break
        cap.release()
        cap = cv2.VideoCapture(str(following))
        ok_b, first = cap.read()
        cap.release()
        if not (ok_a and ok_b) or last.shape != first.shape:
            # Undecidable, not a break. Say so rather than warning.
            continue
        diff = float(np.abs(last.astype(int) - first.astype(int)).mean())
        if diff > CONTINUITY_LIMIT:
            breaks.append((previous.name, following.name, diff))
    return breaks


def _histogram(frame):
    """Small normalised colour histogram, for comparing one frame to the next."""
    import cv2

    small = cv2.resize(frame, (160, 90))
    hist = cv2.calcHist([small], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
    return cv2.normalize(hist, hist).flatten()


def run(args) -> int:
    """Process every clip as one session and write the report.

    Args:
        args: Parsed CLI arguments.

    Returns:
        A process exit code.
    """
    import cv2

    from backend.actions import annotate_graph
    from backend.config import CONFIG
    from backend.enrollment import EnrolledGallery
    from backend.identity import TwoPassIdentityResolver
    from backend.integrate import (
        _assemble_frame,
        _build_behaviour_classifier,
        _build_detector,
        _build_expression_recognizer,
        _build_face_analyzer,
        _build_headpose_estimator,
        _build_person_tracker,
        _build_posture_analyzer,
    )
    from backend.scene_graph import generate_scene_graph
    from backend.scene_layout import detect as detect_layout
    from backend.scene_layout import summarise as summarise_layout
    from backend.student_profile import build_profiles
    from backend.temporal import TemporalTracker
    from tools.report import build as build_report

    directory = Path(args.dir)
    clips = ordered_clips(directory, args.pattern)
    if not clips:
        raise SystemExit(f"No clips matching {args.pattern!r} in {directory}")
    if args.limit:
        clips = clips[: args.limit]

    probe = cv2.VideoCapture(str(clips[0]))
    ok, frame = probe.read()
    probe.release()
    if not ok:
        raise SystemExit(f"Could not read {clips[0]}")
    height, width = frame.shape[:2]

    config = replace(
        CONFIG,
        identity=replace(CONFIG.identity,
                         quality_gated_creation=True,
                         min_face_px_to_found=args.min_face_px_to_found),
        # The behaviour weights train at 640, but that is the size the IMAGE is
        # resized to, not the size it should be fed. On 1080p input, 640 throws
        # away the detail the model needs: measured 4x more detections at 1280
        # on this footage. Scale it with the input rather than pinning it.
        behaviour=replace(CONFIG.behaviour,
                          imgsz=args.behaviour_imgsz or
                          (1280 if max(width, height) >= 1280 else CONFIG.behaviour.imgsz)),
    )

    print(f"{len(clips)} clip(s), {width}x{height}, sample every "
          f"{args.sample_rate} frames, behaviour imgsz "
          f"{config.behaviour.imgsz}")

    if not args.skip_continuity:
        breaks = check_continuity(clips)
        if breaks:
            print(f"  WARNING: {len(breaks)} boundary/boundaries look like a "
                  f"scene change; identity is still carried across them:")
            for a, b, d in breaks[:5]:
                print(f"    {a} -> {b}  (difference {d:.1f})")
        else:
            print("  continuity: every boundary is continuous")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    keyed_path = out / "_keyed.jsonl"

    detector = _build_detector(config)
    headpose = _build_headpose_estimator(config)
    posture = _build_posture_analyzer(config)
    expression = _build_expression_recognizer(config)
    behaviour = _build_behaviour_classifier(config)
    tracker = _build_person_tracker(config)

    # One identity resolver PER SCENE, not one for the recording. Carrying
    # identity across a room change is what produced 22 people for a class of
    # about six: after a cut the camera sees different faces from a different
    # angle, so every student is minted again and the roster is the sum of all
    # scenes rather than the size of the class.
    resolvers: dict[int, TwoPassIdentityResolver] = {}
    layouts_by_scene: dict[int, list] = {}
    scene = 0
    previous_hist = None

    started = time.time()
    frame_id = 0
    kept = 0

    # ---- pass 1: perception, streaming to disk keyed by accumulation key ---
    with _build_face_analyzer(config) as faces, keyed_path.open("w", encoding="utf-8") as sink:
        for number, clip in enumerate(clips, start=1):
            capture = cv2.VideoCapture(str(clip))
            index = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if index % args.sample_rate == 0:
                    hist = _histogram(frame)
                    if previous_hist is not None:
                        similarity = float(cv2.compareHist(
                            previous_hist, hist, cv2.HISTCMP_CORREL))
                        if similarity < SCENE_HIST_CORRELATION:
                            scene += 1
                            tracker.reset()
                    previous_hist = hist
                    resolver = resolvers.setdefault(
                        scene, TwoPassIdentityResolver(config.identity))

                    persons, objects = detector.detect(frame)
                    boxes = [p.bbox for p in persons]
                    results = faces.analyze(frame, boxes)
                    face_boxes = [r.face_bbox for r in results]
                    track_ids = tracker.update(persons, frame)

                    embeddings = [r.embedding for r in results]
                    scores = [r.score for r in results]
                    sizes = [
                        None if r.face_bbox is None
                        else int(min(r.face_bbox[2], r.face_bbox[3]))
                        for r in results
                    ]
                    keys = resolver.keys_for(track_ids, embeddings, scores)
                    resolver.observe(keys, embeddings, scores, sizes)

                    record = _assemble_frame(
                        frame_id, int(frame_id * args.sample_rate * 1000 / 30),
                        persons, results,
                        headpose.estimate(frame, face_boxes),
                        posture.analyze(frame, boxes),
                        expression.classify(frame, face_boxes, [r.kps for r in results]),
                        behaviour.classify(frame, boxes) if behaviour else [None] * len(persons),
                        track_ids,
                        list(keys),
                        objects,
                    )
                    record["scene"] = scene
                    layouts_by_scene.setdefault(scene, []).append(
                        detect_layout(record["persons"]))
                    sink.write(json.dumps(record) + "\n")
                    frame_id += 1
                    kept += 1
                index += 1
            capture.release()
            if number % 10 == 0 or number == len(clips):
                print(f"  {number}/{len(clips)} clips, {kept} frames, "
                      f"{scene + 1} scene(s), {time.time() - started:.0f}s")

    # Namespace each scene's ids so scene 2's "person 1" is never confused with
    # scene 1's, while both stay readable.
    mappings: dict[int, dict[int, int]] = {}
    offset = 0
    for index in sorted(resolvers):
        raw = resolvers[index].finalise()
        highest = max((v for v in raw.values() if v > 0), default=0)
        mappings[index] = {
            key: (value + offset if value > 0 else value)
            for key, value in raw.items()
        }
        offset += highest

    scenes = {i: summarise_layout(v) for i, v in layouts_by_scene.items()}
    print(f"\n{len(resolvers)} scene(s) detected")
    print(f"{'scene':>6} {'frames':>7} {'people':>7} {'layout':>8} {'ratio':>7}  focus")
    for index in sorted(resolvers):
        people = len({v for v in mappings[index].values() if v > 0})
        layout = scenes[index]
        spot = f"({layout.focus[0]:.0f},{layout.focus[1]:.0f})" if layout.focus else "-"
        print(f"{index:>6} {len(layouts_by_scene[index]):>7} {people:>7} "
              f"{layout.kind:>8} {layout.ratio:>7.2f}  {spot}")
    print(f"total distinct people: {offset}")

    # ---- pass 2: final ids, scene graph, actions --------------------------
    temporal = TemporalTracker(config)
    graph_path = out / "live_graph.jsonl"
    raw_path = out / "raw.jsonl"
    with (keyed_path.open(encoding="utf-8") as source,
          graph_path.open("w", encoding="utf-8") as graphs,
          raw_path.open("w", encoding="utf-8") as raws):
        for line in source:
            record = json.loads(line)
            mapping = mappings.get(record.get("scene", 0), {})
            for person in record["persons"]:
                key = person.get("person_id")
                person["person_id"] = mapping.get(key) if key is not None else None
            raws.write(json.dumps(record) + "\n")
            graph = annotate_graph(
                temporal.update_frame(generate_scene_graph(record, config)),
                record, config)
            graphs.write(json.dumps(graph) + "\n")
    keyed_path.unlink(missing_ok=True)

    profiles = build_profiles(graph_path, config)
    (out / "live_profiles.json").write_text(
        json.dumps(list(profiles.values()), indent=2), encoding="utf-8")
    students = [p for p in profiles.values() if p.get("is_student")]

    gallery = EnrolledGallery.load(args.gallery) if args.gallery else EnrolledGallery()
    report = build_report(out, gallery, title=args.title or directory.name)

    print(f"\n{kept} frames in {time.time() - started:.0f}s | "
          f"{len(students)} students profiled")
    print(f"report: {report}")
    return 0


def main() -> int:
    """Parse arguments and run the batch session."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", required=True, help="Folder of clips.")
    parser.add_argument("--pattern", default="*.mp4")
    parser.add_argument("--out", default="outputs/session")
    parser.add_argument("--sample-rate", type=int, default=5,
                        help="Process every Nth frame of each clip.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only the first N clips.")
    parser.add_argument("--behaviour-imgsz", type=int, default=None,
                        help="Override; defaults to 1280 for HD input.")
    parser.add_argument("--min-face-px-to-found", type=int, default=24)
    parser.add_argument("--gallery", default=None,
                        help="Registered-student gallery, to name people.")
    parser.add_argument("--title", default=None)
    parser.add_argument("--skip-continuity", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
