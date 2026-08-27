"""ClassGraph as one window: register students, run a session, see the results.

Everything happens inside a single application window -- there is no typing in
a terminal half way through, and no separate commands to chain. Three screens:

    HOME      who is registered; start a session or add a student
    REGISTER  type a name in the window, hold still, it captures the face
    SESSION   live view with a per-student panel updating as it runs

When the session ends it writes the scene graph and profiles and opens an HTML
report in the browser.

Speed
-----

The offline pipeline is configured for accuracy on recorded video: MediaPipe in
``static_image_mode`` (full detection every frame, no tracking), refined face
landmarks, and a detector input size of 1920 tuned for wide classroom shots.
Measured on one 640x480 webcam frame that costs 233 ms -- 4.3 fps, which does
not feel like a live application.

Live settings and a running cadence bring that down without changing what is
measured: the detector runs at the frame's own resolution, MediaPipe tracks
between frames instead of re-detecting, and the stages whose output cannot
meaningfully change in 60 ms -- expression, head pose, posture -- run every few
frames and hold their last value in between. Identity and detection run every
frame, because those are what the overlay is drawn from.

Privacy: a gallery is biometric data about named people (see
backend/enrollment.py). Frames are never written to disk.

Run:
    python -m tools.classroom_demo
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import webbrowser
from collections import Counter, defaultdict, deque
from dataclasses import replace
from pathlib import Path

import numpy as np

from tools import ui

logger = logging.getLogger(__name__)

VIDEO_W, VIDEO_H = 860, 645
SIDE_W = 380
WIN_W, WIN_H = VIDEO_W + SIDE_W, VIDEO_H
SHOTS = 8
UNKNOWN = -1

HOME, REGISTER, SESSION = "home", "register", "session"


def live_config(config, frame_shape, imgsz=None):
    """Return ``config`` retuned for live webcam frames.

    Args:
        config: The base pipeline config.
        frame_shape: ``(h, w, ...)`` of the camera frames.
        imgsz: Explicit detector size, or ``None`` to derive one.

    Returns:
        A new config. The detector size matters most: 1920 on a 640px frame
        upscales 3x and loses the person entirely (measured: found at 416-1600,
        not found at 1920), so it is capped at the frame's longer side.
    """
    h, w = frame_shape[:2]
    size = imgsz or min(config.detection.imgsz, max(w, h))
    return replace(
        config,
        detection=replace(config.detection, imgsz=size),
        posture=replace(config.posture, static_image_mode=False, model_complexity=0),
        face=replace(config.face, static_image_mode=False, refine_landmarks=False),
    )


class Engine:
    """Runs the perception stack over live frames on a cadence."""

    def __init__(self, config, gallery, stride=3):
        """Build every model once.

        Args:
            config: A live-tuned config.
            gallery: The registered students.
            stride: Run expression / head pose / posture every ``stride``
                frames, holding the previous result in between.
        """
        from backend.integrate import (
            _build_detector,
            _build_expression_recognizer,
            _build_headpose_estimator,
            _build_posture_analyzer,
        )
        from backend.scene_graph import generate_scene_graph
        from backend.temporal import TemporalTracker

        self.config = config
        self.gallery = gallery
        self.stride = max(1, stride)
        self.detector = _build_detector(config)
        self.headpose = _build_headpose_estimator(config)
        self.posture = _build_posture_analyzer(config)
        self.expression = _build_expression_recognizer(config)
        self.temporal = TemporalTracker(config)
        self._graph_of = generate_scene_graph
        self._held: dict = {}
        self.index = 0

    def step(self, faces_module, frame, elapsed_ms):
        """Process one frame.

        Args:
            faces_module: A live :class:`backend.face.FaceAnalyzer`.
            frame: The BGR frame.
            elapsed_ms: Milliseconds since the session started.

        Returns:
            ``(record, graph, person_ids)`` for this frame.
        """
        from backend.integrate import _assemble_frame

        persons, objects = self.detector.detect(frame)
        boxes = [p.bbox for p in persons]
        faces = faces_module.analyze(frame, boxes)

        person_ids = []
        floor = self.config.identity.min_face_score_for_identity
        for face in faces:
            usable = face.embedding is not None and (face.score or 0.0) >= floor
            hit = self.gallery.identify(face.embedding) if usable else None
            if hit is None:
                person_ids.append(UNKNOWN if face.face_bbox else None)
            else:
                person_ids.append(hit[0].person_id)

        face_boxes = [f.face_bbox for f in faces]
        heavy = self.index % self.stride == 0 or self._held.get("n") != len(persons)
        if heavy:
            self._held = {
                "n": len(persons),
                "pose": self.headpose.estimate(frame, face_boxes),
                "posture": self.posture.analyze(frame, boxes),
                "expr": self.expression.classify(frame, face_boxes, [f.kps for f in faces]),
            }
        held = self._held

        record = _assemble_frame(
            self.index, int(elapsed_ms), persons, faces,
            held["pose"], held["posture"], held["expr"],
            [None] * len(persons), [None] * len(persons), person_ids, objects,
        )
        graph = self.temporal.update_frame(self._graph_of(record, self.config))
        self.index += 1
        return record, graph, person_ids


class Tally:
    """Live per-student counters, for the session panel."""

    def __init__(self):
        self.gaze = defaultdict(Counter)
        self.expr = defaultdict(Counter)
        self.frames = Counter()
        self.recent = defaultdict(lambda: deque(maxlen=90))
        self.last_seen = {}

    def update(self, graph, index):
        """Fold one frame's graph into the running totals."""
        for node in graph.get("nodes", []):
            pid = node.get("person_id")
            if pid is None or pid <= 0:
                continue
            feat = node.get("features") or {}
            self.frames[pid] += 1
            self.last_seen[pid] = index
            gaze = feat.get("gaze_label")
            if gaze:
                self.gaze[pid][gaze] += 1
            expr = feat.get("expression")
            if expr:
                self.expr[pid][expr] += 1
            self.recent[pid].append(1.0 if gaze == "teacher" else 0.0)

    def attention(self, pid):
        """Share of graded frames this student looked at the teacher."""
        counts = self.gaze[pid]
        total = sum(counts.values())
        return counts["teacher"] / total if total else None


# --------------------------------------------------------------------------- #
# screens
# --------------------------------------------------------------------------- #


def draw_home(canvas, gallery, message):
    """Draw the home screen onto ``canvas``."""
    canvas.text(48, 44, "ClassGraph", size=38, bold=True)
    canvas.text(48, 92, "Attention, posture and emotion, per registered student",
                size=17, colour=ui.MUTED)

    canvas.text(48, 156, f"REGISTERED  ({len(gallery)})", size=13, colour=ui.MUTED, bold=True)
    y = 186
    if not len(gallery):
        canvas.text(48, y, "Nobody yet — press R to add the first student.",
                    size=17, colour=ui.MUTED)
    for person in gallery.people[:9]:
        canvas.rect(48, y, WIN_W - 96, 44, ui.PANEL_2, radius=8)
        canvas.rect(48, y, 4, 44, ui.ACCENT)
        canvas.text(68, y + 22, person.name, size=18, anchor="lm")
        canvas.text(WIN_W - 68, y + 22, f"#{person.person_id}  ·  {person.shots} shots",
                    size=14, colour=ui.MUTED, anchor="rm")
        y += 52

    canvas.rect(0, WIN_H - 84, WIN_W, 84, ui.PANEL_2)
    for i, (key, label) in enumerate((("R", "register a student"),
                                      ("S", "start session"),
                                      ("Q", "quit"))):
        x = 48 + i * 260
        canvas.rect(x, WIN_H - 56, 30, 30, ui.LINE, radius=6)
        canvas.text(x + 15, WIN_H - 41, key, size=15, bold=True, anchor="mm")
        canvas.text(x + 44, WIN_H - 41, label, size=16, colour=ui.MUTED, anchor="lm")
    if message:
        canvas.text(48, WIN_H - 108, message, size=15, colour=ui.ACCENT)


def draw_register(canvas, frame, name, captured, status, ready):
    """Draw the registration screen."""
    canvas.image[0:VIDEO_H, 0:VIDEO_W] = ui.fit(frame, VIDEO_W, VIDEO_H)
    canvas.rect(VIDEO_W, 0, SIDE_W, WIN_H, ui.PANEL)

    x = VIDEO_W + 28
    canvas.text(x, 44, "Register", size=28, bold=True)
    canvas.text(x, 84, "Type the student's name, then hold still.",
                size=14, colour=ui.MUTED)

    canvas.text(x, 140, "NAME", size=12, colour=ui.MUTED, bold=True)
    canvas.rect(x, 162, SIDE_W - 56, 46, ui.PANEL_2, radius=8)
    caret = "|" if int(time.time() * 2) % 2 == 0 else " "
    canvas.text(x + 14, 185, (name or "") + caret, size=19, anchor="lm")

    canvas.text(x, 244, "CAPTURE", size=12, colour=ui.MUTED, bold=True)
    canvas.bar(x, 268, SIDE_W - 56, 10, captured / SHOTS,
               ui.ACCENT if ready else ui.DIM)
    canvas.text(x, 296, f"{captured} of {SHOTS} shots", size=15, colour=ui.MUTED)

    canvas.rect(x, 344, SIDE_W - 56, 68, ui.PANEL_2, radius=8)
    canvas.text(x + 16, 378, status, size=15,
                colour=ui.ACCENT if ready else ui.WARN, anchor="lm")

    canvas.text(x, WIN_H - 96, "ENTER   save", size=14, colour=ui.MUTED)
    canvas.text(x, WIN_H - 72, "ESC     cancel", size=14, colour=ui.MUTED)


def draw_session(canvas, frame, record, graph, tally, names, fps, seconds):
    """Draw the live session screen with the per-student panel."""
    scaled = ui.fit(frame, VIDEO_W, VIDEO_H)
    sx = VIDEO_W / frame.shape[1]
    sy = VIDEO_H / frame.shape[0]
    scale = min(sx, sy)
    ox = (VIDEO_W - frame.shape[1] * scale) / 2
    oy = (VIDEO_H - frame.shape[0] * scale) / 2
    canvas.image[0:VIDEO_H, 0:VIDEO_W] = scaled

    feats = {n.get("person_id"): (n.get("features") or {}) for n in graph.get("nodes", [])}
    for person in record["persons"]:
        pid = person.get("person_id")
        bx, by, bw, bh = person["bbox"]
        x, y = ox + bx * scale, oy + by * scale
        w, h = bw * scale, bh * scale
        known = pid is not None and pid > 0
        gaze = (feats.get(pid) or {}).get("gaze_label")
        colour = ui.GAZE_COLOURS.get(gaze, ui.ACCENT if known else ui.BAD)
        canvas.outline(x, y, w, h, colour, 2)
        label = names.get(pid, "unknown") if known else "unknown"
        tw = max(96, 9 * len(label) + 24)
        canvas.rect(x, max(0, y - 30), tw, 28, colour, alpha=0.92, radius=6)
        canvas.text(x + 10, max(0, y - 30) + 14, label, size=15, bold=True,
                    colour=(20, 20, 20), anchor="lm")

    canvas.rect(VIDEO_W, 0, SIDE_W, WIN_H, ui.PANEL)
    x = VIDEO_W + 24
    canvas.text(x, 34, "Live session", size=24, bold=True)
    canvas.text(x, 66, f"{seconds:0.0f}s   ·   {fps:0.1f} fps   ·   "
                       f"{len(record['persons'])} in frame",
                size=14, colour=ui.MUTED)

    y = 108
    for pid, name in sorted(names.items()):
        present = tally.last_seen.get(pid, -99) >= record["frame_id"] - 3
        pct = tally.attention(pid)
        canvas.rect(x, y, SIDE_W - 48, 118, ui.PANEL_2, radius=10)
        canvas.rect(x, y, 4, 118, ui.ACCENT if present else ui.DIM)
        canvas.text(x + 18, y + 26, name, size=17, bold=True,
                    colour=ui.INK if present else ui.MUTED, anchor="lm")
        canvas.text(x + SIDE_W - 66, y + 26,
                    "—" if pct is None else f"{pct * 100:.0f}%",
                    size=20, bold=True, anchor="rm",
                    colour=ui.ACCENT if (pct or 0) >= .7 else ui.WARN if pct else ui.MUTED)

        gaze = tally.gaze[pid]
        if sum(gaze.values()):
            parts = [(v, ui.GAZE_COLOURS.get(k, ui.DIM)) for k, v in gaze.most_common()]
            canvas.stacked_bar(x + 18, y + 46, SIDE_W - 84, 7, parts)
            canvas.text(x + 18, y + 66, "  ".join(f"{k} {v}" for k, v in gaze.most_common(3)),
                        size=12, colour=ui.MUTED)
        expr = tally.expr[pid]
        canvas.text(x + 18, y + 86,
                    "  ".join(f"{k} {v}" for k, v in expr.most_common(2)) or "no expression yet",
                    size=12, colour=ui.MUTED)
        canvas.sparkline(x + 18, y + 96, SIDE_W - 84, 16,
                         list(tally.recent[pid]), ui.ACCENT if present else ui.DIM)
        y += 128
        if y > WIN_H - 150:
            break

    canvas.rect(0, WIN_H - 46, WIN_W, 46, ui.PANEL_2)
    canvas.text(24, WIN_H - 23, "Q  end session and open the report",
                size=15, colour=ui.MUTED, anchor="lm")


# --------------------------------------------------------------------------- #


def run(args) -> int:
    """Run the application loop.

    Args:
        args: Parsed CLI arguments.

    Returns:
        A process exit code.
    """
    import cv2

    from backend.config import CONFIG
    from backend.enrollment import EnrolledGallery
    from backend.face import FaceAnalyzer
    from backend.student_profile import build_profiles

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gallery = EnrolledGallery.load(args.gallery, CONFIG.identity)

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        print(f"Could not open camera {args.camera}.")
        return 1
    for _ in range(args.warmup_frames):
        ok, frame = capture.read()
    if not ok:
        print("Camera opened but returned no frames.")
        return 1
    config = live_config(CONFIG, frame.shape, args.imgsz)
    if args.yaw_reference is not None:
        config = replace(config, headpose=replace(
            config.headpose, yaw_reference_deg=args.yaw_reference))
    print(f"Camera {args.camera}: {frame.shape[1]}x{frame.shape[0]}, "
          f"imgsz={config.detection.imgsz}, stride={args.stride}")
    if float(frame.mean()) < 20:
        print("  warning: the camera image is near-black — check for a privacy "
              "shutter or camera permissions.")

    window = "ClassGraph"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)

    analyzer = FaceAnalyzer(config.face)
    analyzer.__enter__()
    engine = None
    state = HOME
    autostart = args.auto_start and len(gallery) > 0
    message = ""
    name = ""
    shots: list = []
    tally = Tally()
    started = 0.0
    fps = 0.0
    frames_out = graph_out = None
    code = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            tick = time.perf_counter()
            base = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
            base[:] = ui.PANEL
            canvas = ui.Canvas(base)

            if state == HOME and autostart:
                autostart = False
                engine = Engine(config, gallery, args.stride)
                tally = Tally()
                frames_out = (out / "live_frames.jsonl").open("w", encoding="utf-8")
                graph_out = (out / "live_graph.jsonl").open("w", encoding="utf-8")
                started = time.time()
                state = SESSION

            if state == HOME:
                draw_home(canvas, gallery, message)

            elif state == REGISTER:
                faces = analyzer.detect_faces(frame)
                good = [
                    f for f in faces
                    if f.embedding is not None
                    and f.score >= config.identity.min_face_score_for_identity
                    and min(f.bbox[2], f.bbox[3]) >= args.min_face_px
                ]
                ready = len(good) == 1 and bool(name.strip())
                if ready and len(shots) < SHOTS:
                    shots.append(good[0].embedding)
                if not name.strip():
                    status = "Type a name to begin"
                elif not good:
                    status = "No face — move closer, face the camera"
                elif len(good) > 1:
                    status = f"{len(good)} faces — one person at a time"
                elif len(shots) >= SHOTS:
                    status = "Done — press ENTER to save"
                else:
                    status = "Hold still…"
                for f in faces:
                    bx, by, bw, bh = f.bbox
                    s = min(VIDEO_W / frame.shape[1], VIDEO_H / frame.shape[0])
                    ox = (VIDEO_W - frame.shape[1] * s) / 2
                    oy = (VIDEO_H - frame.shape[0] * s) / 2
                    canvas.outline(ox + bx * s, oy + by * s, bw * s, bh * s,
                                   ui.ACCENT if ready else ui.WARN, 2)
                draw_register(canvas, frame, name, len(shots), status, ready)

            elif state == SESSION:
                elapsed = time.time() - started
                record, graph, _ = engine.step(analyzer, frame, elapsed * 1000)
                tally.update(graph, record["frame_id"])
                frames_out.write(json.dumps(record) + "\n")
                graph_out.write(json.dumps(graph) + "\n")
                names = {p.person_id: p.name for p in gallery.people}
                draw_session(canvas, frame, record, graph, tally, names, fps, elapsed)
                if args.seconds and elapsed >= args.seconds:
                    key = ord("q")
                    cv2.imshow(window, canvas.finish())
                    cv2.waitKey(1)
                    frames_out.close(); graph_out.close()
                    state = HOME
                    message = _finish(out, gallery, args, build_profiles)
                    print(message)
                    if args.exit_after_session:
                        break
                    continue

            cv2.imshow(window, canvas.finish())
            key = cv2.waitKey(1) & 0xFF
            dt = time.perf_counter() - tick
            fps = (0.85 * fps + 0.15 / dt) if fps else 1 / max(dt, 1e-6)

            if key == 255:
                continue

            if state == HOME:
                if key in (ord("q"), 27):
                    break
                if key == ord("r"):
                    state, name, shots, message = REGISTER, "", [], ""
                elif key == ord("s"):
                    if not len(gallery):
                        message = "Register at least one student first (R)."
                        continue
                    engine = Engine(config, gallery, args.stride)
                    tally = Tally()
                    frames_out = (out / "live_frames.jsonl").open("w", encoding="utf-8")
                    graph_out = (out / "live_graph.jsonl").open("w", encoding="utf-8")
                    started = time.time()
                    state = SESSION

            elif state == REGISTER:
                if key == 27:
                    state, message = HOME, ""
                elif key in (13, 10):
                    if len(shots) >= 3 and name.strip():
                        person = gallery.register(name.strip(), shots)
                        gallery.save(args.gallery)
                        message = f"{person.name} registered as #{person.person_id}."
                        state = HOME
                elif key == 8:
                    name, shots = name[:-1], []
                elif 32 <= key <= 126:
                    name, shots = name + chr(key), []

            elif state == SESSION and key in (ord("q"), 27):
                frames_out.close(); graph_out.close()
                state = HOME
                message = _finish(out, gallery, args, build_profiles)

    except KeyboardInterrupt:
        code = 0
    finally:
        for handle in (frames_out, graph_out):
            if handle and not handle.closed:
                handle.close()
        analyzer.__exit__(None, None, None)
        capture.release()
        cv2.destroyAllWindows()
    return code


def _finish(out: Path, gallery, args, build_profiles) -> str:
    """Write profiles and the report at the end of a session.

    Args:
        out: Session output directory.
        gallery: The registered students.
        args: Parsed CLI arguments.
        build_profiles: ``backend.student_profile.build_profiles``.

    Returns:
        A short message for the home screen.
    """
    from tools.report import build

    graph_path = out / "live_graph.jsonl"
    if not graph_path.exists() or not graph_path.stat().st_size:
        return "Session ended with no frames."
    profiles = build_profiles(graph_path, None)
    (out / "live_profiles.json").write_text(
        json.dumps(list(profiles.values()), indent=2), encoding="utf-8")
    report = build(out, gallery)
    students = [p for p in profiles.values() if p.get("is_student")]
    if not args.no_browser:
        webbrowser.open(report.resolve().as_uri())
    return f"Session saved — {len(students)} student(s). Report opened."


def main() -> int:
    """Parse arguments and start the app."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--gallery", default="outputs/enrollment/gallery.json")
    parser.add_argument("--out", default="outputs/live")
    parser.add_argument("--stride", type=int, default=3,
                        help="Run expression/head-pose/posture every Nth frame.")
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--min-face-px", type=int, default=60)
    parser.add_argument("--warmup-frames", type=int, default=20)
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--yaw-reference", type=float, default=None)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--auto-start", action="store_true",
                        help="Go straight into a session if students are registered.")
    parser.add_argument("--exit-after-session", action="store_true",
                        help="Quit once the timed session finishes.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
