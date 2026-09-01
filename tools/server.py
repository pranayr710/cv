"""ClassGraph as a web dashboard: enrol students, run a session, watch it live.

Serves a single page that does the whole flow in a browser -- registration,
a live session with the video and per-student analytics streaming over a
WebSocket, and the finished report -- so a demonstration is one URL rather than
a sequence of commands.

Why a server rather than a desktop window
-----------------------------------------

The OpenCV window can draw a frame and little else: no charts, no scrolling
tables, no layout that survives a resize. Everything interesting about a
session is comparative -- this student against that one, now against a minute
ago -- and that is a job for a document, not a framebuffer. Streaming JPEG
frames plus a JSON payload per frame keeps the vision work in Python and lets
the page render what Python is bad at.

Camera ownership: one session at a time. The camera is opened when a session
starts and released when it stops, so registration and the session never
contend for the device.

Privacy: the gallery is biometric data about named people (see
backend/enrollment.py). Frames are streamed to the browser but never written to
disk; only derived per-frame records, the graph and the profiles are stored.

Run:
    python -m tools.server
    # then open http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import threading
import time
from collections import Counter, defaultdict, deque
from dataclasses import replace
from pathlib import Path

# Imported at module scope, not inside create_app, because this module uses
# `from __future__ import annotations`: every annotation becomes a string, and
# FastAPI resolves them against MODULE globals. A locally-imported WebSocket is
# invisible there, so FastAPI silently reclassifies the parameter as a missing
# query field and closes every handshake with a bare 403 -- no traceback, no
# log line, and the route still listed in app.routes.
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

STATIC = Path(__file__).resolve().parent / "static"
UNKNOWN = -1


class Session:
    """One live capture, owning the camera and every model."""

    def __init__(self, config, gallery, out: Path, camera: int, stride: int,
                 min_face_px: int = 60):
        """Prepare a session without opening the camera yet.

        Args:
            config: Live-tuned pipeline config.
            gallery: Registered students.
            out: Directory for the per-frame records, graph and profiles.
            camera: Camera index.
            stride: Run expression / head pose / posture every Nth frame.
            min_face_px: Face size below which identification is not expected
                to work, used only to warn the operator.
        """
        self.config = config
        self.gallery = gallery
        self.out = out
        self.camera = camera
        self.stride = max(1, stride)
        self.min_face_px = min_face_px
        self.running = False
        self.frame_jpeg: bytes | None = None
        self.payload: dict = {}
        self.started = 0.0
        self.frames = 0
        self.fps = 0.0
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self.gaze = defaultdict(Counter)
        self.expr = defaultdict(Counter)
        self.actions = defaultdict(Counter)
        self.recent = defaultdict(lambda: deque(maxlen=120))
        self.last_seen: dict[int, int] = {}
        self.pairs: Counter = Counter()
        #: track_id -> the person that track was last recognised as. A face is
        #: needed to LEARN who a track is, but not to keep knowing it.
        self.track_identity: dict[int, int] = {}

    def start(self):
        """Open the camera and begin processing in a background thread."""
        if self.running:
            return
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop processing and wait for the camera to be released."""
        self.running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def _loop(self):
        """Capture and process frames until stopped."""
        import cv2

        from backend.actions import annotate_graph
        from backend.face import FaceAnalyzer
        from backend.integrate import (
            _assemble_frame,
            _build_detector,
            _build_expression_recognizer,
            _build_headpose_estimator,
            _build_person_tracker,
            _build_posture_analyzer,
        )
        from backend.scene_graph import generate_scene_graph
        from backend.scene_layout import annotate as annotate_layout
        from backend.temporal import TemporalTracker

        capture = cv2.VideoCapture(self.camera)
        if not capture.isOpened():
            logger.error("Could not open camera %s", self.camera)
            self.running = False
            return
        for _ in range(20):
            capture.read()

        detector = _build_detector(self.config)
        headpose = _build_headpose_estimator(self.config)
        posture = _build_posture_analyzer(self.config)
        expression = _build_expression_recognizer(self.config)
        # Motion tracking carries identity across frames where the face cannot
        # be read -- a student turning to a neighbour, or away from the camera.
        # Without it every frame is judged alone and a known student becomes a
        # stranger the moment they look away, which is exactly when the system
        # most needs to keep following them.
        tracker = _build_person_tracker(self.config)
        temporal = TemporalTracker(self.config)
        names = {p.person_id: p.name for p in self.gallery.people}
        floor = self.config.identity.min_face_score_for_identity

        self.out.mkdir(parents=True, exist_ok=True)
        frames_file = (self.out / "live_frames.jsonl").open("w", encoding="utf-8")
        graph_file = (self.out / "live_graph.jsonl").open("w", encoding="utf-8")
        self.started = time.time()
        held: dict = {}
        index = 0

        try:
            with FaceAnalyzer(self.config.face) as analyzer:
                while self.running:
                    tick = time.perf_counter()
                    ok, frame = capture.read()
                    if not ok:
                        break

                    persons, objects = detector.detect(frame)
                    boxes = [p.bbox for p in persons]
                    faces = analyzer.analyze(frame, boxes)

                    track_ids = tracker.update(persons, frame)
                    person_ids = []
                    for face, track in zip(faces, track_ids):
                        usable = face.embedding is not None and (face.score or 0) >= floor
                        hit = self.gallery.identify(face.embedding) if usable else None
                        if hit is not None:
                            # A readable face teaches the track who it is.
                            if track is not None:
                                self.track_identity[track] = hit[0].person_id
                            person_ids.append(hit[0].person_id)
                            continue
                        # No usable face this frame: fall back to whoever this
                        # track was last recognised as.
                        remembered = self.track_identity.get(track) if track is not None else None
                        if remembered is not None:
                            person_ids.append(remembered)
                        else:
                            person_ids.append(UNKNOWN if face.face_bbox else None)

                    face_boxes = [f.face_bbox for f in faces]
                    if index % self.stride == 0 or held.get("n") != len(persons):
                        held = {
                            "n": len(persons),
                            "pose": headpose.estimate(frame, face_boxes),
                            "posture": posture.analyze(frame, boxes),
                            "expr": expression.classify(
                                frame, face_boxes, [f.kps for f in faces]),
                        }

                    elapsed = time.time() - self.started
                    record = _assemble_frame(
                        index, int(elapsed * 1000), persons, faces,
                        held["pose"], held["posture"], held["expr"],
                        [None] * len(persons), track_ids,
                        person_ids, objects,
                    )
                    # Layout first: actions read `oriented` from it.
                    graph = annotate_graph(
                        annotate_layout(
                            temporal.update_frame(
                                generate_scene_graph(record, self.config)),
                            record,
                        ),
                        record, self.config,
                    )
                    frames_file.write(json.dumps(record) + "\n")
                    graph_file.write(json.dumps(graph) + "\n")
                    self._fold(graph, index)

                    ok, buf = cv2.imencode(".jpg", frame,
                                           [int(cv2.IMWRITE_JPEG_QUALITY), 72])
                    dt = time.perf_counter() - tick
                    self.fps = 0.85 * self.fps + 0.15 / max(dt, 1e-6) if self.fps else 1 / max(dt, 1e-6)
                    index += 1
                    self.frames = index

                    with self._lock:
                        if ok:
                            self.frame_jpeg = buf.tobytes()
                        self.payload = self._snapshot(record, graph, names, elapsed)
        finally:
            capture.release()
            frames_file.close()
            graph_file.close()
            self.running = False

    def _fold(self, graph, index):
        """Accumulate one frame's graph into the running session totals."""
        present = []
        for node in graph.get("nodes", []):
            pid = node.get("person_id")
            if pid is None or pid <= 0:
                continue
            feat = node.get("features") or {}
            present.append(pid)
            self.last_seen[pid] = index
            if feat.get("gaze_label"):
                self.gaze[pid][feat["gaze_label"]] += 1
            if feat.get("expression"):
                self.expr[pid][feat["expression"]] += 1
            if feat.get("action"):
                self.actions[pid][feat["action"]] += 1
            self.recent[pid].append(1 if feat.get("gaze_label") == "teacher" else 0)
        for edge in graph.get("edges", []):
            if edge["type"] != "shared_action":
                continue
            of = {n["id"]: n.get("person_id") for n in graph.get("nodes", [])}
            a, b = of.get(edge["source"]), of.get(edge["target"])
            if a and b and a > 0 and b > 0:
                self.pairs[tuple(sorted((a, b)))] += 1

    def _layout_now(self, record):
        """The room's measured layout for this frame, for the live view.

        The layout verdict is the project's distinguishing measurement and it
        was computed on every frame and never sent anywhere, so the one thing
        the system does that others do not was invisible while it ran.
        """
        from backend.scene_layout import detect as detect_layout

        layout = detect_layout(record.get("persons", []))
        return {
            "kind": layout.kind,
            "ratio": round(layout.ratio, 2) if layout.focus else None,
            "focus": [round(layout.focus[0]), round(layout.focus[1])] if layout.focus else None,
            "centre": [round(layout.centre[0]), round(layout.centre[1])] if layout.centre else None,
            "radius": round(layout.radius) if layout.centre else None,
            "n": layout.n,
        }

    def _snapshot(self, record, graph, names, elapsed) -> dict:
        """Build the JSON payload the browser renders each frame."""
        feats = {n.get("person_id"): (n.get("features") or {})
                 for n in graph.get("nodes", [])}
        boxes = []
        small_unknown = 0
        for person in record["persons"]:
            pid = person.get("person_id")
            feat = feats.get(pid) or {}
            face = person.get("face") or {}
            box = face.get("bbox")
            face_px = int(min(box[2], box[3])) if box else None
            if (pid is None or pid <= 0) and face_px and face_px < self.min_face_px:
                small_unknown += 1
            posture = person.get("posture") or {}
            boxes.append({
                "bbox": person["bbox"],
                "person_id": pid,
                "name": names.get(pid) if pid and pid > 0 else None,
                "gaze": feat.get("gaze_label"),
                "action": feat.get("action"),
                "face_px": face_px,
                # The shoulder direction each person contributes to the focus
                # calculation, so the viewer can see the geometry being used.
                "facing": posture.get("facing_direction"),
                "oriented": feat.get("oriented"),
            })

        students = []
        for pid, name in sorted(names.items()):
            counts = self.gaze[pid]
            graded = sum(counts.values())
            students.append({
                "id": pid,
                "name": name,
                "present": self.last_seen.get(pid, -99) >= record["frame_id"] - 4,
                "attention": (counts["teacher"] / graded) if graded else None,
                "gaze": dict(counts),
                "expression": dict(self.expr[pid]),
                "actions": dict(self.actions[pid]),
                "action": (feats.get(pid) or {}).get("action"),
                "evidence": (feats.get(pid) or {}).get("action_evidence"),
                "recent": list(self.recent[pid]),
            })

        # A face too small to identify is the single most common reason a demo
        # shows "unknown" with a perfectly clear picture on screen. Measured:
        # at 28px the similarity to an enrolled reference was 0.149 against a
        # 0.35 threshold, and the detector score 0.31 against a 0.50 floor.
        # Saying so beats letting the operator guess.
        faceless = sum(1 for b in boxes if b["face_px"] is None)
        hint = None
        if small_unknown:
            hint = ("Face too small to identify — move closer to the camera "
                    f"(need about {self.min_face_px}px, seeing "
                    f"{min(b['face_px'] for b in boxes if b['face_px'])}px).")
        elif faceless and not any(b["name"] for b in boxes):
            hint = ("Somebody is in frame but no face is visible — turn towards "
                    "the camera. Identity needs a face, not a body.")
        elif record["persons"] and not any(b["name"] for b in boxes):
            hint = "Someone is in frame but not recognised — register them first."

        return {
            "hint": hint,
            "layout": self._layout_now(record),
            "frame": record["frame_id"],
            "seconds": round(elapsed, 1),
            "fps": round(self.fps, 1),
            "in_frame": len(record["persons"]),
            "objects": [o["cls"] for o in record.get("objects", [])],
            "boxes": boxes,
            "students": students,
            "pairs": [{"a": a, "b": b, "frames": n}
                      for (a, b), n in self.pairs.most_common(12)],
        }


def create_app(args):
    """Build the FastAPI application.

    Args:
        args: Parsed CLI arguments.

    Returns:
        A configured :class:`fastapi.FastAPI`.
    """
    import cv2
    import numpy as np

    from backend.config import CONFIG
    from backend.enrollment import EnrolledGallery
    from backend.face import FaceAnalyzer

    app = FastAPI(title="ClassGraph")
    state = {"session": None, "config": None}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    def gallery():
        return EnrolledGallery.load(args.gallery, CONFIG.identity)

    def live_config(shape):
        """Retune the pipeline for live webcam frames of this size."""
        if state["config"] is not None:
            return state["config"]
        h, w = shape[:2]
        cfg = replace(
            CONFIG,
            detection=replace(CONFIG.detection,
                              imgsz=args.imgsz or min(CONFIG.detection.imgsz, max(w, h))),
            posture=replace(CONFIG.posture, static_image_mode=False, model_complexity=0),
            face=replace(CONFIG.face, static_image_mode=False, refine_landmarks=False),
        )
        if args.yaw_reference is not None:
            cfg = replace(cfg, headpose=replace(
                cfg.headpose, yaw_reference_deg=args.yaw_reference))
        state["config"] = cfg
        return cfg

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/api/students")
    async def students():
        g = gallery()
        return {"students": [
            {"id": p.person_id, "name": p.name, "shots": p.shots} for p in g.people]}

    @app.delete("/api/students/{name}")
    async def forget(name: str):
        g = gallery()
        if not g.forget(name):
            raise HTTPException(404, f"No student named {name!r}")
        g.save(args.gallery)
        return {"ok": True}

    @app.post("/api/enroll")
    async def enroll(payload: dict):
        """Register a student from browser-captured frames.

        Args:
            payload: ``{"name": str, "shots": [dataURL, ...]}``.

        Returns:
            The new student, or a 400 explaining which shots were unusable.
        """
        name = (payload.get("name") or "").strip()
        shots = payload.get("shots") or []
        if not name:
            raise HTTPException(400, "A name is required.")

        cfg = state["config"] or CONFIG
        embeddings, rejected = [], []
        with FaceAnalyzer(cfg.face) as analyzer:
            for i, data_url in enumerate(shots):
                raw = base64.b64decode(data_url.split(",", 1)[-1])
                img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if img is None:
                    rejected.append(f"shot {i + 1}: unreadable")
                    continue
                faces = analyzer.detect_faces(img)
                good = [f for f in faces
                        if f.embedding is not None
                        and f.score >= CONFIG.identity.min_face_score_for_identity
                        and min(f.bbox[2], f.bbox[3]) >= args.min_face_px]
                if len(good) == 1:
                    embeddings.append(good[0].embedding)
                else:
                    rejected.append(
                        f"shot {i + 1}: {'no usable face' if not good else f'{len(good)} faces'}")

        if len(embeddings) < 3:
            raise HTTPException(400, json.dumps({
                "message": f"Only {len(embeddings)} usable shot(s); at least 3 are "
                           f"needed for a reliable reference.",
                "detail": rejected[:6],
            }))
        g = gallery()
        person = g.register(name, embeddings)
        g.save(args.gallery)
        return {"id": person.person_id, "name": person.name, "shots": person.shots}

    @app.post("/api/session/start")
    async def start():
        if state["session"] and state["session"].running:
            return {"ok": True, "already": True}
        g = gallery()
        if not len(g):
            raise HTTPException(400, "Register at least one student first.")
        probe = cv2.VideoCapture(args.camera)
        ok, frame = probe.read()
        probe.release()
        if not ok:
            raise HTTPException(500, f"Camera {args.camera} returned no frame.")
        session = Session(live_config(frame.shape), g, out, args.camera,
                          args.stride, args.min_face_px)
        session.start()
        state["session"] = session
        return {"ok": True}

    @app.post("/api/session/stop")
    async def stop():
        session = state["session"]
        if not session:
            raise HTTPException(400, "No session is running.")
        session.stop()
        report = _finish(out, session.gallery)
        return {"ok": True, "frames": session.frames,
                "report": f"/report?t={int(time.time())}" if report else None}

    @app.get("/report", response_class=HTMLResponse)
    async def report():
        path = out / "report.html"
        if not path.exists():
            return HTMLResponse("<p>No report yet — run a session first.</p>", 404)
        return path.read_text(encoding="utf-8")

    @app.get("/api/status")
    async def status():
        session = state["session"]
        return JSONResponse({
            "running": bool(session and session.running),
            "frames": session.frames if session else 0,
        })

    @app.websocket("/ws")
    async def ws(socket: WebSocket):
        """Push the live frame and analytics to the page."""
        await socket.accept()
        try:
            while True:
                session = state["session"]
                if session and session.running:
                    with session._lock:
                        jpeg, payload = session.frame_jpeg, session.payload
                    if jpeg and payload:
                        payload = dict(payload)
                        payload["image"] = base64.b64encode(jpeg).decode("ascii")
                        await socket.send_text(json.dumps(payload))
                else:
                    await socket.send_text(json.dumps({"idle": True}))
                await asyncio.sleep(0.06)
        except (WebSocketDisconnect, RuntimeError):
            return

    if STATIC.exists():
        app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


def _finish(out: Path, gallery) -> Path | None:
    """Build profiles and the report at the end of a session.

    Args:
        out: Session output directory.
        gallery: The registered students.

    Returns:
        The report path, or ``None`` if the session produced nothing.
    """
    from backend.student_profile import build_profiles
    from tools.report import build

    graph_path = out / "live_graph.jsonl"
    if not graph_path.exists() or not graph_path.stat().st_size:
        return None
    profiles = build_profiles(graph_path, None)
    (out / "live_profiles.json").write_text(
        json.dumps(list(profiles.values()), indent=2), encoding="utf-8")
    return build(out, gallery)


def main() -> int:
    """Parse arguments and serve the dashboard."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--gallery", default="outputs/enrollment/gallery.json")
    parser.add_argument("--out", default="outputs/live")
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--min-face-px", type=int, default=60)
    parser.add_argument("--yaw-reference", type=float, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    import uvicorn

    print(f"ClassGraph dashboard -> http://{args.host}:{args.port}")
    # websockets 14 removed the legacy API uvicorn's default "websockets"
    # implementation is built on, and the mismatch shows up as every WebSocket
    # handshake being refused with a bare 403 before the route is even reached.
    # The sans-io implementation is the maintained path and works with 14+.
    uvicorn.run(create_app(args), host=args.host, port=args.port,
                log_level="warning", ws="websockets-sansio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
