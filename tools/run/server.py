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
    python -m tools.run.server
    # then open http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import logging
import os
import re
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
from fastapi import FastAPI, File, Form, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

logger = logging.getLogger(__name__)

#: Video containers OpenCV reads reliably. Anything else is
#: rejected with its name rather than failing inside the decoder.
VIDEO_SUFFIXES = frozenset({".mp4", ".avi", ".mov", ".mkv", ".webm"})

#: Upload ceiling. Above this the browser round trip is the wrong
#: tool -- `classgraph video` streams from disk with no copy.
MAX_UPLOAD_BYTES = 512 * 1024 * 1024

#: Roughly the smallest face the recogniser can match against the gallery.
#: Measured, not guessed: on the overhead classroom footage 86 faces came out
#: at a median of 56px with only 9 above this line, and the enrolment scores
#: for those crops ran 0.36-0.45 against a 0.5 acceptance gate. Below this,
#: students are numbered rather than named, and the page should say so while
#: the run is still going rather than leave it to be inferred from the result.
RECOGNISABLE_FACE_PX = 60

#: Object confidence below which a detection is shown faded and not counted
#: as something the room contained. The detector is run at 0.35 (0.25 for
#: books, 0.15 beside a person) deliberately, because raising it loses real
#: laptops -- so weak detections exist by design and the display, not the
#: detector, is where they have to be marked as weak.
REPORTABLE_OBJECT_CONF = 0.50

#: How many analysed recordings to keep alongside their results.
#:
#: Every upload used to be kept forever. Two days of testing left 20 copies
#: of 14 videos in outputs/live/uploads -- 219 MB, and far more to the point,
#: 219 MB of classroom footage of children with no retention policy at all,
#: in a project that has a retention sweep for the face gallery and a
#: unified erasure path for everything else. The analysis is what the upload
#: was for; keeping the input after it is a copy nobody asked for.
#:
#: Not zero. `tools/run/export_faces_to_enrol.py` needs the source video to pull
#: face crops for enrolment, and that is the obvious next step after watching
#: a run, so the last few stay.
KEEP_RECENT_UPLOADS = 3


def prune_uploads(uploads: Path, keep: int = KEEP_RECENT_UPLOADS) -> int:
    """Delete all but the most recent ``keep`` analysed recordings.

    Args:
        uploads: The uploads directory.
        keep: How many of the newest entries to leave in place.

    Returns:
        How many entries were removed.

    Entries are named from the upload time and each run stages its video into
    a folder of the same name, so both a bare file and its staged folder are
    handled. A failure to delete is logged and skipped rather than raised:
    a video still open in a reader must not stop the next analysis starting.
    """
    import shutil

    if not uploads.is_dir():
        return 0
    entries = sorted(uploads.iterdir(),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    removed = 0
    for entry in entries[keep:]:
        try:
            shutil.rmtree(entry) if entry.is_dir() else entry.unlink()
        except OSError as exc:
            logger.warning("could not remove old upload %s: %s", entry, exc)
            continue
        removed += 1
    if removed:
        logger.info("removed %d recording(s) older than the last %d",
                    removed, keep)
    return removed

STATIC = Path(__file__).resolve().parent / "static"
UNKNOWN = -1


class Session:
    """One live capture, owning the camera and every model."""

    def __init__(self, config, gallery, out: Path, camera, stride: int,
                 min_face_px: int = 60, sample_rate: int = 1):
        """Prepare a session without opening its source yet.

        Args:
            config: Live-tuned pipeline config.
            gallery: Registered students.
            out: Directory for the per-frame records, graph and profiles.
            camera: A camera index, or the path to a video file. The capture
                loop is identical either way -- a recording is a camera that
                runs out -- so the only differences are that a file needs no
                warm-up frames and that reaching its end is success rather
                than the failure it would be on a camera.
            stride: Run expression / head pose / posture every Nth frame.
            min_face_px: Face size below which identification is not expected
                to work, used only to warn the operator.
            sample_rate: Process every Nth frame. 1 for a camera, which is
                already slow enough to keep up with; higher for a file, where
                30 fps of near-identical frames is work with no new
                information in it.
        """
        self.config = config
        self.gallery = gallery
        self.out = out
        self.camera = camera
        #: True when reading a file rather than a camera.
        self.is_file = not isinstance(camera, int)
        self.sample_rate = max(1, sample_rate)
        #: Total frames in the source, when it is a file and says so. Lets the
        #: page show progress instead of an unbounded count.
        self.total_frames = 0
        self.stride = max(1, stride)
        self.min_face_px = min_face_px
        self.running = False
        # Why the last session ended, when it ended on its own.
        self.stop_reason: str | None = None
        self.frame_jpeg: bytes | None = None
        self.payload: dict = {}
        self.started = 0.0
        #: Frames read from the source. On a sampled file this is the
        #: position in the recording, which is what progress is measured
        #: against -- not how many were analysed.
        self.frames = 0
        #: Frames actually put through the models. Lower than `frames`
        #: whenever sample_rate skips any, and the number that matches
        #: the rows in live_graph.jsonl.
        self.analysed = 0
        self.fps = 0.0
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        #: Bumped whenever a new frame and payload are published, so
        #: the socket can send on change instead of on a timer.
        self.revision = 0
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
        from backend.attention_score import (
            AttentionScorer,
            annotate_attention,
        )
        from backend.face import FaceAnalyzer
        from backend.mamba_score import MambaScorer
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
            self.stop_reason = (
                f"Could not open {self.camera}."
                if self.is_file else
                f"Could not open camera {self.camera}. Another program is "
                "most likely using it.")
            logger.error("%s", self.stop_reason)
            self.running = False
            return
        if self.is_file:
            self.total_frames = int(
                capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        else:
            # Webcams return dark or malformed frames while exposure settles.
            # A file has no such warm-up, and discarding its first 20 frames
            # would silently drop the opening second of the recording.
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
        # One scorer per session: it carries each student's rolling
        # window across frames, so it cannot be rebuilt per frame.
        scorer = AttentionScorer()
        if not scorer.available:
            logger.warning(
                "no trained attention model found; scores will be "
                "absent and the rule verdict is the only answer. "
                "Train one with tools/train/train_window_model.py.")
        # The sequence model, scored on the same windows and shown beside the
        # aggregate score. Optional: a missing checkpoint leaves one number
        # on screen instead of two and takes nothing down.
        mamba = MambaScorer()
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
                        if self.is_file:
                            # The end of a recording is usually the job
                            # finishing. It is not when the decoder gave up
                            # early: a damaged file stops returning frames
                            # exactly as a finished one does, and reporting
                            # both as "finished" would let somebody believe
                            # they had analysed a whole lecture when they had
                            # analysed its first few seconds.
                            short = (self.total_frames
                                     and index < self.total_frames * 0.9)
                            if short:
                                self.stop_reason = (
                                    f"The video ended after {index} of about "
                                    f"{self.total_frames} frames, so it is "
                                    "most likely damaged. The results below "
                                    "cover only the part that could be read.")
                                logger.warning("%s", self.stop_reason)
                            else:
                                self.stop_reason = (
                                    f"Finished the video: {index} frames "
                                    "read.")
                                logger.info("%s", self.stop_reason)
                        else:
                            # Almost always another process holding the camera
                            # -- a server left running from an earlier session.
                            # On Windows those survive Ctrl+C in a way that is
                            # easy to miss, and the symptom is a session that
                            # stops on its own with nothing explaining it.
                            self.stop_reason = (
                                "The camera stopped returning frames after "
                                f"{index} frames. Another program is most "
                                "likely using it -- check for a server left "
                                "running from an earlier session.")
                            logger.warning("%s", self.stop_reason)
                        break

                    # Sampling a file: a 30 fps recording holds thirty nearly
                    # identical frames per second, and processing all of them
                    # costs thirty times as long for no extra information.
                    if self.sample_rate > 1 and index % self.sample_rate:
                        index += 1
                        continue
                    self.analysed += 1

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
                    # Layout first: actions read `oriented` from it. Attention
                    # last: its window features are built from everything the
                    # earlier stages produce.
                    graph = annotate_attention(
                        annotate_graph(
                            annotate_layout(
                                temporal.update_frame(
                                    generate_scene_graph(record, self.config)),
                                record,
                            ),
                            record, self.config,
                        ),
                        scorer, mamba,
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
                        self.revision += 1
                        if ok:
                            self.frame_jpeg = buf.tobytes()
                        self.payload = self._snapshot(record, graph, names, elapsed)
        except Exception:
            # Without this the thread dies silently and the only symptom is a
            # session that stopped by itself. The traceback goes to the log and
            # the reason to the UI, so a failure says what it was.
            self.stop_reason = (
                "The capture loop failed. See the server log for the "
                "traceback.")
            logger.exception("capture loop failed after %d frames", index)
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
                # Posture, reported in its own right. It used to reach the
                # page only when it won the action contest, which it lost to
                # every object rule and both gaze rules -- 2 frames in 424
                # on one recording, against 334 in which it had been measured.
                "lean": feat.get("lean_state"),
                "lean_value": feat.get("vertical_lean"),
            })

        # Where each student actually is in the frame, and which way they
        # face. The graph drew students on a four-column grid in whatever
        # order the roster happened to be in, which is a picture of the array
        # rather than of the room -- and these positions had been measured
        # all along.
        where = {}
        for person in record["persons"]:
            pid = person.get("person_id")
            box = person.get("bbox")
            if not pid or pid <= 0 or not box:
                continue
            where[pid] = {
                "cx": round(box[0] + box[2] / 2.0),
                "cy": round(box[1] + box[3] / 2.0),
                "facing": (person.get("posture") or {}).get(
                    "facing_direction"),
            }

        students = []
        for pid, name in sorted(names.items()):
            counts = self.gaze[pid]
            graded = sum(counts.values())
            spot = where.get(pid) or {}
            students.append({
                "id": pid,
                "name": name,
                "present": self.last_seen.get(pid, -99) >= record["frame_id"] - 4,
                # The model's score, not the old gaze ratio. counts["teacher"]
                # over graded measured how often a student faced the front,
                # which is a different claim from whether they were engaged --
                # and in group work it is close to the opposite one.
                "attention_score": (feats.get(pid) or {}).get(
                    "attention_score"),
                # The Mamba sequence model's score for the same student,
                # shown beside the above rather than replacing it. Out of
                # fold on 179 labelled windows the two came out 0.609 and
                # 0.592 -- too close to call a winner on that many labels, so
                # both are reported and the viewer can see them agree or not.
                "attention_score_mamba": (feats.get(pid) or {}).get(
                    "attention_score_mamba"),
                "lean": (feats.get(pid) or {}).get("lean_state"),
                "attention_label": (feats.get(pid) or {}).get(
                    "attention_label"),
                "attention_reason": (feats.get(pid) or {}).get(
                    "attention_reason"),
                "attention_source": (feats.get(pid) or {}).get(
                    "attention_source"),
                # Kept so the two can be shown side by side rather than one
                # quietly standing in for the other.
                "facing_front_pct": (counts["teacher"] / graded)
                if graded else None,
                "gaze": dict(counts),
                "expression": dict(self.expr[pid]),
                "actions": dict(self.actions[pid]),
                "action": (feats.get(pid) or {}).get("action"),
                "evidence": (feats.get(pid) or {}).get("action_evidence"),
                # The object THIS student was judged to be handling, and how
                # sure the detector was of it. The graph used to draw an arc
                # from every student to every object in the frame, which drew
                # a book for people who never touched one. assign_objects
                # already decides ownership exclusively; the answer simply
                # was not being sent.
                "object": (feats.get(pid) or {}).get("object"),
                "object_conf": (feats.get(pid) or {}).get("object_confidence"),
                "oriented": (feats.get(pid) or {}).get("oriented"),
                "cx": spot.get("cx"),
                "cy": spot.get("cy"),
                "facing": spot.get("facing"),
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
            # Class and confidence, not class alone. A bottle at 0.28 and a
            # laptop at 0.91 were two identical chips on screen, so a weak
            # detection read as a fact about the room. The page shows the
            # number and fades the chip below REPORTABLE_OBJECT_CONF.
            "objects": [{"cls": o["cls"],
                         "conf": round(float(o.get("confidence") or 0.0), 2)}
                        for o in record.get("objects", [])],
            "boxes": boxes,
            "students": students,
            "pairs": [{"a": a, "b": b, "frames": n}
                      for (a, b), n in self.pairs.most_common(12)],
        }


class VideoJob:
    """An uploaded recording, processed by the batch pipeline in a thread.

    Presents the same surface as :class:`Session` -- ``running``, ``frames``,
    ``stop_reason``, ``stop()`` -- so the page and the stop endpoint do not
    need to know which produced a result. What differs is underneath: this
    runs tools.run.batch_session, which resolves identity in two passes over the
    whole recording rather than deciding on first sight.

    There is no live frame preview while it runs, because the batch pipeline
    does not emit one. Progress comes from counting the graph rows it writes,
    which is the honest measure anyway: those rows are the analysis.
    """

    def __init__(self, video: Path, out: Path, sample_rate: int,
                 gallery_path, db_path):
        self.video = Path(video)
        self.out = Path(out)
        self.sample_rate = max(1, sample_rate)
        self.gallery_path = gallery_path
        self.db_path = db_path
        self.running = False
        self.stop_reason: str | None = None
        self.frames = 0
        self.analysed = 0
        self.total_frames = 0
        self.is_file = True
        self.started = time.time()
        #: Set when the first preview frame is downscaled, so the
        #: resolved redraw lands on the same pixels.
        self._preview_scale = 1.0
        # The page reads these; a batch run has no live frame to show.
        self.frame_jpeg = None
        self.payload: dict = {}
        self.gallery = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        #: Bumped whenever a new frame and payload are published, so
        #: the socket can send on change instead of on a timer.
        self.revision = 0

    def start(self):
        import cv2

        capture = cv2.VideoCapture(str(self.video))
        self.total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        capture.release()

        self.running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        import argparse as _argparse

        from tools.run import batch_session

        # batch_session works on a folder of clips, treating them as one
        # continuous session. A single upload is a folder of one.
        folder = self.video.parent / self.video.stem
        folder.mkdir(parents=True, exist_ok=True)
        staged = folder / self.video.name
        if not staged.exists():
            self.video.replace(staged)

        # A folder used for a live session and then a batch run holds one
        # per-frame file from each, with different person ids in them. Every
        # tool that reads the folder then has to guess which is current, and
        # one of them guessed wrong. Removing the live file makes the folder
        # describe a single run.
        stale = self.out / "live_frames.jsonl"
        if stale.exists():
            stale.unlink()

        job_args = _argparse.Namespace(
            dir=str(folder), pattern=self.video.name,
            out=str(self.out), sample_rate=self.sample_rate, limit=None,
            behaviour_imgsz=None, min_face_px_to_found=35,
            db=self.db_path, no_db=False, gallery=self.gallery_path,
            title=self.video.name, skip_continuity=True, verbose=False)

        from backend.enrollment import EnrolledGallery

        try:
            batch_session.run(job_args, on_frame=self._preview)
            # Empty for the same reason batch_session uses an empty one:
            # this run's person ids are local to it and mean nothing in the
            # enrolled gallery.
            self.gallery = EnrolledGallery()
            self._publish_resolved()
            self.stop_reason = self._completion_note()
        except Exception as exc:  # noqa: BLE001 - surfaced, never swallowed
            self.stop_reason = f"Processing failed: {exc}"
            logger.exception("video job failed for %s", self.video)
        finally:
            self.running = False

    def _preview(self, frame, record, kept: int) -> None:
        """Publish the frame being analysed, for the page to display.

        Called from pass 1, where detections exist but identity has not been
        resolved and the action rules have not run. So boxes carry no name and
        no action, and the page is told as much rather than being handed empty
        strings that would render as though nothing was happening.

        Downscaled before encoding. A 1080p JPEG per frame is several hundred
        kilobytes through a WebSocket that is only driving a preview.
        """
        import cv2

        height, width = frame.shape[:2]
        if width > 960:
            frame = cv2.resize(frame, (960, int(height * 960 / width)))
        ok, buffer = cv2.imencode(".jpg", frame,
                                  [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        if not ok:
            return

        self._preview_scale = frame.shape[1] / width
        scale = self._preview_scale
        boxes = []
        for person in record.get("persons", []):
            box = person.get("bbox")
            if not box:
                continue
            face = (person.get("face") or {}).get("bbox")
            boxes.append({
                "bbox": [int(v * scale) for v in box[:4]],
                "person_id": None,          # resolved in pass 2
                "name": None,
                "action": None,
                "gaze": None,
                "face_px": (int(min(face[2], face[3]) * scale)
                            if face else None),
            })

        elapsed = max(time.time() - self.started, 1e-6)
        with self._lock:
            self.revision += 1
            self.frame_jpeg = buffer.tobytes()
            # Every field the page reads, including the ones a batch run has
            # no answer for. A partial payload is worse than a dull one: the
            # page reads d.objects.length, which throws on a missing key, and
            # the handler then aborts before repainting -- leaving the last
            # live session's boxes drawn over the video, labelled with its
            # names. Absent data has to be present and empty.
            self.payload = {
                "boxes": boxes,
                "students": [],
                "pairs": [],
                "objects": [],
                "layout": None,
                "in_frame": len(boxes),
                "frame": kept,
                "frames": kept,
                "fps": round(kept / elapsed, 1),
                "seconds": int(elapsed),
                "phase": "detecting",
                "hint": self._detecting_hint(boxes),
            }

    def _detecting_hint(self, boxes) -> str:
        """What pass 1 can honestly say, including what it will never give.

        "Resolved in a second pass" explains why boxes say unidentified now,
        and implies names are coming. On footage where the faces are too
        small to recognise they are not, and the question gets asked again at
        the end. Both facts belong in the same sentence.
        """
        faces = [b["face_px"] for b in boxes if b.get("face_px")]
        base = ("Detecting people. Identity and actions are resolved in a "
                "second pass, once the whole recording has been seen.")
        if not faces:
            return base
        median = sorted(faces)[len(faces) // 2]
        if median >= RECOGNISABLE_FACE_PX:
            return base + (f" Faces here are about {median}px, large enough "
                           "to match against registered students.")
        return base + (
            f" Faces here are about {median}px and matching a registered "
            f"student needs around {RECOGNISABLE_FACE_PX}px, so students "
            "will be numbered rather than named.")

    def _publish_resolved(self) -> None:
        """Redraw the last frame with what pass 2 concluded.

        Every box the page showed during a run said "unidentified", because
        pass 1 is before identity exists. Pass 2 then resolves five students
        and the picture is never updated, so the analysis appears to have
        found nobody even though the results panel lists them -- the two
        halves of the screen disagree, and the one the eye goes to is wrong.

        The last graph row describes the same frame as the last preview, so
        the existing JPEG can be relabelled rather than re-encoded.
        """
        graph = self.out / "live_graph.jsonl"
        if not graph.exists() or self.frame_jpeg is None:
            return
        try:
            last = None
            for line in graph.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    last = line
            if last is None:
                return
            row = json.loads(last)
        except (OSError, json.JSONDecodeError):
            return

        boxes = []
        for node in row.get("nodes", []):
            features = node.get("features") or {}
            box = features.get("bbox")
            pid = node.get("person_id")
            if not box or pid is None or int(pid) <= 0:
                continue
            boxes.append({
                "bbox": [int(v * self._preview_scale) for v in box[:4]],
                "person_id": int(pid),
                # Still no name: a batch run numbers people within itself, so
                # the gallery cannot say who they are.
                "name": None,
                "action": features.get("action"),
                "gaze": features.get("gaze_label"),
                "face_px": None,
                "lean": features.get("lean_state"),
                "lean_value": features.get("vertical_lean"),
            })
        if not boxes:
            return

        students, objects = self._resolved_students(row)
        with self._lock:
            self.revision += 1
            self.payload = {
                **self.payload,
                "boxes": boxes,
                "students": students,
                "objects": objects,
                "in_frame": len(boxes),
                "phase": "resolved",
                "hint": (f"Resolved {len(boxes)} students. Names need faces "
                         "of about 60px; these are smaller, so they are "
                         "numbered instead."),
            }

    def _resolved_students(self, row) -> tuple[list, list]:
        """The roster the scene graph draws, from a finished batch run.

        A video run used to send ``students: []`` on every payload including
        the final one, so the graph card -- the seating map, the both-models
        comparison, the object links -- stayed empty for the whole analysis
        and stayed empty afterwards. The results panel listed five students
        at the same moment the graph said nobody was in frame. All of it had
        been computed; none of it was being sent.

        Scores come from the profiles file, which is the session mean per
        student, and position from the final frame.
        """
        profiles_path = self.out / "live_profiles.json"
        by_id: dict[int, dict] = {}
        if profiles_path.exists():
            try:
                data = json.loads(profiles_path.read_text(encoding="utf-8"))
                entries = data if isinstance(data, list) else list(data.values())
                for entry in entries:
                    pid = entry.get("person_id")
                    if pid is not None and int(pid) > 0:
                        by_id[int(pid)] = entry
            except (OSError, json.JSONDecodeError):
                pass

        students, objects = [], []
        for node in row.get("nodes", []):
            pid = node.get("person_id")
            if pid is None or int(pid) <= 0:
                continue
            pid = int(pid)
            features = node.get("features") or {}
            profile = by_id.get(pid, {})
            box = features.get("bbox") or [0, 0, 0, 0]
            obj = features.get("object")
            conf = features.get("object_confidence")
            if obj:
                objects.append({"cls": obj,
                                "conf": round(float(conf or 0.0), 2)})
            students.append({
                "id": pid,
                # A batch run numbers people within itself; the gallery
                # cannot say who they are, so the page shows "Student 3".
                "name": None,
                "present": True,
                "attention_score": profile.get("attention_score"),
                "attention_score_mamba": profile.get(
                    "attention_score_mamba"),
                "attention_label": None,
                "attention_reason": (
                    f"session mean over "
                    f"{profile.get('attention_windows', 0)} windows"),
                "attention_source": "model",
                "action": features.get("action"),
                "evidence": features.get("action_evidence"),
                "lean": features.get("lean_state"),
                "object": obj,
                "object_conf": conf,
                "oriented": features.get("oriented"),
                "cx": round(box[0] + box[2] / 2.0),
                "cy": round(box[1] + box[3] / 2.0),
                "facing": None,
                "facing_front_pct": None,
                # Empty rather than absent: the page reads these and a
                # missing key throws before it repaints.
                "gaze": {}, "expression": {}, "actions": {}, "recent": [],
            })
        return students, objects

    def _completion_note(self) -> str:
        """What to tell the operator, including when the file was damaged."""
        rows = self._graph_rows()
        read = rows * self.sample_rate
        if self.total_frames and read < self.total_frames * 0.9:
            # Reached only after the decoder has already been given 30 tries
            # to step over each bad frame (MAX_CONSECUTIVE_BAD_FRAMES in
            # tools/run/batch_session.py). Before that recovery existed, one
            # rejected h264 packet 16 frames into a 288-frame recording ended
            # the read and produced this message for a file that was fine.
            return (f"Only {read} of {self.total_frames} frames could be "
                    "decoded, even after skipping past the frames that "
                    "failed. The results below cover that part only.")
        return (f"Finished: {rows} frames analysed"
                + (f" from {self.total_frames} in the recording."
                   if self.total_frames else "."))

    def _graph_rows(self) -> int:
        path = self.out / "live_graph.jsonl"
        if not path.exists():
            return 0
        try:
            with path.open("rb") as handle:
                return sum(1 for _ in handle)
        except OSError:
            return 0

    def poll(self):
        """Refresh the counters the status endpoint reads."""
        self.analysed = self._graph_rows()
        self.frames = self.analysed * self.sample_rate

    def stop(self):
        """Wait for the run to finish.

        A batch run cannot be interrupted part way without leaving a session
        folder that looks complete and is not, so this waits rather than
        pretending to cancel.
        """
        if self._thread:
            self._thread.join(timeout=600)
            self._thread = None
        self.running = False


def _session_folders(root: Path) -> list[Path]:
    """Output folders holding a session, found rather than configured.

    A session writes to whichever folder the run chose, so a hand-maintained
    list would go stale -- and a stale list here means records that survive a
    deletion request.
    """
    base = Path(root).parent if Path(root).name else Path(root)
    if not base.is_dir():
        return []
    return sorted(p for p in base.iterdir()
                  if p.is_dir() and (p / "live_profiles.json").exists())


def create_app(args):
    """Build the FastAPI application.

    Args:
        args: Parsed CLI arguments.

    Returns:
        A configured :class:`fastapi.FastAPI`.
    """
    import cv2
    import numpy as np

    from backend import webauth
    from backend.config import CONFIG
    from backend.enrollment import EnrolledGallery
    from backend.face import FaceAnalyzer

    app = FastAPI(title="ClassGraph")

    # Access control before anything else is registered. The dashboard can
    # start a camera pointed at minors and stream it, so an unauthenticated
    # route is not a missing feature but a hole.
    token, generated = webauth.resolve_token()
    webauth.install(app, token)
    app.state.classgraph_token = token

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
        # Served without the token. An earlier version injected it into a
        # meta tag, reasoning that a cross-origin page cannot read this
        # response -- true, and irrelevant: `curl http://host:8000/ | grep
        # classgraph-token` hands any device on the network full API access,
        # which is the threat the token exists to stop. The operator pastes
        # the token printed at startup; the page keeps it in sessionStorage.
        page = (STATIC / "index.html").read_text(encoding="utf-8")

        # The cache-busting query was a hand-maintained "?v=2", so editing
        # app.js without remembering to bump it served every browser the old
        # file. That is indistinguishable from the fix not working, and it
        # cost a debugging session. Stamped from the file's own modification
        # time instead, which cannot drift from its contents.
        for asset in ("app.js", "style.css"):
            path = STATIC / asset
            if path.exists():
                stamp = int(path.stat().st_mtime)
                page = re.sub(rf"/static/{re.escape(asset)}(\?v=[^\"']*)?",
                              f"/static/{asset}?v={stamp}", page)
        return page

    @app.get("/api/students")
    async def students():
        g = gallery()
        return {"students": [
            {"id": p.person_id, "name": p.name, "shots": p.shots} for p in g.people]}

    @app.delete("/api/students/{name}")
    async def forget(name: str):
        """Erase a student from every store, not only from the gallery.

        This used to drop the face record and leave the session database and
        every output folder untouched -- per-window scores, per-frame boxes
        and the name rendered into a report all survived, and the call
        returned ok. A deletion request is one request.
        """
        from backend.erasure import erase

        report = erase(name, gallery_path=Path(args.gallery),
                       db_path=Path(args.db),
                       output_dirs=_session_folders(out))
        if not report.anything_found:
            raise HTTPException(404, f"No student named {name!r}")
        return {
            "ok": True,
            "face_record_removed": report.gallery_removed,
            "database_sessions": report.db_sessions,
            "files_filtered": len(report.files_rewritten),
            "files_removed": len(report.files_removed),
            # Returned rather than hidden: an operator acting on a deletion
            # request needs to know what this could not reach.
            "unreachable": report.unreachable,
        }

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

    @app.post("/api/video/analyse")
    async def analyse_video(file: UploadFile = File(...),
                            sample_rate: int = Form(5)):
        """Analyse an uploaded recording through the live pipeline.

        The same Session that drives the camera, pointed at a file. That is
        deliberate: a second code path would drift from the first, and the
        value of showing a video in this page is that it is processed exactly
        as a live session would be.

        One job at a time, for the same reason a camera session is: the
        models are loaded per session and two at once would contend for the
        GPU and produce two sets of output in one folder.
        """
        if state["session"] and state["session"].running:
            raise HTTPException(409, "A session is already running.")

        suffix = Path(file.filename or "upload.mp4").suffix.lower()
        if suffix not in VIDEO_SUFFIXES:
            raise HTTPException(
                400, f"{suffix or 'that file'} is not a video. Accepted: "
                     + ", ".join(sorted(VIDEO_SUFFIXES)))

        uploads = out / "uploads"
        uploads.mkdir(parents=True, exist_ok=True)
        prune_uploads(uploads)
        # Named from the upload rather than reusing one path, so a second
        # analysis cannot overwrite a file the first is still reading.
        target = uploads / f"{int(time.time())}{suffix}"
        size = 0
        with target.open("wb") as handle:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    handle.close()
                    target.unlink(missing_ok=True)
                    raise HTTPException(
                        413, f"That video is larger than "
                             f"{MAX_UPLOAD_BYTES // (1 << 20)} MB. Process it "
                             "with `classgraph video` instead, which streams "
                             "from disk.")
                handle.write(chunk)

        # Run the batch pipeline, not the live loop. The two differ in how
        # identity is decided, and on this footage that is the whole result:
        # the live resolver must name a student the first time it sees them
        # and refuses when the face is too small, while the batch resolver
        # sees the entire recording before deciding and can use a good face
        # from any frame. Measured on the same clips, batch resolves 87% of
        # detections into 6 students and the live path resolves 0%.
        #
        # A file is finite, so there is no reason to accept the streaming
        # compromise that a camera forces.
        job = VideoJob(target, out, sample_rate=max(1, min(sample_rate, 30)),
                       gallery_path=args.gallery, db_path=args.db)
        state["session"] = job
        job.start()
        return {"ok": True, "filename": file.filename,
                "megabytes": round(size / (1 << 20), 1),
                "sample_rate": job.sample_rate}

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

    @app.get("/api/session/results")
    async def session_results():
        """Per-student results from the finished session.

        The roster panel lists registered students, so a recording of people
        nobody has enrolled had nowhere to appear: the analysis completed, the
        report was written, and the page showed the same three "not seen"
        names it started with. This returns what the run actually found.
        """
        profiles_path = out / "live_profiles.json"
        if not profiles_path.exists():
            return JSONResponse({"students": [], "ready": False})

        data = json.loads(profiles_path.read_text(encoding="utf-8"))
        entries = data if isinstance(data, list) else list(data.values())

        # Gallery names apply only when identity was resolved against the
        # gallery, which is the live path. A batch run numbers people within
        # itself, so looking its ids up in the gallery names a stranger after
        # whoever happens to hold that number -- and the numbers overlap.
        session = state["session"]
        from_gallery = session is not None and not getattr(
            session, "is_file", False)
        names = ({p.person_id: p.name for p in gallery().people}
                 if from_gallery else {})

        students = []
        for entry in entries:
            pid = entry.get("person_id")
            # Negative ids are tracks the resolver could not turn into a
            # person, and a rejected profile was explicitly refused. Neither
            # is a student to report on.
            if pid is None or int(pid) <= 0 or entry.get("rejected_reason"):
                continue
            if entry.get("role") not in (None, "student"):
                continue
            actions = (entry.get("actions") or {}).get("counts") or {}
            top = max(actions.items(), key=lambda kv: kv[1])[0] if actions else None
            students.append({
                "id": int(pid),
                "name": names.get(int(pid)),
                "attention": entry.get("attention_score"),
                "attention_windows": entry.get("attention_windows"),
                # The sequence model's answer for the same student, so the
                # finished run shows both and not just the one that ships.
                "attention_mamba": entry.get("attention_score_mamba"),
                "attention_windows_mamba": entry.get(
                    "attention_windows_mamba"),
                "on_task_pct": entry.get("on_task_pct"),
                "engagement_pct": entry.get("engagement_pct"),
                "frames": entry.get("frames_seen"),
                "top_action": top,
            })
        students.sort(key=lambda s: (s["attention"] is None, s["attention"] or 0))
        return JSONResponse({"students": students, "ready": bool(students)})

    @app.get("/api/status")
    async def status():
        session = state["session"]
        # A file knows how long it is, so progress is reportable; a camera
        # does not, and a progress bar that cannot end would be a lie.
        # A batch job counts its progress from the rows it has written, so it
        # has to be asked before being read.
        if session is not None and hasattr(session, "poll"):
            session.poll()
        total = getattr(session, "total_frames", 0) if session else 0
        # Measured on frames READ, not frames analysed. Multiplying the
        # analysed count by the sample rate reached 1.0 a fifth of the way in,
        # because it was reconstructing the read position from a number that
        # already was it.
        done = min(1.0, session.frames / total) if session and total else 0
        return JSONResponse({
            "running": bool(session and session.running),
            "stop_reason": session.stop_reason if session else None,
            "frames": session.frames if session else 0,
            "analysed": session.analysed if session else 0,
            "source": ("video" if session and session.is_file
                       else "camera" if session else None),
            "total_frames": total,
            "progress": round(done, 3) if total else None,
        })

    @app.websocket("/ws")
    async def ws(socket: WebSocket):
        """Push the live frame and analytics to the page.

        Checked before accepting, not after. The same-origin policy does not
        apply to WebSocket handshakes, so any page a teacher visits could
        otherwise open this socket and receive the live video; accepting and
        then closing would already have completed the handshake it needs.
        """
        if not await webauth.websocket_allowed(socket, token):
            await socket.close(code=1008)
            return
        await socket.accept()
        # What this client was last sent. The socket used to push on a 0.06s
        # timer whatever the session held, so a video analysed at 1.2 frames
        # per second was sent the same frame about thirteen times -- measured
        # on one 58-frame run: 535 payloads, each carrying a 72 KB base64
        # JPEG, about 38 MB for eleven seconds of footage, with the browser
        # decoding sixteen identical images a second to draw one.
        #
        # Tracked per connection rather than on the session, so a page that
        # connects midway is still sent the current frame at once instead of
        # a blank panel until something next changes.
        sent: int | str | None = None
        try:
            while True:
                session = state["session"]
                # Not `session.running`. A video job publishes its resolved
                # frame -- the one where the boxes finally carry student ids
                # instead of "unidentified" -- after the run ends, so gating
                # on running meant the page never received the only frame
                # that showed what the analysis concluded. It also blanked
                # the feed the instant a session stopped, discarding the last
                # thing anyone was looking at.
                if session is not None and session.frame_jpeg:
                    with session._lock:
                        revision = session.revision
                        jpeg, payload = session.frame_jpeg, session.payload
                    if jpeg and payload and revision != sent:
                        sent = revision
                        payload = dict(payload)
                        payload["image"] = base64.b64encode(jpeg).decode("ascii")
                        await socket.send_text(json.dumps(payload))
                elif sent != "idle":
                    # Idle is a state, not a heartbeat. Repeating it sixteen
                    # times a second told the page nothing it did not know.
                    sent = "idle"
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
    from tools.run.report import build

    graph_path = out / "live_graph.jsonl"
    if not graph_path.exists() or not graph_path.stat().st_size:
        return None
    profiles = build_profiles(graph_path, None)
    (out / "live_profiles.json").write_text(
        json.dumps(list(profiles.values()), indent=2), encoding="utf-8")
    return build(out, gallery)


def app(**overrides):
    """ASGI factory, so `uvicorn tools.run.server:app --factory` works.

    There is deliberately no module-level `app` object. Building one at
    import time would open the camera and load every model the moment
    anything imported this module -- including the test suite.

    This exists because `uvicorn tools.run.server:app` is the first thing
    anyone tries, and it failed with `Attribute "app" not found`, which says
    nothing about what to do instead. The supported invocation is still
    `python -m tools.run.server`, which prints the access token the dashboard
    needs; a bare uvicorn command cannot, so it generates one you then have
    to read out of the environment.
    """
    import argparse as _argparse

    defaults = {
        "host": "127.0.0.1", "port": 8000, "camera": 0,
        "gallery": "outputs/enrollment/gallery.json", "out": "outputs/live",
        "db": "outputs/classgraph.db", "stride": 3, "imgsz": None,
        "min_face_px": 60, "yaw_reference": None, "verbose": False,
    }
    defaults.update(overrides)
    built = create_app(_argparse.Namespace(**defaults))
    print(f"  access token: {built.state.classgraph_token}")
    return built


def main() -> int:
    """Parse arguments and serve the dashboard."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--gallery", default="outputs/enrollment/gallery.json")
    parser.add_argument("--out", default="outputs/live")
    parser.add_argument("--db", default="outputs/classgraph.db",
                        help="session database to erase from")
    parser.add_argument("--stride", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--min-face-px", type=int, default=60)
    parser.add_argument("--yaw-reference", type=float, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    from backend import webauth

    import uvicorn

    app = create_app(args)
    token = app.state.classgraph_token
    print(f"ClassGraph dashboard -> http://{args.host}:{args.port}")
    if os.environ.get(webauth.TOKEN_ENV, "").strip():
        print("  access token: taken from " + webauth.TOKEN_ENV)
    else:
        print(f"  access token: {token}")
        print("  (generated for this run; set "
              f"{webauth.TOKEN_ENV} to keep one across restarts)")
    if args.host not in ("127.0.0.1", "localhost"):
        print(f"  WARNING: bound to {args.host}. Anything that can reach this")
        print("           port can reach the camera. Put it behind TLS and a")
        print("           reverse proxy before using it on a real network.")
    # websockets 14 removed the legacy API uvicorn's default "websockets"
    # implementation is built on, and the mismatch shows up as every WebSocket
    # handshake being refused with a bare 403 before the route is even reached.
    # The sans-io implementation is the maintained path and works with 14+.
    uvicorn.run(app, host=args.host, port=args.port,
                log_level="warning", ws="websockets-sansio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
