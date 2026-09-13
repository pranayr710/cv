"""Render what each model draws on the same frame.

"What does the output look like?" is a fair question that a table of parameter
counts cannot answer. So each model is run on one classroom frame and its own
output is drawn -- boxes for YOLO, five keypoints for SCRFD, a skeleton for
Pose, the 468-point mesh for Face Mesh, a rotation gizmo for SixDRepNet.

The point of putting them on the *same* frame is that the pipeline is a chain:
SCRFD searches inside YOLO's boxes, Face Mesh runs on SCRFD's crop, and so on.
Six separate demo images would hide that.

Writes ppt_assets/model_io_<name>.jpg plus a combined model_io_grid.jpg.
"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, ".")

from backend.config import CONFIG
from backend.detection import Detector
from backend.face import FaceAnalyzer
from backend.headpose import HeadPoseEstimator
from backend.posture import PostureAnalyzer

OUT = Path("ppt_assets")
IMAGES = Path("dataset/23-08/test/images")

FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (90, 200, 90)
AMBER = (40, 170, 235)
CYAN = (235, 200, 60)
MAGENTA = (200, 90, 220)
RED = (70, 70, 225)
WHITE = (255, 255, 255)
INK = (50, 35, 20)

#: MediaPipe Pose indices we actually consume, as (a, b) bone pairs.
BONES = (("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"),
         ("left_elbow", "left_wrist"), ("right_shoulder", "right_elbow"),
         ("right_elbow", "right_wrist"), ("shoulder_mid", "hip_mid"),
         ("shoulder_mid", "nose"))


def banner(img, title, subtitle, colour=GREEN):
    h = 74
    strip = img[:h].copy()
    cv2.rectangle(strip, (0, 0), (img.shape[1], h), INK, -1)
    img[:h] = cv2.addWeighted(strip, 0.85, img[:h], 0.15, 0)
    cv2.putText(img, title, (16, 30), FONT, 0.82, WHITE, 2, cv2.LINE_AA)
    cv2.putText(img, subtitle, (16, 60), FONT, 0.62, colour, 2, cv2.LINE_AA)
    return img


def pick_frame(det):
    """The frame where every model has something to draw.

    Ranked on objects first, then people: a frame with no held objects makes the
    detection panel look like a person detector, which is half its job.
    """
    best, score = None, (-1, -1)
    for path in sorted(IMAGES.glob("*.jpg"))[:25]:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        persons, objects = det.detect(frame)
        if len(persons) < 6:
            continue
        rank = (len(objects), len(persons))
        if rank > score:
            best, score = frame, rank
    return best, score


def main() -> int:
    OUT.mkdir(exist_ok=True)
    det = Detector(CONFIG.detection)
    frame, _rank = pick_frame(det)
    if frame is None:
        print("no dataset frame found")
        return 1
    persons, objects = det.detect(frame)
    boxes = [p.bbox for p in persons]
    print(f"frame: {frame.shape[1]}x{frame.shape[0]}, {len(persons)} persons, "
          f"{len(objects)} objects")

    panels = {}

    # --- 1. YOLO11m: boxes, classes, confidences ------------------------------
    img = frame.copy()
    for p in persons:
        x, y, w, h = p.bbox
        cv2.rectangle(img, (x, y), (x + w, y + h), GREEN, 2)
        cv2.putText(img, f"person {getattr(p, 'confidence', 0):.2f}",
                    (x, max(80, y - 6)), FONT, 0.42, GREEN, 1, cv2.LINE_AA)
    for o in objects:
        x, y, w, h = o.bbox
        cv2.rectangle(img, (x, y), (x + w, y + h), AMBER, 2)
        cv2.putText(img, f"{o.cls} {o.confidence:.2f}", (x, max(80, y - 6)),
                    FONT, 0.42, AMBER, 1, cv2.LINE_AA)
    panels["yolo"] = banner(img, "YOLO11m  -  detection",
                            f"{len(persons)} persons + {len(objects)} objects, "
                            f"each a box + class + score")

    with FaceAnalyzer(CONFIG.face) as fa, PostureAnalyzer(CONFIG.posture) as pa:
        faces = fa.analyze(frame, boxes)
        postures = pa.analyze(frame, boxes)

        # --- 2. SCRFD: face boxes and 5 keypoints -------------------------
        img = frame.copy()
        n_face = 0
        for f in faces:
            if f.face_bbox is None:
                continue
            n_face += 1
            x, y, w, h = f.face_bbox
            cv2.rectangle(img, (x, y), (x + w, y + h), CYAN, 2)
            for pt in (f.kps if f.kps is not None else []):
                cv2.circle(img, (int(pt[0]), int(pt[1])), 3, MAGENTA, -1)
        panels["scrfd"] = banner(
            img, "SCRFD det_10g  -  face detection",
            f"{n_face} faces, each with 5 keypoints (magenta)",
            CYAN)

        # --- 3. MediaPipe Pose: the skeleton we consume -------------------
        img = frame.copy()
        n_pose = 0
        for ps in postures:
            if ps is None:
                continue
            pts = {k: getattr(ps, k, None) for k in
                   ("nose", "left_shoulder", "right_shoulder", "shoulder_mid",
                    "hip_mid", "left_elbow", "right_elbow", "left_wrist",
                    "right_wrist")}
            if not pts["left_shoulder"] or not pts["right_shoulder"]:
                continue
            n_pose += 1
            for a, b in BONES:
                if pts.get(a) and pts.get(b):
                    cv2.line(img, tuple(map(int, pts[a])), tuple(map(int, pts[b])),
                             GREEN, 2)
            for v in pts.values():
                if v:
                    cv2.circle(img, tuple(map(int, v)), 3, AMBER, -1)
            # The facing ray -- the input to the room-layout measurement.
            fd = getattr(ps, "facing_direction", None)
            if fd and pts.get("shoulder_mid"):
                sx, sy = pts["shoulder_mid"]
                cv2.arrowedLine(img, (int(sx), int(sy)),
                                (int(sx + fd[0] * 70), int(sy + fd[1] * 70)),
                                MAGENTA, 2, tipLength=0.3)
        panels["pose"] = banner(
            img, "MediaPipe Pose  -  body geometry",
            f"{n_pose} skeletons; magenta = facing_direction")

        # --- 4. Face Mesh: 468 landmarks on one face ----------------------
        crop = None
        for f in faces:
            if f.landmarks and f.face_bbox:
                x, y, w, h = f.face_bbox
                pad = int(max(w, h) * 0.55)
                H, W = frame.shape[:2]
                x0, y0 = max(0, x - pad), max(0, y - pad)
                x1, y1 = min(W, x + w + pad), min(H, y + h + pad)
                crop = frame[y0:y1, x0:x1].copy()
                scale = 520 / max(crop.shape[:2])
                crop = cv2.resize(crop, None, fx=scale, fy=scale)
                for lx, ly in f.landmarks:
                    cv2.circle(crop, (int((lx - x0) * scale), int((ly - y0) * scale)),
                               1, CYAN, -1)
                crop = banner(crop, "MediaPipe Face Mesh  -  468 landmarks",
                              f"EAR {f.ear:.3f}" if f.ear is not None
                              else "EAR unavailable", CYAN)
                break
        if crop is not None:
            panels["facemesh"] = crop

    # --- 5. SixDRepNet: rotation drawn as an axis gizmo ---------------------
    hp = HeadPoseEstimator(CONFIG.headpose)
    poses = hp.estimate(frame, [f.face_bbox for f in faces])
    img = frame.copy()
    n_hp = 0
    for f, po in zip(faces, poses):
        if po is None or f.face_bbox is None:
            continue
        n_hp += 1
        x, y, w, h = f.face_bbox
        cx, cy, size = x + w // 2, y + h // 2, max(w, h) * 0.9
        yaw, pitch, roll = (np.radians(po.yaw), np.radians(po.pitch),
                            np.radians(po.roll))
        # Standard head-pose gizmo: project the three body axes into the image.
        axes = (
            ((np.cos(yaw) * np.cos(roll)), (np.cos(pitch) * np.sin(roll)
             + np.cos(roll) * np.sin(pitch) * np.sin(yaw)), (0, 0, 255)),
            ((-np.cos(yaw) * np.sin(roll)), (np.cos(pitch) * np.cos(roll)
             - np.sin(pitch) * np.sin(yaw) * np.sin(roll)), (0, 255, 0)),
            ((np.sin(yaw)), (-np.cos(yaw) * np.sin(pitch)), (255, 0, 0)),
        )
        for dx, dy, colour in axes:
            cv2.arrowedLine(img, (cx, cy),
                            (int(cx + size * dx), int(cy + size * dy)),
                            colour, 2, tipLength=0.25)
    panels["headpose"] = banner(
        img, "SixDRepNet  -  head pose",
        f"{n_hp} heads; red/green/blue are the three rotation axes", RED)

    for name, im in panels.items():
        cv2.imwrite(str(OUT / f"model_io_{name}.jpg"), im)
        print(f"  -> model_io_{name}.jpg  {im.shape[1]}x{im.shape[0]}")

    # Combined 2x2 of the full-frame panels, for a single overview slide.
    order = [panels[k] for k in ("yolo", "scrfd", "pose", "headpose") if k in panels]
    if len(order) == 4:
        h = min(im.shape[0] for im in order)
        row = [cv2.resize(im, (int(im.shape[1] * h / im.shape[0]), h)) for im in order]
        grid = cv2.vconcat([cv2.hconcat(row[:2]), cv2.hconcat(row[2:])])
        cv2.imwrite(str(OUT / "model_io_grid.jpg"), grid)
        print(f"  -> model_io_grid.jpg  {grid.shape[1]}x{grid.shape[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
