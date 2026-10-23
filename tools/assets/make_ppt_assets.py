"""Render the images the deck refers to.

build_ppt.py skips a missing asset silently, which is how the deck came to ship
with empty picture slots. Worse than empty is an image that does not show what
its caption claims: the first version of this script drew every person box on a
crowded room, where a 17-vs-19 difference is invisible, and scattered pose dots
under a caption about a skeleton nobody could see.

So each image here is chosen to make one claim legible:

* ``imgsz_collapse.jpg`` -- the same webcam frame at three inference sizes. The
  box shrinks, then the person disappears. The failure is the picture.
* ``fallback_coverage.jpg`` -- a room where most faces are unreadable, with each
  person coloured by which signal reached them. Picked by measurement (the
  frame with the most pose recoveries), not by eye.

Both burn their own counts into the image, so a caption cannot drift away from
what is on screen.
"""
import dataclasses
import sys
from pathlib import Path

import cv2

sys.path.insert(0, ".")

from backend.config import CONFIG
from backend.detection import Detector
from backend.face import FaceAnalyzer
from backend.posture import PostureAnalyzer

OUT = Path("ppt_assets")
IMAGES = Path("dataset/23-08/test/images")

FONT = cv2.FONT_HERSHEY_SIMPLEX
GREEN = (90, 200, 90)
AMBER = (40, 170, 235)
RED = (70, 70, 225)
WHITE = (255, 255, 255)
INK = (50, 35, 20)


def banner(img, lines, colours, height=None):
    """A caption bar across the top, so the count travels with the picture."""
    h = height or int(30 + 34 * len(lines))
    bar = img[:h].copy()
    cv2.rectangle(bar, (0, 0), (img.shape[1], h), INK, -1)
    img[:h] = cv2.addWeighted(bar, 0.82, img[:h], 0.18, 0)
    for i, (text, colour) in enumerate(zip(lines, colours)):
        cv2.putText(img, text, (16, 34 + i * 34), FONT, 0.78, colour, 2, cv2.LINE_AA)
    return img


def collapse(webcam: Path) -> None:
    """The same frame at three inference sizes, side by side."""
    frame = cv2.imread(str(webcam))
    if frame is None:
        print(f"  skipped imgsz_collapse: no webcam frame at {webcam}")
        return
    # The cap is in effect now, so asking for imgsz 1920 no longer gets 1920.
    # Lifting it here is what reproduces the failure the slide is about -- the
    # picture has to show the bug, not the code that prevents it.
    import backend.detection as det_mod

    original_cap = det_mod.MAX_UPSCALE
    det_mod.MAX_UPSCALE = 1e9

    panels = []
    for size in (640, 1600, 1920):
        det = Detector(dataclasses.replace(CONFIG.detection, imgsz=size))
        persons, _ = det.detect(frame)
        panel = frame.copy()
        for p in persons:
            x, y, w, h = p.bbox
            cv2.rectangle(panel, (x, y), (x + w, y + h), GREEN, 3)
        conf = max((getattr(p, "confidence", 0.0) for p in persons), default=0.0)
        found = (f"{len(persons)} person   conf {conf:.2f}" if persons
                 else "NOBODY FOUND")
        banner(panel, [f"imgsz {size}  ({size / max(frame.shape[:2]):.1f}x)", found],
               [WHITE, GREEN if persons else RED])
        panels.append(panel)
        print(f"  imgsz {size}: {len(persons)} persons")
    det_mod.MAX_UPSCALE = original_cap
    cv2.imwrite(str(OUT / "imgsz_collapse.jpg"), cv2.hconcat(panels))
    print("  -> imgsz_collapse.jpg  (cap lifted to reproduce the pre-fix behaviour)")


def _shoulders(p) -> bool:
    return (p is not None and getattr(p, "left_shoulder", None)
            and getattr(p, "right_shoulder", None))


def coverage() -> None:
    """One room, each person coloured by which signal actually reached them."""
    det = Detector(CONFIG.detection)
    best = None
    with FaceAnalyzer(CONFIG.face) as fa, PostureAnalyzer(CONFIG.posture) as pa:
        for path in sorted(IMAGES.glob("*.jpg"))[:30]:
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            persons, _ = det.detect(frame)
            if len(persons) < 8:
                continue
            boxes = [p.bbox for p in persons]
            faces, posts = fa.analyze(frame, boxes), pa.analyze(frame, boxes)
            rec = sum(1 for f, q in zip(faces, posts)
                      if f.face_bbox is None and _shoulders(q))
            # Prefer the room where the fallback carries the most weight -- that
            # is the claim the picture has to support.
            score = (rec, -sum(1 for f in faces if f.face_bbox is not None))
            if best is None or score > best[0]:
                best = (score, path, frame, persons, faces, posts)

    if best is None:
        print("  skipped fallback_coverage: no suitable image")
        return
    _, path, frame, persons, faces, posts = best
    img = frame.copy()
    n_face = n_pose = n_none = 0
    for p, f, q in zip(persons, faces, posts):
        x, y, w, h = p.bbox
        if f.face_bbox is not None:
            colour, n_face = GREEN, n_face + 1
        elif _shoulders(q):
            colour, n_pose = AMBER, n_pose + 1
        else:
            colour, n_none = RED, n_none + 1
        cv2.rectangle(img, (x, y), (x + w, y + h), colour, 3)
        if colour is AMBER:
            a = tuple(map(int, q.left_shoulder))
            b = tuple(map(int, q.right_shoulder))
            cv2.line(img, a, b, AMBER, 3)
            cv2.circle(img, a, 5, AMBER, -1)
            cv2.circle(img, b, 5, AMBER, -1)

    total = len(persons)
    banner(img,
           [f"{total} students detected",
            f"{n_face} readable face", f"{n_pose} recovered by body pose",
            f"{n_none} neither -- reported as unknown"],
           [WHITE, GREEN, AMBER, RED])
    cv2.imwrite(str(OUT / "fallback_coverage.jpg"), img)
    print(f"  {path.name[:32]}: {total} persons, "
          f"{n_face} face / {n_pose} pose / {n_none} neither")
    print(f"  coverage {100 * n_face / total:.0f}% -> "
          f"{100 * (n_face + n_pose) / total:.0f}%  -> fallback_coverage.jpg")


def main() -> int:
    OUT.mkdir(exist_ok=True)
    scratch = Path(sys.argv[1]) / "room.jpg" if len(sys.argv) > 1 else Path("room.jpg")
    print("imgsz collapse:")
    collapse(scratch)
    print("pose fallback coverage:")
    coverage()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
