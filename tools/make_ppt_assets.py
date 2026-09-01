"""Render the images the deck refers to.

build_ppt.py degrades gracefully when these are missing, which is how the deck
ended up shipping with three empty picture slots and an argument nobody could
see. The images are regenerated from the dataset rather than kept as binaries so
they always show what the current code actually does.
"""
import sys
from pathlib import Path

import cv2

sys.path.insert(0, ".")

from backend.config import CONFIG
from backend.detection import Detector
from backend.posture import PostureAnalyzer

OUT = Path("ppt_assets")
IMAGES = sorted(Path("dataset/23-08/test/images").glob("*.jpg"))


def busiest() -> Path:
    """The image with the most students, so the comparison has something to show."""
    d = Detector(CONFIG.detection)
    best, count = IMAGES[0], -1
    for p in IMAGES[:20]:
        f = cv2.imread(str(p))
        if f is None:
            continue
        n = len(d.detect(f)[0])
        if n > count:
            best, count = p, n
    print(f"busiest image: {best.name} ({count} persons)")
    return best


def draw(frame, persons, colour):
    out = frame.copy()
    for p in persons:
        x, y, w, h = p.bbox
        cv2.rectangle(out, (x, y), (x + w, y + h), colour, max(2, out.shape[1] // 640))
    return out


def main() -> int:
    import dataclasses

    OUT.mkdir(exist_ok=True)
    src = busiest()
    frame = cv2.imread(str(src))

    lo = Detector(dataclasses.replace(CONFIG.detection, imgsz=960, person_conf=0.40))
    hi = Detector(dataclasses.replace(CONFIG.detection, imgsz=1536, person_conf=0.30))
    a, b = lo.detect(frame)[0], hi.detect(frame)[0]

    cv2.imwrite(str(OUT / "annot_baseline_960_c40.jpg"), draw(frame, a, (60, 60, 220)))
    cv2.imwrite(str(OUT / "annot_candidate_1536_c30.jpg"), draw(frame, b, (80, 190, 90)))
    print(f"baseline 960/0.40: {len(a)} persons -> annot_baseline_960_c40.jpg")
    print(f"candidate 1536/0.30: {len(b)} persons -> annot_candidate_1536_c30.jpg")

    # The posture fallback, drawn on the same scene: skeletons for students the
    # face pipeline cannot reach.
    with PostureAnalyzer(CONFIG.posture) as poser:
        postures = poser.analyze(frame, [p.bbox for p in b])
    pose_img = frame.copy()
    drawn = 0
    for ps in postures:
        if ps is None:
            continue
        pts = [getattr(ps, n, None) for n in
               ("nose", "left_shoulder", "right_shoulder", "left_wrist", "right_wrist")]
        pts = [p for p in pts if p]
        if len(pts) < 2:
            continue
        drawn += 1
        sh = [p for p in (getattr(ps, "left_shoulder", None),
                          getattr(ps, "right_shoulder", None)) if p]
        if len(sh) == 2:
            cv2.line(pose_img, tuple(map(int, sh[0])), tuple(map(int, sh[1])),
                     (80, 190, 90), 2)
        for p in pts:
            cv2.circle(pose_img, tuple(map(int, p)), 4, (40, 220, 255), -1)
    cv2.imwrite(str(OUT / "t1_pose.jpg"), pose_img)
    print(f"posture skeletons drawn for {drawn} of {len(b)} persons -> t1_pose.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
