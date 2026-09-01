"""Print the raw signals the action rules read, one line per frame.

Every complaint about a wrong label is really a question about a number: the
mouth-opening ratio behind a missed yawn, the eye aspect ratio behind a blink
scored as a closure, the vertical lean behind a posture that reads upright when
somebody is lying down. Those numbers are computed and then thrown away, so the
only thing left to argue with is the label.

This prints them. Perform a behaviour, read the column, and the threshold that
is wrong becomes obvious -- which is also the calibration the yawn threshold
has never had (see actions.YAWN_RATIO).
"""
import argparse
import sys
import time

import cv2

sys.path.insert(0, ".")

from backend.actions import mouth_open_ratio
from backend.config import CONFIG
from backend.detection import Detector
from backend.face import FaceAnalyzer
from backend.headpose import HeadPoseEstimator
from backend.posture import PostureAnalyzer


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=30.0)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"camera {args.camera} did not open", file=sys.stderr)
        return 1
    for _ in range(10):          # let exposure settle; early frames come back black
        cap.read()

    detector = Detector(CONFIG.detection)
    print(f"{'t':>5} {'ppl':>4} {'mouth':>6} {'ear':>6} {'lean':>6} {'pitch':>6}  objects")
    print(f"{'':>5} {'':>4} {'>' + str(0.35):>6} {'<0.21':>6} {'':>6} {'':>6}")
    start = time.time()
    hp = HeadPoseEstimator(CONFIG.headpose)
    with FaceAnalyzer(CONFIG.face) as analyzer, PostureAnalyzer(CONFIG.posture) as poser:
        while time.time() - start < args.seconds:
            ok, frame = cap.read()
            if not ok:
                break
            persons, objects = detector.detect(frame)
            boxes = [p.bbox for p in persons]
            if not boxes:
                continue
            faces = analyzer.analyze(frame, boxes)
            poses = hp.estimate(frame, [f.face_bbox for f in faces])
            postures = poser.analyze(frame, boxes)

            f, po, ps = faces[0], poses[0], postures[0]
            ratio = mouth_open_ratio(f.landmarks) if f.landmarks else None
            lean = getattr(ps, "vertical_lean", None)
            pitch = getattr(po, "pitch", None)

            def show(v, fmt="{:.3f}"):
                return "  -   " if v is None else fmt.format(v)

            print(f"{time.time() - start:>5.1f} {len(persons):>4} "
                  f"{show(ratio):>6} {show(f.ear):>6} {show(lean, '{:+.2f}'):>6} "
                  f"{show(pitch, '{:+.0f}'):>6}  "
                  f"{','.join(sorted({o['cls'] for o in objects})) or '-'}")
    cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
