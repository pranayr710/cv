"""Show what the detector sees, and what it nearly saw.

When an object is labelled wrongly -- a held phone reported as a laptop -- the
pipeline shows only the winner, so there is no way to tell whether the right
class was a close second or never considered at all. Those need different
fixes: the first is a threshold, the second is the detector's training.

This holds an object up to the camera and prints the ranked detections,
including ones far below the shipping threshold, so the distinction is visible.

    python tools/probe_objects.py                 # one frame, now
    python tools/probe_objects.py --seconds 10    # keep sampling
    python tools/probe_objects.py --save shot.jpg
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import CONFIG
from backend.detection import Detector

#: Far below anything the pipeline acts on, so a near-miss is still listed.
PROBE_CONF = 0.03

#: Classes most often confused with one another on a held object, printed even
#: at zero so their absence is as visible as their presence.
WATCH = ("cell phone", "laptop", "book", "remote", "tv", "keyboard", "mouse")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="keep sampling for this long (0 = a single frame)")
    ap.add_argument("--save", type=Path, default=None,
                    help="write the frame that was analysed")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"camera {args.camera} did not open. Another program may be "
              "using it -- try tools/stop_servers.py --kill")
        return 1
    for _ in range(12):        # let exposure settle
        cap.read()

    detector = Detector(dataclasses.replace(
        CONFIG.detection, object_conf=PROBE_CONF,
        object_conf_near_person=PROBE_CONF))

    print(f"shipping thresholds: object_conf={CONFIG.detection.object_conf}, "
          f"near a person={CONFIG.detection.object_conf_near_person}")
    print(f"probing at {PROBE_CONF} so near-misses are visible\n")

    deadline = time.time() + max(args.seconds, 0.0)
    first = True
    while first or time.time() < deadline:
        first = False
        ok, frame = cap.read()
        if not ok:
            print("camera stopped returning frames")
            break
        persons, objects = detector.detect(frame)
        ranked = sorted(objects, key=lambda o: -o.confidence)

        print(f"[{time.strftime('%H:%M:%S')}]  {len(persons)} person(s), "
              f"{len(ranked)} object(s)")
        if not ranked:
            print("    nothing at all, even at 0.03 -- the detector is not "
                  "seeing an object here")
        for obj in ranked[:8]:
            _x, _y, w, h = obj.bbox
            shipped = ("SHIPS" if obj.confidence
                       >= CONFIG.detection.object_conf_near_person else "  -  ")
            print(f"    {shipped}  {obj.cls:<14} {obj.confidence:.3f}  "
                  f"{w}x{h}px")

        seen = {o.cls: o.confidence for o in ranked}
        missing = [c for c in WATCH if c not in seen]
        if missing:
            print(f"    not detected at all: {', '.join(missing)}")
        print()
        if args.seconds:
            time.sleep(1.0)

    if args.save is not None and "frame" in dir():
        cv2.imwrite(str(args.save), frame)
        print(f"frame written to {args.save}")
    cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
