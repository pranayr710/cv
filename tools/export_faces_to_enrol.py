"""Pull each student's clearest faces out of a processed video, ready to name.

Registering someone from a recording is a chicken-and-egg problem: the gallery
turns a face into a name, but the only faces you have are inside the video, and
the dashboard's own enrolment captures from a webcam. So there is no way to put
a name on a student who exists only in footage.

This closes that. It reads a session the pipeline has already produced, picks
the clearest face for each student it found, and writes them into one folder
per student. Rename the folders to the students' names and register them:

    python tools/export_faces_to_enrol.py outputs/live --video <the file>
    # look at outputs/enrol_crops/, rename person_3 -> "Asha", etc.
    python tools/register_faces.py --images outputs/enrol_crops

**On whether this will work.** Registration quality is set by face size, and
classroom footage is unkind: ArcFace separates a 112 px face reliably and a
30 px one not at all. This reports the sizes it found so the answer is visible
before the time is spent, rather than after a gallery that matches nobody.
Faces below the floor are still exported, marked, and left for a human to
decide on -- a blurry crop a person recognises is better evidence than a sharp
one of the wrong student.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

#: Shorter side below which ArcFace embeddings stop separating people. The
#: same floor registration applies, repeated here so the warning arrives
#: before the work rather than after it.
RELIABLE_FACE_PX = 60

#: How many crops per student. More shots average out a bad frame; past about
#: a dozen the mean stops moving and the extra files only cost review time.
SHOTS_PER_PERSON = 12

#: Pixels of context around the face box. A tight crop loses the hairline and
#: jaw that a person uses to recognise someone, and this is reviewed by eye
#: before it is registered.
PAD = 0.35


def best_faces(frames_path: Path) -> dict[int, list[tuple[int, int, list]]]:
    """``{person_id: [(frame_id, face_px, bbox), ...]}``, clearest first.

    Sorted by face size, which is the measure that predicts whether an
    embedding will be usable. The detector's own confidence is not: it reports
    how sure it is that a face is there, not how much of one was visible.
    """
    found: dict[int, list[tuple[int, int, list]]] = defaultdict(list)
    for line in frames_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        frame_id = int(row.get("frame_id") or 0)
        for person in row.get("persons", []):
            pid = person.get("person_id")
            face = person.get("face") or {}
            box = face.get("bbox")
            if not pid or pid <= 0 or not box:
                continue
            found[int(pid)].append((frame_id, int(min(box[2], box[3])), box))
    for shots in found.values():
        shots.sort(key=lambda s: -s[1])
    return found


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", type=Path, nargs="?",
                    default=Path("outputs/live"),
                    help="a processed session folder")
    ap.add_argument("--video", type=Path, required=True,
                    help="the recording that session was produced from")
    ap.add_argument("--out", type=Path, default=Path("outputs/enrol_crops"))
    ap.add_argument("--shots", type=int, default=SHOTS_PER_PERSON)
    args = ap.parse_args()

    import cv2

    # Both files can exist in one folder: the live loop writes
    # live_frames.jsonl and the batch pipeline writes raw.jsonl, and a folder
    # used for both holds one of each. Choosing by name reads whichever is
    # checked first -- which was a stale session that had resolved nobody,
    # while a newer one beside it had resolved five. Choosing by result
    # cannot make that mistake.
    candidates = [p for p in (args.session / "raw.jsonl",
                              args.session / "live_frames.jsonl")
                  if p.exists()]
    if not candidates:
        print(f"No per-frame records in {args.session}.")
        print("Analyse the video first, from the dashboard or with "
              "`classgraph video`.")
        return 1
    if not args.video.exists():
        print(f"{args.video} not found.")
        return 1

    frames_path, people = max(
        ((path, best_faces(path)) for path in candidates),
        key=lambda pair: len(pair[1]))
    if len(candidates) > 1:
        print(f"reading {frames_path.name} -- "
              f"{len(people)} student(s) resolved there\n")
    if not people:
        print("No student was resolved to a stable id in this session, so")
        print("there is nobody to register. That usually means the faces were")
        print("too small for the resolver to found an identity -- the session")
        print("report says what sizes it saw.")
        return 1

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        print(f"Could not open {args.video}.")
        return 1

    # Frames are read in order once, rather than seeking per crop: seeking in
    # a compressed video is slow and, on a damaged file, unreliable.
    wanted: dict[int, list[tuple[int, list]]] = defaultdict(list)
    for pid, shots in people.items():
        for frame_id, _px, box in shots[:args.shots]:
            wanted[frame_id].append((pid, box))

    args.out.mkdir(parents=True, exist_ok=True)
    written: dict[int, int] = defaultdict(int)
    frame_id = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        for pid, box in wanted.get(frame_id, []):
            x, y, w, h = (int(v) for v in box[:4])
            pad_x, pad_y = int(w * PAD), int(h * PAD)
            x0 = max(0, x - pad_x)
            y0 = max(0, y - pad_y)
            x1 = min(frame.shape[1], x + w + pad_x)
            y1 = min(frame.shape[0], y + h + pad_y)
            crop = frame[y0:y1, x0:x1]
            if crop.size == 0:
                continue
            folder = args.out / f"person_{pid}"
            folder.mkdir(exist_ok=True)
            written[pid] += 1
            cv2.imwrite(str(folder / f"shot_{written[pid]:02d}.jpg"), crop)
        frame_id += 1
    capture.release()

    print(f"{len(people)} student(s) found in {args.session}\n")
    print(f"  {'student':<12}{'crops':>7}{'best face':>12}{'usable?':>10}")
    usable = 0
    for pid in sorted(people):
        best = people[pid][0][1]
        ok_size = best >= RELIABLE_FACE_PX
        usable += ok_size
        print(f"  person_{pid:<5}{written[pid]:>7}{best:>9} px"
              f"{'yes' if ok_size else 'marginal':>10}")

    print(f"\n  crops written to {args.out}")
    print("\nNext:")
    print(f"  1. Open {args.out} and look at each folder.")
    print("  2. Rename person_1, person_2 ... to the students' names.")
    print("     Delete any folder that holds more than one person, or")
    print("     nobody you can identify -- a wrong name is worse than none.")
    print(f"  3. python tools/register_faces.py --images {args.out}")
    print("\n  After that, every video gives those students the same name.")

    print()
    print("  Before spending time on this: on our own overhead footage")
    print("  registration scores these crops 0.36 to 0.45 against a 0.5")
    print("  floor, and refuses all of them. That refusal is right -- a face")
    print("  embedded below the floor does not separate people, so a gallery")
    print("  built from them matches the wrong student confidently instead")
    print("  of failing honestly.")
    print()
    print("  This works on faces of about 100px or more, looking toward the")
    print("  camera. For an overhead classroom shot, register the students")
    print("  from a webcam instead, or keep the stable Student 1..N labels,")
    print("  which are what the report is keyed on anyway.")

    if usable < len(people):
        print(f"\n  {len(people) - usable} of {len(people)} have a best face "
              f"below {RELIABLE_FACE_PX} px. ArcFace separates a 112 px face")
        print("  reliably and a 30 px one not at all, so those may register")
        print("  and then match the wrong student, or nobody. Prefer footage")
        print("  shot closer, or register those students from a webcam.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
