"""Render a finished run back onto its video, as the artifact a reviewer watches.

Everything Part 1 claims is per-person and persistent: the same student keeps one
id from the first frame to the last, and carries an expression, a behaviour and a
concentration figure. None of that is checkable from a JSONL file or a summary
table -- a reviewer has to see the id stay on the same face while the camera
pans, and see the label sitting on the right person.

So this draws the run's own output back onto its own frames:

* green box    a student, face-verified id
* amber box    a student whose id was never confirmed by a face (negative id)
* red box      rejected -- a printed face, transient noise, or the instructor
* blue box     the declared instructor
* cyan box     the face the id was matched on
* header       frame time, how many students are on screen, running totals

Two things are deliberately visible rather than hidden. Rejected detections are
drawn, in red, with the reason -- a viewer should be able to see what the
pipeline threw away and disagree with it. And a concentration figure computed
with no behaviour evidence behind it is marked with "?" rather than shown as a
confident number, matching the caveat backend/student_profile.py already emits.

Run:
    python -m tools.render_video --jsonl outputs/wa_no_posters.jsonl \\
        --video "dataset/23-08/vedio/WhatsApp Video 2026-08-23 at 10.16.09.mp4" \\
        --out outputs/annotated.mp4 --fps 3
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

GREEN = (0, 220, 0)
AMBER = (0, 200, 255)
RED = (0, 0, 255)
BLUE = (255, 160, 0)
CYAN = (255, 200, 0)
WHITE = (255, 255, 255)

FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX


def _colour_for(profile: dict | None) -> tuple[int, int, int]:
    if profile is None:
        return AMBER
    if profile.get("role") == "instructor":
        return BLUE
    if not profile.get("is_student", True):
        return RED
    return GREEN if profile.get("face_verified") else AMBER


def _labels_for(person: dict, profile: dict | None) -> list[tuple[str, tuple[int, int, int]]]:
    """The lines drawn under a person's box, each with its own colour."""
    out: list[tuple[str, tuple[int, int, int]]] = []
    expr = (person.get("expression") or {}).get("label")
    if expr:
        out.append((expr, WHITE))
    beh = person.get("behaviour") or {}
    if beh.get("label"):
        # Reliability comes straight from the run; a weak label is marked so a
        # reviewer does not read it as a firm finding.
        weak = beh.get("reliability") == "weak"
        out.append((f"{beh['label']}{'?' if weak else ''}", WHITE))
    if profile and profile.get("is_student"):
        conc = profile.get("concentration") or {}
        pct = conc.get("concentration_pct")
        if pct is not None:
            # "?" where nothing off-task was ever detectable, so the figure
            # comes from absence of evidence rather than evidence of attention.
            mark = "" if conc.get("off_task_detectable") else "?"
            out.append((f"conc {pct:.0f}%{mark}", WHITE))
    return out


def render(jsonl: Path, video: Path, out: Path, fps: float, max_frames: int | None) -> None:
    import cv2

    from backend.student_profile import build_profiles

    records = [
        json.loads(line)
        for line in jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records.sort(key=lambda r: r["frame_id"])
    if max_frames:
        records = records[:max_frames]

    profiles = build_profiles(jsonl)
    students = {k for k, v in profiles.items() if v.get("is_student")}
    print(f"{len(profiles)} identities, {len(students)} reported as students")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise SystemExit(f"Could not open writer for {out}")

    written = 0
    for rec in records:
        cap.set(cv2.CAP_PROP_POS_FRAMES, rec["frame_id"])
        ok, frame = cap.read()
        if not ok:
            continue

        on_screen = 0
        for person in rec["persons"]:
            pid = person.get("person_id")
            profile = profiles.get(pid) if pid is not None else None
            colour = _colour_for(profile)
            x, y, bw, bh = person["bbox"]
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), colour, 2)

            if profile and profile.get("is_student"):
                on_screen += 1
                tag = f"id {pid}"
            elif profile and profile.get("role") == "instructor":
                tag = f"id {pid} INSTRUCTOR"
            elif profile:
                # First word of the reason is enough on-frame; the full text
                # stays in the profile output.
                why = (profile.get("rejected_reason") or "rejected").split(":")[0]
                tag = f"id {pid} [{why}]"
            else:
                tag = "unidentified"

            cv2.putText(frame, tag, (x, max(12, y - 5)), FONT, 0.45, colour, 1)
            for i, (text, tc) in enumerate(_labels_for(person, profile)):
                cv2.putText(frame, text, (x, y + bh + 13 + 13 * i), FONT, 0.4, tc, 1)

            face = person.get("face")
            if face and face.get("bbox"):
                fx, fy, fw_, fh_ = face["bbox"]
                cv2.rectangle(frame, (fx, fy), (fx + fw_, fy + fh_), CYAN, 1)

        cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.putText(
            frame,
            f"t={rec['timestamp_ms'] / 1000:6.1f}s   on screen: {on_screen}"
            f"   roster: {len(students)} students",
            (8, 16), FONT, 0.45, WHITE, 1,
        )
        cv2.putText(
            frame,
            "green=student  amber=unverified  red=rejected  blue=instructor"
            "  cyan=face   '?'=weak/unevidenced",
            (8, 33), FONT, 0.38, (180, 180, 180), 1,
        )

        writer.write(frame)
        written += 1

    writer.release()
    cap.release()
    print(f"wrote {written} frames at {fps} fps -> {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument(
        "--video",
        default="dataset/23-08/vedio/WhatsApp Video 2026-08-23 at 10.16.09.mp4",
    )
    parser.add_argument("--out", default="outputs/annotated.mp4")
    parser.add_argument(
        "--fps", type=float, default=3.0,
        help="Playback rate. The run is sampled at 1 fps, so 3 gives a "
             "watchable pace without dropping any analysed frame.",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    render(Path(args.jsonl), Path(args.video), Path(args.out), args.fps, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
