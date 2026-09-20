"""Re-derive the rule's verdict on hand-labelled windows, and score it.

Changing an action rule is cheap; knowing whether it helped is not, because the
obvious way to find out is to re-run the pipeline over the footage. This takes
the shorter path: per-frame records in raw.jsonl already hold every input the
action rules consume, so the rules can be re-applied to them directly and the
result compared against the same human labels the model was judged on.

That makes a rule change measurable in seconds against real footage, which is
the difference between a fix and a hope.

    python tools/rescore_rule.py
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import CONFIG
from backend.engagement import classify_engagement
from backend.scene_graph import _phone_overlaps

RAW = Path("outputs/final2/raw.jsonl")
LABELS = Path("labeling/labels.json")


def verdict_for_window(frames: list[tuple[dict, list]]) -> str | None:
    """Majority on/off over a window, re-deriving the engagement verdict.

    This reproduces the path scene_graph takes, not the action rules. The
    labelled windows carry the ENGAGEMENT verdict, and engagement is decided by
    backend.engagement.classify_engagement from gaze, behaviour, a phone check
    and eye closure -- a different function from the one that assigns actions.

    Getting that wrong the first time produced a rule that never answered
    "off", because the off-task routes it depends on were never given their
    inputs. A measurement harness that does not reproduce the thing it measures
    reports on itself.
    """
    verdicts = []
    for person, objects in frames:
        face = person.get("face") or {}
        eyes_closed = None
        if face.get("ear") is not None:
            eyes_closed = face["ear"] < CONFIG.face.ear_closed_threshold
        verdict = classify_engagement(
            (person.get("head_pose") or {}).get("gaze_label"),
            (person.get("behaviour") or {}).get("label"),
            CONFIG.engagement,
            phone_nearby=_phone_overlaps(person["bbox"], objects, CONFIG,
                                         person.get("posture")),
            eyes_closed=eyes_closed,
        )
        if verdict:
            verdicts.append(verdict)
    if not verdicts:
        return None
    return Counter(verdicts).most_common(1)[0][0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=RAW)
    ap.add_argument("--labels", type=Path, default=LABELS)
    args = ap.parse_args()

    if not args.labels.exists():
        print(f"{args.labels} not found -- nothing to score against.")
        return 1

    labels = {k: v for k, v in
              json.loads(args.labels.read_text(encoding="utf-8")).items()
              if v["label"] in ("on", "off")}
    print(f"{len(labels)} labelled windows\n  reading {args.raw}...")

    # frame_id -> timestamp, and (frame_id, person_id) -> (person, objects)
    per_frame: dict[tuple[int, int], tuple[dict, list]] = {}
    frame_time: dict[int, int] = {}
    for line in args.raw.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        fid = int(rec.get("frame_id", -1))
        frame_time[fid] = int(rec.get("timestamp_ms") or 0)
        objects = rec.get("objects") or []
        for person in rec.get("persons", []):
            pid = person.get("person_id")
            if pid and pid > 0 and person.get("bbox"):
                per_frame[(fid, int(pid))] = (person, objects)

    fresh, stored, truth = [], [], []
    for entry in labels.values():
        pid = entry["person_id"]
        frames = [v for (fid, p), v in per_frame.items()
                  if p == pid
                  and entry["start_ms"] <= frame_time.get(fid, -1) < entry["end_ms"]]
        if not frames:
            continue
        fresh.append(verdict_for_window(frames))
        stored.append(entry.get("rule_verdict"))
        truth.append(entry["label"])

    def score(pred):
        decided = [(p, t) for p, t in zip(pred, truth) if p is not None]
        if not decided:
            return 0.0, 0, len(pred)
        correct = sum(1 for p, t in decided if p == t)
        return correct / len(decided), len(decided), len(pred) - len(decided)

    old_acc, old_n, old_abs = score(stored)
    new_acc, new_n, new_abs = score(fresh)

    print(f"  matched {len(truth)} windows to frames\n")
    print(f"  {'':<22} {'accuracy':>9} {'decided':>9} {'abstained':>10}")
    print(f"  {'rule, as recorded':<22} {old_acc:>9.3f} {old_n:>9} {old_abs:>10}")
    print(f"  {'rule, re-derived now':<22} {new_acc:>9.3f} {new_n:>9} {new_abs:>10}")

    off_old = sum(1 for p in stored if p == "off")
    off_new = sum(1 for p in fresh if p == "off")
    human_off = sum(1 for t in truth if t == "off")
    print(f"\n  says 'off':  was {off_old}, now {off_new}, "
          f"human says {human_off} of {len(truth)}")
    delta = new_acc - old_acc
    print(f"  change: {delta:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
