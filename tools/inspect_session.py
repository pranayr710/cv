"""What did a session actually see, and what did it conclude from it?

When a result looks wrong -- a phone flagged with no phone present -- the
question is which stage was wrong, and the report does not say. The detector
may have hallucinated an object, or detected nothing and the action rules
guessed, or both been right and the display misleading.

This separates them. It prints what the detector proposed and at what
confidence, what the action layer concluded, and which evidence it cited, so a
wrong answer can be traced to the stage that produced it rather than argued
about.

    python tools/inspect_session.py                     # the last live session
    python tools/inspect_session.py outputs/final2
    python tools/inspect_session.py --student 3
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line
            in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, nargs="?",
                    default=Path("outputs/live"),
                    help="a session output folder (default: outputs/live)")
    ap.add_argument("--student", type=int, default=None,
                    help="restrict the action summary to one person_id")
    args = ap.parse_args()

    raw_path = args.folder / "live_frames.jsonl"
    if not raw_path.exists():
        raw_path = args.folder / "raw.jsonl"
    graph = _read(args.folder / "live_graph.jsonl")
    raw = _read(raw_path)

    if not graph and not raw:
        print(f"No session data in {args.folder}.")
        print("Run a session first, or point this at another folder.")
        return 1

    print(f"{args.folder}")
    print(f"  {len(raw)} detector frames, {len(graph)} graph frames\n")

    # --- stage 1: what the detector proposed --------------------------------
    confs: dict[str, list[float]] = defaultdict(list)
    for row in raw:
        for obj in row.get("objects", []):
            confs[obj.get("cls")].append(float(obj.get("confidence") or 0))

    print("  WHAT THE DETECTOR SAW")
    if not confs:
        print("    no objects at all this session")
    else:
        print(f"    {'class':<14}{'frames':>8}{'min':>7}{'median':>8}{'max':>7}")
        for cls, values in sorted(confs.items(), key=lambda kv: -len(kv[1])):
            values.sort()
            print(f"    {str(cls):<14}{len(values):>8}{values[0]:>7.2f}"
                  f"{values[len(values) // 2]:>8.2f}{values[-1]:>7.2f}")
        print("\n    A low median here is the detector guessing. Anything it")
        print("    named is in the whitelist; everything else it saw was")
        print("    discarded before this point.")

    # --- stage 2: what the action layer concluded ---------------------------
    actions: Counter = Counter()
    evidence: Counter = Counter()
    people: set[int] = set()
    for row in graph:
        for node in row.get("nodes", []):
            pid = node.get("person_id")
            if pid is None:
                continue
            if args.student is not None and pid != args.student:
                continue
            people.add(pid)
            feat = node.get("features") or {}
            actions[feat.get("action")] += 1
            action = feat.get("action")
            if action in ("on_phone", "on_laptop", "eating", "drinking",
                          "reading", "writing", "typing"):
                evidence[(action, feat.get("action_evidence"),
                          feat.get("action_confidence"))] += 1

    who = (f"student {args.student}" if args.student is not None
           else f"{len(people)} student(s)")
    print(f"\n  WHAT IT CONCLUDED -- {who}")
    if not actions:
        print("    nothing; no student was tracked in these frames")
    else:
        total = sum(actions.values())
        for action, count in actions.most_common():
            bar = "#" * int(30 * count / total)
            print(f"    {str(action):<18}{count:>6}  {count / total:>5.1%}  {bar}")

    if evidence:
        print("\n  WHY -- the evidence each object-driven action cited")
        print(f"    {'action':<12}{'evidence':<46}{'how':<10}{'frames':>7}")
        for (action, why, how), count in evidence.most_common():
            print(f"    {action:<12}{str(why)[:45]:<46}{str(how):<10}{count:>7}")
        print("\n    'direct' means a model detected it. 'inferred' means a")
        print("    rule concluded it from geometry, which is weaker evidence")
        print("    and is labelled so you can discount it.")

    # --- the cross-check that usually answers the question -------------------
    print("\n  CROSS-CHECK")
    phone_frames = sum(n for (a, _w, _h), n in evidence.items()
                       if a == "on_phone")
    phone_seen = len(confs.get("cell phone", []))
    print(f"    frames the detector named a phone : {phone_seen}")
    print(f"    frames a student was 'on phone'   : {phone_frames}")
    if phone_frames and not phone_seen:
        print("    -> concluded without a detection: a rule inferred it.")
    elif phone_seen and confs.get("cell phone"):
        low = sorted(confs["cell phone"])[len(confs["cell phone"]) // 2]
        if low < 0.4:
            print(f"    -> detected, but weakly (median {low:.2f}). A phone")
            print("       below about 0.4 is usually the detector guessing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
