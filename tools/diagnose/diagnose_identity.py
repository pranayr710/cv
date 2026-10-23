"""Why does the resolver report more students than the room contains?

The duplicate-box explanation was tested and ruled out: NMS runs at iou=0.50
so two surviving person boxes cannot overlap more than that, and across
130,000 pairs in five recordings the highest overlap seen was 0.501, with
every high-overlap pair differing in size and position like two students
rather than one seen twice.

Two candidates remain, and nothing has measured either. They need different
fixes, so guessing between them is not useful:

**Tracker fragmentation.** One student occluded, or briefly lost, comes back
as a new track id. Each fragment then founds its own cluster. Signature: many
short-lived tracks, and ids whose time spans do not overlap.

**Surrogate keys.** ``keys_for`` mints an id for a detection the tracker never
confirmed. Signature: ids that exist for very few frames and never acquire a
usable face.

Both are visible in a finished run without re-running detection. The third
possibility -- that the clustering simply refuses to merge genuine matches --
is checked by looking at whether distinct ids have similar mean embeddings.

    python tools/diagnose/diagnose_identity.py outputs/final2
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, nargs="?",
                    default=Path("outputs/final2"))
    ap.add_argument("--expected", type=int, default=None,
                    help="how many people were really in the room")
    args = ap.parse_args()

    graph = args.folder / "live_graph.jsonl"
    if not graph.exists():
        print(f"{graph} not found.")
        return 1

    spans: dict[int, list[int]] = defaultdict(list)
    frames_seen: dict[int, int] = defaultdict(int)
    for line in graph.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ts = int(row.get("timestamp_ms") or 0)
        for node in row.get("nodes", []):
            pid = node.get("person_id")
            if pid is None:
                continue
            spans[int(pid)].append(ts)
            frames_seen[int(pid)] += 1

    positive = {p: v for p, v in spans.items() if p > 0}
    negative = {p: v for p, v in spans.items() if p <= 0}
    print(f"{len(positive)} resolved ids, {len(negative)} unresolved\n")

    counts = sorted(frames_seen[p] for p in positive)
    if not counts:
        print("No resolved ids.")
        return 1
    total = sum(counts)
    print("  frames per id")
    print(f"    median {counts[len(counts) // 2]}, "
          f"min {counts[0]}, max {counts[-1]}")
    thin = [p for p in positive if frames_seen[p] < 30]
    print(f"    ids seen in fewer than 30 frames: {len(thin)} of "
          f"{len(positive)}  ({len(thin) / len(positive):.0%})")
    print(f"    they account for "
          f"{sum(frames_seen[p] for p in thin) / total:.1%} of all observations")

    # Fragmentation: ids whose lifetimes do not overlap are candidates for
    # being the same person split in two. Ids that co-occur cannot be.
    ranges = {p: (min(v), max(v)) for p, v in positive.items()}
    ids = sorted(positive)
    disjoint = 0
    pairs = 0
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            pairs += 1
            if ranges[a][1] < ranges[b][0] or ranges[b][1] < ranges[a][0]:
                disjoint += 1
    print(f"\n  id pairs whose lifetimes never overlap: {disjoint} of {pairs}"
          f"  ({disjoint / pairs:.0%})" if pairs else "")
    print("    these pairs COULD be one student split by a tracking gap;")
    print("    overlapping pairs could not be, whatever their faces look like.")

    # Sequential appearance is the fingerprint of fragmentation: an id ending
    # as another begins.
    starts = sorted((ranges[p][0], p) for p in positive)
    handoffs = 0
    for (_t1, a), (t2, _b) in zip(starts, starts[1:]):
        if abs(ranges[a][1] - t2) < 2000:
            handoffs += 1
    print(f"\n  ids beginning within 2s of another ending: {handoffs}")
    print("    a tracking gap looks exactly like this; genuine arrivals do not")
    print("    cluster against someone else's departure.")

    if args.expected:
        over = len(positive) - args.expected
        print(f"\n  over-count: {len(positive)} ids for {args.expected} "
              f"people  (+{over})")
        print(f"  thin ids alone would account for {min(len(thin), over)} "
              "of that")
        if len(thin) >= over:
            print("\n  A founding floor on frames seen would close the gap")
            print("  without touching the clustering. That is a different fix")
            print("  from merging fragments, and a cheaper one.")
        else:
            print("\n  Thin ids cannot explain it alone; fragments of"
                  "\n  well-observed students are being kept apart.")
    else:
        print("\n  Pass --expected N to compare against the real headcount.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
