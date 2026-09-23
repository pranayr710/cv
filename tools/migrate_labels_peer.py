"""Recompute stored label features after the peer columns were added.

A label records the feature vector as it stood when the judgement was made, so
that training can check the weights are being attached to the columns they
were fitted for. Adding peer context changes the vector from 15 values to 18,
which makes every stored vector stale -- and the trainer refuses to run rather
than silently mismatch them.

The human judgement itself is unaffected: a rater watched a video and said on
or off, and that answer does not depend on how the pipeline summarises the
same span. So the labels are kept and only the vectors are recomputed, from
the same graph and the same window boundaries they were originally derived
from.

The previous file is copied to labels.json.bak first. This was needed once
before, when n_frames was removed, and the backup is what made that safe.

    python tools/migrate_labels_peer.py --dry-run
    python tools/migrate_labels_peer.py
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.engagement_features import FEATURE_NAMES, iter_windows

LABELS = Path("labeling/labels.json")
GRAPH = Path("outputs/final2/live_graph.jsonl")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change and write nothing")
    args = ap.parse_args()

    store = json.loads(LABELS.read_text(encoding="utf-8"))
    fresh = {(w.person_id, w.start_ms): w for w in iter_windows(GRAPH)}
    print(f"{len(store)} labels, {len(fresh)} windows recomputed "
          f"with {len(FEATURE_NAMES)} features")

    updated = missing = unchanged = 0
    for entry in store.values():
        key = (entry["person_id"], entry["start_ms"])
        window = fresh.get(key)
        if window is None:
            missing += 1
            continue
        if list(entry.get("feature_names") or []) == list(FEATURE_NAMES):
            unchanged += 1
            continue
        entry["features"] = list(window.features)
        entry["feature_names"] = list(FEATURE_NAMES)
        updated += 1

    print(f"  updated   : {updated}")
    print(f"  unchanged : {unchanged}")
    print(f"  no window : {missing}"
          + ("  (these keep their old vector and will be skipped in training)"
             if missing else ""))

    if args.dry_run:
        print("\n  dry run -- nothing written.")
        return 0

    shutil.copy2(LABELS, LABELS.with_suffix(".json.bak"))
    LABELS.write_text(json.dumps(store, indent=1), encoding="utf-8")
    print(f"\n  backup  -> {LABELS.with_suffix('.json.bak')}")
    print(f"  written -> {LABELS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
