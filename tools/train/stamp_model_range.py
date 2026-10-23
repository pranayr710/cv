"""Record on a trained model the range of features it was fitted on.

Models saved before `lo`/`hi` existed carry weights but no record of what
they were fitted on, so they cannot tell when they are being asked about
footage unlike anything they saw. They extrapolate instead, and because the
last layer is a sigmoid they extrapolate straight to 0 or to 1.

That is not hypothetical. On a one-person webcam session the shipping model
returned probabilities of 0.023, 0.014 and 0.001 -- every window scored
1/10 -- because six of its twenty-one features were outside the fitted range
entirely: there are no peers in a one-person session, and a room focus needs
three students before it can be measured at all.

This reads the training graph, recomputes the per-feature range, and writes
it into the model file. **The weights are not touched**, so the model's
measured accuracy is unchanged -- this only gives it the means to say "I was
not fitted for this".

    python tools/train/stamp_model_range.py --check
    python tools/train/stamp_model_range.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np

from backend.engagement_features import iter_windows
from backend.engagement_model import DEFAULT_MODEL

#: The run the shipping model was fitted on.
TRAINING_GRAPH = Path("outputs/final2/live_graph.jsonl")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--graph", type=Path, default=TRAINING_GRAPH)
    ap.add_argument("--check", action="store_true",
                    help="report what would be written, and change nothing")
    args = ap.parse_args()

    if not args.graph.exists():
        print(f"  {args.graph} missing -- cannot recover the fitted range.")
        return 1

    payload = json.loads(args.model.read_text(encoding="utf-8"))
    names = payload["feature_names"]

    x = np.array([list(w.features) for w in iter_windows(args.graph)])
    if not len(x):
        print(f"  no windows in {args.graph}")
        return 1
    if x.shape[1] != len(names):
        print(f"  {args.graph} yields {x.shape[1]} features, the model "
              f"expects {len(names)} -- the feature set has changed and the "
              "model needs refitting, not stamping.")
        return 1

    lo, hi = x.min(axis=0), x.max(axis=0)
    print(f"  {len(x)} windows from {args.graph}")
    for i, n in enumerate(names):
        print(f"      {n:<30s} {lo[i]:8.3f}  to {hi[i]:8.3f}")

    if args.check:
        print("\n  --check: nothing written.")
        return 0

    payload["lo"] = lo.tolist()
    payload["hi"] = hi.tolist()
    args.model.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"\n  range written to {args.model}; weights untouched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
