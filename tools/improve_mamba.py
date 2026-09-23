"""The self-improving loop for the Mamba scorer: ask, label, refit, repeat.

"Self-improving" here means the model chooses what it needs to be taught. Each
round it scores every window it has never been told about, ranks them by how
uncertain it is, and queues the ones nearest its own decision boundary. A
window it scores 0.97 confirms what it already believes; a window it scores
0.51 is one it genuinely cannot separate, and a human judgement there moves
the boundary rather than reinforcing it.

What this deliberately is **not** is self-training. Refitting on the model's
own predicted labels would compound its existing mistakes and hand back a
sharper version of whatever it already thought, with higher confidence and no
new information. Confidence that rises without new evidence is a bug that
looks like progress. New information enters through a person or not at all.

    python tools/improve_mamba.py --status     # where the model stands
    python tools/improve_mamba.py --next 60    # queue what to label
    python tools/improve_mamba.py --retrain    # refit on everything labelled

The cycle: --next, label with labeling/label_gui.py, --retrain, repeat.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.engagement_sequence import (
    FRAME_FEATURE_NAMES,
    sequences_from_graph,
)
from backend.mamba_model import MambaAttentionScorer, MambaConfig

LABELS = Path("labeling/labels.json")
GRAPH = Path("outputs/final2/live_graph.jsonl")
MODEL = Path("outputs/mamba_attention_model.pt")
QUEUE = Path("labeling/to_label_next_mamba.json")
REPORT = Path("outputs/mamba_model_report.json")


def load_model() -> MambaAttentionScorer | None:
    """The trained scorer, or None if it has never been fitted."""
    if not MODEL.exists():
        return None
    blob = torch.load(MODEL, map_location="cpu", weights_only=False)
    names = blob.get("feature_names")
    if names and list(names) != list(FRAME_FEATURE_NAMES):
        print("  the saved model was fitted to a different frame feature set;"
              "\n  retrain before trusting it.")
        return None
    model = MambaAttentionScorer(
        MambaConfig(n_features=len(FRAME_FEATURE_NAMES)))
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model


def uncertainty(model: MambaAttentionScorer) -> list[tuple[float, tuple, float]]:
    """Unlabelled windows ranked by how close the model is to undecided."""
    labelled = set()
    if LABELS.exists():
        for entry in json.loads(LABELS.read_text(encoding="utf-8")).values():
            labelled.add((entry["person_id"], entry["start_ms"]))

    seqs = sequences_from_graph(GRAPH)
    todo = {k: v for k, v in seqs.items() if k not in labelled}
    if not todo:
        return []

    keys = list(todo)
    batch = torch.tensor([todo[k].steps for k in keys], dtype=torch.float32)
    with torch.no_grad():
        probs = torch.sigmoid(model(batch)).numpy()
    # Distance from the boundary: 0 is maximally uncertain.
    return sorted(((abs(float(p) - 0.5), k, float(p))
                   for k, p in zip(keys, probs)), key=lambda r: r[0])


def status() -> int:
    n_lab = 0
    if LABELS.exists():
        store = json.loads(LABELS.read_text(encoding="utf-8"))
        n_lab = sum(1 for e in store.values() if e["label"] in ("on", "off"))
    print(f"  usable labels : {n_lab}")
    if REPORT.exists():
        r = json.loads(REPORT.read_text(encoding="utf-8"))
        print(f"  parameters    : {r.get('mamba_parameters'):,}")
        for name, res in (r.get("results") or {}).items():
            print(f"    {name:<32}{res['accuracy']:.3f}")
        print(f"  best          : {r.get('best')}")
    model = load_model()
    if model is None:
        print("  no trained model yet -- run tools/train_mamba_scorer.py")
        return 1
    ranked = uncertainty(model)
    print(f"  unlabelled    : {len(ranked)} windows available to label")
    if ranked:
        print(f"  most uncertain: p={ranked[0][2]:.3f} "
              f"(person {ranked[0][1][0]}, {ranked[0][1][1]}ms)")
    return 0


def queue(n: int) -> int:
    model = load_model()
    if model is None:
        print("  train the model first: tools/train_mamba_scorer.py")
        return 1
    ranked = uncertainty(model)
    if not ranked:
        print("  every window is already labelled -- nothing left to ask.")
        return 0
    picked = ranked[:n]
    QUEUE.write_text(json.dumps(
        [{"person_id": k[0], "start_ms": k[1], "probability": round(p, 4),
          "margin": round(m, 4)} for m, k, p in picked], indent=1),
        encoding="utf-8")
    lo = min(p for _, _, p in picked)
    hi = max(p for _, _, p in picked)
    print(f"  queued {len(picked)} windows -> {QUEUE}")
    print(f"  their probabilities span {lo:.3f}-{hi:.3f}; the model is nearest"
          "\n  undecided here, so a human answer is worth most.")
    print("\n  label them:  cd labeling && python label_gui.py")
    print("  then:        python tools/improve_mamba.py --retrain")
    return 0


def retrain() -> int:
    print("  refitting on every label collected so far...\n")
    return subprocess.call(
        [sys.executable, "tools/train_mamba_scorer.py"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--status", action="store_true",
                    help="how the model stands and what is left to label")
    ap.add_argument("--next", type=int, metavar="N", default=None,
                    help="queue the N windows the model is least sure about")
    ap.add_argument("--retrain", action="store_true",
                    help="refit on everything labelled and re-measure")
    args = ap.parse_args()

    if args.retrain:
        return retrain()
    if args.next is not None:
        return queue(args.next)
    return status()


if __name__ == "__main__":
    raise SystemExit(main())
