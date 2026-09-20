"""Close the improvement loop: find what to label next, then retrain.

The attention score comes from a model fitted to hand-labelled windows, so it
improves exactly when it gets more labels -- and not all labels are worth the
same. A window the model already scores 0.95 teaches it almost nothing; a
window it scores 0.51 is one it cannot currently tell apart, and a human
judgement there moves the boundary. This picks the second kind.

There is deliberately no self-training step. Refitting on the model's own
predictions would sharpen whatever it already believed, wrong parts included,
and return the resulting confidence as though it were progress. New information
enters through a person or not at all.

    python tools/improve_model.py --status      # how the model stands
    python tools/improve_model.py --next 60     # queue the next windows
    python tools/improve_model.py --retrain     # refit on everything labelled

The usual cycle is: --next, label them with labeling/label_gui.py, --retrain,
and repeat. Each round the queue is drawn from wherever the current model is
least sure, so it stops asking about cases it has already learned.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.attention_score import UNCERTAIN_BAND, AttentionScorer
from backend.engagement_model import DEFAULT_MODEL, load_or_none

GRAPH = Path("outputs/final2/live_graph.jsonl")
LABELS = Path("labeling/labels.json")
QUEUE = Path("labeling/to_label_next.json")
REPORT = Path("outputs/window_model_report.json")


def status() -> int:
    """Where the model stands and what it rests on."""
    model = load_or_none()
    print(f"model file      : {DEFAULT_MODEL}")
    print(f"model loaded    : {'yes' if model else 'NO -- rules are in use'}")

    if LABELS.exists():
        store = json.loads(LABELS.read_text(encoding="utf-8"))
        counts = Counter(v["label"] for v in store.values())
        usable = counts["on"] + counts["off"]
        students = len({v["person_id"] for v in store.values()
                        if v["label"] in ("on", "off")})
        print(f"labels          : {usable} usable "
              f"({counts['on']} on, {counts['off']} off, "
              f"{counts['unclear']} unclear) across {students} students")
    else:
        print("labels          : none yet")

    if REPORT.exists():
        r = json.loads(REPORT.read_text(encoding="utf-8"))
        rule = r.get("rule_baseline", {}).get("accuracy")
        learned = r.get("learned_model", {}).get("accuracy")
        print(f"last evaluation : model {learned}  vs  rule {rule}")
        print(f"                  ({r.get('evaluation')})")
    else:
        print("last evaluation : none yet")
    return 0


def queue_next(count: int) -> int:
    """Write the windows the model is least sure about."""
    scorer = AttentionScorer()
    if not scorer.available:
        print("no trained model, so nothing can be ranked by uncertainty.")
        print("Label a first batch with labeling/label_gui.py, then --retrain.")
        return 1
    if not GRAPH.exists():
        print(f"{GRAPH} not found -- run the pipeline over footage first.")
        return 1

    done = set()
    if LABELS.exists():
        done = set(json.loads(LABELS.read_text(encoding="utf-8")))

    for line in GRAPH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        g = json.loads(line)
        for node in g.get("nodes", []):
            pid = node.get("person_id")
            if pid and pid > 0:
                scorer.update(int(pid), int(g.get("timestamp_ms") or 0),
                              node.get("features") or {})

    # Skip anything already judged: asking again buys nothing.
    fresh = [w for w in scorer.uncertain_windows(limit=count * 4)
             if f"p{w['person_id']}_{w['start_ms']}" not in done][:count]
    QUEUE.parent.mkdir(parents=True, exist_ok=True)
    QUEUE.write_text(json.dumps(fresh, indent=1), encoding="utf-8")

    print(f"{len(fresh)} windows queued to {QUEUE}")
    if fresh:
        near = sum(1 for w in fresh if abs(w["probability"] - 0.5) < 0.05)
        print(f"  {near} of them sit within 0.05 of the decision boundary")
        print(f"  (the band searched is {UNCERTAIN_BAND})")
    print("\nLabel them with:  python labeling/label_gui.py")
    print("Then:             python tools/improve_model.py --retrain")
    return 0


def retrain() -> int:
    """Refit on every label gathered so far and report the movement."""
    before = None
    if REPORT.exists():
        before = json.loads(REPORT.read_text(encoding="utf-8"))

    print("retraining on all labels...\n")
    result = subprocess.run(
        [sys.executable, "tools/train_window_model.py"],
        capture_output=True, text=True, check=False)
    print(result.stdout.strip() or result.stderr.strip())
    if result.returncode != 0:
        return result.returncode

    if before and REPORT.exists():
        after = json.loads(REPORT.read_text(encoding="utf-8"))
        old = before.get("learned_model", {}).get("accuracy")
        new = after.get("learned_model", {}).get("accuracy")
        old_n, new_n = before.get("n_labels"), after.get("n_labels")
        if old is not None and new is not None:
            print(f"\n  accuracy {old:.3f} -> {new:.3f} "
                  f"({new - old:+.3f}) on {old_n} -> {new_n} labels")
            if new <= old:
                print("  No improvement. More labels do not always help: if "
                      "this repeats,\n  the limit is more likely the features "
                      "than the label count.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--status", action="store_true",
                    help="report where the model stands")
    ap.add_argument("--next", type=int, metavar="N", default=None,
                    help="queue the N windows most worth labelling")
    ap.add_argument("--retrain", action="store_true",
                    help="refit on every label gathered so far")
    args = ap.parse_args()

    if args.next is not None:
        return queue_next(args.next)
    if args.retrain:
        return retrain()
    return status()


if __name__ == "__main__":
    raise SystemExit(main())
