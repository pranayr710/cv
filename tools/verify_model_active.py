"""Check that the attention score really comes from the model, end to end.

"Did the switch actually happen" is not a question to answer by reading code.
Wiring can be present and still not reach the screen -- that is exactly what
happened here once, when the pipeline computed a model score into the output
files while the browser kept showing a gaze ratio from before the change.

So this runs the real pipeline path over the recorded footage and checks the
chain link by link: the model loads, the annotation stage attaches a score,
every score carries the source that produced it, and the numbers differ from
the rule's answer often enough that one is clearly not standing in for the
other.

    python tools/verify_model_active.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.attention_score import AttentionScorer, annotate_attention
from backend.engagement_model import DEFAULT_MODEL, load_or_none

GRAPH = Path("outputs/final2/live_graph.jsonl")

PASS, FAIL = "  PASS  ", "  FAIL  "


def main() -> int:
    checks: list[tuple[bool, str]] = []

    model = load_or_none()
    checks.append((model is not None,
                   f"a trained model loads from {DEFAULT_MODEL}"))
    if model is not None:
        checks.append((type(model).__name__ in
                       ("EngagementModel", "MLPEngagementModel"),
                       f"model type is {type(model).__name__}"))

    if not GRAPH.exists():
        print(f"{GRAPH} not found -- run the pipeline over footage first.")
        return 1

    scorer = AttentionScorer()
    checks.append((scorer.available, "the scorer reports a model available"))

    rows = [json.loads(line) for line in
            GRAPH.read_text(encoding="utf-8").splitlines() if line.strip()]
    scored = 0
    sources = Counter()
    agree = disagree = comparable = 0
    example = None

    for graph in rows:
        annotate_attention(graph, scorer)
        for node in graph.get("nodes", []):
            features = node.get("features") or {}
            score = features.get("attention_score")
            if score is None:
                continue
            scored += 1
            sources[features.get("attention_source")] += 1
            if example is None and features.get("attention_label"):
                example = (node.get("person_id"), features)
            rule = features.get("engagement")
            learned = features.get("attention_label")
            if rule in ("on", "off") and learned in ("on", "off"):
                comparable += 1
                if rule == learned:
                    agree += 1
                else:
                    disagree += 1

    checks.append((scored > 0, f"{scored} nodes carry an attention score"))
    checks.append((set(sources) == {"model"},
                   f"every score is sourced from the model ({dict(sources)})"))
    checks.append((example is not None,
                   "at least one score carries a verdict and a reason"))
    if example:
        _, features = example
        checks.append((bool(features.get("attention_reason")),
                       "scores explain themselves"))
    # If the two never disagreed, the model would be reproducing the rule and
    # the replacement would be cosmetic.
    summary = (f"model and rule disagree on {disagree} of {comparable} "
               f"comparable windows")
    checks.append((disagree > 0, summary))

    for ok, text in checks:
        print(f"{PASS if ok else FAIL}{text}")

    if example:
        pid, features = example
        print(f"\n  example -- student {pid}")
        print(f"    attention_score  : {features['attention_score']}/10")
        print(f"    attention_label  : {features['attention_label']}")
        print(f"    attention_source : {features['attention_source']}")
        print(f"    rule said        : {features.get('engagement')}")
        print(f"    why              : {features['attention_reason'][:80]}")

    failed = [t for ok, t in checks if not ok]
    print(f"\n  {len(checks) - len(failed)}/{len(checks)} checks passed")
    if failed:
        print("  still wrong:")
        for text in failed:
            print(f"    - {text}")
        return 1
    print("  The attention score is computed by the model.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
