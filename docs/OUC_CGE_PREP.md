# OUC-CGE: Acquisition & Preparation Status

*Owner: Person C (month-3 tasks 1–2). Companion tooling:
`tools/prepare_ouccge.py` (this doc's commands), evaluation via
`tools/eval_group_activity.py`.*

## What OUC-CGE is and why it is the right external check

OUC-CGE is a public classroom group-engagement dataset: ~12h50m of real
classroom video, 17 participants, clip-level engagement labels on a
three-level ordinal scale (**high / medium / low**), openly archived on OSF
under a permanent DOI, with an MIT-licensed codebase. It matches our target
task almost exactly — which is exactly why it is used for **evaluation only,
never training**: training on it would make our headline number a
self-fulfilling artifact instead of an external check.

## Current status

| Step | Status |
|---|---|
| Dataset identified, licence + permanence verified | ✅ |
| Download executed | ❌ **pending** (~13h of video; not fetched during this work session) |
| `labels.csv` obtained from archive metadata | ❌ pending download |
| Preparation script (`tools/prepare_ouccge.py`) | ✅ written against published layout; **layout assumption flagged in its docstring — run `--verify-only` first** |
| Split definition (by source video/clip) | ✅ implemented + unit-tested; enforced again at eval time |
| Evaluation harness | ✅ `tools/eval_group_activity.py`, unit-tested |
| Any measured number | ❌ **none exists yet** — none may be quoted until rows above close |

## Runbook

```bash
# 1) download from OSF (permanent DOI) into data/OUC-CGE/
#    (osfclient or browser; keep the DOI in the final report's references)

# 2) place/verify the label sheet, then integrity-check BEFORE anything else:
python -m tools.prepare_ouccge --root data/OUC-CGE --verify-only

# 3) build splits + manifest:
python -m tools.prepare_ouccge --root data/OUC-CGE --out data/OUC-CGE/prepared

# 4) later, once backend/group_activity.py has a trained model_fn:
python -m tools.eval_group_activity \
    --manifest data/OUC-CGE/prepared/manifest.csv \
    --predictions outputs/group_predictions.jsonl \
    --split test --abstain exclude --human-agreement <kappa from BOSS study>
```

## The split rule (the part that must never regress)

Splits are assigned **by source recording**, whole sources to one fold each,
greedy-stratified by label, fully deterministic (no RNG). Rationale: frames
inside one classroom recording are near-duplicates; any frame- or clip-level
split leaks them across folds and inflates accuracy that will not survive
contact with the real deployment classroom. Both tools re-verify this rule
independently — the manifest builder by construction, the evaluator by an
explicit leakage abort — because "we were careful" is not a control, checks
are.

With only ~17 underlying participants, expect small *source*-level folds even
with thousands of clips: the evaluator therefore prints per-class support and
a Wilson CI, and metrics are reported next to human inter-rater agreement
(BOSS protocol output, docs/BOSS_VALIDATION.md), never alone.
