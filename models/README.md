# models/

What the running system loads. Both files are tracked, both are kilobytes,
and both are the actual deliverable: without them the pipeline silently falls
back to the rule-based verdict, which measured **0.447** accuracy against the
model's **0.717**.

| File | Loaded by | What it is |
|---|---|---|
| `window_engagement_model.json` | `backend/engagement_model.py` | The shipping attention model — an MLP over 21 aggregate window features |
| `mamba_attention_model.pt` | `backend/mamba_score.py` | The Mamba sequence scorer, 2,897 parameters, run beside the above so both scores reach the dashboard |

Retrain with `tools/train/train_window_model.py` and
`tools/train/train_mamba_scorer.py`. Both need `outputs/final2/live_graph.jsonl`
and `labeling/labels.json`.

## Why these are not in `outputs/`

They were, and `.gitignore` carried a negation rule for each so they would
survive the blanket `outputs/` ignore. Two problems followed from that.

The rules drifted: `reg_report.json` stayed tracked long after anything
referenced it, while `label_ceiling_report.json` — which holds the κ = 0.261
and the 0.77 ceiling every accuracy in this project is quoted against — was
never tracked at all.

And `outputs/` is a Docker `VOLUME`. The image baked these two files into it
and the documented `-v ./outputs:/app/outputs` mount then hid them, so a
deployed container fell back to the rule verdict with no warning. A directory
cannot be both disposable and a deliverable.

`outputs/` is now entirely what a run produces, and ignored with no
exceptions. The measurement record lives in [`../results/`](../results/).
