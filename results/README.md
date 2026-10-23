# results/

The measurement behind every number this project quotes. Nothing here is read
by the running system — these exist so a claim can be checked, and so a claim
that stopped being true gets noticed.

Each file is rewritten by exactly one tool. Re-run that tool and the number
changes; that is the point.

| File | Rewritten by | The number it holds |
|---|---|---|
| `label_ceiling_report.json` | `tools/evaluate/label_ceiling.py` | **κ = 0.261**, 23% label error, **ceiling 0.77**. Read this one first — it bounds everything else |
| `trusted_eval_report.json` | `tools/evaluate/trusted_eval.py` | **0.717 accuracy / 0.708 macro F1** on the 82 twice-confirmed labels |
| `window_model_report.json` | `tools/train/train_window_model.py` | 0.564 against a 0.410 rule baseline, 557 windows, grouped by student |
| `classifier_comparison.json` | `tools/evaluate/compare_classifiers.py` | Five estimators on the same folds — why logistic regression ships |
| `mamba_model_report.json` | `tools/train/train_mamba_scorer.py` | MLP **0.609** vs Mamba **0.592** — the tie that put both on the dashboard |
| `vision_mamba_report.json` | `tools/train/train_vision_mamba.py` | Vision Mamba 0.618 vs ResNet18 0.664 on 12,582 images |
| `engagement_scorer_report.json` | `tools/train/train_engagement_scorer.py` | **0.664 grouped vs 0.968 random** — the near-duplicate leak, caught |
| `score_method_comparison.json` | `tools/data/score_all_images.py` | Calibrated classifier ρ=0.515 beats direct ridge ρ=0.353 |
| `semi_supervised_report.json` | `tools/train/semi_supervised.py` | Self-training **hurt** |
| `label_spreading_check.json` | `tools/diagnose/check_label_spreading.py` | The gain **evaporated** under fold permutation |
| `candidate_features.json` | `tools/diagnose/try_features.py` | 12 literature features tested; 9 came back "noise" |

## Four of these record failures

`semi_supervised_report.json`, `label_spreading_check.json`,
`candidate_features.json` and the Mamba comparison all record something that
did **not** work. They are kept deliberately. "We tried Mamba and it tied" is
only worth saying if there is a file behind it, and a project that keeps only
its wins cannot tell you what it already ruled out.

## Why this is not in `outputs/`

`outputs/` is gitignored, because a pipeline run writes hundreds of megabytes
into it. These were there too, surviving on ten negation rules that had
drifted out of step with reality — `reg_report.json` tracked long after
anything referenced it, `label_ceiling_report.json` and
`trusted_eval_report.json` not tracked at all, though they hold the two
most-quoted numbers in the project and existed on one laptop.
