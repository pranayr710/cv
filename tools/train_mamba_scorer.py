"""Train the Mamba attention scorer, and measure it against what already ships.

The number this produces is only meaningful next to a baseline, so the same
labels, the same folds and the same group-aware split are used for three
predictors: the precedence rule, the aggregate-feature MLP that currently
ships, and the Mamba sequence model. Splitting is by student because windows
overlap by 50% -- a random split puts near-identical windows of one person on
both sides and inflates every number.

The comparison is deliberately unfair in the *baseline's* favour in one way:
the Mamba sees exactly the unaggregated form of the features the MLP gets, so
a win cannot come from better inputs. It can only come from reading order.

    python tools/train_mamba_scorer.py
    python tools/train_mamba_scorer.py --epochs 200 --seed 1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.engagement_sequence import (
    FRAME_FEATURE_NAMES,
    SEQ_LEN,
    sequences_from_graph,
)
from backend.mamba_model import MambaAttentionScorer, MambaConfig

LABELS = Path("labeling/labels.json")
GRAPH = Path("outputs/final2/live_graph.jsonl")
MODEL_OUT = Path("outputs/mamba_attention_model.pt")
REPORT = Path("outputs/mamba_model_report.json")


def load() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """Sequences, aggregate features, target, student id and rule verdict."""
    store = json.loads(LABELS.read_text(encoding="utf-8"))
    seqs = sequences_from_graph(GRAPH)

    X, A, y, g, rule = [], [], [], [], []
    missing = 0
    for entry in store.values():
        if entry["label"] not in ("on", "off"):
            continue
        key = (entry["person_id"], entry["start_ms"])
        if key not in seqs:
            missing += 1
            continue
        X.append(seqs[key].steps)
        A.append(entry["features"])
        y.append(1 if entry["label"] == "on" else 0)
        g.append(entry["person_id"])
        rule.append(entry.get("rule_verdict"))
    if missing:
        print(f"  note: {missing} labels had no sequence and were skipped")
    return (np.array(X, dtype=np.float32), np.array(A, dtype=float),
            np.array(y), np.array(g), rule)


def train_fold(x: np.ndarray, y: np.ndarray, epochs: int, seed: int,
               device: str) -> MambaAttentionScorer:
    """Fit one fold. Class weighting keeps an imbalanced fold from collapsing
    onto the majority answer, which at this sample size it otherwise does."""
    torch.manual_seed(seed)
    model = MambaAttentionScorer(
        MambaConfig(n_features=len(FRAME_FEATURE_NAMES))).to(device)
    pos = float(y.sum())
    weight = torch.tensor(
        [(len(y) - pos) / pos if pos else 1.0], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)

    xt = torch.tensor(x, device=device)
    yt = torch.tensor(y, dtype=torch.float32, device=device)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(xt), yt)
        loss.backward()
        # Long recurrences are prone to exploding gradients; this is cheap
        # insurance rather than a response to an observed problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model


def accuracy(pred: list, truth: np.ndarray) -> dict:
    """Accuracy over decided windows, with abstentions reported separately."""
    decided = [(p, t) for p, t in zip(pred, truth) if p is not None]
    if not decided:
        return {"accuracy": float("nan"), "decided": 0, "abstained": len(pred)}
    correct = sum(1 for p, t in decided if (p == "on") == bool(t))
    return {"accuracy": round(correct / len(decided), 4),
            "decided": len(decided), "abstained": len(pred) - len(decided)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    from sklearn.model_selection import GroupKFold
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, agg, y, g, rule = load()
    print(f"{len(y)} labelled windows ({y.sum()} on, {len(y) - y.sum()} off) "
          f"across {len(set(g))} students")
    print(f"sequences: {SEQ_LEN} steps x {len(FRAME_FEATURE_NAMES)} features "
          f"| device: {device}")
    n_params = MambaAttentionScorer(
        MambaConfig(n_features=len(FRAME_FEATURE_NAMES))).n_params
    print(f"Mamba parameters: {n_params:,}\n")

    oof_mamba: list = [None] * len(y)
    oof_mlp: list = [None] * len(y)
    for k, (tr, te) in enumerate(
            GroupKFold(n_splits=args.folds).split(x, y, g), 1):
        model = train_fold(x[tr], y[tr], args.epochs, args.seed, device)
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(
                model(torch.tensor(x[te], device=device))).cpu().numpy()
        for i, p in zip(te, probs):
            oof_mamba[i] = "on" if p >= 0.5 else "off"

        scaler = StandardScaler().fit(agg[tr])
        mlp = MLPClassifier((16,), max_iter=3000, random_state=args.seed)
        mlp.fit(scaler.transform(agg[tr]), y[tr])
        for i, p in zip(te, mlp.predict(scaler.transform(agg[te]))):
            oof_mlp[i] = "on" if p else "off"
        print(f"  fold {k}/{args.folds} done ({len(te)} held out)")

    res = {
        "rule (precedence)": accuracy(rule, y),
        "MLP on aggregates (shipping)": accuracy(oof_mlp, y),
        "Mamba on sequences": accuracy(oof_mamba, y),
    }
    print(f"\n  out-of-fold over {args.folds} folds, split by student\n")
    print(f"  {'':<30}{'accuracy':>10}{'decided':>9}{'abstained':>11}")
    for name, r in res.items():
        print(f"  {name:<30}{r['accuracy']:>10.3f}{r['decided']:>9}"
              f"{r['abstained']:>11}")

    best = max(res, key=lambda k: res[k]["accuracy"])
    print(f"\n  best on this evidence: {best}")
    if best != "Mamba on sequences":
        print("  The sequence model did not beat the aggregate model. On this"
              "\n  many labels that is the expected result, and it is reported"
              "\n  rather than tuned away.")

    torch.save({"state_dict": train_fold(x, y, args.epochs, args.seed,
                                         device).state_dict(),
                "config": {"n_features": len(FRAME_FEATURE_NAMES)},
                "feature_names": list(FRAME_FEATURE_NAMES),
                "seq_len": SEQ_LEN}, MODEL_OUT)
    print(f"\n  final model -> {MODEL_OUT}")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps({
        "n_labels": len(y), "n_students": len(set(g)),
        "evaluation": "out-of-fold, grouped by student",
        "mamba_parameters": n_params,
        "seq_len": SEQ_LEN, "n_frame_features": len(FRAME_FEATURE_NAMES),
        "epochs": args.epochs, "results": res, "best": best,
    }, indent=1), encoding="utf-8")
    print(f"  report      -> {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
