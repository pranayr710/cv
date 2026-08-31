"""Group-level activity recognition for ClassGraph (Stage 5).

Status: SCAFFOLD, NOT A RESULT. Everything in this module is engineering
plumbing built ahead of its inputs. The scene graph this consumes does not
exist yet (Person B's Stage 3 work), and no trained weight for any layer below
exists either. Nothing here has produced a measured number on real footage,
and none of its outputs may be quoted anywhere until they have. The honest
deliverable today is: a defined input contract, an abstention-first readout,
and a test suite proving the plumbing does what the docstrings say.

Theoretical basis — ARG used directly, with documented deviations.

The architecture follows Actor Relation Graphs for Group Activity Recognition
(Wang et al., AAAI 2020): build a graph whose nodes are people, connect them
with typed relations (appearance similarity + spatial position), let each
relation type carry its own learned weight, run message passing (a GCN over
the relation-weighted adjacency), then pool node features into a whole-graph
prediction. We take that structure directly rather than inventing a variant,
with three deviations, all forced by our problem being different:

1. Class level, not activities. ARG predicts what a group is doing
   (crossing arms, watching). Our target is the OUC-CGE-style ordinal group
   engagement level: high / medium / low.
2. Three ordered classes, tiny N. OUC-CGE has 17 participants and ~3k labelled
   clips. That rules out deep stacks; the shipped readout is a 2-layer GCN at
   most, and the primary reported metric will sit next to human inter-rater
   agreement, never alone (see ``tools/eval_group_activity.py``).
3. Abstention is part of the contract. ARG assumes every person is detected
   every frame. Classroom footage does not cooperate: occlusion and detector
   gaps are normal, and emitting a confident "high engagement" from two
   visible students out of ten would be the exact failure mode this project
   exists to avoid. Below ``min_students`` nodes or above
   ``max_unknown_rate`` missing features, :func:`classify_group` returns
   ``None`` with a reason instead of a guess.

Input contract (PROVISIONAL until B freezes graph_schema):

    nodes = [
        {
            "track_id": int,            # stable per-student track id
            "role": str,                # "student" | "instructor"
            "center": [x, y],           # image coords, any scale
            "bbox_wh": [w, h],
            "features": [float, ...],   # per-node descriptor, e.g. one-hot
                                        # behaviour verdicts + gaze verdicts +
                                        # behavioural proxy stats. Missing
                                        # features -> pass floats, mark unknown
                                        # via "feature_mask".
            "appearance": [float, ...], # embedding for similarity edges;
                                        # optional, falls back to position
        },
        ...
    ]

Usage::

    g = GroupActivityBuilder(GroupActivityConfig())
    graph = g.build_graph(frame_nodes)
    result = classify_group(graph, model_fn=my_trained_gcn)
    # result.label in {"high","medium","low", None}; None == abstained

Training is out of scope here: once B's graph lands and OUC-CGE is prepared
(``tools/prepare_ouccge.py``), train the relation weights and GCN with PyTorch
externally and inject the callable. The pure-numpy forward pass exists so the
plumbing is testable without torch, not because an untrained network means
anything.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

GroupLabel = Literal["high", "medium", "low"]

# Ordered class index convention used everywhere in this module and in
# tools/eval_group_activity.py. Ordinality is real (high > medium > low
# engagement); confusion matrices should respect it when read.
LABELS: tuple[GroupLabel, ...] = ("high", "medium", "low")

# Injectable signature: (adj [N,N], features [N,F]) -> logits [N,3]
ModelFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass
class GroupActivityConfig:
    """Knobs for graph construction and the abstention rule.

    Defaults are ENGINEERING GUESSES, chosen to fail safe (prefer abstaining),
    not tuned values. They must be re-derived from OUC-CGE before any claim.
    """

    # Relation types, mirroring ARG: appearance similarity + spatial position.
    use_appearance_relation: bool = True
    use_position_relation: bool = True

    # Appearance edges: each node connects to its k nearest neighbours by
    # cosine similarity (ARG uses fully-connected weighted graphs; kNN keeps
    # the adjacency sparse and robust to junk embeddings).
    appearance_knn_k: int = 3

    # Position edges: pairs closer than this fraction of frame diagonal are
    # neighbours. Normalized distance, so 0..~1.4 possible; 0.35 ~= adjacent
    # desks.
    position_radius: float = 0.35

    # --- Abstention (first-class, see module docstring deviation 3) -------- #
    # Fewer distinct students than this in the frame -> abstain. Two visible
    # students cannot speak for a ten-person group.
    min_students: int = 4
    # Share of nodes with NO usable feature vector -> abstain. Graph topology
    # alone (who stands near whom) carries almost no engagement signal.
    max_unknown_feature_rate: float = 0.5

    # Frames pooled per clip decision (majority vote of per-frame labels).
    clip_window_frames: int = 15
    # If more than this share of frames in the window abstained, the clip
    # abstains too -- a majority built on a minority of frames is not a
    # majority.
    max_abstained_frame_rate: float = 0.5


@dataclass
class GraphResult:
    """One frame's graph plus bookkeeping the caller should not have to
    recompute. ``abstain_reason`` is None iff the frame was usable."""

    adjacencies: dict[str, np.ndarray]          # relation name -> [N, N]
    features: np.ndarray                        # [N, F]
    node_ids: list[int]
    unknown_node_idx: list[int]                 # nodes lacking features
    n_total_in_frame: int                       # incl. undetected students?
    abstain_reason: str | None = None


@dataclass
class ClipDecision:
    """Final per-clip output. ``label`` None == explicit abstention."""

    label: GroupLabel | None
    confidence: float | None                    # softmax prob of the winner
    votes: dict[str, int] = field(default_factory=dict)
    n_frames_used: int = 0
    n_frames_abstained: int = 0
    reason: str | None = None                   # set iff label is None


def _cosine_similarity_matrix(feats: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity with a safe zero for zero vectors."""
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    normed = np.divide(
        feats, norms, out=np.zeros_like(feats, dtype=float), where=norms > 0
    )
    return normed @ normed.T


def _normalized_distance_matrix(centers: np.ndarray) -> np.ndarray:
    """Pairwise distances normalized by the spread (diagonal of the bbox of
    all centers), making ``position_radius`` resolution-independent."""
    if len(centers) < 2:
        return np.zeros((len(centers), len(centers)))
    span = centers.max(axis=0) - centers.min(axis=0)
    diag = float(np.linalg.norm(span)) or 1.0
    diff = centers[:, None, :] - centers[None, :, :]
    return np.linalg.norm(diff, axis=2) / diag


def build_graph(
    nodes: Sequence[Mapping[str, Any]], cfg: GroupActivityConfig | None = None
) -> GraphResult:
    """Build ARG-style relations for one frame.

    Returns a :class:`GraphResult` whose ``abstain_reason`` explains any
    refusal to produce a graph-level guess. Never raises for degenerate input
    (empty list, one node) -- it abstains; callers upstream crash on real
    classroom footage often enough already.
    """
    cfg = cfg or GroupActivityConfig()
    n = len(nodes)

    def abstain(reason: str, adj: dict[str, np.ndarray] | None = None) -> GraphResult:
        return GraphResult(
            adjacencies=adj or {},
            features=np.zeros((n, 0)),
            node_ids=[int(nd.get("track_id", -1)) for nd in nodes],
            unknown_node_idx=list(range(n)),
            n_total_in_frame=n,
            abstain_reason=reason,
        )

    if n == 0:
        return abstain("no nodes")
    if n < cfg.min_students:
        return abstain(f"only {n} nodes (< min_students={cfg.min_students})")

    # Build the feature matrix manually: node feature vectors may be missing
    # or different lengths, and np.asarray on ragged input raises under
    # modern numpy instead of object-wrapping.
    raw_feats = [list(nd.get("features") or []) for nd in nodes]
    width = max((len(r) for r in raw_feats), default=0)
    feats = np.zeros((n, width), dtype=float)
    for i, row in enumerate(raw_feats):
        if row:
            feats[i, : len(row)] = row
    known = [i for i in range(n) if width > 0 and feats[i].any()]
    unknown_rate = 1.0 - len(known) / n
    if unknown_rate > cfg.max_unknown_feature_rate:
        return abstain(
            f"{unknown_rate:.0%} of nodes lack features "
            f"(> max_unknown_feature_rate={cfg.max_unknown_feature_rate:.0%})"
        )

    centers = np.asarray([nd.get("center") or (0.0, 0.0) for nd in nodes], dtype=float)

    adjacencies: dict[str, np.ndarray] = {}

    if cfg.use_appearance_relation and len(known) >= 2:
        sim = _cosine_similarity_matrix(feats[known])
        k = min(cfg.appearance_knn_k, max(len(known) - 1, 1))
        app = np.zeros((n, n))
        for row, src in enumerate(known):
            sims = sim[row].copy()
            sims[row] = -np.inf                      # never self-edge
            top = np.argsort(sims)[-k:]
            for col in top:
                app[src, int(known[col])] = float(max(sim[row, col], 0.0))
                app[int(known[col]), src] = app[src, int(known[col])]
        adjacencies["appearance"] = app

    if cfg.use_position_relation:
        dist = _normalized_distance_matrix(centers)
        pos = (dist <= cfg.position_radius).astype(float)
        np.fill_diagonal(pos, 0.0)
        if pos.sum() > 0:
            adjacencies["position"] = pos

    if not adjacencies:
        return abstain("no relation edges survived thresholds")

    return GraphResult(
        adjacencies=adjacencies,
        features=feats,
        node_ids=[int(nd.get("track_id", -1)) for nd in nodes],
        unknown_node_idx=[i for i in range(n) if i not in known],
        n_total_in_frame=n,
        abstain_reason=None,
    )


def _renormalize(adj_sum: np.ndarray) -> np.ndarray:
    """Kipf & Welling renormalization: D^-1/2 (A + I) D^-1/2."""
    a_tilde = adj_sum + np.eye(len(adj_sum))
    deg = a_tilde.sum(axis=1)
    d_inv_sqrt = np.divide(
        1.0, np.sqrt(deg), out=np.zeros_like(deg), where=deg > 0
    )
    return a_tilde * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]


def default_numpy_forward(
    graph: GraphResult, rel_weights: Mapping[str, float] | None = None
) -> np.ndarray:
    """Unttrained reference forward pass -- plumbing, not a predictor.

    Fuses relations with fixed weights (uniform unless overridden), runs two
    mean-aggregation propagation steps, mean-pools nodes, and applies a fixed
    linear head that returns near-uniform logits. Its ONLY purposes: letting
    tests exercise the full path end-to-end, and serving as the shape
    contract an externally-trained PyTorch GCN must match. Do not read its
    outputs as predictions.
    """
    n = len(graph.node_ids)
    if n == 0 or graph.abstain_reason is not None:
        raise ValueError(f"cannot forward an abstained graph ({graph.abstain_reason})")
    weights = rel_weights or {name: 1.0 for name in graph.adjacencies}
    fused = sum(w * graph.adjacencies[name] for name, w in weights.items() if name in graph.adjacencies)
    a_hat = _renormalize(fused)
    h = graph.features if graph.features.shape[1] else np.zeros((n, 1))
    # Pad/truncate to a fixed width so the fixed head always matches.
    target_w = 16
    if h.shape[1] < target_w:
        h = np.pad(h, ((0, 0), (0, target_w - h.shape[1])))
    h = h[:, :target_w]
    for _ in range(2):
        h = np.tanh(a_hat @ h)
    pooled = h.mean(axis=0)
    logits = pooled[: len(LABELS)]
    return logits - logits.mean()


def classify_frame(
    graph: GraphResult,
    model_fn: ModelFn | None = None,
    rel_weights: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """One frame -> per-frame group label or an explicit abstention.

    Returns ``{"label": "high"|"medium"|"low"|None, "confidence": float|None,
    "abstain_reason": str|None}``. An injected ``model_fn`` (the future
    trained GCN) replaces :func:`default_numpy_forward` wholesale.
    """
    if graph.abstain_reason is not None:
        return {"label": None, "confidence": None, "abstain_reason": graph.abstain_reason}
    fn: ModelFn = (
        model_fn
        if model_fn is not None
        else (lambda adj_unused, feats_unused: default_numpy_forward(graph, rel_weights))
    )
    fused_adj = sum(graph.adjacencies.values())
    logits = np.asarray(fn(fused_adj, graph.features), dtype=float).reshape(-1)
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    idx = int(np.argmax(probs))
    return {
        "label": LABELS[idx],
        "confidence": float(probs[idx]),
        "abstain_reason": None,
    }


def classify_clip(
    frames: Sequence[Sequence[Mapping[str, Any]]],
    cfg: GroupActivityConfig | None = None,
    model_fn: ModelFn | None = None,
    rel_weights: Mapping[str, float] | None = None,
) -> ClipDecision:
    """Pool per-frame decisions into a per-clip label, majority-vote style,
    with abstention propagating upward: a clip whose window is mostly
    abstained frames abstains too (config: ``max_abstained_frame_rate``).
    Ties in the vote are broken toward the LOWER engagement class -- the same
    safer-error direction the rest of this project uses."""
    cfg = cfg or GroupActivityConfig()
    window = frames[-cfg.clip_window_frames :] if cfg.clip_window_frames else list(frames)
    votes = {name: 0 for name in LABELS}
    used = abstained = 0
    confidences: dict[str, list[float]] = {name: [] for name in LABELS}
    for frame_nodes in window:
        graph = build_graph(frame_nodes, cfg)
        decision = classify_frame(graph, model_fn=model_fn, rel_weights=rel_weights)
        if decision["label"] is None:
            abstained += 1
            continue
        used += 1
        votes[decision["label"]] += 1
        confidences[decision["label"]].append(decision["confidence"])

    if used == 0:
        return ClipDecision(
            label=None,
            confidence=None,
            votes=votes,
            n_frames_used=used,
            n_frames_abstained=abstained,
            reason=f"all {len(window)} frames abstained",
        )
    if abstained / max(used + abstained, 1) > cfg.max_abstained_frame_rate:
        return ClipDecision(
            label=None,
            confidence=None,
            votes=votes,
            n_frames_used=used,
            n_frames_abstained=abstained,
            reason=(
                f"{abstained}/{used + abstained} frames abstained "
                f"(> max_abstained_frame_rate={cfg.max_abstained_frame_rate:.0%})"
            ),
        )

    best = max(votes.values())
    tied = [name for name in LABELS if votes[name] == best]
    # LABELS is ordered high->low, so the LAST tied member is the lowest
    # engagement class: ties break downward, matching the docstring.
    winner = tied[-1]
    confs = confidences[winner]
    return ClipDecision(
        label=winner,
        confidence=float(sum(confs) / len(confs)) if confs else None,
        votes=votes,
        n_frames_used=used,
        n_frames_abstained=abstained,
        reason=None,
    )


class TorchGCN:
    """Shape-contract sketch for the future trained model.

    Present so the injection point is concrete rather than hypothetical.
    Import-guarded; requires ``torch``. Weights are randomly initialised --
    i.e. MEANINGLESS -- until trained on prepared OUC-CGE splits. This class
    existing is not a claim that a model is trained; it is the opposite: a
    named slot showing exactly what is still missing.
    """

    def __init__(self, feat_dim: int, hidden: int = 64, rel_names: Sequence[str] = ("appearance", "position")):
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "TorchGCN needs torch; install requirements.txt extras or "
                "inject your own model_fn instead."
            ) from exc
        self.torch = torch
        self.rel_names = tuple(rel_names)
        self.rel_fc = torch.nn.ModuleDict(
            {name: torch.nn.Linear(feat_dim, hidden, bias=False) for name in rel_names}
        )
        self.out = torch.nn.Linear(hidden, len(LABELS))

    def __call__(self, adj: np.ndarray, feats: np.ndarray) -> np.ndarray:
        torch = self.torch
        a = torch.tensor(_renormalize(adj.astype(float)), dtype=torch.float32)
        x = torch.tensor(feats.astype(float), dtype=torch.float32)
        if x.shape[1] != next(self.rel_fc[self.rel_names[0]].parameters()).shape[1]:
            raise ValueError(
                f"feature dim mismatch: graph gives {x.shape[1]}, model expects "
                "different -- retrain or pad features consistently"
            )
        rel_outs = [fc(x) for fc in self.rel_fc.values()]
        h = sum(rel_outs) / len(rel_outs)
        h = torch.tanh(a @ h)
        h = torch.tanh(a @ h)
        return self.out(h.mean(dim=0)).detach().numpy()
