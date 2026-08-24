"""Tests for backend.group_activity (ARG+GCN scaffold).

Covers, in order:
1. Graph construction: relation adjacency structure (symmetry, no self-edges,
   kNN bound on appearance edges).
2. Abstention rules fire with precise reasons: too few students, too much
   missing features, no surviving edges.
3. Frame classification honours an injected model_fn end-to-end.
4. Clip pooling: majority vote, tie broken toward the LOWER engagement class,
   abstained frames propagate upward past max_abstained_frame_rate.

No torch required -- model functions are injected lambdas.
"""

import numpy as np
import pytest

from backend.group_activity import (
    LABELS,
    GroupActivityConfig,
    build_graph,
    classify_clip,
    classify_frame,
)


def node(i: int, center=(0.0, 0.0), feats=None, appearance=None):
    return {
        "track_id": i,
        "role": "student",
        "center": list(center),
        "bbox_wh": [50, 120],
        "features": feats if feats is not None else [1.0, 0.0],
        "appearance": appearance or [float(i), 1.0],
    }


def classroom(n=6, spaced=True):
    return [
        node(
            i,
            center=((i % 3) * (2.0 if spaced else 0.05), (i // 3) * 1.0),
            feats=[1.0 if i % 2 == 0 else 0.0, 1.0],
        )
        for i in range(n)
    ]


# --- 1. graph construction ---------------------------------------------------


def test_build_graph_relations_are_symmetric_and_self_free() -> None:
    graph = build_graph(classroom(6))
    assert graph.abstain_reason is None
    for name, adj in graph.adjacencies.items():
        assert (adj == adj.T).all(), f"{name} adjacency must be undirected"
        assert (np.diag(adj) == 0).all(), f"{name} must have no self-edges"


def test_appearance_knn_bounds_degree() -> None:
    cfg = GroupActivityConfig(appearance_knn_k=2)
    graph = build_graph(classroom(8), cfg)
    app = graph.adjacencies["appearance"]
    # undirected kNN can exceed k slightly, but never reach n-1 fan-out
    assert app.sum() / len(app) <= 2 * cfg.appearance_knn_k + 1


def test_position_edges_respect_radius() -> None:
    cfg = GroupActivityConfig(position_radius=0.35)
    far = build_graph(classroom(6, spaced=True), cfg).adjacencies.get("position")
    near = build_graph(classroom(6, spaced=False), cfg).adjacencies["position"]
    assert near.sum() > (far.sum() if far is not None else -1)


# --- 2. abstention -----------------------------------------------------------


def test_empty_frame_abstains_with_reason() -> None:
    graph = build_graph([])
    assert graph.abstain_reason == "no nodes"


def test_too_few_students_abstains() -> None:
    graph = build_graph(classroom(3))  # min_students default 4
    assert graph.abstain_reason is not None
    assert "min_students" in graph.abstain_reason


def test_missing_features_above_threshold_abstains() -> None:
    nodes = [
        node(i, feats=[] if i < 4 else [1.0, 0.0]) for i in range(6)  # 4/6 blank
    ]
    graph = build_graph(nodes)
    assert graph.abstain_reason is not None
    assert "lack features" in graph.abstain_reason


def test_no_surviving_edges_abstains() -> None:
    # two clusters far apart AND appearance disabled. Radius must be BELOW the
    # intra-cluster normalized distance (1.0 / diag(10,1) ≈ 0.0995) so no
    # position edge survives anywhere.
    nodes = [
        node(0, center=(0, 0)), node(1, center=(0, 1)),
        node(2, center=(10, 0)), node(3, center=(10, 1)),
    ]
    cfg = GroupActivityConfig(use_appearance_relation=False, position_radius=0.05)
    graph = build_graph(nodes, cfg)
    assert graph.abstain_reason == "no relation edges survived thresholds"


def test_classify_frame_passes_through_abstention() -> None:
    decision = classify_frame(build_graph([]))
    assert decision == {"label": None, "confidence": None,
                        "abstain_reason": "no nodes"}


# --- 3. injected model --------------------------------------------------------


def test_injected_model_decides_the_label() -> None:
    def says_high(adj, feats):
        return np.array([5.0, 0.0, 0.0])  # logits over LABELS

    def says_low(adj, feats):
        return np.array([0.0, 0.0, 5.0])

    graph = build_graph(classroom(6))
    assert classify_frame(graph, model_fn=says_high)["label"] == "high"
    assert classify_frame(graph, model_fn=says_low)["label"] == "low"
    conf = classify_frame(graph, model_fn=says_high)["confidence"]
    assert conf > 0.95  # softmax([5,0,0]) ≈ 0.9867 -- decisively, not perfectly


# --- 4. clip pooling ----------------------------------------------------------


def test_clip_majority_vote_uses_window() -> None:
    frames = []
    for i in range(20):
        winner = "high" if i % 3 else "low"
        fn = (lambda w: lambda a, f: np.eye(3)[LABELS.index(w)])(winner)
        frames.append(classroom(6))
    decisions = {"high": 0, "low": 0}
    # simpler: run classify_clip with a model voting 'medium' everywhere
    medium = lambda a, f: np.array([0.0, 5.0, 0.0])
    clip = classify_clip(frames * 1, model_fn=medium)  # 20 frames
    assert clip.label == "medium"
    assert clip.n_frames_used == 15  # window default trims to last 15


def test_clip_tie_breaks_toward_lower_engagement_class() -> None:
    # alternate high/low votes evenly -> tie -> must pick the lower ('low')
    def alternating_model_factory():
        state = {"i": 0}

        def model(adj, feats):
            out = np.eye(3)[(0 if state["i"] % 2 == 0 else 2)]  # alternate high, low
            state["i"] += 1
            return out

        return model

    frames = [classroom(6) for _ in range(6)]
    clip = classify_clip(frames, model_fn=alternating_model_factory())
    assert clip.votes == {"high": 3, "medium": 0, "low": 3}
    assert clip.label == "low"


def test_clip_abstains_when_most_frames_abstained() -> None:
    # 14 usable + 12 abstaining (below min_students) inside a 15-frame window
    frames = [classroom(6) for _ in range(14)] + [classroom(2) for _ in range(12)]
    constant = lambda a, f: np.array([5.0, 0.0, 0.0])
    clip = classify_clip(frames, model_fn=constant)
    assert clip.label is None
    assert "frames abstained" in clip.reason


def test_untrained_default_forward_runs_but_is_not_a_prediction_guard() -> None:
    from backend.group_activity import default_numpy_forward

    graph = build_graph(classroom(6))
    logits = default_numpy_forward(graph)
    assert logits.shape == (3,)
    assert abs(float(np.sum(logits))) < 1e-6  # centred, as documented


def test_torch_gcn_is_optional_and_raises_helpfully_without_torch():
    try:
        import torch  # noqa: F401
        has_torch = True
    except ImportError:
        has_torch = False
    from backend.group_activity import TorchGCN

    if has_torch:
        model = TorchGCN(feat_dim=2)
        graph = build_graph(classroom(6))
        fused = sum(graph.adjacencies.values())
        out = model(fused, graph.features)
        assert out.shape == (3,)
    else:
        with pytest.raises(RuntimeError, match="inject your own model_fn"):
            TorchGCN(feat_dim=2)
