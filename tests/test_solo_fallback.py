"""Scoring a room too small for the full model, without pretending.

The full model reads three features that describe the room -- how many peers
are in frame and what they are doing -- and `peer_count` is n/(n+8), so its
fitted minimum of 0.333 means four peers. Below five people in frame it has
never seen anything like the vector it is handed, and refuses.

That refusal is right for that model and wrong as a final answer: eighteen
of the twenty-one features are about the student alone and are in range at
any room size. The solo model is fitted on those, measured the same way, and
used only where the full one declines.

Measured out of fold on the 82 twice-confirmed labels: full 0.634 accuracy /
0.620 macro F1, solo 0.598 / 0.587. Peer context is worth about 0.03 macro
F1, which is why this is a fallback and not a replacement -- and why a score
that came from it says so.
"""
import pytest

from backend.attention_score import SOLO_FEATURE_NAMES, AttentionScorer
from backend.engagement_features import (
    BASE_FEATURE_NAMES,
    BODY_FEATURE_NAMES,
    PEER_FEATURE_NAMES,
)
from backend.engagement_model import EngagementModel

ALL_NAMES = (list(BASE_FEATURE_NAMES) + list(BODY_FEATURE_NAMES)
             + list(PEER_FEATURE_NAMES))


class TestTheFeatureSets:
    def test_solo_is_the_full_set_minus_the_peer_features(self):
        assert SOLO_FEATURE_NAMES == [n for n in ALL_NAMES
                                      if n not in set(PEER_FEATURE_NAMES)]

    def test_it_drops_exactly_three(self):
        assert len(ALL_NAMES) - len(SOLO_FEATURE_NAMES) == 3

    def test_no_peer_feature_survives(self):
        assert not set(SOLO_FEATURE_NAMES) & set(PEER_FEATURE_NAMES)

    def test_the_order_is_preserved(self):
        """The columns are positional once loaded; a reordering here would
        feed the weights the wrong features silently."""
        kept = [n for n in ALL_NAMES if n in set(SOLO_FEATURE_NAMES)]
        assert kept == SOLO_FEATURE_NAMES


class TestTheLoaderStillChecks:
    """Relaxing the feature-set check for everyone would let a stale model
    score against the wrong columns. The caller states what it expects."""

    def test_a_subset_model_loads_when_declared(self, tmp_path):
        from backend.engagement_model import load_or_none

        path = tmp_path / "solo.json"
        EngagementModel([0.0] * len(SOLO_FEATURE_NAMES), 0.0,
                        SOLO_FEATURE_NAMES,
                        [0.0] * len(SOLO_FEATURE_NAMES),
                        [1.0] * len(SOLO_FEATURE_NAMES)).save(path)
        assert load_or_none(path, expect=SOLO_FEATURE_NAMES) is not None

    def test_the_same_model_is_refused_as_the_full_one(self, tmp_path):
        from backend.engagement_model import load_or_none

        path = tmp_path / "solo.json"
        EngagementModel([0.0] * len(SOLO_FEATURE_NAMES), 0.0,
                        SOLO_FEATURE_NAMES,
                        [0.0] * len(SOLO_FEATURE_NAMES),
                        [1.0] * len(SOLO_FEATURE_NAMES)).save(path)
        assert load_or_none(path) is None


class TestTheShippedPair:
    def test_both_models_are_present(self):
        s = AttentionScorer()
        if s.model is None:
            pytest.skip("no trained model present")
        assert s.solo is not None, (
            "the solo model is missing, so a small room scores nothing -- "
            "run tools/train/train_solo_model.py")

    def test_the_columns_line_up(self):
        s = AttentionScorer()
        if s.model is None or s.solo is None:
            pytest.skip("models not present")
        assert s.solo_columns is not None
        picked = [s.model.feature_names[i] for i in s.solo_columns]
        assert picked == s.solo.feature_names

    def test_a_small_room_is_scored_and_says_which_model(self):
        """The symptom this exists for: two people on screen, both blank."""
        import json
        from pathlib import Path

        from backend.attention_score import annotate_attention

        graph = Path("outputs/live/live_graph.jsonl")
        s = AttentionScorer()
        if s.model is None or s.solo is None or not graph.exists():
            pytest.skip("models or the small-room session not present")

        scored, by_solo = 0, 0
        for line in graph.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            g = annotate_attention(json.loads(line), s)
            for n in g.get("nodes", []):
                f = n.get("features") or {}
                reason = f.get("attention_reason") or ""
                if f.get("attention_score") is not None \
                        and "probability" in reason:
                    scored += 1
                    by_solo += "solo model" in reason
        assert scored > 0, "a small room still scores nothing"
        assert by_solo == scored, (
            "a small room was scored by the full model, which was not fitted "
            "for it")

    def test_the_full_model_still_handles_its_own_footage(self):
        """The fallback must not quietly take over where the better model
        applies."""
        import json
        from pathlib import Path

        from backend.attention_score import annotate_attention

        graph = Path("outputs/final2/live_graph.jsonl")
        s = AttentionScorer()
        if s.model is None or s.solo is None or not graph.exists():
            pytest.skip("models or the classroom graph not present")

        full = solo = 0
        for line in graph.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            g = annotate_attention(json.loads(line), s)
            for n in g.get("nodes", []):
                reason = (n.get("features") or {}).get("attention_reason") or ""
                if "probability" not in reason:
                    continue
                if "solo model" in reason:
                    solo += 1
                else:
                    full += 1
        assert full > solo * 5, (
            f"the full model scored {full} and the fallback {solo} on "
            "classroom footage; the fallback should be rare there")
