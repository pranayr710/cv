"""Tests for tools.eval_group_activity metrics (the audited metric source).

Covers:
1. Wilson CI bounds stay in [0,1] and bracket the point estimate, including
   the degenerate 0/0 and n/n cases.
2. evaluate(): accuracy, macro-F1, confusion orientation (rows=prediction).
3. Abstain policies are explicit and differ: 'exclude' drops them from the
   denominator; 'wrong' charges them — and both echo their counts.
"""

import pytest

from tools.eval_group_activity import evaluate, wilson_ci


def pair(label, truth):
    return {"clip_id": "x", "label": label}, {"source_video": "s", "label": truth, "split": "test"}


def test_wilson_stays_in_unit_interval_at_extremes() -> None:
    for successes, n in [(0, 5), (5, 5), (1, 100), (99, 100)]:
        lo, hi = wilson_ci(successes, n)
        assert 0.0 <= lo <= hi <= 1.0
        p = successes / n
        assert lo - 1e-9 <= p <= hi + 1e-9


def test_wilson_degenerate_zero_samples() -> None:
    assert wilson_ci(0, 0) == (0.0, 1.0)


def test_evaluate_accuracy_and_macro_f1() -> None:
    pairs = [
        pair("high", "high"),
        pair("medium", "medium"),
        pair("low", "low"),
        pair("high", "low"),   # wrong
    ]
    res = evaluate(pairs, abstain_policy="exclude")
    assert res["n_scored"] == 4
    assert res["accuracy"] == 0.75
    lo, hi = res["wilson95"]
    assert lo < 0.75 < hi
    # per-class F1: high=2/3, med=1.0, low=2/3 -> macro = 7/9 ≈ .7778
    assert abs(res["macro_f1"] - 7 / 9) < 1e-3
    conf = res["confusion_rows_pred_cols_true"]
    assert conf["high"]["high"] == 1
    assert conf["high"]["low"] == 1   # row=prediction(high), col=truth(low)
    assert conf["low"]["low"] == 1


def test_abstain_policies_differ_and_are_counted() -> None:
    pairs = [
        pair("high", "high"),
        pair(None, "low"),     # abstained
        pair(None, "high"),    # abstained
    ]
    excl = evaluate(pairs, abstain_policy="exclude")
    wrong = evaluate(pairs, abstain_policy="wrong")
    assert excl["n_abstained"] == wrong["n_abstained"] == 2
    assert excl["accuracy"] == 1.0            # only scored clip is correct
    assert wrong["accuracy"] == pytest.approx(1 / 3, abs=1e-3)  # tool rounds 4dp
    assert excl["wilson95"][0] > wrong["wilson95"][0]
