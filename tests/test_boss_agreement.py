"""Tests for tools.boss_agreement (Cohen's kappa + coverage discipline).

Covers:
1. Kappa math against a hand-computed example.
2. Chance-only agreement scores near 0 (the reason kappa exists).
3. Perfect and inverse agreement extremes.
4. Coverage reporting: unmatched rows are counted loudly, never dropped silently.
5. Degenerate single-category raters (kappa undefined case) don't divide by zero.
"""

import pytest

from tools.boss_agreement import Row, cohen_kappa, compare, load_rows

# --- 1. hand-computed example ---------------------------------------------------
# a = on,on,off,unknown,on ; b = on,off,off,unknown,on
# observed = 4/5 = .8 ; expected = (3*2+1*2+1*1)/25 = 9/25 = .36
# kappa = (.8-.36)/(.64) = .6875
HAND_CASE_A = ["on", "on", "off", "unknown", "on"]
HAND_CASE_B = ["on", "off", "off", "unknown", "on"]


def test_kappa_matches_hand_computation() -> None:
    assert cohen_kappa(HAND_CASE_A, HAND_CASE_B) == pytest.approx(0.6875)


def test_chance_agreement_scores_near_zero() -> None:
    # independent shuffles of the same marginals: expected == observed
    a = ["on", "on", "off", "off"]
    b = ["on", "off", "on", "off"]  # same marginals, half agreement by chance
    assert abs(cohen_kappa(a, b)) < 1e-9 or cohen_kappa(a, b) < 0.2


def test_perfect_and_inverse_extremes() -> None:
    codes = ["on", "off", "unknown", "on"]
    assert cohen_kappa(codes, list(codes)) == pytest.approx(1.0)
    flipped = {"on": "off", "off": "on", "unknown": "unknown"}
    assert cohen_kappa(codes, [flipped[c] for c in codes]) < 0


# --- 4/5. compare(): joining + degenerate cases ---------------------------------


def _rows(pairs):
    return {
        (interval, sid): Row(interval, sid, code)
        for interval, sid, code in pairs
    }


def test_compare_reports_coverage_gaps_loudly() -> None:
    human = _rows([("i1", "1", "on"), ("i2", "1", "off"), ("i3", "1", "on")])
    system = _rows([("i1", "1", "on"), ("i2", "1", "off"), ("i9", "1", "off")])
    res = compare(human, system)
    assert res["n_joined"] == 2
    assert res["coverage"] == pytest.approx(2 / 3, abs=1e-3)  # tool rounds to 4dp
    assert res["n_only_in_human"] == 1
    assert res["n_only_in_system"] == 1
    assert res["percent_agreement"] == 1.0
    assert res["cohen_kappa"] == 1.0


def test_compare_with_single_category_raters_does_not_divide_by_zero() -> None:
    human = _rows([(f"i{i}", "1", "on") for i in range(3)])
    system_same = _rows([(f"i{i}", "1", "on") for i in range(3)])
    res = compare(human, system_same)
    assert res["cohen_kappa"] == 1.0

    system_other = _rows([(f"i{i}", "1", "off") for i in range(3)])
    res2 = compare(human, system_other)
    assert res2["cohen_kappa"] == 0.0


def test_load_rows_rejects_unknown_codes(tmp_path) -> None:
    path = tmp_path / "r.csv"
    path.write_text("interval,student_id,code\ni1,1,sleeping\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="not in"):
        load_rows(path)


def test_load_rows_rejects_duplicate_keys(tmp_path) -> None:
    path = tmp_path / "r.csv"
    path.write_text(
        "interval,student_id,code\ni1,1,on\ni1,1,off\n", encoding="utf-8"
    )
    with pytest.raises(SystemExit, match="duplicate"):
        load_rows(path)
