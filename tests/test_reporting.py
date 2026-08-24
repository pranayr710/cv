"""Tests for backend.reporting (O6 layer).

Covers, in order:
1. Class-level aggregation math: medians, off-task rate, unknown rate.
2. Empty buckets are RETAINED with unknown_rate 1.0 (gaps are information).
3. Defensive reads: records missing fields degrade to 'unknown', never crash
   and never contaminate averages.
4. student_trajectory is an explicit per-id drill-down and nothing else.
5. render_html carries the standing caveat and exposes no individual rows.
"""

import json

import pytest

from backend.reporting import (
    class_trends,
    load_profiles,
    render_html,
    student_trajectory,
)


def profile(sid=1, t=0.0, on=3, off=1, unk=0):
    graded = on + off
    pct = (on / graded * 100.0) if graded else None
    return {
        "student_id": sid,
        "t_s": t,
        "concentration": {
            "on": on,
            "off": off,
            "unknown": unk,
            "behavioral_proxy_pct": pct,
            "concentration_pct": pct,
        },
    }


# --- 1. aggregation ----------------------------------------------------------


def test_class_trends_aggregates_across_all_students() -> None:
    profiles = [
        profile(sid=1, t=10.0, on=3, off=1),
        profile(sid=2, t=20.0, on=1, off=3),
    ]
    buckets = class_trends(profiles, duration_s=60.0, bucket_seconds=30.0)
    assert len(buckets) == 2
    first = buckets[0]
    assert first.n_students_observed == 2
    # graded = (4 + 4) = 8; off = 1 + 3 = 4 -> off-task rate 0.5
    assert first.off_task_rate == pytest.approx(0.5)
    assert first.unknown_rate == pytest.approx(0.0)
    # per-student pcts: 75.0 and 25.0 -> median 50.0
    assert first.median_on_task_pct == pytest.approx(50.0)


def test_median_handles_even_and_odd_counts() -> None:
    odd = class_trends(
        [
            profile(1, 5.0, on=9, off=1),   # 90%
            profile(2, 6.0, on=1, off=1),   # 50%
            profile(3, 7.0, on=0, off=1),   # 0%
        ],
        duration_s=30.0,
        bucket_seconds=30.0,
    )[0]
    assert odd.median_on_task_pct == pytest.approx(50.0)

    even = class_trends(
        [profile(1, 5.0, on=9, off=1), profile(2, 6.0, on=1, off=1)],
        duration_s=30.0,
    )[0]
    assert even.median_on_task_pct == pytest.approx(70.0)  # (90+50)/2


# --- 2. gaps retained ---------------------------------------------------------


def test_empty_buckets_are_kept_not_dropped() -> None:
    profiles = [profile(1, t=0.0)]
    buckets = class_trends(profiles, duration_s=300.0, bucket_seconds=100.0)
    assert len(buckets) == 3
    gap = buckets[2]
    assert gap.n_observations == 0
    assert gap.unknown_rate == 1.0          # honest default: everything unknown
    assert gap.median_on_task_pct is None
    assert gap.off_task_rate == 0.0


def test_timestamps_outside_range_do_not_crash() -> None:
    profiles = [profile(1, t=-5.0), profile(1, t=999.0), profile(2, t=10.0)]
    buckets = class_trends(profiles, duration_s=60.0)
    assert buckets[0].n_students_observed == 1
    assert any("lacked usable timestamps" in n for n in buckets[0].notes)


# --- 3. defensive reads --------------------------------------------------------


def test_records_without_concentration_or_time_degrade_safely(tmp_path) -> None:
    bad = [{"student_id": 9}, {"t_s": 5.0}]  # no concentration / no timestamp
    good = [profile(1, t=5.0)]
    buckets = class_trends(bad + good, duration_s=30.0)
    assert buckets[0].n_observations == 4  # only the good record counted


def test_load_profiles_rejects_corrupt_lines_with_location(tmp_path) -> None:
    path = tmp_path / "profiles.jsonl"
    path.write_text('{"ok": 1}\n{not json}\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"profiles\.jsonl:2"):
        load_profiles(str(path))


def test_load_profiles_roundtrips(tmp_path) -> None:
    path = tmp_path / "p.jsonl"
    path.write_text("\n".join(json.dumps(profile(i)) for i in (1, 2)) + "\n", encoding="utf-8")
    records = load_profiles(str(path))
    assert [r["student_id"] for r in records] == [1, 2]


# --- 4. explicit drill-down -----------------------------------------------------


def test_student_trajectory_returns_only_that_student_sorted() -> None:
    profiles = [
        profile(2, t=40.0),
        profile(1, t=20.0),
        profile(1, t=10.0),
        profile(3, t=30.0),
    ]
    traj = student_trajectory(profiles, student_id=1)
    assert [point["t_s"] for point in traj] == [10.0, 20.0]


# --- 5. rendering ----------------------------------------------------------------


def test_render_html_carries_caveat_and_no_individual_rows() -> None:
    buckets = class_trends([profile(1, t=5.0)], duration_s=60.0)
    html = render_html(buckets)
    assert "Behavioral proxy" in html or "behavioral proxy" in html.lower()
    assert "Individual students are intentionally absent" in html
    # aggregated numbers may appear; individual ids must not
    assert ">1<" not in html.split("<caption>")[0]
