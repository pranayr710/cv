"""Tests for tools.prepare_ouccge (dataset prep + THE split rule).

Covers, in order:
1. labels.csv parsing: valid parse; unknown label rejected; duplicate id
   rejected; missing media warned not fatal.
2. THE RULE: whole source videos land in exactly one split — no clip of a
   source ever shares a fold with another source's clips.
3. Determinism: identical input -> byte-identical split assignment.
4. Stratification is approximate and reported (all three splits non-empty on
   a reasonable mix).
5. Manifest writing: schema + sortedness + roundtrip through the evaluator's
   leakage check (tools.eval_group_activity.load_manifest).
"""

import csv
import pytest

from tools.eval_group_activity import load_manifest as eval_load_manifest
from tools.prepare_ouccge import (
    VALID_LABELS,
    assign_splits,
    build_manifest,
    discover_clips,
    verify,
)


def make_tree(tmp_path, rows, media_ids):
    root = tmp_path
    (root / "videos").mkdir(exist_ok=True)
    for mid in media_ids:
        (root / "videos" / f"{mid}.mp4").touch()
    labels = tmp_path / "labels.csv"
    with open(labels, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["clip_id", "source_video", "camera", "label"]
        )
        writer.writeheader()
        writer.writerows(rows)
    return root, labels


BASIC_ROWS = [
    {"clip_id": f"c{i:03d}", "source_video": f"src{i % 4}", "camera": "front", "label": lbl}
    for i, lbl in enumerate(["High", "High", "Medium", "Medium", "Low", "Low"] * 2)
]


# --- 1. parsing ------------------------------------------------------------------


def test_discover_clips_parses_and_matches_media(tmp_path) -> None:
    ids = [r["clip_id"] for r in BASIC_ROWS]
    root, labels = make_tree(tmp_path, BASIC_ROWS, ids)
    clips = discover_clips(root, labels)
    assert len(clips) == len(BASIC_ROWS)
    assert {c.label for c in clips} == set(VALID_LABELS)


def test_unknown_label_rejected(tmp_path) -> None:
    rows = [{"clip_id": "c1", "source_video": "s1", "camera": "front", "label": "ecstatic"}]
    root, labels = make_tree(tmp_path, rows, ["c1"])
    with pytest.raises(SystemExit, match="not one of"):
        discover_clips(root, labels)


def test_duplicate_clip_id_rejected(tmp_path) -> None:
    row = {"clip_id": "c1", "source_video": "s1", "camera": "front", "label": "High"}
    root, labels = make_tree(tmp_path, [row, dict(row)], ["c1"])
    with pytest.raises(SystemExit, match="duplicate"):
        discover_clips(root, labels)


def test_missing_media_warns_but_parses(capsys, tmp_path) -> None:
    rows = BASIC_ROWS[:3]
    root, labels = make_tree(tmp_path, rows, [])  # NO media files at all
    clips = discover_clips(root, labels)
    assert len(clips) == 3
    assert "WARN" in capsys.readouterr().err


# --- 2. THE RULE ---------------------------------------------------------------


def test_whole_sources_land_in_exactly_one_split() -> None:
    clips = [
        type("C", (), {"clip_id": r["clip_id"], "source_video": r["source_video"], "camera": r["camera"], "label": r["label"]})()
        for r in BASIC_ROWS
    ]
    split_of = assign_splits(clips)
    source_splits: dict[str, set[str]] = {}
    for c in clips:
        source_splits.setdefault(c.source_video, set()).add(split_of[c.source_video])
    assert all(len(v) == 1 for v in source_splits.values())


# --- 3. determinism ---------------------------------------------------------------


def test_assignment_is_deterministic() -> None:
    mk = lambda: [
        type("C", (), {"clip_id": r["clip_id"], "source_video": r["source_video"], "camera": r["camera"], "label": r["label"]})()
        for r in BASIC_ROWS
    ]
    assert assign_splits(mk()) == assign_splits(mk())


# --- 4. stratification sanity ------------------------------------------------------


def test_all_splits_get_content_on_mixed_sources() -> None:
    rows = [
        {"clip_id": f"c{i:03d}", "source_video": f"src{i}", "camera": "front", "label": ["High", "Medium", "Low"][i % 3]}
        for i in range(12)
    ]
    clips = [
        type("C", (), {"clip_id": r["clip_id"], "source_video": r["source_video"], "camera": r["camera"], "label": r["label"]})()
        for r in rows
    ]
    split_of = assign_splits(clips)
    used = set(split_of.values())
    assert used == {"train", "val", "test"}
    findings = verify(clips, split_of)
    assert not any("LEAKAGE" in f for f in findings)


# --- 5. manifest + downstream compatibility -------------------------------------------


def test_manifest_roundtrips_through_evaluator_leakage_check(tmp_path) -> None:
    rows = BASIC_ROWS[:6]
    ids = [r["clip_id"] for r in rows]
    root, labels = make_tree(tmp_path, rows, ids)
    clips = discover_clips(root, labels)
    split_of = assign_splits(clips)
    manifest = build_manifest(clips, split_of, tmp_path / "prepared")
    loaded = eval_load_manifest(manifest)  # must NOT raise SystemExit
    assert len(loaded) == 6


def test_evaluator_aborts_on_hand_edited_leaky_manifest(tmp_path) -> None:
    leaky = tmp_path / "leaky.csv"
    with open(leaky, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["clip_id", "source_video", "camera", "label", "split"])
        w.writerow(["c1", "srcA", "front", "high", "train"])
        w.writerow(["c2", "srcA", "side", "low", "test"])  # same source, other fold
    with pytest.raises(SystemExit, match="leakage"):
        eval_load_manifest(leaky)
