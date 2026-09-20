"""Tests for the DIPSER label adapter.

Written against the schema in the authors' own loader (classes/label_reader.py
and classes/data_processor.py), not against the data, which is not present
here. Each test pins one assumption that, if wrong, would produce a target set
that trains without error and means nothing.
"""
import json
from pathlib import Path

import pytest

from backend.dipser import (
    LabelledWindow,
    Rating,
    iter_labelled_windows,
    parse_time_ms,
    read_ratings,
    summarise,
    time_ms_from_image,
    to_label,
)


def _subject(root: Path, experiment: str, subject: str,
             per_rater: dict[str, list[tuple[str, int]]]) -> Path:
    """Build a DIPSER-shaped subject folder with the given ratings."""
    subject_dir = root / experiment / subject
    labels = subject_dir / "labels"
    labels.mkdir(parents=True, exist_ok=True)
    for rater, tags in per_rater.items():
        payload = [{"datetime": stamp, "attention": value, "emotion": "Hope"}
                   for stamp, value in tags]
        (labels / f"{rater}.json").write_text(json.dumps(payload),
                                              encoding="utf-8")
    return subject_dir


def test_parse_time_handles_dipser_stamp():
    """HH:MM:SS:ffffff, where the last field is microseconds."""
    assert parse_time_ms("00:00:01:000000") == 1000
    assert parse_time_ms("01:00:00:000000") == 3_600_000
    assert parse_time_ms("00:00:00:500000") == 500


def test_fractional_field_is_padded_not_truncated():
    """'5' means 500000 microseconds, not 5.

    Reading it as 5 would put every sub-second timestamp at effectively zero
    offset, silently collapsing frames onto the same instant.
    """
    assert parse_time_ms("00:00:00:5") == 500


def test_parse_time_rejects_malformed():
    assert parse_time_ms("") is None
    assert parse_time_ms("noon") is None
    assert parse_time_ms("12:30") is None


def test_image_filename_is_the_timestamp():
    """DIPSER encodes the time in the filename with underscores for colons."""
    assert time_ms_from_image(Path("images/00_00_02_000000.png")) == 2000


def test_each_label_file_is_a_rater(tmp_path: Path):
    """Several files per subject means several opinions, not several sessions.

    Treating them as one stream would hide disagreement and inflate the
    apparent amount of independent evidence.
    """
    subject = _subject(tmp_path, "experiment_a", "subject_1", {
        "expert1": [("00:00:01:000000", 5)],
        "expert2": [("00:00:01:000000", 1)],
    })
    ratings = read_ratings(subject)
    assert {r.rater for r in ratings} == {"expert1", "expert2"}
    assert sorted(r.attention for r in ratings) == [1, 5]


def test_tags_without_attention_are_skipped(tmp_path: Path):
    """DIPSER tags may carry only an emotion."""
    subject_dir = tmp_path / "experiment_a" / "subject_1" / "labels"
    subject_dir.mkdir(parents=True)
    (subject_dir / "expert1.json").write_text(json.dumps([
        {"datetime": "00:00:01:000000", "emotion": "Boredom"},
        {"datetime": "00:00:02:000000", "attention": 4, "emotion": "Hope"},
    ]), encoding="utf-8")
    ratings = read_ratings(subject_dir.parent)
    assert len(ratings) == 1
    assert ratings[0].attention == 4


def test_out_of_range_attention_is_rejected(tmp_path: Path):
    """The scale is 1-5; anything else is a parsing mistake, not a value."""
    subject = _subject(tmp_path, "experiment_a", "subject_1", {
        "expert1": [("00:00:01:000000", 0), ("00:00:02:000000", 9),
                    ("00:00:03:000000", 3)],
    })
    ratings = read_ratings(subject)
    assert [r.attention for r in ratings] == [3]


def test_missing_labels_dir_is_empty_not_an_error(tmp_path: Path):
    (tmp_path / "experiment_a" / "subject_1").mkdir(parents=True)
    assert read_ratings(tmp_path / "experiment_a" / "subject_1") == []


def test_malformed_json_does_not_kill_the_walk(tmp_path: Path):
    labels = tmp_path / "experiment_a" / "subject_1" / "labels"
    labels.mkdir(parents=True)
    (labels / "broken.json").write_text("{not json", encoding="utf-8")
    (labels / "expert1.json").write_text(json.dumps([
        {"datetime": "00:00:01:000000", "attention": 4}]), encoding="utf-8")
    ratings = read_ratings(labels.parent)
    assert len(ratings) == 1


def test_middle_of_the_scale_abstains():
    """A 3 on a five-point scale is the rater saying they could not tell.

    Collapsing at a single midpoint would turn every ambiguous window into a
    confident on or off, which is the opposite of what the scale is saying.
    """
    assert to_label(5.0) == "on"
    assert to_label(4.0) == "on"
    assert to_label(3.0) is None
    assert to_label(2.0) == "off"
    assert to_label(1.0) == "off"


def test_windows_carry_subject_identity(tmp_path: Path):
    """Targets must name their subject so the split can be by student.

    Overlapping windows from one student are correlated; splitting at random
    would leak the same student into train and test.
    """
    _subject(tmp_path, "experiment_a", "subject_7", {
        "expert1": [(f"00:00:{s:02d}:000000", 5) for s in range(1, 20)],
    })
    windows = list(iter_labelled_windows(tmp_path))
    assert windows
    assert all(isinstance(w, LabelledWindow) for w in windows)
    assert {w.subject for w in windows} == {"subject_7"}
    assert {w.experiment for w in windows} == {"experiment_a"}


def test_thin_windows_are_skipped(tmp_path: Path):
    """A target averaged from one stray rating is not a target."""
    _subject(tmp_path, "experiment_a", "subject_1", {
        "expert1": [("00:00:01:000000", 5)],
    })
    assert list(iter_labelled_windows(tmp_path, min_ratings=3)) == []


def test_rater_spread_exposes_disagreement(tmp_path: Path):
    """Two experts three points apart must not look like agreement."""
    _subject(tmp_path, "experiment_a", "subject_1", {
        "expert1": [(f"00:00:{s:02d}:000000", 5) for s in range(1, 6)],
        "expert2": [(f"00:00:{s:02d}:000000", 1) for s in range(1, 6)],
    })
    windows = list(iter_labelled_windows(tmp_path))
    assert windows
    first = windows[0]
    assert first.rater_spread == pytest.approx(4.0)
    # Means to 3.0, which is the ambiguous band -- exactly right for a window
    # two experts read in opposite directions.
    assert first.label is None


def test_single_rater_has_zero_spread(tmp_path: Path):
    _subject(tmp_path, "experiment_a", "subject_1", {
        "expert1": [(f"00:00:{s:02d}:000000", 4) for s in range(1, 6)],
    })
    windows = list(iter_labelled_windows(tmp_path))
    assert windows[0].rater_spread == 0.0
    assert windows[0].label == "on"


def test_summarise_reports_class_balance(tmp_path: Path):
    """Class balance has to be visible before anyone trains on this."""
    _subject(tmp_path, "experiment_a", "subject_1", {
        "expert1": [(f"00:00:{s:02d}:000000", 5) for s in range(1, 20)],
    })
    windows = list(iter_labelled_windows(tmp_path))
    report = summarise(windows)
    assert report["windows"] == len(windows)
    assert report["subjects"] == 1
    assert report["by_label"]["on"] == len(windows)


def test_summarise_handles_nothing():
    assert summarise([])["windows"] == 0


def test_rating_is_hashable_and_ordered(tmp_path: Path):
    """Ratings come back in time order, so windowing can scan them once."""
    _subject(tmp_path, "experiment_a", "subject_1", {
        "expert1": [("00:00:09:000000", 2), ("00:00:01:000000", 4)],
    })
    ratings = read_ratings(tmp_path / "experiment_a" / "subject_1")
    assert [r.time_ms for r in ratings] == sorted(r.time_ms for r in ratings)
    assert isinstance(ratings[0], Rating)
