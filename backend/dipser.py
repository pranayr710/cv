"""Read DIPSER's attention labels and align them to our engagement windows.

DIPSER (Marquez-Carpintero et al., CC BY 4.0) records in-person classroom
sessions and labels each student's attention on a 1-5 scale, from four expert
raters plus the student's own self-assessment. That makes it the closest public
substitute for hand-labelling our own footage: it is a real classroom with
several students, so the room-layout features stay meaningful, and its labels
are finer-grained than our 15-second windows rather than coarser.

Layout, taken from the authors' own loader rather than guessed::

    <root>/
      experiment_<name>/
        subject_<id>/
          images/                      HH_MM_SS_ffffff.png
          labels/
            <rater>.json               [{"datetime": "HH:MM:SS:ffffff",
                                         "attention": 1-5,
                                         "emotion": "..."}, ...]

Two details that matter and are easy to get wrong:

* The image filename IS the timestamp, with underscores standing in for colons.
* Each label file is one RATER, not one session. Several files per subject means
  several opinions on the same instant, and that disagreement is information --
  it is kept per rater here rather than averaged away at read time.

What this module does NOT do is produce features. The feature vector comes from
our own pipeline (:mod:`backend.engagement_features`), so using DIPSER means
running our detectors over its image sequences and pairing the result with the
attention target this module extracts. Whether our detectors hold up on their
camera geometry is an open question that has to be measured, not assumed.
"""
from __future__ import annotations

import json
import statistics
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

#: Window length and stride, matching backend.engagement_features so a DIPSER
#: target lines up with a feature vector computed over the same span.
WINDOW_MS: int = 15_000
STRIDE_MS: int = 7_500

#: DIPSER attention runs 1 (not attending) to 5 (fully attending).
ATTENTION_MIN, ATTENTION_MAX = 1, 5

#: Mean attention at or above this reads as on-task; at or below OFF_AT reads as
#: off-task. The band between them is returned as ``None``.
#:
#: The gap is deliberate. Our own labelling tool offers "unclear" as a third
#: answer precisely so an ambiguous window is excluded rather than decided by a
#: coin flip, and collapsing a 1-5 scale at a single midpoint would do the
#: opposite -- it would turn every genuinely middling window into a confident
#: on or off. A 3.0 on a five-point scale is the rater saying they could not
#: tell, and that should survive into the target.
ON_AT: float = 3.5
OFF_AT: float = 2.5

#: How far a label may be carried forward to cover a frame that has none.
#: The authors' loader forward-fills without limit, which silently stretches one
#: judgement across an arbitrary gap. Attention is not that stable, so a gap
#: wider than this leaves the frame unlabelled instead.
MAX_CARRY_MS: int = 2_000


@dataclass(frozen=True)
class Rating:
    """One rater's attention score at one instant."""

    rater: str
    time_ms: int
    attention: int


@dataclass(frozen=True)
class LabelledWindow:
    """A DIPSER attention target over one window, for one subject.

    Attributes:
        experiment: The experiment folder name.
        subject: The subject folder name.
        start_ms: Window start, milliseconds since midnight.
        end_ms: Window end.
        mean_attention: Mean over every rating in the window, across raters.
        per_rater_mean: Each rater's own mean over the window, so disagreement
            stays visible rather than being flattened into one number.
        n_ratings: How many individual ratings the mean is built from.
        label: ``"on"``, ``"off"``, or ``None`` when the mean falls in the
            ambiguous band between :data:`OFF_AT` and :data:`ON_AT`.
    """

    experiment: str
    subject: str
    start_ms: int
    end_ms: int
    mean_attention: float
    per_rater_mean: dict[str, float]
    n_ratings: int
    label: str | None

    @property
    def rater_spread(self) -> float:
        """Max minus min of the per-rater means; 0.0 with a single rater.

        A window where experts disagree by three points on a five-point scale
        is not the same evidence as one where they agree, and a model trained
        as though it were would be learning from noise it could have excluded.
        """
        if len(self.per_rater_mean) < 2:
            return 0.0
        values = self.per_rater_mean.values()
        return max(values) - min(values)


def parse_time_ms(stamp: str) -> int | None:
    """Milliseconds since midnight from DIPSER's ``HH:MM:SS:ffffff``.

    Args:
        stamp: A timestamp as it appears in a label's ``datetime`` field, or an
            image filename with underscores already replaced by colons.

    Returns:
        Milliseconds since midnight, or ``None`` if the string is malformed.
    """
    parts = stamp.strip().split(":")
    if len(parts) < 3:
        return None
    fraction = parts[3] if len(parts) > 3 else ""
    try:
        hours, minutes, seconds = int(parts[0]), int(parts[1]), int(parts[2])
        # The fractional field is microseconds written left-aligned, so "5"
        # means 500000, not 5. Reading it as written would collapse every
        # sub-second timestamp onto the same instant.
        micros = int(fraction.ljust(6, "0")[:6]) if fraction else 0
    except ValueError:
        return None
    return ((hours * 3600 + minutes * 60 + seconds) * 1000) + micros // 1000


def time_ms_from_image(path: Path) -> int | None:
    """The timestamp encoded in an image filename.

    DIPSER names frames ``HH_MM_SS_ffffff.png``; the underscores are colons.
    """
    return parse_time_ms(path.stem.replace("_", ":"))


def read_ratings(subject_dir: Path) -> list[Rating]:
    """Every rating for one subject, across all rater files.

    Args:
        subject_dir: A ``subject_<id>`` directory containing a ``labels``
            subdirectory.

    Returns:
        Ratings sorted by time. Entries without an ``attention`` field are
        skipped -- DIPSER's tags may carry only ``emotion``.
    """
    labels_dir = subject_dir / "labels"
    if not labels_dir.is_dir():
        return []

    ratings: list[Rating] = []
    for label_file in sorted(labels_dir.glob("*.json")):
        try:
            tags = json.loads(label_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(tags, list):
            continue
        rater = label_file.stem
        for tag in tags:
            if not isinstance(tag, dict) or "attention" not in tag:
                continue
            time_ms = parse_time_ms(str(tag.get("datetime", "")))
            if time_ms is None:
                continue
            try:
                attention = int(tag["attention"])
            except (TypeError, ValueError):
                continue
            if ATTENTION_MIN <= attention <= ATTENTION_MAX:
                ratings.append(Rating(rater, time_ms, attention))
    ratings.sort(key=lambda r: r.time_ms)
    return ratings


def to_label(mean_attention: float,
             on_at: float = ON_AT,
             off_at: float = OFF_AT) -> str | None:
    """Collapse a 1-5 mean onto our on/off verdict, or abstain."""
    if mean_attention >= on_at:
        return "on"
    if mean_attention <= off_at:
        return "off"
    return None


def iter_labelled_windows(root: Path,
                          window_ms: int = WINDOW_MS,
                          stride_ms: int = STRIDE_MS,
                          min_ratings: int = 3) -> Iterator[LabelledWindow]:
    """Walk a DIPSER root and yield an attention target per subject-window.

    Args:
        root: The extracted dataset root, containing ``experiment_*`` folders.
        window_ms: Window length, matched to our feature windows.
        stride_ms: Distance between window starts.
        min_ratings: Windows backed by fewer ratings than this are skipped;
            a target averaged from one stray rating is not a target.

    Yields:
        :class:`LabelledWindow` per subject per span, in time order.
    """
    for experiment_dir in sorted(p for p in root.glob("experiment*") if p.is_dir()):
        for subject_dir in sorted(p for p in experiment_dir.glob("subject*")
                                  if p.is_dir()):
            ratings = read_ratings(subject_dir)
            if not ratings:
                continue
            start = ratings[0].time_ms
            last = ratings[-1].time_ms
            while start <= last:
                end = start + window_ms
                inside = [r for r in ratings if start <= r.time_ms < end]
                if len(inside) >= min_ratings:
                    per_rater: dict[str, list[int]] = {}
                    for r in inside:
                        per_rater.setdefault(r.rater, []).append(r.attention)
                    per_rater_mean = {k: statistics.fmean(v)
                                      for k, v in per_rater.items()}
                    mean_attention = statistics.fmean(r.attention for r in inside)
                    yield LabelledWindow(
                        experiment=experiment_dir.name,
                        subject=subject_dir.name,
                        start_ms=start,
                        end_ms=end,
                        mean_attention=mean_attention,
                        per_rater_mean=per_rater_mean,
                        n_ratings=len(inside),
                        label=to_label(mean_attention),
                    )
                start += stride_ms


def summarise(windows: list[LabelledWindow]) -> dict[str, object]:
    """Counts a caller can print before committing to training on this.

    Returns:
        Totals by label, subject and rater count, plus how many windows the
        raters disagreed sharply on. Reported up front because a target set
        that is 90% one class, or one where raters routinely differ by three
        points, is not worth building a model on.
    """
    if not windows:
        return {"windows": 0}
    by_label: dict[str, int] = {}
    for w in windows:
        key = w.label or "ambiguous"
        by_label[key] = by_label.get(key, 0) + 1
    multi = [w for w in windows if len(w.per_rater_mean) > 1]
    wide = [w for w in multi if w.rater_spread >= 2.0]
    return {
        "windows": len(windows),
        "subjects": len({(w.experiment, w.subject) for w in windows}),
        "by_label": by_label,
        "mean_attention": round(
            statistics.fmean(w.mean_attention for w in windows), 3),
        "multi_rater_windows": len(multi),
        "rater_spread_2_or_more": len(wide),
    }
