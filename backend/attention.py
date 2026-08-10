"""Windowed, per-student attention signal — exploratory Stage 3 first slice.

Operates on the Stage 1+2 JSONL output (``gaze_label``, EAR, posture presence,
``objects``, ``track_id``) after the fact. Nothing here is wired into
``schema.json`` or the live capture loop in ``integrate.py`` — this reads a
finished JSONL file, not a live frame.

Motivation: the raw per-frame contract Stage 1 emits is exactly right for
Stage 1's job (perception), but four research findings, gathered specifically
to answer this, say that scoring attention frame-by-frame is the wrong unit
of analysis:

* A single ~2-second break was shown to *eliminate* vigilance decline over a
  50-minute task rather than being a symptom of it (Ariga & Lleras, 2011).
  Flagging brief lapses would penalise the mechanism that protects sustained
  attention, not detect its failure.
* Minds wander roughly a quarter to half of waking hours in the general
  population (Killingsworth & Gilbert, 2010) — at any instant, most of an
  attentive classroom is already doing this. A per-frame verdict would flag
  almost everyone, almost always.
* Real gaze-based mind-wandering detectors built for lecture footage get
  their best results aggregating features over roughly a 12-second window,
  not single frames (Faber, Bixler & D'Mello).
* Most self-reported classroom attention lapses last under a minute and get
  *shorter and more frequent* as a lecture continues — attention oscillates
  in quick cycles, it does not decay along one long curve (Bunce, Flens &
  Neiles, 2010).

Full sourcing for all of the above lives in the "Reading the Room" research
artifact this module implements decisions from.

Grounding the taxonomy itself, not just the windowing: a field-wide critical
review (Khan, Abedi & Colella, 2022/23) found most engagement-detection
systems invent their own ad hoc category scheme rather than grounding it in
a validated educational-psychology instrument -- exactly the failure mode to
avoid here. A follow-up research pass checked ORIENTATION (the six
categories below) against real, published classroom-observation instruments
from school psychology, rather than treating "we made these up but they
sound reasonable" as good enough:

* BOSS (Behavioral Observation of Students in Schools -- Shapiro; reviewed
  alongside six sibling instruments in Volpe, DiPerna, Hintze & Shapiro,
  2005, School Psychology Review 34(4)) is a published, validated,
  individual-student momentary-time-sampling instrument with almost exactly
  this shape: Active/Passive Engaged Time (on-task) vs. Off-task
  Motor/Verbal/Passive. Its standard sampling interval is 15 seconds, with
  dedicated methodological work (Zakszeski, Hojnoski & Wood, 2017) finding
  shorter momentary intervals (5-15s) track continuous observation most
  accurately -- an independent confirmation of this module's
  ``window_seconds = 15.0`` default that has nothing to do with the
  mind-wandering literature above, which is worth having two unrelated
  sources agree on.
* DBR (Direct Behavior Rating -- Chafouleas, Riley-Tillman et al.) defines
  "Academic Engagement" as essentially BOSS's two on-task categories
  collapsed into one, rated per observation period rather than per interval
  -- a weaker direct fit for frame-level output, useful mainly as a
  convergent secondary reference.
* CLASS (Pianta, La Paro & Hamre) was also checked and is a poor fit for
  THIS taxonomy specifically: its engagement dimension is a single holistic
  rating for the whole classroom over 15-25 minutes, not a per-student
  behavioural code -- the right instrument if this project ever adds
  classroom-climate features, the wrong one for what this module does.

Proposed mapping (a recommendation for the team to review, not a renaming
already applied to the code below -- category identifiers are unchanged):

    ClassGraph category      BOSS category                          Fit
    attending_teacher    ->  Passive Engaged Time                   clean but partial
    oriented_away         -> Off-task Passive (if sustained, and    forced -- BOSS's
                              not sanctioned peer talk)               own exception
                                                                       needs audio too
    head_down_with_device -> Off-task Motor                         reasonably clean
    head_down_writing     -> Active Engaged Time                    clean when the
                                                                     book is detected
    head_down_no_device   -> ambiguous: Active Engaged Time         NO clean mapping
                              (silent reading/writing) OR             exists -- see below
                              Off-task Passive
    posture_only          -> no BOSS analogue (CV artifact)         n/a
    no_signal              -> no BOSS analogue (CV artifact)         n/a

``head_down_writing`` was added after a reviewer asked the obvious question:
if a student is writing in a book with a pen, the system should say so. Before
it existed, a studying student and a disengaged one both landed in
``head_down_no_device`` and the pipeline could not credit anyone for working.
A detected book near a bowed head is the one piece of positive evidence
available in the existing schema, so it gets its own on-task category.

Its limit is the book detector, not the logic: COCO's ``book`` class was
trained on bookshelves and closed books, not an open notebook seen from an
elevated classroom camera, so it misses. Every miss lands the student back in
``head_down_no_device`` -- an under-count of engagement, never an over-count,
which is the right direction for this error to run. Fine-tuning on
SCB-Dataset's ``write``/``read`` labels is the real fix if the threshold
adjustment in ``DetectionConfig.object_conf_per_class`` is not enough.

The remaining unresolved cell, ``head_down_no_device``, is not a gap unique to
this project: BOSS's own two hardest categories to tell apart by eye are
exactly "quiet, head down, reading/writing" (on-task) versus "quiet, head
down, spaced out" (off-task) -- a trained human observer resolves it by
watching for sustained duration and context a bounding box cannot see
either. Leaving this bucket ambiguous mirrors the established literature's
own unresolved difficulty; forcing a confident split here would be less
honest than the established instrument itself is willing to be. Off-task
Verbal (BOSS) has no equivalent here at all -- it is defined by audible
sound, which this vision-only pipeline does not have access to; this is a
genuine, permanent gap, not an oversight.

Not implemented yet, and worth naming rather than quietly omitting: pairing
the book with a **wrist keypoint** would separate "writing" from "a book is
merely open on the desk". :mod:`backend.posture` currently extracts only
nose/shoulder/hip landmarks, so the wrists are not available without extending
that module and the schema again. Deferred deliberately, not overlooked.

What this module does NOT claim:

* It does not detect peer interaction. Kendon's F-formation research gives a
  real geometric definition (reciprocal, sustained body orientation between
  two people), but detecting it needs pairing across tracked students, which
  this first slice does not implement. ``gaze_label`` "left"/"right" is
  reported as its own ambiguous "oriented_away" bucket, never folded into an
  off-task count — collapsing "turned toward a neighbour" into "distracted"
  is exactly the mistake the CSCL literature warns against (a productive
  academic discussion and idle chat look identical from vision alone; the
  field's own answer, when it needed that distinction, was to add a
  microphone, not a smarter camera heuristic).
* It does not treat "gaze down"/"gaze back" alone as off-task. Gaze aversion
  during effortful thinking is a documented, opposite-reading confound
  (Doherty-Sneddon et al.) — a bowed head is equally consistent with reading
  or writing. It is only treated as a meaningful signal alongside a detected
  "cell phone" nearby, which is the one case with a defensible reading in
  the existing schema.
* It never produces a bare "engaged"/"disengaged" verdict for an individual.
  ``summarise_classroom`` — the class-level, aggregate view — is this
  module's default output; a single student's data is available only by
  explicitly asking for one track_id, mirroring the "individual data is a
  drill-down, not the default view" guardrail from the ethics research.
* It does not use the word "emotion" anywhere, and never will in this
  module — behavioural/geometric signal only.

Usage (library):
    from backend.attention import RollingAttentionTracker, iter_jsonl_signals
    tracker = RollingAttentionTracker()
    for track_id, timestamp_ms, signal in iter_jsonl_signals("stage1.jsonl"):
        tracker.update(track_id, timestamp_ms, signal)
    print(tracker.summarise_classroom())

Usage (CLI):
    python -m backend.attention --jsonl outputs/stage1.jsonl
    python -m backend.attention --jsonl outputs/stage1.jsonl --student 3
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from backend.config import CONFIG, AttentionConfig

logger = logging.getLogger(__name__)

Bbox = tuple[float, float, float, float]

# The full set of per-frame orientation categories. Deliberately not an
# "engaged"/"disengaged" scale -- see the module docstring for why each
# ambiguous case is its own bucket rather than collapsed into one.
Orientation = Literal[
    "attending_teacher",
    "oriented_away",
    "head_down_with_device",
    "head_down_writing",
    "head_down_no_device",
    "posture_only",
    "no_signal",
]

ALL_ORIENTATIONS: tuple[Orientation, ...] = (
    "attending_teacher",
    "oriented_away",
    "head_down_with_device",
    "head_down_writing",
    "head_down_no_device",
    "posture_only",
    "no_signal",
)


@dataclass(frozen=True)
class FrameSignal:
    """One person's classified visible behaviour for one frame.

    Attributes:
        orientation: Which of :data:`ALL_ORIENTATIONS` this frame falls into.
        eyes_closed: Whether EAR was below :data:`FaceConfig.ear_closed_threshold`,
            or ``None`` when no face/EAR was available this frame. Tracked
            separately from ``orientation`` because it can co-occur with any
            gaze label and is a distinct signal (possible drowsiness) from
            gaze direction.
    """

    orientation: Orientation
    eyes_closed: bool | None


def _bbox_overlaps(a: Bbox, b: Bbox, min_iou: float) -> bool:
    """Whether two ``(x, y, w, h)`` boxes overlap by at least ``min_iou``.

    Args:
        a: First box.
        b: Second box.
        min_iou: Minimum intersection-over-union to count as overlapping.
            ``0.0`` means "any positive overlap counts."

    Returns:
        ``True`` if the boxes overlap enough to satisfy ``min_iou``.
    """
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    inter_w = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    inter_h = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = inter_w * inter_h
    if inter <= 0.0:
        return False  # zero overlap never counts, regardless of min_iou
    if min_iou <= 0.0:
        return True  # any positive overlap satisfies a zero threshold
    union = aw * ah + bw * bh - inter
    return union > 0 and (inter / union) > min_iou


def classify_frame(
    person: dict,
    objects: list[dict],
    config: AttentionConfig | None = None,
    ear_closed_threshold: float | None = None,
) -> FrameSignal:
    """Classify one person's visible behaviour in one frame.

    Reads only fields already in the frozen schema (``face``, ``head_pose``,
    ``posture``) plus the frame's ``objects`` list -- nothing new is inferred
    beyond combining what Stage 1/1B already output.

    Args:
        person: One entry from a JSONL record's ``persons`` list.
        objects: That record's ``objects`` list.
        config: Attention settings. Defaults to ``CONFIG.attention``.
        ear_closed_threshold: Overrides ``CONFIG.face.ear_closed_threshold``
            when set (kept as a parameter, not a silent import, so this stays
            a pure function of its inputs).

    Returns:
        The classified :class:`FrameSignal` for this person this frame.

    Raises:
        KeyError: If ``person`` is missing a required schema field.
    """
    cfg = config if config is not None else CONFIG.attention
    ear_closed = (
        ear_closed_threshold
        if ear_closed_threshold is not None
        else CONFIG.face.ear_closed_threshold
    )

    face = person["face"]
    head_pose = person["head_pose"]
    posture = person["posture"]

    eyes_closed: bool | None = None
    if face is not None and face["ear"] is not None:
        eyes_closed = face["ear"] < ear_closed

    if head_pose is None:
        orientation: Orientation = (
            "posture_only" if posture is not None else "no_signal"
        )
        return FrameSignal(orientation=orientation, eyes_closed=eyes_closed)

    gaze = head_pose["gaze_label"]
    if gaze == "teacher":
        return FrameSignal(orientation="attending_teacher", eyes_closed=eyes_closed)
    if gaze not in cfg.device_gaze_labels:
        # "left" / "right": ambiguous, possibly peer interaction -- see
        # module docstring. Never counted toward an off-task signal.
        return FrameSignal(orientation="oriented_away", eyes_closed=eyes_closed)

    person_bbox: Bbox = tuple(person["bbox"])  # type: ignore[assignment]

    def _near(classes: tuple[str, ...]) -> bool:
        """Whether any object of these classes overlaps this person's box."""
        return any(
            obj["cls"] in classes
            and _bbox_overlaps(
                person_bbox, tuple(obj["bbox"]), cfg.device_proximity_iou
            )
            for obj in objects
        )

    # A bowed head means different things depending on what is under it. Phone
    # is checked first: when both a phone and a book are detected near the same
    # student the evidence is contradictory, and the more concerning reading is
    # the safer default -- crediting a student as "working" on the strength of a
    # book that happens to be open on the desk would be the easier error to make
    # and the worse one to make.
    if _near(cfg.device_object_classes):
        orientation: Orientation = "head_down_with_device"
    elif _near(cfg.writing_object_classes):
        orientation = "head_down_writing"
    else:
        # Still genuinely ambiguous: a bowed head with nothing detected near it
        # is equally consistent with reading, writing on loose paper, or
        # disengagement. Kept as its own bucket rather than guessed at -- and
        # note book detection is the weakest link feeding the branch above, so
        # some students who *are* writing land here.
        orientation = "head_down_no_device"
    return FrameSignal(orientation=orientation, eyes_closed=eyes_closed)


def iter_jsonl_signals(
    path: str | Path, config: AttentionConfig | None = None
) -> Iterator[tuple[int, int, FrameSignal]]:
    """Read a Stage 1+2 JSONL file and yield ``(track_id, timestamp_ms, signal)``.

    Persons with ``track_id is None`` (an unconfirmed track this frame — see
    ``backend.tracking``) are skipped: there is no identity to aggregate them
    under.

    Args:
        path: Path to a JSONL file matching ``schema.json``.
        config: Attention settings. Defaults to ``CONFIG.attention``.

    Yields:
        One tuple per tracked person per frame, in file order.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    src = Path(path)
    if not src.is_file():
        raise FileNotFoundError(f"JSONL file not found: {src}")

    with src.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            timestamp_ms = record["timestamp_ms"]
            objects = record["objects"]
            for person in record["persons"]:
                if person["track_id"] is None:
                    continue
                signal = classify_frame(person, objects, config)
                yield person["track_id"], timestamp_ms, signal


@dataclass
class _TrackState:
    """Internal per-track rolling state (module-private)."""

    history: deque = field(default_factory=deque)
    calibration_start_ms: int | None = None
    calibration_attending: int = 0
    calibration_total: int = 0
    calibration_baseline: float | None = None
    off_task_streak_start_ms: int | None = None


class RollingAttentionTracker:
    """Maintains a rolling per-track window of :class:`FrameSignal` history.

    One instance covers one video/session. Feed it every ``(track_id,
    timestamp_ms, signal)`` in chronological order via :meth:`update`.

    Attributes:
        config: The :class:`AttentionConfig` in effect.
    """

    def __init__(self, config: AttentionConfig | None = None) -> None:
        """Create an empty tracker.

        Args:
            config: Attention settings. Defaults to ``CONFIG.attention``.
        """
        self.config: AttentionConfig = (
            config if config is not None else CONFIG.attention
        )
        self._tracks: dict[int, _TrackState] = {}

    def update(self, track_id: int, timestamp_ms: int, signal: FrameSignal) -> None:
        """Record one frame's signal for one tracked student.

        Args:
            track_id: The Stage 2 track id (must not be ``None`` — filter
                those out before calling, e.g. via :func:`iter_jsonl_signals`).
            timestamp_ms: Frame timestamp in milliseconds. Must be
                non-decreasing per track_id (chronological input).
            signal: This frame's classified behaviour for this student.
        """
        state = self._tracks.setdefault(track_id, _TrackState())
        state.history.append((timestamp_ms, signal))

        window_start = timestamp_ms - int(self.config.window_seconds * 1000)
        while state.history and state.history[0][0] < window_start:
            state.history.popleft()

        if state.calibration_start_ms is None:
            state.calibration_start_ms = timestamp_ms
        if state.calibration_baseline is None:
            elapsed = timestamp_ms - state.calibration_start_ms
            state.calibration_total += 1
            if signal.orientation == "attending_teacher":
                state.calibration_attending += 1
            if (
                elapsed >= self.config.calibration_seconds * 1000
                and state.calibration_total > 0
            ):
                state.calibration_baseline = (
                    state.calibration_attending / state.calibration_total
                )

        window_counts = self._window_counts(state)
        window_total = sum(window_counts.values())
        off_task_now = (
            window_total > 0
            and window_counts.get("head_down_with_device", 0) / window_total
            >= self.config.off_task_majority_fraction
        )
        if off_task_now:
            if state.off_task_streak_start_ms is None:
                state.off_task_streak_start_ms = timestamp_ms
        else:
            state.off_task_streak_start_ms = None

    @staticmethod
    def _window_counts(state: _TrackState) -> dict[Orientation, int]:
        counts: dict[Orientation, int] = {o: 0 for o in ALL_ORIENTATIONS}
        for _, signal in state.history:
            counts[signal.orientation] += 1
        return counts

    def window_distribution(self, track_id: int) -> dict[Orientation, float]:
        """Fraction of the current rolling window in each orientation category.

        Args:
            track_id: A track previously passed to :meth:`update`.

        Returns:
            A mapping from every :data:`ALL_ORIENTATIONS` value to its
            fraction of frames in the current window (``0.0`` for a category
            with no frames). All-zero if the track is unknown or has no
            history in the current window.
        """
        state = self._tracks.get(track_id)
        if state is None or not state.history:
            return {o: 0.0 for o in ALL_ORIENTATIONS}
        counts = self._window_counts(state)
        total = sum(counts.values())
        return {o: (counts[o] / total if total else 0.0) for o in ALL_ORIENTATIONS}

    def window_eyes_closed_ratio(self, track_id: int) -> float | None:
        """Fraction of window frames with EAR below the closed-eye threshold.

        Args:
            track_id: A track previously passed to :meth:`update`.

        Returns:
            The ratio, or ``None`` if no frame in the window had EAR data
            (e.g. no face was ever visible in this window).
        """
        state = self._tracks.get(track_id)
        if state is None:
            return None
        judged = [s.eyes_closed for _, s in state.history if s.eyes_closed is not None]
        if not judged:
            return None
        return sum(judged) / len(judged)

    def is_sustained_device_distraction(self, track_id: int) -> bool:
        """Whether this track has been majority phone-while-head-down for
        at least :data:`AttentionConfig.sustained_seconds`, continuously.

        Args:
            track_id: A track previously passed to :meth:`update`.

        Returns:
            ``True`` only once the current off-task streak has lasted at
            least ``sustained_seconds`` — a single missed glance-back does
            not trip this (see :data:`AttentionConfig.sustained_seconds`'s
            docstring for the research behind that threshold).
        """
        state = self._tracks.get(track_id)
        if state is None or state.off_task_streak_start_ms is None or not state.history:
            return False
        now_ms = state.history[-1][0]
        elapsed = now_ms - state.off_task_streak_start_ms
        return elapsed >= self.config.sustained_seconds * 1000

    def personal_baseline(self, track_id: int) -> float | None:
        """This student's own baseline "attending_teacher" rate.

        Computed once from their first :data:`AttentionConfig.calibration_seconds`
        of data. Per-student calibration was the one concrete, literature-
        measured accuracy lever found in this research (+0.084 AUC in a real
        classroom deployment) — see the module docstring.

        Args:
            track_id: A track previously passed to :meth:`update`.

        Returns:
            The baseline rate in ``[0, 1]``, or ``None`` if fewer than
            ``calibration_seconds`` of data have been seen for this track yet.
        """
        state = self._tracks.get(track_id)
        return None if state is None else state.calibration_baseline

    def deviation_from_baseline(self, track_id: int) -> float | None:
        """Current window's attending rate minus this student's own baseline.

        Args:
            track_id: A track previously passed to :meth:`update`.

        Returns:
            A signed value: negative means currently below this student's
            own typical baseline, positive means above it. ``None`` until a
            baseline exists (see :meth:`personal_baseline`).
        """
        baseline = self.personal_baseline(track_id)
        if baseline is None:
            return None
        current = self.window_distribution(track_id)["attending_teacher"]
        return current - baseline

    def known_track_ids(self) -> list[int]:
        """All track ids seen so far, in first-seen order."""
        return list(self._tracks.keys())

    def summarise_classroom(self) -> dict[str, object]:
        """Class-level aggregate across every known track.

        This — not a per-student score — is the intended default view.
        Individual students are a deliberate drill-down via
        :meth:`window_distribution`/:meth:`personal_baseline` for one
        track_id, not the default output, per the "never a bare individual
        verdict" guardrail from the ethics research this implements.

        Returns:
            A dict with ``student_count``, the classroom-averaged
            ``distribution`` across :data:`ALL_ORIENTATIONS`, and
            ``sustained_device_distraction_count`` (how many students are
            currently flagged, not which ones).
        """
        ids = self.known_track_ids()
        if not ids:
            return {
                "student_count": 0,
                "distribution": {o: 0.0 for o in ALL_ORIENTATIONS},
                "sustained_device_distraction_count": 0,
            }
        per_student = [self.window_distribution(t) for t in ids]
        distribution = {
            o: sum(d[o] for d in per_student) / len(per_student)
            for o in ALL_ORIENTATIONS
        }
        sustained = sum(1 for t in ids if self.is_sustained_device_distraction(t))
        return {
            "student_count": len(ids),
            "distribution": distribution,
            "sustained_device_distraction_count": sustained,
        }


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser.

    Returns:
        A configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog="python -m backend.attention",
        description=(
            "Summarise a Stage 1+2 JSONL file into a windowed, class-level "
            "attention signal. Defaults to a classroom-level view; pass "
            "--student to drill into one tracked student."
        ),
    )
    parser.add_argument(
        "--jsonl", required=True, type=str, help="Path to a stage1 JSONL file."
    )
    parser.add_argument(
        "--student",
        type=int,
        default=None,
        help="Track id to drill into, instead of the classroom-level summary.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=CONFIG.log_level,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]`` when ``None``.

    Returns:
        Process exit code (``0`` on success, ``1`` on a handled failure).
    """
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(
        level=args.log_level, format="%(asctime)s %(levelname)-8s %(name)s: %(message)s"
    )

    try:
        tracker = RollingAttentionTracker()
        last_seen: dict[int, int] = {}
        for track_id, timestamp_ms, signal in iter_jsonl_signals(args.jsonl):
            tracker.update(track_id, timestamp_ms, signal)
            last_seen[track_id] = timestamp_ms
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as exc:
        logger.error("Failed to process %s: %s", args.jsonl, exc)
        return 1

    if not last_seen:
        print("No tracked persons found in this file.")
        return 0

    if args.student is not None:
        if args.student not in last_seen:
            logger.error("Track id %d was never seen in this file.", args.student)
            return 1
        dist = tracker.window_distribution(args.student)
        baseline = tracker.personal_baseline(args.student)
        deviation = tracker.deviation_from_baseline(args.student)
        eyes_closed = tracker.window_eyes_closed_ratio(args.student)
        print(
            f"Student track_id={args.student} — last {CONFIG.attention.window_seconds:.0f}s window"
        )
        for o in ALL_ORIENTATIONS:
            print(f"  {o:<24} {dist[o] * 100:5.1f}%")
        print(
            f"  eyes closed (of frames w/ EAR): {eyes_closed * 100:.1f}%"
            if eyes_closed is not None
            else "  eyes closed: no EAR data"
        )
        print(
            f"  personal baseline (attending):  {baseline * 100:.1f}%"
            if baseline is not None
            else "  personal baseline: not yet calibrated"
        )
        print(
            f"  deviation from own baseline:    {deviation * 100:+.1f} pts"
            if deviation is not None
            else ""
        )
        print(
            f"  sustained device distraction:   {tracker.is_sustained_device_distraction(args.student)}"
        )
    else:
        summary = tracker.summarise_classroom()
        print(
            f"Classroom summary — {summary['student_count']} tracked students, last {CONFIG.attention.window_seconds:.0f}s window each"
        )
        for o in ALL_ORIENTATIONS:
            print(f"  {o:<24} {summary['distribution'][o] * 100:5.1f}%")
        print(
            f"  students w/ sustained device distraction: {summary['sustained_device_distraction_count']} / {summary['student_count']}"
        )

    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via CLI
    import sys

    sys.exit(main())
