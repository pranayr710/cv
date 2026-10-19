"""Turn a student's time window into a fixed-length feature vector.

The engagement verdict is currently a precedence rule: off-task routes are
checked, then on-task routes, and the first match wins (see
:func:`backend.engagement.classify_engagement`). The thresholds and the
ordering were set by hand and defended from the BOSS literature, but they were
never fitted to data.

This module is the bridge to fitting them. It does NOT replace the rules --
it consumes what the rules and the perception models already produce and
summarises a window of frames into one vector, so a small classifier can learn
the *weighting* that the precedence ordering currently hard-codes.

Design note. The rules stay as feature extractors rather than being discarded,
for three reasons:

* They encode real prior knowledge (BOSS's "behaviour overrides orientation"),
  which 343 labelled windows cannot rediscover on their own.
* A learned model over these features keeps the abstention property: coverage
  is itself a feature, so a window with almost no readable frames can be
  refused rather than scored from nothing.
* Feature importances over named features are inspectable. A panel can be told
  which signal the model leans on; a black box over raw pixels could not.

Nothing here is trained. It produces X; the labels are y, and y has to come
from a human -- deriving labels from the same rules the model would then learn
is circular and would measure nothing.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

#: Window length in milliseconds. Matches backend.attention's rolling window so
#: a learned score and the rule score describe the same span of time.
WINDOW_MS: int = 15_000

#: Step between window starts. Half the window, so consecutive examples overlap
#: by 50% -- roughly doubles the sample count from a fixed recording, at the
#: cost of correlated examples that must be split by STUDENT, never at random.
STRIDE_MS: int = 7_500

#: Below this fraction of frames carrying any usable signal, the window is not
#: scored at all. The rule pipeline abstains on thin evidence and a learned
#: scorer must be able to do the same, so coverage is both a gate and a feature.
MIN_COVERAGE: float = 0.30

#: Actions counted as off-task, mirroring backend.actions.OFF_TASK so the
#: learned features and the rule verdict describe the same behaviours.
OFF_TASK_ACTIONS = frozenset({
    "on_phone", "eyes_closed", "head_down", "looking_away", "slouching",
    "yawning", "drinking", "eating",
})

#: Actions counted as on-task.
ON_TASK_ACTIONS = frozenset({
    "attentive", "writing", "reading", "studying", "typing", "raising_hand",
    "on_laptop", "leaning_forward",
})

#: The feature names, in the fixed order :func:`window_features` emits them.
#: Named rather than positional because feature importances are only useful if
#: the importance can be attached to a word.
#:
#: n_frames was removed after it broke the model on live webcam input. It was a
#: raw count, so it scaled with frame rate: the batch pipeline samples at 3 fps
#: and produced windows of about 41 frames, while a webcam session produced
#: 130. Standardised against the training mean that is a shift of nearly 13
#: standard deviations, on a feature carrying no information the model needed
#: -- coverage already expresses how full a window is, normalised. Every
#: student in a live session scored 1/10 because of it.
BASE_FEATURE_NAMES: tuple[str, ...] = (
    "coverage",              # fraction of frames with any readable signal
    "frac_on_task_action",   # fraction of read frames whose action is on-task
    "frac_off_task_action",  # ... off-task
    "frac_unknown_action",   # ... unattributable
    "frac_phone",            # fraction with a phone assigned to this student
    "frac_eyes_closed",      # fraction with a sustained eye closure
    "frac_oriented",         # fraction oriented toward the room focus
    "frac_gaze_attending",   # fraction whose head pose reads as attending
    "mean_focus_offset_deg",  # mean angle between facing and room focus
    "action_entropy",        # how varied the actions are across the window
    "action_switch_rate",    # fraction of adjacent frames where action changed
    "frac_head_down_action",  # fraction whose action was read as head-down
    "mean_vertical_lean",    # mean nose-to-shoulder offset, normalised
    "frac_expr_negative",    # fraction with a sad expression
    "frac_expr_uncertain",   # fraction where expression could not be read
)

#: What the rest of the room was doing in the same window.
#:
#: Every other feature describes one student in isolation, which is not how
#: engagement works: a teacher asking a question moves the whole room at once,
#: and a student looking away while everyone else also looks away means
#: something different from one looking away alone. Measured across ten seeds,
#: adding these lifted accuracy from 0.580 to 0.636 and macro F1 from 0.574 to
#: 0.634 -- a gain larger than the seed spread, and the largest found from any
#: change that needed no new labels.
#:
#: Lagged features (the same student one window earlier) were tried alongside
#: these and *hurt*, dropping accuracy to 0.562, so they are deliberately
#: absent.
PEER_FEATURE_NAMES: tuple[str, ...] = (
    "peer_frac_on_task",      # mean on-task fraction across other students
    "peer_frac_gaze_attending",  # mean attending-gaze fraction across others
    "peer_count",             # how many others were visible, scaled
)

FEATURE_NAMES: tuple[str, ...] = BASE_FEATURE_NAMES + PEER_FEATURE_NAMES

#: Half-saturation point for peer_count: this many peers reads as 0.5.
#:
#: An earlier version divided the raw count by 10, which rescales but does not
#: bound. A class of 30 then gives 2.9 against a training mean of 0.72 and a
#: scale of 0.138 -- +16.5 standard deviations, the same defect n_frames had,
#: and it would appear the first time this ran in a bigger room than the one
#: it was developed in.
#:
#: n / (n + K) is bounded in [0, 1) for any class size, so the feature says
#: "a few peers" or "many peers" rather than carrying the roll number. The
#: difference between 25 and 30 classmates is not information about whether
#: one student is paying attention; the difference between 1 and 6 is.
PEER_COUNT_HALF: float = 8.0


@dataclass(frozen=True)
class Window:
    """One (student, time span) example, ready for a classifier.

    Attributes:
        person_id: The stable identity this window belongs to.
        start_ms: Window start, in milliseconds from session start.
        end_ms: Window end.
        features: Values in :data:`FEATURE_NAMES` order.
        rule_verdict: What the existing rule pipeline concluded over the same
            span -- ``"on"``, ``"off"`` or ``None``. Carried so a learned model
            can be compared against the rules on identical spans, NOT so it can
            be trained on them.
        scene: The scene index, for splitting.
        frame_ids: The pipeline frame ids this window covers. Carried from the
            graph log rather than recomputed from timestamps, because the two
            disagree: the log is written per PROCESSED frame, and reconstructing
            an index from elapsed milliseconds silently assumed every source
            clip had been processed.
    """

    person_id: int
    start_ms: int
    end_ms: int
    features: tuple[float, ...]
    rule_verdict: str | None
    scene: int
    frame_ids: tuple[int, ...]


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _entropy(counts: Mapping[str, int]) -> float:
    """Shannon entropy in bits over an action distribution.

    A student doing one thing for the whole window scores 0; one cycling
    between many actions scores high. Included because sustained attention and
    restlessness differ in variability, not only in which action dominates.
    """
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    h = 0.0
    for n in counts.values():
        if n > 0:
            p = n / total
            h -= p * math.log2(p)
    return h


def peer_features(peer_vectors: Sequence[Sequence[float]]) -> tuple[float, ...]:
    """Summarise what everyone *else* was doing, from their base vectors.

    Args:
        peer_vectors: One base-feature vector per other student visible in the
            same window. Empty when nobody else was present.

    Returns:
        Values in :data:`PEER_FEATURE_NAMES` order. All zero with no peers,
        which is honest: a student alone in frame has no room to compare to.
    """
    if not peer_vectors:
        return (0.0, 0.0, 0.0)
    i_on = BASE_FEATURE_NAMES.index("frac_on_task_action")
    i_gaze = BASE_FEATURE_NAMES.index("frac_gaze_attending")
    n = len(peer_vectors)
    return (
        sum(v[i_on] for v in peer_vectors) / n,
        sum(v[i_gaze] for v in peer_vectors) / n,
        n / (n + PEER_COUNT_HALF),
    )


def peer_vectors_in_span(
    frames_by_student: Mapping[int, Sequence[tuple[int, Mapping[str, Any]]]],
    person_id: int,
    start_ms: int,
    end_ms: int,
    expected_frames: int,
    min_coverage: float = MIN_COVERAGE,
) -> list[tuple[float, ...]]:
    """Base vectors for every other student visible over the same span.

    Shared by the batch and the live paths so the two cannot drift, which they
    had. The batch path grouped peers by exact equality of window start, and a
    student's windows begin at their own first observed frame, so students
    present for the same fifteen seconds who arrived at different moments were
    invisible to each other. The live path had the opposite defect, counting
    any student with buffered frames regardless of span, which pushed
    peer_count 1.8 standard deviations above what the model was trained on.

    Defining peers by the span itself fixes both, and makes a window scored
    live agree with the same window scored from the log.

    Args:
        frames_by_student: ``{person_id: [(timestamp_ms, features), ...]}``.
        person_id: The student being scored, excluded from their own summary.
        start_ms: Window start, inclusive.
        end_ms: Window end, exclusive.
        expected_frames: Frames a fully-present student would contribute.
        min_coverage: Peers seen less than this are omitted rather than
            counted, since a student glimpsed for one frame says little about
            what the room was doing.

    Returns:
        One base-feature vector per qualifying peer. Only base columns are
        taken: including a peer's own peer columns would make each student's
        score depend on a summary that already contains it.
    """
    peers: list[tuple[float, ...]] = []
    for other, rows in frames_by_student.items():
        if int(other) == int(person_id):
            continue
        frames = [f for t, f in rows if start_ms <= t < end_ms]
        if not frames:
            continue
        base = window_features(frames, expected_frames)[
            :len(BASE_FEATURE_NAMES)]
        if base[0] >= min_coverage:
            peers.append(base)
    return peers


def window_features(frames: Sequence[Mapping[str, Any]],
                    expected_frames: int,
                    peer_vectors: Sequence[Sequence[float]] = ()
                    ) -> tuple[float, ...]:
    """Summarise one student's frames within a window.

    Args:
        frames: Per-frame node ``features`` dicts for one student, in time
            order, all inside the window.
        expected_frames: How many frames the window would contain if the
            student were present throughout. Coverage is measured against this
            rather than against ``len(frames)``, so a student who vanishes for
            half the window is recorded as half-covered rather than fully
            covered over fewer frames.
        peer_vectors: Base vectors for the other students in the same window.
            Defaults to empty, which zeroes the peer columns rather than
            omitting them, so the vector length never varies.

    Returns:
        Values in :data:`FEATURE_NAMES` order.
    """
    n = len(frames)
    # Clamped at 1.0. expected_frames is estimated from the frame rate, so a
    # window can legitimately hold a frame or two more than predicted -- but
    # left unclamped this turns a rate mismatch into an unbounded feature. A
    # 13 fps webcam against the live scorer's 3 fps default gives coverage
    # 4.33, standardising to +21 sigma: the defect n_frames had, in another
    # column. Coverage also gates abstention, so an inflated value silently
    # disables the refusal that is meant to catch thin evidence.
    coverage = min(1.0, _safe_div(n, expected_frames))
    if n == 0:
        return tuple([0.0] * len(FEATURE_NAMES))

    actions = [f.get("action") for f in frames]
    counts = Counter(a for a in actions if a)
    read = sum(1 for a in actions if a and a != "unknown")

    on_task = sum(1 for a in actions if a in ON_TASK_ACTIONS)
    off_task = sum(1 for a in actions if a in OFF_TASK_ACTIONS)
    unknown = sum(1 for a in actions if a in (None, "unknown"))

    phone = sum(1 for f in frames if f.get("object") == "cell phone")
    eyes = sum(1 for f in frames if f.get("is_eyes_closed_sustained"))
    oriented = sum(1 for f in frames if f.get("oriented") is True)
    from backend.config import CONFIG
    attending_labels = set(CONFIG.engagement.attending_gaze_labels)
    attending = sum(1 for f in frames if f.get("gaze_label") in attending_labels)

    offsets = [f["focus_offset_deg"] for f in frames
               if f.get("focus_offset_deg") is not None]
    mean_offset = sum(offsets) / len(offsets) if offsets else 0.0

    # Fraction of adjacent frame pairs where the action changed. Bounded in
    # [0, 1] and independent of frame rate; a per-minute rate was neither.
    switches = sum(1 for a, b in pairwise(actions) if a != b and a and b)
    switch_rate = _safe_div(switches, max(n - 1, 1))

    head_down = sum(1 for f in frames if f.get("action") == "head_down")

    leans = [(f.get("posture") or {}).get("vertical_lean") for f in frames]
    leans = [v for v in leans if v is not None]
    mean_lean = sum(leans) / len(leans) if leans else 0.0

    expr = [f.get("expression") for f in frames]
    neg = sum(1 for e in expr if e == "sad")
    unc = sum(1 for e in expr if e in (None, "uncertain"))

    return (
        coverage,
        _safe_div(on_task, read or n),
        _safe_div(off_task, read or n),
        _safe_div(unknown, n),
        _safe_div(phone, n),
        _safe_div(eyes, n),
        _safe_div(oriented, n),
        _safe_div(attending, n),
        mean_offset,
        _entropy(counts),
        switch_rate,
        _safe_div(head_down, n),
        mean_lean,
        _safe_div(neg, n),
        _safe_div(unc, n),
    ) + peer_features(peer_vectors)


def _rule_verdict(frames: Sequence[Mapping[str, Any]]) -> str | None:
    """The existing pipeline's majority verdict over the same frames."""
    verdicts = [f.get("engagement") for f in frames if f.get("engagement")]
    if not verdicts:
        return None
    return Counter(verdicts).most_common(1)[0][0]


def iter_windows(graph_path: Path,
                 window_ms: int = WINDOW_MS,
                 stride_ms: int = STRIDE_MS,
                 min_coverage: float = MIN_COVERAGE) -> Iterator[Window]:
    """Read a graph log and yield one :class:`Window` per student per span.

    Args:
        graph_path: A ``live_graph.jsonl`` written by the pipeline.
        window_ms: Window length in milliseconds.
        stride_ms: Distance between window starts.
        min_coverage: Windows below this coverage are skipped entirely --
            there is nothing to learn from, and nothing a human could label.

    Yields:
        :class:`Window` instances in (student, time) order.
    """
    by_student: dict[int, list[tuple[int, int, Mapping[str, Any]]]] = {}
    times: list[int] = []
    for line in graph_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        g = json.loads(line)
        ts = int(g.get("timestamp_ms") or 0)
        times.append(ts)
        fid = int(g.get("frame_id") or 0)
        for node in g.get("nodes", []):
            pid = node.get("person_id")
            if pid and pid > 0:
                by_student.setdefault(int(pid), []).append(
                    (ts, int(g.get("scene") or 0), node.get("features") or {},
                     fid))

    if not times:
        return
    # Frame rate is needed to know what full coverage means for a window.
    span = max(times) - min(times)
    fps = _safe_div(len(set(times)) - 1, span / 1000.0) if span else 0.0
    expected = max(round(fps * window_ms / 1000.0), 1)

    # Peers are resolved by time span, not by a shared window-start key. An
    # earlier version grouped students whose `start` matched exactly, and a
    # student's windows begin at their own first observed frame -- so two
    # students present for the identical fifteen seconds, who entered the room
    # at different moments, never appeared in each other's cohort. On a
    # 14-student recording that left 40 per cent of windows reporting no peers
    # while six to eleven students were visible.
    spans = {pid: [(t, f) for (t, _scene, f, _fid) in rows]
             for pid, rows in by_student.items()}

    for pid, rows in sorted(by_student.items()):
        rows.sort(key=lambda r: r[0])
        first, last = rows[0][0], rows[-1][0]
        start = first
        while start + window_ms <= last + stride_ms:
            end = start + window_ms
            inside = [(sc, f, fid) for (t, sc, f, fid) in rows
                      if start <= t < end]
            if inside:
                frames = [f for _, f, _ in inside]
                base = window_features(frames, expected)[
                    :len(BASE_FEATURE_NAMES)]
                if base[0] >= min_coverage:
                    peers = peer_vectors_in_span(
                        spans, pid, start, end, expected, min_coverage)
                    yield Window(
                        person_id=pid,
                        start_ms=start,
                        end_ms=end,
                        features=tuple(base) + peer_features(peers),
                        rule_verdict=_rule_verdict(frames),
                        scene=inside[0][0],
                        frame_ids=tuple(fid for _, _, fid in inside),
                    )
            start += stride_ms
