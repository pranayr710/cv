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
    "frac_head_down_pitch",  # fraction with head pitch below the down threshold
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

#: Divisor that keeps peer_count in roughly unit range. A raw count would
#: scale with class size and reintroduce exactly the frame-rate style of bug
#: that n_frames caused.
PEER_COUNT_SCALE: float = 10.0


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
        n / PEER_COUNT_SCALE,
    )


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
    coverage = _safe_div(n, expected_frames)
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

    # Two passes. Peer context needs every student's base vector for a given
    # window start, and that is not known while still walking one student, so
    # nothing can be yielded until all of them have been summarised.
    staged: dict[int, list[dict[str, Any]]] = {}
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
                    staged.setdefault(start, []).append({
                        "person_id": pid,
                        "start_ms": start,
                        "end_ms": end,
                        "base": base,
                        "rule_verdict": _rule_verdict(frames),
                        "scene": inside[0][0],
                        "frame_ids": tuple(fid for _, _, fid in inside),
                    })
            start += stride_ms

    for start in sorted(staged):
        cohort = staged[start]
        for item in cohort:
            peers = [o["base"] for o in cohort
                     if o["person_id"] != item["person_id"]]
            yield Window(
                person_id=item["person_id"],
                start_ms=item["start_ms"],
                end_ms=item["end_ms"],
                features=tuple(item["base"]) + peer_features(peers),
                rule_verdict=item["rule_verdict"],
                scene=item["scene"],
                frame_ids=item["frame_ids"],
            )
