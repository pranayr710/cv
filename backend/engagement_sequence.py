"""Per-frame feature sequences, for models that read order rather than averages.

:mod:`backend.engagement_features` reduces a 15-second window to 15 aggregate
numbers -- means and fractions. That reduction throws away the one thing a
sequence model exists to use: *order*. A student who looks away and comes back
and a student who drifts away and stays away produce nearly identical aggregate
vectors, because the fraction of frames spent looking away is the same in both.

This module keeps the window as a sequence: one vector per frame, in time
order, plus a mask marking which timesteps the student was actually visible
for. Whether that extra information is worth anything is an empirical question
-- see ``tools/train_mamba_scorer.py``, which measures it against the
aggregate model on the same labels under the same group-aware split.

The per-frame vector is deliberately the *unaggregated* form of the same
signals the window features are built from, so a win cannot be explained by
having fed the sequence model better inputs.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.config import CONFIG
from backend.engagement_features import (
    OFF_TASK_ACTIONS,
    ON_TASK_ACTIONS,
    WINDOW_MS,
)

#: One entry per column of the per-frame vector, in order.
FRAME_FEATURE_NAMES: tuple[str, ...] = (
    "present",
    "on_task_action",
    "off_task_action",
    "unknown_action",
    "phone",
    "eyes_closed",
    "oriented",
    "gaze_attending",
    "focus_offset",
    "head_down_pitch",
    "vertical_lean",
    "expr_negative",
    "expr_uncertain",
)

#: Timesteps per window. Windows are 15s and the pipeline runs near 3fps, so
#: 48 covers a full window with room to spare; shorter windows are padded and
#: the mask marks the padding.
SEQ_LEN: int = 48

#: Head pitch beyond this reads as looking down at a desk or a lap.
HEAD_DOWN_PITCH_DEG: float = 20.0

_NEGATIVE = frozenset({"sad", "angry", "disgust", "fear"})
_UNCERTAIN = frozenset({"neutral", "unknown", None})


def _num(value: Any) -> float:
    """A float, or 0.0 for anything missing or unparseable."""
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def frame_vector(frame: Mapping[str, Any] | None) -> list[float]:
    """One timestep of the sequence.

    ``None`` means the student was not detected in that frame, which is
    information rather than absence: it is returned as an all-zero vector whose
    ``present`` flag is 0, so a model can tell "not visible" from "visible and
    showing nothing".
    """
    if frame is None:
        return [0.0] * len(FRAME_FEATURE_NAMES)

    action = frame.get("action")
    expr = frame.get("expression")
    posture = frame.get("posture") or {}
    if not isinstance(posture, Mapping):
        posture = {}

    pitch = _num(frame.get("focus_offset_deg"))
    return [
        1.0,
        1.0 if action in ON_TASK_ACTIONS else 0.0,
        1.0 if action in OFF_TASK_ACTIONS else 0.0,
        1.0 if action in (None, "unknown") else 0.0,
        1.0 if frame.get("object") == "cell phone" else 0.0,
        1.0 if frame.get("is_eyes_closed_sustained") else 0.0,
        1.0 if frame.get("oriented") else 0.0,
        (1.0 if frame.get("gaze_label")
         in CONFIG.engagement.attending_gaze_labels else 0.0),
        # Degrees scaled to roughly unit range so no column dominates before
        # standardisation.
        pitch / 90.0,
        1.0 if abs(pitch) > HEAD_DOWN_PITCH_DEG else 0.0,
        _num(posture.get("vertical_lean")),
        1.0 if expr in _NEGATIVE else 0.0,
        1.0 if expr in _UNCERTAIN else 0.0,
    ]


@dataclass(frozen=True)
class Sequence3:
    """A labelled window, kept as a sequence.

    Attributes:
        person_id: Who this is, for group-aware splitting.
        start_ms: Window start, matching the key in ``labels.json``.
        steps: ``SEQ_LEN`` vectors in time order, padded with absent frames.
        n_real: How many timesteps were real rather than padding.
    """

    person_id: int
    start_ms: int
    steps: list[list[float]]
    n_real: int


def sequence_for_window(frames_by_ms: Mapping[int, Mapping[str, Any]],
                        start_ms: int,
                        person_id: int) -> Sequence3:
    """Resample one student's frames onto a fixed grid of ``SEQ_LEN`` steps.

    A fixed grid rather than the raw frames is what makes windows comparable:
    the pipeline's frame rate varies between a batch run and a live webcam, and
    a model fed variable-length sequences would learn that difference. Each
    slot takes the nearest frame within half a slot, or an absent vector.
    """
    slot = WINDOW_MS / SEQ_LEN
    times = sorted(frames_by_ms)
    steps, n_real = [], 0
    for i in range(SEQ_LEN):
        centre = start_ms + (i + 0.5) * slot
        best, best_gap = None, slot / 2.0
        for t in times:
            gap = abs(t - centre)
            if gap <= best_gap:
                best, best_gap = t, gap
        if best is None:
            steps.append(frame_vector(None))
        else:
            steps.append(frame_vector(frames_by_ms[best]))
            n_real += 1
    return Sequence3(person_id, start_ms, steps, n_real)


def sequences_from_graph(graph_path: Path) -> dict[tuple[int, int], Sequence3]:
    """Every window in a graph log, keyed by ``(person_id, start_ms)``.

    Window boundaries come from :func:`backend.engagement_features.iter_windows`
    rather than being re-derived here. That is deliberate: windows start at
    each student's *first* observed frame and step by the stride, so a student
    who appears late has boundaries no global grid would reproduce. Re-deriving
    them produced keys that matched only 23 of 179 labels.

    Sharing the one implementation means a sequence and its aggregate feature
    vector always describe the same span, which is what makes the comparison
    between the two models fair.
    """
    from backend.engagement_features import iter_windows

    per_person: dict[int, dict[int, Mapping[str, Any]]] = {}
    for line in graph_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ts = int(row.get("timestamp_ms") or 0)
        for node in row.get("nodes", []):
            pid = node.get("person_id")
            if pid is None:
                continue
            per_person.setdefault(int(pid), {})[ts] = node.get("features") or {}

    out: dict[tuple[int, int], Sequence3] = {}
    for window in iter_windows(graph_path):
        frames = per_person.get(window.person_id, {})
        inside = {t: f for t, f in frames.items()
                  if window.start_ms <= t < window.end_ms}
        if inside:
            out[(window.person_id, window.start_ms)] = sequence_for_window(
                inside, window.start_ms, window.person_id)
    return out


def stack(seqs: Sequence[Sequence3]) -> tuple[list[list[list[float]]], list[int]]:
    """Sequences and their person ids, ready for tensor conversion."""
    return [s.steps for s in seqs], [s.person_id for s in seqs]
