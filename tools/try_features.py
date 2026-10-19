"""Measure whether a candidate feature earns its place.

Adding features is cheap and almost always looks like it helped, because a
larger feature set fits the training data better. This evaluates each
candidate the way the shipping model is evaluated -- out of fold, grouped by
student -- and then again on the only labels known to be repeatable, so a gain
that is really an artefact of noisy targets is visible as such.

Two things make the answer trustworthy:

**Every candidate is measured alone, against the same baseline.** Adding eight
features together and seeing an improvement says nothing about which of them
did it, and a feature that hurts can be hidden by one that helps.

**Seeds are averaged and the spread is reported.** A single run of an MLP on
557 labels moves by more than most real feature gains. This project has
already had a result reverse under fold permutation; a difference smaller than
the spread is not a difference.

    python tools/try_features.py
    python tools/try_features.py --seeds 10
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

warnings.filterwarnings("ignore")

from backend.engagement_features import (
    MIN_COVERAGE,
    WINDOW_MS,
    _safe_div,
)

GRAPH = Path("outputs/final2/live_graph.jsonl")
LABELS = Path("labeling/labels.json")
RECHECK = Path("labeling/recheck.json")
REPORT = Path("outputs/candidate_features.json")


# --------------------------------------------------------------------------
# Helpers shared by the candidates.
# --------------------------------------------------------------------------

def _pt(posture: Mapping[str, Any] | None, name: str) -> tuple[float, float] | None:
    """One landmark as (x, y), or None when it was not detected."""
    if not isinstance(posture, Mapping):
        return None
    value = posture.get(name)
    if not value or len(value) < 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None


def _shoulder_width(posture: Mapping[str, Any] | None) -> float | None:
    """Distance between the shoulders, the natural scale for this person.

    Every spatial measurement below is divided by it, which is what makes
    these features independent of how far the student sits from the camera.
    A pixel distance is not comparable between the front row and the back.
    """
    left, right = _pt(posture, "left_shoulder"), _pt(posture, "right_shoulder")
    if left is None or right is None:
        return None
    width = math.dist(left, right)
    return width if width > 1.0 else None


def _spread(values: Sequence[float]) -> float:
    """Standard deviation, or 0.0 for fewer than two values."""
    return float(np.std(values)) if len(values) > 1 else 0.0


# --------------------------------------------------------------------------
# Candidates. Each takes the window's frames and returns one float.
# --------------------------------------------------------------------------

def motion_energy(frames: Sequence[Mapping[str, Any]],
                  times: Sequence[int] | None = None) -> float:
    """Head movement in shoulder-widths per second.

    Motion Energy Analysis is the standard nonverbal measure here, and the
    existing features describe what a student is doing rather than how much
    they are moving while doing it -- fidgeting and stillness share an action
    label.

    Divided by elapsed time as well as by shoulder width. Shoulder width alone
    removes the distance to the camera; without the time division a step size
    still carries the sampling interval, which is the defect that has broken
    this project twice.
    """
    steps = []
    previous = None
    previous_t = None
    for i, frame in enumerate(frames):
        posture = frame.get("posture")
        nose, width = _pt(posture, "nose"), _shoulder_width(posture)
        now = times[i] if times and i < len(times) else None
        if nose is None or width is None:
            previous = previous_t = None
            continue
        if previous is not None:
            gap = ((now - previous_t) / 1000.0
                   if now is not None and previous_t is not None else None)
            if gap is None or gap <= 0:
                gap = 1.0 / 3.0          # the batch pipeline's nominal rate
            # Clipped: tracker jitter on a small crop produces single frames
            # of implausible motion that would otherwise dominate the mean.
            steps.append(min(math.dist(nose, previous) / width / gap, 5.0))
        previous, previous_t = nose, now
    return float(np.mean(steps)) if steps else 0.0


def longest_attend_bout(frames: Sequence[Mapping[str, Any]]) -> float:
    """Longest unbroken run of attending frames, as a fraction of the window.

    frac_gaze_attending of 0.5 is produced identically by one sustained
    seven-second look and by twenty flickers. Sustained attention is the
    quantity the engagement literature actually localises, and the fraction
    cannot see it. action_switch_rate covers the action stream only, not gaze.
    """
    from backend.config import CONFIG

    attending = CONFIG.engagement.attending_gaze_labels
    best = run = 0
    for frame in frames:
        if frame.get("gaze_label") in attending:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return _safe_div(best, len(frames))


def longest_off_task_bout(frames: Sequence[Mapping[str, Any]]) -> float:
    """Longest unbroken off-task run, as a fraction of the window.

    The same argument in the other direction, and the one a teacher cares
    about: a student who drifted away once for ten seconds is a different
    case from one who glanced away ten times.
    """
    from backend.engagement_features import OFF_TASK_ACTIONS

    best = run = 0
    for frame in frames:
        if frame.get("action") in OFF_TASK_ACTIONS:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return _safe_div(best, len(frames))


def head_torso_misalign(frames: Sequence[Mapping[str, Any]]) -> float:
    """Mean disagreement between where the torso points and where the head looks.

    Separates a student whose body is turned to a neighbour but whose head
    snapped forward -- compliance -- from one genuinely oriented at the
    lesson. frac_oriented collapses both into one bit.

    Built from the 2D shoulder separation rather than MediaPipe's z, which is
    unreliable at the distance these crops are taken from.
    """
    values = []
    for frame in frames:
        posture = frame.get("posture")
        width = _shoulder_width(posture)
        shoulder, hip = _pt(posture, "shoulder_mid"), _pt(posture, "hip_mid")
        offset = frame.get("focus_offset_deg")
        if width is None or shoulder is None or hip is None or offset is None:
            continue
        torso = math.dist(shoulder, hip)
        if torso <= 1.0:
            continue
        # Narrow shoulders against torso length means the body is turned.
        body_turn = 1.0 - min(width / torso, 1.0)
        head_turn = min(abs(float(offset)) / 90.0, 1.0)
        values.append(abs(head_turn - body_turn))
    return float(np.mean(values)) if values else 0.0


def head_stillness(frames: Sequence[Mapping[str, Any]]) -> float:
    """How little the head wandered, as 1 / (1 + spread).

    Bounded in (0, 1]: 1.0 is perfectly still. Distinct from motion_energy,
    which measures step size -- a student can make many small movements
    around one point, or drift steadily with few large steps.
    """
    xs, ys = [], []
    for frame in frames:
        posture = frame.get("posture")
        nose, width = _pt(posture, "nose"), _shoulder_width(posture)
        if nose is None or width is None:
            continue
        xs.append(nose[0] / width)
        ys.append(nose[1] / width)
    if len(xs) < 2:
        return 0.0
    return 1.0 / (1.0 + _spread(xs) + _spread(ys))


def gaze_stability(frames: Sequence[Mapping[str, Any]]) -> float:
    """Spread of the angle to the room focus, scaled to roughly unit range.

    The feature set carries the *mean* offset but nothing about whether it
    held steady. A student fixed 40 degrees off (watching a neighbour work)
    and one swinging through 40 degrees on average (scanning the room) are
    different, and currently identical.
    """
    angles = [float(f["focus_offset_deg"]) for f in frames
              if f.get("focus_offset_deg") is not None]
    return _spread(angles) / 90.0 if len(angles) > 1 else 0.0


def hand_to_face(frames: Sequence[Mapping[str, Any]]) -> float:
    """Fraction of frames with a wrist within one shoulder width of the nose.

    Chin-resting and head-propping are documented postures of disengagement,
    and nothing in the current set can see a hand at all.
    """
    near = total = 0
    for frame in frames:
        posture = frame.get("posture")
        nose, width = _pt(posture, "nose"), _shoulder_width(posture)
        if nose is None or width is None:
            continue
        total += 1
        for side in ("left_wrist", "right_wrist"):
            wrist = _pt(posture, side)
            if wrist is not None and math.dist(wrist, nose) < width:
                near += 1
                break
    return _safe_div(near, total)


def hands_visible(frames: Sequence[Mapping[str, Any]]) -> float:
    """Fraction of frames where at least one wrist was detected.

    Hands on the desk are visible; hands in a lap or under a desk are not.
    A proxy for whether the student is working with their hands, and cheap
    because the detection already happened.
    """
    seen = total = 0
    for frame in frames:
        posture = frame.get("posture")
        if not isinstance(posture, Mapping):
            continue
        total += 1
        if _pt(posture, "left_wrist") or _pt(posture, "right_wrist"):
            seen += 1
    return _safe_div(seen, total)


def lean_variability(frames: Sequence[Mapping[str, Any]]) -> float:
    """Spread of vertical lean. The set has its mean but not its movement."""
    leans = []
    for frame in frames:
        posture = frame.get("posture")
        if isinstance(posture, Mapping) and posture.get("vertical_lean") is not None:
            leans.append(float(posture["vertical_lean"]))
    return _spread(leans)


def facing_variability(frames: Sequence[Mapping[str, Any]]) -> float:
    """Spread of torso facing angle, in turns.

    Shoulder geometry rather than head pose, so it survives at a distance
    where the face is 30 pixels. A student whose body keeps turning is
    oriented somewhere other than the lesson.
    """
    angles = []
    for frame in frames:
        posture = frame.get("posture")
        if not isinstance(posture, Mapping):
            continue
        facing = posture.get("facing_direction")
        if facing and len(facing) >= 2:
            angles.append(math.atan2(float(facing[1]), float(facing[0]))
                          / (2 * math.pi))
    return _spread(angles)


def posture_openness(frames: Sequence[Mapping[str, Any]]) -> float:
    """Mean shoulder width over torso height.

    A slumped or hunched student foreshortens; an upright one does not. Both
    distances come from the same body, so the ratio is scale-free by
    construction rather than by normalisation.
    """
    ratios = []
    for frame in frames:
        posture = frame.get("posture")
        width = _shoulder_width(posture)
        shoulder, hip = _pt(posture, "shoulder_mid"), _pt(posture, "hip_mid")
        if width is None or shoulder is None or hip is None:
            continue
        torso = math.dist(shoulder, hip)
        if torso > 1.0:
            ratios.append(width / torso)
    return float(np.mean(ratios)) if ratios else 0.0


def peer_synchrony(frames: Sequence[Mapping[str, Any]],
                   peers: Sequence[Sequence[Mapping[str, Any]]]) -> float:
    """Agreement between this student's on-task frames and the room's.

    Peer context is already in the set as a level -- what fraction of the
    room was on task. This is the *correlation*: a student moving with the
    room is following the lesson, one moving against it is not, and the two
    are different at the same peer level.
    """
    if not peers:
        return 0.0
    from backend.engagement_features import ON_TASK_ACTIONS

    mine = [1.0 if f.get("action") in ON_TASK_ACTIONS else 0.0 for f in frames]
    if len(mine) < 2 or len(set(mine)) < 2:
        return 0.0
    theirs = []
    for i in range(len(mine)):
        votes = [1.0 if p[i].get("action") in ON_TASK_ACTIONS else 0.0
                 for p in peers if i < len(p)]
        theirs.append(float(np.mean(votes)) if votes else 0.0)
    if len(set(theirs)) < 2:
        return 0.0
    return float(np.corrcoef(mine, theirs)[0, 1])


#: Candidates that need only this student's frames.
SIMPLE: dict[str, Callable[[Sequence[Mapping[str, Any]]], float]] = {
    "longest_attend_bout": longest_attend_bout,
    "longest_off_task_bout": longest_off_task_bout,
    "head_torso_misalign": head_torso_misalign,
    "head_stillness": head_stillness,
    "gaze_stability": gaze_stability,
    "hand_to_face": hand_to_face,
    "hands_visible": hands_visible,
    "lean_variability": lean_variability,
    "facing_variability": facing_variability,
    "posture_openness": posture_openness,
}


def build(graph_path: Path) -> dict[tuple[int, int], dict[str, float]]:
    """Compute every candidate for every window in a graph log."""
    from backend.engagement_features import iter_windows

    by_student: dict[int, list[tuple[int, Mapping[str, Any]]]] = {}
    for line in graph_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        ts = int(row.get("timestamp_ms") or 0)
        for node in row.get("nodes", []):
            pid = node.get("person_id")
            if pid and int(pid) > 0:
                by_student.setdefault(int(pid), []).append(
                    (ts, node.get("features") or {}))

    out: dict[tuple[int, int], dict[str, float]] = {}
    for window in iter_windows(graph_path):
        rows = by_student.get(window.person_id, [])
        frames = [f for t, f in rows
                  if window.start_ms <= t < window.end_ms]
        if not frames:
            continue
        values = {name: fn(frames) for name, fn in SIMPLE.items()}
        stamps = [t for t, _f in rows
                  if window.start_ms <= t < window.end_ms]
        values["motion_energy"] = motion_energy(frames, stamps)

        peer_frames = []
        for other, other_rows in by_student.items():
            if other == window.person_id:
                continue
            got = [f for t, f in other_rows
                   if window.start_ms <= t < window.end_ms]
            if len(got) >= len(frames) * MIN_COVERAGE:
                peer_frames.append(got)
        values["peer_synchrony"] = peer_synchrony(frames, peer_frames)
        out[(window.person_id, window.start_ms)] = values
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()

    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import GroupKFold
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    print(f"computing candidates over {WINDOW_MS / 1000:.0f}s windows...")
    extra = build(GRAPH)
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    recheck = (json.loads(RECHECK.read_text(encoding="utf-8"))
               if RECHECK.exists() else {})
    trusted_ids = {w for w in recheck
                   if labels.get(w, {}).get("label") in ("on", "off")
                   and recheck[w] in ("on", "off")
                   and labels[w]["label"] == recheck[w]}

    base, y, groups, cand, trusted = [], [], [], [], []
    for wid, entry in labels.items():
        if entry["label"] not in ("on", "off"):
            continue
        key = (entry["person_id"], entry["start_ms"])
        if key not in extra:
            continue
        base.append(entry["features"])
        y.append(1 if entry["label"] == "on" else 0)
        groups.append(entry["person_id"])
        cand.append(extra[key])
        trusted.append(wid in trusted_ids)

    base = np.array(base, float)
    y = np.array(y)
    groups = np.array(groups)
    trusted = np.array(trusted)
    names = sorted(SIMPLE) + ["motion_energy", "peer_synchrony"]
    print(f"{len(y)} labelled windows, {len(names)} candidates, "
          f"{trusted.sum()} trusted\n")

    def evaluate(matrix, seed):
        pred = np.zeros(len(y))
        for tr, te in GroupKFold(n_splits=5).split(matrix, y, groups):
            scaler = StandardScaler().fit(matrix[tr])
            clf = MLPClassifier((16,), max_iter=3000, random_state=seed)
            clf.fit(scaler.transform(matrix[tr]), y[tr])
            pred[te] = clf.predict(scaler.transform(matrix[te]))
        return pred

    def score(matrix):
        accs, f1s, tacc = [], [], []
        for seed in range(args.seeds):
            pred = evaluate(matrix, seed)
            accs.append(accuracy_score(y, pred))
            f1s.append(f1_score(y, pred, average="macro"))
            if trusted.sum() >= 30:
                tacc.append(accuracy_score(y[trusted], pred[trusted]))
        return (float(np.mean(accs)), float(np.std(accs)), float(np.mean(f1s)),
                float(np.mean(tacc)) if tacc else float("nan"))

    b_acc, b_sd, b_f1, b_tr = score(base)
    print(f"  {'feature set':<26}{'acc':>8}{'sd':>7}{'F1':>8}"
          f"{'trusted':>9}{'delta F1':>10}")
    print(f"  {'baseline (18)':<26}{b_acc:>8.3f}{b_sd:>7.3f}{b_f1:>8.3f}"
          f"{b_tr:>9.3f}{'':>10}")

    results = {}
    for name in names:
        column = np.array([[c[name]] for c in cand], float)
        acc, sd, f1, tr = score(np.hstack([base, column]))
        verdict = "helps" if f1 - b_f1 > (sd + b_sd) else (
            "hurts" if b_f1 - f1 > (sd + b_sd) else "noise")
        results[name] = {"accuracy": round(acc, 4), "sd": round(sd, 4),
                         "macro_f1": round(f1, 4),
                         "trusted_accuracy": None if tr != tr else round(tr, 4),
                         "delta_f1": round(f1 - b_f1, 4), "verdict": verdict}
        print(f"  + {name:<24}{acc:>8.3f}{sd:>7.3f}{f1:>8.3f}{tr:>9.3f}"
              f"{f1 - b_f1:>+10.3f}  {verdict}")

    helpful = [n for n, r in results.items() if r["verdict"] == "helps"]
    if helpful:
        columns = np.array([[c[n] for n in helpful] for c in cand], float)
        acc, sd, f1, tr = score(np.hstack([base, columns]))
        print(f"\n  all {len(helpful)} that helped, together:"
              f"  acc {acc:.3f}  F1 {f1:.3f}  trusted {tr:.3f}"
              f"  ({f1 - b_f1:+.3f})")
        results["_combined"] = {"features": helpful, "macro_f1": round(f1, 4),
                                "delta_f1": round(f1 - b_f1, 4)}
    else:
        print("\n  None beat the seed spread. The existing 18 already carry "
              "what\n  these measure, or the labels are too noisy to show the "
              "difference.")

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(
        {"n_windows": int(len(y)), "seeds": args.seeds,
         "baseline": {"accuracy": round(b_acc, 4), "macro_f1": round(b_f1, 4),
                      "trusted_accuracy": round(b_tr, 4)},
         "candidates": results}, indent=1), encoding="utf-8")
    print(f"\n  report -> {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
