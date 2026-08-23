"""Per-student summary profiles from a finished Stage 1+2 JSONL run.

Turns the frame-by-frame log into the thing actually asked for: one record per
student, keyed by their re-identified ``person_id`` (see
:mod:`backend.identity`), holding everything known about them for the whole
video — how long they were seen, their expression breakdown, their behaviour
breakdown, and a single concentration percentage.

What "complete details" means here, precisely
----------------------------------------------

This aggregates only what the pipeline itself produced for that student —
expression labels, behaviour labels, gaze-derived concentration. It does not
add anything the frame-level pipeline did not already compute, and it carries
forward every honesty flag those modules already attach (an "uncertain"
expression stays uncertain in the counts; a "weak" behaviour reading is
reported as weak, not folded in as equally trustworthy — see
:mod:`backend.behaviour`'s ``reliability`` field).

Grouping is by ``person_id``, not ``track_id``, specifically because
``person_id`` is the field designed to survive occlusion and reappearance
(:mod:`backend.identity`); grouping by the raw ``track_id`` would silently
split one real student into several profiles across a single occlusion gap,
defeating the entire point of adding re-identification.

A real absence-of-evidence trap this module guards against explicitly: "off"
is only reachable through a behaviour reading (see
:mod:`backend.engagement` — a bare non-attending gaze is deliberately left
"unknown", never "off"). A student who never once gets a behaviour reading
(e.g. the behaviour model finds zero detections on unfamiliar footage — a real
case hit on out-of-distribution video, same generalization failure documented
in ``CHALLENGES_AND_SOLUTIONS.md`` section 18) will show 100% concentration
purely because off-task was structurally unreachable, not because they were
attentive. ``concentration.off_task_detectable`` and ``concentration.caveat``
surface this directly on the profile rather than requiring a reader to
cross-reference the separate ``behaviour.classified`` count to notice it.

Usage (CLI):
    python -m backend.student_profile --jsonl outputs/stage1.jsonl --out outputs/students.json

Usage (API):
    from backend.student_profile import build_profiles
    profiles = build_profiles("outputs/stage1.jsonl")
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

from backend.config import CONFIG
from backend.engagement import classify_engagement, summarise_engagement

logger = logging.getLogger(__name__)


def _tally(labels: list[str | None]) -> dict[str, object]:
    """Count non-``None`` labels, matching the shape of the project's other
    ``summarise_*`` helpers (``backend.expression``, ``backend.behaviour``).

    Args:
        labels: One label (or ``None``) per frame that student was seen in.

    Returns:
        A dict with ``classified``, ``unavailable``, and ``counts`` per label.
    """
    present = [label for label in labels if label is not None]
    counts = dict(Counter(present))
    return {
        "classified": len(present),
        "unavailable": len(labels) - len(present),
        "counts": counts,
    }


def build_profiles(
    jsonl_path: str | Path, config=None
) -> dict[int, dict[str, object]]:
    """Read a finished Stage 1+2 JSONL file and build one profile per student.

    Args:
        jsonl_path: Path to a JSONL file matching ``schema.json``, already
            containing ``person_id`` (i.e. produced after the
            :mod:`backend.identity` wiring in :mod:`backend.integrate`).
        config: The full pipeline config, for engagement rules. Defaults to
            ``CONFIG``.

    Returns:
        A dict keyed by ``person_id``. Each value has:

        * ``person_id``, ``face_verified`` (``True`` unless the id is negative
          — see :mod:`backend.identity` for what a negative id means),
        * ``frames_seen``, ``first_seen_ms``, ``last_seen_ms``,
          ``duration_ms``,
        * ``expression`` — output of :func:`_tally` over expression labels,
        * ``behaviour`` — output of :func:`_tally` over behaviour labels, plus
          ``weak_labels`` (behaviour classes with poor measured recall that
          were nonetheless reported — see :mod:`backend.behaviour`),
        * ``concentration`` — output of
          :func:`~backend.engagement.summarise_engagement`.

        Students who are never assigned a ``person_id`` (an unconfirmed track,
        ``track_id`` present but re-identification never resolved — should not
        occur in practice, since every ``track_id`` produces a ``person_id``,
        but a record with ``track_id is None`` is skipped, matching how
        ``person_id`` is defined) are excluded, since there is no stable key
        to file them under.

    Raises:
        FileNotFoundError: If ``jsonl_path`` does not exist.
    """
    cfg = config if config is not None else CONFIG
    src = Path(jsonl_path)
    if not src.is_file():
        raise FileNotFoundError(f"JSONL file not found: {src}")

    frames_seen: dict[int, int] = defaultdict(int)
    first_ms: dict[int, int] = {}
    last_ms: dict[int, int] = {}
    expression_labels: dict[int, list[str | None]] = defaultdict(list)
    behaviour_labels: dict[int, list[str | None]] = defaultdict(list)
    behaviour_reliability: dict[int, set[str]] = defaultdict(set)
    engagement_verdicts: dict[int, list[str | None]] = defaultdict(list)

    with src.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            ts_ms = record["timestamp_ms"]
            for person in record["persons"]:
                person_id = person.get("person_id")
                if person_id is None:
                    continue

                frames_seen[person_id] += 1
                first_ms[person_id] = min(first_ms.get(person_id, ts_ms), ts_ms)
                last_ms[person_id] = max(last_ms.get(person_id, ts_ms), ts_ms)

                expression = person.get("expression")
                expression_labels[person_id].append(
                    expression["label"] if expression else None
                )

                behaviour = person.get("behaviour")
                behaviour_labels[person_id].append(
                    behaviour["label"] if behaviour else None
                )
                if behaviour and behaviour.get("reliability") == "weak":
                    behaviour_reliability[person_id].add(behaviour["label"])

                gaze_label = (
                    person["head_pose"]["gaze_label"]
                    if person.get("head_pose")
                    else None
                )
                behaviour_label = behaviour["label"] if behaviour else None
                engagement_verdicts[person_id].append(
                    classify_engagement(gaze_label, behaviour_label, cfg.engagement)
                )

    profiles: dict[int, dict[str, object]] = {}
    for person_id, seen_count in frames_seen.items():
        behaviour_summary = _tally(behaviour_labels[person_id])
        behaviour_summary["weak_labels"] = sorted(behaviour_reliability[person_id])
        concentration = summarise_engagement(engagement_verdicts[person_id])

        # Off-task can ONLY be reached through a behaviour reading (see
        # backend.engagement -- a bare non-attending gaze is deliberately left
        # "unknown", never "off"). So if this student never once got a
        # behaviour reading, "off" was structurally unreachable for them, and
        # a resulting 100% concentration_pct is an absence-of-evidence
        # artifact, not a finding. Found for real: the behaviour model
        # returned zero readings on an out-of-distribution video (a different
        # generalization failure, same shape as CHALLENGES_AND_SOLUTIONS.md
        # section 18) and every student's concentration silently read ~100%.
        # Surfaced here rather than left for a reader to notice by
        # cross-referencing two unrelated fields.
        if behaviour_summary["classified"] == 0 and concentration["frames"] > 0:
            concentration["off_task_detectable"] = False
            concentration["caveat"] = (
                "No behaviour reading was ever obtained for this student, so "
                "off-task states (phone/sleep) could not be detected at all -- "
                "concentration_pct reflects gaze only and should not be read "
                "as a real attentiveness score."
            )
        else:
            concentration["off_task_detectable"] = True

        profiles[person_id] = {
            "person_id": person_id,
            # A negative id means this student was never matched by face --
            # see backend/identity.py. Carried forward explicitly so a report
            # cannot present an unverified id as a confirmed re-identification.
            "face_verified": person_id > 0,
            "frames_seen": seen_count,
            "first_seen_ms": first_ms[person_id],
            "last_seen_ms": last_ms[person_id],
            "duration_ms": last_ms[person_id] - first_ms[person_id],
            "expression": _tally(expression_labels[person_id]),
            "behaviour": behaviour_summary,
            "concentration": concentration,
        }
    return profiles


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m backend.student_profile",
        description=(
            "Build one per-student summary profile (expression, behaviour, "
            "concentration) from a finished Stage 1+2 JSONL run."
        ),
    )
    parser.add_argument("--jsonl", required=True, help="Input JSONL path.")
    parser.add_argument("--out", required=True, help="Output JSON path.")
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point.

    Example:
        python -m backend.student_profile --jsonl outputs/stage1.jsonl --out outputs/students.json
    """
    args = _build_arg_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(levelname)s %(name)s: %(message)s")

    try:
        profiles = build_profiles(args.jsonl)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Sorted by person_id for a stable, readable diff between runs; verified
    # ids (>=1) naturally sort before unverified negative ones in reverse, so
    # sort by absolute value with verified-first as the tiebreak instead.
    ordered = sorted(profiles.values(), key=lambda p: (not p["face_verified"], abs(p["person_id"])))
    out_path.write_text(json.dumps(ordered, indent=2), encoding="utf-8")

    logger.info(
        "Wrote %d student profile(s) to %s (%d face-verified, %d unverified).",
        len(ordered),
        out_path,
        sum(1 for p in ordered if p["face_verified"]),
        sum(1 for p in ordered if not p["face_verified"]),
    )
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
