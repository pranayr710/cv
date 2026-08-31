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

from backend.actions import OFF_TASK as OFF_TASK_ACTIONS

#: Seconds of on-task frames tolerated inside an off-task stretch before it
#: counts as two separate episodes. Set from the sampling rate rather than the
#: literature: at the rates this pipeline runs, a couple of seconds is a few
#: frames, enough to absorb a label flickering between on_phone and head_down
#: as a hand moves, and short enough that genuinely returning to task ends the
#: episode.
FLICKER_BRIDGE_SECONDS = 2.0
from backend.config import CONFIG
from backend.engagement import classify_engagement, summarise_engagement

logger = logging.getLogger(__name__)


def _people_in(record: dict) -> tuple[list[dict], bool]:
    """The per-person entries of one record, from either input format.

    This module was written against Stage 1+2 JSONL (``persons``). Stage 3
    turns each of those into a graph ``node`` and Stage 4 adds temporal
    features to it, so reading only Stage 1 meant everything the graph computed
    -- the engagement verdict, the rolling attention percentage, sustained
    distraction -- was recomputed here at best and lost at worst. Accepting
    both shapes lets the profile be built from the richest record available
    without duplicating Stage 3's logic.

    Args:
        record: One decoded JSONL line, in either format.

    Returns:
        ``(people, from_graph)``, where each person is normalised to the
        Stage-1 field names this module already uses, and ``from_graph`` says
        whether the graph's own richer fields are present.

    Raises:
        KeyError: If the record has neither ``persons`` nor ``nodes``.
    """
    if "persons" in record:
        return record["persons"], False
    if "nodes" not in record:
        raise KeyError(
            "record has neither 'persons' (Stage 1+2) nor 'nodes' (Stage 3+4); "
            "is this a pipeline output file?"
        )

    people = []
    for node in record["nodes"]:
        features = node.get("features") or {}
        expression = features.get("expression")
        behaviour = features.get("behaviour")
        gaze = features.get("gaze_label")
        people.append({
            "person_id": node.get("person_id"),
            "role": node.get("role"),
            "bbox": features.get("bbox"),
            "expression": {"label": expression} if expression else None,
            "behaviour": {"label": behaviour} if behaviour else None,
            "head_pose": {"gaze_label": gaze} if gaze else None,
            "posture": features.get("posture"),
            # Already decided by Stage 3/4 -- carried, not recomputed.
            "engagement": features.get("engagement"),
            "eyes_closed": features.get("eyes_closed"),
            "rolling_engagement_pct": features.get("rolling_engagement_pct"),
            "is_sustained_distracted": features.get("is_sustained_distracted"),
            "is_eyes_closed_sustained": features.get("is_eyes_closed_sustained"),
            "is_poster": features.get("is_poster"),
            # What the student was doing, from backend/actions.py. Only Stage 3
            # graphs carry it; a Stage 1 record has no such field and the
            # profile simply reports no actions rather than inventing them.
            "action": features.get("action"),
            # Orientation relative to the room's measured focus
            # (backend/scene_layout.py), independent of any action.
            "oriented": features.get("oriented"),
            "layout": features.get("layout"),
        })
    return people, True


def _summarise_posture(samples: list[dict]) -> dict[str, object]:
    """Aggregate one student's posture geometry over the video.

    Args:
        samples: Per-frame posture dicts (or ``None`` entries) for one student.

    Returns:
        Frames with and without body keypoints, and the mean forward lean over
        the frames that had it. Raw geometry only -- this deliberately does not
        classify posture into "slouching"/"upright", because nothing in this
        project has validated such a mapping (see :mod:`backend.posture`).
    """
    present = [s for s in samples if s]
    leans = [
        s["vertical_lean"] for s in present if s.get("vertical_lean") is not None
    ]
    return {
        "frames_with_keypoints": len(present),
        "frames_without_keypoints": len(samples) - len(present),
        "mean_vertical_lean": (sum(leans) / len(leans)) if leans else None,
    }


def _engagement_pct(flags: list[bool | None]) -> float | None:
    """Share of readable frames the student faced the room's focus.

    Args:
        flags: Per-frame ``True``/``False``/``None`` from
            :func:`backend.scene_layout.annotate`.

    Returns:
        A percentage, or ``None`` when the student's shoulders were never
        readable.

    ``None`` frames are excluded rather than counted against the student: not
    having seen someone is not evidence that they looked away. This is reported
    beside ``on_task_pct`` and never merged with it -- one measures what a
    student was doing, the other where they were facing, and a single blended
    figure would obscure which evidence it rested on.
    """
    readable = [f for f in flags if f is not None]
    if not readable:
        return None
    return round(100.0 * sum(1 for f in readable if f) / len(readable), 1)


def _off_task_runs(actions, times, minimum_seconds):
    """Indices of off-task frames belonging to a run long enough to count.

    Args:
        actions: Per-frame action names.
        times: Per-frame timestamps in milliseconds, index-aligned.
        minimum_seconds: Shortest run that counts as off task.

    Returns:
        The set of frame indices inside a qualifying run.

    Behavioural-observation research treats behaviour as off task once it lasts
    beyond about five seconds. Below that it is a glance -- looking at a
    neighbour, checking the time, following someone walking past -- and
    counting glances turns ordinary human movement into disengagement. A
    student who glanced twelve times used to score the same as one who spent a
    minute on their phone.
    """
    # Per-frame labels flicker: a student on a phone reads as on_phone, then
    # head_down, then on_phone again as the hand moves. Segmenting naively
    # chops one long lapse into many short runs that each fail the test, and
    # the score then measures label stability rather than behaviour -- measured
    # on real footage, mean on-task swung from 47% to 95% across plausible
    # thresholds before this. Brief on-task interruptions inside an otherwise
    # off-task stretch are bridged, the standard debounce for event
    # segmentation.
    # Fixed, and deliberately not tied to `minimum_seconds`. Tying them made
    # the bridge grow with the threshold, so raising the threshold merged
    # ever-longer stretches and the sweep measured the two effects together
    # instead of separately.
    bridge_ms = FLICKER_BRIDGE_SECONDS * 1000
    spans: list[list[int]] = []
    for index, action in enumerate(actions):
        if action not in OFF_TASK_ACTIONS:
            continue
        if spans and times[index] - times[spans[-1][1]] <= bridge_ms:
            spans[-1][1] = index
        else:
            spans.append([index, index])

    counted: set[int] = set()
    for first, last in spans:
        if times[last] - times[first] >= minimum_seconds * 1000:
            counted.update(range(first, last + 1))
    return counted


def _below_scene_coverage(person_id, seen_count, scene_of, scene_frames,
                          minimum_pct) -> bool:
    """Whether an identity appeared in too little of its scene to be a student.

    Args:
        person_id: The identity.
        seen_count: Frames it was seen in.
        scene_of: ``{person_id: scene}``.
        scene_frames: ``{scene: set of frame ids}``.
        minimum_pct: Required share of the scene.

    Returns:
        ``True`` when the identity is a fragment. Returns ``False`` when the
        scene is unknown or empty -- a missing scene id is not evidence against
        a student, and single-video callers have no scenes at all.
    """
    if minimum_pct <= 0:
        return False
    total = len(scene_frames.get(scene_of.get(person_id), ()))
    if total <= 0:
        return False
    return (100.0 * seen_count / total) < minimum_pct


def _on_task_pct(actions: list[str | None], times=None,
                 minimum_seconds: float = 0.0) -> float | None:
    """Share of graded frames the student was NOT visibly off task.

    Args:
        actions: Per-frame action names, ``None`` where nothing was graded.
        times: Per-frame timestamps in milliseconds, for the glance filter.
        minimum_seconds: Off-task runs shorter than this are treated as
            glances and not counted against the student.

    Returns:
        A percentage, or ``None`` when nothing was graded at all.

    Unlike ``concentration``, this can actually reach a low number, because
    :mod:`backend.actions` produces positive evidence of being off task -- a
    detected phone, closed eyes, a head turned away -- rather than inferring
    it from the absence of a behaviour label. A profile that reports 100% here
    means no off-task evidence was seen, not that none could be.
    """
    graded = [a for a in actions if a and a != "unknown"]
    if not graded:
        return None
    if times and minimum_seconds > 0:
        counted = _off_task_runs(actions, times, minimum_seconds)
        off = sum(1 for i, a in enumerate(actions)
                  if a and a != "unknown" and i in counted)
    else:
        off = sum(1 for a in graded if a in OFF_TASK_ACTIONS)
    return round(100.0 * (len(graded) - off) / len(graded), 1)


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


def _phone_overlaps(person_bbox, objects, cfg) -> bool:
    """Whether a phone-class object overlaps this student's box at all.

    Independent of the fine-tuned behaviour model, so it still yields an
    off-task signal when that model produces nothing. Uses "any positive
    overlap" rather than an IoU threshold, matching
    :data:`~backend.config.AttentionConfig.device_proximity_iou`'s own default
    of ``0.0`` and its reasoning: a phone box and a student box barely overlap
    even when the phone is plainly in that student's hands.

    Args:
        person_bbox: The student's ``[x, y, w, h]``.
        objects: The frame's ``objects`` list.
        cfg: The full config (reads ``engagement.fallback_off_task_objects``).

    Returns:
        ``True`` if any configured phone-class object overlaps.
    """
    if not cfg.engagement.use_object_fallback:
        return False
    px, py, pw, ph = person_bbox
    for obj in objects:
        if obj.get("cls") not in cfg.engagement.fallback_off_task_objects:
            continue
        ox, oy, ow, oh = obj["bbox"]
        if (
            max(0, min(px + pw, ox + ow) - max(px, ox)) > 0
            and max(0, min(py + ph, oy + oh) - max(py, oy)) > 0
        ):
            return True
    return False


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
    off_task_evidence: dict[int, bool] = defaultdict(bool)
    # Set by tools/reject_static_faces.py when an identity was measured to
    # be a printed face (wall poster) rather than a student.
    is_poster: dict[int, bool] = defaultdict(bool)
    # Signals the graph already carries. Previously unreachable here, because
    # this module only ever read Stage 1.
    gaze_labels: dict[int, list[str | None]] = defaultdict(list)
    action_labels: dict[int, list[str | None]] = defaultdict(list)
    action_times: dict[int, list[int]] = defaultdict(list)
    scene_frames: dict[object, set] = defaultdict(set)
    scene_of: dict[int, object] = {}
    oriented_flags: dict[int, list[bool | None]] = defaultdict(list)
    layout_kinds: dict[int, list[str]] = defaultdict(list)
    posture_samples: dict[int, list[dict | None]] = defaultdict(list)
    rolling_pct: dict[int, list[float]] = defaultdict(list)
    sustained_distracted: dict[int, int] = defaultdict(int)
    eyes_closed_sustained: dict[int, int] = defaultdict(int)
    graph_roles: dict[int, str] = {}

    with src.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            ts_ms = record["timestamp_ms"]
            people, from_graph = _people_in(record)
            for person in people:
                person_id = person.get("person_id")
                if person_id is None:
                    continue

                if person.get("is_poster"):
                    is_poster[person_id] = True

                posture_samples[person_id].append(person.get("posture"))
                if from_graph:
                    if person.get("role"):
                        graph_roles[person_id] = person["role"]
                    pct = person.get("rolling_engagement_pct")
                    if pct is not None:
                        rolling_pct[person_id].append(float(pct))
                    if person.get("is_sustained_distracted"):
                        sustained_distracted[person_id] += 1
                    if person.get("is_eyes_closed_sustained"):
                        eyes_closed_sustained[person_id] += 1

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
                gaze_labels[person_id].append(gaze_label)
                action_labels[person_id].append(person.get("action"))
                action_times[person_id].append(record.get("timestamp_ms", 0))
                scene = record.get("scene")
                scene_frames[scene].add(record.get("frame_id"))
                scene_of.setdefault(person_id, scene)
                oriented_flags[person_id].append(person.get("oriented"))
                if person.get("layout"):
                    layout_kinds[person_id].append(person["layout"])

                # Fallback signals, used when the behaviour model produced
                # nothing -- both already computed by the pipeline and
                # previously discarded. See EngagementConfig for why.
                face = person.get("face")
                eyes_closed = person.get("eyes_closed")
                if eyes_closed is None and face and face.get("ear") is not None:
                    eyes_closed = face["ear"] < cfg.face.ear_closed_threshold
                phone_nearby = _phone_overlaps(
                    person.get("bbox") or [0, 0, 0, 0], record.get("objects", []), cfg
                )

                if from_graph and person.get("engagement") is not None:
                    # Stage 3 already applied exactly this rule and Stage 4
                    # refined it over time. Re-deriving it here would be a
                    # second, silently diverging copy of the precedence logic.
                    engagement_verdicts[person_id].append(person["engagement"])
                else:
                    engagement_verdicts[person_id].append(
                        classify_engagement(
                            gaze_label,
                            behaviour_label,
                            cfg.engagement,
                            phone_nearby=phone_nearby,
                            eyes_closed=eyes_closed,
                        )
                    )
                # Track whether ANY off-task-capable evidence was ever
                # available for this student, across all three routes. Used
                # below to decide whether a 100% score is a real finding or
                # just absence of evidence.
                if (
                    behaviour_label is not None
                    or phone_nearby
                    or eyes_closed is not None
                ):
                    off_task_evidence[person_id] = True

    profiles: dict[int, dict[str, object]] = {}
    for person_id, seen_count in frames_seen.items():
        behaviour_summary = _tally(behaviour_labels[person_id])
        behaviour_summary["weak_labels"] = sorted(behaviour_reliability[person_id])
        concentration = summarise_engagement(engagement_verdicts[person_id])

        # "off" is only reachable through concrete evidence -- a behaviour
        # reading, a nearby phone, or eyes-closed-while-head-down (see
        # backend.engagement; a bare non-attending gaze is deliberately left
        # "unknown", never "off"). If NONE of those three was ever available
        # for this student, "off" was structurally unreachable and a resulting
        # 100% concentration_pct is an absence-of-evidence artifact, not a
        # finding.
        #
        # Found for real: the behaviour model returned zero readings across an
        # entire out-of-distribution video (same generalization failure as
        # CHALLENGES_AND_SOLUTIONS.md section 18) and every student silently
        # read ~100%. The object/eye-closure fallbacks were added because of
        # that, so this check now covers all three routes rather than only the
        # behaviour one -- otherwise it would keep flagging students who DO
        # have usable fallback evidence.
        if not off_task_evidence[person_id] and concentration["frames"] > 0:
            concentration["off_task_detectable"] = False
            # Prepended onto the standing behavioral-proxy caveat that
            # summarise_engagement always attaches, so BOTH honesty layers
            # survive in the emitted record (this branch used to overwrite
            # the standing one).
            concentration["caveat"] = (
                "No off-task evidence of any kind was available for this "
                "student (no behaviour reading, no nearby phone detection, and "
                "no eye-closure measurement), so off-task states could not be "
                "detected at all -- the score reflects gaze only and should "
                "not be read as a real attentiveness score. "
                + concentration["caveat"]
            )
        else:
            concentration["off_task_detectable"] = True

        # Reject entries that are not students. Marked rather than deleted
        # (ProfileConfig.report_rejected) so a reviewer can see what was
        # dropped and why -- silently discarding detections is how a pipeline
        # starts misreporting its own recall.
        rejected: str | None = None
        # The graph already assigned a role; honour it rather than re-deriving.
        role = graph_roles.get(person_id, "student")
        if person_id in cfg.profile.instructor_ids:
            # Stated by whoever ran the video, not inferred: four geometric
            # signals were measured and none separated the teacher from the
            # students (see ProfileConfig.instructor_ids). Reported, not
            # deleted, but kept out of the student roster -- the audited video
            # had the teacher as its highest-sighting "student".
            role = "instructor"
            rejected = (
                "instructor: named as the person teaching this recording, so "
                "not counted as a student (role is declared, not inferred -- "
                "no measured signal separates the two on this footage)"
            )
        elif is_poster[person_id]:
            rejected = (
                "printed face: appearance did not change across its sightings, "
                "measured as a wall poster/portrait rather than a student "
                "(see backend/identity.py appearance_invariance)"
            )
        elif cfg.profile.require_face_verified and person_id < 0:
            rejected = (
                "unidentified: no face good enough to establish who this is, so "
                "this detection cannot be attributed to a student (reported as "
                "unidentified rather than counted as one -- see "
                "ProfileConfig.require_face_verified)"
            )
        elif _below_scene_coverage(person_id, seen_count, scene_of, scene_frames,
                                   cfg.profile.min_scene_coverage_pct):
            scene = scene_of.get(person_id)
            total = len(scene_frames.get(scene, ())) or 1
            rejected = (
                f"fragment: present in {seen_count} of {total} frames "
                f"({100 * seen_count / total:.1f}%) of its scene, below the "
                f"{cfg.profile.min_scene_coverage_pct:.0f}% minimum -- too brief "
                f"to be a student sitting in the room, and an absolute frame "
                f"count cannot say so because three frames means something "
                f"different in a 20-frame clip and a 700-frame scene"
            )
        elif seen_count < cfg.profile.min_frames_for_profile:
            rejected = (
                f"transient: seen in only {seen_count} frame(s), below the "
                f"{cfg.profile.min_frames_for_profile}-frame minimum -- almost "
                f"certainly detection noise rather than a student"
            )

        profiles[person_id] = {
            "person_id": person_id,
            # A negative id means this student was never matched by face --
            # see backend/identity.py. Carried forward explicitly so a report
            # cannot present an unverified id as a confirmed re-identification.
            "face_verified": person_id > 0,
            "role": role,
            "is_student": rejected is None,
            "rejected_reason": rejected,
            "frames_seen": seen_count,
            "first_seen_ms": first_ms[person_id],
            "last_seen_ms": last_ms[person_id],
            "duration_ms": last_ms[person_id] - first_ms[person_id],
            "expression": _tally(expression_labels[person_id]),
            "behaviour": behaviour_summary,
            # SUPERSEDED by on_task_pct and engagement_pct, and kept only
            # because backend/reporting.py and tools/dashboard.py still read
            # it. It disagrees with both -- it can only reach "on task" through
            # a gaze label matched against a global reference, which cannot be
            # right for every seat, so on real footage it reported 0% for
            # students the action layer scored at 50%. Removing it is a
            # cross-owner change, not a local one; see docs/PROJECT_REVIEW.md.
            "concentration": {**concentration, "superseded_by":
                              ["on_task_pct", "engagement_pct"]},
            # Where the student was looking, frame by frame, tallied. Head-pose
            # gaze is the one attention signal available on every face, and it
            # was previously consumed to compute concentration and then thrown
            # away, so a reader could not see what the verdict rested on.
            "attention": _tally(gaze_labels[person_id]),
            # What they were doing, as opposed to where they were looking.
            "actions": _tally(action_labels[person_id]),
            "on_task_pct": _on_task_pct(
                action_labels[person_id], action_times[person_id],
                cfg.temporal.min_off_task_seconds),
            # Deliberately a SECOND number rather than folded into the first.
            # One is about what a student did, the other about where they
            # faced; averaging them would hide which evidence a score rests on.
            "engagement_pct": _engagement_pct(oriented_flags[person_id]),
            "layout": (Counter(layout_kinds[person_id]).most_common(1)[0][0]
                       if layout_kinds[person_id] else None),
            "posture": _summarise_posture(posture_samples[person_id]),
            # Only populated from a Stage 4 input; a Stage 1 file has no
            # temporal analysis to carry, and reporting zeros there would read
            # as "measured, none found" instead of "not measured".
            "temporal": (
                {
                    "mean_rolling_engagement_pct": (
                        sum(rolling_pct[person_id]) / len(rolling_pct[person_id])
                        if rolling_pct[person_id]
                        else None
                    ),
                    "frames_sustained_distracted": sustained_distracted[person_id],
                    "frames_eyes_closed_sustained": eyes_closed_sustained[person_id],
                }
                if person_id in graph_roles
                else None
            ),
        }

    if not cfg.profile.report_rejected:
        profiles = {k: v for k, v in profiles.items() if v["is_student"]}
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
