"""Derive what a student is *doing* from signals the pipeline already produces.

The project has a behaviour classifier for exactly this -- reading, writing,
using a device, sleeping -- but it needs fine-tuned weights that are not in the
repository, so ``behaviour`` comes back ``null`` on every frame and the scene
graph carries no actions at all.

This fills that gap without training anything. Three signals the pipeline
already computes are enough for the actions that matter in a classroom:

* **objects** -- the detector is already configured to find ``cell phone``,
  ``laptop`` and ``book`` (:data:`DetectionConfig.object_whitelist`), and does:
  on the audited clip it found 37 books, 4 laptops and 1 phone. An object
  overlapping a student's box is direct evidence of what they are handling.
* **head pose** -- pitch-down separates "reading the desk" from "watching the
  front", and yaw separates "looking at a neighbour" from either.
* **eye aspect ratio** -- eyes closed for a sustained period.

What this is not
----------------

It is a rule over observable evidence, not a trained action recogniser. It
cannot tell reading from writing -- both are a book plus a bowed head -- which
is precisely the distinction the literature says is hardest (SCB-ST-Dataset4
reaches only 57.8% on writing with a strong temporal model). So those two are
deliberately reported as one ``studying`` action rather than guessed apart.

Every action carries the evidence that produced it, so a reviewer can see
whether "on phone" came from a detected phone or from a bowed head.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Ordered by how strongly each overrides the others. The first action whose
#: rule fires wins, so a visible phone beats a head-down posture that would
#: otherwise read as studying.
PRIORITY = (
    "on_phone",
    "studying",
    "on_laptop",
    "eyes_closed",
    "looking_away",
    "head_down",
    "attentive",
    "unknown",
)

#: Actions that mean the student is not engaged with the lesson.
OFF_TASK = frozenset({"on_phone", "eyes_closed", "looking_away"})

LABELS = {
    "on_phone": "on phone",
    "studying": "reading / writing",
    "on_laptop": "on laptop",
    "eyes_closed": "eyes closed",
    "looking_away": "looking away",
    "head_down": "head down",
    "attentive": "attentive",
    "unknown": "no face read",
}


@dataclass(frozen=True)
class Action:
    """One student's action in one frame.

    Attributes:
        name: A key from :data:`PRIORITY`.
        evidence: Short human-readable reason, e.g. ``"cell phone overlap"``.
        off_task: Whether this action counts against engagement.
        obj: The object class this action was read from, if any. Carried so
            the scene graph can hold a person-object edge -- which is a real
            relation even when only one person is present.
    """

    name: str
    evidence: str
    off_task: bool
    obj: str | None = None

    @property
    def label(self) -> str:
        """Display form, e.g. ``"on phone"``."""
        return LABELS.get(self.name, self.name)


def _overlaps(person_bbox, obj_bbox, min_fraction: float = 0.0) -> bool:
    """Whether an object box overlaps a person box.

    Args:
        person_bbox: ``(x, y, w, h)``.
        obj_bbox: ``(x, y, w, h)``.
        min_fraction: Minimum share of the OBJECT that must fall inside the
            person box. Judged against the object, not the union, because a
            phone is tiny next to a person and IoU would never fire.

    Returns:
        ``True`` when they overlap by at least ``min_fraction``.
    """
    px, py, pw, ph = person_bbox
    ox, oy, ow, oh = obj_bbox
    ix = max(0.0, min(px + pw, ox + ow) - max(px, ox))
    iy = max(0.0, min(py + ph, oy + oh) - max(py, oy))
    inter = ix * iy
    area = max(ow * oh, 1e-6)
    return inter / area > min_fraction


def classify(
    person_bbox,
    objects,
    gaze_label: str | None,
    pitch: float | None,
    ear: float | None,
    config=None,
) -> Action:
    """Decide what one student is doing this frame.

    Args:
        person_bbox: The student's box, ``(x, y, w, h)``.
        objects: This frame's detected objects, dicts with ``cls`` and
            ``bbox``.
        gaze_label: Head-pose gaze class, or ``None`` if no face was read.
        pitch: Head pitch in degrees, positive downward, or ``None``.
        ear: Eye aspect ratio, or ``None``.
        config: Full pipeline config, for thresholds. Defaults to ``CONFIG``.

    Returns:
        The highest-priority :class:`Action` whose rule fires. With no face
        read at all the answer is ``unknown``, not ``attentive`` -- there is no
        evidence of anything, and reading attentiveness out of silence is the
        confident wrong answer this project exists to avoid.
    """
    from backend.config import CONFIG

    cfg = config if config is not None else CONFIG

    near = {}
    for obj in objects or ():
        cls = obj.get("cls")
        bbox = obj.get("bbox")
        if cls and bbox and _overlaps(person_bbox, bbox):
            near[cls] = True

    down = pitch is not None and pitch >= cfg.headpose.pitch_down_threshold

    if near.get("cell phone"):
        return Action("on_phone", "cell phone overlap", True, "cell phone")
    if near.get("book"):
        return Action("studying", "book overlap" + (" + head down" if down else ""),
                      False, "book")
    if near.get("laptop"):
        return Action("on_laptop", "laptop overlap", False, "laptop")
    if ear is not None and ear < cfg.face.ear_closed_threshold:
        return Action("eyes_closed", f"eye aspect ratio {ear:.2f}", True)
    if gaze_label in ("left", "right", "back"):
        return Action("looking_away", f"gaze {gaze_label}", True)
    if down:
        return Action("head_down", f"pitch {pitch:.0f} deg", False)
    if gaze_label is None:
        return Action("unknown", "no face read", False)
    return Action("attentive", f"gaze {gaze_label}", False)


def annotate_graph(graph: dict, record: dict, config=None) -> dict:
    """Add an ``action`` to every node of a scene graph, in place.

    Args:
        graph: A scene graph from :func:`backend.scene_graph.generate_scene_graph`.
        record: The Stage 1 record the graph was built from, for objects and
            the per-person eye aspect ratio.
        config: Full pipeline config.

    Returns:
        The same graph, with ``features.action`` and ``features.action_evidence``
        filled in, and ``shared_action`` edges added between students doing the
        same thing.
    """
    objects = record.get("objects") or []
    by_person = {}
    for person in record.get("persons", []):
        by_person[person.get("person_id")] = person

    actions: dict[int, str] = {}
    for node in graph.get("nodes", []):
        feat = node.setdefault("features", {})
        person = by_person.get(node.get("person_id"))
        if person is None:
            continue
        head = person.get("head_pose") or {}
        face = person.get("face") or {}
        action = classify(
            person.get("bbox"),
            objects,
            feat.get("gaze_label"),
            head.get("pitch"),
            face.get("ear"),
            config,
        )
        feat["action"] = action.name
        feat["action_evidence"] = action.evidence
        feat["object"] = action.obj
        if node.get("role") == "student":
            actions[node["id"]] = action.name

    # Two students doing the same thing at the same time is the relation worth
    # drawing: it is what turns the picture from a seating chart into something
    # about behaviour.
    ids = sorted(actions)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if actions[a] != actions[b] or actions[a] in ("attentive", "unknown"):
                continue
            graph.setdefault("edges", []).append({
                "type": "shared_action",
                "source": a,
                "target": b,
                "features": {"action": actions[a]},
            })
    return graph
