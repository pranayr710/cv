"""Name what a student is doing, from signals the pipeline already produces.

The project has a behaviour classifier for this -- reading, writing, using a
device, sleeping -- but it needs fine-tuned weights that are not in the
repository, so ``behaviour`` comes back ``null`` on every frame and the scene
graph carried no actions at all.

This fills the gap without training anything, using three sources that were all
being computed and then discarded:

* **Objects.** YOLO detects all 80 COCO classes regardless; the whitelist only
  decides which reach the output. A bottle at the mouth is drinking, a keyboard
  under the hands is typing, a phone is a phone. Widening that list from three
  classes to seventeen turned "what is this student holding" from unanswerable
  into direct evidence.
* **Hands.** MediaPipe Pose returns 33 landmarks and the pipeline extracted 5.
  Wrists and elbows came free with the same inference. They are what separate a
  raised hand from a resting one, writing from reading, and a head propped on a
  palm from a head merely bowed.
* **Face and head pose.** Eye aspect ratio for closed eyes, mouth opening for a
  yawn, pitch for a bowed head, yaw for a turned one.

Confidence is part of the answer
--------------------------------

Actions are not equally well evidenced, and pretending otherwise is how a demo
becomes a lie. Each :class:`Action` carries a ``confidence``:

* ``"direct"`` -- a detected object overlapping the student, or landmark
  geometry that is unambiguous (a wrist held clear above the head is a raised
  hand).
* ``"inferred"`` -- a plausible reading of weaker geometry. ``writing`` is the
  clearest case: reading and writing differ only by what the hands are doing,
  and the closest published work reaches 57.8% on writing even with a strong
  temporal model. So writing is reported as inferred, and when the hands are
  not visible at all the answer falls back to ``studying`` rather than guessing.

Every action also carries the evidence that produced it, so "on phone" can be
traced to a detected phone rather than an assumption.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Evaluated in order; the first rule that fires wins. The ordering encodes
#: which evidence overrides which -- a visible phone beats a bowed head,
#: because the bowed head is *why* they are looking at the phone.
PRIORITY = (
    "raising_hand",
    "on_phone",
    "drinking",
    "eating",
    "typing",
    "writing",
    "reading",
    "studying",
    "on_laptop",
    "yawning",
    "eyes_closed",
    "head_on_hand",
    "looking_away",
    "head_down",
    "slouching",
    "leaning_forward",
    "attentive",
    "unknown",
)

#: Actions that count against engagement. Deliberately narrow: a bowed head or
#: a slouch is posture, not misbehaviour, and counting those as off task would
#: penalise students for reading.
OFF_TASK = frozenset({"on_phone", "eyes_closed", "looking_away", "yawning"})

#: Actions that are positive evidence of participation.
ON_TASK = frozenset({"raising_hand", "writing", "reading", "studying", "typing",
                     "on_laptop", "attentive", "leaning_forward"})

LABELS = {
    "raising_hand": "raising hand",
    "on_phone": "on phone",
    "drinking": "drinking",
    "eating": "eating",
    "typing": "typing",
    "writing": "writing",
    "reading": "reading",
    "studying": "reading or writing",
    "on_laptop": "on laptop",
    "yawning": "yawning",
    "eyes_closed": "eyes closed",
    "head_on_hand": "head resting on hand",
    "looking_away": "looking away",
    "head_down": "head down",
    "slouching": "slouching",
    "leaning_forward": "leaning forward",
    "attentive": "attentive",
    "unknown": "no face read",
}

#: Object classes that mean drinking when held up near the head.
DRINK_CLASSES = frozenset({"bottle", "cup"})
#: Object classes that mean eating when held up near the head.
FOOD_CLASSES = frozenset({"sandwich", "apple", "banana", "donut", "pizza"})
#: Object classes that mean typing when under the hands.
TYPING_CLASSES = frozenset({"keyboard", "mouse"})

#: MediaPipe Face Mesh indices for the inner lip, used for mouth opening. Inner
#: rather than outer, so lip colour or a beard cannot inflate the measurement.
_UPPER_LIP, _LOWER_LIP = 13, 14
_LEFT_CORNER, _RIGHT_CORNER = 78, 308

#: Milliseconds the eyes must stay continuously shut before it counts as
#: "eyes closed" rather than a blink. Human blinks run 100-400 ms; 600 ms sits
#: clear of the longest of them while still catching a genuine closure well
#: before anyone would call it sleep. Without this every blink was an off-task
#: frame -- 12.5% of frames on a real session.
BLINK_MS = 600

#: Mouth opening (vertical over horizontal) above which a face reads as a yawn.
#:
#: Was 0.55, reasoned from an assumption that speech peaks near 0.35. Measured
#: on a real session that assumption was wrong by a factor of two: across 192
#: readings the mouth ratio ran 0.000 to 0.274 with a median of 0.039, so the
#: threshold sat above anything the signal could reach and the yawn label was
#: unreachable rather than merely rare.
#:
#: 0.35 sits about 28% above the highest non-yawn opening observed, which keeps
#: it clear of speech while being attainable. It remains UNCALIBRATED in the
#: strict sense -- no labelled yawn has been measured -- so the action is
#: reported as inferred, and calibrating it needs footage of somebody actually
#: yawning.
YAWN_RATIO = 0.35


@dataclass(frozen=True)
class Action:
    """One student's action in one frame.

    Attributes:
        name: A key from :data:`PRIORITY`.
        evidence: Short human-readable reason, e.g. ``"cell phone overlap"``.
        off_task: Whether this counts against engagement.
        obj: The object class the action was read from, if any. Carried so the
            scene graph can hold a person-object edge.
        confidence: ``"direct"`` for detected objects and unambiguous landmark
            geometry, ``"inferred"`` for a plausible reading of weaker
            evidence. Surfaced in the interface so an inferred label is never
            mistaken for a measured one.
    """

    name: str
    evidence: str
    off_task: bool
    obj: str | None = None
    confidence: str = "direct"

    @property
    def label(self) -> str:
        """Display form, e.g. ``"raising hand"``."""
        return LABELS.get(self.name, self.name)


def _overlaps(person_bbox, obj_bbox, region: str = "any") -> bool:
    """Whether an object box overlaps a person box, optionally only part of it.

    Args:
        person_bbox: ``(x, y, w, h)``.
        obj_bbox: ``(x, y, w, h)``.
        region: ``"any"``, ``"head"`` (top third) or ``"lap"`` (bottom half).
            A bottle at the mouth is drinking; the same bottle on the desk is
            not, and only the region separates them.

    Returns:
        ``True`` when any part of the object falls inside the chosen region.
        Judged against the object's own area rather than IoU, because a phone
        is tiny beside a person and IoU would never fire.
    """
    px, py, pw, ph = person_bbox
    if region == "head":
        ph = ph / 3.0
    elif region == "lap":
        py, ph = py + ph / 2.0, ph / 2.0
    ox, oy, ow, oh = obj_bbox
    ix = max(0.0, min(px + pw, ox + ow) - max(px, ox))
    iy = max(0.0, min(py + ph, oy + oh) - max(py, oy))
    return (ix * iy) / max(ow * oh, 1e-6) > 0.0


def mouth_open_ratio(landmarks) -> float | None:
    """Vertical mouth opening over mouth width, from Face Mesh landmarks.

    Args:
        landmarks: The 468-point landmark list, or ``None``.

    Returns:
        The ratio, or ``None`` if landmarks are absent or degenerate. Scale
        free, so it does not change with how close the student is sitting.
    """
    if not landmarks or len(landmarks) <= _RIGHT_CORNER:
        return None
    try:
        upper, lower = landmarks[_UPPER_LIP], landmarks[_LOWER_LIP]
        left, right = landmarks[_LEFT_CORNER], landmarks[_RIGHT_CORNER]
        vertical = abs(float(lower[1]) - float(upper[1]))
        horizontal = abs(float(right[0]) - float(left[0]))
    except (IndexError, TypeError, ValueError):
        return None
    return vertical / horizontal if horizontal > 1e-6 else None


def _hand_up(posture, person_bbox) -> tuple[bool, str]:
    """Whether either wrist is genuinely raised, not resting against the face.

    Args:
        posture: The person's posture dict, or ``None``.
        person_bbox: ``(x, y, w, h)``, for scale.

    Returns:
        ``(raised, evidence)``.

    Measured against the NOSE, not the shoulders. A hand propping up a bored
    head also sits above the shoulders, so a shoulder test called that a raised
    hand -- the two most opposite states in a classroom collapsing into one
    label. A raised hand clears the head and is held out from it; a propped one
    is at face height and touching. Both conditions are required.
    """
    if not posture:
        return False, ""
    nose = posture.get("nose")
    if not nose:
        return False, ""
    margin = person_bbox[3] * 0.04
    near = person_bbox[3] * 0.18
    for side in ("left_wrist", "right_wrist"):
        wrist = posture.get(side)
        if not wrist:
            continue
        if float(wrist[1]) >= float(nose[1]) - margin:
            continue
        dx = float(wrist[0]) - float(nose[0])
        dy = float(wrist[1]) - float(nose[1])
        if (dx * dx + dy * dy) ** 0.5 < near:
            continue  # touching the head: propped, not raised
        return True, f"{side.split('_')[0]} wrist raised above the head"
    return False, ""


def _hand_near_face(posture, person_bbox) -> bool:
    """Whether a wrist sits close to the nose, i.e. a head propped on a hand."""
    if not posture:
        return False
    nose = posture.get("nose")
    if not nose:
        return False
    near = person_bbox[3] * 0.18
    for side in ("left_wrist", "right_wrist"):
        wrist = posture.get(side)
        if not wrist:
            continue
        dx = float(wrist[0]) - float(nose[0])
        dy = float(wrist[1]) - float(nose[1])
        if (dx * dx + dy * dy) ** 0.5 < near:
            return True
    return False


def _hands_low(posture, person_bbox) -> bool:
    """Whether every visible wrist sits well below the shoulders, near a desk."""
    if not posture:
        return False
    shoulder = posture.get("shoulder_mid")
    if not shoulder:
        return False
    drop = person_bbox[3] * 0.15
    seen = [posture.get("left_wrist"), posture.get("right_wrist")]
    seen = [w for w in seen if w]
    return bool(seen) and all(float(w[1]) > float(shoulder[1]) + drop for w in seen)


def classify(
    person_bbox,
    objects,
    gaze_label: str | None,
    pitch: float | None,
    ear: float | None,
    config=None,
    posture: dict | None = None,
    landmarks=None,
    oriented: bool | None = None,
    eyes_closed_ms: int | None = None,
) -> Action:
    """Decide what one student is doing this frame.

    Args:
        person_bbox: The student's box, ``(x, y, w, h)``.
        objects: This frame's detected objects, dicts with ``cls`` and ``bbox``.
        gaze_label: Head-pose gaze class, or ``None`` if no face was read.
        pitch: Head pitch in degrees, positive downward, or ``None``.
        ear: Eye aspect ratio, or ``None``.
        config: Full pipeline config, for thresholds. Defaults to ``CONFIG``.
        posture: The student's posture dict (wrists, shoulders, lean), or
            ``None``.
        landmarks: Face Mesh landmarks, for the mouth-opening ratio.

    Returns:
        The highest-priority :class:`Action` whose rule fires. With no face and
        no posture the answer is ``unknown``, never ``attentive`` -- there is no
        evidence of anything, and reading attentiveness out of silence is the
        confident wrong answer this project exists to avoid.
    """
    from backend.config import CONFIG

    cfg = config if config is not None else CONFIG

    near_any, near_head = set(), set()
    for obj in objects or ():
        cls, bbox = obj.get("cls"), obj.get("bbox")
        if not cls or not bbox:
            continue
        if _overlaps(person_bbox, bbox):
            near_any.add(cls)
        if _overlaps(person_bbox, bbox, "head"):
            near_head.add(cls)

    down = pitch is not None and pitch >= cfg.headpose.pitch_down_threshold

    raised, why = _hand_up(posture, person_bbox)
    if raised:
        return Action("raising_hand", why, False)

    if "cell phone" in near_any:
        return Action("on_phone", "cell phone overlap", True, "cell phone")

    drink = near_head & DRINK_CLASSES
    if drink:
        obj = min(drink)
        return Action("drinking", f"{obj} at head height", False, obj)

    food = near_head & FOOD_CLASSES
    if food:
        obj = min(food)
        return Action("eating", f"{obj} at head height", False, obj)

    typing = near_any & TYPING_CLASSES
    if typing:
        obj = min(typing)
        return Action("typing", f"{obj} overlap", False, obj)

    if "book" in near_any:
        # Reading and writing differ only by what the hands are doing. When the
        # hands are visible the geometry is worth reporting, marked inferred;
        # when they are not, the honest answer is that it is one or the other.
        if _hands_low(posture, person_bbox):
            return Action("writing", "book with hands low at the desk", False,
                          "book", "inferred")
        if posture and posture.get("shoulder_mid"):
            return Action("reading", "book with hands not at the desk", False,
                          "book", "inferred")
        return Action("studying", "book overlap, hands not visible", False, "book")

    if "laptop" in near_any:
        return Action("on_laptop", "laptop overlap", False, "laptop")

    ratio = mouth_open_ratio(landmarks)
    if ratio is not None and ratio > YAWN_RATIO:
        return Action("yawning", f"mouth opening ratio {ratio:.2f}", True,
                      confidence="inferred")

    # Shut eyes are only meaningful once they stay shut. With no duration we
    # cannot tell a blink from a closure, so this stays silent rather than
    # guessing -- a blink wrongly scored off task is worse than a closure missed.
    if (ear is not None and ear < cfg.face.ear_closed_threshold
            and eyes_closed_ms is not None and eyes_closed_ms >= BLINK_MS):
        return Action("eyes_closed", f"eyes shut {eyes_closed_ms / 1000:.1f}s", True)

    if _hand_near_face(posture, person_bbox):
        return Action("head_on_hand", "wrist close to the face", False,
                      confidence="inferred")

    # Prefer the room-relative answer when it exists. `gaze_label` compares
    # head yaw against a single global reference, which cannot be right for
    # every seat: on real footage students sat at yaws from -58 to +84 degrees
    # and a student facing the front read as "looking right" purely from where
    # they sat. `oriented` compares the student's shoulders against the focus
    # the room was measured to have, so it carries no camera constant at all.
    if oriented is False:
        return Action("looking_away", "facing away from the room focus", True)
    if oriented is None and gaze_label in ("left", "right", "back"):
        return Action("looking_away", f"gaze {gaze_label}", True,
                      confidence="inferred")

    if down:
        return Action("head_down", f"pitch {pitch:.0f} deg", False)

    lean = (posture or {}).get("vertical_lean")
    if lean is not None:
        # vertical_lean is nose.y - shoulder_mid.y over person height. More
        # negative means the head sits higher above the shoulders (upright);
        # closer to zero means it has sunk towards them.
        if lean > -0.05:
            return Action("slouching", f"lean {lean:+.2f}", False,
                          confidence="inferred")
        if lean < -0.30:
            return Action("leaning_forward", f"lean {lean:+.2f}", False,
                          confidence="inferred")

    if gaze_label is None:
        return Action("unknown", "no face read", False)
    return Action("attentive", f"gaze {gaze_label}", False)


def assign_objects(persons: list, objects: list) -> dict[int, list]:
    """Give each detected object to at most ONE person.

    Args:
        persons: This frame's person records, each with a ``bbox``.
        objects: This frame's detected objects.

    Returns:
        ``{index into persons: [objects]}``.

    Overlap alone is not ownership. Measured on a 10-minute lecture, **35% of
    detected phones overlapped more than one student box** -- some overlapped
    three, one overlapped five -- because students sit shoulder to shoulder and
    a phone on one desk intrudes into the neighbour's box. Attributing it to
    everyone it touches inflated ``on_phone`` to 30% of all actions and marked
    students off-task for sitting next to somebody.

    Each object goes to the person whose box contains the largest share of it.
    A phone is small relative to a person, so "share of the object inside the
    box" discriminates where IoU would not.
    """
    assignment: dict[int, list] = {}
    for obj in objects or ():
        bbox = obj.get("bbox")
        if not bbox:
            continue
        ox, oy, ow, oh = bbox
        area = max(ow * oh, 1e-6)
        best_index, best_share = None, 0.0
        for index, person in enumerate(persons):
            pbox = person.get("bbox")
            if not pbox:
                continue
            px, py, pw, ph = pbox
            ix = max(0.0, min(px + pw, ox + ow) - max(px, ox))
            iy = max(0.0, min(py + ph, oy + oh) - max(py, oy))
            share = (ix * iy) / area
            if share > best_share:
                best_index, best_share = index, share
        if best_index is not None and best_share > 0.0:
            assignment.setdefault(best_index, []).append(obj)
    return assignment


def annotate_graph(graph: dict, record: dict, config=None) -> dict:
    """Add an action to every node of a scene graph, in place.

    Args:
        graph: A scene graph from
            :func:`backend.scene_graph.generate_scene_graph`.
        record: The Stage 1 record the graph was built from, for objects,
            posture, landmarks and eye aspect ratio.
        config: Full pipeline config.

    Returns:
        The same graph, with ``features.action``, ``features.action_evidence``,
        ``features.action_confidence`` and ``features.object`` filled in, plus
        ``shared_action`` edges between students doing the same thing.
    """
    objects = record.get("objects") or []
    persons = record.get("persons", [])
    by_person = {p.get("person_id"): p for p in persons}
    # Exclusive: an object belongs to one student, not to everyone it touches.
    owned = assign_objects(persons, objects)
    index_of = {id(p): i for i, p in enumerate(persons)}

    actions: dict[int, str] = {}
    centres: dict[int, tuple[float, float]] = {}
    for node in graph.get("nodes", []):
        feat = node.setdefault("features", {})
        person = by_person.get(node.get("person_id"))
        if person is None:
            continue
        mine = owned.get(index_of.get(id(person), -1), [])
        head = person.get("head_pose") or {}
        face = person.get("face") or {}
        action = classify(
            person.get("bbox"),
            mine,
            feat.get("gaze_label"),
            head.get("pitch"),
            face.get("ear"),
            config,
            posture=person.get("posture") or feat.get("posture"),
            landmarks=face.get("landmarks"),
            # Set by backend.scene_layout.annotate, which must run first. When
            # present it replaces the camera-relative gaze test entirely.
            oriented=feat.get("oriented"),
            eyes_closed_ms=feat.get("eyes_closed_ms"),
        )
        feat["action"] = action.name
        feat["action_evidence"] = action.evidence
        feat["action_confidence"] = action.confidence
        feat["object"] = action.obj
        if node.get("role") == "student":
            actions[node["id"]] = action.name
            box = person.get("bbox")
            if box:
                centres[node["id"]] = (box[0] + box[2] / 2.0, box[1] + box[3] / 2.0)

    # Two students doing the same thing at the same time is the relation worth
    # drawing: it turns the picture from a seating chart into something about
    # behaviour. "attentive" is excluded because it is the default state --
    # linking everyone watching the front would bury the edges that mean
    # something.
    ids = sorted(actions)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if actions[a] != actions[b] or actions[a] in ("attentive", "unknown"):
                continue
            # The schema requires every edge to carry the full feature set,
            # so a consumer never has to branch on edge type to read a field.
            # The pair-interaction fields do not apply to a shared action and
            # are null rather than absent -- "not measured here" is a different
            # claim from "measured as zero".
            centre_a, centre_b = centres.get(a), centres.get(b)
            distance = (
                math.dist(centre_a, centre_b)
                if centre_a and centre_b else None
            )
            graph.setdefault("edges", []).append({
                "type": "shared_action",
                "source": a,
                "target": b,
                "features": {
                    "action": actions[a],
                    "distance_px": distance,
                    "oriented_fraction": None,
                    "shared_object_class": None,
                    "is_sustained_interaction": None,
                    "rolling_interaction_fraction": None,
                },
            })
    return graph
