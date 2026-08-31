"""Work out what the room is doing, so attention can be judged against it.

The pipeline's original model assumed one shape of room: students in rows
facing a teacher, where attention means *oriented toward the front*. Run that
model on a group discussion around a table and it inverts -- students facing
each other, which is exactly what collaboration looks like, gets labelled
"looking away" and scored as disengagement. On real footage that mislabelled
23% of every action.

So the room is measured rather than assumed.

Where do they look?
-------------------

Each student contributes a **ray**: their position, and the direction their
shoulders face. :mod:`backend.posture` already computes ``facing_direction`` in
image space from the shoulder line, which matters -- it never passes through
the camera-relative yaw that makes head pose uncomparable between seats.

Those rays are intersected in a least-squares sense. The point they converge on
is the room's **focus**: the thing the students are collectively oriented
toward. Nothing about a teacher, a board or a camera is assumed; the focus is
wherever the geometry says it is.

What kind of room is it?
------------------------

The focus lands in one of two places relative to the students themselves:

* **inside** the group -- they face each other. Group work. Attention means
  engagement with the group, and facing "away from the front" is correct
  behaviour.
* **outside**, beyond the seating -- they face a common external thing. A
  lecture. Attention means orientation toward that focus.

This is one measurement, not a per-video setting, so a recording that moves
between a lab table and a lecture room is handled without being told.

What it deliberately does not do
--------------------------------

It does not identify *what* the focus is. A convergence point outside the
seating could be a teacher, a screen, a window or a door. The geometry says
"they are collectively oriented at that place"; calling it a teacher would be
an assumption this module has no evidence for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Minimum students with a usable facing direction before a layout is claimed.
#: Two rays always intersect somewhere, which would make the answer arbitrary.
MIN_STUDENTS = 3

#: How far outside the seating the focus must fall, as a multiple of the
#: group's own radius, before the room reads as a lecture. Inside 1.0 is
#: literally among the students; beyond 1.6 is clear of them with room for
#: error in the facing estimates.
LECTURE_FOCUS_RATIO = 1.6

#: Degrees of misalignment still counted as oriented toward the focus. Matches
#: the +-20 degree yaw band the head-pose config already treats as "forward",
#: so the two agree rather than each carrying its own notion of straight ahead.
ORIENTED_WITHIN_DEG = 35.0


@dataclass(frozen=True)
class Layout:
    """What the room is doing, measured from one frame or a window of frames.

    Attributes:
        kind: ``"lecture"``, ``"group"`` or ``"unknown"``.
        focus: The point the students collectively face, in image pixels, or
            ``None`` when the rays do not converge.
        centre: Mean student position.
        radius: Mean distance from ``centre`` to the students -- the size of
            the group, used to judge whether the focus is inside it.
        ratio: ``|focus - centre| / radius``. Below 1.0 the focus sits among
            the students; above :data:`LECTURE_FOCUS_RATIO` it is clear of them.
        n: How many students contributed a facing ray.
    """

    kind: str
    focus: tuple[float, float] | None
    centre: tuple[float, float] | None
    radius: float
    ratio: float
    n: int


def _rays(people) -> list[tuple[float, float, float, float]]:
    """Extract ``(x, y, dx, dy)`` facing rays from person records.

    Args:
        people: Person records carrying ``bbox`` and ``posture``.

    Returns:
        One ray per student whose shoulders gave a usable facing direction.
    """
    rays = []
    for person in people:
        bbox = person.get("bbox")
        posture = person.get("posture") or {}
        facing = posture.get("facing_direction")
        if not bbox or not facing:
            continue
        dx, dy = float(facing[0]), float(facing[1])
        norm = math.hypot(dx, dy)
        if norm < 1e-6:
            continue
        x = bbox[0] + bbox[2] / 2.0
        y = bbox[1] + bbox[3] / 2.0
        rays.append((x, y, dx / norm, dy / norm))
    return rays


def _converge(rays) -> tuple[float, float] | None:
    """Least-squares point closest to every facing ray.

    Args:
        rays: ``(x, y, dx, dy)`` tuples.

    Returns:
        The convergence point, or ``None`` if the rays are parallel (no
        meaningful intersection -- everyone facing the same way in a corridor,
        for instance).

    Each ray contributes the constraint that the focus lies on the line through
    ``(x, y)`` with direction ``(dx, dy)``. Minimising the squared perpendicular
    distance to every line gives a 2x2 normal system, solved directly.
    """
    if len(rays) < 2:
        return None
    # Perpendicular form: n . p = n . a, where n is the ray normal.
    sxx = sxy = syy = bx = by = 0.0
    for x, y, dx, dy in rays:
        nx, ny = -dy, dx
        c = nx * x + ny * y
        sxx += nx * nx
        sxy += nx * ny
        syy += ny * ny
        bx += nx * c
        by += ny * c
    det = sxx * syy - sxy * sxy
    if abs(det) < 1e-9:
        return None
    return ((syy * bx - sxy * by) / det, (sxx * by - sxy * bx) / det)


def _fraction_facing(rays, point) -> float:
    """Share of students with ``point`` in front of them rather than behind.

    Args:
        rays: ``(x, y, dx, dy)`` facing rays.
        point: The candidate focus.

    Returns:
        A fraction in ``[0, 1]``.

    The least-squares fit works on infinite LINES, and a line is identical in
    either direction -- so a ring of students with their backs to the table
    converges on exactly the same point as a ring facing it. Without this
    check, people facing away from each other would be reported as
    collaborating, which is the precise opposite of the truth.
    """
    if not rays:
        return 0.0
    ahead = 0
    for x, y, dx, dy in rays:
        if dx * (point[0] - x) + dy * (point[1] - y) > 0:
            ahead += 1
    return ahead / len(rays)


#: Share of students that must have the focus in front of them for the
#: convergence to describe something they are collectively attending to.
MIN_FACING_FRACTION = 0.6


def detect(people) -> Layout:
    """Measure the room's layout from one frame's people.

    Args:
        people: Person records with ``bbox`` and ``posture``.

    Returns:
        A :class:`Layout`. ``kind`` is ``"unknown"`` when there are too few
        students or the rays do not converge -- an honest answer, since forcing
        a choice between lecture and group on two people would be a coin flip
        dressed as a measurement.
    """
    rays = _rays(people)
    if len(rays) < MIN_STUDENTS:
        return Layout("unknown", None, None, 0.0, 0.0, len(rays))

    cx = sum(r[0] for r in rays) / len(rays)
    cy = sum(r[1] for r in rays) / len(rays)
    radius = sum(math.hypot(r[0] - cx, r[1] - cy) for r in rays) / len(rays)

    focus = _converge(rays)
    if focus is None or radius < 1e-6:
        return Layout("unknown", focus, (cx, cy), radius, 0.0, len(rays))

    # The focus has to be in front of most of them, or the convergence is an
    # artefact of fitting undirected lines.
    if _fraction_facing(rays, focus) < MIN_FACING_FRACTION:
        return Layout("unknown", focus, (cx, cy), radius, 0.0, len(rays))

    ratio = math.hypot(focus[0] - cx, focus[1] - cy) / radius
    kind = "lecture" if ratio > LECTURE_FOCUS_RATIO else "group"
    return Layout(kind, focus, (cx, cy), radius, ratio, len(rays))


def oriented_toward(person, focus) -> tuple[bool, float] | tuple[None, None]:
    """Whether a student's shoulders point at the focus, and by what margin.

    Args:
        person: A person record with ``bbox`` and ``posture``.
        focus: The point to test against, from :func:`detect`.

    Returns:
        ``(is_oriented, degrees_off)``, or ``(None, None)`` when the student's
        facing direction could not be read. ``None`` is not ``False``: a
        student whose shoulders were not visible has not been shown to be
        looking away.
    """
    if focus is None:
        return (None, None)
    bbox = person.get("bbox")
    posture = person.get("posture") or {}
    facing = posture.get("facing_direction")
    if not bbox or not facing:
        return (None, None)

    x = bbox[0] + bbox[2] / 2.0
    y = bbox[1] + bbox[3] / 2.0
    fx, fy = float(facing[0]), float(facing[1])
    tx, ty = focus[0] - x, focus[1] - y
    fn, tn = math.hypot(fx, fy), math.hypot(tx, ty)
    if fn < 1e-6 or tn < 1e-6:
        return (None, None)

    cosine = max(-1.0, min(1.0, (fx * tx + fy * ty) / (fn * tn)))
    off = math.degrees(math.acos(cosine))
    return (off <= ORIENTED_WITHIN_DEG, off)


def summarise(layouts) -> Layout:
    """Reduce many per-frame layouts to one verdict for a scene.

    Args:
        layouts: Per-frame :class:`Layout` results.

    Returns:
        A :class:`Layout` whose ``kind`` is the majority of the decided frames
        and whose focus is the median of the frames that agreed with it. Median
        rather than mean because a single frame where somebody turned right
        round can throw the convergence point across the room.
    """
    decided = [x for x in layouts if x.kind != "unknown"]
    if not decided:
        return Layout("unknown", None, None, 0.0, 0.0, 0)

    lectures = sum(1 for x in decided if x.kind == "lecture")
    kind = "lecture" if lectures * 2 > len(decided) else "group"
    agreeing = [x for x in decided if x.kind == kind and x.focus]
    if not agreeing:
        return Layout(kind, None, None, 0.0, 0.0, len(decided))

    def median(values):
        ordered = sorted(values)
        return ordered[len(ordered) // 2]

    focus = (median([x.focus[0] for x in agreeing]),
             median([x.focus[1] for x in agreeing]))
    centre = (median([x.centre[0] for x in agreeing]),
              median([x.centre[1] for x in agreeing]))
    return Layout(kind, focus, centre,
                  median([x.radius for x in agreeing]),
                  median([x.ratio for x in agreeing]),
                  max(x.n for x in agreeing))


def annotate(graph: dict, record: dict) -> dict:
    """Record each student's orientation relative to the room's focus.

    Args:
        graph: A scene graph whose nodes carry ``person_id``.
        record: The Stage 1 record it was built from, for ``bbox`` and
            ``posture``.

    Returns:
        The same graph. Every node gains ``layout`` (what the room is doing),
        ``oriented`` (``True``/``False``/``None``) and ``focus_offset_deg``.

    The focus is measured on this frame rather than inherited from a scene
    summary, so the same code serves a live session and an offline pass. A
    student whose shoulders could not be read gets ``None``, never ``False``:
    not having seen someone is not evidence that they looked away.
    """
    people = record.get("persons", [])
    layout = detect(people)
    by_person = {p.get("person_id"): p for p in people}

    for node in graph.get("nodes", []):
        feat = node.setdefault("features", {})
        feat["layout"] = layout.kind
        person = by_person.get(node.get("person_id"))
        if person is None:
            feat["oriented"] = None
            feat["focus_offset_deg"] = None
            continue
        ok, off = oriented_toward(person, layout.focus)
        feat["oriented"] = ok
        feat["focus_offset_deg"] = None if off is None else round(off, 1)
    return graph
