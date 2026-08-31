"""Tests for room-layout detection (backend/scene_layout.py).

The module's whole job is to stop the attention model inverting itself: in a
group discussion, facing away from the front IS engagement, and judging it
against a lecture's rules scores collaboration as distraction.
"""

from __future__ import annotations

import math

from backend.scene_layout import (
    LECTURE_FOCUS_RATIO,
    detect,
    oriented_toward,
    summarise,
)


def _person(x, y, dx, dy, size=80):
    """A student at ``(x, y)`` whose shoulders face ``(dx, dy)``."""
    return {
        "bbox": [x - size / 2, y - size / 2, size, size],
        "posture": {"facing_direction": [dx, dy]},
    }


def _ring(n=6, cx=500.0, cy=500.0, r=200.0, inward=True):
    """Students evenly spaced round a table, facing in or out."""
    people = []
    for i in range(n):
        a = 2 * math.pi * i / n
        x, y = cx + r * math.cos(a), cy + r * math.sin(a)
        dx, dy = (cx - x), (cy - y)
        if not inward:
            dx, dy = -dx, -dy
        norm = math.hypot(dx, dy)
        people.append(_person(x, y, dx / norm, dy / norm))
    return people


def _rows(n=8, y0=600.0, focus=(500.0, 0.0)):
    """Students in a row, all facing one distant point."""
    people = []
    for i in range(n):
        x = 150.0 + i * 100.0
        dx, dy = focus[0] - x, focus[1] - y0
        norm = math.hypot(dx, dy)
        people.append(_person(x, y0, dx / norm, dy / norm))
    return people


class TestLayoutKind:
    def test_a_ring_facing_inward_is_group_work(self):
        """Students round a table converge on a point among themselves."""
        layout = detect(_ring())
        assert layout.kind == "group"
        assert layout.ratio < LECTURE_FOCUS_RATIO

    def test_rows_facing_a_distant_point_is_a_lecture(self):
        layout = detect(_rows())
        assert layout.kind == "lecture"
        assert layout.ratio > LECTURE_FOCUS_RATIO

    def test_the_focus_of_a_ring_is_the_table(self):
        layout = detect(_ring(cx=500.0, cy=500.0))
        assert layout.focus is not None
        assert math.isclose(layout.focus[0], 500.0, abs_tol=25)
        assert math.isclose(layout.focus[1], 500.0, abs_tol=25)

    def test_the_focus_of_rows_is_where_they_look(self):
        layout = detect(_rows(focus=(500.0, 0.0)))
        assert layout.focus is not None
        assert math.isclose(layout.focus[0], 500.0, abs_tol=60)
        assert layout.focus[1] < 200.0

    def test_too_few_students_is_unknown_not_a_guess(self):
        """Two rays always meet somewhere; calling that a layout would be a
        coin flip dressed as a measurement."""
        assert detect(_ring(n=2)).kind == "unknown"

    def test_students_facing_the_same_way_do_not_converge(self):
        """Parallel rays have no intersection. Reporting one would invent a
        focus that is purely numerical noise."""
        people = [_person(100.0 + i * 100, 500.0, 0.0, -1.0) for i in range(5)]
        assert detect(people).kind == "unknown"

    def test_missing_posture_is_skipped_not_assumed(self):
        people = _ring(n=5)
        people.append({"bbox": [10, 10, 50, 50], "posture": None})
        assert detect(people).kind == "group"

    def test_a_ring_facing_outward_is_not_group_work(self):
        """Facing away from each other is the opposite of collaboration, and
        must not be read as it just because the seating is a ring."""
        assert detect(_ring(inward=False)).kind != "group"


class TestOrientation:
    def test_facing_the_focus_counts_as_oriented(self):
        person = _person(300.0, 500.0, 1.0, 0.0)
        ok, off = oriented_toward(person, (900.0, 500.0))
        assert ok is True
        assert off < 5.0

    def test_facing_away_does_not(self):
        person = _person(300.0, 500.0, -1.0, 0.0)
        ok, off = oriented_toward(person, (900.0, 500.0))
        assert ok is False
        assert off > 150.0

    def test_unreadable_posture_is_none_not_false(self):
        """A student whose shoulders were not visible has not been shown to be
        looking away -- that distinction is the project's honesty claim."""
        ok, off = oriented_toward({"bbox": [0, 0, 10, 10], "posture": {}}, (5, 5))
        assert ok is None and off is None

    def test_no_focus_is_none(self):
        ok, off = oriented_toward(_person(1, 1, 1, 0), None)
        assert ok is None and off is None


class TestSummarise:
    def test_majority_decides_the_scene(self):
        layouts = [detect(_ring()) for _ in range(7)] + [detect(_rows()) for _ in range(2)]
        assert summarise(layouts).kind == "group"

    def test_undecided_frames_are_ignored(self):
        layouts = [detect(_ring(n=2)) for _ in range(9)] + [detect(_rows())]
        assert summarise(layouts).kind == "lecture"

    def test_all_unknown_stays_unknown(self):
        assert summarise([detect(_ring(n=2)) for _ in range(5)]).kind == "unknown"

    def test_empty_is_unknown(self):
        assert summarise([]).kind == "unknown"

    def test_a_single_wild_frame_does_not_move_the_focus(self):
        """One student spinning round can throw the convergence point across
        the room, so the scene focus is a median rather than a mean."""
        good = [detect(_ring(cx=500.0, cy=500.0)) for _ in range(9)]
        wild = detect(_ring(cx=5000.0, cy=5000.0))
        focus = summarise([*good, wild]).focus
        assert focus is not None
        assert math.isclose(focus[0], 500.0, abs_tol=40)
