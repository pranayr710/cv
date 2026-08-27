"""Stage 2 — person tracking, filling the Stage 1 ``track_id`` field.

Wraps ultralytics' own ``BYTETracker`` (the same tracker ``yolo track`` uses)
rather than reimplementing tracking logic. ``track_id`` is single-class
("person" only) and assigned per **video**: a :class:`PersonTracker` (or a
fresh :meth:`PersonTracker.reset`) must be created once per video, because
ByteTrack carries Kalman-filter state and a lost-track buffer across frames of
one continuous sequence, and its ID counter starts over at construction.

This boundary is more than a numbering detail. ``track_id`` here is assigned
from motion/IoU only -- there is no appearance embedding or face-recognition
model anywhere in this codebase (verified by inspection, not assumed), so
identity can never survive past the ``PersonTracker`` instance it was
computed in. As long as callers follow the contract above (fresh instance, or
an explicit ``.reset()``, per video), this is the code-level line between
"attention analytics" scoped to one session and persistent facial
recognition across sessions -- the latter is what several jurisdictions
regulate or ban outright in schools (e.g. Sweden's first-ever GDPR fine
targeted a school system that persisted identity across sessions; New York
State banned facial recognition in schools statewide). ``process_video`` in
``integrate.py`` always builds a fresh tracker when none is injected, so the
default path is safe by construction; a caller who injects and reuses one
``PersonTracker`` across more than one video is responsible for calling
``.reset()`` between them -- ``test_reusing_one_tracker_without_reset_does_leak_identity``
in ``tests/test_integrate.py`` demonstrates exactly what happens if they
don't, so this is an enforced, tested contract rather than a comment someone
could miss.

This module does not detect people (Person A) or bind faces/pose to them —
it only assigns identity continuity to the ``persons`` list ``detection.py``
already produced, frame by frame.

Confirmed behaviour (by running the real tracker, not assumed from docs):

* A brand-new person is "unconfirmed" on their first sighting and gets
  ``None``, not a fresh id — the id appears from their **second consecutive**
  match onward. The single exception is the very first call to
  :meth:`PersonTracker.update` after construction/:meth:`reset` (ByteTrack's
  own ``frame_id == 1``), where a new track is activated immediately. This
  means every person's *first* appearance in the output (after frame 1) has
  ``track_id: null``, which is expected, not a bug.
* That internal frame counter advances on **every** call, including
  ``update([])`` for a frame with nobody in it. So if a video opens on an
  empty frame, the "instant activation" bonus is spent on nothing, and
  whoever is first detected on a later frame still needs two consecutive
  sightings to get an id.
* A track survives a gap of up to :data:`TrackingConfig.track_buffer` frames
  with no matching detection and resumes with the **same** id. Once the gap
  exceeds the buffer the old track is dropped; the person reappearing goes
  through the same unconfirmed-then-assigned sequence as a new person, and
  gets a **new**, higher id — never the old one back.

Usage:
    from backend.tracking import PersonTracker
    tracker = PersonTracker()
    track_ids = tracker.update(persons)  # index-aligned with `persons`
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np

from backend.config import CONFIG, TrackingConfig

if TYPE_CHECKING:  # pragma: no cover - typing only
    from backend.detection import Person

logger = logging.getLogger(__name__)


class _TrackerInput:
    """Duck-typed detections adapter for ``ultralytics.trackers.BYTETracker``.

    BYTETracker consumes a "Results-like" object exposing ``xywh``/``xyxy``,
    ``conf``, ``cls``, ``__len__`` and boolean-mask ``__getitem__``. This small
    adapter satisfies that interface directly, rather than depending on
    ultralytics' own ``Results``/``Boxes`` classes, which are built around
    torch tensors and an image context this module does not have.

    Attributes:
        xywh: Boxes as ``(center_x, center_y, w, h)``, shape ``(N, 4)``.
        conf: Detection confidence, shape ``(N,)``.
        cls: Class id per detection, shape ``(N,)``. Always ``0`` here (person
            is the only class tracked).
    """

    def __init__(self, xywh: np.ndarray, conf: np.ndarray, cls: np.ndarray) -> None:
        self.xywh = xywh
        self.conf = conf
        self.cls = cls

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, mask: np.ndarray) -> _TrackerInput:
        return _TrackerInput(self.xywh[mask], self.conf[mask], self.cls[mask])

    @property
    def xyxy(self) -> np.ndarray:
        """Boxes as ``(x1, y1, x2, y2)``, derived from :attr:`xywh`."""
        cx, cy, w, h = (self.xywh[:, i] for i in range(4))
        return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)

    @staticmethod
    def empty() -> _TrackerInput:
        """An input with zero detections, for frames with nobody in them."""
        z4 = np.zeros((0, 4), dtype=np.float32)
        z1 = np.zeros((0,), dtype=np.float32)
        return _TrackerInput(z4, z1, z1)


class PersonTracker:
    """ByteTrack wrapper assigning a stable ``track_id`` to each detected person.

    Attributes:
        config: The :class:`TrackingConfig` in effect for this tracker.
    """

    def __init__(self, config: TrackingConfig | None = None) -> None:
        """Build a fresh ByteTrack tracker for one video.

        Args:
            config: Tracking settings. Defaults to ``CONFIG.tracking``.

        Raises:
            ImportError: If ``ultralytics`` is not installed.
        """
        self.config: TrackingConfig = config if config is not None else CONFIG.tracking
        try:
            from ultralytics.trackers import BOTSORT, BYTETracker
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "ultralytics is required for tracking. "
                "Install it via requirements.txt (`pip install ultralytics`)."
            ) from exc

        self._tracker_cls = (
            BOTSORT if self.config.tracker == "botsort" else BYTETracker
        )
        self._tracker = self._new_tracker()
        logger.info(
            "PersonTracker ready (%s): high=%.2f low=%.2f new=%.2f buffer=%d "
            "match=%.2f fuse_score=%s",
            self.config.tracker,
            self.config.track_high_thresh,
            self.config.track_low_thresh,
            self.config.new_track_thresh,
            self.config.track_buffer,
            self.config.match_thresh,
            self.config.fuse_score,
        )

    def _new_tracker(self):
        """Construct a fresh ``BYTETracker`` instance from ``self.config``."""
        args = SimpleNamespace(
            track_high_thresh=self.config.track_high_thresh,
            track_low_thresh=self.config.track_low_thresh,
            new_track_thresh=self.config.new_track_thresh,
            track_buffer=self.config.track_buffer,
            match_thresh=self.config.match_thresh,
            fuse_score=self.config.fuse_score,
        )
        if self.config.tracker == "botsort":
            # BoT-SORT reads these off the same args object; ByteTrack ignores
            # them entirely, so they are only attached when they are wanted.
            args.with_reid = self.config.with_reid
            args.gmc_method = self.config.gmc_method
            args.proximity_thresh = self.config.proximity_thresh
            args.appearance_thresh = self.config.appearance_thresh
            args.model = self.config.reid_model
        return self._tracker_cls(args)

    def reset(self) -> None:
        """Discard all track state and start a new, empty sequence.

        Call this between videos when reusing one :class:`PersonTracker`
        instance; otherwise construct a new instance per video.
        """
        self._tracker = self._new_tracker()

    def update(
        self, persons: Sequence[Person], frame: object | None = None
    ) -> list[int | None]:
        """Assign a ``track_id`` to each person, aligned index-wise with ``persons``.

        Must be called once per **consecutive** processed frame of the same
        video, in order; ByteTrack's motion model assumes fixed frame spacing.

        Args:
            persons: This frame's detected persons, in detection order.
            frame: The BGR image this frame, optional. BoT-SORT's appearance
                re-identification needs the pixels to embed each box; without
                them ``with_reid`` loads an encoder and then silently does
                nothing, which is indistinguishable from ReID not helping.
                ByteTrack ignores it.

        Returns:
            A list the same length as ``persons``. A person ByteTrack has not
            (yet) confirmed as a track this frame gets ``None`` in that slot.
            Confirmation takes one extra consecutive frame for anyone other
            than a frame-1 detection (see the module docstring), so ``None``
            on a first sighting is expected, not a dropped detection — the
            person may get a real id starting next frame.
        """
        n = len(persons)
        if n == 0:
            self._tracker.update(_TrackerInput.empty(), frame)
            return []

        boxes = np.empty((n, 4), dtype=np.float32)
        conf = np.empty((n,), dtype=np.float32)
        for i, person in enumerate(persons):
            x, y, w, h = person.bbox
            boxes[i] = (x + w / 2.0, y + h / 2.0, w, h)
            conf[i] = person.confidence
        cls = np.zeros((n,), dtype=np.float32)

        tracked = self._tracker.update(_TrackerInput(boxes, conf, cls), frame)

        track_ids: list[int | None] = [None] * n
        for row in tracked:
            # Columns are [x1, y1, x2, y2, track_id, score, cls, idx]; idx is
            # the position in the *full* per-frame detections array we passed
            # in, regardless of which association stage matched it.
            idx = int(row[7])
            track_ids[idx] = int(row[4])
        return track_ids
