"""Within-video face-recognition re-identification for ClassGraph.

Fills a gap ByteTrack alone cannot: a student who is briefly fully occluded,
turns away, or leaves and re-enters frame gets a **new** track_id from motion
tracking, because there is nothing to associate across the gap. Measured on the
one real continuous clip available (204s, ~9 concurrent people at most):
ByteTrack alone assigned **28 distinct track_ids** — roughly 3 ids per real
person. Matching a reappearing face against faces already seen in this video
recovers the original identity instead of minting a new one.

Scope and privacy boundary — read this before wiring it in elsewhere
---------------------------------------------------------------------

This is re-identification **within one video only**. An :class:`IdentityGallery`
is built fresh per video (one instance per call to
:func:`backend.integrate.process_video`) and discarded when processing ends.
No embedding is written to the JSONL output (see
:func:`backend.integrate._person_id_only`, which is all that reaches disk) and
none is persisted across videos. This preserves the project's existing
session-reset identity property — verified for the tracker in
``tests/test_tracking.py`` — and extends the same guarantee to this module in
``tests/test_identity.py``. It only makes identity more robust *inside* one
session; it does not add cross-session or cross-day tracking, and it does not
match against a roster of named students (no such roster exists in this
project — every id here is an anonymous session-local number).

What this deliberately does not solve
--------------------------------------

If a person's face is **never visible** during a gap — fully turned away, fully
blocked by another body, or genuinely absent from frame — face-matching cannot
help, and the gap still relies on :mod:`backend.tracking`'s motion-based
continuity alone. That case can still produce a new id for the same person.
This is a real, stated limitation, not a subtle edge case being glossed over.

The match threshold (:data:`~backend.config.IdentityConfig.match_threshold`) is
also an uncalibrated starting point — see that config's docstring — because
this project has no labelled same/different-identity pairs to calibrate
against yet.

Usage:
    from backend.identity import IdentityResolver
    resolver = IdentityResolver()  # one per video
    person_ids = resolver.resolve(track_ids, embeddings, scores)
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from backend.config import CONFIG, IdentityConfig

logger = logging.getLogger(__name__)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two vectors, safe against a zero-norm input.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        A value in ``[-1.0, 1.0]``; ``0.0`` if either vector has zero norm
        (degenerate, should not occur for a real ArcFace embedding, but this
        must not raise on malformed input).
    """
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class IdentityGallery:
    """A session-local (one video) gallery of face embeddings to person ids.

    Not thread-safe; used from the single-threaded per-frame loop in
    :func:`backend.integrate.process_video`.

    Attributes:
        config: The :class:`IdentityConfig` in effect.
    """

    def __init__(self, config: IdentityConfig | None = None) -> None:
        """Create an empty gallery.

        Args:
            config: Identity settings. Defaults to ``CONFIG.identity``.
        """
        self.config: IdentityConfig = config if config is not None else CONFIG.identity
        self._embeddings: dict[int, np.ndarray] = {}
        self._next_id: int = 1

    def match_or_register(self, embedding: np.ndarray) -> tuple[int, bool]:
        """Match an embedding against the gallery, or register it as new.

        Args:
            embedding: A 512-d L2-normalised ArcFace embedding.

        Returns:
            ``(person_id, is_new)``. ``is_new`` is ``True`` when no existing
            entry cleared :data:`IdentityConfig.match_threshold` and a new
            person id was minted.
        """
        best_id: int | None = None
        best_sim = -1.0
        for person_id, stored in self._embeddings.items():
            sim = _cosine_similarity(embedding, stored)
            if sim > best_sim:
                best_id, best_sim = person_id, sim

        if best_id is not None and best_sim >= self.config.match_threshold:
            # Exponential moving average toward the new sighting, so one
            # poor-quality frame cannot overwrite a good representative
            # embedding built from many earlier sightings.
            rate = self.config.embedding_update_rate
            updated = (1 - rate) * self._embeddings[best_id] + rate * embedding
            norm = np.linalg.norm(updated)
            self._embeddings[best_id] = updated / norm if norm > 0 else updated
            return best_id, False

        new_id = self._next_id
        self._next_id += 1
        self._embeddings[new_id] = embedding
        return new_id, True

    def known_person_ids(self) -> list[int]:
        """All person ids registered so far, in ascending order."""
        return sorted(self._embeddings)


class IdentityResolver:
    """Reconciles per-frame ByteTrack ``track_id``s with a stable ``person_id``.

    Cheap in the common case: once a ``track_id`` has been resolved to a
    ``person_id``, later frames reuse that mapping directly without
    re-matching, since ByteTrack already keeps a track_id stable while the
    person remains continuously trackable. Matching against the face gallery
    only happens the moment a track_id is seen for the first time — which is
    exactly when a person could either be genuinely new, or a reappearance
    that motion tracking lost.

    Attributes:
        gallery: The underlying :class:`IdentityGallery`.
    """

    def __init__(self, config: IdentityConfig | None = None) -> None:
        """Create a resolver with a fresh, empty gallery.

        Args:
            config: Identity settings. Defaults to ``CONFIG.identity``.
        """
        self.config: IdentityConfig = config if config is not None else CONFIG.identity
        self.gallery = IdentityGallery(self.config)
        self._track_to_person: dict[int, int] = {}
        self._faceless_next_id: int = -1

    def resolve(
        self,
        track_ids: Sequence[int | None],
        embeddings: Sequence[np.ndarray | None],
        face_scores: Sequence[float | None] | None = None,
    ) -> list[int | None]:
        """Assign a stable ``person_id`` to each person in one frame.

        Args:
            track_ids: Per-person ByteTrack ids for this frame, index-aligned
                with ``embeddings``. ``None`` when the tracker has not (yet)
                confirmed a track this frame — such a person gets ``None``
                back too; identity needs a track to attach to.
            embeddings: Per-person face embeddings for this frame (``None``
                where no face was matched to that person), index-aligned with
                ``track_ids``.
            face_scores: Optional per-person face detection confidence,
                index-aligned with ``embeddings``. A face below
                :data:`IdentityConfig.min_face_score_for_identity` is not
                trusted to register or match an identity, even if present.
                Defaults to treating every embedding as trusted when omitted.

        Returns:
            Per-person ``person_id``, index-aligned with ``track_ids``. ``None``
            wherever the input ``track_id`` was ``None``.

        Raises:
            ValueError: If the input sequences have mismatched lengths.
        """
        n = len(track_ids)
        if len(embeddings) != n or (face_scores is not None and len(face_scores) != n):
            raise ValueError(
                f"track_ids ({n}), embeddings ({len(embeddings)}) and "
                f"face_scores ({len(face_scores) if face_scores is not None else n}) "
                f"must be the same length."
            )
        scores = face_scores if face_scores is not None else [1.0] * n

        results: list[int | None] = []
        for track_id, embedding, score in zip(track_ids, embeddings, scores):
            if track_id is None:
                results.append(None)
                continue

            known = self._track_to_person.get(track_id)
            if known is not None:
                results.append(known)
                continue

            # First time this track_id has been seen. This is exactly the
            # moment a reappearing person (lost by motion tracking, given a
            # fresh track_id) needs to be recognised by face instead.
            trusted_face = (
                embedding is not None
                and score is not None
                and score >= self.config.min_face_score_for_identity
            )
            if trusted_face:
                person_id, is_new = self.gallery.match_or_register(embedding)
                if not is_new:
                    logger.debug(
                        "track_id %d re-identified as existing person_id %d "
                        "(motion tracking had lost them).",
                        track_id,
                        person_id,
                    )
            else:
                # No trustworthy face to match against. Cannot do better than
                # a brand-new id -- stated limitation, see module docstring.
                person_id = self._faceless_next_id
                self._faceless_next_id -= 1
                # Negative ids mark a person minted without ever being
                # face-matched, distinguishing "never verified by face" from
                # a real gallery id (>=1) for anyone inspecting the mapping.

            self._track_to_person[track_id] = person_id
            results.append(person_id)

        return results

    def known_person_ids(self) -> list[int]:
        """Every distinct ``person_id`` assigned so far this video."""
        return sorted(set(self._track_to_person.values()))
