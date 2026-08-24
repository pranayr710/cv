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
from collections.abc import Container, Sequence

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

    def match_or_register(
        self,
        embedding: np.ndarray,
        forbidden: Container[int] = (),
    ) -> tuple[int, bool]:
        """Match an embedding against the gallery, or register it as new.

        Args:
            embedding: A 512-d L2-normalised ArcFace embedding.
            forbidden: Person ids this embedding may **not** be assigned to,
                however similar it looks. Used to enforce that two people seen
                in the same frame cannot be the same person -- see
                :class:`TwoPassIdentityResolver`. Matching against these is
                skipped entirely rather than penalised: co-occurrence is proof,
                not a soft preference.

        Returns:
            ``(person_id, is_new)``. ``is_new`` is ``True`` when no permitted
            entry cleared :data:`IdentityConfig.match_threshold` and a new
            person id was minted.
        """
        best_id: int | None = None
        best_sim = -1.0
        for person_id, stored in self._embeddings.items():
            if person_id in forbidden:
                continue
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


class TwoPassIdentityResolver:
    """Assigns person ids after seeing the WHOLE video, not on first sighting.

    Fixes a real defect in :class:`IdentityResolver`, which decides a track's
    identity the first frame that track appears. Measured consequence on a real
    5.5-minute video: 6 tracks were permanently stamped with an unverified
    negative id, and **3 of those had a perfectly good face in a later frame**
    that was never consulted, because the decision had already been made.

    It also improves matching quality independent of that bug. A single frame's
    embedding from a small classroom face is noisy -- measured same-person
    similarity on real footage had a p10 of 0.51 against a median of 0.80, so
    roughly the worst 7% of pairs fall below the match threshold and split one
    student into two ids. Averaging every embedding a track produced, then
    matching once on that mean, suppresses exactly that noise.

    Cost: identity is only known once the video ends, so this cannot be used
    for live streaming -- :class:`IdentityResolver` remains for that case. For
    offline video processing (what this project actually does) there is no
    downside beyond buffering, which is negligible: 512 floats per track.

    Usage:
        resolver = TwoPassIdentityResolver()
        for frame in frames:                       # pass 1
            resolver.observe(track_ids, embeddings, scores)
        mapping = resolver.finalise()              # pass 2
        person_id = mapping[track_id]
    """

    def __init__(self, config: IdentityConfig | None = None) -> None:
        """Create an empty resolver.

        Args:
            config: Identity settings. Defaults to ``CONFIG.identity``.
        """
        self.config: IdentityConfig = config if config is not None else CONFIG.identity
        self._sums: dict[int, np.ndarray] = {}
        self._counts: dict[int, int] = {}
        self._seen_tracks: list[int] = []
        #: track_id -> every other track seen alive in the same frame. Two
        #: co-occurring tracks are provably different people, which is a hard
        #: constraint the embeddings alone cannot supply.
        self._conflicts: dict[int, set[int]] = {}
        self._next_surrogate: int = self.config.surrogate_key_base

    def keys_for(
        self,
        track_ids: Sequence[int | None],
        embeddings: Sequence[np.ndarray | None],
        face_scores: Sequence[float | None] | None = None,
    ) -> list[int | None]:
        """Per-person accumulation keys, inventing one where the tracker failed.

        A detection the tracker never picked up (``track_id is None``) still has
        a face, and a face is enough to identify someone. Rather than discarding
        it -- measured at 20.5% of all person detections on real footage -- it
        gets a fresh single-frame **surrogate** key so it takes part in gallery
        matching like any other observation. See
        :data:`IdentityConfig.identify_untracked` for why box-overlap tracking
        fails on this footage in the first place.

        A surrogate is minted only when there is a trustworthy face to match on;
        without one there is nothing to identify the person by, and a surrogate
        would just mint a meaningless negative id per frame.

        Args:
            track_ids: Per-person ByteTrack ids for this frame, ``None`` where
                the tracker did not confirm a track.
            embeddings: Per-person face embeddings, index-aligned.
            face_scores: Optional per-person face confidence, index-aligned.

        Returns:
            A list index-aligned with the inputs: the real ``track_id`` where
            there is one, a fresh surrogate key where there is not but a good
            face exists, and ``None`` where the person cannot be identified at
            all. Feed this to :meth:`observe` in place of ``track_ids``.

        Raises:
            ValueError: If the input sequences have mismatched lengths.
        """
        n = len(track_ids)
        if len(embeddings) != n or (face_scores is not None and len(face_scores) != n):
            raise ValueError(
                "track_ids, embeddings and face_scores must be the same length."
            )
        scores = face_scores if face_scores is not None else [1.0] * n

        keys: list[int | None] = []
        for track_id, embedding, score in zip(track_ids, embeddings, scores):
            if track_id is not None or not self.config.identify_untracked:
                keys.append(track_id)
                continue
            usable = (
                embedding is not None
                and score is not None
                and score >= self.config.min_face_score_for_identity
            )
            if not usable:
                keys.append(None)
                continue
            keys.append(self._next_surrogate)
            self._next_surrogate += 1
        return keys

    def is_surrogate(self, key: int) -> bool:
        """Whether ``key`` was invented by :meth:`keys_for` rather than tracked."""
        return key >= self.config.surrogate_key_base

    def observe(
        self,
        track_ids: Sequence[int | None],
        embeddings: Sequence[np.ndarray | None],
        face_scores: Sequence[float | None] | None = None,
    ) -> None:
        """Accumulate one frame's evidence without assigning anything yet.

        Args:
            track_ids: Per-person ByteTrack ids for this frame.
            embeddings: Per-person face embeddings, index-aligned.
            face_scores: Optional per-person face confidence, index-aligned.
                Embeddings from a face below
                :data:`IdentityConfig.min_face_score_for_identity` are ignored,
                same gate as the streaming resolver.

        Raises:
            ValueError: If the input sequences have mismatched lengths.
        """
        n = len(track_ids)
        if len(embeddings) != n or (face_scores is not None and len(face_scores) != n):
            raise ValueError(
                "track_ids, embeddings and face_scores must be the same length."
            )
        scores = face_scores if face_scores is not None else [1.0] * n

        # Everything alive in this frame is mutually exclusive: one person
        # cannot occupy two boxes at once. Recorded before the face-quality
        # gate below, because the constraint holds whether or not a usable
        # face was read this frame.
        present = [t for t in track_ids if t is not None]
        for track_id in present:
            others = self._conflicts.setdefault(track_id, set())
            others.update(t for t in present if t != track_id)

        for track_id, embedding, score in zip(track_ids, embeddings, scores):
            if track_id is None:
                continue
            if track_id not in self._counts:
                self._seen_tracks.append(track_id)
                self._counts[track_id] = 0
            if (
                embedding is None
                or score is None
                or score < self.config.min_face_score_for_identity
            ):
                continue
            vec = np.asarray(embedding, dtype=np.float32)
            if track_id in self._sums:
                self._sums[track_id] = self._sums[track_id] + vec
            else:
                self._sums[track_id] = vec.copy()
            self._counts[track_id] += 1

    def finalise(self) -> dict[int, int]:
        """Assign a person id to every observed track via constrained clustering.

        Replaces an earlier sequential-greedy version (each track matched
        once, in isolation, against a growing gallery) after a visual audit
        found it merging three different people into one identity. Root
        cause, per docs/LITERATURE_REVIEW.md section 2: sequential matching
        against an EMA-updated gallery entry is a pairwise VERIFICATION rule
        applied to what is actually a CLUSTERING problem, and it fails by
        TRANSITIVITY -- if track A matches track B's (already-drifted)
        stored embedding, and B's embedding had earlier drifted toward C, A
        can end up sharing an id with C despite A and C never being compared
        to each other directly.

        Constrained AGGLOMERATIVE clustering (Wu et al., CVPR 2013,
        "Constrained Clustering and Its Application to Face Clustering in
        Videos") avoids this structurally: at every step, the two most
        similar clusters merge only if BOTH hold -- their similarity clears
        :data:`IdentityConfig.match_threshold`, AND no member of one cluster
        ever co-occurred with a member of the other (the same hard
        cannot-link constraint the previous version enforced, but checked
        globally between whole clusters at every merge, not through one
        growing centroid a later track happens to land near).

        Cost is O(k^3) in the number of evidenced tracks k, fine for a single
        video's handful-to-low-hundreds of tracks; this would not scale to a
        gallery shared across many videos, which is out of scope (see the
        module's session-reset privacy boundary).

        Returns:
            A ``{track_id: person_id}`` mapping covering every observed track.
            Tracks that never produced a trustworthy face get a negative id,
            with the same meaning as in :class:`IdentityResolver`.
        """
        mapping: dict[int, int] = {}
        faceless_next = -1
        for track_id in self._seen_tracks:
            if self._counts.get(track_id, 0) == 0 or track_id not in self._sums:
                mapping[track_id] = faceless_next
                faceless_next -= 1

        evidenced = [t for t in self._seen_tracks if t not in mapping]
        if evidenced:
            for person_id, members in enumerate(
                self._cluster(evidenced), start=1
            ):
                for track_id in members:
                    mapping[track_id] = person_id

        logger.info(
            "Two-pass identity (constrained clustering): %d tracks -> %d "
            "distinct person ids (%d never face-matched).",
            len(mapping),
            len(set(mapping.values())),
            sum(1 for v in mapping.values() if v < 0),
        )
        return mapping

    def _cluster(self, evidenced: list[int]) -> list[set[int]]:
        """Constrained average-linkage clustering over evidenced tracks.

        Args:
            evidenced: Track ids that produced at least one trustworthy face
                observation (have an entry in ``self._sums``/``self._counts``).

        Returns:
            A list of disjoint track-id sets, one per resulting person,
            ordered by descending total evidence (most-observed person first)
            to match the previous version's "best evidence establishes the
            identity" convention.
        """
        embedding: dict[int, np.ndarray] = {}
        weight: dict[int, int] = {}
        members: dict[int, set[int]] = {}
        conflicts: dict[int, set[int]] = {}
        for t in evidenced:
            mean = self._sums[t] / self._counts[t]
            norm = np.linalg.norm(mean)
            embedding[t] = mean / norm if norm > 0 else mean
            weight[t] = self._counts[t]
            members[t] = {t}
            conflicts[t] = set(self._conflicts.get(t, ()))

        clusters = list(evidenced)
        threshold = self.config.match_threshold

        while len(clusters) > 1:
            best_pair: tuple[int, int] | None = None
            best_sim = threshold  # a merge below the floor is never allowed
            for i in range(len(clusters)):
                a = clusters[i]
                for j in range(i + 1, len(clusters)):
                    b = clusters[j]
                    # Cannot-link: provably different people if any member of
                    # one cluster was ever alive in the same frame as any
                    # member of the other. Checked before spending a
                    # similarity computation on a pair that can never merge.
                    if (members[b] & conflicts[a]) or (members[a] & conflicts[b]):
                        continue
                    sim = _cosine_similarity(embedding[a], embedding[b])
                    if sim > best_sim:
                        best_sim = sim
                        best_pair = (a, b)
            if best_pair is None:
                break

            a, b = best_pair
            total = weight[a] + weight[b]
            merged = (weight[a] * embedding[a] + weight[b] * embedding[b]) / total
            norm = np.linalg.norm(merged)
            embedding[a] = merged / norm if norm > 0 else merged
            weight[a] = total
            members[a] |= members[b]
            conflicts[a] |= conflicts[b]
            clusters.remove(b)
            del embedding[b], weight[b], members[b], conflicts[b]

        clusters.sort(key=lambda c: -weight[c])
        return [members[c] for c in clusters]


def appearance_invariance(crops: Sequence[np.ndarray]) -> float | None:
    """Mean pairwise similarity between one identity's own face crops.

    Distinguishes a **printed face** (wall poster, portrait, textbook photo)
    from a real student. A real face blinks, turns and changes expression
    between frames; a printed one is pixel-identical apart from camera motion
    and lighting.

    This exists because a visual identity audit
    (``tools/audit_identity.py``) found the pipeline profiling two wall posters
    as students -- tracked for 27 and 20 sightings, each contributing a
    permanently "attentive, neutral" phantom to every class-level aggregate.
    Nothing asked whether a face ever changed, because to a face detector a
    printed face is a perfectly good face.

    A **positional** test was tried first and rejected on measurement: the
    audited camera pans, so the posters appeared to move as much as some
    students (1.38-1.74 face-widths of centre drift, versus a barely-moving
    real student at 0.39). Position cannot separate them. Appearance can:

    ==============  =====================
    identity        mean self-similarity
    ==============  =====================
    poster              0.906, 0.909
    student         0.311 - 0.817
    ==============  =====================

    Args:
        crops: Face crops for one identity, any sizes. Compared as
            lighting-normalised 48x48 grayscale, so brightness changes across
            a video do not register as a change in the face itself.

    Returns:
        Mean pairwise similarity in roughly ``[-1, 1]`` (1.0 = identical every
        frame), or ``None`` if fewer than three usable crops were supplied --
        too little evidence to judge, in which case the caller must not reject.
    """
    import cv2

    vectors: list[np.ndarray] = []
    for crop in crops:
        if crop is None or getattr(crop, "size", 0) == 0:
            continue
        grey = cv2.cvtColor(cv2.resize(crop, (48, 48)), cv2.COLOR_BGR2GRAY)
        grey = grey.astype(np.float32)
        # Normalising per crop removes global brightness/contrast drift, so the
        # score reflects the face changing rather than the room lighting.
        vectors.append(
            ((grey - grey.mean()) / (grey.std() + 1e-6)).ravel()
        )

    if len(vectors) < 3:
        return None

    sims = [
        float(np.dot(vectors[i], vectors[j]) / len(vectors[i]))
        for i in range(len(vectors))
        for j in range(i + 1, len(vectors))
    ]
    return float(np.mean(sims)) if sims else None


def is_static_face(
    crops: Sequence[np.ndarray], config: IdentityConfig | None = None
) -> bool:
    """Whether these crops look like a printed face rather than a student.

    Args:
        crops: Face crops for one identity across its lifetime.
        config: Identity settings. Defaults to ``CONFIG.identity``.

    Returns:
        ``True`` only when rejection is enabled, there were at least
        :data:`IdentityConfig.static_face_min_sightings` crops, and the
        measured invariance exceeds
        :data:`IdentityConfig.static_face_similarity`. Returns ``False`` on
        insufficient evidence -- never rejects a student for lack of data.
    """
    cfg = config if config is not None else CONFIG.identity
    if not cfg.reject_static_faces:
        return False
    if len(crops) < cfg.static_face_min_sightings:
        return False
    score = appearance_invariance(crops)
    if score is None:
        return False
    return score > cfg.static_face_similarity
