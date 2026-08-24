"""Registered-face identity: stable person ids that persist across videos.

Everything else in this project assigns identity *within* one video and throws
it away (see :mod:`backend.identity` for that boundary). This module is the
deliberate exception: a **gallery of registered people**, built once from
enrollment photos, saved to disk, and reused to give the same human the same
``person_id`` in every video they appear in.

Why this exists
---------------

Unsupervised clustering has to *discover* how many people are in a video from
whatever face quality it happens to see. Measured on the audited 5.5-minute
clip (7 students + 1 teacher = 8 people), it over-splits: 13 ids at the shipped
``match_threshold`` of 0.35, and **11 at best across a full 0.20-0.60 sweep**,
so no threshold value reaches the project's "<= 10 ids for 8 people" gate.
Registration removes the guessing: the number of people is known up front, and
matching becomes classification against known references instead of clustering.

It also repairs over-split as a side effect, which is the part worth
understanding. Two clusters that are really the same person will both match
that person's registered reference, so both collapse onto one id -- even though
the clustering step could not merge them directly.

What it does not fix
--------------------

A face too small or too blurred to embed reliably stays unusable whether or not
a reference exists to compare it against. Registration converts some of the
over-split into *unmatched* detections instead, which is a different number on
a different gate (``no id %``), not free accuracy. The back-of-room face-size
limit is a detection problem and is unaffected by anything in this module.

Privacy -- read this before enabling it
---------------------------------------

This crosses a line the rest of the codebase is careful to hold.
:mod:`backend.tracking` documents why: persistent facial recognition in schools
is regulated or banned in several jurisdictions (Sweden's first GDPR fine
targeted a school system that persisted identity across sessions; New York
State banned it in schools statewide), and ``docs/PROJECT_PLAN.md`` commits to
session-scoped ids with regression tests enforcing no cross-session leakage.

A gallery file is **biometric data about named people**. Consequences:

* This module is **opt-in**. Nothing constructs it unless a caller passes a
  gallery explicitly, so the default pipeline keeps the session-scoped
  property untouched.
* The default gallery path lives under ``outputs/``, which is gitignored.
  Never commit a gallery.
* Registration requires informed consent from every person enrolled, and
  under the DPDP Act 2023 child-data provisions, from a guardian where the
  person is a minor.
* :meth:`EnrolledGallery.forget` exists so a person can be removed on request.

Usage:
    from backend.enrollment import EnrolledGallery, EnrolledIdentityResolver
    gallery = EnrolledGallery.load("outputs/enrollment/gallery.json")
    resolver = EnrolledIdentityResolver(gallery)   # drop-in for TwoPass
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.config import CONFIG, IdentityConfig
from backend.identity import TwoPassIdentityResolver, _cosine_similarity

logger = logging.getLogger(__name__)

#: Bumped if the on-disk shape changes incompatibly.
GALLERY_FORMAT_VERSION = 1

#: Where a gallery lives unless the caller says otherwise. Under ``outputs/``
#: on purpose: that directory is gitignored, so a biometric file cannot be
#: committed by accident.
DEFAULT_GALLERY_PATH = Path("outputs/enrollment/gallery.json")


def _unit(vec: np.ndarray) -> np.ndarray:
    """L2-normalise, leaving a zero vector unchanged rather than dividing by 0."""
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    return arr / norm if norm > 0.0 else arr


@dataclass(frozen=True)
class EnrolledPerson:
    """One registered human.

    Attributes:
        person_id: The stable id this person gets in every video. Assigned at
            registration and never reused, so it survives re-enrolling.
        name: Human-readable label, used only for reporting and for
            :meth:`EnrolledGallery.forget`.
        embedding: 512-d L2-normalised mean of every enrollment shot. Averaging
            matters: a single classroom-resolution embedding is noisy (measured
            same-person cosine p10 0.51 against a median of 0.80), and the mean
            of several shots suppresses exactly that noise.
        shots: How many enrollment faces the mean was built from. Reported so a
            thin, unreliable enrollment is visible rather than silent.
    """

    person_id: int
    name: str
    embedding: np.ndarray
    shots: int


class EnrolledGallery:
    """A persistent set of registered people, matched by face embedding.

    Unlike :class:`backend.identity.IdentityGallery`, which is built fresh per
    video and discarded, this is loaded from disk and outlives any one video.
    That is the entire point, and also the reason for the privacy notes in this
    module's docstring.
    """

    def __init__(
        self,
        people: Iterable[EnrolledPerson] = (),
        config: IdentityConfig | None = None,
        next_id: int | None = None,
    ) -> None:
        """Create a gallery.

        Args:
            people: Already-registered people, e.g. from :meth:`load`.
            config: Identity settings; supplies ``match_threshold``. Defaults
                to ``CONFIG.identity``.
            next_id: The id the next registration takes. Restored by
                :meth:`load` so a retired id is not handed out again after a
                reload. Never below ``max(existing) + 1``, so a hand-edited or
                truncated gallery file cannot cause a collision.
        """
        self.config: IdentityConfig = config if config is not None else CONFIG.identity
        self._people: list[EnrolledPerson] = list(people)
        floor = max((p.person_id for p in self._people), default=0) + 1
        self._next_id: int = max(floor, next_id) if next_id is not None else floor

    def __len__(self) -> int:
        return len(self._people)

    @property
    def people(self) -> list[EnrolledPerson]:
        """Every registered person, in registration order."""
        return list(self._people)

    def next_person_id(self) -> int:
        """The id the next registration will take.

        A high-water mark, not ``max(current) + 1``: forgetting a person
        retires their id permanently. Reusing it would silently reattribute
        their historical records to whoever came next.
        """
        return self._next_id

    def register(self, name: str, embeddings: Sequence[np.ndarray]) -> EnrolledPerson:
        """Add or replace a person from one or more enrollment face embeddings.

        Re-registering an existing name **keeps that person's id** and replaces
        their reference embedding, so adding better enrollment shots later does
        not renumber anyone or invalidate previously produced output.

        Args:
            name: Human-readable label, matched case-insensitively.
            embeddings: One or more 512-d ArcFace embeddings of this person.

        Returns:
            The stored :class:`EnrolledPerson`.

        Raises:
            ValueError: If ``embeddings`` is empty.
        """
        if len(embeddings) == 0:
            raise ValueError(f"Cannot register {name!r} with no embeddings.")
        mean = _unit(np.mean([_unit(e) for e in embeddings], axis=0))

        existing = self._find(name)
        person = EnrolledPerson(
            person_id=existing.person_id if existing else self.next_person_id(),
            name=name,
            embedding=mean,
            shots=len(embeddings),
        )
        if existing:
            self._people = [
                person if p.person_id == existing.person_id else p for p in self._people
            ]
        else:
            self._people.append(person)
            self._next_id += 1
        return person

    def forget(self, name: str) -> bool:
        """Remove a person entirely, for a deletion request.

        Their id is not reused, so old output stays interpretable as "someone
        who is no longer registered" rather than silently becoming someone else.

        Args:
            name: The person to remove, matched case-insensitively.

        Returns:
            ``True`` if someone was removed.
        """
        target = self._find(name)
        if target is None:
            return False
        self._people = [p for p in self._people if p.person_id != target.person_id]
        return True

    def identify(
        self, embedding: np.ndarray | None, threshold: float | None = None
    ) -> tuple[EnrolledPerson, float] | None:
        """Match one embedding against the gallery.

        Args:
            embedding: A 512-d ArcFace embedding.
            threshold: Minimum cosine similarity to accept. Defaults to
                ``config.match_threshold``.

        Returns:
            The best-matching person and their similarity, or ``None`` if
            nobody cleared the threshold. Returning ``None`` is a real answer,
            not a failure: an unregistered person genuinely is not in here.
        """
        if embedding is None or not self._people:
            return None
        floor = self.config.match_threshold if threshold is None else threshold
        probe = _unit(embedding)
        best: tuple[EnrolledPerson, float] | None = None
        for person in self._people:
            sim = _cosine_similarity(probe, person.embedding)
            if sim >= floor and (best is None or sim > best[1]):
                best = (person, sim)
        return best

    def _find(self, name: str) -> EnrolledPerson | None:
        """The person registered under ``name``, case-insensitively, or None."""
        key = name.strip().casefold()
        return next((p for p in self._people if p.name.casefold() == key), None)

    def save(self, path: str | Path = DEFAULT_GALLERY_PATH) -> Path:
        """Write the gallery to JSON, creating parent directories.

        JSON rather than a binary blob so the file is inspectable -- somebody
        auditing what biometric data this project stores can read it without
        running any of this code.

        Args:
            path: Destination file.

        Returns:
            The path written.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": GALLERY_FORMAT_VERSION,
            # Saved so a retired id stays retired across a reload, not just
            # within one process.
            "next_person_id": self._next_id,
            "people": [
                {
                    "person_id": p.person_id,
                    "name": p.name,
                    "shots": p.shots,
                    "embedding": [round(float(x), 6) for x in p.embedding],
                }
                for p in self._people
            ],
        }
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("Wrote gallery of %d people to %s", len(self._people), target)
        return target

    @classmethod
    def load(
        cls,
        path: str | Path = DEFAULT_GALLERY_PATH,
        config: IdentityConfig | None = None,
    ) -> EnrolledGallery:
        """Read a gallery written by :meth:`save`.

        Args:
            path: The gallery file.
            config: Identity settings to attach.

        Returns:
            The loaded gallery; an empty one if the file does not exist, so a
            first run before anyone is registered is not an error.

        Raises:
            ValueError: If the file was written by an incompatible version.
        """
        source = Path(path)
        if not source.exists():
            logger.info("No gallery at %s; starting empty.", source)
            return cls(config=config)
        payload = json.loads(source.read_text(encoding="utf-8"))
        version = payload.get("format_version")
        if version != GALLERY_FORMAT_VERSION:
            raise ValueError(
                f"{source} has format_version {version!r}, expected "
                f"{GALLERY_FORMAT_VERSION}. Re-register rather than guessing."
            )
        people = [
            EnrolledPerson(
                person_id=int(entry["person_id"]),
                name=str(entry["name"]),
                embedding=_unit(np.asarray(entry["embedding"], dtype=np.float32)),
                shots=int(entry.get("shots", 1)),
            )
            for entry in payload.get("people", [])
        ]
        stored_next = payload.get("next_person_id")
        return cls(
            people,
            config=config,
            next_id=int(stored_next) if stored_next is not None else None,
        )


class EnrolledIdentityResolver(TwoPassIdentityResolver):
    """Two-pass identity that relabels each cluster with a registered id.

    A drop-in replacement for :class:`~backend.identity.TwoPassIdentityResolver`
    -- same ``keys_for`` / ``observe`` / ``finalise`` contract, so
    :func:`backend.integrate.process_video` accepts it with no changes.

    Clustering still runs exactly as before. What changes is the final naming
    step: instead of numbering clusters 1..k in the order they happen to appear,
    each cluster's mean embedding is matched against the gallery, and a match
    takes that person's **stable** id.

    Two consequences, in order of importance:

    * **Ids become comparable across videos.** Person 3 is the same human on
      Tuesday as on Monday. That is the whole reason to register.
    * **Over-split collapses.** Two clusters the similarity step could not
      merge -- because a low-quality mean landed below the threshold, or
      because a cannot-link edge forbade the merge -- both match the same
      registered reference and therefore both receive that person's id.

    The cannot-link constraint is still honoured, and this is the subtle part.
    If two clusters match the same registered person but were ever alive in the
    same frame, they are provably two different humans, so only the
    better-matching cluster keeps the registered id; the other falls through to
    an unregistered id. Co-occurrence remains proof, and matching never
    overrides it.

    Clusters matching nobody get ids above every registered id, so an
    unregistered visitor is still reported rather than dropped -- and is
    visibly not one of the registered people.
    """

    def __init__(
        self, gallery: EnrolledGallery, config: IdentityConfig | None = None
    ) -> None:
        """Create a resolver bound to a gallery.

        Args:
            gallery: The registered people to match against.
            config: Identity settings. Defaults to ``CONFIG.identity``.
        """
        super().__init__(config)
        self.gallery = gallery

    def _cluster_mean(self, members: set[int]) -> np.ndarray:
        """Evidence-weighted mean embedding of one cluster.

        Args:
            members: Track ids belonging to the cluster.

        Returns:
            A 512-d L2-normalised vector. ``self._sums`` already holds a *sum*
            of embeddings per track, so adding them weights each track by how
            many frames it contributed, which is what we want -- a track seen
            80 times should count for more than one seen twice.
        """
        total = np.sum([self._sums[t] for t in members], axis=0)
        return _unit(total)

    def _conflicts_between(self, a: set[int], b: set[int]) -> bool:
        """Whether any member of ``a`` ever co-occurred with any member of ``b``."""
        return any(self._conflicts.get(track_id, set()) & b for track_id in a)

    def finalise(self) -> dict[int, int]:
        """Cluster, then relabel each cluster with its registered person id.

        Returns:
            A ``{track_id: person_id}`` mapping over every observed track.
            Registered people get their gallery id; unregistered clusters get
            ids above the gallery's range; tracks that never produced a
            trustworthy face keep the negative ids
            :class:`~backend.identity.TwoPassIdentityResolver` gives them.
        """
        mapping: dict[int, int] = {}
        faceless_next = -1
        for track_id in self._seen_tracks:
            if self._counts.get(track_id, 0) == 0 or track_id not in self._sums:
                mapping[track_id] = faceless_next
                faceless_next -= 1

        evidenced = [t for t in self._seen_tracks if t not in mapping]
        clusters = self._cluster(evidenced) if evidenced else []

        # Best matches first, so when two clusters compete for one registered
        # person the more confident one wins the id.
        matches = []
        for index, members in enumerate(clusters):
            hit = self.gallery.identify(self._cluster_mean(members))
            if hit is not None:
                matches.append((hit[1], index, hit[0]))
        matches.sort(key=lambda match: -match[0])

        assigned: dict[int, EnrolledPerson] = {}
        claimed: dict[int, list[int]] = {}
        for similarity, index, person in matches:
            prior = claimed.get(person.person_id, [])
            if any(
                self._conflicts_between(clusters[index], clusters[other])
                for other in prior
            ):
                # Provably a different human: they shared a frame with a
                # cluster already holding this id. Proof beats similarity.
                logger.info(
                    "Cluster %d matched %s at %.3f but co-occurs with a cluster "
                    "already holding that id; leaving it unregistered.",
                    index,
                    person.name,
                    similarity,
                )
                continue
            assigned[index] = person
            claimed.setdefault(person.person_id, []).append(index)

        next_unregistered = self.gallery.next_person_id()
        for index, members in enumerate(clusters):
            person = assigned.get(index)
            if person is None:
                person_id = next_unregistered
                next_unregistered += 1
            else:
                person_id = person.person_id
            for track_id in members:
                mapping[track_id] = person_id

        distinct = len({mapping[t] for t in evidenced}) if evidenced else 0
        logger.info(
            "Registered identity: %d clusters -> %d ids (%d registered people "
            "matched, %d clusters collapsed onto an already-claimed id, "
            "%d unregistered).",
            len(clusters),
            distinct,
            len(claimed),
            len(assigned) - len(claimed),
            len(clusters) - len(assigned),
        )
        return mapping
