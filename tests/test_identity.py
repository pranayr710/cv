"""Unit tests for :mod:`backend.identity`.

Pure logic on synthetic embeddings -- no model is loaded, so these run
everywhere.

Test coverage:
    1. A track_id gets a stable person_id once resolved, reused on later frames.
    2. A NEW track_id whose face matches an earlier one is re-identified with
       the SAME person_id -- the core "survives occlusion" requirement.
    3. A genuinely different face registers as a genuinely new person_id.
    4. A face below the confidence gate never registers or matches an identity.
    5. A person with no face at all still gets an id, but a negative
       (never-verified) one, and it is never confused with a real gallery id.
    6. None track_ids pass through as None person_ids.
    7. The gallery never persists across two separate resolver instances --
       the project's session-reset identity property, now also true for face
       embeddings, not just for the tracker.
    8. Mismatched input lengths raise, rather than silently misaligning.
"""

from __future__ import annotations

import numpy as np
import pytest

from backend.config import IdentityConfig
from backend.identity import IdentityGallery, IdentityResolver, _cosine_similarity


def _embedding(seed: int, dim: int = 512) -> np.ndarray:
    """A deterministic, L2-normalised pseudo-embedding for a given identity."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _nudged(embedding: np.ndarray, noise: float = 0.02, seed: int = 0) -> np.ndarray:
    """A near-duplicate embedding, simulating the same face on a later frame."""
    rng = np.random.default_rng(seed)
    v = embedding + noise * rng.normal(size=embedding.shape).astype(np.float32)
    return v / np.linalg.norm(v)


PERSON_A = _embedding(1)
PERSON_B = _embedding(2)  # a genuinely different identity


# --------------------------------------------------------------------------- #
# _cosine_similarity
# --------------------------------------------------------------------------- #


def test_cosine_similarity_of_identical_vectors_is_one() -> None:
    assert _cosine_similarity(PERSON_A, PERSON_A) == pytest.approx(1.0)


def test_cosine_similarity_handles_zero_vector_without_raising() -> None:
    zero = np.zeros_like(PERSON_A)
    assert _cosine_similarity(zero, PERSON_A) == 0.0


# --------------------------------------------------------------------------- #
# IdentityGallery
# --------------------------------------------------------------------------- #


def test_first_sighting_registers_a_new_person() -> None:
    gallery = IdentityGallery()
    person_id, is_new = gallery.match_or_register(PERSON_A)
    assert is_new is True
    assert person_id in gallery.known_person_ids()


def test_a_near_duplicate_embedding_matches_the_same_person() -> None:
    """The core capability: a slightly different embedding of the same face
    (different lighting/angle/frame) must resolve to the same person_id."""
    gallery = IdentityGallery()
    first_id, _ = gallery.match_or_register(PERSON_A)
    second_id, is_new = gallery.match_or_register(_nudged(PERSON_A, seed=1))
    assert is_new is False
    assert second_id == first_id


def test_a_different_face_registers_as_a_new_person() -> None:
    gallery = IdentityGallery()
    id_a, _ = gallery.match_or_register(PERSON_A)
    id_b, is_new = gallery.match_or_register(PERSON_B)
    assert is_new is True
    assert id_b != id_a


def test_gallery_never_merges_two_genuinely_different_people() -> None:
    """A wrong merge (two people sharing one id) is the worse failure mode."""
    gallery = IdentityGallery()
    gallery.match_or_register(PERSON_A)
    _id_b, is_new_b = gallery.match_or_register(PERSON_B)
    _id_a2, is_new_a2 = gallery.match_or_register(_nudged(PERSON_A, seed=2))
    assert is_new_b is True
    assert is_new_a2 is False
    assert len(gallery.known_person_ids()) == 2


# --------------------------------------------------------------------------- #
# IdentityResolver -- the actual integration surface
# --------------------------------------------------------------------------- #


def test_stable_track_id_keeps_its_person_id_across_frames() -> None:
    resolver = IdentityResolver()
    frame1 = resolver.resolve([10], [PERSON_A], [0.9])
    frame2 = resolver.resolve([10], [None], [None])  # no face this frame
    assert frame1[0] is not None
    assert frame2[0] == frame1[0]


def test_reappearing_face_after_a_new_track_id_recovers_the_same_person() -> None:
    """The actual requirement this module exists for: a student who is lost by
    motion tracking (gets a brand-new track_id) and then has their face seen
    again must be reported as the SAME person, not a new one."""
    resolver = IdentityResolver()
    frame1 = resolver.resolve([10], [PERSON_A], [0.9])
    # Occlusion: track 10 is dropped by ByteTrack; a new track_id (11) starts
    # when the same student reappears.
    frame2 = resolver.resolve([11], [_nudged(PERSON_A, seed=3)], [0.9])
    assert frame2[0] == frame1[0]


def test_two_different_people_never_share_a_person_id() -> None:
    resolver = IdentityResolver()
    ids = resolver.resolve([1, 2], [PERSON_A, PERSON_B], [0.9, 0.9])
    assert ids[0] != ids[1]


def test_low_confidence_face_does_not_register_or_match_an_identity() -> None:
    """A wrong merge from a distorted low-quality embedding is worse than an
    unrecognised reappearance -- the gate must actually refuse to trust it."""
    resolver = IdentityResolver(IdentityConfig(min_face_score_for_identity=0.5))
    frame1 = resolver.resolve([10], [PERSON_A], [0.9])
    # New track, same face, but reported at low confidence -- must NOT match.
    frame2 = resolver.resolve([11], [PERSON_A], [0.2])
    assert frame2[0] != frame1[0]
    assert frame2[0] is not None  # still gets *an* id, just not a re-identified one


def test_person_with_no_face_still_gets_an_id() -> None:
    resolver = IdentityResolver()
    ids = resolver.resolve([10], [None], [None])
    assert ids[0] is not None


def test_faceless_id_is_negative_and_never_collides_with_a_real_gallery_id() -> None:
    """Distinguishes 'never verified by face' from a real re-identified id, so
    a consumer cannot mistake an unverified guess for a confirmed match."""
    resolver = IdentityResolver()
    faceless_ids = resolver.resolve([10, 11], [None, None], [None, None])
    real_ids = resolver.resolve([12], [PERSON_A], [0.9])
    assert all(pid is not None and pid < 0 for pid in faceless_ids)
    assert all(pid > 0 for pid in real_ids)


def test_none_track_id_passes_through_as_none_person_id() -> None:
    resolver = IdentityResolver()
    ids = resolver.resolve([None, 5], [None, PERSON_A], [None, 0.9])
    assert ids[0] is None
    assert ids[1] is not None


def test_mismatched_lengths_raise_rather_than_misalign() -> None:
    resolver = IdentityResolver()
    with pytest.raises(ValueError):
        resolver.resolve([1, 2], [PERSON_A], [0.9])


# --------------------------------------------------------------------------- #
# Session-reset property: no gallery survives across two resolver instances.
# Same guarantee this project already tests for the tracker itself; extended
# here to cover the new face-embedding gallery specifically.
# --------------------------------------------------------------------------- #


def test_a_new_resolver_does_not_recognise_a_face_from_a_previous_one() -> None:
    """A fresh video (fresh IdentityResolver) must not carry identity over from
    a previous video's gallery -- the whole point of scoping it per video.
    """
    first_video = IdentityResolver()
    ids_in_first_video = first_video.resolve([1], [PERSON_A], [0.9])

    second_video = IdentityResolver()
    ids_in_second_video = second_video.resolve([1], [PERSON_A], [0.9])

    # Both mint id 1 independently (fresh counters), which is exactly the
    # expected behaviour of two unrelated, unconnected galleries -- not
    # evidence they are the same gallery, just confirmation each starts clean.
    assert ids_in_first_video == [1]
    assert ids_in_second_video == [1]
    assert second_video.gallery is not first_video.gallery


def test_known_person_ids_reports_only_ids_actually_assigned() -> None:
    resolver = IdentityResolver()
    resolver.resolve([1, 2], [PERSON_A, PERSON_B], [0.9, 0.9])
    assert resolver.known_person_ids() == sorted(resolver.known_person_ids())
    assert len(resolver.known_person_ids()) == 2


# --------------------------------------------------------------------------- #
# TwoPassIdentityResolver -- fixes the "decide on first sighting" defect
# --------------------------------------------------------------------------- #


def test_two_pass_rescues_a_track_whose_face_appears_later() -> None:
    """The exact defect this class exists for. Measured on real video: 3 tracks
    were permanently stamped unverified despite having a clear face in a later
    frame, because the streaming resolver had already decided."""
    from backend.identity import TwoPassIdentityResolver

    streaming = IdentityResolver()
    two_pass = TwoPassIdentityResolver()

    # Track 10 appears with NO face, then track 10 gets a good face later.
    frames = [
        ([10], [None], [None]),
        ([10], [PERSON_A], [0.9]),
    ]
    for tids, embs, scores in frames:
        streaming.resolve(tids, embs, scores)
        two_pass.observe(tids, embs, scores)

    # Streaming: stuck with the unverified negative id it assigned on frame 1.
    assert all(pid < 0 for pid in streaming.known_person_ids())
    # Two-pass: uses the later face, so the track is properly identified.
    mapping = two_pass.finalise()
    assert mapping[10] > 0


def test_two_pass_merges_two_tracks_of_the_same_person() -> None:
    from backend.identity import TwoPassIdentityResolver

    r = TwoPassIdentityResolver()
    r.observe([1], [PERSON_A], [0.9])
    r.observe([2], [_nudged(PERSON_A, seed=7)], [0.9])
    mapping = r.finalise()
    assert mapping[1] == mapping[2]


def test_two_pass_keeps_different_people_separate() -> None:
    from backend.identity import TwoPassIdentityResolver

    r = TwoPassIdentityResolver()
    r.observe([1], [PERSON_A], [0.9])
    r.observe([2], [PERSON_B], [0.9])
    mapping = r.finalise()
    assert mapping[1] != mapping[2]


def test_two_pass_track_with_no_face_ever_gets_negative_id() -> None:
    from backend.identity import TwoPassIdentityResolver

    r = TwoPassIdentityResolver()
    r.observe([5], [None], [None])
    mapping = r.finalise()
    assert mapping[5] < 0


def test_two_pass_ignores_low_confidence_faces_when_averaging() -> None:
    """A low-confidence embedding must not pollute the averaged one."""
    from backend.identity import TwoPassIdentityResolver

    r = TwoPassIdentityResolver(IdentityConfig(min_face_score_for_identity=0.5))
    r.observe([1], [PERSON_A], [0.2])  # below gate -- ignored
    mapping = r.finalise()
    assert mapping[1] < 0  # no trustworthy face was ever contributed


def test_two_pass_mismatched_lengths_raise() -> None:
    from backend.identity import TwoPassIdentityResolver

    r = TwoPassIdentityResolver()
    with pytest.raises(ValueError):
        r.observe([1, 2], [PERSON_A], [0.9])


def test_two_pass_covers_every_observed_track() -> None:
    from backend.identity import TwoPassIdentityResolver

    r = TwoPassIdentityResolver()
    r.observe([1, 2, None], [PERSON_A, None, PERSON_B], [0.9, None, 0.9])
    mapping = r.finalise()
    assert set(mapping) == {1, 2}  # None track is not an identity


# --------------------------------------------------------------------------- #
# Poster rejection -- found by a visual audit, not by any metric
# --------------------------------------------------------------------------- #


def _fake_crop(seed: int, size: int = 60):
    """A noisy colour crop, standing in for a real face."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8)


def test_identical_crops_are_detected_as_a_printed_face() -> None:
    """A wall poster is pixel-identical every frame. Measured on real footage:
    posters scored 0.906-0.909 invariance vs 0.311-0.817 for students."""
    from backend.identity import appearance_invariance, is_static_face

    poster = _fake_crop(1)
    crops = [poster.copy() for _ in range(10)]
    score = appearance_invariance(crops)
    assert score is not None and score > 0.95
    assert is_static_face(crops) is True


def test_varying_crops_are_not_a_printed_face() -> None:
    from backend.identity import appearance_invariance, is_static_face

    crops = [_fake_crop(i) for i in range(10)]
    score = appearance_invariance(crops)
    assert score is not None and score < 0.5
    assert is_static_face(crops) is False


def test_invariance_is_robust_to_lighting_change() -> None:
    """Global brightness drift across a video must not read as a changing face,
    nor mask a genuinely static one -- crops are normalised per crop."""
    from backend.identity import appearance_invariance

    base = _fake_crop(2)
    brightened = [
        np.clip(base.astype(np.int16) + delta, 0, 255).astype(np.uint8)
        for delta in range(0, 100, 10)
    ]
    score = appearance_invariance(brightened)
    assert score is not None and score > 0.95


def test_too_few_crops_never_rejects() -> None:
    """Insufficient evidence must not reject a student -- a person seen in 4
    frames would trivially look invariant. Measured case: id 12 scored 0.872
    (above threshold) on 4 sightings and was correctly NOT rejected."""
    from backend.identity import appearance_invariance, is_static_face

    poster = _fake_crop(3)
    few = [poster.copy() for _ in range(2)]
    assert appearance_invariance(few) is None
    assert is_static_face(few) is False

    cfg = IdentityConfig(static_face_min_sightings=8)
    seven = [poster.copy() for _ in range(7)]
    assert is_static_face(seven, cfg) is False


def test_rejection_can_be_disabled() -> None:
    from backend.identity import is_static_face

    poster = _fake_crop(4)
    crops = [poster.copy() for _ in range(10)]
    assert is_static_face(crops, IdentityConfig(reject_static_faces=False)) is False


def test_empty_crops_are_handled() -> None:
    from backend.identity import appearance_invariance

    assert appearance_invariance([]) is None
