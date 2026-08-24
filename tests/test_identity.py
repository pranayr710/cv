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
from backend.identity import (
    IdentityGallery,
    IdentityResolver,
    TwoPassIdentityResolver,
    _cosine_similarity,
)


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


CONFIG_SURROGATE_BASE = IdentityConfig().surrogate_key_base

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


# --------------------------------------------------------------------------- #
# Mutual exclusion: one person cannot be in two places at once.
#
# Measured on the real 5.5-min video BEFORE this constraint existed: 56 of 331
# frames contained a duplicated person_id (id 1 in 28 frames, id 8 in 21, id 5
# in 7). Ground truth for that clip is 8 people, so a duplicate is never a
# legitimate reading -- it is always an error.
# --------------------------------------------------------------------------- #


def test_forbidden_ids_are_never_matched_however_similar() -> None:
    """The constraint is hard, not a penalty: co-occurrence is proof."""
    gallery = IdentityGallery(IdentityConfig(match_threshold=0.35))
    vec = PERSON_A
    first, is_new = gallery.match_or_register(vec)
    assert is_new

    # The identical embedding, but that id is forbidden -> must mint a new one.
    second, is_new = gallery.match_or_register(vec, forbidden={first})
    assert is_new is True
    assert second != first


def test_no_forbidden_ids_behaves_exactly_as_before() -> None:
    """Default argument must not change existing matching behaviour."""
    gallery = IdentityGallery(IdentityConfig(match_threshold=0.35))
    vec = PERSON_A
    first, _ = gallery.match_or_register(vec)
    again, is_new = gallery.match_or_register(vec)
    assert again == first
    assert is_new is False


def test_two_lookalikes_in_one_frame_get_different_ids() -> None:
    """The exact real bug: two different students, similar low-res embeddings,
    both alive in the same frame. Before the constraint both were matched to
    one gallery entry and reported as the same student."""
    base = PERSON_A
    a, b = _nudged(base, 0.05, 1), _nudged(base, 0.05, 2)

    resolver = TwoPassIdentityResolver(IdentityConfig(match_threshold=0.35))
    for _ in range(10):
        resolver.observe([1, 2], [a, b], [0.9, 0.9])
    mapping = resolver.finalise()

    assert mapping[1] != mapping[2], (
        "two tracks alive in the same frame were given the same person id"
    )


def test_tracks_that_never_co_occur_may_still_merge() -> None:
    """The constraint must not block legitimate re-identification -- a student
    who leaves frame and returns as a new track is exactly the case the whole
    identity module exists to handle."""
    base = PERSON_A
    same_person = _nudged(base, 0.02, 3)

    resolver = TwoPassIdentityResolver(IdentityConfig(match_threshold=0.35))
    for _ in range(5):                      # track 1 present, alone
        resolver.observe([1], [same_person], [0.9])
    for _ in range(5):                      # track 1 gone, track 2 appears
        resolver.observe([2], [same_person], [0.9])
    mapping = resolver.finalise()

    assert mapping[1] == mapping[2]


def test_no_duplicate_person_id_among_co_occurring_tracks() -> None:
    """Whole-frame invariant, on more tracks than the gallery would naturally
    separate: every track alive together must end up distinct."""
    base = PERSON_A
    tracks = list(range(1, 7))
    embs = [_nudged(base, 0.03, s) for s in tracks]

    resolver = TwoPassIdentityResolver(IdentityConfig(match_threshold=0.35))
    for _ in range(8):
        resolver.observe(tracks, embs, [0.9] * len(tracks))
    mapping = resolver.finalise()

    assigned = [mapping[t] for t in tracks]
    assert len(set(assigned)) == len(tracks), f"duplicates in {assigned}"


def test_conflicts_recorded_even_when_face_is_unusable() -> None:
    """A track with no readable face still occupies space -- it must still
    constrain, otherwise a student facing away stops being mutually exclusive."""
    base = PERSON_A
    good = _nudged(base, 0.02, 4)

    resolver = TwoPassIdentityResolver(IdentityConfig(match_threshold=0.35))
    resolver.observe([1, 2], [good, None], [0.9, None])       # 2 has no face
    for _ in range(4):
        resolver.observe([1], [good], [0.9])
    for _ in range(4):
        resolver.observe([2], [good], [0.9])                  # now it does
    mapping = resolver.finalise()

    assert mapping[1] != mapping[2], (
        "tracks that co-occurred while one had no usable face were still merged"
    )


# --------------------------------------------------------------------------- #
# Identifying people the tracker never picked up.
#
# Measured on the real video: 164 of 801 person detections (20.5%) had no
# track_id, and identity was keyed on track_id alone -- so those people were
# detected and then silently dropped from the roster. Cause was not detection
# confidence (mean 0.527 vs a 0.25 threshold) but that ByteTrack needs two
# consecutive IoU matches, while this pipeline samples 1 fps from a panning
# camera: 12.6% of boxes fall below its association floor of IoU 0.20.
# --------------------------------------------------------------------------- #


def test_untracked_person_with_a_good_face_gets_a_key() -> None:
    resolver = TwoPassIdentityResolver()
    keys = resolver.keys_for([None], [PERSON_A], [0.9])
    assert keys[0] is not None
    assert resolver.is_surrogate(keys[0])


def test_tracked_person_keeps_their_real_track_id() -> None:
    resolver = TwoPassIdentityResolver()
    keys = resolver.keys_for([7], [PERSON_A], [0.9])
    assert keys == [7]
    assert not resolver.is_surrogate(7)


def test_untracked_person_with_no_face_gets_no_key() -> None:
    """Without a face there is nothing to identify them by, so a surrogate
    would only mint a meaningless id per frame."""
    resolver = TwoPassIdentityResolver()
    assert resolver.keys_for([None], [None], [None]) == [None]


def test_untracked_person_with_a_weak_face_gets_no_key() -> None:
    cfg = IdentityConfig(min_face_score_for_identity=0.50)
    resolver = TwoPassIdentityResolver(cfg)
    assert resolver.keys_for([None], [PERSON_A], [0.20]) == [None]


def test_surrogate_keys_are_unique_per_detection() -> None:
    resolver = TwoPassIdentityResolver()
    first = resolver.keys_for([None, None], [PERSON_A, PERSON_B], [0.9, 0.9])
    second = resolver.keys_for([None], [PERSON_A], [0.9])
    assert len(set(first + second)) == 3


def test_surrogate_keys_cannot_collide_with_tracker_ids() -> None:
    """ByteTrack ids are small positive ints; surrogates must stay clear of
    them or an invented key would silently merge into a real track."""
    resolver = TwoPassIdentityResolver()
    keys = resolver.keys_for([None], [PERSON_A], [0.9])
    assert keys[0] >= CONFIG_SURROGATE_BASE


def test_untracked_sightings_of_one_person_resolve_to_one_id() -> None:
    """The point of the whole change: someone the tracker never held onto is
    recovered by appearance across their separate sightings."""
    resolver = TwoPassIdentityResolver(IdentityConfig(match_threshold=0.35))
    for seed in range(6):
        emb = _nudged(PERSON_A, 0.02, seed)
        keys = resolver.keys_for([None], [emb], [0.9])
        resolver.observe(keys, [emb], [0.9])
    mapping = resolver.finalise()
    assert len(set(mapping.values())) == 1
    assert all(v > 0 for v in mapping.values()), "should be face-verified, not negative"


def test_untracked_people_in_one_frame_stay_separate() -> None:
    """Surrogates must not defeat mutual exclusion: two untracked people
    visible together are still two people."""
    resolver = TwoPassIdentityResolver(IdentityConfig(match_threshold=0.35))
    a, b = _nudged(PERSON_A, 0.03, 11), _nudged(PERSON_A, 0.03, 12)
    for _ in range(4):
        keys = resolver.keys_for([None, None], [a, b], [0.9, 0.9])
        resolver.observe(keys, [a, b], [0.9, 0.9])
    mapping = resolver.finalise()
    # Every frame contributed one pair that co-occurred, so no pair may share.
    assert len(set(mapping.values())) > 1


def test_the_feature_can_be_disabled() -> None:
    cfg = IdentityConfig(identify_untracked=False)
    resolver = TwoPassIdentityResolver(cfg)
    assert resolver.keys_for([None], [PERSON_A], [0.9]) == [None]


def test_keys_for_rejects_mismatched_lengths() -> None:
    resolver = TwoPassIdentityResolver()
    with pytest.raises(ValueError):
        resolver.keys_for([None, None], [PERSON_A], [0.9])


# --------------------------------------------------------------------------- #
# Constrained clustering replaces sequential greedy matching.
#
# The visual audit (docs/IDENTITY_AUDIT.md) found id 2 merging three different
# people. Root cause: sequential matching against an EMA-updated gallery entry
# is vulnerable to TRANSITIVITY -- A matches B's drifted embedding, B had
# earlier drifted toward C, so A ends up sharing an id with C despite A and C
# never being compared directly. Constrained clustering (Wu et al., CVPR 2013)
# fixes this structurally by comparing whole clusters at every merge, not one
# growing centroid. See docs/LITERATURE_REVIEW.md section 2.
# --------------------------------------------------------------------------- #


def test_transitivity_does_not_chain_three_different_people() -> None:
    """The exact real failure mode: A~B is close, B~C is close, but A~C is
    NOT close enough on its own. Sequential greedy matching against a single
    drifting gallery entry could still chain all three into one id; clustering
    must not, because a direct A-vs-C comparison is what governs the final
    grouping, not a chain of intermediate steps."""
    # A 2-D-like construction embedded in 512 dims: three points spaced so
    # consecutive pairs are similar but the endpoints are not, all normalised.
    base = np.zeros(512, dtype=np.float32)
    a = base.copy(); a[0] = 1.0
    b = base.copy(); b[0] = 0.6; b[1] = 0.8
    c = base.copy(); c[1] = 1.0
    a, b, c = (v / np.linalg.norm(v) for v in (a, b, c))

    assert _cosine_similarity(a, b) > 0.5
    assert _cosine_similarity(b, c) > 0.5
    assert _cosine_similarity(a, c) < 0.35  # below match_threshold: NOT the same person

    resolver = TwoPassIdentityResolver(IdentityConfig(match_threshold=0.35))
    for _ in range(5):
        resolver.observe([1], [a], [0.9])
    for _ in range(5):
        resolver.observe([2], [b], [0.9])
    for _ in range(5):
        resolver.observe([3], [c], [0.9])
    mapping = resolver.finalise()

    assert mapping[1] != mapping[3], (
        "A and C were merged via a B-shaped transitivity chain despite "
        "never being similar enough to match directly"
    )


def test_clustering_still_merges_a_genuine_reappearance() -> None:
    """Sanity check that fixing transitivity did not also break the module's
    core purpose: a real re-identification across a track gap must still
    merge, with no intermediate lookalike involved."""
    resolver = TwoPassIdentityResolver(IdentityConfig(match_threshold=0.35))
    for _ in range(5):
        resolver.observe([10], [PERSON_A], [0.9])
    for _ in range(5):
        resolver.observe([11], [_nudged(PERSON_A, 0.02, 7)], [0.9])
    mapping = resolver.finalise()
    assert mapping[10] == mapping[11]


def test_best_evidenced_cluster_gets_person_id_one() -> None:
    """Preserves the previous convention: the most-observed identity should
    be the one a reviewer sees numbered first."""
    resolver = TwoPassIdentityResolver(IdentityConfig(match_threshold=0.35))
    for _ in range(20):
        resolver.observe([1], [PERSON_A], [0.9])
    for _ in range(3):
        resolver.observe([2], [PERSON_B], [0.9])
    mapping = resolver.finalise()
    assert mapping[1] == 1
