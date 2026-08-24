"""Register people into a persistent face gallery, so their ids stay constant.

Reads enrollment photos, embeds the face in each with the same SCRFD + ArcFace
path the pipeline uses, averages them per person, and writes the result to a
gallery :mod:`backend.enrollment` can load. Once someone is registered, every
video gives them the same ``person_id``.

Read the privacy section of ``backend/enrollment.py`` before running this. A
gallery is biometric data about named people, it persists across sessions, and
it is exactly the thing the rest of this project is careful not to keep.

Enrollment quality drives everything downstream, so this is strict on purpose
and tells you what it rejected rather than quietly averaging a bad shot in:

* the face must clear ``--min-score`` (default: the same
  ``min_face_score_for_identity`` gate identity already applies)
* the face must be at least ``--min-size`` pixels on its shorter side, because
  a small face embeds unreliably no matter how confident the detector is
* a photo with more than one face is skipped unless ``--largest-face`` is
  given, since guessing which face to enroll is how the wrong person ends up
  with somebody else's id

Run:
    # one folder per person: enroll/pranay/*.jpg, enroll/asha/*.jpg, ...
    python -m tools.register_faces --images enroll/

    # capture straight from a webcam
    python -m tools.register_faces --webcam --name pranay --shots 12

    # inspect or edit the gallery
    python -m tools.register_faces --list
    python -m tools.register_faces --forget pranay
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _iter_person_dirs(root: Path):
    """Yield ``(name, [image paths])`` for each subdirectory of ``root``.

    Args:
        root: A directory holding one subdirectory per person, named after
            them.

    Yields:
        The person's name and their enrollment images, sorted for
        reproducibility.

    Raises:
        SystemExit: If ``root`` is not a directory.
    """
    if not root.is_dir():
        raise SystemExit(f"--images must be a directory, got: {root}")
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        images = sorted(
            p for p in child.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES
        )
        if images:
            yield child.name, images
        else:
            print(f"  {child.name}: no images found, skipped")


def _embed(analyzer, image, source: str, args) -> list | None:
    """Extract the single enrollable face embedding from one image.

    Args:
        analyzer: A live :class:`backend.face.FaceAnalyzer`.
        image: A BGR frame.
        source: Label used in messages (a filename, or "webcam frame 3").
        args: Parsed CLI arguments, for the quality gates.

    Returns:
        The 512-d embedding, or ``None`` with a printed reason if this image
        cannot be enrolled.
    """
    faces = analyzer.detect_faces(image)
    if not faces:
        print(f"  {source}: no face detected, skipped")
        return None

    if len(faces) > 1 and not args.largest_face:
        print(
            f"  {source}: {len(faces)} faces detected -- skipped. Enrolling the "
            f"wrong face assigns someone else's id. Re-shoot with one person, "
            f"or pass --largest-face."
        )
        return None

    # detect() returns strongest-first; --largest-face instead trusts size,
    # which is the better proxy when a bystander is sharply in focus behind
    # the person being enrolled.
    face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3]) if args.largest_face else faces[0]

    if face.score < args.min_score:
        print(f"  {source}: face score {face.score:.2f} < {args.min_score}, skipped")
        return None
    shorter = min(face.bbox[2], face.bbox[3])
    if shorter < args.min_size:
        print(f"  {source}: face is {shorter}px, under --min-size {args.min_size}, skipped")
        return None
    if face.embedding is None:
        print(
            f"  {source}: no embedding returned -- is "
            f"FaceConfig.enable_recognition on? Skipped."
        )
        return None
    return face.embedding


def _from_images(analyzer, gallery, args) -> int:
    """Register every person found under ``args.images``.

    Args:
        analyzer: A live face analyzer.
        gallery: The gallery to register into.
        args: Parsed CLI arguments.

    Returns:
        How many people were registered.
    """
    import cv2

    registered = 0
    for name, paths in _iter_person_dirs(Path(args.images)):
        print(f"{name}: {len(paths)} image(s)")
        embeddings = []
        for path in paths:
            image = cv2.imread(str(path))
            if image is None:
                print(f"  {path.name}: unreadable, skipped")
                continue
            embedding = _embed(analyzer, image, path.name, args)
            if embedding is not None:
                embeddings.append(embedding)

        if len(embeddings) < args.min_shots:
            print(
                f"  -> NOT registered: {len(embeddings)} usable shot(s), "
                f"--min-shots is {args.min_shots}.\n"
            )
            continue
        person = gallery.register(name, embeddings)
        print(f"  -> registered as person_id {person.person_id} from {person.shots} shot(s)\n")
        registered += 1
    return registered


def _from_webcam(analyzer, gallery, args) -> int:
    """Capture enrollment shots from a camera and register one person.

    Args:
        analyzer: A live face analyzer.
        gallery: The gallery to register into.
        args: Parsed CLI arguments; ``--name`` and ``--shots`` are used here.

    Returns:
        ``1`` if the person was registered, ``0`` otherwise.

    Raises:
        SystemExit: If ``--name`` was not given or the camera cannot be opened.
    """
    import cv2

    if not args.name:
        raise SystemExit("--webcam requires --name.")
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}.")

    print(
        f"Capturing {args.shots} shots for {args.name!r}. Look at the camera and "
        f"move your head slightly between shots -- varied pose makes the mean "
        f"embedding generalise to a classroom view.\n"
    )
    embeddings = []
    attempts = 0
    try:
        while len(embeddings) < args.shots and attempts < args.shots * 10:
            attempts += 1
            ok, frame = capture.read()
            if not ok:
                break
            embedding = _embed(analyzer, frame, f"webcam frame {attempts}", args)
            if embedding is not None:
                embeddings.append(embedding)
                print(f"  captured {len(embeddings)}/{args.shots}")
    finally:
        capture.release()

    if len(embeddings) < args.min_shots:
        print(f"\n-> NOT registered: only {len(embeddings)} usable shot(s).")
        return 0
    person = gallery.register(args.name, embeddings)
    print(f"\n-> registered {args.name!r} as person_id {person.person_id} from {person.shots} shot(s)")
    return 1


def _identify(analyzer, gallery, args) -> int:
    """Report who the gallery thinks is in each given image.

    The honest way to test a registration: enroll from one set of photos, then
    identify a **held-out** photo the gallery has never seen. Checking against
    an enrollment photo only proves the file round-tripped.

    Args:
        analyzer: A live face analyzer.
        gallery: The gallery to query.
        args: Parsed CLI arguments; ``--identify`` names a file or directory.

    Returns:
        A process exit code: non-zero if any image could not be identified, so
        this is usable as a check in a script.
    """
    import cv2

    target = Path(args.identify)
    images = (
        sorted(p for p in target.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if target.is_dir()
        else [target]
    )
    if not images:
        raise SystemExit(f"No images found at {target}")

    print(f"Matching {len(images)} image(s) against {len(gallery)} registered "
          f"people, threshold {args.min_similarity:.2f}:\n")
    unmatched = 0
    for path in images:
        image = cv2.imread(str(path))
        if image is None:
            print(f"  {path.name}: unreadable")
            unmatched += 1
            continue
        embedding = _embed(analyzer, image, path.name, args)
        if embedding is None:
            unmatched += 1
            continue
        hit = gallery.identify(embedding, threshold=args.min_similarity)
        if hit is None:
            # Show the closest miss too: "no match" is far more actionable when
            # you can see whether it missed narrowly or was never close.
            closest = gallery.identify(embedding, threshold=-1.0)
            near = f" (closest: {closest[0].name} at {closest[1]:.3f})" if closest else ""
            print(f"  {path.name}: NOT REGISTERED{near}")
            unmatched += 1
        else:
            print(f"  {path.name}: {hit[0].name}  (person_id {hit[0].person_id}, "
                  f"similarity {hit[1]:.3f})")
    print(f"\n{len(images) - unmatched}/{len(images)} identified.")
    return 1 if unmatched else 0


def _print_gallery(gallery, path: Path) -> None:
    """Print every registered person.

    Args:
        gallery: The gallery to describe.
        path: Where it was loaded from, for the header.
    """
    if not len(gallery):
        print(f"No one registered in {path}.")
        return
    print(f"{len(gallery)} registered in {path}:\n")
    print(f"{'id':>4}  {'shots':>5}  name")
    for person in gallery.people:
        print(f"{person.person_id:>4}  {person.shots:>5}  {person.name}")


def main() -> int:
    """Parse arguments and run the requested registration action.

    Returns:
        A process exit code.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--images",
        help="Directory holding one subdirectory per person, named after them.",
    )
    source.add_argument(
        "--webcam", action="store_true", help="Capture shots from a camera instead."
    )
    source.add_argument("--list", action="store_true", help="Show the gallery and exit.")
    source.add_argument("--forget", metavar="NAME", help="Remove a person and exit.")
    source.add_argument(
        "--identify",
        metavar="PATH",
        help="Test the gallery: report who is in this image (or every image in "
             "this directory). Use a held-out photo, not an enrollment one.",
    )

    parser.add_argument("--gallery", default=None, help="Gallery file to read and write.")
    parser.add_argument("--name", help="Person being enrolled, with --webcam.")
    parser.add_argument("--shots", type=int, default=10, help="Webcam shots to capture.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument(
        "--min-shots",
        type=int,
        default=3,
        help="Refuse to register anyone with fewer usable shots than this. One "
             "shot is a noisy reference; several averaged is measurably better.",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Minimum face-detector confidence. Defaults to the identity gate.",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=80,
        help="Minimum face size in pixels on the shorter side.",
    )
    parser.add_argument(
        "--largest-face",
        action="store_true",
        help="With several faces in a photo, enroll the largest instead of skipping.",
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=None,
        help="Cosine similarity required to call a match, with --identify. "
             "Defaults to IdentityConfig.match_threshold.",
    )
    parser.add_argument("--verbose", action="store_true", help="Show library logging.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    from backend.config import CONFIG
    from backend.enrollment import DEFAULT_GALLERY_PATH, EnrolledGallery

    if args.min_score is None:
        args.min_score = CONFIG.identity.min_face_score_for_identity
    if args.min_similarity is None:
        args.min_similarity = CONFIG.identity.match_threshold

    path = Path(args.gallery) if args.gallery else DEFAULT_GALLERY_PATH
    gallery = EnrolledGallery.load(path)

    if args.list:
        _print_gallery(gallery, path)
        return 0

    if args.forget:
        if gallery.forget(args.forget):
            gallery.save(path)
            print(f"Removed {args.forget!r}. Their id is retired, not reused.")
            return 0
        print(f"No one named {args.forget!r} is registered.")
        return 1

    if not args.images and not args.webcam and not args.identify:
        parser.error("choose one of --images, --webcam, --identify, --list, --forget")

    from backend.face import FaceAnalyzer

    if args.identify:
        if len(gallery) == 0:
            print(f"Nobody is registered in {path}; nothing to match against.")
            return 1
        with FaceAnalyzer() as analyzer:
            return _identify(analyzer, gallery, args)

    with FaceAnalyzer() as analyzer:
        if args.images:
            count = _from_images(analyzer, gallery, args)
        else:
            count = _from_webcam(analyzer, gallery, args)

    if count:
        gallery.save(path)
        print(f"Gallery now holds {len(gallery)} people -> {path}")
    else:
        print("Nobody registered; gallery left unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
