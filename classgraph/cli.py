"""One command for everything ClassGraph does.

Before this, running the system meant knowing which of forty scripts in
``tools/`` to invoke and from which directory. That is fine for the person who
wrote them and a barrier for everyone else, which is the difference between a
project and something that can be installed.

Each subcommand delegates to the module that already did the work, so there is
one implementation of each behaviour and this file adds no logic of its own
beyond argument routing and the environment check.

    classgraph doctor              # can this machine run it?
    classgraph serve               # web dashboard on :8000
    classgraph video <folder>      # process recorded clips
    classgraph live                # webcam session, no browser
    classgraph train               # refit the attention model
    classgraph label               # queue windows to label
"""
from __future__ import annotations

import argparse
import importlib
import os
import shutil
import sys
from pathlib import Path

#: Model files the pipeline needs that are not installed by pip. Checked by
#: ``doctor`` because their absence produces a failure deep inside a library
#: rather than a message anyone can act on.
WEIGHTS = (
    ("yolo11m.pt", "person and object detection (COCO)"),
    ("runs/behaviour/merged4_aug/weights/best.pt", "behaviour classes"),
)

#: Learned artifacts. Their absence is not fatal -- the pipeline falls back to
#: the rule-based verdict -- so they are reported separately.
OPTIONAL = (
    ("outputs/window_engagement_model.json", "learned attention score"),
    ("outputs/mamba_attention_model.pt", "Mamba sequence scorer"),
)


def _ok(text: str) -> None:
    print(f"  [ ok ] {text}")


def _warn(text: str) -> None:
    print(f"  [warn] {text}")


def _fail(text: str) -> None:
    print(f"  [FAIL] {text}")


def doctor(_args: argparse.Namespace) -> int:
    """Check this machine before anything slow or confusing fails midway."""
    print("ClassGraph environment check\n")
    problems = 0

    print("Python and packages")
    _ok(f"Python {sys.version_info.major}.{sys.version_info.minor}")

    for module, why in (("cv2", "video and image IO"),
                        ("numpy", "arrays"),
                        ("ultralytics", "YOLO detection"),
                        ("mediapipe", "pose and face mesh"),
                        ("insightface", "face recognition"),
                        ("sklearn", "the attention model"),
                        ("fastapi", "the web dashboard")):
        try:
            importlib.import_module(module)
            _ok(f"{module} -- {why}")
        except ImportError:
            _fail(f"{module} missing -- {why}")
            problems += 1

    try:
        import torch  # noqa: PLC0415 - optional extra
        where = "GPU" if torch.cuda.is_available() else "CPU only"
        _ok(f"torch {torch.__version__} ({where})")
        if not torch.cuda.is_available():
            _warn("no CUDA: video processing will be several times slower")
    except ImportError:
        _warn("torch missing -- the Mamba scorer is unavailable "
              "(pip install 'classgraph[learned]')")

    print("\nModel weights")
    for path, why in WEIGHTS:
        if Path(path).exists():
            size = Path(path).stat().st_size / 1e6
            _ok(f"{path} ({size:.0f} MB) -- {why}")
        else:
            _fail(f"{path} missing -- {why}")
            problems += 1

    print("\nTrained artifacts (optional)")
    for path, why in OPTIONAL:
        if Path(path).exists():
            _ok(f"{path} -- {why}")
        else:
            _warn(f"{path} missing -- {why}; the rule-based verdict is used")

    print("\nCamera")
    try:
        import cv2  # noqa: PLC0415 - only needed here
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if os.name == "nt" else 0)
        if cap.isOpened() and cap.read()[0]:
            _ok("camera 0 opens and returns frames")
        else:
            _warn("camera 0 did not return a frame -- another program may "
                  "hold it; recorded video still works")
        cap.release()
    except Exception as exc:  # noqa: BLE001 - any camera failure is a warning
        _warn(f"camera check failed: {exc}")

    print("\nDisk")
    free = shutil.disk_usage(".").free / 1e9
    (_ok if free > 5 else _warn)(f"{free:.1f} GB free")

    print()
    if problems:
        print(f"{problems} problem(s) must be fixed before ClassGraph will "
              "run.")
        return 1
    print("Ready.")
    return 0


def _delegate(module: str, argv: list[str]) -> int:
    """Run another module's main() with the given arguments."""
    sys.argv = [module, *argv]
    mod = importlib.import_module(module)
    return int(mod.main() or 0)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="classgraph", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("doctor", help="check this machine can run ClassGraph")
    sub.add_parser("serve", help="web dashboard (register, run, report)")
    sub.add_parser("video", help="process a folder of recorded clips")
    sub.add_parser("live", help="webcam session without the browser")
    sub.add_parser("train", help="refit the attention model on all labels")
    sub.add_parser("label", help="queue the windows worth labelling next")
    sub.add_parser("db", help="query sessions and students across runs")
    sub.add_parser("erase", help="remove one person from every store")

    # A table rather than a chain of ifs, so adding a command is one line
    # and the help text above cannot drift from what actually dispatches.
    routes = {
        "serve": ("tools.server", []),
        "video": ("tools.batch_session", []),
        "live": ("tools.live_session", []),
        "train": ("tools.train_window_model", []),
        "label": ("tools.improve_model", ["--next", "60"]),
        "db": ("tools.ingest_session", ["--list"]),
        "erase": ("tools.erase_person", []),
    }

    args, rest = parser.parse_known_args()
    if args.command == "doctor":
        return doctor(args)
    if args.command in routes:
        module, fallback = routes[args.command]
        # The fallback is the default *action*, so it still applies when the
        # only arguments given are settings like --db. Without this, naming a
        # database silently turned `classgraph db` into a help screen.
        if fallback and not any(a.startswith("--") and a not in ("--db",)
                                for a in rest):
            rest = [*rest, *fallback]
        return _delegate(module, rest)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
