"""Erase one person from every store, as a single operation.

A deletion request is one request. The data lives in three places that know
nothing about each other -- the face gallery, the session database, and the
pipeline's output folders -- and deleting through any one of them left the
other two intact while reporting success.

    python tools/run/erase_person.py "Asha" --dry-run
    python tools/run/erase_person.py "Asha"
    python tools/run/erase_person.py "Asha" --outputs outputs/live outputs/final2

Always run --dry-run first. This is not reversible and there is no undo: the
whole point is that the data stops existing.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.erasure import erase
from backend.store import DEFAULT_DB

DEFAULT_GALLERY = Path("outputs/enrollment/gallery.json")


def _default_output_dirs() -> list[Path]:
    """Every pipeline output folder under outputs/.

    Found rather than configured, because the folder a session wrote to is
    chosen per run and a person's records are wherever the run put them. A
    list that had to be maintained by hand would go stale, and a stale list
    here means data that survives a deletion request.
    """
    root = Path("outputs")
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir()
                  if p.is_dir() and (p / "live_profiles.json").exists())


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help="the person to erase (case-insensitive)")
    ap.add_argument("--gallery", type=Path, default=DEFAULT_GALLERY)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--outputs", type=Path, nargs="*", default=None,
                    help="output folders to filter (default: every folder "
                         "under outputs/ that holds a session)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would be removed and change nothing")
    ap.add_argument("--yes", action="store_true",
                    help="skip the confirmation prompt")
    args = ap.parse_args()

    folders = args.outputs if args.outputs is not None else _default_output_dirs()
    print(f"gallery : {args.gallery}")
    print(f"database: {args.db}")
    print(f"outputs : {len(folders)} folder(s)")
    for folder in folders:
        print(f"    {folder}")
    print()

    if args.dry_run:
        report = erase(args.name, gallery_path=args.gallery, db_path=args.db,
                       output_dirs=list(folders), dry_run=True)
        print(report.render())
        print("\n  dry run -- nothing was changed.")
        return 0

    if not args.yes:
        # Interactive confirmation, because this is irreversible and the
        # name is matched loosely enough that a typo could hit someone else.
        print(f"This permanently erases every record of {args.name!r}.")
        print("There is no undo. Run with --dry-run first if unsure.")
        try:
            answer = input("Type the name again to confirm: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  cancelled.")
            return 1
        if answer.lower() != args.name.strip().lower():
            print("  names did not match -- cancelled.")
            return 1

    report = erase(args.name, gallery_path=args.gallery, db_path=args.db,
                   output_dirs=list(folders))
    print(report.render())
    if report.files_removed:
        print("\n  Reports were deleted rather than filtered. Regenerate any")
        print("  you still need from the session folder.")
    return 0 if report.anything_found else 1


if __name__ == "__main__":
    raise SystemExit(main())
