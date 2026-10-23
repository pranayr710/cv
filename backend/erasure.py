"""Erase one person from everything this system holds about them.

A deletion request is one request. Until now it needed three, because the data
lives in three places that knew nothing about each other: the face gallery
(name and a 512-dimension embedding), the session database (per-window
attention scores), and the pipeline's output folders (per-frame boxes, derived
profiles, and the person's name rendered into an HTML report). Deleting via
the dashboard removed the gallery entry and left the rest; deleting via the
database CLI did the reverse. Both reported success.

That is the shape of a privacy failure nobody notices, and this system holds
biometric data about children.

**What erasure means here, precisely.** The person's name and face embedding
are destroyed. Their derived records -- attention scores, action tallies,
per-frame boxes -- are destroyed with them, because a per-frame bounding box
tied to a stable id is still personal data even without a name attached.

**What is deliberately kept.** The *identifier* stays retired rather than
reused, and session-level aggregates (how many students, class mean) are not
recomputed. An id that came back as a different child would corrupt every
historical record that mentions it, and a class mean is about the class.

**What this cannot reach.** Anything copied elsewhere: a report emailed to a
colleague, a backup, a folder outside the ones named. And the uploaded
recording itself, which shows this person but is also every other student's
source and so cannot be filtered without destroying their records too. The
report names all of it rather than pretending otherwise, because a deletion
that silently missed a copy is worse than one that says where it stopped.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

#: Files in an output folder that carry per-student records.
DERIVED_FILES = ("live_graph.jsonl", "live_frames.jsonl", "raw.jsonl")

#: Files rebuilt from the above rather than filtered line by line.
REBUILT_FILES = ("live_profiles.json", "report.html")


@dataclass
class ErasureReport:
    """What was removed, and what could not be reached.

    Attributes:
        name: Who was erased.
        gallery_removed: Whether a face record was destroyed.
        db_sessions: Session records deleted from the database.
        files_rewritten: Paths whose per-student records were filtered out.
        files_removed: Paths deleted entirely, because they could not be
            filtered safely.
        unreachable: Places this could not reach, stated so the operator can
            act on them.
    """

    name: str
    gallery_removed: bool = False
    db_sessions: int = 0
    files_rewritten: list[str] = field(default_factory=list)
    files_removed: list[str] = field(default_factory=list)
    unreachable: list[str] = field(default_factory=list)

    @property
    def anything_found(self) -> bool:
        return bool(self.gallery_removed or self.db_sessions
                    or self.files_rewritten or self.files_removed)

    def render(self) -> str:
        lines = [f"Erasure report for {self.name!r}", ""]
        lines.append(f"  face record removed : {self.gallery_removed}")
        lines.append(f"  database sessions   : {self.db_sessions}")
        lines.append(f"  files filtered      : {len(self.files_rewritten)}")
        for path in self.files_rewritten:
            lines.append(f"      {path}")
        if self.files_removed:
            lines.append(f"  files deleted       : {len(self.files_removed)}")
            for path in self.files_removed:
                lines.append(f"      {path}")
        if self.unreachable:
            lines.append("")
            lines.append("  NOT reached -- handle these by hand:")
            for item in self.unreachable:
                lines.append(f"      {item}")
        if not self.anything_found:
            lines.append("")
            lines.append("  Nothing found under this name.")
        return "\n".join(lines)



def atomic_write(path: Path, text: str) -> None:
    """Replace a file's contents without a window where it is half-written.

    A bare write_text truncates first. A crash or a full disk between the
    truncate and the write leaves a corrupt JSONL or destroys the gallery,
    with no copy to recover from -- and these are the files holding the data
    an erasure or a retention sweep is in the middle of protecting.

    os.replace is atomic on POSIX and on Windows, so a reader sees either the
    old file or the new one.
    """
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")

    # Retried, because on Windows os.replace fails with PermissionError when
    # anything else holds either file open for an instant -- Defender and the
    # search indexer both do, on a file that was just written. The failure is
    # transient and rare, which is the worst combination: it surfaced twice as
    # an unreproducible test failure before it was understood.
    for attempt in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.05 * (attempt + 1))


def _person_ids_for(gallery, name: str) -> set[int]:
    """Every pipeline id the gallery associates with this person."""
    lowered = name.strip().lower()
    return {p.person_id for p in gallery.people
            if p.name.strip().lower() == lowered}


def _filter_jsonl(path: Path, person_ids: set[int]) -> bool:
    """Drop this person's nodes from a JSONL stream, keeping the frames.

    Frames are kept because removing them would change the timeline every
    other student's windows are measured against -- erasing one person would
    silently alter everybody else's scores.
    """
    if not path.exists() or not person_ids:
        return False

    out: list[str] = []
    changed = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            out.append(line)
            continue

        for key in ("nodes", "persons"):
            items = row.get(key)
            if not isinstance(items, list):
                continue
            kept = [it for it in items
                    if it.get("person_id") not in person_ids]
            if len(kept) != len(items):
                row[key] = kept
                changed = True

        edges = row.get("edges")
        if isinstance(edges, list):
            kept_edges = [e for e in edges
                          if e.get("source") not in person_ids
                          and e.get("target") not in person_ids]
            if len(kept_edges) != len(edges):
                row["edges"] = kept_edges
                changed = True

        out.append(json.dumps(row))

    if changed:
        path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return changed


def _filter_profiles(path: Path, person_ids: set[int]) -> bool:
    """Drop this person's profile from a profiles JSON."""
    if not path.exists() or not person_ids:
        return False
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data if isinstance(data, list) else list(data.values())
    kept = [e for e in entries if e.get("person_id") not in person_ids]
    if len(kept) == len(entries):
        return False
    atomic_write(path, json.dumps(kept, indent=2))
    return True


#: Containers an uploaded recording may arrive in. Mirrors VIDEO_SUFFIXES in
#: tools/run/server.py.
RECORDING_SUFFIXES = (".mp4", ".avi", ".mov", ".mkv", ".webm")


def _recordings_in(folder: Path) -> list[str]:
    """Source recordings under this output folder, as warnings to the caller.

    The dashboard stages each analysed upload under ``uploads/``, so the
    video a session was built from sits beside its results. That file is the
    richest biometric record this system holds and nothing in the erasure
    path had ever looked at it.
    """
    uploads = folder / "uploads"
    if not uploads.is_dir():
        return []
    found = sorted(
        str(p) for p in uploads.rglob("*")
        if p.is_file() and p.suffix.lower() in RECORDING_SUFFIXES)
    return [f"the recording this session was built from, which shows this "
            f"person and cannot be filtered without destroying every other "
            f"student's source: {path}" for path in found]


def erase(name: str, *, gallery_path: Path, db_path: Path,
          output_dirs: list[Path], dry_run: bool = False) -> ErasureReport:
    """Remove every record of one person, across all three stores.

    Args:
        name: The person to erase, matched case-insensitively.
        gallery_path: The enrolment gallery.
        db_path: The session database.
        output_dirs: Pipeline output folders to filter.
        dry_run: Report what would be removed and change nothing.

    Returns:
        An :class:`ErasureReport`, including anything that could not be
        reached.
    """
    from backend.enrollment import EnrolledGallery

    report = ErasureReport(name=name)

    gallery = EnrolledGallery.load(gallery_path)
    person_ids = _person_ids_for(gallery, name)
    if person_ids and not dry_run:
        report.gallery_removed = gallery.forget(name)
        gallery.save(gallery_path)
    else:
        report.gallery_removed = bool(person_ids)

    # The database half, which has two ways of missing records.
    #
    # Matching was case-sensitive here while the gallery's is not, so
    # erase("asha") removed the face record, reported success, and left every
    # attention score in place.
    #
    # And forget_person only reaches students with a person_id. Ingest leaves
    # that NULL -- linking is a separate, optional step -- so a student who
    # was never linked survived a deletion request entirely. They are reached
    # here by the gallery id the pipeline recorded as local_id.
    if Path(db_path).exists():
        from backend.store import connect, forget_person

        with connect(db_path) as conn:
            row = conn.execute(
                "SELECT id FROM person WHERE display_name = ? COLLATE NOCASE",
                (name,)).fetchone()
            if row is not None:
                if dry_run:
                    report.db_sessions = int(conn.execute(
                        "SELECT COUNT(*) FROM student WHERE person_id = ?",
                        (int(row["id"]),)).fetchone()[0])
                else:
                    report.db_sessions = forget_person(conn, int(row["id"]))

            if person_ids:
                placeholders = ",".join("?" * len(person_ids))
                unlinked = conn.execute(
                    "SELECT id FROM student WHERE person_id IS NULL "  # noqa: S608
                    f"AND local_id IN ({placeholders})",
                    tuple(person_ids)).fetchall()
                if unlinked and not dry_run:
                    conn.executemany(
                        "DELETE FROM student WHERE id = ?",
                        [(int(r["id"]),) for r in unlinked])
                    conn.commit()
                report.db_sessions += len(unlinked)

    # Output folders. Without a gallery entry there is no id to match on, so
    # say so rather than reporting a clean sweep that did nothing.
    if not person_ids:
        report.unreachable.append(
            "no gallery entry, so pipeline outputs could not be matched to "
            "this person by id")
    else:
        for folder in output_dirs:
            folder = Path(folder)
            if not folder.is_dir():
                continue
            for filename in DERIVED_FILES:
                path = folder / filename
                if dry_run:
                    if path.exists():
                        report.files_rewritten.append(str(path))
                elif _filter_jsonl(path, person_ids):
                    report.files_rewritten.append(str(path))

            profiles = folder / "live_profiles.json"
            if dry_run:
                if profiles.exists():
                    report.files_rewritten.append(str(profiles))
            elif _filter_profiles(profiles, person_ids):
                report.files_rewritten.append(str(profiles))

            # The report is generated HTML with the name rendered into it.
            # Filtering it would mean parsing a document that is rebuilt from
            # the files above anyway, so it is removed and regenerated.
            html = folder / "report.html"
            if html.exists():
                if not dry_run:
                    html.unlink()
                report.files_removed.append(str(html))

            # The uploaded recording this folder was produced from.
            #
            # It is named rather than deleted. A video cannot be filtered the
            # way a JSONL can: this student's frames are the same frames as
            # everyone else's, so removing them means removing the recording,
            # and that destroys the source for every other student in the
            # room. Deleting it is a decision about other people's records
            # and belongs to the operator.
            #
            # Saying nothing was the previous behaviour and it was the worse
            # of the two: a deletion request reported success having scrubbed
            # the derived rows while leaving the child's face, in full, in a
            # video file this module knew the path of.
            report.unreachable.extend(_recordings_in(folder))

    report.unreachable.append(
        "any copy outside these paths -- an emailed report, a backup, or an "
        "output folder not listed")
    return report
