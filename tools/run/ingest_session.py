"""Load a finished session into the database, and query across sessions.

A pipeline run leaves four files in its output folder. Those answer questions
about that one recording. The questions a school actually has span recordings
-- how is this student doing this term, is the class more engaged after the
seating change -- and answering those from files means reparsing 150 MB per
question.

    python tools/run/ingest_session.py outputs/final2 --source "lecture 2026-10-19"
    python tools/run/ingest_session.py --list
    python tools/run/ingest_session.py --session 1
    python tools/run/ingest_session.py --link 1:5="Asha" --link 1:6="Ben"
    python tools/run/ingest_session.py --history "Asha"
    python tools/run/ingest_session.py --forget "Asha"

Re-ingesting the same folder creates a second session row rather than merging
into the first: two runs of the pipeline over the same footage are two
measurements, and silently overwriting one with the other would hide a change
in the model between them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.store import (
    DEFAULT_DB,
    connect,
    forget_person,
    ingest_profiles,
    link_person,
    person_history,
    record_session,
    session_summary,
)


def _session_facts(folder: Path) -> dict:
    """Frame count and duration, read from the graph log if it is present."""
    graph = folder / "live_graph.jsonl"
    if not graph.exists():
        return {}
    frames = 0
    last = 0
    for line in graph.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        frames += 1
        try:
            last = max(last, int(json.loads(line).get("timestamp_ms") or 0))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    fps = round(frames / (last / 1000.0), 2) if last else None
    return {"n_frames": frames, "duration_ms": last, "fps": fps}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, nargs="?",
                    help="a pipeline output folder to ingest")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--source", default=None,
                    help="what this recording was (defaults to the folder)")
    ap.add_argument("--list", action="store_true", help="list sessions")
    ap.add_argument("--session", type=int, help="show one session's students")
    ap.add_argument("--link", action="append", default=[],
                    metavar="SESSION:LOCAL=NAME",
                    help="attach a session's student to a named person")
    ap.add_argument("--history", metavar="NAME",
                    help="one person's attention across every session")
    ap.add_argument("--forget", metavar="NAME",
                    help="delete a person and every observation of them")
    args = ap.parse_args()

    with connect(args.db) as conn:
        if args.folder:
            profiles = args.folder / "live_profiles.json"
            if not profiles.exists():
                print(f"{profiles} not found.")
                return 1
            facts = _session_facts(args.folder)
            source = args.source or str(args.folder)
            # Two runs over the same footage are two measurements and are
            # kept apart deliberately. But an accidental re-ingest
            # double-counts the student in every cross-session average, and
            # the two look identical afterwards, so say which this is now.
            seen = conn.execute(
                "SELECT id, started_at FROM session WHERE source = ?",
                (source,)).fetchall()
            if seen:
                print(f"  NOTE: {len(seen)} session(s) already recorded under "
                      f"{source!r}:")
                for row in seen:
                    print(f"      session {row['id']}, {row['started_at'][:19]}")
                print("  Adding another. If this is a re-ingest rather than")
                print("  a second recording, remove the old one first, or")
                print("  person averages will count this student twice.")
            session_id = record_session(conn, source=source, **facts)
            n = ingest_profiles(conn, session_id, profiles)
            print(f"session {session_id}: {n} students from {args.folder}")
            if facts:
                print(f"  {facts['n_frames']} frames, "
                      f"{facts['duration_ms'] / 1000:.0f}s, "
                      f"{facts['fps']} fps")

        for spec in args.link:
            try:
                where, name = spec.split("=", 1)
                sess, local = (int(v) for v in where.split(":", 1))
            except ValueError:
                print(f"  could not read --link {spec!r}; "
                      "expected SESSION:LOCAL=NAME")
                continue
            row = conn.execute(
                "SELECT id FROM student WHERE session_id = ? AND local_id = ?",
                (sess, local)).fetchone()
            if row is None:
                print(f"  no student {local} in session {sess}")
                continue
            person = link_person(conn, int(row["id"]), name.strip('"\''))
            print(f"  session {sess} student {local} -> person {person} "
                  f"({name})")

        if args.forget:
            row = conn.execute(
                "SELECT id FROM person WHERE display_name = ?",
                (args.forget,)).fetchone()
            if row is None:
                print(f"  no person named {args.forget!r}")
            else:
                n = forget_person(conn, int(row["id"]))
                print(f"  deleted {args.forget!r} and {n} session record(s), "
                      "with their windows and action tallies")

        if args.history:
            row = conn.execute(
                "SELECT id FROM person WHERE display_name = ?",
                (args.history,)).fetchone()
            if row is None:
                print(f"  no person named {args.history!r}")
            else:
                rows = person_history(conn, int(row["id"]))
                print(f"\n  {args.history} across {len(rows)} session(s)")
                print(f"  {'session':>8}{'when':>22}{'attention':>11}"
                      f"{'on task %':>11}")
                for r in rows:
                    score = ("-" if r["attention_score"] is None
                             else f"{r['attention_score']}/10")
                    pct = ("-" if r["on_task_pct"] is None
                           else f"{r['on_task_pct']:.1f}")
                    print(f"  {r['session_id']:>8}{r['started_at'][:19]:>22}"
                          f"{score:>11}{pct:>11}")

        if args.list:
            rows = conn.execute(
                "SELECT s.id, s.started_at, s.source, s.n_frames,"
                "       (SELECT COUNT(*) FROM student st"
                "          WHERE st.session_id = s.id) AS n_students"
                " FROM session s ORDER BY s.started_at").fetchall()
            print(f"\n  {len(rows)} session(s) in {args.db}")
            print(f"  {'id':>4}{'when':>22}{'students':>10}{'frames':>9}"
                  f"  source")
            for r in rows:
                print(f"  {r['id']:>4}{r['started_at'][:19]:>22}"
                      f"{r['n_students']:>10}{r['n_frames'] or 0:>9}"
                      f"  {r['source']}")

        if args.session:
            rows = session_summary(conn, args.session)
            print(f"\n  session {args.session}: {len(rows)} students, "
                  "least attentive first")
            print(f"  {'local id':>9}{'name':>14}{'attention':>11}"
                  f"{'on task %':>11}{'windows':>9}")
            for r in rows:
                score = ("-" if r["attention_score"] is None
                         else f"{r['attention_score']}/10")
                pct = ("-" if r["on_task_pct"] is None
                       else f"{r['on_task_pct']:.1f}")
                print(f"  {r['local_id']:>9}{r['display_name'] or '-':>14}"
                      f"{score:>11}{pct:>11}{r['n_windows']:>9}")

        if not any([args.folder, args.list, args.session, args.link,
                    args.history, args.forget]):
            ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
