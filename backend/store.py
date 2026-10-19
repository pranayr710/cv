"""Durable storage for sessions, students and attention scores.

Until now a session produced four files: two JSONL streams, a JSON summary and
an HTML report. That is fine for one recording and wrong for a deployment,
because the questions people actually ask span sessions -- how has this student
been over the term, which class is least engaged on Monday mornings, did the
new seating plan change anything. Answering any of them meant parsing a 36 MB
JSONL and a 115 MB one, per question.

**Why SQLite and not Postgres.** The requirement is that this installs
anywhere, including a school laptop with no server and no administrator. SQLite
is in the Python standard library, needs no process, no port and no
credentials, and the whole database is one file to copy or delete. It handles
far more than a classroom produces: a 40-minute session is roughly 600 windows
and 12,000 observations, so a year of daily sessions is a few million rows,
which SQLite reads without noticing. If this ever needs concurrent writers
across machines, the schema moves to Postgres unchanged -- nothing here uses a
SQLite-specific type.

**Why the raw frame stream stays in JSONL.** Per-frame records are 115 MB per
session and are read once, in order, by the tools that rebuild reports. That is
what a file is good at and what a database is wasteful at. The database holds
what gets queried: sessions, students, windows and action tallies.

**Identity across sessions.** The pipeline assigns a person_id per session, so
the same student is a different id next week. ``person`` is the durable human;
``student`` is one person as seen in one session. Linking them is a separate
decision -- face registration, or a teacher confirming a roster -- and the
schema stores the link without asserting how it was made.

**Privacy.** This holds inferences about named people. Everything belonging to
one person is reachable from their row and removable with
:func:`forget_person`, because a system that cannot delete someone should not
be storing them. Scores are model output, not fact: a 3/10 is a reading of
posture and gaze at roughly 0.72 macro F1 against human judgement that itself
agrees with itself 65 per cent of the time.
"""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: Bumped whenever the schema changes in a way that needs migrating.
SCHEMA_VERSION = 1

DEFAULT_DB = Path("outputs/classgraph.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (
    version     INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS session (
    id            INTEGER PRIMARY KEY,
    started_at    TEXT    NOT NULL,
    source        TEXT    NOT NULL,
    n_frames      INTEGER,
    duration_ms   INTEGER,
    fps           REAL,
    scorer        TEXT,
    notes         TEXT,
    created_at    TEXT    NOT NULL
);

-- A real human, persisting across sessions.
CREATE TABLE IF NOT EXISTS person (
    id            INTEGER PRIMARY KEY,
    display_name  TEXT    NOT NULL,
    created_at    TEXT    NOT NULL
);

-- One person as seen in one session. person_id is NULL until something
-- establishes the link, so an unresolved student is representable rather
-- than being silently attached to the wrong human.
CREATE TABLE IF NOT EXISTS student (
    id                INTEGER PRIMARY KEY,
    session_id        INTEGER NOT NULL REFERENCES session(id) ON DELETE CASCADE,
    person_id         INTEGER REFERENCES person(id) ON DELETE SET NULL,
    local_id          INTEGER NOT NULL,
    attention_score   INTEGER,
    attention_windows INTEGER,
    on_task_pct       REAL,
    UNIQUE (session_id, local_id)
);

CREATE TABLE IF NOT EXISTS window (
    id            INTEGER PRIMARY KEY,
    student_id    INTEGER NOT NULL REFERENCES student(id) ON DELETE CASCADE,
    start_ms      INTEGER NOT NULL,
    end_ms        INTEGER NOT NULL,
    score_10      INTEGER,
    label         TEXT,
    probability   REAL,
    source        TEXT,
    coverage      REAL,
    UNIQUE (student_id, start_ms)
);

CREATE TABLE IF NOT EXISTS action_tally (
    student_id    INTEGER NOT NULL REFERENCES student(id) ON DELETE CASCADE,
    action        TEXT    NOT NULL,
    frames        INTEGER NOT NULL,
    PRIMARY KEY (student_id, action)
);

CREATE INDEX IF NOT EXISTS idx_student_session ON student(session_id);
CREATE INDEX IF NOT EXISTS idx_student_person  ON student(person_id);
CREATE INDEX IF NOT EXISTS idx_window_student  ON window(student_id, start_ms);
CREATE INDEX IF NOT EXISTS idx_session_started ON session(started_at);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect(path: Path | str = DEFAULT_DB) -> Iterator[sqlite3.Connection]:
    """Open the database, creating it if needed, with sane defaults.

    Foreign keys are off by default in SQLite, which would let a delete leave
    orphaned windows behind -- the exact failure that makes "forget this
    student" quietly incomplete. WAL lets the dashboard read while a session
    is still writing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(SCHEMA)
        row = conn.execute("SELECT version FROM schema_version").fetchone()
        if row is None:
            conn.execute("INSERT INTO schema_version (version) VALUES (?)",
                         (SCHEMA_VERSION,))
        conn.commit()
        yield conn
    finally:
        conn.close()


def record_session(conn: sqlite3.Connection, *, source: str,
                   started_at: str | None = None, n_frames: int | None = None,
                   duration_ms: int | None = None, fps: float | None = None,
                   scorer: str | None = None, notes: str | None = None) -> int:
    """Create a session row and return its id."""
    cur = conn.execute(
        "INSERT INTO session (started_at, source, n_frames, duration_ms, fps,"
        " scorer, notes, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (started_at or _now(), source, n_frames, duration_ms, fps, scorer,
         notes, _now()))
    conn.commit()
    return int(cur.lastrowid)


def upsert_student(conn: sqlite3.Connection, session_id: int, local_id: int,
                   **fields: Any) -> int:
    """Create or update one student within a session; return the row id."""
    conn.execute(
        "INSERT OR IGNORE INTO student (session_id, local_id) VALUES (?, ?)",
        (session_id, local_id))
    allowed = {"person_id", "attention_score", "attention_windows",
               "on_task_pct"}
    sets = {k: v for k, v in fields.items() if k in allowed}
    if sets:
        assignments = ", ".join(f"{k} = ?" for k in sets)
        conn.execute(
            f"UPDATE student SET {assignments} "  # noqa: S608 - keys are fixed
            "WHERE session_id = ? AND local_id = ?",
            (*sets.values(), session_id, local_id))
    conn.commit()
    row = conn.execute(
        "SELECT id FROM student WHERE session_id = ? AND local_id = ?",
        (session_id, local_id)).fetchone()
    return int(row["id"])


def add_windows(conn: sqlite3.Connection, student_id: int,
                windows: Iterable[Mapping[str, Any]]) -> int:
    """Insert attention windows for one student. Re-running is safe."""
    rows = [(student_id, int(w["start_ms"]), int(w["end_ms"]),
             w.get("score_10"), w.get("label"), w.get("probability"),
             w.get("source"), w.get("coverage"))
            for w in windows]
    if not rows:
        return 0
    conn.executemany(
        "INSERT OR REPLACE INTO window (student_id, start_ms, end_ms,"
        " score_10, label, probability, source, coverage)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
    conn.commit()
    return len(rows)


def add_action_tally(conn: sqlite3.Connection, student_id: int,
                     tally: Mapping[str, int]) -> int:
    """Store how many frames a student spent in each action."""
    rows = [(student_id, action, int(frames))
            for action, frames in tally.items() if action]
    if not rows:
        return 0
    conn.executemany(
        "INSERT OR REPLACE INTO action_tally (student_id, action, frames)"
        " VALUES (?, ?, ?)", rows)
    conn.commit()
    return len(rows)


def link_person(conn: sqlite3.Connection, student_id: int,
                display_name: str) -> int:
    """Attach a session-local student to a durable person, by name.

    Matching on the display name is deliberate and limited: it is what a
    teacher confirming a roster can supply. Anything stronger -- face
    registration, a school id -- should write the person row itself and pass
    the id, rather than inventing a second identity here.
    """
    row = conn.execute("SELECT id FROM person WHERE display_name = ?",
                       (display_name,)).fetchone()
    if row is None:
        cur = conn.execute(
            "INSERT INTO person (display_name, created_at) VALUES (?, ?)",
            (display_name, _now()))
        person_id = int(cur.lastrowid)
    else:
        person_id = int(row["id"])
    conn.execute("UPDATE student SET person_id = ? WHERE id = ?",
                 (person_id, student_id))
    conn.commit()
    return person_id


def forget_person(conn: sqlite3.Connection, person_id: int) -> int:
    """Delete a person and every observation belonging to them.

    Returns the number of student rows removed. Cascades to windows and
    tallies, which is why foreign keys are enabled on every connection: with
    them off this would leave the scores behind and report success.
    """
    students = conn.execute(
        "SELECT id FROM student WHERE person_id = ?", (person_id,)).fetchall()
    conn.execute("DELETE FROM student WHERE person_id = ?", (person_id,))
    conn.execute("DELETE FROM person WHERE id = ?", (person_id,))
    conn.commit()
    return len(students)


def session_summary(conn: sqlite3.Connection, session_id: int) -> list[dict]:
    """Per-student figures for one session, worst attention first."""
    rows = conn.execute(
        "SELECT s.local_id, s.attention_score, s.attention_windows,"
        "       s.on_task_pct, p.display_name,"
        "       (SELECT COUNT(*) FROM window w WHERE w.student_id = s.id)"
        "         AS n_windows"
        " FROM student s LEFT JOIN person p ON p.id = s.person_id"
        " WHERE s.session_id = ?"
        " ORDER BY s.attention_score IS NULL, s.attention_score ASC",
        (session_id,)).fetchall()
    return [dict(r) for r in rows]


def person_history(conn: sqlite3.Connection, person_id: int) -> list[dict]:
    """One person's attention across every session they appear in.

    This is the query the file-based output could not answer at all, and the
    reason for the database.
    """
    rows = conn.execute(
        "SELECT se.id AS session_id, se.started_at, se.source,"
        "       st.attention_score, st.on_task_pct,"
        "       (SELECT AVG(w.score_10) FROM window w"
        "          WHERE w.student_id = st.id) AS mean_window_score"
        " FROM student st JOIN session se ON se.id = st.session_id"
        " WHERE st.person_id = ?"
        " ORDER BY se.started_at", (person_id,)).fetchall()
    return [dict(r) for r in rows]


def ingest_profiles(conn: sqlite3.Connection, session_id: int,
                    profiles_path: Path) -> int:
    """Load a ``live_profiles.json`` into the database.

    Returns how many students were stored. Students the pipeline could not
    identify (negative ids) are skipped: they are not people, and storing them
    would put noise into every per-student average computed afterwards.
    """
    data = json.loads(Path(profiles_path).read_text(encoding="utf-8"))
    students = data if isinstance(data, list) else data.get("students", data)
    if isinstance(students, Mapping):
        students = list(students.values())

    stored = 0
    for entry in students:
        local = entry.get("person_id", entry.get("id"))
        # Negative ids are tracks the resolver could not turn into a person,
        # and a profile carrying rejected_reason was explicitly refused. Both
        # would otherwise land in every class average computed afterwards.
        if local is None or int(local) <= 0:
            continue
        if entry.get("rejected_reason") or entry.get("is_student") is False:
            continue
        # The teacher is in the roster but is not a student, and including
        # them would skew a class mean by one adult who is attending to
        # nobody.
        if entry.get("role") not in (None, "student"):
            continue

        student_id = upsert_student(
            conn, session_id, int(local),
            attention_score=entry.get("attention_score"),
            attention_windows=entry.get("attention_windows"),
            on_task_pct=entry.get("on_task_pct"))

        # Action counts are nested under a classified/unavailable summary in
        # the pipeline's own format; a flat mapping is also accepted so a
        # caller can pass one directly.
        actions = entry.get("action_counts") or entry.get("actions")
        if isinstance(actions, Mapping):
            tally = actions.get("counts") if "counts" in actions else actions
            if isinstance(tally, Mapping):
                add_action_tally(
                    conn, student_id,
                    {k: v for k, v in tally.items() if isinstance(v, int)})
        stored += 1
    return stored
