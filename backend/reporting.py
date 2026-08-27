"""Reporting layer for ClassGraph (O6) — class-level trends by default,
per-student views only on explicit request.

Design rules this module enforces structurally (from docs/PROJECT_PLAN.md O6
and docs/PRIVACY_ETHICS.md), so a caller has to work to violate them:

1. CLASS-LEVEL BY DEFAULT. :func:`class_trends` aggregates across all
   students; there is no parameter that quietly narrows it to one person.
2. DRILL-DOWN IS EXPLICIT. Per-student output exists only in
   :func:`student_trajectory`, which requires a ``student_id`` argument.
   The CLI (``tools/dashboard.py``) exposes it only behind ``--per-student``.
3. TRAJECTORIES, NOT SNAPSHOTS. Everything is returned as time-bucketed
   series. A single-number "current class engagement: 73%" is exactly the
   misreadable artifact this project refuses to produce.
4. ABSTENTION IS FIRST-CLASS. Every aggregate carries an ``unknown_rate``, and
   buckets where nothing could be measured are kept in the series (with
   ``n_observations == 0``) rather than dropped — gaps are information.

Input contract: reads the JSONL that ``backend.student_profile`` emits — one
record per student per run, each carrying a ``"concentration"`` summary dict
as produced by ``backend.engagement.summarise_engagement`` (keys ``on``,
``off``, ``unknown``, ``behavioral_proxy_pct`` / legacy ``concentration_pct``
— see the naming note there). Fields are read defensively: a record missing a
field contributes to ``unknown_rate``, never to a guessed average.

Rendering (:func:`render_html`) is dependency-free — inline SVG sparklines,
no JavaScript, no CDN calls, because a privacy section that promises no
third-party data flows should not ship a page that phones home for jQuery.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrendBucket:
    """One time bucket of CLASS-level aggregates."""

    start_s: float
    end_s: float
    n_students_observed: int          # students with >= 1 graded verdict
    n_observations: int               # graded verdicts in bucket
    unknown_rate: float               # share of verdicts unresolvable
    median_on_task_pct: float | None  # None when no student had graded frames
    off_task_rate: float              # off / graded, 0..1
    notes: list[str] = field(default_factory=list)


def load_profiles(path: str) -> list[dict[str, Any]]:
    """Load a finished-run profiles JSONL into plain dicts.

    ``utf-8-sig`` so a BOM left by a Windows text editor doesn't turn the
    first record's first key into ``\\ufeffstudent_id``.
    """
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8-sig") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: not valid JSON") from exc
    return records


def _concentration_of(record: Mapping[str, Any]) -> Mapping[str, Any] | None:
    conc = record.get("concentration")
    return conc if isinstance(conc, Mapping) else None


def _timestamp_of(record: Mapping[str, Any]) -> float | None:
    """Best-effort timestamp in seconds: explicit field, else frame/fps."""
    for key in ("t_s", "time_s", "timestamp_s"):
        if isinstance(record.get(key), (int, float)):
            return float(record[key])
    fps = record.get("fps")
    frame = record.get("frame")
    if isinstance(fps, (int, float)) and fps and isinstance(frame, (int, float)):
        return float(frame) / float(fps)
    return None


def class_trends(
    profiles: Sequence[Mapping[str, Any]],
    duration_s: float | None = None,
    bucket_seconds: float = 300.0,
) -> list[TrendBucket]:
    """Aggregate ALL students into class-level time-bucketed trends.

    This is the default view of the system. Buckets with no data are RETAINED
    (empty) so the rendered trajectory shows the gap instead of smoothing
    over it. ``duration_s`` bounds the series when the last observation is
    earlier than the video's end; pass it from the run metadata when known.
    """
    if duration_s is None:
        stamps = [t for t in (_timestamp_of(r) for r in profiles) if t is not None]
        if not stamps:
            raise ValueError(
                "no timestamps in profiles; pass duration_s explicitly"
            )
        duration_s = max(stamps)
    if bucket_seconds <= 0:
        raise ValueError("bucket_seconds must be positive")

    n_buckets = max(1, math.ceil(duration_s / bucket_seconds))
    # per bucket: [graded_on, graded_off, unknown, students_seen(set), per-student pct lists]
    acc: list[dict[str, Any]] = [
        {"on": 0, "off": 0, "unknown": 0, "students": set(), "pcts": {}}
        for _ in range(n_buckets)
    ]
    undated = 0

    for rec in profiles:
        conc = _concentration_of(rec)
        if conc is None:
            continue
        t = _timestamp_of(rec)
        if t is None or t < 0 or t >= duration_s:
            undated += 1
            continue
        b = acc[min(int(t // bucket_seconds), n_buckets - 1)]
        on = int(conc.get("on", 0) or 0)
        off = int(conc.get("off", 0) or 0)
        unk = int(conc.get("unknown", 0) or 0)
        b["on"] += on
        b["off"] += off
        b["unknown"] += unk
        sid = rec.get("student_id", rec.get("track_id"))
        b["students"].add(sid)
        pct = conc.get("behavioral_proxy_pct", conc.get("concentration_pct"))
        if sid is not None and isinstance(pct, (int, float)):
            b["pcts"][sid] = float(pct)

    buckets: list[TrendBucket] = []
    for i, b in enumerate(acc):
        graded = b["on"] + b["off"]
        total = graded + b["unknown"]
        pcts = sorted(b["pcts"].values())
        med = (
            pcts[len(pcts) // 2]
            if len(pcts) % 2
            else (pcts[len(pcts) // 2 - 1] + pcts[len(pcts) // 2]) / 2.0
        ) if pcts else None
        notes = []
        if undated and i == 0:
            notes.append(f"{undated} records lacked usable timestamps")
        buckets.append(
            TrendBucket(
                start_s=i * bucket_seconds,
                end_s=min((i + 1) * bucket_seconds, duration_s),
                n_students_observed=len(b["students"]),
                n_observations=total,
                unknown_rate=(b["unknown"] / total) if total else 1.0,
                median_on_task_pct=med,
                off_task_rate=(b["off"] / graded) if graded else 0.0,
                notes=notes,
            )
        )
    return buckets


def student_trajectory(
    profiles: Sequence[Mapping[str, Any]], student_id: Any
) -> list[dict[str, Any]]:
    """EXPLICIT drill-down: one named student's observations over time.

    Exists as a separate function requiring ``student_id`` so that class-level
    default cannot silently become an individual surveillance view. Callers
    must name the student; the CLI gates it behind ``--per-student``.
    """
    out: list[dict[str, Any]] = []
    for rec in profiles:
        if rec.get("student_id", rec.get("track_id")) != student_id:
            continue
        conc = _concentration_of(rec) or {}
        out.append(
            {
                "t_s": _timestamp_of(rec),
                "on": conc.get("on"),
                "off": conc.get("off"),
                "unknown": conc.get("unknown"),
                "behavioral_proxy_pct": conc.get(
                    "behavioral_proxy_pct", conc.get("concentration_pct")
                ),
                "caveat": conc.get("caveat"),
            }
        )
    out.sort(key=lambda d: (d["t_s"] is None, d["t_s"]))
    return out


# ---------------------------------------------------------------- rendering --


def _sparkline(values: Sequence[float | None], width: int = 240, height: int = 48) -> str:
    """Inline SVG polyline; None values create gaps (abstention visible)."""
    pts: list[str] = []
    n = len(values)
    finite = [v for v in values if v is not None]
    if not finite or n == 0:
        return (
            f'<svg width="{width}" height="{height}" role="img">'
            f'<text x="4" y="{height - 6}" font-size="10" fill="#888">no data</text></svg>'
        )
    lo, hi = min(finite), max(finite)
    span = (hi - lo) or 1.0
    step_x = width / max(n - 1, 1)
    open_run = False
    for i, v in enumerate(values):
        if v is None:
            open_run = False
            continue
        x = round(i * step_x, 1)
        y = round(height - 4 - (v - lo) / span * (height - 10), 1)
        if not open_run:
            pts.append(f'M{x},{y}')
            open_run = True
        else:
            pts.append(f'L{x},{y}')
    path = "".join(pts)
    label = f"{lo:.0f}–{hi:.0f}" if hi != lo else f"{hi:.0f}"
    return (
        f'<svg width="{width}" height="{height}" role="img" '
        f'aria-label="trend {label}"><path d="{path}" fill="none" '
        f'stroke="#2563eb" stroke-width="2"/></svg>'
    )


def render_html(buckets: Sequence[TrendBucket], title: str = "ClassGraph — class report") -> str:
    """Dependency-free HTML report of the class-level trends.

    Deliberately renders ONLY what was passed in: class aggregates. There is
    no API here that turns a bucket back into individuals — drill-down pages
    must be requested explicitly via :func:`student_trajectory`.
    """
    rows: list[str] = []
    for b in buckets:
        mm = f"{int(b.start_s // 60):02d}:{int(b.start_s % 60):02d}–{int(b.end_s // 60):02d}:{int(b.end_s % 60):02d}"
        med = "—" if b.median_on_task_pct is None else f"{b.median_on_task_pct:.0f}%"
        gap = "" if b.n_observations else ' <em>(no data)</em>'
        rows.append(
            f"<tr><td>{mm}</td><td>{b.n_students_observed}</td>"
            f"<td>{b.n_observations}{gap}</td>"
            f"<td>{b.unknown_rate:.0%}</td><td>{med}</td>"
            f"<td>{b.off_task_rate:.0%}</td></tr>"
        )
    caveat = (
        "Behavioral proxy scores below are observed on-task indicators from a "
        "hand-authored precedence rule over machine labels — not validated "
        "against attention, comprehension, or outcomes."
    )
    table_rows = "\n".join(rows) or '<tr><td colspan="6">no data</td></tr>'
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{title}</title>
<style>
 body {{ font-family: system-ui, sans-serif; margin: 2rem; color: #111 }}
 table {{ border-collapse: collapse; margin-top: 1rem }}
 td, th {{ border: 1px solid #ccc; padding: .35rem .6rem; font-size: .9rem }}
 caption {{ text-align: left; font-weight: 600; padding-bottom: .4rem }}
 .caveat {{ background: #fef3c7; padding: .6rem .8rem; border-radius: 6px;
           font-size: .85rem; max-width: 60ch }}
</style></head><body>
<h1>{title}</h1>
<p class="caveat">{caveat}</p>
<table>
<caption>Class-level trends (all students aggregated)</caption>
<tr><th>interval</th><th>students observed</th><th>observations</th>
<th>unresolved rate</th><th>median on-task</th><th>off-task rate</th></tr>
{table_rows}
</table>
<p>Individual students are intentionally absent from this page. Drill-down
requires an explicit request (see <code>tools/dashboard.py --per-student</code>)
and is logged like any other access.</p>
</body></html>"""
