"""Render the O6 class dashboard from a finished run's profiles JSONL.

Defaults to the CLASS-LEVEL view — that is the product. Per-student pages are
generated only when ``--per-student <id>[,<id>...]`` is passed explicitly;
there is no "include everyone" shortcut, by design (docs/PRIVACY_ETHICS.md,
access-control section: individual views are deliberate acts, not defaults).

Abstention is rendered, not hidden: buckets where nothing could be measured
appear as "(no data)" rows and gaps in sparklines rather than being dropped.

Usage:
    python -m tools.dashboard --profiles outputs/profiles.jsonl \
        --out outputs/dashboard.html --bucket 300

    # explicit drill-down only:
    python -m tools.dashboard --profiles outputs/profiles.jsonl \
        --out outputs/drill.html --per-student 3,7 --bucket 300
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from backend.reporting import class_trends, load_profiles, render_html


def _fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def render_student_page(profiles, student_id) -> str:
    """Small explicit drill-down page; kept separate so its use is greppable."""
    from backend.reporting import student_trajectory  # local import keeps the default path free of it

    traj = student_trajectory(profiles, student_id)
    if not traj:
        return (
            f"<!doctype html><meta charset='utf-8'><title>student {student_id}</title>"
            f"<h1>student {student_id}</h1><p>No observations recorded.</p>"
        )
    rows = []
    for point in traj:
        t = "—" if point["t_s"] is None else _fmt_duration(point["t_s"])
        pct = point["behavioral_proxy_pct"]
        pct_s = "?" if pct is None else f"{pct:.0f}%"
        unk = point.get("unknown")
        unk_s = "?" if unk is None else str(unk)
        rows.append(
            f"<tr><td>{t}</td><td>{point['on']}</td><td>{point['off']}</td>"
            f"<td>{unk_s}</td><td>{pct_s}</td></tr>"
        )
    caveat = traj[0].get("caveat") or (
        "Observed on-task indicator, not a concentration measurement."
    )
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>student {student_id}</title>
<style>table{{border-collapse:collapse}}td,th{{border:1px solid #ccc;padding:.3rem .5rem}}</style>
</head><body>
<h1>Student {student_id} — trajectory</h1>
<p class="caveat">{caveat}</p>
<table><tr><th>time</th><th>on</th><th>off</th><th>unresolved</th><th>on-task %</th></tr>
{''.join(rows)}</table>
</body></html>"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--profiles", required=True, help="profiles JSONL from backend.student_profile")
    parser.add_argument("--out", required=True, help="output HTML path")
    parser.add_argument("--bucket", type=float, default=300.0, help="class-trend bucket seconds")
    parser.add_argument("--duration", type=float, default=None, help="video duration in seconds")
    parser.add_argument("--title", default="ClassGraph — class report")
    parser.add_argument(
        "--per-student",
        default=None,
        help=(
            "EXPLICIT drill-down: comma-separated student ids to also emit "
            "individual pages (<out stem>_student_<id>.html). Omitted = no "
            "individual pages exist at all."
        ),
    )
    args = parser.parse_args(argv)

    profiles = load_profiles(args.profiles)
    duration = args.duration
    if duration is None:
        stamps = [r.get("t_s") or r.get("timestamp_s") for r in profiles]
        stamps = [s for s in stamps if isinstance(s, (int, float))]
        duration = max(stamps) + 1.0 if stamps else None

    buckets = class_trends(profiles, duration_s=duration, bucket_seconds=args.bucket)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_html(buckets, title=args.title), encoding="utf-8")
    empty = sum(1 for b in buckets if b.n_observations == 0)
    print(
        f"wrote {out_path} ({len(buckets)} buckets, {empty} without data — "
        "gaps shown, not dropped)"
    )

    if args.per_student:
        ids = [part.strip() for part in args.per_student.split(",") if part.strip()]
        stem = out_path.with_suffix("")
        for sid in ids:
            page = Path(f"{stem}_student_{sid}.html")
            page.write_text(render_student_page(profiles, sid), encoding="utf-8")
            print(f"wrote {page} (explicit drill-down)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
