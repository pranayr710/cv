"""Render a session report a person can read and explain.

The previous version put everything it knew on the page at once: seventeen
action colours, a transition graph per student, a combined scene graph, and
about 12,900 SVG rectangles across sixteen students. Every individual chart was
defensible and the page as a whole was unreadable -- there is no order in which
to explain thirty-four colours.

This version answers three questions in order, one visual each.

1. **Who needs attention?** One sorted chart of every student. The eye lands on
   the bottom of the list without being told where to look.
2. **When did the class drift?** Every student's timeline on ONE shared axis, so
   a whole-class dip is visible as a vertical band and an individual's lapse as
   a gap in one row. Separate per-student axes made that comparison impossible.
3. **What was this student doing?** A compact card each, on demand.

Seventeen actions collapse to four buckets for anything visual. The detail is
still reported, as text, where text is better than colour: nobody can hold
seventeen hues in their head, but "on phone 704" is exact and instantly read.

Colour follows the data-viz method rather than taste. The four bucket hues were
run through the validator's six checks in both light and dark mode -- lightness
band, chroma floor, colourblind separation, normal-vision separation, and
contrast against the surface. Adjacent-pair CVD separation is dE 23.1 light and
17.3 dark against a target of 8. Every bucket also carries a visible text label,
which is what the one contrast warning (aqua on the light surface, 2.74 against
a 3.0 minimum) obliges.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

#: Seventeen actions in four buckets. The buckets are what gets drawn; the
#: actions are what gets named in text.
BUCKETS = {
    "participating": ("attentive", "raising_hand", "leaning_forward"),
    "working": ("writing", "reading", "studying", "typing", "on_laptop"),
    "passive": ("head_down", "head_on_hand", "slouching", "drinking", "eating"),
    "off_task": ("on_phone", "eyes_closed", "looking_away", "yawning"),
}
BUCKET_OF = {a: b for b, actions in BUCKETS.items() for a in actions}
BUCKET_ORDER = ("participating", "working", "passive", "off_task", "unknown")
BUCKET_LABEL = {
    "participating": "participating",
    "working": "working",
    "passive": "passive",
    "off_task": "off task",
    "unknown": "not seen",
}

ACTION_LABEL = {
    "raising_hand": "raising hand", "on_phone": "on phone", "drinking": "drinking",
    "eating": "eating", "typing": "typing", "writing": "writing",
    "reading": "reading", "studying": "reading or writing", "on_laptop": "on laptop",
    "yawning": "yawning", "eyes_closed": "eyes closed",
    "head_on_hand": "head on hand", "looking_away": "looking away",
    "head_down": "head down", "slouching": "slouching",
    "leaning_forward": "leaning forward", "attentive": "attentive",
    "unknown": "not seen",
}


def read_session(graph_path: Path):
    """Read a session graph into per-student series.

    Args:
        graph_path: The session's scene-graph JSONL.

    Returns:
        ``(timeline, positions, pairs, frames, objects, layouts)`` where
        ``timeline`` maps ``person_id`` to ``(timestamp_ms, bucket)`` samples.
    """
    timeline: dict[int, list] = defaultdict(list)
    sums: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0, 0])
    pairs: Counter = Counter()
    objects: Counter = Counter()
    layouts: Counter = Counter()
    frames = 0

    for line in graph_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        graph = json.loads(line)
        frames += 1
        ts = graph.get("timestamp_ms", 0)
        of_node = {}
        for node in graph.get("nodes", []):
            pid = node.get("person_id")
            of_node[node["id"]] = pid
            if pid is None or pid <= 0:
                continue
            feat = node.get("features") or {}
            action = feat.get("action")
            timeline[pid].append((ts, BUCKET_OF.get(action, "unknown")))
            if feat.get("layout"):
                layouts[feat["layout"]] += 1
            if feat.get("object"):
                objects[(pid, feat["object"])] += 1
            bbox = feat.get("bbox")
            if bbox:
                entry = sums[pid]
                entry[0] += bbox[0] + bbox[2] / 2
                entry[1] += bbox[1] + bbox[3] / 2
                entry[2] += 1
        for edge in graph.get("edges", []):
            if edge["type"] != "shared_action":
                continue
            a, b = of_node.get(edge["source"]), of_node.get(edge["target"])
            if a and b and a > 0 and b > 0 and a != b:
                pairs[tuple(sorted((a, b)))] += 1

    positions = {p: (x / n, y / n) for p, (x, y, n) in sums.items() if n}
    return timeline, positions, dict(pairs), frames, dict(objects), layouts


def _runs(series):
    """Collapse consecutive same-bucket samples into runs.

    Args:
        series: ``(timestamp_ms, bucket)`` samples in order.

    Returns:
        ``(start_ms, end_ms, bucket)`` runs.

    One rectangle per sample produced about 12,900 of them for a sixteen-student
    session, which is both unreadable and a megabyte of SVG. Runs carry exactly
    the same information -- a band is a band whether drawn once or thirty times.
    """
    runs = []
    for ts, bucket in series:
        if runs and runs[-1][2] == bucket:
            runs[-1][1] = ts
        else:
            runs.append([ts, ts, bucket])
    return runs


def _timeline_row(series, duration_ms, width=1000, height=18):
    """One student's session as a strip of coloured runs."""
    if not series or duration_ms <= 0:
        return ""
    parts = []
    for start, end, bucket in _runs(series):
        x = (start / duration_ms) * width
        w = max(1.0, ((end - start) / duration_ms) * width)
        parts.append(f'<rect x="{x:.1f}" y="0" width="{w:.1f}" height="{height}" '
                     f'fill="var(--{bucket})"><title>{BUCKET_LABEL[bucket]} · '
                     f'{start / 1000:.0f}-{end / 1000:.0f}s</title></rect>')
    return "".join(parts)


def _class_timeline(timeline, names, duration_ms, order):
    """Every student on one shared time axis.

    Args:
        timeline: Per-student samples.
        names: ``{person_id: label}``.
        duration_ms: Session length, so every row shares an x-axis.
        order: Person ids, top to bottom.

    Returns:
        An SVG fragment. A shared axis is the entire point: a whole-class dip
        reads as a vertical band, which per-student axes made impossible to see.
    """
    if not order or duration_ms <= 0:
        return '<p class="muted">No timeline to draw.</p>'
    row_h, gap, label_w = 18, 6, 120
    height = len(order) * (row_h + gap) + 26
    width = 1000

    opening = (f'<svg viewBox="0 0 {label_w + width} {height}" class="wide" '
               f'role="img" aria-label="Every student over the session">')
    parts = [opening]
    for i, pid in enumerate(order):
        y = i * (row_h + gap)
        parts.append(f'<text x="{label_w - 10}" y="{y + 13}" class="rowlab">'
                     f'{names.get(pid, f"#{pid}")}</text>')
        parts.append(f'<g transform="translate({label_w},{y})">'
                     f'{_timeline_row(timeline.get(pid, []), duration_ms, width, row_h)}'
                     f'</g>')
    base = len(order) * (row_h + gap)
    for i in range(6):
        x = label_w + width * i / 5
        parts.append(f'<text x="{x:.0f}" y="{base + 14}" class="tick">'
                     f'{duration_ms / 1000 * i / 5:.0f}s</text>')
    parts.append("</svg>")
    return "".join(parts)


def _overview(students, scores, engagement, names):
    """Sorted bars: who needs attention, without being told where to look."""
    if not students:
        return '<p class="muted">Nobody was recognised.</p>'
    order = sorted(students, key=lambda p: (scores.get(p["person_id"]) is None,
                                            scores.get(p["person_id"], 0)))
    row_h, gap, label_w, bar_w = 22, 8, 120, 620
    height = len(order) * (row_h + gap) + 8
    opening = (f'<svg viewBox="0 0 {label_w + bar_w + 190} {height}" class="wide" '
               f'role="img" aria-label="Students by on-task percentage">')
    parts = [opening]
    for i, p in enumerate(order):
        pid = p["person_id"]
        y = i * (row_h + gap)
        on = scores.get(pid)
        en = engagement.get(pid)
        parts.append(f'<text x="{label_w - 10}" y="{y + 15}" class="rowlab">'
                     f'{names.get(pid, f"#{pid}")}</text>')
        parts.append(f'<rect x="{label_w}" y="{y}" width="{bar_w}" height="{row_h}" '
                     f'rx="3" fill="var(--track)"/>')
        if on is not None:
            parts.append(f'<rect x="{label_w}" y="{y}" width="{bar_w * on / 100:.1f}" '
                         f'height="{row_h}" rx="3" fill="var(--working)"/>')
            parts.append(f'<text x="{label_w + bar_w + 12}" y="{y + 15}" '
                         f'class="val">{on:.0f}% on task</text>')
        else:
            parts.append(f'<text x="{label_w + bar_w + 12}" y="{y + 15}" '
                         f'class="val muted">not graded</text>')
        if en is not None:
            cx = label_w + bar_w * en / 100
            parts.append(f'<circle cx="{cx:.1f}" cy="{y + row_h / 2}" r="5" '
                         f'fill="var(--surface)" stroke="var(--participating)" '
                         f'stroke-width="2.5"><title>{en:.0f}% facing the room '
                         f'focus</title></circle>')
    parts.append("</svg>")
    return "".join(parts)


def _bucket_bar(counts):
    """Four-bucket proportion bar with visible labels."""
    totals = Counter()
    for action, n in counts.items():
        totals[BUCKET_OF.get(action, "unknown")] += n
    total = sum(totals.values())
    if not total:
        return '<span class="muted">no data</span>'
    segs, keys = [], []
    for bucket in BUCKET_ORDER:
        n = totals.get(bucket, 0)
        if not n:
            continue
        segs.append(f'<span style="width:{100 * n / total:.2f}%;'
                    f'background:var(--{bucket})" title="{BUCKET_LABEL[bucket]}: {n}">'
                    f'</span>')
        keys.append(f'<span class="key"><i style="background:var(--{bucket})"></i>'
                    f'{BUCKET_LABEL[bucket]} <b>{100 * n / total:.0f}%</b></span>')
    return f'<div class="stack">{"".join(segs)}</div><div class="legend">{"".join(keys)}</div>'


OBJECT_CSS = {"cell phone": "off_task", "laptop": "working", "book": "working",
              "keyboard": "working", "mouse": "working", "bottle": "passive",
              "cup": "passive", "tv": "participating"}


def _scene_graph(names, on_task, objects, pairs, present):
    """Students, what they handled, and who did the same thing at the same time.

    Args:
        names: ``{person_id: label}``.
        on_task: ``{person_id: percentage or None}``, for node colour.
        objects: ``{(person_id, object class): frames}``.
        pairs: ``{(a, b): frames}`` shared-action links.
        present: Person ids to draw.

    Returns:
        An SVG fragment. Students on the left, the objects they were seen
        handling on the right, links weighted by how long each lasted. Curved
        links on the left join students who were doing the same thing at the
        same time.

    Laid out as two columns rather than a force-directed cloud. A spring layout
    of sixteen students and their objects is a hairball, and position in it
    means nothing; two columns mean "person" and "thing", which is a claim a
    reader can check.
    """
    people = [p for p in sorted(present)]
    if not people:
        return '<p class="muted">Nobody was recognised, so there is nothing to link.</p>'

    # Only the links worth drawing: the strongest per student, so one busy
    # student cannot bury everybody else's.
    per_student = defaultdict(list)
    for (pid, obj), n in objects.items():
        if pid in present:
            per_student[pid].append((n, obj))
    links = []
    for pid, entries in per_student.items():
        for n, obj in sorted(entries, reverse=True)[:3]:
            links.append((pid, obj, n))
    used = sorted({o for _, o, _ in links})

    if not links and not pairs:
        return ('<p class="muted">No objects were attributed to anyone and no '
                'two students were doing the same thing at the same time, so '
                'there are no links to draw.</p>')

    rows = max(len(people), len(used), 1)
    row_h, pad = 46, 40
    H = rows * row_h + pad * 2
    W, px, ox = 760, 210, 540

    def y_of(i, total):
        return pad + (H - 2 * pad) * ((i + 0.5) / max(total, 1))

    py = {pid: y_of(i, len(people)) for i, pid in enumerate(people)}
    oy = {obj: y_of(i, len(used)) for i, obj in enumerate(used)}

    top_obj = max((n for _, _, n in links), default=1)
    top_pair = max(pairs.values(), default=1) if pairs else 1
    opening = (f'<svg viewBox="0 0 {W} {H}" class="wide" role="img" '
               f'aria-label="Students, the objects they used, and shared actions">')
    out = [opening]

    for (a, b), n in sorted(pairs.items(), key=lambda kv: kv[1]):
        if a not in py or b not in py:
            continue
        bend = px - 70 - 40 * (abs(py[a] - py[b]) / max(H, 1))
        out.append(f'<path d="M{px},{py[a]:.0f} Q{bend:.0f},'
                   f'{(py[a] + py[b]) / 2:.0f} {px},{py[b]:.0f}" fill="none" '
                   f'stroke="var(--passive)" stroke-opacity=".55" '
                   f'stroke-width="{1 + 4 * n / top_pair:.1f}">'
                   f'<title>{names.get(a, a)} and {names.get(b, b)}: same action '
                   f'in {n} frames</title></path>')

    for pid, obj, n in sorted(links, key=lambda t: t[2]):
        out.append(f'<line x1="{px}" y1="{py[pid]:.0f}" x2="{ox}" '
                   f'y2="{oy[obj]:.0f}" stroke="var(--{OBJECT_CSS.get(obj, "unknown")})" '
                   f'stroke-opacity=".5" stroke-width="{1 + 5 * n / top_obj:.1f}">'
                   f'<title>{names.get(pid, pid)} - {obj}: {n} frames</title></line>')

    for pid in people:
        pct = on_task.get(pid)
        fill = ("var(--unknown)" if pct is None else
                "var(--working)" if pct >= 60 else
                "var(--passive)" if pct >= 30 else "var(--off_task)")
        out.append(f'<circle cx="{px}" cy="{py[pid]:.0f}" r="9" fill="{fill}"/>'
                   f'<text x="{px - 16}" y="{py[pid] + 4:.0f}" class="rowlab">'
                   f'{names.get(pid, f"#{pid}")}</text>')
    for obj in used:
        out.append(f'<rect x="{ox}" y="{oy[obj] - 12:.0f}" width="150" height="24" '
                   f'rx="6" fill="var(--{OBJECT_CSS.get(obj, "unknown")})" '
                   f'fill-opacity=".18" stroke="var(--{OBJECT_CSS.get(obj, "unknown")})"/>'
                   f'<text x="{ox + 10}" y="{oy[obj] + 4:.0f}" class="objlab">{obj}</text>')
    out.append("</svg>")
    return "".join(out)


def _legend():
    """The one legend for the whole page."""
    return "".join(
        f'<span class="key"><i style="background:var(--{b})"></i>'
        f'{BUCKET_LABEL[b]}</span>' for b in BUCKET_ORDER)


def build(out_dir: Path, gallery, title="Classroom session") -> Path:
    """Write ``report.html`` for a finished session.

    Args:
        out_dir: Directory holding ``live_graph.jsonl`` and ``live_profiles.json``.
        gallery: Registered students, for names.
        title: Page title.

    Returns:
        The path written.

    Raises:
        FileNotFoundError: If the session outputs are not present.
    """
    graph_path = out_dir / "live_graph.jsonl"
    profiles_path = out_dir / "live_profiles.json"
    if not graph_path.exists() or not profiles_path.exists():
        raise FileNotFoundError(f"No finished session in {out_dir}")

    names = {p.person_id: p.name for p in gallery.people}
    profiles = json.loads(profiles_path.read_text(encoding="utf-8"))
    students = [p for p in profiles if p.get("is_student")]
    timeline, _positions, pairs, frames, objects, layouts = read_session(graph_path)
    duration = max((s[-1][0] for s in timeline.values() if s), default=0)

    label = {p["person_id"]: names.get(p["person_id"], f"Student {p['person_id']}")
             for p in students}
    on_task = {p["person_id"]: p.get("on_task_pct") for p in students}
    engaged = {p["person_id"]: p.get("engagement_pct") for p in students}
    order = sorted((p["person_id"] for p in students),
                   key=lambda pid: (on_task.get(pid) is None, on_task.get(pid, 0)))

    cards = []
    for p in sorted(students, key=lambda p: p["person_id"]):
        pid = p["person_id"]
        counts = p.get("actions", {}).get("counts", {})
        top = ", ".join(f"{ACTION_LABEL.get(k, k)} <b>{v}</b>"
                        for k, v in sorted(counts.items(), key=lambda kv: -kv[1])[:4])
        lean = p["posture"]["mean_vertical_lean"]
        cards.append(f"""
    <article class="card">
      <header><h3>{label[pid]}</h3>
        <span class="id">{p["frames_seen"]} frames</span></header>
      <div class="pair">
        <div><b>{"—" if on_task[pid] is None else f"{on_task[pid]:.0f}%"}</b>
             <small>on task</small></div>
        <div><b>{"—" if engaged[pid] is None else f"{engaged[pid]:.0f}%"}</b>
             <small>facing the room</small></div>
      </div>
      {_bucket_bar(counts)}
      <dl>
        <dt>Most of the time</dt><dd>{top or "—"}</dd>
        <dt>Posture</dt><dd>{"not read" if lean is None else f"mean lean {lean:+.2f}"}</dd>
      </dl>
    </article>""")

    kind = layouts.most_common(1)[0][0] if layouts else "unknown"
    explain = {
        "group": "Students faced each other, so <b>facing the room</b> means "
                 "engaged with the group rather than with a front.",
        "lecture": "Students faced a common point outside the seating, so "
                   "<b>facing the room</b> means oriented toward it.",
        "unknown": "The room layout could not be determined, so the "
                   "<b>facing the room</b> figure is unavailable or unreliable.",
    }[kind]

    html = f"""<title>{title}</title>
<style>
:root{{
  --surface:#fcfcfb; --panel:#ffffff; --ink:#0b0b0b; --muted:#52514e;
  --line:#e4e3df; --track:#eeedea;
  --participating:#2a78d6; --working:#1baf7a; --passive:#4a3aa7;
  --off_task:#eb6834; --unknown:#b9b8b3;
}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{
  --surface:#1a1a19; --panel:#232321; --ink:#ffffff; --muted:#c3c2b7;
  --line:#33332f; --track:#2c2c29;
  --participating:#3987e5; --working:#199e70; --passive:#9085e9;
  --off_task:#d95926; --unknown:#6b6a65;
}}}}
:root[data-theme="dark"]{{
  --surface:#1a1a19; --panel:#232321; --ink:#ffffff; --muted:#c3c2b7;
  --line:#33332f; --track:#2c2c29;
  --participating:#3987e5; --working:#199e70; --passive:#9085e9;
  --off_task:#d95926; --unknown:#6b6a65;
}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--surface);color:var(--ink);padding:40px 20px 72px;
font:15px/1.55 "Segoe UI",-apple-system,BlinkMacSystemFont,Roboto,sans-serif}}
.wrap{{max-width:1080px;margin:0 auto}}
h1{{font-size:1.7rem;margin:0 0 6px;letter-spacing:-.02em}}
.sub{{color:var(--muted);margin:0 0 8px}}
.note{{color:var(--muted);font-size:.88rem;margin:0 0 30px;
border-left:3px solid var(--line);padding-left:14px}}
h2{{font-size:.78rem;text-transform:uppercase;letter-spacing:.1em;
color:var(--muted);margin:38px 0 6px}}
.q{{color:var(--muted);font-size:.9rem;margin:0 0 14px}}
.panel{{background:var(--panel);border:1px solid var(--line);border-radius:12px;
padding:20px;overflow-x:auto}}
.wide{{width:100%;height:auto;min-width:680px;display:block}}
.rowlab{{font-size:12px;fill:var(--ink);text-anchor:end}}
.objlab{{font-size:12px;fill:var(--ink)}}
.val{{font-size:12px;fill:var(--ink)}}
.tick{{font-size:10px;fill:var(--muted);text-anchor:middle}}
.grid{{display:grid;gap:14px;grid-template-columns:repeat(auto-fill,minmax(310px,1fr))}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}}
.card header{{display:flex;justify-content:space-between;align-items:baseline}}
.card h3{{margin:0;font-size:1rem}}
.id{{color:var(--muted);font-size:.8rem}}
.pair{{display:flex;gap:26px;margin:10px 0 14px}}
.pair b{{font-size:1.6rem;font-weight:600;letter-spacing:-.02em;display:block}}
.pair small{{color:var(--muted);font-size:.68rem;text-transform:uppercase;
letter-spacing:.07em}}
.stack{{display:flex;height:10px;border-radius:5px;overflow:hidden;background:var(--track);
gap:2px}}
.stack span{{display:block}}
.legend{{margin-top:8px;font-size:.78rem;color:var(--muted)}}
.key{{margin-right:12px;white-space:nowrap;display:inline-block}}
.key i{{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px;
vertical-align:baseline}}
dl{{margin:14px 0 0;display:grid;grid-template-columns:110px 1fr;gap:8px 12px}}
dt{{font-size:.7rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}}
dd{{margin:0;font-size:.88rem}}
.muted{{color:var(--muted)}}
.pagelegend{{margin:0 0 22px;font-size:.82rem;color:var(--muted)}}
</style>
<div class="wrap">
  <h1>{title}</h1>
  <p class="sub">{len(students)} students · {frames} frames · {duration / 1000:.0f}s
     · room read as <b>{kind}</b></p>
  <p class="note">{explain} <b>On task</b> is what a student was doing, from
     detected objects and body pose. The two are reported separately and never
     averaged, because they rest on different evidence.</p>
  <p class="pagelegend">{_legend()}</p>

  <h2>Who needs attention</h2>
  <p class="q">Sorted lowest first. The ring marks how much of the time each
     student faced the room.</p>
  <div class="panel">{_overview(students, on_task, engaged, label)}</div>

  <h2>When the class drifted</h2>
  <p class="q">Every student on one shared clock, so a whole-class dip reads as
     a vertical band and one student's lapse as a gap in a single row.</p>
  <div class="panel">{_class_timeline(timeline, label, duration, order)}</div>

  <h2>Who used what, and who acted together</h2>
  <p class="q">A dot per student, coloured by how much of the time they were on
     task. Lines to the right show the objects they were seen handling, thicker
     the longer it lasted. Curves on the left join students who were doing the
     same thing at the same moment.</p>
  <div class="panel">{_scene_graph(label, on_task, objects, pairs, set(on_task))}</div>

  <h2>Each student</h2>
  <div class="grid">{"".join(cards) or '<p class="muted">Nobody recognised.</p>'}</div>
</div>
"""
    path = out_dir / "report.html"
    path.write_text(html, encoding="utf-8")
    return path
