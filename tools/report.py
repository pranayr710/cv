"""Render a session report: what each student did, and how it changed.

The interaction graph alone is a poor summary of a session, and for a single
student it is one dot and nothing else. What actually carries the session is
**time**: when someone looked away, for how long, and whether it drifted. So
the report leads with a per-student attention timeline and keeps the node graph
for what it is genuinely good at -- who was near whom.

Everything is inline (no external CSS, fonts or scripts), so the file can be
opened straight from disk or sent to somebody as one attachment.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

#: Gaze labels in a fixed order, so a colour always means the same thing.
GAZE_ORDER = ("teacher", "left", "right", "down", "back")
GAZE_CSS = {
    "teacher": "#3f9e5f", "left": "#d98324", "right": "#a855b8",
    "down": "#2f7fd1", "back": "#8a8f98",
}
ACTION_CSS = {
    "attentive": "#3f9e5f", "studying": "#4a86c9", "on_laptop": "#3d7ea6",
    "head_down": "#7d848f", "looking_away": "#d98324", "on_phone": "#c0522f",
    "eyes_closed": "#8b3a52", "unknown": "#3a3f46",
}
ACTION_LABEL = {
    "on_phone": "on phone", "studying": "reading / writing",
    "on_laptop": "on laptop", "eyes_closed": "eyes closed",
    "looking_away": "looking away", "head_down": "head down",
    "attentive": "attentive", "unknown": "no face read",
}
EXPR_CSS = {
    "neutral": "#8a8f98", "happy": "#3f9e5f", "sad": "#2f7fd1",
    "angry": "#c0522f", "surprise": "#d98324", "fear": "#a855b8",
    "disgust": "#7a6a3a", "contempt": "#6b5b95", "uncertain": "#b9bec6",
}


def read_session(graph_path: Path):
    """Read a session graph into per-student series and pair counts.

    Args:
        graph_path: The session's graph JSONL, one scene graph per line.

    Returns:
        ``(timeline, positions, pairs, frames)`` where ``timeline`` maps
        ``person_id`` to a list of ``(timestamp_ms, gaze_label|None)``,
        ``positions`` maps to a mean on-screen point, ``pairs`` counts frames
        each pair was linked, and ``frames`` is the frame count.
    """
    timeline: dict[int, list] = defaultdict(list)
    sums: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0, 0])
    pairs: Counter = Counter()
    objects: Counter = Counter()
    transitions: Counter = Counter()
    previous: dict[int, str] = {}
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
            timeline[pid].append((ts, feat.get("gaze_label"), action))
            if feat.get("object"):
                objects[(pid, feat["object"])] += 1
            # A change of state is an edge: this is the graph a single student
            # still has, because behaviour moves between states over time even
            # when there is nobody to interact with.
            if action and action != "unknown":
                last = previous.get(pid)
                if last and last != action:
                    transitions[(pid, last, action)] += 1
                previous[pid] = action
            bbox = feat.get("bbox")
            if bbox:
                entry = sums[pid]
                entry[0] += bbox[0] + bbox[2] / 2
                entry[1] += bbox[1] + bbox[3] / 2
                entry[2] += 1
        for edge in graph.get("edges", []):
            a, b = of_node.get(edge["source"]), of_node.get(edge["target"])
            if a and b and a > 0 and b > 0 and a != b:
                pairs[tuple(sorted((a, b)))] += 1

    positions = {p: (x / n, y / n) for p, (x, y, n) in sums.items() if n}
    return (timeline, positions, dict(pairs), frames,
            dict(objects), dict(transitions))


def _timeline_svg(series, duration_ms: int, width=680, height=64, key="action") -> str:
    """Draw one student's attention over the session as a banded strip.

    Args:
        series: ``(timestamp_ms, gaze_label|None)`` in order.
        duration_ms: Total session length, so every student shares an x-axis.
        width: SVG width in user units.
        height: SVG height in user units.

    Returns:
        An SVG fragment. Each sample is a vertical band coloured by where the
        student was looking, so a glance away is visible as a stripe rather
        than being averaged out of existence.
    """
    if not series or duration_ms <= 0:
        return '<p class="muted">No timeline — nothing was graded.</p>'

    palette = ACTION_CSS if key == "action" else GAZE_CSS
    band = max(1.6, width / max(len(series), 1))
    bars = []
    for ts, gaze, action in series:
        label = action if key == "action" else gaze
        x = (ts / duration_ms) * width
        colour = palette.get(label, "#3a3f46") if label else "#2b2f35"
        bars.append(f'<rect x="{x:.2f}" y="0" width="{band:.2f}" height="{height}" '
                    f'fill="{colour}"/>')

    ticks = []
    for i in range(6):
        x = width * i / 5
        secs = duration_ms / 1000 * i / 5
        ticks.append(f'<line x1="{x:.1f}" y1="{height}" x2="{x:.1f}" y2="{height + 4}" '
                     f'stroke="var(--line)"/>'
                     f'<text x="{x:.1f}" y="{height + 16}" class="tick">{secs:.0f}s</text>')

    return (f'<svg viewBox="0 -1 {width} {height + 22}" class="timeline" '
            f'preserveAspectRatio="none" role="img" aria-label="Attention over time">'
            f'{"".join(bars)}{"".join(ticks)}</svg>')


def _graph_svg(positions, pairs, names, scores) -> str:
    """Draw who sat near whom, or explain why there is nothing to draw."""
    if len(positions) < 2:
        return ('<p class="muted">An interaction graph needs at least two '
                'recognised students — with one person there are no pairs to '
                'link. Register a second student and run again.</p>')

    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    W, H, PAD = 680, 340, 64

    def place(pid):
        px, py = positions[pid]
        fx = 0.5 if x1 == x0 else (px - x0) / (x1 - x0)
        fy = 0.5 if y1 == y0 else (py - y0) / (y1 - y0)
        return PAD + fx * (W - 2 * PAD), PAD + fy * (H - 2 * PAD)

    top = max(pairs.values()) if pairs else 1
    opening = (f'<svg viewBox="0 0 {W} {H}" class="graph" role="img" '
               f'aria-label="Who sat near whom">')
    out = [opening]
    for (a, b), n in sorted(pairs.items(), key=lambda kv: kv[1]):
        if a not in positions or b not in positions:
            continue
        ax, ay = place(a)
        bx, by = place(b)
        out.append(f'<line x1="{ax:.0f}" y1="{ay:.0f}" x2="{bx:.0f}" y2="{by:.0f}" '
                   f'stroke="var(--edge)" stroke-width="{1 + 6 * n / top:.1f}" '
                   f'stroke-linecap="round"><title>{n} frames</title></line>')
    for pid in positions:
        cx, cy = place(pid)
        pct = scores.get(pid)
        fill = ("#5a6068" if pct is None else
                "#3f9e5f" if pct >= .7 else "#d98324" if pct >= .4 else "#c0522f")
        out.append(
            f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="28" fill="{fill}"/>'
            f'<text x="{cx:.0f}" y="{cy + 5:.0f}" class="nodeval">'
            f'{"—" if pct is None else f"{pct * 100:.0f}%"}</text>'
            f'<text x="{cx:.0f}" y="{cy + 48:.0f}" class="nodename">'
            f'{names.get(pid, f"#{pid}")}</text>')
    out.append("</svg>")
    return "".join(out)


def _action_key(counts: dict) -> str:
    """Legend for the action timeline, only for actions actually observed."""
    if not counts:
        return '<span class="muted">no actions graded</span>'
    return "".join(
        f'<span class="key"><i style="background:{ACTION_CSS.get(k, "#6b7280")}"></i>'
        f'{ACTION_LABEL.get(k, k)} <b>{v}</b></span>'
        for k, v in sorted(counts.items(), key=lambda kv: -kv[1]))


OBJECT_CSS = {"book": "#4a86c9", "laptop": "#3d7ea6", "cell phone": "#c0522f"}


def _scene_graph_svg(names, scores, pairs, objects, present) -> str:
    """Draw students, the objects they handled, and links between them.

    Args:
        names: ``{person_id: name}``.
        scores: ``{person_id: attention fraction or None}``.
        pairs: ``{(a, b): frames}`` shared-action links between students.
        objects: ``{(person_id, object class): frames}``.
        present: Person ids that appear in this session.

    Returns:
        An SVG fragment. A student sitting alone still produces a graph here,
        because a person handling a book is a relation -- the seating-only
        version had nothing to draw with one node, which made an empty picture
        look like a broken feature rather than an absent relationship.
    """
    if not present:
        return '<p class="muted">Nobody was recognised, so there is nothing to link.</p>'

    people = sorted(present)
    used = sorted({o for _, o in objects})
    W, H = 680, max(240, 120 + 80 * max(len(people), len(used)))
    px = 190
    ox = 500

    place_p = {p: (px, 80 + i * (H - 140) / max(len(people) - 1, 1) if len(people) > 1
                   else H / 2) for i, p in enumerate(people)}
    place_o = {o: (ox, 80 + i * (H - 140) / max(len(used) - 1, 1) if len(used) > 1
                   else H / 2) for i, o in enumerate(used)}

    top_obj = max(objects.values(), default=1)
    top_pair = max(pairs.values(), default=1)
    opening = (f'<svg viewBox="0 0 {W} {H}" class="graph" role="img" '
               f'aria-label="Students, objects and the links between them">')
    out = [opening]

    for (pid, obj), n in sorted(objects.items(), key=lambda kv: kv[1]):
        if pid not in place_p:
            continue
        ax, ay = place_p[pid]
        bx, by = place_o[obj]
        out.append(f'<line x1="{ax:.0f}" y1="{ay:.0f}" x2="{bx:.0f}" y2="{by:.0f}" '
                   f'stroke="{OBJECT_CSS.get(obj, "#6b7280")}" stroke-opacity=".55" '
                   f'stroke-width="{1 + 6 * n / top_obj:.1f}" stroke-linecap="round">'
                   f'<title>{n} frames</title></line>'
                   f'<text x="{(ax + bx) / 2:.0f}" y="{(ay + by) / 2 - 6:.0f}" '
                   f'class="edgelab">{n}</text>')

    for (a, b), n in sorted(pairs.items(), key=lambda kv: kv[1]):
        if a not in place_p or b not in place_p:
            continue
        ax, ay = place_p[a]
        bx, by = place_p[b]
        mid = ax - 70
        out.append(f'<path d="M{ax:.0f},{ay:.0f} Q{mid:.0f},{(ay + by) / 2:.0f} '
                   f'{bx:.0f},{by:.0f}" fill="none" stroke="var(--edge)" '
                   f'stroke-width="{1 + 6 * n / top_pair:.1f}"><title>'
                   f'{n} frames doing the same thing</title></path>')

    for pid, (x, y) in place_p.items():
        pct = scores.get(pid)
        fill = ("#5a6068" if pct is None else "#3f9e5f" if pct >= .7
                else "#d98324" if pct >= .4 else "#c0522f")
        out.append(
            f'<circle cx="{x:.0f}" cy="{y:.0f}" r="30" fill="{fill}"/>'
            f'<text x="{x:.0f}" y="{y + 5:.0f}" class="nodeval">'
            f'{"—" if pct is None else f"{pct * 100:.0f}%"}</text>'
            f'<text x="{x:.0f}" y="{y + 50:.0f}" class="nodename">'
            f'{names.get(pid, f"#{pid}")}</text>')

    for obj, (x, y) in place_o.items():
        colour = OBJECT_CSS.get(obj, "#6b7280")
        out.append(
            f'<rect x="{x - 34:.0f}" y="{y - 22:.0f}" width="68" height="44" rx="9" '
            f'fill="{colour}" fill-opacity=".22" stroke="{colour}" stroke-width="2"/>'
            f'<text x="{x:.0f}" y="{y + 5:.0f}" class="objname">{obj}</text>')

    out.append("</svg>")
    return "".join(out)


def _transition_svg(pid, transitions) -> str:
    """Draw one student's action-transition graph.

    Args:
        pid: The student.
        transitions: ``{(person_id, from, to): count}``.

    Returns:
        An SVG fragment, or a note when the student never changed state. Nodes
        are actions and edges are moves between them, so this is a genuine
        graph for a single person -- what they did, and what it led to.
    """
    mine = {(a, b): n for (p, a, b), n in transitions.items() if p == pid}
    if not mine:
        return ('<p class="muted">No state changes — this student stayed in one '
                'action for the whole session.</p>')

    states = sorted({a for a, _ in mine} | {b for _, b in mine})
    W, H, R = 660, 200, 26
    step = W / (len(states) + 1)
    at = {s: ((i + 1) * step, H / 2) for i, s in enumerate(states)}
    top = max(mine.values())

    opening = (f'<svg viewBox="0 0 {W} {H}" class="graph" role="img" '
               f'aria-label="How this student moved between actions">')
    marker = ('<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" '
              'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
              '<path d="M0,0 L10,5 L0,10 z" fill="var(--edge)"/></marker></defs>')
    out = [opening, marker]

    for (a, b), n in sorted(mine.items(), key=lambda kv: kv[1]):
        ax, ay = at[a]
        bx, by = at[b]
        lift = 58 if ax < bx else -58
        out.append(
            f'<path d="M{ax:.0f},{ay - (R if lift > 0 else -R):.0f} '
            f'Q{(ax + bx) / 2:.0f},{ay - lift * 1.5:.0f} '
            f'{bx:.0f},{by - (R if lift > 0 else -R):.0f}" fill="none" '
            f'stroke="var(--edge)" stroke-width="{1 + 5 * n / top:.1f}" '
            f'marker-end="url(#ah)"><title>{a} to {b}: {n} times</title></path>'
            f'<text x="{(ax + bx) / 2:.0f}" y="{ay - lift * 1.1:.0f}" '
            f'class="edgelab">{n}</text>')

    for state, (x, y) in at.items():
        out.append(
            f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{R}" '
            f'fill="{ACTION_CSS.get(state, "#6b7280")}"/>'
            f'<text x="{x:.0f}" y="{y + 44:.0f}" class="nodename">'
            f'{ACTION_LABEL.get(state, state)}</text>')
    out.append("</svg>")
    return "".join(out)


def _stack(counts: dict, palette: dict) -> str:
    """A stacked proportion bar with a legend, widest segment first."""
    total = sum(counts.values())
    if not total:
        return '<span class="muted">no data</span>'
    items = sorted(counts.items(), key=lambda kv: -kv[1])
    segs = "".join(
        f'<span style="width:{100 * v / total:.2f}%;background:'
        f'{palette.get(k, "#6b7280")}" title="{k}: {v}"></span>' for k, v in items)
    keys = "".join(
        f'<span class="key"><i style="background:{palette.get(k, "#6b7280")}"></i>'
        f'{k} <b>{v}</b></span>' for k, v in items)
    return f'<div class="stack">{segs}</div><div class="legend">{keys}</div>'


def build(out_dir: Path, gallery, title="Classroom session") -> Path:
    """Write ``report.html`` for a finished session.

    Args:
        out_dir: Directory holding ``live_graph.jsonl`` and
            ``live_profiles.json``.
        gallery: The registered students, for names.
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
    students = sorted((p for p in profiles if p.get("is_student")),
                      key=lambda p: p["person_id"])
    timeline, positions, pairs, frames, objects, transitions = read_session(graph_path)
    duration = max((s[-1][0] for s in timeline.values() if s), default=0)

    scores = {}
    for p in students:
        counts = p["attention"]["counts"]
        total = sum(counts.values())
        scores[p["person_id"]] = (counts.get("teacher", 0) / total) if total else None

    keep = set(scores)
    positions = {k: v for k, v in positions.items() if k in keep}
    pairs = {k: v for k, v in pairs.items() if k[0] in keep and k[1] in keep}

    cards = []
    for p in students:
        pid = p["person_id"]
        pct = scores[pid]
        on_task = p.get("on_task_pct")
        lean = p["posture"]["mean_vertical_lean"]
        away = sum(v for k, v in p["attention"]["counts"].items() if k != "teacher")
        cards.append(f"""
    <article class="card">
      <header>
        <div><h3>{names.get(pid, f"Student #{pid}")}</h3>
             <span class="id">id {pid} · seen in {p["frames_seen"]} frames</span></div>
        <div class="scores">
          <div class="big" style="color:{"#3f9e5f" if (on_task or 0) >= 70 else "#d98324"}">
            {"—" if on_task is None else f"{on_task:.0f}%"}<small>on task</small></div>
          <div class="big small2" style="color:{"#3f9e5f" if (pct or 0) >= .7 else "#d98324"}">
            {"—" if pct is None else f"{pct * 100:.0f}%"}<small>looking forward</small></div>
        </div>
      </header>
      <div class="tl-head">What they were doing</div>
      {_timeline_svg(timeline.get(pid, []), duration, key="action")}
      <div class="legend">{_action_key(p.get("actions", {}).get("counts", {}))}</div>
      <dl>
        <dt>Actions</dt><dd>{_stack(p.get("actions", {}).get("counts", {}), ACTION_CSS)}</dd>
        <dt>Where they looked</dt><dd>{_stack(p["attention"]["counts"], GAZE_CSS)}</dd>
        <dt>Expression</dt><dd>{_stack(p["expression"]["counts"], EXPR_CSS)}</dd>
        <dt>Posture</dt><dd>{"no body keypoints" if lean is None else
          f"mean lean {lean:+.3f} · keypoints in {p['posture']['frames_with_keypoints']} frames"}</dd>
        <dt>Looked away</dt><dd>{away} frame(s)</dd>
      </dl>
      <div class="tl-head" style="margin-top:20px">How they moved between actions</div>
      {_transition_svg(pid, transitions)}
    </article>""")

    caveat = (students[0]["concentration"]["caveat"] if students else "")
    unknown = len(profiles) - len(students)

    html = f"""<title>{title}</title>
<style>
:root{{--bg:#0f1113;--panel:#17191c;--panel2:#1e2126;--ink:#eceef1;--muted:#9aa1ab;
--line:#2b2f35;--edge:#3a4049;}}
@media (prefers-color-scheme: light){{:root:not([data-theme="dark"]){{
--bg:#f7f7f6;--panel:#fff;--panel2:#f1f2f4;--ink:#16181b;--muted:#666d78;
--line:#e2e5e9;--edge:#c3c9d1;}}}}
:root[data-theme="light"]{{--bg:#f7f7f6;--panel:#fff;--panel2:#f1f2f4;--ink:#16181b;
--muted:#666d78;--line:#e2e5e9;--edge:#c3c9d1;}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--ink);padding:40px 20px 72px;
font:15px/1.55 "Segoe UI",-apple-system,BlinkMacSystemFont,Roboto,sans-serif}}
.wrap{{max-width:820px;margin:0 auto}}
h1{{font-size:1.75rem;margin:0 0 6px;letter-spacing:-.02em}}
.sub{{color:var(--muted);margin:0 0 34px}}
h2{{font-size:.78rem;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);
margin:40px 0 14px}}
.panel{{background:var(--panel);border:1px solid var(--line);border-radius:14px;
padding:22px;overflow-x:auto}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:14px;
padding:22px;margin-bottom:18px}}
.card header{{display:flex;justify-content:space-between;align-items:flex-start;
gap:16px;margin-bottom:16px}}
.card h3{{margin:0;font-size:1.15rem}}
.id{{color:var(--muted);font-size:.82rem}}
.big{{font-size:2.3rem;font-weight:600;line-height:1;letter-spacing:-.03em;
text-align:right}}
.big small{{display:block;font-size:.68rem;font-weight:400;color:var(--muted);
text-transform:uppercase;letter-spacing:.08em;margin-top:6px}}
.timeline{{width:100%;height:86px;display:block;border-radius:6px;overflow:hidden;
background:var(--panel2)}}
.tick{{font-size:9px;fill:var(--muted);text-anchor:middle}}
.graph{{width:100%;height:auto;min-width:520px;display:block}}
.nodeval{{font-size:13px;font-weight:600;fill:#fff;text-anchor:middle}}
.nodename{{font-size:12px;fill:var(--ink);text-anchor:middle}}
dl{{margin:18px 0 0;display:grid;grid-template-columns:150px 1fr;gap:10px 16px;
align-items:center}}
dt{{font-size:.72rem;text-transform:uppercase;letter-spacing:.06em;color:var(--muted)}}
dd{{margin:0}}
.stack{{display:flex;height:10px;border-radius:5px;overflow:hidden;background:var(--line)}}
.stack span{{display:block}}
.legend{{margin-top:7px;font-size:.78rem;color:var(--muted)}}
.key{{margin-right:12px;white-space:nowrap}}
.key i{{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:5px}}
.muted{{color:var(--muted)}}
.note{{border-left:3px solid var(--line);padding:4px 0 4px 16px;color:var(--muted);
font-size:.86rem;margin-top:34px}}
</style>
<div class="wrap">
  <h1>{title}</h1>
  <p class="sub">{len(students)} student(s) · {frames} frames ·
     {duration / 1000:.0f}s{f" · {unknown} unidentified detection(s) not counted" if unknown else ""}</p>

  <h2>Per student</h2>
  {"".join(cards) or '<p class="muted">Nobody was recognised.</p>'}

  <h2>Scene graph</h2>
  <div class="panel">
    {_scene_graph_svg(names, scores, pairs, objects, set(scores))}
    <p class="muted" style="margin:14px 0 0;font-size:.85rem">
      Circles are students, coloured by attention. Rectangles are objects they
      were seen handling; a line's thickness is how many frames that lasted.
      Curved links between students mean they were doing the same thing at the
      same time.</p>
  </div>

  <h2>Where they sat</h2>
  <div class="panel">{_graph_svg(positions, pairs, names, scores)}</div>

  <p class="note">{caveat}</p>
</div>
"""
    path = out_dir / "report.html"
    path.write_text(html, encoding="utf-8")
    return path
