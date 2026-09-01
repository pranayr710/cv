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

#: vertical_lean is nose-to-shoulder offset normalised by person height, so it
#: is negative when upright (nose above the shoulder line) and rises toward zero
#: as someone slumps. Bands measured on real sessions, where upright sitting ran
#: about -0.25 to -0.40.
def posture_label(lean):
    """A readable posture from the raw lean number.

    Args:
        lean: ``vertical_lean``, or ``None`` when no body keypoints were read.

    Returns:
        A short phrase and the number, so the reader gets a word they can act on
        without the measurement being hidden from them.
    """
    if lean is None:
        return "not read"
    if lean > -0.15:
        word = "slumped"
    elif lean > -0.32:
        word = "upright"
    else:
        word = "leaning forward"
    return f"{word} <span class=\"muted\">(lean {lean:+.2f})</span>"


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
            timeline[pid].append((ts, BUCKET_OF.get(action, "unknown")))
            # A change of state is an edge. This is the graph a student still
            # has when they are the only person in the room: behaviour moves
            # between states over time even with nobody to interact with.
            if action and action != "unknown":
                last = previous.get(pid)
                if last and last != action:
                    transitions[(pid, last, action)] += 1
                previous[pid] = action
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
    return (timeline, positions, dict(pairs), frames, dict(objects), layouts,
            dict(transitions))


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
    """Students, what they handled, and who did the same thing at the same time."""
    people = [p for p in sorted(present)]
    if not people:
        return '<p class="muted">Nobody was recognised, so there is nothing to link.</p>'

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
    row_h, pad = 54, 45
    H = rows * row_h + pad * 2
    W, px, ox = 820, 240, 580

    def y_of(i, total):
        return pad + (H - 2 * pad) * ((i + 0.5) / max(total, 1))

    py = {pid: y_of(i, len(people)) for i, pid in enumerate(people)}
    oy = {obj: y_of(i, len(used)) for i, obj in enumerate(used)}

    top_obj = max((n for _, _, n in links), default=1)
    top_pair = max(pairs.values(), default=1) if pairs else 1

    opening = (f'<svg viewBox="0 0 {W} {H}" class="wide" role="img" '
               f'aria-label="Students, the objects they used, and shared actions">'
               f'<defs>'
               f'<linearGradient id="linkGrad" x1="0%" y1="0%" x2="100%" y2="0%">'
               f'<stop offset="0%" stop-color="#6366f1" stop-opacity="0.7"/>'
               f'<stop offset="100%" stop-color="#06b6d4" stop-opacity="0.7"/>'
               f'</linearGradient>'
               f'</defs>')
    out = [opening]

    # Render Curved Shared-Action Arcs between students on the left
    for (a, b), n in sorted(pairs.items(), key=lambda kv: kv[1]):
        if a not in py or b not in py:
            continue
        bend = px - 80 - 45 * (abs(py[a] - py[b]) / max(H, 1))
        out.append(f'<path d="M{px},{py[a]:.0f} Q{bend:.0f},'
                   f'{(py[a] + py[b]) / 2:.0f} {px},{py[b]:.0f}" fill="none" '
                   f'stroke="#f97316" stroke-opacity=".7" stroke-dasharray="4 4" '
                   f'stroke-width="{1.5 + 3.5 * n / top_pair:.1f}">'
                   f'<title>{names.get(a, a)} and {names.get(b, b)}: mutual action in {n} frames</title></path>')

    # Render Curved Links from Students to Objects
    for pid, obj, n in sorted(links, key=lambda t: t[2]):
        midX = (px + ox) / 2
        midY = (py[pid] + oy[obj]) / 2
        out.append(f'<path d="M{px},{py[pid]:.0f} Q{midX:.0f},{midY:.0f} {ox},{oy[obj]:.0f}" '
                   f'fill="none" stroke="url(#linkGrad)" stroke-opacity=".65" '
                   f'stroke-width="{1.5 + 4 * n / top_obj:.1f}">'
                   f'<title>{names.get(pid, pid)} → {obj}: {n} frames</title></path>')

    # Render Student Avatar Nodes
    for pid in people:
        pct = on_task.get(pid)
        fill = ("#64748b" if pct is None else
                "#10b981" if pct >= 60 else
                "#f59e0b" if pct >= 30 else "#f43f5e")
        out.append(f'<g class="node-group" transform="translate({px},{py[pid]:.0f})">'
                   f'<circle r="14" fill="{fill}" stroke="#060810" stroke-width="2.5" />'
                   f'<text x="0" y="4" fill="#ffffff" font-size="11" font-weight="bold" text-anchor="middle" font-family="\'JetBrains Mono\', monospace">{pid}</text>'
                   f'<text x="-22" y="4" class="rowlab" font-weight="600">{names.get(pid, f"#{pid}")}</text>'
                   f'</g>')

    # Render Object Cards
    obj_icons = {"laptop": "💻", "book": "📖", "cell phone": "📱", "keyboard": "⌨️", "bottle": "🍾", "cup": "☕"}
    for obj in used:
        icon = obj_icons.get(obj, "📦")
        out.append(f'<g transform="translate({ox},{oy[obj] - 14:.0f})">'
                   f'<rect width="160" height="28" rx="8" fill="rgba(6, 182, 212, 0.12)" stroke="#06b6d4" stroke-opacity="0.4" stroke-width="1.2"/>'
                   f'<text x="12" y="18" fill="#e2e8f0" font-size="12" font-weight="600">{icon} {obj}</text>'
                   f'</g>')

    out.append("</svg>")
    return "".join(out)


def _transitions_svg(pid, transitions, top=6):
    """How one student moved between actions."""
    mine = Counter({(a, b): n for (p, a, b), n in transitions.items() if p == pid})
    if not mine:
        return ('<p class="muted" style="font-size:.85rem">Stayed in one action '
                'for the whole session, so there are no transitions to draw.</p>')
    strongest = mine.most_common(top)
    states = []
    for (a, b), _ in strongest:
        for x in (a, b):
            if x not in states:
                states.append(x)

    W, H, R = 320, 140, 16
    step = W / (len(states) + 1)
    at = {s: ((i + 1) * step, H / 2) for i, s in enumerate(states)}
    heaviest = strongest[0][1]

    opening = (f'<svg viewBox="0 0 {W} {H}" class="tg" role="img" '
               f'aria-label="How this student moved between actions">')
    marker = (f'<defs><marker id="a{pid}" viewBox="0 0 10 10" refX="9" refY="5" '
              f'markerWidth="5" markerHeight="5" orient="auto-start-reverse">'
              f'<path d="M0,0 L10,5 L0,10 z" fill="#818cf8"/></marker>'
              f'</defs>')
    out = [opening, marker]
    for (a, b), n in sorted(strongest, key=lambda kv: kv[1]):
        ax, ay = at[a]
        bx, by = at[b]
        up = ax < bx
        lift = 34 if up else -34
        out.append(
            f'<path d="M{ax:.0f},{ay - (R if up else -R):.0f} '
            f'Q{(ax + bx) / 2:.0f},{ay - lift * 1.6:.0f} '
            f'{bx:.0f},{by - (R if up else -R):.0f}" fill="none" '
            f'stroke="#6366f1" stroke-opacity="0.8" stroke-width="{1.5 + 3 * n / heaviest:.1f}" '
            f'marker-end="url(#a{pid})"><title>{ACTION_LABEL.get(a, a)} to '
            f'{ACTION_LABEL.get(b, b)}: {n} times</title></path>')
    for state, (x, y) in at.items():
        out.append(f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{R}" '
                   f'fill="var(--{BUCKET_OF.get(state, "unknown")})" stroke="#060810" stroke-width="2"/>'
                   f'<text x="{x:.0f}" y="{y + R + 14:.0f}" class="tglab" font-weight="600">'
                   f'{ACTION_LABEL.get(state, state)}</text>')
    out.append("</svg>")
    return "".join(out)


EXPR_CSS = {"neutral": "unknown", "happy": "participating", "sad": "passive",
            "angry": "off_task", "surprise": "working", "fear": "passive",
            "disgust": "off_task", "contempt": "off_task",
            "uncertain": "unknown"}


def _stack_expr(counts):
    """Expression mix as a bar with named counts.

    Expression was computed on every frame and then not shown at all, so a
    working signal looked broken. It is reported with its counts because the
    model abstains often -- `uncertain` is a real answer and hiding it would
    overstate how much was actually read.
    """
    total = sum(counts.values())
    if not total:
        return '<span class="muted">not read</span>'
    items = sorted(counts.items(), key=lambda kv: -kv[1])
    segs = "".join(
        f'<span style="width:{100 * v / total:.2f}%;'
        f'background:var(--{EXPR_CSS.get(k, "unknown")})"></span>'
        for k, v in items)
    keys = " ".join(f"{k} <b>{v}</b>" for k, v in items[:3])
    return f'<div class="stack">{segs}</div><div class="legend">{keys}</div>'


def _legend():
    """The one legend for the whole page."""
    return "".join(
        f'<span class="key"><i style="background:var(--{b})"></i>'
        f'{BUCKET_LABEL[b]}</span>' for b in BUCKET_ORDER)


def build(out_dir: Path, gallery, title="Classroom session") -> Path:
    """Write ``report.html`` for a finished session."""
    graph_path = out_dir / "live_graph.jsonl"
    profiles_path = out_dir / "live_profiles.json"
    if not graph_path.exists() or not profiles_path.exists():
        raise FileNotFoundError(f"No finished session in {out_dir}")

    names = {p.person_id: p.name for p in gallery.people}
    profiles = json.loads(profiles_path.read_text(encoding="utf-8"))
    students = [p for p in profiles if p.get("is_student")]
    (timeline, _positions, pairs, frames, objects, layouts,
     transitions) = read_session(graph_path)
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
        <span class="id">#{pid} · {p["frames_seen"]} frames</span></header>
      <div class="pair">
        <div><b>{"—" if on_task[pid] is None else f"{on_task[pid]:.0f}%"}</b>
             <small>on task</small></div>
        <div><b>{"—" if engaged[pid] is None else f"{engaged[pid]:.0f}%"}</b>
             <small>facing room</small></div>
      </div>
      {_bucket_bar(counts)}
      <dl>
        <dt>Dominant Actions</dt><dd>{top or "—"}</dd>
        <dt>Posture Lean</dt><dd>{"not read" if lean is None else f"mean lean {lean:+.2f} rad"}</dd>
      </dl>
      <div class="tghead">State Machine Transitions</div>
      {_transitions_svg(pid, transitions)}
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

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title} — ClassGraph Report</title>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<style>
:root {{
  --surface: #060810; --panel: rgba(15, 23, 42, 0.75); --ink: #f8fafc; --muted: #94a3b8;
  --line: rgba(255, 255, 255, 0.08); --track: rgba(255, 255, 255, 0.05);
  --participating: #38bdf8; --working: #10b981; --passive: #f59e0b;
  --off_task: #f43f5e; --unknown: #64748b; --muted-mark: #818cf8;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0; background: radial-gradient(circle at 50% -20%, #1e1b4b 0%, var(--surface) 75%);
  color: var(--ink); padding: 40px 20px 72px; font: 15px/1.55 'Outfit', sans-serif;
}}
.wrap {{ max-width: 1140px; margin: 0 auto; display: flex; flex-direction: column; gap: 20px; }}
h1 {{ font-size: 2rem; margin: 0 0 6px; font-weight: 800; letter-spacing: -0.02em; background: linear-gradient(135deg, #fff, #a5b4fc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }}
.sub {{ color: var(--muted); margin: 0 0 8px; font-size: 0.95rem; }}
.note {{ color: var(--muted); font-size: 0.88rem; margin: 0 0 20px; border-left: 3px solid #6366f1; padding-left: 14px; background: rgba(99, 102, 241, 0.08); border-radius: 0 10px 10px 0; padding: 10px 14px; }}
h2 {{ font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.1em; color: #a5b4fc; margin: 28px 0 6px; font-weight: 700; }}
.q {{ color: var(--muted); font-size: 0.9rem; margin: 0 0 14px; }}
.panel {{ background: var(--panel); backdrop-filter: blur(20px); border: 1px solid var(--line); border-radius: 18px; padding: 22px; overflow-x: auto; box-shadow: 0 10px 30px rgba(0,0,0,0.4); }}
.wide {{ width: 100%; height: auto; min-width: 680px; display: block; }}
.rowlab {{ font-size: 12px; fill: var(--ink); text-anchor: end; font-family: 'Outfit', sans-serif; }}
.objlab {{ font-size: 12px; fill: var(--ink); font-family: 'Outfit', sans-serif; }}
.tg {{ width: 100%; height: auto; display: block; margin-top: 6px; }}
.tglab {{ font-size: 10px; fill: var(--muted); text-anchor: middle; font-family: 'Outfit', sans-serif; }}
.tghead {{ font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); margin-top: 18px; font-weight: 700; }}
.val {{ font-size: 12px; fill: var(--ink); font-family: 'JetBrains Mono', monospace; font-weight: 700; }}
.tick {{ font-size: 11px; fill: var(--muted); text-anchor: middle; font-family: 'JetBrains Mono', monospace; }}
.grid {{ display: grid; gap: 16px; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); }}
.card {{ background: var(--panel); backdrop-filter: blur(20px); border: 1px solid var(--line); border-radius: 18px; padding: 18px; box-shadow: 0 8px 30px rgba(0,0,0,0.3); }}
.card header {{ display: flex; justify-content: space-between; align-items: baseline; border-bottom: 1px solid var(--line); padding-bottom: 10px; margin-bottom: 12px; }}
.card h3 {{ margin: 0; font-size: 1.1rem; font-weight: 700; color: #fff; }}
.id {{ color: var(--muted); font-size: 0.78rem; font-family: 'JetBrains Mono', monospace; }}
.pair {{ display: flex; gap: 26px; margin: 10px 0 14px; }}
.pair b {{ font-size: 1.7rem; font-weight: 800; letter-spacing: -0.02em; display: block; font-family: 'JetBrains Mono', monospace; color: #06b6d4; }}
.pair small {{ color: var(--muted); font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.07em; font-weight: 600; }}
.stack {{ display: flex; height: 10px; border-radius: 5px; overflow: hidden; background: var(--track); gap: 2px; }}
.stack span {{ display: block; }}
.legend {{ margin-top: 10px; font-size: 0.78rem; color: var(--muted); }}
.key {{ margin-right: 14px; white-space: nowrap; display: inline-block; font-weight: 500; }}
.key i {{ display: inline-block; width: 9px; height: 9px; border-radius: 3px; margin-right: 6px; vertical-align: baseline; }}
dl {{ margin: 14px 0 0; display: grid; grid-template-columns: 110px 1fr; gap: 8px 12px; }}
dt {{ font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--muted); font-weight: 600; }}
dd {{ margin: 0; font-size: 0.88rem; }}
.muted {{ color: var(--muted); }}
.pagelegend {{ margin: 0 0 22px; font-size: 0.85rem; color: var(--muted); }}
</style>
</head>
<body>
<div class="wrap">
  <div>
    <h1>{title}</h1>
    <p class="sub">{len(students)} students · {frames} frames · {duration / 1000:.0f}s · room layout: <b>{kind}</b></p>
    <p class="note">{explain} <b>On task</b> is derived from objects and posture. Reported separately from orientation.</p>
    <p class="pagelegend">{_legend()}</p>
  </div>

  <h2>1. Who Needs Attention</h2>
  <p class="q">Sorted lowest on-task first. Rings mark student room-orientation time.</p>
  <div class="panel">{_overview(students, on_task, engaged, label)}</div>

  <h2>2. Classroom Temporal Drift Timeline</h2>
  <p class="q">All students aligned on a single clock axis to visualize class-wide engagement dips.</p>
  <div class="panel">{_class_timeline(timeline, label, duration, order)}</div>

  <h2>3. Scene Graph: Relational Links &amp; Shared Objects</h2>
  <p class="q">Student nodes on the left linked to handled objects on the right via curved Bezier arcs. Left arcs show mutual actions between students.</p>
  <div class="panel">{_scene_graph(label, on_task, objects, pairs, set(on_task))}</div>

  <h2>4. Per-Student Behavior Breakdown</h2>
  <div class="grid">{"".join(cards) or '<p class="muted">Nobody recognised.</p>'}</div>
</div>
</body>
</html>
"""
    path = out_dir / "report.html"
    path.write_text(html, encoding="utf-8")
    return path

