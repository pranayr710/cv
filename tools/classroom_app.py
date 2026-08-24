"""The whole thing as one guided app: register, record, then see the results.

Three steps, in order, with nothing to wire up in between:

    1. REGISTER  every student, once each, from the webcam
    2. RECORD    the session live, with an overlay showing what is being read
    3. REPORT    an HTML page: per-student scores and the interaction graph

Each step hands its output to the next, so the gallery from step 1 is what
names people in step 2, and the graph from step 2 is what step 3 draws.

Run:
    python -m tools.classroom_app

    # skip registration and use the students already registered
    python -m tools.classroom_app --skip-register

    # rebuild the report from the last session without recording again
    python -m tools.classroom_app --report-only

Privacy: a gallery is biometric data about named people -- see the privacy
section of backend/enrollment.py. Video frames are never saved; only the
derived per-frame records, the graph, and the profiles are.
"""

from __future__ import annotations

import argparse
import json
import logging
import webbrowser
from argparse import Namespace
from collections import Counter, defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

SHOTS_PER_STUDENT = 8


# --------------------------------------------------------------------------- #
# Step 1 -- registration
# --------------------------------------------------------------------------- #


def register_students(args) -> int:
    """Walk the operator through registering every student from the webcam.

    Args:
        args: Parsed CLI arguments.

    Returns:
        How many students are registered when the step finishes.
    """
    import cv2

    from backend.config import CONFIG
    from backend.enrollment import EnrolledGallery
    from backend.face import FaceAnalyzer

    gallery = EnrolledGallery.load(args.gallery, CONFIG.identity)
    if len(gallery):
        print("Already registered: " + ", ".join(p.name for p in gallery.people))

    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}.")

    print(
        "\nSTEP 1 of 3 -- REGISTER STUDENTS\n"
        "  One student at a time, facing the camera.\n"
        "  Type their name and press Enter, then hold still while it captures.\n"
        "  Press Enter on an empty name when everyone is registered.\n"
    )
    try:
        with FaceAnalyzer() as analyzer:
            while True:
                name = input("Student name (blank to finish): ").strip()
                if not name:
                    break

                print(f"  capturing {SHOTS_PER_STUDENT} shots for {name} -- "
                      f"look at the camera and move your head slightly")
                embeddings = []
                attempts = 0
                while len(embeddings) < SHOTS_PER_STUDENT and attempts < 300:
                    attempts += 1
                    ok, frame = capture.read()
                    if not ok:
                        break
                    faces = analyzer.detect_faces(frame)
                    good = [
                        f for f in faces
                        if f.embedding is not None
                        and f.score >= CONFIG.identity.min_face_score_for_identity
                        and min(f.bbox[2], f.bbox[3]) >= args.min_face_px
                    ]
                    if len(good) == 1:
                        embeddings.append(good[0].embedding)
                    if not args.no_window:
                        for f in faces:
                            x, y, w, h = (int(v) for v in f.bbox)
                            colour = (80, 220, 80) if len(good) == 1 else (80, 80, 220)
                            cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 2)
                        status = (
                            f"{name}: {len(embeddings)}/{SHOTS_PER_STUDENT}"
                            if len(good) == 1
                            else ("move closer / face the camera" if not good
                                  else f"{len(good)} faces -- only one person please")
                        )
                        cv2.putText(frame, status, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.imshow("Register students", frame)
                        cv2.waitKey(1)

                if len(embeddings) < 3:
                    print(f"  -> NOT registered: only {len(embeddings)} usable shots. "
                          f"Try again with better light, or move closer.\n")
                    continue
                person = gallery.register(name, embeddings)
                gallery.save(args.gallery)
                print(f"  -> {name} registered as student #{person.person_id} "
                      f"({len(embeddings)} shots)\n")
    finally:
        capture.release()
        if not args.no_window:
            cv2.destroyAllWindows()

    print(f"{len(gallery)} student(s) registered -> {args.gallery}\n")
    return len(gallery)


# --------------------------------------------------------------------------- #
# Step 3 -- the report
# --------------------------------------------------------------------------- #


def _aggregate_graph(graph_path: Path) -> tuple[dict, dict]:
    """Collapse a session's per-frame graphs into one summary graph.

    Args:
        graph_path: The session's graph JSONL.

    Returns:
        ``(positions, edges)`` -- mean on-screen position per person_id, and
        ``{(a, b): frames}`` counting how many frames each pair was linked in.
    """
    sums: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0, 0])
    pairs: Counter = Counter()
    for line in graph_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        graph = json.loads(line)
        node_person = {}
        for node in graph.get("nodes", []):
            pid = node.get("person_id")
            node_person[node["id"]] = pid
            bbox = (node.get("features") or {}).get("bbox")
            if pid is None or not bbox:
                continue
            entry = sums[pid]
            entry[0] += bbox[0] + bbox[2] / 2
            entry[1] += bbox[1] + bbox[3] / 2
            entry[2] += 1
        for edge in graph.get("edges", []):
            a, b = node_person.get(edge["source"]), node_person.get(edge["target"])
            if a is None or b is None or a == b:
                continue
            pairs[tuple(sorted((a, b)))] += 1
    positions = {
        pid: (x / n, y / n) for pid, (x, y, n) in sums.items() if n
    }
    return positions, dict(pairs)


def _svg_graph(positions: dict, edges: dict, names: dict, scores: dict) -> str:
    """Draw the session graph as inline SVG.

    Args:
        positions: Mean screen position per person_id.
        edges: ``{(a, b): frames}`` link strengths.
        names: ``{person_id: name}``.
        scores: ``{person_id: attention percentage or None}``.

    Returns:
        An SVG fragment. Node placement mirrors where people actually sat, so
        the picture matches the room rather than an arbitrary layout.
    """
    if not positions:
        return '<p class="muted">No graph to draw — nobody was recognised.</p>'

    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    W, H, PAD = 720, 380, 70

    def place(pid):
        px, py = positions[pid]
        fx = 0.5 if x1 == x0 else (px - x0) / (x1 - x0)
        fy = 0.5 if y1 == y0 else (py - y0) / (y1 - y0)
        return PAD + fx * (W - 2 * PAD), PAD + fy * (H - 2 * PAD)

    strongest = max(edges.values()) if edges else 1
    opening = (
        f'<svg viewBox="0 0 {W} {H}" class="graph" role="img" '
        f'aria-label="Student interaction graph">'
    )
    parts = [opening]
    for (a, b), n in sorted(edges.items(), key=lambda kv: kv[1]):
        if a not in positions or b not in positions:
            continue
        ax, ay = place(a)
        bx, by = place(b)
        width = 1 + 5 * (n / strongest)
        parts.append(
            f'<line x1="{ax:.0f}" y1="{ay:.0f}" x2="{bx:.0f}" y2="{by:.0f}" '
            f'stroke="var(--edge)" stroke-width="{width:.1f}" stroke-linecap="round"/>'
        )
    for pid in positions:
        cx, cy = place(pid)
        pct = scores.get(pid)
        fill = "var(--unknown)" if pct is None else (
            "var(--good)" if pct >= 70 else "var(--mid)" if pct >= 40 else "var(--low)"
        )
        label = names.get(pid, f"#{pid}")
        parts.append(
            f'<circle cx="{cx:.0f}" cy="{cy:.0f}" r="26" fill="{fill}" '
            f'stroke="var(--ring)" stroke-width="2"/>'
            f'<text x="{cx:.0f}" y="{cy + 5:.0f}" class="nodeval">'
            f'{"—" if pct is None else f"{pct:.0f}%"}</text>'
            f'<text x="{cx:.0f}" y="{cy + 46:.0f}" class="nodename">{label}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _bar(counts: dict) -> str:
    """A one-line stacked bar of label counts, widest first."""
    total = sum(counts.values())
    if not total:
        return '<span class="muted">no data</span>'
    order = sorted(counts.items(), key=lambda kv: -kv[1])
    cells = "".join(
        f'<span class="seg seg-{k}" style="width:{100 * v / total:.1f}%" '
        f'title="{k}: {v}"></span>' for k, v in order
    )
    legend = " ".join(f'<span class="key key-{k}">{k} {v}</span>' for k, v in order)
    return f'<div class="bar">{cells}</div><div class="legend">{legend}</div>'


def build_report(args) -> Path:
    """Render the session report and return the file written.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Path to the written HTML report.

    Raises:
        SystemExit: If the session outputs are missing.
    """
    from backend.enrollment import EnrolledGallery

    out = Path(args.out)
    profiles_path = out / "live_profiles.json"
    graph_path = out / "live_graph.jsonl"
    if not profiles_path.exists() or not graph_path.exists():
        raise SystemExit(
            f"No session found in {out}. Run without --report-only first."
        )

    names = {p.person_id: p.name for p in EnrolledGallery.load(args.gallery).people}
    profiles = json.loads(profiles_path.read_text(encoding="utf-8"))
    students = [p for p in profiles if p.get("is_student")]
    positions, edges = _aggregate_graph(graph_path)
    scores = {
        p["person_id"]: p["concentration"]["behavioral_proxy_pct"] for p in students
    }
    # Draw only people who became students. Unidentified detections are counted
    # in the header, but putting them on the graph would show the room as more
    # crowded than the roster and invite reading a circle as a person we can
    # actually name.
    keep = set(scores)
    positions = {pid: xy for pid, xy in positions.items() if pid in keep}
    edges = {
        pair: n for pair, n in edges.items() if pair[0] in keep and pair[1] in keep
    }

    cards = []
    for p in sorted(students, key=lambda p: p["person_id"]):
        pid = p["person_id"]
        pct = p["concentration"]["behavioral_proxy_pct"]
        lean = p["posture"]["mean_vertical_lean"]
        cards.append(f"""
        <article class="card">
          <header>
            <h3>{names.get(pid, f"Student #{pid}")}</h3>
            <span class="id">#{pid}</span>
          </header>
          <div class="score">{"—" if pct is None else f"{pct:.0f}%"}
            <small>attention proxy</small></div>
          <dl>
            <dt>Seen</dt><dd>{p["frames_seen"]} frames</dd>
            <dt>Where they looked</dt><dd>{_bar(p["attention"]["counts"])}</dd>
            <dt>Emotion</dt><dd>{_bar(p["expression"]["counts"])}</dd>
            <dt>Posture</dt><dd>{"no keypoints" if lean is None else
                f"mean lean {lean:+.3f} · {p['posture']['frames_with_keypoints']} frames with body keypoints"}</dd>
          </dl>
        </article>""")

    unknown = sum(1 for p in profiles if not p.get("is_student"))
    caveat = students[0]["concentration"]["caveat"] if students else ""

    html = f"""<title>Classroom Session Report</title>
<style>
:root {{
  --bg:#fbfaf9; --panel:#fff; --ink:#1c1a19; --muted:#6b6663; --line:#e6e2df;
  --good:#2f855a; --mid:#b7791f; --low:#c05621; --unknown:#8b8683;
  --edge:#c8c2be; --ring:#fff;
  --teacher:#2f855a; --left:#4c6ef5; --right:#7048e8; --down:#b7791f; --back:#868e96;
  --neutral:#868e96; --happy:#2f855a; --sad:#4c6ef5; --uncertain:#adb5bd;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --bg:#151313; --panel:#1e1c1b; --ink:#f0ece9; --muted:#a29c98; --line:#332f2d;
    --edge:#4a4441; --ring:#1e1c1b; --unknown:#6b6663;
  }}
}}
:root[data-theme="dark"] {{
  --bg:#151313; --panel:#1e1c1b; --ink:#f0ece9; --muted:#a29c98; --line:#332f2d;
  --edge:#4a4441; --ring:#1e1c1b; --unknown:#6b6663;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:15px/1.55 -apple-system,
  BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; padding:32px 20px 64px; }}
.wrap {{ max-width:980px; margin:0 auto; }}
h1 {{ font-size:1.6rem; margin:0 0 4px; letter-spacing:-.01em; }}
.sub {{ color:var(--muted); margin:0 0 28px; }}
h2 {{ font-size:1.05rem; margin:36px 0 14px; letter-spacing:.02em;
  text-transform:uppercase; color:var(--muted); }}
.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:12px;
  padding:20px; overflow-x:auto; }}
.graph {{ width:100%; height:auto; min-width:520px; display:block; }}
.nodeval {{ font-size:13px; font-weight:600; fill:#fff; text-anchor:middle; }}
.nodename {{ font-size:12px; fill:var(--ink); text-anchor:middle; }}
.grid {{ display:grid; gap:16px; grid-template-columns:repeat(auto-fill,minmax(300px,1fr)); }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:18px; }}
.card header {{ display:flex; align-items:baseline; justify-content:space-between; }}
.card h3 {{ margin:0; font-size:1.05rem; }}
.id {{ color:var(--muted); font-size:.85rem; }}
.score {{ font-size:2.1rem; font-weight:600; margin:10px 0 14px; letter-spacing:-.02em; }}
.score small {{ display:block; font-size:.72rem; font-weight:400; color:var(--muted);
  text-transform:uppercase; letter-spacing:.06em; }}
dl {{ margin:0; }}
dt {{ font-size:.72rem; text-transform:uppercase; letter-spacing:.06em;
  color:var(--muted); margin-top:12px; }}
dd {{ margin:4px 0 0; }}
.bar {{ display:flex; height:9px; border-radius:5px; overflow:hidden; background:var(--line); }}
.seg {{ display:block; }}
.seg-teacher{{background:var(--teacher)}} .seg-left{{background:var(--left)}}
.seg-right{{background:var(--right)}} .seg-down{{background:var(--down)}}
.seg-back{{background:var(--back)}} .seg-neutral{{background:var(--neutral)}}
.seg-happy{{background:var(--happy)}} .seg-sad{{background:var(--sad)}}
.seg-uncertain{{background:var(--uncertain)}}
.legend {{ margin-top:6px; font-size:.78rem; color:var(--muted); }}
.key {{ margin-right:10px; white-space:nowrap; }}
.muted {{ color:var(--muted); }}
.note {{ border-left:3px solid var(--line); padding:2px 0 2px 14px; color:var(--muted);
  font-size:.88rem; margin-top:28px; }}
</style>
<div class="wrap">
  <h1>Classroom session</h1>
  <p class="sub">{len(students)} student(s) recognised · {unknown} unidentified
     detection(s) not counted</p>

  <h2>Interaction graph</h2>
  <div class="panel">
    {_svg_graph(positions, edges, names, scores)}
    <p class="muted" style="margin:14px 0 0;font-size:.85rem">
      Each circle is a registered student, placed where they actually sat.
      Colour and number are the attention proxy. Line thickness is how many
      frames two students were linked in the scene graph.</p>
  </div>

  <h2>Per student</h2>
  <div class="grid">{"".join(cards) or '<p class="muted">Nobody was recognised.</p>'}</div>

  <p class="note">{caveat}</p>
</div>
"""
    report = out / "report.html"
    report.write_text(html, encoding="utf-8")
    return report


# --------------------------------------------------------------------------- #


def main() -> int:
    """Run the guided three-step flow.

    Returns:
        A process exit code.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--gallery", default="outputs/enrollment/gallery.json")
    parser.add_argument("--out", default="outputs/live")
    parser.add_argument("--seconds", type=float, default=None,
                        help="Stop recording automatically after this long.")
    parser.add_argument("--min-face-px", type=int, default=60,
                        help="Smallest face accepted during registration.")
    parser.add_argument("--skip-register", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--no-window", action="store_true")
    parser.add_argument("--yaw-reference", type=float, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    Path(args.out).mkdir(parents=True, exist_ok=True)

    if not args.report_only:
        if not args.skip_register and register_students(args) == 0:
            print("Nobody registered, so nothing could be stored under a student id.")
            return 1

        print("STEP 2 of 3 -- RECORD THE SESSION")
        from tools.live_session import run as run_session

        code = run_session(Namespace(
            camera=args.camera, gallery=args.gallery, out=args.out,
            seconds=args.seconds, no_window=args.no_window,
            yaw_reference=args.yaw_reference, verbose=args.verbose,
        ))
        if code != 0:
            return code

    print("\nSTEP 3 of 3 -- REPORT")
    report = build_report(args)
    print(f"  wrote {report}")
    if not args.no_window:
        webbrowser.open(report.resolve().as_uri())
        print("  opened in your browser")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
