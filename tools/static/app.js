// The server requires a token on every route. It is printed in the terminal
// at startup and pasted here once, then kept in sessionStorage for this tab.
//
// It is deliberately NOT served with the page. An earlier version put it in a
// meta tag, which meant `curl http://host:8000/` handed it to anyone who
// asked -- the token protected nothing against exactly the networked attacker
// it exists to stop.
const CG_TOKEN_KEY = "classgraph-token";

function cgToken() {
  let token = sessionStorage.getItem(CG_TOKEN_KEY);
  if (!token) {
    token = window.prompt(
      "Access token\n\nClassGraph prints one in the terminal when it " +
      "starts.\nPaste it here to use this dashboard.");
    if (token) sessionStorage.setItem(CG_TOKEN_KEY, token.trim());
  }
  return (token || "").trim();
}

/** Forget a rejected token, so a wrong paste can be corrected. */
function cgForgetToken() {
  sessionStorage.removeItem(CG_TOKEN_KEY);
}

/** Why a request failed, in words a person can act on.
 *
 * The server answers a rejected token with {error, hint} and a rejected
 * request with {detail}. Reading only one of them showed "undefined" in an
 * alert box, which tells the operator nothing about what to do next.
 */
async function cgReason(response) {
  if (response.status === 401) {
    return "That access token was not accepted.\n\n" +
           "ClassGraph prints the current one in the terminal each time it " +
           "starts.\nReload the page to enter it again.";
  }
  try {
    const body = await response.json();
    return body.detail || body.error || `Request failed (${response.status}).`;
  } catch {
    return `Request failed (${response.status}).`;
  }
}

/** fetch(), with the token attached. Used in place of fetch() throughout. */
async function cgFetch(url, options = {}) {
  const headers = { ...(options.headers || {}),
                    "X-ClassGraph-Token": cgToken() };
  const response = await fetch(url, { ...options, headers });
  if (response.status === 401) {
    // A stored token that no longer works is worse than none: the server
    // restarts generate a new one, and without this the page would keep
    // retrying the stale one forever.
    cgForgetToken();
  }
  return response;
}

/* ClassGraph dashboard: roster, live overlay and enrollment.
 *
 * The page owns presentation only. Every number comes from the WebSocket
 * payload the server computes, so the browser never re-derives a metric and
 * cannot disagree with the report.
 */
"use strict";

const GAZE = {
  teacher: "#4ea36b", left: "#d99038", right: "#a862c0",
  down: "#4a86c9", back: "#7d848f",
};
const ACTION_LABEL = {
  on_phone: "on phone", studying: "reading / writing", on_laptop: "on laptop",
  eyes_closed: "eyes closed", looking_away: "looking away",
  head_down: "head down", attentive: "attentive", unknown: "no face read",
};
const ACTION_COLOUR = {
  on_phone: "#c9553d", eyes_closed: "#c9553d", looking_away: "#d99038",
  studying: "#4a86c9", on_laptop: "#4a86c9", head_down: "#7d848f",
  attentive: "#4ea36b", unknown: "#4a5058",
};

//: Mirrors REPORTABLE_OBJECT_CONF in tools/server.py.
const REPORTABLE_OBJECT_CONF = 0.5;

const LEAN_LABEL = {
  slouching: "slouching", leaning_forward: "leaning forward", upright: "upright",
};

/** The colour band for a 1-10 attention score. Shared by both models, so a
 *  5 from one is the same colour as a 5 from the other. */
function scoreColour(score) {
  if (score == null) return "var(--muted)";
  return score >= 7 ? "var(--accent)" : score >= 4 ? "var(--warn)" : "var(--bad)";
}

const $ = (id) => document.getElementById(id);
const feed = $("feed");
const overlay = $("overlay");
const ctx = overlay.getContext("2d");
let running = false;
let socket = null;
let enrollStream = null;

/* ---------------------------------------------------------------- roster */

async function loadRoster() {
  const { students } = await (await cgFetch("/api/students")).json();
  const box = $("roster");
  if (!students.length) {
    box.innerHTML = '<p class="empty">Nobody registered yet.</p>';
    return;
  }
  box.innerHTML = students
    .map((s) => `<div class="student" data-id="${s.id}">
        <div class="row"><span class="nm">${escape(s.name)}</span>
          <span class="pct">—</span></div>
        <div class="act">#${s.id} · ${s.shots} shots</div>
        <div class="bar"></div>
        <svg class="spark" viewBox="0 0 120 22" preserveAspectRatio="none"></svg>
      </div>`)
    .join("");
}

function paintRoster(students) {
  for (const s of students) {
    const el = document.querySelector(`.student[data-id="${s.id}"]`);
    if (!el) continue;
    el.classList.toggle("present", s.present);
    // The learned model's 1-10 score. A dash until it has seen a full
    // window of this student: inventing a number for someone who just walked
    // in would contradict how the rest of this system treats thin evidence.
    const pct = el.querySelector(".pct");
    const score = s.attention_score;
    const mamba = s.attention_score_mamba;
    // Two models, side by side. The aggregate model is the one that ships;
    // the Mamba sequence model scores the same 15-second window from the
    // frames in order rather than from their averages. Out of fold on 179
    // labelled windows they came out at 0.609 and 0.592 -- closer together
    // than 179 labels can separate -- so neither is presented as the answer
    // and a disagreement between them is information the viewer should see.
    pct.innerHTML = score == null
      ? '<span class="sc sc-none">—</span>'
      : `<span class="sc" style="color:${scoreColour(score)}">${score}</span>`
        + '<span class="sc-sep">/</span>'
        + (mamba == null
            ? '<span class="sc sc-none" title="Mamba: no full window yet">'
              + "–</span>"
            : `<span class="sc sc-mamba" style="color:${scoreColour(mamba)}">`
              + `${mamba}</span>`);
    pct.title = score == null
      ? "waiting for a full 15s window"
      : `aggregate model ${score}/10`
        + (mamba == null ? "" : `, Mamba sequence model ${mamba}/10`)
        + ` · ${s.attention_label || "no verdict"}`
        + ` · ${s.attention_reason || ""}`;

    const act = s.action ? ACTION_LABEL[s.action] || s.action : "not seen";
    // Posture is shown next to the action, not instead of it. A student can
    // be on their phone and slouching at the same time, and the old display
    // could only ever say one of those because lean competed with every
    // object rule for a single slot and lost.
    const lean = s.lean && s.lean !== "upright"
      ? ` <span class="lean">${escape(LEAN_LABEL[s.lean] || s.lean)}</span>`
      : "";
    el.querySelector(".act").innerHTML =
      `<span style="color:${ACTION_COLOUR[s.action] || "var(--muted)"}">●</span> ${escape(act)}` +
      lean +
      (s.evidence ? ` <span style="opacity:.65">· ${escape(s.evidence)}</span>` : "");

    const total = Object.values(s.gaze).reduce((a, b) => a + b, 0);
    el.querySelector(".bar").innerHTML = total
      ? Object.entries(s.gaze).sort((a, b) => b[1] - a[1])
          .map(([k, v]) => `<span style="width:${(100 * v) / total}%;background:${
            GAZE[k] || "#555"}" title="${k}: ${v}"></span>`).join("")
      : "";

    const svg = el.querySelector(".spark");
    const pts = s.recent || [];
    if (pts.length > 1) {
      const step = 120 / (pts.length - 1);
      const d = pts.map((v, i) => `${i ? "L" : "M"}${(i * step).toFixed(1)},${
        (20 - v * 18).toFixed(1)}`).join("");
      svg.innerHTML = `<path d="${d}" fill="none" stroke="${
        s.present ? "#4ea36b" : "#4a5058"}" stroke-width="1.6"/>`;
    }
  }
}

function paintPairs(pairs, students) {
  const name = Object.fromEntries(students.map((s) => [s.id, s.name]));
  $("pairs").innerHTML = pairs && pairs.length
    ? `<div class="chips">${pairs.map((p) =>
        `<span class="chip">${escape(name[p.a] || p.a)} + ${
          escape(name[p.b] || p.b)} <b>${p.frames}</b></span>`).join("")}</div>`
    : '<p class="empty">Needs two students doing the same thing.</p>';
}

function paintRoom(L) {
  const box = $("room"), kind = $("roomKind"), why = $("roomWhy");
  if (!L) return;
  box.classList.toggle("group", L.kind === "group");
  box.classList.toggle("lecture", L.kind === "lecture");
  if (L.kind === "group") {
    kind.textContent = "Group work";
    why.innerHTML = `They face <b>each other</b> — the focus falls among them `
      + `(ratio ${L.ratio}). Engagement here means engagement with the group, `
      + `so turning away from the front is correct behaviour.`;
  } else if (L.kind === "lecture") {
    kind.textContent = "Lecture";
    why.innerHTML = `They share a focus <b>outside the seating</b> `
      + `(ratio ${L.ratio}). Engagement means orientation toward it.`;
  } else {
    kind.textContent = "Layout unknown";
    why.textContent = L.n < 3
      ? `Needs 3 people to fit the facing rays — ${L.n} in frame. `
        + `The second score stays blank rather than being guessed.`
      : "The facing directions do not converge, so no shared focus can be claimed.";
  }
}

/* ---------------------------------------------------------------- overlay */

function paintOverlay(boxes) {
  const w = feed.naturalWidth, h = feed.naturalHeight;
  if (!w || !h) return;
  // Match the letterboxing that object-fit:contain applies to the <img>.
  const box = feed.getBoundingClientRect();
  overlay.width = box.width; overlay.height = box.height;
  const scale = Math.min(box.width / w, box.height / h);
  const ox = (box.width - w * scale) / 2, oy = (box.height - h * scale) / 2;

  ctx.clearRect(0, 0, overlay.width, overlay.height);
  const X = (v) => ox + v * scale, Y = (v) => oy + v * scale;

  // The geometry the layout verdict is built from: one ray per student along
  // their shoulder direction, and the point those rays converge on.
  if (window._layout && window._layout.focus) {
    const L = window._layout, fx = X(L.focus[0]), fy = Y(L.focus[1]);
    const hue = L.kind === "group" ? "#4ea36b" : "#4a86c9";
    ctx.save();
    ctx.strokeStyle = hue; ctx.globalAlpha = .38; ctx.lineWidth = 1.5;
    for (const b of boxes) {
      if (!b.facing) continue;
      const cx = X(b.bbox[0] + b.bbox[2] / 2), cy = Y(b.bbox[1] + b.bbox[3] / 2);
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(fx, fy); ctx.stroke();
    }
    ctx.globalAlpha = 1; ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.arc(fx, fy, 13, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(fx - 20, fy); ctx.lineTo(fx + 20, fy);
    ctx.moveTo(fx, fy - 20); ctx.lineTo(fx, fy + 20); ctx.stroke();
    ctx.fillStyle = hue; ctx.font = "600 12px 'Segoe UI',sans-serif";
    ctx.fillText("room focus", fx + 26, fy + 4);
    ctx.restore();
  }
  ctx.lineWidth = 2;
  ctx.font = "600 13px 'Segoe UI',sans-serif";
  for (const b of boxes) {
    const [x, y, bw, bh] = b.bbox;
    // Three states, not two. A registered student has a name. An unregistered
    // one still has a stable person_id -- the tracker is following them and
    // every score in the report is keyed on it -- and calling that "unknown"
    // throws away the identity the system did establish. Only a person the
    // resolver could not place at all is unknown.
    const tracked = b.person_id && b.person_id > 0;
    const colour = b.name ? (ACTION_COLOUR[b.action] || "#4ea36b")
                  : tracked ? "#c9a14d"
                  : "#c9553d";
    ctx.strokeStyle = colour;
    ctx.strokeRect(ox + x * scale, oy + y * scale, bw * scale, bh * scale);
    const who = b.name || (tracked ? `Student ${b.person_id}` : "unidentified");
    let what = ACTION_LABEL[b.action] || "";
    if (b.lean && b.lean !== "upright") {
      const posture = LEAN_LABEL[b.lean] || b.lean;
      what = what ? `${what}, ${posture}` : posture;
    }
    const label = b.name || tracked
      ? (what ? `${who} · ${what}` : who)
      : (b.face_px ? `unidentified · face ${b.face_px}px` : "unidentified");
    const tw = ctx.measureText(label).width + 14;
    const ty = Math.max(0, oy + y * scale - 22);
    ctx.fillStyle = colour;
    ctx.fillRect(ox + x * scale, ty, tw, 20);
    ctx.fillStyle = "#0d0f11";
    ctx.fillText(label, ox + x * scale + 7, ty + 14);
  }
}

/* ---------------------------------------------------------------- socket */

function connect() {
  socket = new WebSocket(
    `ws://${location.host}/ws?token=${encodeURIComponent(cgToken())}`);
  socket.onmessage = (event) => {
    const d = JSON.parse(event.data);
    if (d.idle) return;
    if (d.image) {
      feed.hidden = false;
      $("ph").hidden = true;
      feed.src = `data:image/jpeg;base64,${d.image}`;
    }
    // Defaulted, because a sender that omits a field must not freeze the
    // display. Reading d.objects.length on a payload without it threw, the
    // handler aborted before repainting, and the previous session's boxes
    // stayed on screen over the new video -- still carrying its names.
    const students = d.students || [];
    const objects = d.objects || [];
    const pairs = d.pairs || [];

    $("tInFrame").textContent = d.in_frame ?? "";
    $("tKnown").textContent = students.filter((s) => s.present).length;
    $("tFrames").textContent = d.frame ?? "";
    $("tFps").textContent = d.fps ?? "";
    $("hdr").textContent = d.seconds == null
      ? "" : `${d.seconds}s · ${d.fps ?? "?"} fps`;
    paintRoom(d.layout);
    const hint = $("hint");
    if (d.hint) { hint.textContent = d.hint; hint.hidden = false; }
    else hint.hidden = true;
    // The detector runs at a deliberately low threshold, so weak detections
    // exist by design. They used to arrive here as bare class names and were
    // drawn identically to strong ones -- a bottle the detector was 28% sure
    // of looked exactly like a laptop it was 91% sure of. The number is shown
    // and anything under the reportable confidence is dimmed and marked.
    $("objects").innerHTML = objects.length
      ? objects.map((o) => {
          const weak = o.conf < REPORTABLE_OBJECT_CONF;
          return `<span class="chip${weak ? " weak" : ""}" title="${
            weak ? "below the reportable confidence; shown but not counted"
                 : "detector confidence"}">${escape(o.cls)}` +
            `<b>${Math.round(o.conf * 100)}%</b></span>`;
        }).join("")
      : '<span class="empty">none</span>';
    // The graph draws only what the detector was reasonably sure of; an
    // object seen once at 0.26 is not a thing that was in the room.
    const strong = objects.filter((o) => o.conf >= REPORTABLE_OBJECT_CONF)
                          .map((o) => o.cls);
    window.lastStudents = students;
    window.lastObjects = strong;
    window.lastPairs = pairs;
    paintRoster(students);
    paintPairs(pairs, students);
    paintLiveGraph(students, strong, pairs);
    window._layout = d.layout;
    requestAnimationFrame(() => paintOverlay(d.boxes));
  };
  socket.onclose = () => setTimeout(connect, 1200);
}

let currentGraphMode = "room";

/** Switch which measurement the scene graph is showing. */
function setGraphMode(mode) {
  currentGraphMode = mode;
  document.querySelectorAll(".graph-tab").forEach(tab => {
    tab.classList.toggle("active", tab.dataset.mode === mode);
  });
  // Named for what each one measures. "Gaze Radar", "Energy Flow" and
  // "Topology Network" described the drawing rather than the quantity, and
  // two of the three were not drawing a quantity at all.
  const labels = {
    room: "Seating and room focus",
    objects: "Who was holding what",
    scores: "Attention: both models",
    heatmap: "Shared actions between students"
  };
  $("graphModeLabel").textContent = labels[mode] || mode;
  // Repaint at once. Without this the tab highlights and the picture does
  // not change until the next socket frame, which on a finished run never
  // arrives -- so every tab looked broken after a video had been analysed.
  paintLiveGraph(window.lastStudents || [], window.lastObjects || [],
                 window.lastPairs || []);
}

function toggleFullscreenGraph() {
  const card = $("graphCard");
  if (card) {
    card.classList.toggle("fullscreen-graph");
  }
}

/* The four graph modes.
 *
 * Each of these draws a measurement. That is a change: two of them used to
 * draw every student joined to every object, which is a complete bipartite
 * graph and carries no information at all -- it put a book beside a student
 * who never touched one. A third drew a "teacher podium" at a fixed place on
 * the canvas with a ray from each student to it, although nothing had
 * measured where the front of the room was or which way anyone faced.
 *
 * The pipeline already decides object ownership exclusively, already measures
 * where the focus of the room is, and already measures where each student
 * sits. The graphs were not short of data; they were not asking for it.
 */

/** Draw the scene graph in whichever mode is selected.
 *
 * `present` is the students the pipeline currently sees; an empty list draws
 * an empty graph that says it is empty, rather than the four invented
 * students this used to fall back on.
 */
function paintLiveGraph(students, detectedObjects, pairs) {
  const svg = $("graphCanvas");
  if (!svg) return;
  const present = (students || []).filter(s => s && s.present);
  const objects = Array.from(new Set(detectedObjects || []));
  const pairList = pairs || [];

  if (!present.length) {
    emptyGraph(svg, "No registered student is in frame. "
      + "The graph fills in as people are recognised.");
    return;
  }

  const W = 900, H = 420;
  if (currentGraphMode === "heatmap") {
    renderInteractionMatrix(svg, present, pairList, W, H);
  } else if (currentGraphMode === "scores") {
    renderScoreCompare(svg, present, objects, pairList, W, H);
  } else if (currentGraphMode === "objects") {
    renderObjectLinks(svg, present, objects, pairList, W, H);
  } else {
    renderRoomMap(svg, present, objects, pairList, W, H);
  }
}

/** A student's display name, falling back to their tracked number. */
function who(s) {
  return s.name || `Student ${s.id}`;
}

/** An honest empty state, in place of the placeholder dataset.
 *
 * When no student was present the graph used to draw four invented students
 * -- Aarav, Bhavya, Chetan, Divya -- holding an invented laptop, book and
 * phone, in the same colours and shapes as a real measurement and with
 * nothing on screen marking it as fake. That is how objects nobody could see
 * in the room ended up in the results: the graph was not reporting a
 * detection, it was reporting its own filler.
 *
 * A measurement display with a demo mode is one that sometimes lies, and the
 * viewer cannot tell which time.
 */
function emptyGraph(svg, message) {
  svg.innerHTML =
    '<text x="450" y="200" text-anchor="middle" class="graph-empty-title">'
    + "Nothing to draw yet</text>"
    + '<text x="450" y="226" text-anchor="middle" class="graph-empty-note">'
    + escape(message) + "</text>";
}

/** Shared SVG preamble: a legend strip along the top of the canvas. */
function legend(items) {
  let x = 24;
  let out = "";
  for (const item of items) {
    out += `<circle cx="${x}" cy="18" r="5" fill="${item[0]}"/>`
         + `<text x="${x + 11}" y="22" class="graph-legend">`
         + `${escape(item[1])}</text>`;
    x += 26 + item[1].length * 6.6;
  }
  return out;
}

/** Where the students are, and what the room is oriented toward.
 *
 * Positions are the measured bounding-box centres, scaled into the canvas,
 * not a four-column grid in roster order. The focus is the one the pipeline
 * fitted from the students' shoulder directions; when it could not fit one,
 * this says so rather than drawing a podium that was never measured.
 */
function renderRoomMap(svg, present, objects, pairs, W, H) {
  const L = window._layout;
  const pts = present.filter(s => s.cx != null && s.cy != null);
  if (!pts.length) {
    emptyGraph(svg, "No student position has been measured yet.");
    return;
  }

  // Scale the frame's pixel coordinates into the canvas, with one factor for
  // both axes so the seating keeps its shape instead of being stretched.
  const xs = pts.map(s => s.cx).concat(L && L.focus ? [L.focus[0]] : []);
  const ys = pts.map(s => s.cy).concat(L && L.focus ? [L.focus[1]] : []);
  const pad = 70;
  const minX = Math.min.apply(null, xs), maxX = Math.max.apply(null, xs);
  const minY = Math.min.apply(null, ys), maxY = Math.max.apply(null, ys);
  const k = Math.min((W - 2 * pad) / Math.max(maxX - minX, 1),
                     (H - 2 * pad - 30) / Math.max(maxY - minY, 1));
  const X = (v) => pad + (v - minX) * k;
  const Y = (v) => pad + 30 + (v - minY) * k;

  let html = legend([["#4ea36b", "facing the focus"],
                     ["#d99038", "facing away"],
                     ["#7d848f", "shoulders not read"]]);

  if (L && L.focus) {
    const fx = X(L.focus[0]), fy = Y(L.focus[1]);
    const hue = L.kind === "group" ? "#4ea36b" : "#4a86c9";
    for (const s of pts) {
      // The colour is the pipeline's own `oriented` verdict for that student,
      // not a guess from where they land on the canvas. Grey means the
      // shoulders could not be read, which is not the same as facing away.
      const tone = s.oriented === true ? "#4ea36b"
                 : s.oriented === false ? "#d99038" : "#7d848f";
      html += `<line x1="${X(s.cx)}" y1="${Y(s.cy)}" x2="${fx}" y2="${fy}" `
            + `stroke="${tone}" stroke-opacity="0.45" stroke-width="2"/>`;
    }
    html += `<circle cx="${fx}" cy="${fy}" r="13" fill="none" `
          + `stroke="${hue}" stroke-width="2.5"/>`
          + `<line x1="${fx - 19}" y1="${fy}" x2="${fx + 19}" y2="${fy}" `
          + `stroke="${hue}" stroke-width="2.5"/>`
          + `<line x1="${fx}" y1="${fy - 19}" x2="${fx}" y2="${fy + 19}" `
          + `stroke="${hue}" stroke-width="2.5"/>`
          + `<text x="${fx + 26}" y="${fy + 4}" class="graph-note">`
          + `room focus (${escape(L.kind)})</text>`;
  } else {
    const why = L && L.n < 3
      ? `Needs 3 people to fit the facing rays and ${L.n} are in frame.`
      : "The facing directions do not converge on a shared point.";
    html += `<text x="${W / 2}" y="${H - 18}" text-anchor="middle" `
          + `class="graph-note">No room focus could be measured. `
          + `${escape(why)}</text>`;
  }

  for (const s of pts) {
    const colour = ACTION_COLOUR[s.action] || "#7d848f";
    html += `<g transform="translate(${X(s.cx)},${Y(s.cy)})">`
          + `<circle r="15" fill="${colour}" stroke="#0d0f11" `
          + `stroke-width="2.5"/>`
          + `<text y="4" text-anchor="middle" class="graph-id">${s.id}</text>`
          + `<text y="31" text-anchor="middle" class="graph-name">`
          + `${escape(who(s))}</text></g>`;
  }
  svg.innerHTML = html;
}

/** Who was holding what, and how sure the detector was.
 *
 * One link per student, to the single object the pipeline attributed to THEM.
 * Ownership is exclusive -- backend.actions.assign_objects gives each object
 * to the one student whose box contains the largest share of it, because on a
 * real lecture 35% of detected phones overlapped more than one student box
 * and crediting all of them inflated "on phone" to 30% of every action.
 */
function renderObjectLinks(svg, present, objects, pairs, W, H) {
  const held = present.filter(s => s.object);
  if (!held.length) {
    emptyGraph(svg, present.length === 1
      ? "No object was attributed to this student in this frame."
      : "No object was attributed to any student in this frame.");
    return;
  }

  const classes = Array.from(new Set(held.map(s => s.object))).sort();
  const px = 250, ox = 640;
  const yOf = (i, n) => 70 + (H - 120) * ((i + 0.5) / Math.max(n, 1));
  const py = {}, oy = {};
  held.forEach((s, i) => { py[s.id] = yOf(i, held.length); });
  classes.forEach((c, i) => { oy[c] = yOf(i, classes.length); });

  let html = legend([["#4a86c9", "detector confident"],
                     ["#c9a14d", "weak detection"]]);

  for (const s of held) {
    const conf = s.object_conf == null ? 0 : s.object_conf;
    const weak = conf < REPORTABLE_OBJECT_CONF;
    const mid = (px + ox) / 2;
    const midY = (py[s.id] + oy[s.object]) / 2;
    html += `<path d="M${px},${py[s.id]} Q${mid},${midY} ${ox},`
          + `${oy[s.object]}" fill="none" `
          + `stroke="${weak ? "#c9a14d" : "#4a86c9"}" stroke-width="2.5" `
          + `${weak ? 'stroke-dasharray="6 4" ' : ""}stroke-opacity="0.75"/>`
          + `<text x="${mid}" y="${midY - 6}" text-anchor="middle" `
          + `class="graph-note">${Math.round(conf * 100)}%</text>`;
  }

  for (const s of held) {
    html += `<g transform="translate(${px},${py[s.id]})">`
          + `<circle r="15" fill="${ACTION_COLOUR[s.action] || "#7d848f"}" `
          + `stroke="#0d0f11" stroke-width="2.5"/>`
          + `<text y="4" text-anchor="middle" class="graph-id">${s.id}</text>`
          + `<text x="-24" y="4" text-anchor="end" class="graph-name">`
          + `${escape(who(s))}</text></g>`;
  }
  for (const c of classes) {
    html += `<g transform="translate(${ox},${oy[c] - 15})">`
          + `<rect width="160" height="30" rx="8" `
          + `fill="rgba(74,134,201,.14)" stroke="#4a86c9" `
          + `stroke-opacity=".5"/>`
          + `<text x="14" y="20" class="graph-name">${escape(c)}</text></g>`;
  }

  const quiet = present.length - held.length;
  if (quiet > 0) {
    html += `<text x="${W / 2}" y="${H - 14}" text-anchor="middle" `
          + `class="graph-note">${quiet} student`
          + `${quiet === 1 ? " was" : "s were"} holding nothing the detector `
          + "recognised.</text>";
  }
  svg.innerHTML = html;
}

/** Both models' attention scores for each student, side by side.
 *
 * The shipping score comes from a logistic model over 21 aggregate features;
 * the second from the Mamba sequence model reading the same 15-second window
 * frame by frame in order. Out of fold on 179 hand-labelled windows, grouped
 * by student, they scored 0.609 and 0.592 -- a gap smaller than 179 labels
 * can resolve. So neither is shown as the answer, and where they disagree
 * about a student that disagreement is on screen rather than averaged away.
 */
function renderScoreCompare(svg, present, objects, pairs, W, H) {
  const scored = present.filter(
    s => s.attention_score != null || s.attention_score_mamba != null);
  if (!scored.length) {
    emptyGraph(svg, "No student has been visible for a full 15-second "
      + "window yet. Both models abstain until then.");
    return;
  }

  const left = 160, right = W - 90;
  const rowH = Math.min(54, (H - 120) / scored.length);
  const barW = (right - left) / 10;
  let html = legend([["#4a86c9", "aggregate model (shipping)"],
                     ["#a862c0", "Mamba sequence model"]]);

  // Gridlines at each second point of the 1-10 scale, so a bar is read off
  // an axis instead of estimated from its length.
  for (let v = 0; v <= 10; v += 2) {
    const x = left + v * barW;
    html += `<line x1="${x}" y1="56" x2="${x}" `
          + `y2="${56 + scored.length * rowH}" `
          + `stroke="rgba(255,255,255,.08)"/>`
          + `<text x="${x}" y="${72 + scored.length * rowH}" `
          + `text-anchor="middle" class="graph-note">${v}</text>`;
  }

  scored.forEach((s, i) => {
    const y = 62 + i * rowH;
    const h = Math.min(13, rowH / 3.4);
    html += `<text x="${left - 12}" y="${y + h + 3}" text-anchor="end" `
          + `class="graph-name">${escape(who(s))}</text>`;
    const bars = [["#4a86c9", s.attention_score, "aggregate"],
                  ["#a862c0", s.attention_score_mamba, "Mamba"]];
    bars.forEach((bar, k) => {
      const by = y + k * (h + 3);
      const score = bar[1];
      if (score == null) {
        html += `<text x="${left + 6}" y="${by + h - 1}" class="graph-note">`
              + `${bar[2]}: no full window yet</text>`;
        return;
      }
      html += `<rect x="${left}" y="${by}" width="${score * barW}" `
            + `height="${h}" rx="3" fill="${bar[0]}"/>`
            + `<text x="${left + score * barW + 7}" y="${by + h - 1}" `
            + `class="graph-value">${score}</text>`;
    });
  });

  const gaps = scored.filter(s => s.attention_score != null
    && s.attention_score_mamba != null
    && Math.abs(s.attention_score - s.attention_score_mamba) >= 3);
  html += `<text x="${W / 2}" y="${H - 10}" text-anchor="middle" `
        + `class="graph-note">`
        + (gaps.length
            ? escape("The two models differ by 3 or more points on "
                + `${gaps.length} of ${scored.length} students. Neither is `
                + "known to be the better one on this evidence.")
            : "The two models agree within 2 points on every student.")
        + "</text>";
  svg.innerHTML = html;
}

/** How often each pair of students was doing the same thing at the same time.
 *
 * "attentive" is excluded upstream, because linking everyone watching the
 * front would bury the pairs that mean something.
 */
function renderInteractionMatrix(svg, present, pairs, W, H) {
  if (present.length < 2) {
    emptyGraph(svg, "A shared-action matrix needs at least two students in "
      + "frame; there " + (present.length === 1 ? "is one." : "are none."));
    return;
  }

  const size = Math.min(320 / present.length, 55);
  const startX = W / 2 - (present.length * size) / 2;
  const startY = 90;
  const lookup = {};
  (pairs || []).forEach(p => {
    lookup[`${p.a}_${p.b}`] = p.frames;
    lookup[`${p.b}_${p.a}`] = p.frames;
  });
  const counts = Object.keys(lookup).map(k => lookup[k]);
  const top = Math.max(1, Math.max.apply(null, counts.concat([1])));

  let html = `<text x="${W / 2}" y="26" text-anchor="middle" `
    + `class="graph-note">Frames in which both students were doing the same `
    + "thing. Darker is more.</text>";

  present.forEach((s, i) => {
    html += `<text x="${startX + i * size + size / 2}" y="${startY - 12}" `
          + `text-anchor="middle" class="graph-name">${escape(who(s))}</text>`
          + `<text x="${startX - 12}" y="${startY + i * size + size / 2 + 4}" `
          + `text-anchor="end" class="graph-name">${escape(who(s))}</text>`;
  });

  present.forEach((a, r) => {
    present.forEach((b, c) => {
      const x = startX + c * size, y = startY + r * size;
      if (r === c) {
        html += `<rect x="${x + 2}" y="${y + 2}" width="${size - 4}" `
              + `height="${size - 4}" rx="6" fill="rgba(255,255,255,.03)"/>`;
        return;
      }
      const v = lookup[`${a.id}_${b.id}`] || 0;
      // One hue, light to dark, in proportion to the count. The three fixed
      // colours this replaces gave 1 frame and 19 frames the same shade and
      // put a hard edge at 20 that nothing in the data corresponds to.
      html += `<rect x="${x + 2}" y="${y + 2}" width="${size - 4}" `
            + `height="${size - 4}" rx="6" fill="rgba(74,134,201,`
            + `${(0.08 + 0.72 * (v / top)).toFixed(2)})" `
            + `stroke="rgba(255,255,255,.08)"/>`
            + `<text x="${x + size / 2}" y="${y + size / 2 + 4}" `
            + `text-anchor="middle" class="graph-value">${v || ""}</text>`;
    });
  });
  svg.innerHTML = html;
}

/* ---------------------------------------------------------------- session */

$("btnStart").onclick = async () => {
  // Cleared first, or the previous run's boxes, results and counters stay on
  // screen beside a new one's picture.
  resetDisplay();
  const res = await cgFetch("/api/session/start", { method: "POST" });
  if (!res.ok) { alert(await cgReason(res)); return; }
  running = true;
  $("btnStart").hidden = true;
  $("btnStop").hidden = false;
  $("btnEnroll").disabled = true;
  $("dot").classList.add("live");
};


/** Clear what the previous run left on screen.
 *
 * Without this a new run inherits the last one's boxes, results and stats,
 * and the only thing that changes is the picture behind them -- which is how
 * a classroom video came to show one box labelled with a webcam session's
 * name across six students.
 */
function resetDisplay() {
  $("hdr").textContent = "";
  $("resultsCard").hidden = true;
  $("results").innerHTML = "";
  $("btnReport").hidden = true;
  for (const id of ["tInFrame", "tKnown", "tFrames", "tFps"]) {
    $(id).textContent = "";
  }
  window.lastStudents = [];
  window.lastObjects = [];
  window.lastPairs = [];
  paintOverlay([]);
  paintLiveGraph([], [], []);
}

/* ------------------------------------------------------------ video file */

$("btnVideo").onclick = () => $("videoFile").click();

$("videoFile").onchange = async () => {
  const file = $("videoFile").files[0];
  if (!file) return;
  // Cleared immediately so choosing the same file twice still fires change.
  $("videoFile").value = "";

  const body = new FormData();
  body.append("file", file);
  // 1 frame in 5. A recording holds many near-identical frames per second,
  // and analysing all of them costs five times as long for no new
  // information -- the same rate the batch tool and the published figures use.
  body.append("sample_rate", "5");

  resetDisplay();
  $("btnVideo").disabled = true;
  $("btnStart").disabled = true;
  $("hdr").textContent = `Uploading ${file.name}...`;

  const res = await cgFetch("/api/video/analyse", { method: "POST", body });
  if (!res.ok) {
    $("hdr").textContent = "";
    $("btnVideo").disabled = false;
    $("btnStart").disabled = false;
    alert(await cgReason(res));
    return;
  }

  running = true;
  $("btnStart").hidden = true;
  $("btnStop").hidden = false;
  $("btnEnroll").disabled = true;
  $("dot").classList.add("live");
  trackVideoProgress(file.name);
};

/** Show how far through the recording we are, and finish by itself.
 *
 * A video ends on its own, unlike a camera, so nobody should have to watch
 * for that and press Stop. Polling rather than using the WebSocket because
 * the socket carries frames for display and goes quiet between them, which
 * would look identical to the analysis having stalled.
 */
async function trackVideoProgress(filename) {
  while (running) {
    await new Promise((r) => setTimeout(r, 1000));
    const res = await cgFetch("/api/status");
    if (!res.ok) break;
    const s = await res.json();

    if (s.running) {
      const pct = s.progress == null ? null : Math.round(s.progress * 100);
      $("hdr").textContent = pct == null
        ? `${filename} — ${s.analysed} frames analysed`
        : `${filename} — ${pct}% (${s.analysed} frames analysed)`;
      continue;
    }

    // Finished. stop() writes the profiles and builds the report, so it is
    // still the right way to end, even though the loop has already exited.
    const done = await cgFetch("/api/session/stop", { method: "POST" });
    const d = done.ok ? await done.json() : {};
    running = false;
    $("btnStart").hidden = false;
    $("btnStart").disabled = false;
    $("btnStop").hidden = true;
    $("btnVideo").disabled = false;
    $("btnEnroll").disabled = false;
    $("dot").classList.remove("live");
    $("hdr").textContent = s.stop_reason || `${d.frames || 0} frames analysed`;
    await showResults(d.report);
    return;
  }
}

$("btnStop").onclick = async () => {
  const res = await cgFetch("/api/session/stop", { method: "POST" });
  const d = await res.json();
  running = false;
  $("btnStart").hidden = false;
  $("btnStop").hidden = true;
  $("btnEnroll").disabled = false;
  $("dot").classList.remove("live");
  $("hdr").textContent = `${d.frames} frames captured`;
  await showResults(d.report);
  if (d.report) await openReport(d.report);
};

/** Show what a finished run found, and offer its report.
 *
 * The roster panel lists registered students, so an unenrolled recording had
 * nowhere to report: the run completed, the report was written, and the page
 * looked exactly as it had before it started.
 *
 * The report is behind a button rather than opened automatically. window.open
 * from a polling callback is not a user gesture, so browsers block it as a
 * popup -- which is silent, and indistinguishable from the analysis having
 * produced nothing.
 */
async function showResults(reportUrl) {
  const res = await cgFetch("/api/session/results");
  if (!res.ok) return;
  const { students } = await res.json();

  const card = $("resultsCard");
  const box = $("results");
  if (!students.length) {
    box.innerHTML = '<p class="empty">No student was resolved in this run.</p>';
  } else {
    box.innerHTML = students.map((s) => {
      const name = s.name || `Student ${s.id}`;
      // Both models, with the number of windows each rests on. They can and
      // do disagree -- on the recording this was built against one student
      // came out 1/10 from the aggregate model and 10/10 from Mamba -- and
      // that is a fact about how little 179 labels settle, not a fault to
      // hide by picking one.
      const pair = `<span class="sc" style="color:${
        scoreColour(s.attention)}">${s.attention == null ? "—"
        : s.attention}</span><span class="sc-sep">/</span><span class="sc" `
        + `style="color:${scoreColour(s.attention_mamba)}">${
        s.attention_mamba == null ? "–" : s.attention_mamba}</span>`;
      const onTask = s.on_task_pct == null
        ? "" : ` · ${Math.round(s.on_task_pct)}% on task`;
      const did = s.top_action ? ` · mostly ${ACTION_LABEL[s.top_action]
        || s.top_action}` : "";
      const windows = `${s.attention_windows || 0} and ${
        s.attention_windows_mamba || 0} windows scored`;
      return `<div class="student present">
          <div class="row"><b>${escape(name)}</b><span class="pct"
            title="aggregate model / Mamba sequence model, each out of 10 · ${
            windows}">${pair}</span></div>
          <div class="sub">${s.frames || 0} frames${onTask}${did}</div>
        </div>`;
    }).join("");
  }
  card.hidden = false;

  const button = $("btnReport");
  if (reportUrl) {
    button.hidden = false;
    button.onclick = () => openReport(reportUrl);
  } else {
    button.hidden = true;
  }
}

/** Open the finished report in a new tab.
 *
 * Fetched rather than navigated to. A browser navigation cannot carry the
 * X-ClassGraph-Token header, so window.open() on the report URL arrived
 * without one and the page showed a JSON 401 instead of the report.
 *
 * Putting the token in the query string would work and is the wrong trade:
 * it would then sit in browser history and in the logs of any reverse proxy
 * in front of this. A blob URL carries no token at all.
 */
async function openReport(url) {
  const res = await cgFetch(url);
  if (!res.ok) {
    alert(await cgReason(res));
    return;
  }
  const blob = new Blob([await res.text()], { type: "text/html" });
  const href = URL.createObjectURL(blob);
  window.open(href, "_blank");
  // Released once the new tab has had time to load it; revoking immediately
  // races the open and yields a blank tab.
  setTimeout(() => URL.revokeObjectURL(href), 60000);
}

/* ---------------------------------------------------------------- enroll */

const dlg = $("enrollDlg");

$("btnEnroll").onclick = async () => {
  $("enrollName").value = "";
  $("enrollMsg").textContent = "";
  $("enrollMsg").className = "msg";
  $("enrollBar").style.width = "0";
  dlg.showModal();
  try {
    enrollStream = await navigator.mediaDevices.getUserMedia({ video: { width: 640 } });
    $("enrollVideo").srcObject = enrollStream;
  } catch (e) {
    $("enrollMsg").className = "msg err";
    $("enrollMsg").textContent =
      "Cannot open the camera in the browser. If a session is running, stop it first.";
  }
};

function closeEnroll() {
  if (enrollStream) enrollStream.getTracks().forEach((t) => t.stop());
  enrollStream = null;
  dlg.close();
}
$("enrollCancel").onclick = closeEnroll;

$("enrollGo").onclick = async () => {
  const name = $("enrollName").value.trim();
  const msg = $("enrollMsg");
  if (!name) { msg.className = "msg err"; msg.textContent = "Enter a name first."; return; }
  if (!enrollStream) { msg.className = "msg err"; msg.textContent = "No camera."; return; }

  $("enrollGo").disabled = true;
  const video = $("enrollVideo");
  const canvas = document.createElement("canvas");
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  const c = canvas.getContext("2d");
  const shots = [];
  for (let i = 0; i < 8; i++) {
    c.drawImage(video, 0, 0, canvas.width, canvas.height);
    shots.push(canvas.toDataURL("image/jpeg", 0.9));
    $("enrollBar").style.width = `${((i + 1) / 8) * 100}%`;
    msg.className = "msg";
    msg.textContent = `Captured ${i + 1} of 8…`;
    await new Promise((r) => setTimeout(r, 320));
  }

  msg.textContent = "Checking the shots…";
  const res = await cgFetch("/api/enroll", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, shots }),
  });
  $("enrollGo").disabled = false;

  if (!res.ok) {
    let detail;
    try { detail = JSON.parse((await res.json()).detail); } catch { detail = null; }
    msg.className = "msg err";
    msg.textContent = detail ? `${detail.message} ${detail.detail.join("; ")}`
                             : "Registration failed.";
    $("enrollBar").style.width = "0";
    return;
  }
  const person = await res.json();
  msg.className = "msg ok";
  msg.textContent = `${person.name} registered as #${person.id}.`;
  await loadRoster();
  setTimeout(closeEnroll, 900);
};

function escape(s) {
  return String(s).replace(/[&<>"]/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

loadRoster();
connect();
paintLiveGraph([], [], []);
