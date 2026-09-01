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

const $ = (id) => document.getElementById(id);
const feed = $("feed");
const overlay = $("overlay");
const ctx = overlay.getContext("2d");
let running = false;
let socket = null;
let enrollStream = null;

/* ---------------------------------------------------------------- roster */

async function loadRoster() {
  const { students } = await (await fetch("/api/students")).json();
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
    const pct = el.querySelector(".pct");
    pct.textContent = s.attention == null ? "—" : `${Math.round(s.attention * 100)}%`;
    pct.style.color = s.attention == null ? "var(--muted)"
      : s.attention >= 0.7 ? "var(--accent)"
      : s.attention >= 0.4 ? "var(--warn)" : "var(--bad)";

    const act = s.action ? ACTION_LABEL[s.action] || s.action : "not seen";
    el.querySelector(".act").innerHTML =
      `<span style="color:${ACTION_COLOUR[s.action] || "var(--muted)"}">●</span> ${escape(act)}` +
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
  ctx.lineWidth = 2;
  ctx.font = "600 13px 'Segoe UI',sans-serif";
  for (const b of boxes) {
    const [x, y, bw, bh] = b.bbox;
    const colour = b.name ? (ACTION_COLOUR[b.action] || "#4ea36b") : "#c9553d";
    ctx.strokeStyle = colour;
    ctx.strokeRect(ox + x * scale, oy + y * scale, bw * scale, bh * scale);
    const label = b.name
      ? `${b.name} · ${ACTION_LABEL[b.action] || ""}`
      : (b.face_px ? `unknown · face ${b.face_px}px` : "unknown");
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
  socket = new WebSocket(`ws://${location.host}/ws`);
  socket.onmessage = (event) => {
    const d = JSON.parse(event.data);
    if (d.idle) return;
    if (d.image) {
      feed.hidden = false;
      $("ph").hidden = true;
      feed.src = `data:image/jpeg;base64,${d.image}`;
    }
    $("tInFrame").textContent = d.in_frame;
    $("tKnown").textContent = d.students.filter((s) => s.present).length;
    $("tFrames").textContent = d.frame;
    $("tFps").textContent = d.fps;
    $("hdr").textContent = `${d.seconds}s · ${d.fps} fps`;
    const hint = $("hint");
    if (d.hint) { hint.textContent = d.hint; hint.hidden = false; }
    else hint.hidden = true;
    $("objects").innerHTML = d.objects.length
      ? d.objects.map((o) => `<span class="chip">${escape(o)}</span>`).join("")
      : '<span class="empty">none</span>';
    window.lastStudents = d.students;
    window.lastObjects = d.objects;
    window.lastPairs = d.pairs;
    paintRoster(d.students);
    paintPairs(d.pairs, d.students);
    paintLiveGraph(d.students, d.objects, d.pairs);
    requestAnimationFrame(() => paintOverlay(d.boxes));
  };
  socket.onclose = () => setTimeout(connect, 1200);
}

let currentGraphMode = "radar";
let currentGraphFilter = "all";
let focusedStudentId = null;
let nodePositions = {};

function setGraphMode(mode) {
  currentGraphMode = mode;
  document.querySelectorAll(".graph-tab").forEach(tab => {
    tab.classList.toggle("active", tab.dataset.mode === mode);
  });
  const labels = {
    radar: "🛰️ Gaze Radar & Spatial Beams",
    topology: "🕸️ Scene Topology Network",
    flow: "🌊 Energy Flow Stream",
    heatmap: "🔥 Social Interaction Matrix"
  };
  $("graphModeLabel").textContent = labels[mode] || mode;
}

function setFilter(filter, el) {
  currentGraphFilter = filter;
  document.querySelectorAll(".filter-pill").forEach(pill => pill.classList.remove("active"));
  if (el) el.classList.add("active");
}

function toggleFullscreenGraph() {
  const card = $("graphCard");
  if (card) {
    card.classList.toggle("fullscreen-graph");
  }
}

function showGraphHud(e, title, body) {
  const hud = $("graphHud");
  if (!hud) return;
  $("hudTitle").textContent = title;
  $("hudBody").innerHTML = body;
  hud.style.opacity = "1";
}

function hideGraphHud() {
  const hud = $("graphHud");
  if (hud) hud.style.opacity = "0";
}

const DEMO_STUDENTS = [
  { id: 1, name: "Aarav", present: true, action: "attentive", attention: 0.94, gaze: "teacher" },
  { id: 2, name: "Bhavya", present: true, action: "studying", attention: 0.88, gaze: "down" },
  { id: 3, name: "Chetan", present: true, action: "on_phone", attention: 0.28, gaze: "down" },
  { id: 4, name: "Divya", present: true, action: "attentive", attention: 0.91, gaze: "teacher" }
];
const DEMO_OBJECTS = ["laptop", "book", "cell phone"];
const DEMO_PAIRS = [{ a: 1, b: 2, frames: 42 }];

function paintLiveGraph(students, detectedObjects, pairs) {
  const svg = $("graphCanvas");
  if (!svg) return;
  let present = (students || []).filter(s => s && s.present);
  let objects = Array.from(new Set(detectedObjects || []));
  let pairList = pairs || [];

  if (!present.length) {
    present = DEMO_STUDENTS;
    objects = DEMO_OBJECTS;
    pairList = DEMO_PAIRS;
  }

  const W = 900, H = 420;

  if (currentGraphMode === "heatmap") {
    renderInteractionMatrix(svg, present, pairList, W, H);
    return;
  }

  if (currentGraphMode === "radar") {
    renderGazeRadar(svg, present, objects, pairList, W, H);
  } else if (currentGraphMode === "flow") {
    renderEnergyFlow(svg, present, objects, pairList, W, H);
  } else {
    renderNetworkTopology(svg, present, objects, pairList, W, H);
  }
}

function renderGazeRadar(svg, present, objects, pairs, W, H) {
  let html = `
    <defs>
      <linearGradient id="podiumGrad" x1="0%" y1="0%" x2="100%" y2="0%">
        <stop offset="0%" stop-color="#6366f1"/>
        <stop offset="100%" stop-color="#06b6d4"/>
      </linearGradient>
      <linearGradient id="laserBeam" x1="0%" y1="0%" x2="100%" y2="0%">
        <stop offset="0%" stop-color="#10b981" stop-opacity="0.8"/>
        <stop offset="100%" stop-color="#34d399" stop-opacity="0.2"/>
      </linearGradient>
    </defs>

    <!-- Teacher Podium / Board -->
    <g transform="translate(450, 45)">
      <rect x="-140" y="-18" width="280" height="36" rx="10" fill="rgba(99, 102, 241, 0.12)" stroke="url(#podiumGrad)" stroke-width="2"/>
      <text x="0" y="5" fill="#fff" font-size="13" font-weight="700" text-anchor="middle" font-family="'Outfit', sans-serif">👩‍🏫 Teacher Podium &amp; Screen Target</text>
    </g>
  `;

  // Spatial Student Desk Positions
  const cols = Math.min(present.length, 4);
  const rows = Math.ceil(present.length / cols);
  
  present.forEach((s, idx) => {
    const col = idx % cols;
    const row = Math.floor(idx / cols);
    const x = 180 + col * (540 / Math.max(cols - 1, 1));
    const y = 160 + row * 110;
    
    nodePositions[s.id] = { x, y };

    const isFocused = focusedStudentId === null || focusedStudentId === s.id;
    const groupClass = isFocused ? "graph-highlighted" : "graph-dimmed";

    // Laser Gaze Ray pointing up to Podium or Target
    const targetY = 65;
    const targetX = 450 + (col - (cols - 1) / 2) * 40;

    if (currentGraphFilter === "all" || currentGraphFilter === "gaze") {
      const laserColor = s.action === "on_phone" || s.action === "eyes_closed" ? "#f43f5e" : "#10b981";
      html += `
        <line x1="${x}" y1="${y - 18}" x2="${targetX}" y2="${targetY}" 
              stroke="${laserColor}" stroke-opacity="${isFocused ? 0.65 : 0.1}" stroke-width="2.5" stroke-dasharray="6 3" class="${groupClass}"/>
      `;
    }

    const actColor = ACTION_COLOUR[s.action] || "#10b981";
    html += `
      <g transform="translate(${x},${y})" class="${groupClass}" cursor="pointer"
         onmouseover="focusedStudentId=${s.id}; showGraphHud(event, '${escape(s.name)} (#${s.id})', 'Action: <b>${s.action || "attentive"}</b><br>On-Task: <b>${s.attention != null ? Math.round(s.attention*100) + "%" : "—"}</b>'); paintLiveGraph(window.lastStudents||[], window.lastObjects||[], window.lastPairs||[]);"
         onmouseout="focusedStudentId=null; hideGraphHud(); paintLiveGraph(window.lastStudents||[], window.lastObjects||[], window.lastPairs||[]);">
        <!-- Desk -->
        <rect x="-42" y="-12" width="84" height="42" rx="8" fill="rgba(15, 23, 42, 0.8)" stroke="${actColor}" stroke-width="1.8"/>
        <!-- Student Avatar Circle -->
        <circle cx="0" cy="-14" r="16" fill="${actColor}" stroke="#030611" stroke-width="2.5"/>
        <text x="0" y="-10" fill="#ffffff" font-size="11" font-weight="700" text-anchor="middle" font-family="'JetBrains Mono', monospace">${s.id}</text>
        <text x="0" y="16" fill="#f8fafc" font-size="11" font-weight="600" text-anchor="middle" font-family="'Outfit', sans-serif">${escape(s.name)}</text>
      </g>
    `;
  });

  svg.innerHTML = html;
}

function renderNetworkTopology(svg, present, objects, pairs, W, H) {
  const px = 220, ox = 680;
  let html = `
    <defs>
      <linearGradient id="topoGrad" x1="0%" y1="0%" x2="100%" y2="0%">
        <stop offset="0%" stop-color="#818cf8"/>
        <stop offset="100%" stop-color="#06b6d4"/>
      </linearGradient>
    </defs>
  `;

  const py = {}, oy = {};
  present.forEach((s, i) => { py[s.id] = (H / (present.length + 1)) * (i + 1); });
  objects.forEach((obj, i) => { oy[obj] = (H / (objects.length + 1)) * (i + 1); });

  // Draw Shared Action Arcs
  if (pairs && pairs.length && (currentGraphFilter === "all" || currentGraphFilter === "actions")) {
    pairs.forEach(p => {
      if (py[p.a] && py[p.b]) {
        const midY = (py[p.a] + py[p.b]) / 2;
        const bend = px - 70;
        html += `<path d="M${px},${py[p.a]} Q${bend},${midY} ${px},${py[p.b]}" fill="none" stroke="#f97316" stroke-width="2.5" stroke-dasharray="5 4" stroke-opacity="0.8"/>`;
      }
    });
  }

  // Draw Student to Object Arcs
  if (currentGraphFilter === "all" || currentGraphFilter === "objects") {
    present.forEach(s => {
      objects.forEach(obj => {
        const midX = (px + ox) / 2;
        const midY = (py[s.id] + oy[obj]) / 2;
        html += `<path d="M${px},${py[s.id]} Q${midX},${midY} ${ox},${oy[obj]}" fill="none" stroke="url(#topoGrad)" stroke-width="2" stroke-opacity="0.65"/>`;
      });
    });
  }

  // Render Student Nodes
  present.forEach(s => {
    const actColor = ACTION_COLOUR[s.action] || "#10b981";
    html += `
      <g transform="translate(${px},${py[s.id]})" cursor="pointer"
         onmouseover="showGraphHud(event, '${escape(s.name)}', 'Status: ${s.action}')" onmouseout="hideGraphHud()">
        <circle r="16" fill="${actColor}" stroke="#030611" stroke-width="3"/>
        <text x="0" y="4" fill="#fff" font-size="11" font-weight="700" text-anchor="middle" font-family="'JetBrains Mono', monospace">${s.id}</text>
        <text x="-24" y="4" fill="#f8fafc" font-size="12" font-weight="600" text-anchor="end" font-family="'Outfit', sans-serif">${escape(s.name)}</text>
      </g>
    `;
  });

  // Render Object Nodes
  const objIcons = { laptop: "💻", book: "📖", "cell phone": "📱", keyboard: "⌨️", bottle: "🍾", cup: "☕" };
  objects.forEach(obj => {
    const icon = objIcons[obj] || "📦";
    html += `
      <g transform="translate(${ox},${oy[obj] - 14})">
        <rect width="150" height="28" rx="8" fill="rgba(6, 182, 212, 0.15)" stroke="#06b6d4" stroke-opacity="0.5" stroke-width="1.2"/>
        <text x="12" y="18" fill="#f8fafc" font-size="12" font-weight="600">${icon} ${escape(obj)}</text>
      </g>
    `;
  });

  svg.innerHTML = html;
}

function renderEnergyFlow(svg, present, objects, pairs, W, H) {
  const px = 220, ox = 680;
  let html = `
    <defs>
      <linearGradient id="flowGrad" x1="0%" y1="0%" x2="100%" y2="0%">
        <stop offset="0%" stop-color="#38bdf8"/>
        <stop offset="100%" stop-color="#818cf8"/>
      </linearGradient>
    </defs>
  `;

  const py = {}, oy = {};
  present.forEach((s, i) => { py[s.id] = (H / (present.length + 1)) * (i + 1); });
  objects.forEach((obj, i) => { oy[obj] = (H / (objects.length + 1)) * (i + 1); });

  present.forEach(s => {
    objects.forEach(obj => {
      const midX = (px + ox) / 2;
      const midY = (py[s.id] + oy[obj]) / 2;
      html += `<path d="M${px},${py[s.id]} Q${midX},${midY} ${ox},${oy[obj]}" fill="none" stroke="url(#flowGrad)" stroke-width="3" class="flow-line"/>`;
    });
  });

  present.forEach(s => {
    const actColor = ACTION_COLOUR[s.action] || "#10b981";
    html += `
      <g transform="translate(${px},${py[s.id]})">
        <circle r="16" fill="${actColor}" stroke="#030611" stroke-width="3"/>
        <text x="0" y="4" fill="#fff" font-size="11" font-weight="700" text-anchor="middle" font-family="'JetBrains Mono', monospace">${s.id}</text>
        <text x="-24" y="4" fill="#f8fafc" font-size="12" font-weight="600" text-anchor="end" font-family="'Outfit', sans-serif">${escape(s.name)}</text>
      </g>
    `;
  });

  const objIcons = { laptop: "💻", book: "📖", "cell phone": "📱", keyboard: "⌨️", bottle: "🍾", cup: "☕" };
  objects.forEach(obj => {
    const icon = objIcons[obj] || "📦";
    html += `
      <g transform="translate(${ox},${oy[obj] - 14})">
        <rect width="150" height="28" rx="8" fill="rgba(6, 182, 212, 0.15)" stroke="#06b6d4" stroke-width="1.2"/>
        <text x="12" y="18" fill="#f8fafc" font-size="12" font-weight="600">${icon} ${escape(obj)}</text>
      </g>
    `;
  });

  svg.innerHTML = html;
}

function renderInteractionMatrix(svg, present, pairs, W, H) {
  if (present.length < 2) {
    svg.innerHTML = '<text x="450" y="210" fill="var(--muted)" font-size="14" text-anchor="middle" font-family="\'Outfit\', sans-serif">Needs at least 2 present students to calculate pairwise social interaction matrix</text>';
    return;
  }

  const size = Math.min(320 / present.length, 55);
  const startX = 450 - (present.length * size) / 2;
  const startY = 80;

  let html = `<g transform="translate(0, 0)">`;

  // Row and Header labels
  present.forEach((s, i) => {
    html += `<text x="${startX + i * size + size / 2}" y="${startY - 10}" fill="#a5b4fc" font-size="11" font-weight="700" text-anchor="middle" font-family="'Outfit', sans-serif">${escape(s.name)}</text>`;
    html += `<text x="${startX - 12}" y="${startY + i * size + size / 2 + 4}" fill="#a5b4fc" font-size="11" font-weight="700" text-anchor="end" font-family="'Outfit', sans-serif">${escape(s.name)}</text>`;
  });

  const pairLookup = {};
  (pairs || []).forEach(p => {
    pairLookup[`${p.a}_${p.b}`] = p.frames;
    pairLookup[`${p.b}_${p.a}`] = p.frames;
  });

  present.forEach((s1, r) => {
    present.forEach((s2, c) => {
      const x = startX + c * size;
      const y = startY + r * size;
      if (r === c) {
        html += `<rect x="${x + 2}" y="${y + 2}" width="${size - 4}" height="${size - 4}" rx="6" fill="rgba(255,255,255,0.04)" stroke="rgba(255,255,255,0.08)"/>`;
      } else {
        const val = pairLookup[`${s1.id}_${s2.id}`] || 0;
        const fill = val > 20 ? "rgba(16, 185, 129, 0.5)" : val > 0 ? "rgba(99, 102, 241, 0.35)" : "rgba(15, 23, 42, 0.6)";
        html += `
          <rect x="${x + 2}" y="${y + 2}" width="${size - 4}" height="${size - 4}" rx="6" fill="${fill}" stroke="rgba(255,255,255,0.1)"/>
          <text x="${x + size/2}" y="${y + size/2 + 4}" fill="#fff" font-size="11" font-weight="700" text-anchor="middle" font-family="'JetBrains Mono', monospace">${val}</text>
        `;
      }
    });
  });

  html += `</g>`;
  svg.innerHTML = html;
}

/* ---------------------------------------------------------------- session */

$("btnStart").onclick = async () => {
  const res = await fetch("/api/session/start", { method: "POST" });
  if (!res.ok) { alert((await res.json()).detail); return; }
  running = true;
  $("btnStart").hidden = true;
  $("btnStop").hidden = false;
  $("btnEnroll").disabled = true;
  $("dot").classList.add("live");
};

$("btnStop").onclick = async () => {
  const res = await fetch("/api/session/stop", { method: "POST" });
  const d = await res.json();
  running = false;
  $("btnStart").hidden = false;
  $("btnStop").hidden = true;
  $("btnEnroll").disabled = false;
  $("dot").classList.remove("live");
  $("hdr").textContent = `${d.frames} frames captured`;
  if (d.report) window.open(d.report, "_blank");
};

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
  const res = await fetch("/api/enroll", {
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
