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
    const label = b.name ? `${b.name} · ${ACTION_LABEL[b.action] || ""}` : "unknown";
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
    $("objects").innerHTML = d.objects.length
      ? d.objects.map((o) => `<span class="chip">${escape(o)}</span>`).join("")
      : '<span class="empty">none</span>';
    paintRoster(d.students);
    paintPairs(d.pairs, d.students);
    requestAnimationFrame(() => paintOverlay(d.boxes));
  };
  socket.onclose = () => setTimeout(connect, 1200);
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
