# Deploying ClassGraph

Written for someone installing this on a machine that is not the one it was
built on. It assumes no knowledge of the codebase.

---

## 1. What this system does, and what it needs

ClassGraph reads classroom video and reports, per student, how engaged they
appear. It runs in two modes:

| Mode | Input | Command |
|---|---|---|
| **Recorded** | a folder of video clips | `classgraph video --dir <folder>` |
| **Live** | a webcam | `classgraph serve`, then open the browser |

**Hardware.** A GPU is strongly recommended but not required. On CPU the
recorded path runs roughly five times slower and the live path will not keep
up with a webcam. 8 GB RAM minimum; 5 GB free disk for models and output.

**Python 3.10 or 3.11.** Not 3.12 — MediaPipe has no wheel for it at the
versions this is pinned to.

---

## 2. Install

```bash
git clone https://github.com/pranayr710/cv.git
cd cv
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
pip install -e ".[learned]"
```

The `[learned]` extra pulls PyTorch, which is a ~2 GB download. It is needed
only for the Mamba sequence scorer. Without it everything else still runs and
the attention score comes from the logistic/MLP model, which is the one that
ships by default anyway.

### Check the install before trusting it

```bash
classgraph doctor
```

This reports the interpreter, every required package, the model weights, the
camera and free disk, and exits non-zero if anything is missing. It exists
because the alternative is a failure several minutes into a video run, thrown
from inside a library, that says nothing about what to install.

---

## 3. Model weights

Two of the four are fetched or shipped automatically:

| Weight | Size | How it arrives |
|---|---|---|
| `yolo11m.pt` | 40 MB | ultralytics downloads it on first use |
| SCRFD + ArcFace (`buffalo_l`) | 180 MB | insightface downloads on first use |
| SixDRepNet | 150 MB | downloads on first use |
| **`runs/behaviour/merged4_aug/weights/best.pt`** | **40 MB** | **must be copied manually** |

The fine-tuned behaviour model is the one exception. It is too large for the
repository and is not on any public host, so it has to be copied from the
machine that trained it, preserving the path. Without it the pipeline still
runs: behaviour falls back to object-and-posture inference, and the four
trained classes (`read`, `write`, `sleep`, `using_device`) are unavailable.

The trained attention scorers **do** ship, in `outputs/`. They are 76 KB. If
they are missing the system silently falls back to the rule-based verdict,
which measured 0.447 accuracy against the model's 0.650 — a quiet and
substantial downgrade, which is why `classgraph doctor` reports on them.

### First run needs internet

The automatic downloads happen once. After that the system runs offline. On an
air-gapped machine, run it once somewhere with a connection and copy the model
cache (`~/.insightface/`, and the ultralytics cache) across.

---

## 4. Security configuration — read before any network deployment

Two environment variables decide whether this is safe to run. Neither has a
default that protects you, because a default secret in a repository is on
every installation at once.

### `CLASSGRAPH_TOKEN` — who may use the dashboard

Every API route, and the WebSocket, requires it. Without it set, one is
generated per run and printed at startup:

```
ClassGraph dashboard -> http://127.0.0.1:8000
  access token: IunDG4V8jhW1h21XXjLTsSFyenv-yBAG
```

Paste that into the browser when it asks. The page does **not** carry the
token — an earlier version did, which meant `curl http://host:8000/` handed
it to anyone on the network.

For a deployment that survives restarts:

```bash
export CLASSGRAPH_TOKEN="$(python -c 'import secrets;print(secrets.token_urlsafe(24))')"
```

Prefer the header over the query parameter. A token in a URL reaches reverse
proxy and browser history logs:

```bash
curl -H "X-ClassGraph-Token: $CLASSGRAPH_TOKEN" http://127.0.0.1:8000/api/status
```

### `CLASSGRAPH_GALLERY_KEY` — encryption at rest

The gallery holds children's names beside their face embeddings. With this
set, it is encrypted with a scrypt-derived key:

```bash
pip install "classgraph[secure]"
export CLASSGRAPH_GALLERY_KEY="a long passphrase you can retrieve"
```

**If you lose it, the gallery is unrecoverable** — everyone must re-enrol.
Store it where you store other credentials, not next to the data.

With the passphrase set and `cryptography` missing, enrolment **fails** rather
than writing plaintext. That is deliberate: an operator who set the key
believes the file is encrypted.

### Check what is actually in force

```bash
python backend/vault.py --gallery outputs/enrollment/gallery.json
```

Reports whether the file is encrypted, how large it is, and what permissions
it carries.

### Retention

Enrolments are dated. Remove old ones at the end of a term:

```bash
python backend/vault.py --expire-after 120 --dry-run
python backend/vault.py --expire-after 120
```

Undated records count as expired. An encrypted gallery stays encrypted.

### Deleting one person

```bash
python tools/erase_person.py "Name" --dry-run
python tools/erase_person.py "Name"
```

One command covering all three stores: the face record, the session database,
and every output folder. It prints what it could **not** reach — an emailed
report, a backup — because a deletion that silently missed a copy is worse
than one that says where it stopped.

### Before exposing this on a network

- Bind to loopback and put a reverse proxy with TLS in front. `--host 0.0.0.0`
  prints a warning naming what is exposed.
- The token travels in plain HTTP without TLS.
- Confirm `classgraph doctor` passes on the target machine.


## 5. Running it

### Recorded footage

```bash
classgraph video --dir /path/to/clips --out outputs/session1
```

Clips are processed as **one continuous session**, not independently. That
matters: identity is discovered by clustering, so running each clip separately
would give the same student a different id in every file and a 27-minute
lecture would report 157 times the real roster.

Useful flags: `--limit N` (first N clips), `--sample-rate N` (process every Nth
frame; 5 is a reasonable default and what the published figures use).

Output lands in `--out`: `report.html` to open in a browser, plus
`live_graph.jsonl` and `live_profiles.json` for anything downstream.

### Live dashboard

```bash
classgraph serve                      # http://127.0.0.1:8000
classgraph serve --host 0.0.0.0 --port 8080   # on a network
```

One page does registration, the live session and the report.

**Only one process can hold a camera.** A second server will read no frames
and stop, usually with nothing on screen explaining why. If a session ends by
itself, this is almost always the reason:

```bash
python tools/stop_servers.py --kill
```

---

## 6. Docker

For the recorded path. Live capture stays native, because passing a camera
into a container is platform-specific and unsupported on Docker Desktop for
Windows and macOS.

```bash
docker build -t classgraph:1.0 .

docker run --rm \
  -v "$PWD/dataset:/data:ro" \
  -v "$PWD/outputs:/app/outputs" \
  classgraph:1.0 classgraph video --dir /data/clips --out /app/outputs/run

# dashboard
docker run --rm -p 8000:8000 -v "$PWD/outputs:/app/outputs" classgraph:1.0

# with a GPU (needs the NVIDIA Container Toolkit)
docker run --rm --gpus all ... classgraph:1.0
```

The image declares a healthcheck against `/api/status`, so an orchestrator can
tell a running server from a crashed one.

---

## 7. Privacy, before you deploy this anywhere real

The face gallery is **biometric data about named people**. Deploying this in a
school means:

- Frames are streamed to the browser but **never written to disk**. Only
  derived records — the graph, the profiles — are stored.
- The gallery (`outputs/enrollment/`) identifies real individuals by name and
  face embedding. It needs the same handling as any other personal record:
  access control, a retention limit, and a documented way to delete someone.
- Engagement scores are **inferences about people**, not measurements of them.
  A score of 3/10 is the model's reading of posture and gaze over 15 seconds,
  on a model measuring 0.650 accuracy against human judgement. It is not a
  fact about whether a student was paying attention, and anything built on top
  of it should say so where a teacher can see it.

---

## 8. Known limits

State these rather than let someone discover them:

| Limit | Detail |
|---|---|
| **Headcount over-counts** | 13–18 ids for 8 real people. Per-student figures are sound per id; the roster count is not. Cause is still unidentified — the duplicate-box hypothesis was measured and ruled out. |
| **Attention accuracy is 0.650** | Macro F1 0.648, against a rule baseline of 0.447. Measured out-of-fold with student-grouped splits. Label-limited: ~185 more labels reaches 0.70. |
| **No pen, pencil or paper detection** | COCO has 80 classes and none of them. A pen is ~69×5 px at classroom distance and below what the detector can resolve. Writing is read from posture instead. |
| **Emotion is uncalibrated** | The expression model's confidence threshold is a starting point, not a validated one. |
| **Human agreement is unmeasured** | Nobody has measured how often two people agree on these labels, so the ceiling on any accuracy figure is unknown. |

---

## 9. Troubleshooting

| Symptom | Cause |
|---|---|
| Session stops by itself after seconds | Another server holds the camera. `python tools/stop_servers.py --kill` |
| `ModuleNotFoundError: No module named 'backend'` | Run from the repository root, or `pip install -e .` |
| Every student scores 1/10 | A stale model fitted to a different feature set. `classgraph train` |
| No behaviour labels in the report | `best.pt` missing — see §3 |
| Nobody detected on a webcam | Fixed; if it recurs, check `imgsz` against the frame size. Upscaling beyond 1.5× erased people entirely. |
| Very slow | No GPU. `classgraph doctor` reports which is in use. |
