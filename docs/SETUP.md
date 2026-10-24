# ClassGraph — setup and running guide

End to end, from a fresh clone to a working dashboard. Written for someone
who has not seen this project before.

---

## 1. What this project is

ClassGraph measures **per-student engagement in a classroom from a single
fixed camera**. One frame goes in; what comes out is, for each student
visible: who they are, what they are doing, which way they are facing, and an
attention score on a 1–10 scale, aggregated over 15-second windows.

It is a **pipeline of nine models**, not one model. Eight are used at their
published weights; three components are trained here.

| Stage | Model | Answers |
|---|---|---|
| Detect | YOLO11m | where the people and objects are |
| Track | ByteTrack | which box is the same person next frame |
| Face | SCRFD det_10g | where the face is |
| Identity | ArcFace w600k_r50 | who it is |
| Head | SixDRepNet | which way the head points |
| Body | MediaPipe Pose | shoulders, wrists, lean |
| Face mesh | MediaPipe Face Mesh | eye and mouth geometry |
| Expression | EfficientNet-B0 | facial configuration |
| Behaviour | YOLO11m fine-tuned **(trained here)** | read / write / sleep / using_device |
| Attention | MLP over 21 features **(trained here)** | a 1–10 score |
| Attention | Mamba sequence model **(trained here)** | the same, read from order |

Three design rules run through all of it, and they explain most of what looks
unusual in the output:

* **It abstains.** When a model cannot answer, the system says so rather than
  substituting its best guess. An unidentified student is reported as
  unidentified, never as the nearest match.
* **Nothing rests on one frame.** Every verdict is windowed over 15 seconds.
* **It does not infer emotion.** It classifies observable facial
  configuration. (Barrett et al. 2019; EU AI Act Art. 5(1)(f).)

---

## 2. Prerequisites

* **Python 3.10 or newer**
* **~8 GB disk** for the environment and auto-downloaded weights
* A **webcam** only if you want a live session; recorded video works without
* A **GPU is optional.** Everything runs on CPU, roughly 5–10× slower.

---

## 3. Install

```bash
git clone https://github.com/pranayr710/cv.git classgraph
cd classgraph

python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

Then check the machine:

```bash
classgraph doctor
```

It reports every dependency, every model weight, the camera and free disk,
and says plainly what is missing. **Run this first whenever something does
not work** — it answers most questions faster than reading a traceback.

---

## 4. What a clone gives you, and what it does not

A clone is about **6 MB**. It contains all the code, the tests, the
documentation, the two trained attention scorers in `models/`, and the
measurement record in `results/`.

It deliberately does **not** contain the data:

| Missing | Size | How to get it |
|---|---|---|
| `yolo11m.pt` | 40 MB | **Automatic** — ultralytics downloads it on first use |
| InsightFace, SixDRepNet, expression weights | ~370 MB | **Automatic** on first use |
| `runs/behaviour/merged4_aug/weights/best.pt` | 40 MB | Kaggle — see §5 |
| `dataset/` | 1.2 GB | Kaggle — see §5 |
| `labeling/images/` | 1.8 GB | Kaggle — see §5 |

**You can run the dashboard with none of the downloads.** The first run
fetches the public weights itself. Without `best.pt` the four fine-tuned
behaviour classes are unavailable and the system falls back to
object-and-posture inference, which it reports rather than hides.

### Where the downloads go

Unpack the Kaggle archives so the tree looks like this:

```
classgraph/
├── dataset/
│   ├── behaviour_merged/       # train/ val/ data.yaml
│   └── high-2636(2191-2347)/   # the source clips
├── labeling/images/            # the engagement image set
└── runs/behaviour/merged4_aug/weights/best.pt
```

---

## 5. Running it

### The dashboard (the easiest way to see everything)

```bash
python -m tools.run.server
```

It prints an **access token** — copy it. Open <http://127.0.0.1:8000>, paste
the token when asked, and you can register faces, start a webcam session, or
upload a video and watch it being analysed.

> `classgraph serve` does the same thing.

### A recorded video, no browser

```bash
classgraph video --dir "dataset/high-2636(2191-2347)" --out outputs/run1
python -m tools.run.report outputs/run1
```

The first command writes the per-frame records, the scene graph and the
profiles. The second renders a report you can open in a browser.

### A webcam session, no browser

```bash
classgraph live
```

---

## 6. If something does not work

**`Error loading ASGI app. Attribute "app" not found in module
"tools.run.server"`**

You ran `uvicorn tools.run.server:app`. There is no module-level `app` on
purpose — building one at import time would open the camera and load every
model the moment anything imported the module, including the test suite. Use:

```bash
python -m tools.run.server            # supported, and prints the token
uvicorn tools.run.server:app --factory   # also works
```

The first is better: a bare uvicorn command cannot print the access token the
dashboard needs.

**`Pipeline failed: Input video not found: test.mp4`**

There is no video at that path. Either download the dataset (§5) or point the
command at your own file. Paths with spaces or brackets need quoting:

```bash
classgraph video --dir "dataset/high-2636(2191-2347)" --out outputs/run1
```

**The page loads but every button does nothing**

The access token was not accepted. Reload and paste the one the server
printed at startup. To keep a token across restarts:

```bash
# PowerShell
$env:CLASSGRAPH_TOKEN = "choose-something"
```

**Every student shows a dash instead of an attention score**

Expected on a **one-person** session. Three of the model's twenty-one
features are peer and room measurements that do not exist with one student —
there are no peers, and a room focus needs three people to fit. The model
refuses rather than extrapolating. Use multi-student footage.

**`WARNING: no trained attention model found`**

`models/` is missing or empty. It is tracked in git, so re-pull.

---

## 7. Retraining

Everything can be refit from the data. In order:

```bash
# behaviour detector (needs dataset/behaviour_merged)
python -m tools.train.train_behaviour

# attention model (needs outputs/final2/live_graph.jsonl + labeling/labels.json)
classgraph train

# the range it refuses outside of, and its calibration
python -m tools.train.stamp_model_range
python -m tools.train.fit_calibration
```

The last two are not optional if you refit the attention model: without them
it extrapolates on unfamiliar footage and its 1–10 score saturates at the
ends.

---

## 8. The repository

```
backend/      the pipeline: detection, identity, actions, attention scoring
tools/        78 scripts, grouped by what you would open one for
  run/          what you run to use the system
  train/        fitting the models
  evaluate/     measuring against ground truth
  diagnose/     one-off investigations, each answering a question
  data/         building and maintaining the label sets
  assets/       charts and images for the deck
classgraph/   the installed CLI
models/        the two trained scorers the system loads (tracked)
results/       the measurement behind every number quoted (tracked)
outputs/       what a run produces — disposable, gitignored
tests/         the test suite
docs/          plans, literature review, privacy, per-person work split
ppt/           deck builders
```

Run the tests with `pytest`. They need no data and no GPU; the ones that do
skip cleanly rather than failing.

---

## 9. A note on the data

The image and video sets are **real classroom footage of identifiable
students**. `docs/PRIVACY_AND_ETHICS.md` sets out the position this project
takes, which follows India's DPDP Act (2023): purpose limitation, guardian
consent, and no public display of student profiles.

If you redistribute any of it, do so **privately** and only to people
entitled to see it. Do not publish the raw frames.
