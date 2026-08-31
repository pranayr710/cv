# Setup — where everything lives

Everything the project needs is now under one root:

```
c:\Users\prana\Desktop\sem5\projects\vision\
├── .venv/                     the Python environment (6.2 GB)
├── backend/                   pipeline modules
├── tools/                     CLI tools and the web dashboard
├── tests/                     420 tests
├── dataset/                   footage and training data (2.3 GB)
│   ├── 23-08/vedio/           the audited 5.5-minute clip
│   ├── behaviour_merged/      4-class behaviour training set
│   └── lecture_2636/          157 clips, 27 minutes at 1080p
├── runs/behaviour/            trained behaviour weights
└── outputs/                   everything the pipeline writes
```

`dataset/`, `runs/`, `outputs/` and `.venv/` are all gitignored, so a clone
gets the code and tests but none of the data — see "Fresh clone" below.

## Running anything

Use the project's own interpreter. The system Python has no torch.

```bat
cd c:\Users\prana\Desktop\sem5\projects\vision
.venv\Scripts\python.exe -m tools.server
```

Tool defaults now resolve without arguments, because the data sits beside the
code:

```bat
.venv\Scripts\python.exe -m tools.sweep_identity --students 7
.venv\Scripts\python.exe -m tools.bench_pipeline
.venv\Scripts\python.exe -m tools.batch_session --dir dataset\lecture_2636 --limit 60
```

## The three things you actually run

| What | Command |
|---|---|
| Live demo | `.venv\Scripts\python.exe -m tools.server` then open http://127.0.0.1:8000 |
| Offline session | `.venv\Scripts\python.exe -m tools.batch_session --dir dataset\lecture_2636 --limit 60 --out outputs\session` |
| Tests | `.venv\Scripts\python.exe -m pytest -q` |

## Fresh clone

A clone is **not** self-contained. After cloning you need:

1. **A virtual environment** — `python -m venv .venv` then
   `.venv\Scripts\python.exe -m pip install -r requirements.txt`.
   Roughly 6 GB, mostly torch with CUDA.
2. **Footage** — `dataset/` is gitignored. Copy it from a machine that has it.
3. **Behaviour weights** — `runs/behaviour/merged4_aug/weights/best.pt`.
   Regenerate with
   `python -m tools.train_behaviour --data dataset/behaviour_merged/data.yaml --imgsz 640`
   (about 23 minutes on an RTX 4050). Train at 640: 960 with batch 8 wants
   9 GB on a 6.4 GB card, spills to system memory and takes 18 hours.

Without the weights the pipeline still runs; `behaviour` is null and the
rule-based action layer carries the session.

## A note on the venv

It was moved here from `projects/cv/.venv` and verified afterwards: `sys.prefix`
points at the new location, CUDA is available, pip works and all tests pass.
Windows virtual environments do not always survive a move, so if it is ever
relocated again, re-run the test suite before trusting it.
