# Engagement window labelling

This folder is self-contained. To label windows you need only this folder and
two packages — `opencv-python` and `numpy`. You do **not** need the rest of
the ClassGraph repository, the source video, or a GPU.

## Setup (one time)

```
pip install opencv-python numpy
```

## Label

```
python label_gui.py
```

A window opens showing one student's cropped frames, played back slower than
real time. Judge from what you see — nothing on screen tells you what the
system already thinks.

| Key | Action |
|---|---|
| `o` or ← | on task |
| `f` or → | off task |
| `u` | unclear — excluded from training, still counted |
| `space` | replay the window |
| `b` | back one window |
| `s` | skip (revisit later) |
| `q` | save and quit |

Progress saves after every keypress. Quit any time; resume with the same
command and it picks up where you left off.

For a shorter sitting:

```
python label_gui.py --limit 100
```

`--limit` takes an even spread across the whole session, not the first N
windows, so a partial sitting still covers the full recording rather than only
its opening minutes.

## If more than one person is labelling this same folder

Give each person their own copy of the folder (or their own clone of it), and
have each one pass `--who`:

```
python label_gui.py --who alex
python label_gui.py --who priya
```

Each labeller writes to their own `labels_<name>.json`, so nobody overwrites
anyone else's work. Once everyone is done, collect every `labels_*.json` back
into one folder and run:

```
python merge_labels.py --report
```

This writes `labels_merged.json` (majority vote where labellers disagree) and
prints how many windows were double-labelled and how often labellers
disagreed — that overlap is what an inter-rater agreement figure needs, so
don't skip labelling a few windows in common if more than one person is doing
this.

## What NOT to look at while labelling

`manifest.json` carries the rule pipeline's own verdict and the feature values
for every window (`rule_verdict`, `features`). They're there so the finished
labels can be compared against what the rules already conclude — not so you
can check them before deciding. Judge from the video only.

## Regenerating this package

`prepare_package.py` builds `manifest.json` and `images/` from the pipeline's
own outputs (`outputs/final2/raw.jsonl`, `outputs/final2/live_graph.jsonl`) and
the original source clips. It needs the full repository and is not meant to be
run by a labeller — only re-run it if the underlying pipeline run changes.

```
python prepare_package.py
```
