# tools/

Seventy-eight scripts, grouped by **what you would open one for**. They were a
single flat directory until they weren't navigable: `server.py` sat between
`semi_supervised.py` and `static/`, and nothing told a reader which three of
the seventy-eight they actually needed to run the system.

Every script runs two ways — `python tools/<group>/<name>.py` or
`python -m tools.<group>.<name>` — and takes `--help`.

| Group | Count | What it is for |
|---|---|---|
| [`run/`](run/) | 17 | **Start here.** What you run to use ClassGraph |
| [`train/`](train/) | 9 | Fitting the models, and the loop that improves them |
| [`evaluate/`](evaluate/) | 18 | Measuring the system against ground truth |
| [`diagnose/`](diagnose/) | 17 | One-off investigations; each answers a question |
| [`data/`](data/) | 13 | Building and maintaining the label and image sets |
| [`assets/`](assets/) | 4 | Images and charts for the decks and the docs |

## The three you probably want

```bash
python tools/run/server.py                  # the dashboard, at 127.0.0.1:8000
python tools/run/batch_session.py --dir <folder of clips> --out outputs/<name>
python tools/run/report.py outputs/<name>   # the readable report
```

`run/static/` holds the dashboard's HTML and JS. They live beside
`run/server.py` because that is how it finds them — `Path(__file__).parent /
"static"` — so the two move together or not at all.

## On `diagnose/`

These are not dead code. Each one was written to settle a specific question,
and the answer is now a comment somewhere in `backend/`: why the forward-lean
threshold is −0.21, why the IoU gate was rebuilt, why 991 `looking_away`
verdicts were removed. Deleting the script would leave the claim without the
means to re-check it.
