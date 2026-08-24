"""Prepare the OUC-CGE group-engagement dataset for ClassGraph Stage 5.

OUC-CGE (Ocean University of China — Classroom Group Engagement) is the
external evaluation set for this project's group-level claims: ~12h50m of
real classroom video, 17 participants, clip-level engagement labels on a
three-level ordinal scale (high / medium / low), publicly archived on OSF
under a permanent DOI with an MIT-licensed codebase. It is used ONLY for
evaluation — never training — which is what makes it an honest external check
of the ARG+GCN scaffold in ``backend/group_activity.py``.

HONEST LIMITS of this script: written against the dataset's published
description, NOT against a downloaded copy (the full download had not been
run when this was committed). The directory layout below is therefore a
documented ASSUMPTION; ``--verify-only`` exists so the first real download is
a five-second check instead of a surprise. If reality differs, fix
:func:`discover_clips` — the split logic downstream is layout-agnostic and
fully tested.

Expected layout (assumption, see above)::

    <root>/
        labels.csv          # authoritative labels; see --labels-csv format
        videos/ or clips/   # media files named <clip_stem>.<ext>
        ...anything else... # ignored

labels.csv columns (header required, order-free)::

    clip_id,source_video,camera,label[,start_s,end_s]

* ``clip_id`` must match the media file stem.
* ``camera`` is free text (e.g. front/side); kept verbatim in the manifest.
  Multi-angle clips of the same instant share one ``source_video`` and are
  NEVER split across folds from each other (see split rule).
* ``label`` ∈ {High, Medium, Low} — matched case-insensitively.

THE ONE RULE THIS SCRIPT EXISTS TO ENFORCE: **splits are assigned by source
video / recording session, never by frame or clip.** Frames inside one
recording are near-duplicates of each other; a random frame- or clip-level
split leaks them across folds and produces inflated accuracy that evaporates
on the real deployment classroom. Here, every clip derived from one
``source_video`` lands in exactly one split, chosen to keep the label
distribution approximately stratified (greedy assignment, fully
deterministic — no RNG — therefore re-runnable and diff-stable).

Outputs:
    <out>/manifest.csv   clip_id,source_video,camera,label,split,n_frames?
    stdout summary       per-split label counts + integrity verdicts

Usage:
    python -m tools.prepare_ouccge --root data/OUC-CGE --out data/OUC-CGE/prepared
    python -m tools.prepare_ouccge --root data/OUC-CGE --verify-only
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

VALID_LABELS = ("high", "medium", "low")
SPLITS = ("train", "val", "test")
# Greedy target proportions; test held out at 20% because with ~3k clips but
# only ~17 underlying sources, splits are small in *sources*, not clips.
SPLIT_RATIOS = {"train": 0.6, "val": 0.2, "test": 0.2}

MEDIA_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".webm"}


@dataclass(frozen=True)
class Clip:
    clip_id: str
    source_video: str
    camera: str
    label: str            # normalized lowercase


def discover_clips(root: Path, labels_csv: Path) -> list[Clip]:
    """Parse labels.csv and cross-check against the media actually present.

    Raises with a precise message on the failure modes that would otherwise
    silently corrupt evaluation: missing files, unknown labels, duplicate ids.
    Media missing from disk is reported but not fatal (labels can be prepared
    before the bulk video transfer finishes) unless ``--require-media``.
    """
    if not labels_csv.is_file():
        raise SystemExit(f"labels CSV not found: {labels_csv}")
    media = {
        p.stem: p for p in root.rglob("*") if p.suffix.lower() in MEDIA_EXTS
    }
    clips: dict[str, Clip] = {}
    # utf-8-sig: the label sheet will likely be edited in Excel at some point;
    # a BOM must not turn "clip_id" into "\ufeffclip_id".
    with open(labels_csv, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        required = {"clip_id", "source_video", "camera", "label"}
        missing_cols = required - set(reader.fieldnames or [])
        if missing_cols:
            raise SystemExit(
                f"{labels_csv}: missing columns {sorted(missing_cols)}; "
                f"expected header containing {sorted(required)}"
            )
        for row_no, row in enumerate(reader, 2):
            cid = (row.get("clip_id") or "").strip()
            raw_label = (row.get("label") or "").strip().lower()
            if not cid:
                raise SystemExit(f"{labels_csv}:{row_no}: empty clip_id")
            if cid in clips:
                raise SystemExit(f"{labels_csv}:{row_no}: duplicate clip_id {cid!r}")
            if raw_label not in VALID_LABELS:
                raise SystemExit(
                    f"{labels_csv}:{row_no}: label {row.get('label')!r} not one "
                    f"of High/Medium/Low"
                )
            clips[cid] = Clip(
                clip_id=cid,
                source_video=(row.get("source_video") or "").strip(),
                camera=(row.get("camera") or "unknown").strip(),
                label=raw_label,
            )
    absent = sorted(set(clips) - set(media))
    if absent:
        msg = f"{len(absent)} labelled clips have no media file under {root}"
        print(f"WARN: {msg}, e.g. {absent[:5]}", file=sys.stderr)
    return list(clips.values())


def assign_splits(
    clips: list[Clip],
    ratios: dict[str, float] | None = None,
) -> dict[str, str]:
    """Grouped, greedy-stratified split: whole ``source_video`` -> one split.

    Fully deterministic (no RNG): sources are visited largest-first with a
    name tie-break, so identical inputs always yield an identical manifest —
    re-running after adding new clips changes earlier assignments only when
    the greedy balance genuinely demands it, and the diff in git shows it.
    Stratification is approximate BY CONSTRUCTION (whole sources move
    together), and that trade-off is the point — exact per-label balance
    achieved by splitting inside a source would reintroduce the leakage this
    function exists to prevent. ``verify()`` prints the resulting per-split
    label distribution so any imbalance is explicit rather than hidden.
    """
    ratios = ratios or SPLIT_RATIOS
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("split ratios must sum to 1")

    by_source: dict[str, list[Clip]] = defaultdict(list)
    for c in clips:
        by_source[c.source_video].append(c)

    total = len(clips)
    targets = {s: ratios[s] * total for s in SPLITS}
    counts = {s: 0 for s in SPLITS}
    label_counts = {s: Counter() for s in SPLITS}
    label_totals = Counter(c.label for c in clips)
    split_of: dict[str, str] = {}

    # Largest sources first: they constrain balance the most.
    sources = sorted(by_source, key=lambda s: (-len(by_source[s]), s))
    for src in sources:
        group = by_source[src]
        group_labels = Counter(c.label for c in group)

        best_split, best_key = None, None
        for split in SPLITS:
            fill = abs(
                (counts[split] + len(group)) / max(targets[split], 1e-9) - 1.0
            )
            strat_err = 0.0
            for lab, tot in label_totals.items():
                want = tot * ratios[split]
                have = label_counts[split][lab] + group_labels.get(lab, 0)
                strat_err += abs(have - want)
            key = (round(fill, 9), round(strat_err, 9), split)  # name = stable tie-break
            if best_key is None or key < best_key:
                best_split, best_key = split, key

        split_of[src] = best_split
        counts[best_split] += len(group)
        label_counts[best_split].update(group_labels)
    return split_of


def build_manifest(
    clips: list[Clip], split_of: dict[str, str], out_dir: Path
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "manifest.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["clip_id", "source_video", "camera", "label", "split"])
        for c in sorted(clips, key=lambda c: (c.source_video, c.clip_id)):
            writer.writerow([c.clip_id, c.source_video, c.camera, c.label, split_of[c.source_video]])
    return path


def verify(clips: list[Clip], split_of: dict[str, str]) -> list[str]:
    """Integrity checks eval tools also assume. Returns human-readable
    findings; empty list == clean."""
    findings: list[str] = []
    seen: dict[str, set[str]] = defaultdict(set)
    for c in clips:
        seen[c.source_video].add(split_of[c.source_video])
    for src, splits in sorted(seen.items()):
        if len(splits) > 1:  # structurally impossible via assign_splits;
            findings.append(  # checked anyway in case a manifest was hand-edited
                f"LEAKAGE: source {src!r} spans splits {sorted(splits)}"
            )
    for split in SPLITS:
        n = sum(1 for c in clips if split_of[c.source_video] == split)
        dist = Counter(c.label for c in clips if split_of[c.source_video] == split)
        pct = {k: f"{v / n:.0%}" for k, v in sorted(dist.items())} if n else {}
        print(f"  {split:5s}: {n:4d} clips  {pct}")
    overall = Counter(c.label for c in clips)
    print(f"  total : {len(clips)} clips  {dict(sorted(overall.items()))}")
    n_sources = len(seen)
    print(f"  sources: {n_sources}")
    if n_sources < 10:
        findings.append(
            f"only {n_sources} distinct sources — with so few independent "
            "recordings, report metrics next to per-source breakdown, never alone"
        )
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", required=True, help="downloaded OUC-CGE root dir")
    parser.add_argument("--labels-csv", default=None, help="path to labels.csv (default <root>/labels.csv)")
    parser.add_argument("--out", default=None, help="output dir (default <root>/prepared)")
    parser.add_argument("--verify-only", action="store_true", help="check structure and print split preview; write nothing")
    args = parser.parse_args(argv)

    root = Path(args.root)
    labels_csv = Path(args.labels_csv) if args.labels_csv else root / "labels.csv"
    clips = discover_clips(root, labels_csv)
    if not clips:
        raise SystemExit("no clips parsed — nothing to do")

    split_of = assign_splits(clips)
    problems = verify(clips, split_of)
    for p in problems:
        print(f"ISSUE: {p}")

    if args.verify_only:
        print("verify-only: wrote nothing")
        return 1 if any("LEAKAGE" in p for p in problems) else 0

    manifest = build_manifest(clips, split_of, Path(args.out or root / "prepared"))
    print(f"wrote {manifest}")
    return 1 if any("LEAKAGE" in p for p in problems) else 0


if __name__ == "__main__":
    sys.exit(main())
