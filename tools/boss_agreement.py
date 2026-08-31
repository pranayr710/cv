"""Cohen's kappa between blind human BOSS coding and system engagement verdicts.

This is the "cheapest real validation" from docs/WORK_PERSON_C.md: a half-day
study where two raters independently code short intervals of classroom video
with a simplified BOSS protocol (docs/BOSS_VALIDATION.md), the system's
verdicts are pulled for the same intervals, and agreement is computed here.

Why kappa and not raw percent agreement: two raters agree on most intervals by
base rates alone (classrooms are mostly on-task). Cohen's kappa corrects for
chance agreement, so it is comparable across classrooms and against published
BOSS inter-rater figures. Percent agreement is printed too, but kappa is the
number that goes in reports.

Inputs: two CSVs with identical interval keys, one row per (interval, student):

    human CSV : interval,student_id,code      code  in  {on,off,unknown}
    system CSV: interval,student_id,code

Rows are joined on (interval, student_id); unmatched rows are REPORTED as
coverage gaps, never silently dropped — a 40%-coverage kappa is a different
claim than an 95%-coverage one, and this tool refuses to let them look alike.

Also supports rater-vs-rater mode (--rater2) for the pre-study training gate:
two humans should reach kappa >= 0.6 before either codes the real study
(published BOSS studies typically report higher; 0.6 is our floor, stated in
the protocol doc).

Usage:
    python -m tools.boss_agreement --human human.csv --system system.csv
    python -m tools.boss_agreement --human rater1.csv --system rater2.csv --rater2
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

CODES = ("on", "off", "unknown")


@dataclass(frozen=True)
class Row:
    interval: str
    student_id: str
    code: str


def load_rows(path: Path) -> dict[tuple[str, str], Row]:
    rows: dict[tuple[str, str], Row] = {}
    # utf-8-sig: tolerate BOMs from spreadsheet-exported coding sheets.
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        needed = {"interval", "student_id", "code"}
        missing = needed - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{path}: missing columns {sorted(missing)}")
        for line_no, row in enumerate(reader, 2):
            key = (row["interval"].strip(), str(row["student_id"]).strip())
            if key in rows:
                raise SystemExit(f"{path}:{line_no}: duplicate interval/student {key}")
            code = row["code"].strip().lower()
            if code not in CODES:
                raise SystemExit(
                    f"{path}:{line_no}: code {row['code']!r} not in {CODES}; "
                    "map richer BOSS categories first (see docs/BOSS_VALIDATION.md)"
                )
            rows[key] = Row(*key, code)
    return rows


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa over paired categorical codes."""
    n = len(a)
    if n == 0 or len(a) != len(b):
        raise ValueError("kappa needs equal-length non-empty lists")
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    counts_a, counts_b = Counter(a), Counter(b)
    expected = sum(counts_a[c] * counts_b[c] for c in set(a) | set(b)) / (n * n)
    if expected == 1.0:
        # Both raters emitted exactly one category everywhere: kappa is
        # undefined (0/0). Return 1.0 only if they also agreed, else 0.
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def compare(human: dict[tuple[str, str], Row], system: dict[tuple[str, str], Row]) -> dict[str, object]:
    keys_human, keys_system = set(human), set(system)
    common = sorted(keys_human & keys_system)
    only_h = keys_human - keys_system
    only_s = keys_system - keys_human
    conf = Counter((human[k].code, system[k].code) for k in common)
    a = [human[k].code for k in common]
    b = [system[k].code for k in common]
    n = len(common)
    agree = sum(1 for x, y in zip(a, b) if x == y)
    return {
        "n_joined": n,
        "coverage": round(n / len(keys_human), 4) if keys_human else 0.0,
        "n_only_in_human": len(only_h),
        "n_only_in_system": len(only_s),
        "percent_agreement": round(agree / n, 4) if n else None,
        "cohen_kappa": round(cohen_kappa(a, b), 4) if n else None,
        "confusion_human_x_system": {
            h: {s: conf[(h, s)] for s in CODES} for h in CODES
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--human", required=True, help="CSV: interval,student_id,code")
    parser.add_argument("--system", required=True, help="CSV: interval,student_id,code")
    parser.add_argument(
        "--rater2",
        action="store_true",
        help="both inputs are HUMAN raters (training-gate mode; same output, different label)",
    )
    args = parser.parse_args(argv)

    left = load_rows(Path(args.human))
    right = load_rows(Path(args.system))
    res = compare(left, right)

    who = "rater2" if args.rater2 else "system"
    print(f"joined on (interval, student): n={res['n_joined']}  "
          f"coverage of --human rows={res['coverage']:.0%}")
    if res["n_only_in_human"] or res["n_only_in_system"]:
        print(
            f"WARN: unmatched rows — {res['n_only_in_human']} only in --human, "
            f"{res['n_only_in_system']} only in --{who}. Coverage below ~90% "
            "means the kappa describes a subset, not the study."
        )
    if res["percent_agreement"] is None:
        print("no overlapping rows — nothing to score")
        return 1
    print(f"percent agreement = {res['percent_agreement']:.3f}")
    print(f"cohen's kappa     = {res['cohen_kappa']:.3f}"
          f"   ({'rater1 vs rater2' if args.rater2 else 'human vs system'})")
    print(f"\nconfusion (rows=--human, cols=--{who}):")
    for h in CODES:
        cells = "  ".join(
            f"{s}={res['confusion_human_x_system'][h][s]:3d}" for s in CODES
        )
        print(f"  {h:8s}{cells}")
    if args.rater2 and res["cohen_kappa"] is not None:
        gate = res["cohen_kappa"] >= 0.6
        print(f"\ntraining gate: kappa >= 0.6 ? {'PASS' if gate else 'FAIL — recode together and repeat'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
