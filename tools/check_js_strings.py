"""Catch the one JavaScript defect this project keeps reintroducing.

A double- or single-quoted string cannot span a newline in JavaScript, and
writing one through a shell heredoc turns its \\n into a real newline. That
has broken the dashboard three times: each time the whole file failed to
parse, every button went dead, and the symptom -- "nothing happens when I
click" -- looked like a logic bug rather than a syntax error.

There is no JavaScript engine in this environment to parse with, so this
checks the specific defect rather than the whole grammar. Template literals
in backticks may legitimately span lines and are skipped, as are comments and
regular-expression character classes, both of which contain quotes that are
not string delimiters.

    python tools/check_js_strings.py tools/static/app.js
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def broken_strings(source: str) -> list[int]:
    """Line numbers where a quoted string runs past the end of its line."""
    bad: list[int] = []
    quote: str | None = None
    line = 1
    i = 0
    while i < len(source):
        char = source[i]
        if quote and char == "\\":
            i += 2
            continue
        if quote:
            if char == quote:
                quote = None
            elif char == "\n" and quote in ('"', "'"):
                bad.append(line)
                quote = None
        elif char in ('"', "'", "`"):
            quote = char
        elif source[i:i + 2] == "/*":
            # A block comment runs until its terminator, not to end of line.
            # Skipping only the first line left the scanner inside the
            # comment, where an apostrophe in "one's" read as a string that
            # never closed -- a false positive in the checker written to find
            # exactly this class of mistake.
            end = source.find("*/", i + 2)
            end = len(source) if end == -1 else end + 2
            line += source.count("\n", i, end)
            i = end
            continue
        elif source[i:i + 2] in ("//", "/["):
            # Line comments and regex classes hold quotes that delimit
            # nothing, and neither continues past the end of its line.
            while i < len(source) and source[i] != "\n":
                i += 1
            continue
        if char == "\n":
            line += 1
        i += 1
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", type=Path, nargs="*",
                    default=[Path("tools/static/app.js")])
    args = ap.parse_args()

    failed = False
    for path in args.files:
        if not path.exists():
            print(f"  {path}: missing")
            failed = True
            continue
        bad = broken_strings(path.read_text(encoding="utf-8"))
        if bad:
            failed = True
            print(f"  {path}: string literal spans a newline at "
                  f"line(s) {', '.join(str(n) for n in bad)}")
        else:
            print(f"  {path}: ok")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
