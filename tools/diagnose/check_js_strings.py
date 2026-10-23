"""Catch the JavaScript defects this project keeps reintroducing.

All three have the same symptom -- a dead button, or a panel that never
appears, with no error anyone sees unless a console is open -- and none of
them is visible in a diff.

1. **A string literal spanning a newline.** A double- or single-quoted string
   cannot cross a line in JavaScript, and writing one through a shell heredoc
   turns its ``\\n`` into a real newline. That has broken the dashboard three
   times; each time the whole file failed to parse and every button went dead.
2. **An inline handler the script does not define.** ``onclick="name()"`` on a
   button that nothing defines is a button that does nothing.
3. **A bare call to a function that was deleted.** This has happened twice, in
   the same edit: removing a block of dead graph code took ``setGraphMode``
   with it, which left every scene-graph tab inert, and ``emptyGraph``, which
   made every graph mode throw the moment it had nothing to draw -- the most
   common case there is.

There is no JavaScript engine in this environment to parse with, so these
check specific defects rather than the whole grammar. Template literals in
backticks may legitimately span lines and are skipped, as are comments and
regular-expression character classes, both of which contain quotes that are
not string delimiters.

    python tools/diagnose/check_js_strings.py tools/run/static/app.js
"""
from __future__ import annotations

import argparse
import re
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


def code_only(source: str) -> str:
    """The source with strings, template literals and comments blanked out.

    Shares its idea of what a literal is with :func:`broken_strings`, because
    a second half-correct notion of it is how a checker starts reporting
    ``rgba`` and ``translate`` as undefined functions -- those appear inside
    SVG attribute strings, not in code. Newlines survive, so a line number
    still means something.
    """
    out: list[str] = []
    quote: str | None = None
    i = 0
    while i < len(source):
        char = source[i]
        if quote:
            if char == "\\":
                out.append("  ")
                i += 2
                continue
            if char == quote:
                quote = None
            out.append("\n" if char == "\n" else " ")
            i += 1
            continue
        if char in ('"', "'", "`"):
            quote = char
            out.append(" ")
            i += 1
            continue
        if source[i:i + 2] == "/*":
            end = source.find("*/", i + 2)
            end = len(source) if end == -1 else end + 2
            out.append(re.sub(r"[^\n]", " ", source[i:end]))
            i = end
            continue
        if source[i:i + 2] in ("//", "/["):
            while i < len(source) and source[i] != "\n":
                out.append(" ")
                i += 1
            continue
        out.append(char)
        i += 1
    return "".join(out)


#: Called in this file but defined by the browser. Listed explicitly, because
#: the alternative is to parse JavaScript properly and there is no engine here
#: to do it with. A method call like ``a.b()`` needs no entry: those are
#: excluded by the pattern, since guessing at what an object offers would bury
#: the real finding in noise.
_NOT_OURS = frozenset("""
alert atob btoa confirm decodeURIComponent encodeURIComponent fetch isNaN
parseFloat parseInt prompt requestAnimationFrame setInterval setTimeout
""".split())

#: Keywords that take a parenthesis and are not calls.
_KEYWORDS = frozenset("""
async await catch const delete do else for function if in instanceof let new
of return switch typeof var void while yield
""".split())


def defined_names(code: str) -> set[str]:
    """Everything the script declares that could be called by name."""
    names = set(re.findall(r"function\s+(\w+)\s*\(", code))
    names |= set(re.findall(r"(?:const|let|var)\s+(\w+)\s*=", code))
    return names


def undefined_calls(script: str) -> list[str]:
    """Names called as bare functions that this file never defines."""
    code = code_only(script)
    # Lower-case initial only. Every function in this codebase is named that
    # way, and the capitalised bare calls are all browser constructors --
    # Blob, FormData, Promise, Set, String, WebSocket. Listing those by name
    # would mean extending the list on every new one, and reporting them
    # would train the reader to ignore this check, which is worse than not
    # having it.
    called = set(re.findall(r"(?<![.\w$])([a-z_]\w*)\s*\(", code))
    return sorted(called - defined_names(code) - _KEYWORDS - _NOT_OURS)


def missing_handlers(html: str, script: str) -> list[str]:
    """Inline handlers the HTML names that the script does not define.

    Only ``on<something>="name("`` is checked, because that is how this page
    wires its buttons. A handler attached with addEventListener cannot go
    missing this way.
    """
    called = set(re.findall(r'on\w+="(\w+)\s*\(', html))
    builtin = {"alert", "confirm", "print", "close", "open"}
    return sorted(called - defined_names(code_only(script)) - builtin)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", type=Path, nargs="*",
                    default=[Path("tools/run/static/app.js")])
    args = ap.parse_args()

    failed = False

    # The page calls into the script by name, and the two files are only
    # connected at runtime, in a browser, by somebody clicking something.
    html = Path("tools/run/static/index.html")
    script = Path("tools/run/static/app.js")
    if html.exists() and script.exists():
        source = script.read_text(encoding="utf-8")
        gone = missing_handlers(html.read_text(encoding="utf-8"), source)
        if gone:
            failed = True
            print(f"  {html}: calls undefined handler(s): {', '.join(gone)}")
        else:
            print(f"  {html}: every inline handler is defined")

        loose = undefined_calls(source)
        if loose:
            failed = True
            print(f"  {script}: calls undefined function(s): "
                  f"{', '.join(loose)}")
        else:
            print(f"  {script}: every bare call resolves")

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
            print(f"  {path}: no string spans a newline")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
