"""Render docs/diagrams/*.mmd to PNG for the slide deck.

Mermaid is the right source format -- text that lives beside the code, so the
arrows cannot drift when a box moves -- but a deck needs pixels. There is no
Node on this machine, so rendering goes through mermaid.ink, the public renderer
maintained by the Mermaid project.

What is sent: the diagram source only, base64 in the URL. That is model names
and stage labels; no code, no data, no credentials. Run with --check to see
exactly what would be sent without sending it.

    python tools/render_diagrams.py --check
    python tools/render_diagrams.py
"""
import argparse
import base64
from pathlib import Path

import requests

SRC = Path("docs/diagrams")
OUT = Path("ppt_assets")
BASE = "https://mermaid.ink/img"
#: Rendered wider than any slide so the deck downscales rather than upscales --
#: text in an upscaled raster is the thing a projector makes unreadable.
WIDTH = 2400
TIMEOUT = 45


def encode(text: str) -> str:
    """mermaid.ink takes the graph as URL-safe base64 of the source."""
    return base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii")


def render(path: Path, dry: bool) -> bool:
    text = path.read_text(encoding="utf-8")
    # Comments are for the reader of the file, not the renderer.
    body = "\n".join(ln for ln in text.splitlines() if not ln.startswith("%%"))
    url = f"{BASE}/{encode(body)}?type=png&width={WIDTH}&bgColor=ffffff"

    if dry:
        print(f"\n--- {path.name} -> {len(body)} chars would be sent ---")
        print(body[:400] + ("..." if len(body) > 400 else ""))
        return True

    try:
        r = requests.get(url, timeout=TIMEOUT)
    except requests.RequestException as exc:
        print(f"  {path.name}: request failed -- {exc}")
        return False
    if r.status_code != 200 or not r.content.startswith(b"\x89PNG"):
        print(f"  {path.name}: HTTP {r.status_code}, "
              f"{len(r.content)} bytes, not a PNG")
        print(f"  {r.content[:200]!r}")
        return False

    dest = OUT / f"{path.stem}.png"
    dest.write_bytes(r.content)
    from PIL import Image

    with Image.open(dest) as im:
        print(f"  {path.name} -> {dest.name}  {im.width}x{im.height}  "
              f"{dest.stat().st_size / 1000:.0f} kB")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="print what would be sent, send nothing")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    sources = sorted(SRC.glob("*.mmd"))
    if not sources:
        print(f"no .mmd files in {SRC}")
        return 1
    # Materialised deliberately: every diagram must be attempted, and
    # all() on a generator would stop at the first failure.
    results = [render(p, args.check) for p in sources]
    ok = all(results)
    if not args.check:
        print("\nall rendered" if ok else "\nsome diagrams failed")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
