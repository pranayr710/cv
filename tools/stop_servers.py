"""Stop any ClassGraph server still holding the camera.

A session that ends on its own, seconds after starting, is almost always
another server still running from an earlier attempt. Only one process can hold
a webcam, so the second one reads no frames and stops -- with nothing on screen
saying why.

Ctrl+C in the terminal does not reliably end these on Windows, and `pkill` does
not act on Windows processes at all, so they accumulate quietly across a
session of testing. This finds them by command line and ends them.

    python tools/stop_servers.py          # list what is running
    python tools/stop_servers.py --kill   # end them
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

MARKERS = ("tools.server", "tools/server.py", "tools\\server.py")


def running_servers() -> list[tuple[int, str]]:
    """Every python process whose command line looks like this server."""
    if os.name != "nt":
        out = subprocess.run(["ps", "-eo", "pid,args"], capture_output=True,
                             text=True, check=False).stdout
        rows = []
        for line in out.splitlines()[1:]:
            pid, _, args = line.strip().partition(" ")
            if any(m in args for m in MARKERS) and pid.isdigit():
                rows.append((int(pid), args.strip()))
        return rows

    # Win32_Process carries the command line; tasklist does not.
    script = (
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        "ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" }")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", script],
                         capture_output=True, text=True, check=False).stdout
    rows = []
    for line in out.splitlines():
        pid, _, args = line.partition("\t")
        if pid.strip().isdigit() and any(m in args for m in MARKERS):
            rows.append((int(pid.strip()), args.strip()))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kill", action="store_true",
                    help="end them (default: only report)")
    args = ap.parse_args()

    servers = running_servers()
    mine = os.getpid()
    servers = [(pid, cmd) for pid, cmd in servers if pid != mine]

    if not servers:
        print("No server processes running. The camera is free.")
        return 0

    print(f"{len(servers)} server process(es) holding the camera:")
    for pid, cmd in servers:
        print(f"  {pid}  {cmd[:88]}")

    if not args.kill:
        print("\nRe-run with --kill to end them.")
        return 0

    for pid, _ in servers:
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                               capture_output=True, check=False)
            else:
                os.kill(pid, 9)
            print(f"  stopped {pid}")
        except OSError as exc:
            print(f"  could not stop {pid}: {exc}", file=sys.stderr)

    left = [p for p, _ in running_servers() if p != mine]
    print(f"\n{len(left)} remaining. The camera is "
          f"{'free' if not left else 'still held'}.")
    return 0 if not left else 1


if __name__ == "__main__":
    raise SystemExit(main())
