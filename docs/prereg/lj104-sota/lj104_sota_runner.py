#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Serial gate for the pre-registered LJ104 comparison.

A scored start takes an exclusive lock and will not start while another
scored start holds it. Every row this process writes cites the git blob
of this file, HEAD, and the tree of HEAD. Seeds 5000-5029 are refused.
The arm B v2 preset id and the arm D v3 tip are UNFILLED, so this commit
does not dispatch a search.
"""

from __future__ import annotations

import argparse
import fcntl
import subprocess
import sys
import time
from pathlib import Path

HELD_OUT_LO = 5000
HELD_OUT_HI = 5029
PRESET_ID = "UNFILLED"
D_V3_TIP = "UNFILLED"
COLUMNS = (
    "runner_blob",
    "head",
    "tree",
    "seed",
    "code",
    "hit",
    "energy",
    "calls",
)


def script_path() -> Path:
    return Path(__file__).resolve()


def repo_root() -> Path:
    here = script_path().parent
    for candidate in (here, *here.parents):
        if (candidate / ".git").exists():
            return candidate
    print("runner is not inside a git work tree", file=sys.stderr)
    raise SystemExit(1)


def git_out(*args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo_root()), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(proc.stderr.strip() or "git failed", file=sys.stderr)
        raise SystemExit(1)
    return proc.stdout.strip()


def provenance() -> dict[str, str]:
    """Git blob of this runner, HEAD, and the tree of HEAD."""
    fields = {
        "runner_blob": git_out("hash-object", "--", str(script_path())),
        "head": git_out("rev-parse", "HEAD"),
        "tree": git_out("rev-parse", "HEAD^{tree}"),
    }
    missing = [name for name in ("runner_blob", "head", "tree") if not fields[name]]
    if missing:
        print("scored row is missing " + ", ".join(missing), file=sys.stderr)
        raise SystemExit(1)
    return fields


def default_lock_file() -> Path:
    return script_path().parent / ".scored-run.lock"


def acquire_scored_run(lock_file: Path):
    """Exclusive non-blocking lock. Exit 2 when another scored run is live."""
    lock_file.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_file.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        print("another scored run is live", file=sys.stderr)
        raise SystemExit(2)
    return handle


def release_scored_run(handle) -> None:
    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    handle.close()


def iter_seeds(spec: str):
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_text, hi_text = part.split("-", 1)
            lo, hi = int(lo_text), int(hi_text)
            if hi < lo:
                print(f"seed range {part} is reversed", file=sys.stderr)
                raise SystemExit(2)
            yield from range(lo, hi + 1)
        else:
            yield int(part)


def touches_held_out(spec: str) -> bool:
    return any(HELD_OUT_LO <= seed <= HELD_OUT_HI for seed in iter_seeds(spec))


def write_row(path: Path, fields: dict[str, str]) -> None:
    for name in ("runner_blob", "head", "tree"):
        if not fields.get(name):
            print("scored row is missing " + name, file=sys.stderr)
            raise SystemExit(1)
    header = "\t".join(COLUMNS)
    body = "\t".join(fields.get(column, "") for column in COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + "\n" + body + "\n", encoding="utf-8")


def record_serial_boundary(dest: Path, label: str) -> None:
    """Start or end receipt for one strictly serial process.

    The receipt is the kernel boot id, the uptime reading, and lscpu.
    A scored run keeps the start receipt and the end receipt. Two
    different boot ids mean the process did not stay on one boot.
    """
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    uptime = Path("/proc/uptime").read_text(encoding="utf-8").strip()
    listing = subprocess.run(
        ["lscpu"],
        check=False,
        capture_output=True,
        text=True,
    )
    text = listing.stdout if listing.returncode == 0 else listing.stderr
    dest.mkdir(parents=True, exist_ok=True)
    (dest / f"{label}.receipt").write_text(
        "boot_id " + boot_id + "\n" + "uptime " + uptime + "\n" + text,
        encoding="utf-8",
    )


def cmd_hold(args: argparse.Namespace) -> int:
    handle = acquire_scored_run(Path(args.lock_file))
    try:
        print("holding", flush=True)
        time.sleep(args.seconds)
    finally:
        release_scored_run(handle)
    return 0


def cmd_emit_row(args: argparse.Namespace) -> int:
    handle = acquire_scored_run(Path(args.lock_file))
    try:
        fields = provenance()
        fields.update(
            {
                "seed": "unscored",
                "code": "schema",
                "hit": "unscored",
                "energy": "",
                "calls": "",
            }
        )
        write_row(Path(args.output), fields)
    finally:
        release_scored_run(handle)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    if touches_held_out(args.seeds):
        print("seeds 5000-5029 are not run from this commit", file=sys.stderr)
        return 3
    if PRESET_ID == "UNFILLED" or D_V3_TIP == "UNFILLED":
        print(
            "preset id or D v3 tip is UNFILLED; no scored search is started",
            file=sys.stderr,
        )
        return 3
    # The identifiers are filled only by a later commit, and that commit
    # still has to keep the lock and the two receipts around the search.
    handle = acquire_scored_run(Path(args.lock_file))
    try:
        receipt_dir = Path(args.receipt_dir)
        record_serial_boundary(receipt_dir, "start")
        print("this commit does not dispatch a scored search", file=sys.stderr)
        record_serial_boundary(receipt_dir, "end")
        return 3
    finally:
        release_scored_run(handle)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock-file", default=str(default_lock_file()))
    sub = parser.add_subparsers(dest="command", required=True)

    hold = sub.add_parser("hold", help="hold the scored-run lock")
    hold.add_argument("--seconds", type=float, required=True)
    hold.set_defaults(func=cmd_hold)

    emit = sub.add_parser("emit-row", help="write one provenance row")
    emit.add_argument("--output", required=True)
    emit.set_defaults(func=cmd_emit_row)

    run = sub.add_parser("run", help="refused until the pre-registration is filled")
    run.add_argument("--code", required=True)
    run.add_argument("--seeds", required=True)
    run.add_argument("--receipt-dir", default="lj104-sota-receipts")
    run.set_defaults(func=cmd_run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
