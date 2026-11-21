#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Append fresh GFN2 single-point scores for one panel arm."""

import argparse
import csv
import datetime
import hashlib
import re
import shlex
import socket
import subprocess
from pathlib import Path

HARTREE_EV = 27.211386245988
ETEMP_K = 3000
ACCURACY = 0.01
MAX_ITERATIONS = 500
FIELDS = [
    "system",
    "code",
    "seed",
    "geometry_sha256",
    "result",
    "fresh_energy_Eh",
    "fresh_energy_eV",
    "reference_eV",
    "hit",
    "etemp_K",
    "accuracy",
    "max_iterations",
    "xtb_sha256",
    "code_revision",
    "host",
    "date",
    "note",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unavailable(path: Path | None) -> dict[int, str]:
    if path is None:
        return {}
    with path.open(newline="") as handle:
        return {int(row["seed"]): row["reason"] for row in csv.DictReader(handle)}


def valid_xyz(path: Path) -> bool:
    try:
        return int(path.read_text(errors="ignore").splitlines()[0]) > 0
    except (IndexError, ValueError):
        return False


def prior_keys(path: Path) -> set[tuple[str, str, str, str]]:
    if not path.is_file():
        return set()
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if rows and list(rows[0]) != FIELDS:
        raise SystemExit(f"{path} has another header")
    return {
        (row["system"], row["code"], row["code_revision"], row["seed"])
        for row in rows
    }


def append(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        if new:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xtb", type=Path, required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--code", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--geometry-template", required=True)
    parser.add_argument("--unavailable", type=Path)
    parser.add_argument("--reference", type=float, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--revision")
    parser.add_argument("--host", default=socket.gethostname())
    args = parser.parse_args()

    revision = args.revision or subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    missing = unavailable(args.unavailable)
    seen = prior_keys(args.out)
    xtb_hash = sha256(args.xtb)
    rows = []
    for seed in args.seeds:
        key = (args.system, args.code, revision, str(seed))
        if key in seen:
            raise SystemExit(f"{key} is already in {args.out}")
        geometry = Path(args.geometry_template.format(seed=seed))
        where = args.work / args.system / args.code / str(seed)
        where.mkdir(parents=True, exist_ok=True)
        common: dict[str, object] = {
            "system": args.system,
            "code": args.code,
            "seed": seed,
            "geometry_sha256": "",
            "fresh_energy_Eh": "",
            "fresh_energy_eV": "",
            "reference_eV": f"{args.reference:.8f}",
            "hit": "false",
            "etemp_K": ETEMP_K,
            "accuracy": ACCURACY,
            "max_iterations": MAX_ITERATIONS,
            "xtb_sha256": xtb_hash,
            "code_revision": revision,
            "host": args.host,
            "date": datetime.date.today().isoformat(),
            "note": "",
        }
        if seed in missing:
            rows.append(dict(common, result="unavailable", note=missing[seed]))
            continue
        if not geometry.is_file():
            rows.append(dict(common, result="no_geometry", note=str(geometry)))
            continue
        common["geometry_sha256"] = sha256(geometry)
        if not valid_xyz(geometry):
            rows.append(dict(common, result="invalid_geometry", note="zero-atom or malformed XYZ"))
            continue
        target = where / "best.xyz"
        target.write_bytes(geometry.read_bytes())
        command = [
            str(args.xtb),
            target.name,
            "--gfn",
            "2",
            "--sp",
            "--etemp",
            str(ETEMP_K),
            "--acc",
            str(ACCURACY),
            "--iterations",
            str(MAX_ITERATIONS),
        ]
        (where / "command.txt").write_text(shlex.join(command) + "\n")
        process = subprocess.run(command, cwd=where, capture_output=True, text=True, check=False)
        (where / "stdout.txt").write_text(process.stdout)
        (where / "stderr.txt").write_text(process.stderr)
        (where / "exit.code").write_text(f"{process.returncode}\n")
        found = re.findall(r"TOTAL ENERGY\s+([-+]?\d+\.\d+)\s+Eh", process.stdout)
        if process.returncode != 0 or not found:
            rows.append(dict(common, result="scc_failed"))
            continue
        energy_eh = float(found[-1])
        energy_ev = energy_eh * HARTREE_EV
        rows.append(
            dict(
                common,
                result="converged",
                fresh_energy_Eh=f"{energy_eh:.12f}",
                fresh_energy_eV=f"{energy_ev:.8f}",
                hit=str(energy_ev <= args.reference + 0.001).lower(),
            )
        )
    append(args.out, rows)


if __name__ == "__main__":
    main()
