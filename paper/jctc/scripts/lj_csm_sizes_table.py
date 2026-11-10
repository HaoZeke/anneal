#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Read LJ38, LJ55 and LJ98 of the hop from the sealed boards of the lj_csm campaign.

Reads rewrite_2026/data/lj_csm_sizes/: audit/ copies, from results_lj_csm/ of the
commit of anneal_repro that sealed them, the verdict, its sums and its PROVENANCE, and
the PROVENANCE, SHA256SUMS and summary.json of the recommended and derived board of each
size. provenance.txt records the digest of those sums, the conditions of each size, the
hits and the seeds of each board and whether the rerun matched the handoff seed for seed,
and the cell of fig:summary each board fills.

A board is reproduced when the verdict finds no scientific mismatch between the rerun and
the handoff, seed for seed, and the rerun gives the handoff's hits and call PAR-2. A cap
that differs does not undo that: the handoff of LJ55 ran at 400000 and the rerun at
1000000, and every run of its recommended preset hit, at the same count, below both. The script stops when a file
of audit/ is not the one the audit sealed, when a board ran another cluster, cap,
reference, tolerance, anneal commit or host than provenance.txt records, when its summary
and the verdict disagree, when provenance.txt gives a board other hits, seeds or another
grade than the seal, or when it draws a cell from a board that was not reproduced.

Prints, one line a board, the hits, the seeds and the grade, and then each cell.

Usage: lj_csm_sizes_table.py [data-dir]
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "lj_csm_sizes"
AUDIT = "audit"
SIZES = (38, 55, 98)
PRESETS = ("recommended", "derived")
BOARD_FILES = ("PROVENANCE", "summary.json")
AUDIT_FILES = ("PROVENANCE", "verdict.json")
FIRST_SEED = 2000
REPRODUCED = "reproduced"


def die(message: str) -> NoReturn:
    raise SystemExit(f"lj_csm_sizes_table: {message}")


def digest_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sums(path: Path) -> dict[str, str]:
    return {name.removeprefix("./"): digest for digest, name in (line.split("  ", 1) for line in path.read_text().splitlines())}


def _keyed(path: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text().splitlines() if "=" in line)


def provenance(data: Path) -> tuple[dict[str, str], dict[str, str]]:
    """The keyed lines of provenance.txt, and the digest it records for each file."""
    keys: dict[str, str] = {}
    digests: dict[str, str] = {}
    for line in (data / "provenance.txt").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        first, _, rest = line.partition(" ")
        if len(first) == 64 and all(c in "0123456789abcdef" for c in first):
            digests[rest.strip()] = first
        else:
            keys[first] = rest.strip()
    return keys, digests


@dataclass(frozen=True)
class Board:
    name: str
    size: int
    preset: str
    hits: int
    seeds: int
    budget: int
    reproduced: bool


@dataclass(frozen=True)
class Cell:
    system: str
    method: str
    board: Board


def _board(root: Path, name: str, size: int, keys: dict[str, str], sealed: dict[str, str], verdict: dict) -> Board:
    where = f"{AUDIT}/{name}"
    if digest_of(root / name / "SHA256SUMS") != sealed.get(f"{name}/SHA256SUMS"):
        die(f"{where}/SHA256SUMS does not match the digest the audit sealed")
    inner = _sums(root / name / "SHA256SUMS")
    for file in BOARD_FILES:
        if digest_of(root / name / file) != inner.get(file) or inner[file] != sealed.get(f"{name}/{file}"):
            die(f"{where}/{file} does not match the digest the audit sealed")
    summary = json.loads((root / name / "summary.json").read_text())
    record = _keyed(root / name / "PROVENANCE")
    saddle = summary["saddle"]
    budget = int(keys[f"budget_{size}"])
    host = f"Intel Xeon family {summary['cpu_family']} model {summary['cpu_model']}, {summary['hypervisor']}"
    checks = {
        "the board": summary["name"] == record["name"] == name,
        "the cluster": summary["n_atoms"] == size == int(record["n_atoms"]),
        "the cap": summary["budget"] == budget == int(record["budget"]),
        "the reference": summary["reference"] == float(keys[f"reference_{size}"]) == float(record["reference"]),
        "the tolerance": summary["tolerance"] == float(keys["tolerance"])
        and record["hit_rule"] == f"best<=reference+{keys['tolerance']}",
        "the seeds": (summary["first"], summary["last"]) == (FIRST_SEED, FIRST_SEED + summary["n_seeds"] - 1),
        "the anneal commit": summary["anneal_commit"] == record["anneal_commit"] == keys["anneal_commit"],
        "the host": host == keys["host"] and summary["cpu_model"] != summary["paper_cpu_model"],
        "the verdict": {k: verdict[name][k] for k in ("hits", "n_seeds", "call_par2")}
        == {k: summary[k] for k in ("hits", "n_seeds", "call_par2")},
        "the comparison": saddle["compared"] is True and saddle["saddle_rows"] == summary["n_seeds"],
    }
    failed = [what for what, held in checks.items() if not held]
    if failed:
        die(f"{where} does not give {failed[0]} provenance.txt records")
    reproduced = (
        not saddle["scientific_mismatches"]
        and verdict[name]["scientific_mismatches"] == 0
        and saddle["saddle_hits"] == summary["hits"]
        and saddle["saddle_call_par2"] == summary["call_par2"]
    )
    return Board(name, size, summary["arm"], summary["hits"], summary["n_seeds"], budget, reproduced)


def load(data: Path = DATA) -> tuple[dict[str, str], tuple[Board, ...], tuple[Cell, ...]]:
    keys, digests = provenance(data)
    root = data / AUDIT
    name = f"{AUDIT}/SHA256SUMS"
    if digest_of(root / "SHA256SUMS") != digests.get(name):
        die(f"{name} does not match the digest provenance.txt records")
    sealed = _sums(root / "SHA256SUMS")
    for file in AUDIT_FILES:
        if digest_of(root / file) != sealed.get(file):
            die(f"{AUDIT}/{file} does not match the digest the audit sealed")
    if _keyed(root / "PROVENANCE")["anneal_commit"] != keys["anneal_commit"]:
        die("the audit ran another anneal commit than provenance.txt records")
    verdict = {board["name"]: board for board in json.loads((root / "verdict.json").read_text())["boards"]}
    boards = []
    for size in SIZES:
        for preset in PRESETS:
            board = _board(root, f"lj{size}-{preset}", size, keys, sealed, verdict)
            if board.preset != preset:
                die(f"{AUDIT}/{board.name} ran the {board.preset} preset")
            recorded = keys.get(f"board_{board.name}", "")
            grade = REPRODUCED if board.reproduced else f"not {REPRODUCED}"
            if recorded.split(":")[0] != f"{board.hits}/{board.seeds} {grade}":
                die(f"provenance.txt gives {board.name} as {recorded!r}, not {board.hits}/{board.seeds} {grade}")
            boards.append(board)
    by_name = {board.name: board for board in boards}
    cells = []
    for key, value in keys.items():
        if not key.startswith("cell_"):
            continue
        method, board_name = value.split()
        board = by_name.get(board_name)
        if board is None or f"LJ{board.size}" != key.removeprefix("cell_"):
            die(f"provenance.txt draws {key} from {board_name}, which is no board of that size")
        if not board.reproduced:
            die(f"provenance.txt draws {key} from {board_name}, which the rerun did not reproduce")
        cells.append(Cell(f"LJ{board.size}", method, board))
    return keys, tuple(boards), tuple(cells)


def main(argv: list[str]) -> None:
    _, boards, cells = load(Path(argv[0]) if argv else DATA)
    for board in boards:
        grade = REPRODUCED if board.reproduced else f"not {REPRODUCED}"
        print(f"{board.name} {board.hits}/{board.seeds} at {board.budget} {grade}")
    for cell in cells:
        print(f"{cell.system} {cell.method} {cell.board.hits}/{cell.board.seeds} from {cell.board.name}")


if __name__ == "__main__":
    main(sys.argv[1:])
