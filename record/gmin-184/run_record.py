#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Serial anneal against GMIN for the five-system record.

Seeds 2000-2011, paper call budgets, one core, one thread. Even seeds
run anneal then GMIN and odd seeds run GMIN then anneal. Rows are
appended and never rewritten. A stored seed is left in place.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEEDS = tuple(range(2000, 2012))
HIT_TOL = 1.0e-3
ANNEAL_COMMIT = "9837358ae1027067ff654fac8b48969020853452"
# name, kind, atoms, reference, rho, budget, binary
SYSTEMS = (
    ("au55", "gupta", 55, -191.03085280, "", 1_000_000, "h2h_metal"),
    ("m55r6", "morse", 55, -250.286609, "6", 1_000_000, "h2h_morse"),
    ("m55r14", "morse", 55, -220.646208, "14", 1_000_000, "h2h_morse"),
    ("m80r6", "morse", 80, -378.333471, "6", 4_000_000, "h2h_morse"),
    ("m80r14", "morse", 80, -340.811371, "14", 4_000_000, "h2h_morse"),
)
ROW_FIELDS = [
    "arm",
    "system",
    "rho",
    "seed",
    "budget",
    "calls",
    "evals_to_hit",
    "best",
    "hit",
    "wall_s",
    "stop_reason",
    "anneal_commit",
    "anneal_tree",
    "gmin_tarball_date",
    "gmin_tarball_sha256",
    "machine_id",
]
THREAD_KEYS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def spec(name: str) -> tuple:
    for row in SYSTEMS:
        if row[0] == name:
            return row
    raise SystemExit(f"unknown system {name}")


def type7(values: list[float], prob: float) -> float:
    ordered = sorted(values)
    count = len(ordered)
    if count < 1:
        raise ValueError("empty sample")
    if count == 1:
        return ordered[0]
    position = 1.0 + (count - 1) * prob
    lower = int(position)
    frac = position - lower
    if frac == 0.0:
        return ordered[lower - 1]
    return ordered[lower - 1] * (1.0 - frac) + ordered[lower] * frac


def machine_id() -> str:
    raw = Path("/etc/machine-id").read_bytes().strip()
    return hashlib.sha256(raw).hexdigest()


def thread_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("H2H_", "ANNEAL_")) or key.endswith("_NUM_THREADS"):
            del env[key]
    for key in THREAD_KEYS:
        env[key] = "1"
    env["OMP_THREAD_LIMIT"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    return env


def load_identity(path: Path) -> tuple[str, str]:
    data: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" not in line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    date = data.get("date", "")
    digest = data.get("sha256", "")
    if len(date) < 10 or len(digest) != 64:
        raise SystemExit(f"tarball record is incomplete: {path}")
    return date, digest


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def arms_for(seed: int) -> tuple[str, str]:
    if (seed - SEEDS[0]) % 2 == 0:
        return ("anneal", "gmin")
    return ("gmin", "anneal")


def check_inputs(inputs: Path) -> None:
    seen: dict[str, str] = {}
    for name, kind, atoms, _reference, rho, _budget, _binary in SYSTEMS:
        path = inputs / f"{name}.data"
        text = path.read_text()
        if "\r" in text:
            raise SystemExit(f"{path.name} contains CR")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if f"! atoms {atoms}" not in lines:
            raise SystemExit(f"{name} data does not state {atoms} atoms")
        if any(line.startswith("RANSEED") for line in lines):
            raise SystemExit(f"{name} data already has a seed")
        if kind == "gupta":
            head = next(line for line in lines if line.startswith("BGUPTAT "))
            if not head.startswith(f"BGUPTAT {atoms} "):
                raise SystemExit(f"{name} data has the wrong cluster size")
            if any(line.startswith("MORSE") for line in lines):
                raise SystemExit(f"{name} data carries a Morse rho")
        else:
            morse = [line for line in lines if line.startswith("MORSE ")]
            if morse != [f"MORSE {float(rho):.1f}"]:
                raise SystemExit(f"{name} data has rho {morse}, expected {rho}")
            if any(line.startswith("BGUPTAT") for line in lines):
                raise SystemExit(f"{name} data carries a Gupta size")
        seen[name] = text
    if len({seen[name] for name in seen}) != len(seen):
        raise SystemExit("two systems share one data file")


def write_log(path: Path, text: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise RuntimeError(f"log readback mismatch for {path.name}")
    return digest


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ROW_FIELDS:
            raise SystemExit(f"rows header is {reader.fieldnames}")
        return list(reader)


def append_row(path: Path, row: dict[str, str]) -> None:
    fresh = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS, lineterminator="\n")
        if fresh:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def git_output(tree: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(tree), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def run_checked(command: list[str], cwd: Path, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, env=env, capture_output=True, text=True, check=False)


def parse_anneal(text: str, system: str, seed: int, budget: int) -> dict[str, str]:
    header = None
    data = None
    for line in text.splitlines():
        if line.startswith("system,code,seed,"):
            header = line
        elif f",anneal,{seed}," in line and line.startswith(f"{system},"):
            data = line
    if header is None or data is None:
        raise RuntimeError(f"anneal {system} {seed} produced no row")
    row = dict(zip(header.split(","), data.split(",")))
    if row.get("system") != system or int(row["seed"]) != seed or int(row["budget"]) != budget:
        raise RuntimeError(f"anneal {system} {seed} row does not match the request")
    if row.get("code") != "anneal":
        raise RuntimeError(f"anneal {system} {seed} row is not the anneal arm")
    return row


def parse_gmin(text: str, system: str, seed: int, budget: int) -> dict[str, str]:
    lines = [line for line in text.splitlines() if line.startswith("h2h_row ")]
    if len(lines) != 1:
        raise RuntimeError(f"gmin {system} {seed} produced {len(lines)} rows")
    fields = dict(part.split("=", 1) for part in lines[0].split()[1:] if "=" in part)
    if fields.get("system") != system or int(fields["seed"]) != seed or int(fields["budget"]) != budget:
        raise RuntimeError(f"gmin {system} {seed} row does not match the request")
    if fields.get("code") != "gmin":
        raise RuntimeError(f"gmin {system} {seed} row is not the GMIN arm")
    return fields


def accept(system: str, raw: dict[str, str], exit_code: int) -> dict[str, str]:
    _name, _kind, _atoms, reference, _rho, budget, _binary = spec(system)
    best = float(raw["best"])
    calls = int(raw["charged"])
    evals = int(raw["evals_to_hit"])
    wall = float(raw["wall_s"])
    hit_flag = raw["hit"].strip().lower() in ("1", "true")
    measured = best <= reference + HIT_TOL
    if hit_flag != measured:
        raise RuntimeError(f"{system} hit flag {raw['hit']} disagrees with energy {best}")
    if calls <= 0:
        raise RuntimeError(f"{system} reports no calls")
    if hit_flag and not (0 < evals <= calls <= budget):
        raise RuntimeError(f"{system} hit calls {evals} charged {calls}")
    if not hit_flag and (evals != budget or calls != budget):
        raise RuntimeError(f"{system} miss calls {evals} charged {calls}")
    if wall < 0.0:
        raise RuntimeError(f"{system} wall {wall}")
    if exit_code != 0 and not hit_flag and calls != budget:
        raise RuntimeError(f"{system} exited {exit_code} before the budget")
    return {
        "calls": str(calls),
        "evals_to_hit": str(evals),
        "best": raw["best"],
        "hit": "1" if hit_flag else "0",
        "wall_s": f"{wall:.6f}",
        "stop_reason": "hit" if hit_flag else "budget",
    }


def coords_ok(path: Path, atoms: int) -> None:
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    if len(lines) != atoms:
        raise RuntimeError(f"coords has {len(lines)} atoms, expected {atoms}")
    for line in lines:
        parts = line.split()
        if len(parts) != 3:
            raise RuntimeError("coords line is not three coordinates")
        for part in parts:
            float(part)


def data_for_run(template: str, seed: int) -> str:
    body = template if template.endswith("\n") else template + "\n"
    return body + f"RANSEED {seed}\n"


def binary_budget(binary: Path, tree: Path, system: str) -> None:
    _name, kind, atoms, reference, rho, budget, _binary_name = spec(system)
    proc = run_checked(["taskset", "-c", "0", str(binary), "budget", system], tree, thread_env())
    if proc.returncode != 0:
        raise SystemExit(f"{system} budget probe failed\n{proc.stderr}")
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if int(lines[0]) != budget or abs(float(lines[1]) - reference) > 1e-8:
        raise SystemExit(f"{system} binary budget {lines[:2]} is not the paper budget")
    if kind == "morse" and abs(float(lines[2]) - float(rho)) > 1e-8:
        raise SystemExit(f"{system} binary rho {lines[2]} is not {rho}")
    if atoms <= 0:
        raise SystemExit(system)


def run_anneal(binary: Path, tree: Path, system: str, seed: int, budget: int, log: Path) -> dict[str, str]:
    command = ["taskset", "-c", "0", str(binary), "run", system, str(seed)]
    proc = run_checked(command, tree, thread_env())
    text = proc.stdout + proc.stderr
    digest = write_log(log, f"arm=anneal\nsystem={system}\nseed={seed}\nexit={proc.returncode}\n" + text)
    if proc.returncode != 0:
        raise RuntimeError(f"anneal {system} {seed} exited {proc.returncode}")
    parsed = accept(system, parse_anneal(proc.stdout, system, seed, budget), proc.returncode)
    parsed["log_sha256"] = digest
    return parsed


def run_gmin(
    binary: Path,
    anneal: Path,
    tree: Path,
    system: str,
    seed: int,
    budget: int,
    work: Path,
    log: Path,
    template: str,
) -> dict[str, str]:
    _name, _kind, atoms, reference, _rho, _budget, _binary_name = spec(system)
    work.mkdir(parents=True, exist_ok=True)
    coords = work / "coords"
    proc = run_checked(
        ["taskset", "-c", "0", str(anneal), "coords", system, str(seed), str(coords)],
        tree,
        thread_env(),
    )
    if proc.returncode != 0 or not coords.is_file():
        raise RuntimeError(f"coords {system} {seed} exited {proc.returncode}\n{proc.stderr}")
    coords_ok(coords, atoms)
    (work / "data").write_text(data_for_run(template, seed))
    row_path = work / "row.txt"
    env = thread_env()
    env.update(
        {
            "H2H_BUDGET": str(budget),
            "H2H_SYSTEM": system,
            "H2H_SEED": str(seed),
            "H2H_TARGET": f"{reference:.10f}",
            "H2H_TOL": f"{HIT_TOL}",
            "H2H_ROW": str(row_path),
        }
    )
    proc = run_checked(["taskset", "-c", "0", str(binary)], work, env)
    row_text = row_path.read_text() if row_path.is_file() else ""
    digest = write_log(
        log,
        f"arm=gmin\nsystem={system}\nseed={seed}\nexit={proc.returncode}\n" + row_text + proc.stdout + proc.stderr,
    )
    if not row_text:
        raise RuntimeError(f"gmin {system} {seed} exited {proc.returncode} with no row")
    parsed = accept(system, parse_gmin(row_text, system, seed, budget), proc.returncode)
    parsed["log_sha256"] = digest
    return parsed


def summarise(rows: list[dict[str, str]]) -> list[str]:
    lines = [
        "system arm hits n median_calls",
    ]
    for name, _kind, _atoms, _reference, _rho, _budget, _binary in SYSTEMS:
        for arm in ("anneal", "gmin"):
            group = [row for row in rows if row["system"] == name and row["arm"] == arm]
            if not group:
                continue
            hits = sum(row["hit"] == "1" for row in group)
            median = type7([float(row["evals_to_hit"]) for row in group], 0.5)
            shown = f"{median:.2f}".rstrip("0").rstrip(".")
            lines.append(f"{name} {arm} {hits} {len(group)} {shown}")
    return lines


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--identity", type=Path, required=True)
    parser.add_argument("--tarball", type=Path)
    parser.add_argument("--tree", type=Path)
    parser.add_argument("--anneal-bin", type=Path)
    parser.add_argument("--gmin", type=Path)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--logs", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--summarise", action="store_true")
    args = parser.parse_args(argv[1:])
    check_inputs(args.inputs)
    date, digest = load_identity(args.identity)
    if args.tarball is not None and file_sha256(args.tarball) != digest:
        raise SystemExit("tarball file hash does not match the recorded file")
    if args.check:
        print(f"inputs ok date={date}")
        return 0
    if args.summarise:
        for line in summarise(read_rows(args.rows)):
            print(line)
        print(f"rows {len(read_rows(args.rows))}")
        print(f"tarball_date {date}")
        print(f"tarball_sha256 {digest}")
        return 0
    needed = (args.tree, args.anneal_bin, args.gmin, args.work, args.logs)
    if any(value is None for value in needed):
        parser.error("--tree, --anneal-bin, --gmin, --work and --logs are required to run")
    assert args.tree is not None and args.anneal_bin is not None
    assert args.gmin is not None and args.work is not None and args.logs is not None
    head = git_output(args.tree, "rev-parse", "HEAD")
    tree = git_output(args.tree, "rev-parse", "HEAD^{tree}")
    if head != ANNEAL_COMMIT:
        raise SystemExit(f"anneal source is {head}")
    if git_output(args.tree, "status", "--porcelain"):
        raise SystemExit("anneal source tree is dirty")
    for name, _kind, _atoms, _reference, _rho, _budget, binary in SYSTEMS:
        binary_budget(args.anneal_bin / binary, args.tree, name)
    done = {(row["system"], row["arm"], row["seed"]) for row in read_rows(args.rows)}
    identity = machine_id()
    templates = {name: (args.inputs / f"{name}.data").read_text() for name, *_rest in SYSTEMS}
    print(f"start commit={head} tree={tree} tarball_date={date}", flush=True)
    for name, _kind, _atoms, _reference, rho, budget, binary in SYSTEMS:
        for seed in SEEDS:
            for arm in arms_for(seed):
                if (name, arm, str(seed)) in done:
                    print(f"skip {name} {arm} {seed}", flush=True)
                    continue
                print(f"start {name} {arm} {seed}", flush=True)
                log = args.logs / name / f"{arm}-{seed}.log"
                if arm == "anneal":
                    result = run_anneal(args.anneal_bin / binary, args.tree, name, seed, budget, log)
                else:
                    result = run_gmin(
                        args.gmin,
                        args.anneal_bin / binary,
                        args.tree,
                        name,
                        seed,
                        budget,
                        args.work / name / str(seed),
                        log,
                        templates[name],
                    )
                if not result["stop_reason"] or int(result["calls"]) <= 0:
                    raise SystemExit(f"{name} {arm} {seed} has no calls or no stop reason")
                log_sha = result.pop("log_sha256")
                row = {
                    "arm": arm,
                    "system": name,
                    "rho": rho,
                    "seed": str(seed),
                    "budget": str(budget),
                    "anneal_commit": head,
                    "anneal_tree": tree,
                    "gmin_tarball_date": date,
                    "gmin_tarball_sha256": digest,
                    "machine_id": identity,
                    **result,
                }
                append_row(args.rows, row)
                done.add((name, arm, str(seed)))
                print(
                    f"done {name} {arm} {seed} hit={row['hit']} calls={row['calls']} "
                    f"wall_s={row['wall_s']} stop={row['stop_reason']} log_sha256={log_sha} "
                    f"rows={len(done)}",
                    flush=True,
                )
    print(f"complete rows={len(done)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
