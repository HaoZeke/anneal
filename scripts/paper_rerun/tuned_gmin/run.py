#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Fresh measurement of the printed per-cluster GMIN cells.

Reads ``manifest.csv``. Rows with status ``run`` execute at
``scripts/gmin_panel/run_panel.py`` ``PAPER_BUDGETS``, seeds
``seed_first`` through ``seed_last``, both arms, ABBA within each seed.
Status ``unrunnable`` and ``not_tuned`` are not executed.

A hit is ``scripts/target_tol.py`` ``energy_hits``: best <= reference + 0.001.
GMIN keywords come from ``run_panel.gmin_data`` and the committed data
files that function reads. Anneal is ``h2h_metal`` or ``h2h_morse`` at
the base commit named below. ``code_revision`` on an anneal row is that
base; on a GMIN row it is the pin ``run_panel.GMIN_PIN``.

One process at a time. The process must already be pinned to core 0.
Each child is started under ``taskset -c 0``. Thread variables are set
to 1. A stored row for the same system, code, seed, code revision and
harness revision is kept. A failed child exits the campaign without a
substitute row.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

CODE_REVISION = "61fa67cf380e46876b569da45e7cc091647a775b"
HARNESS_FILES = (
    "scripts/paper_rerun/tuned_gmin/manifest.csv",
    "scripts/paper_rerun/tuned_gmin/run.py",
    "scripts/paper_rerun/tuned_gmin/summarize.py",
)
THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
ROW_TAIL = ("utc", "harness_blob", "session")


def load_panel(root: Path):
    path = root / "scripts" / "gmin_panel" / "run_panel.py"
    spec = importlib.util.spec_from_file_location("gmin_panel_run", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row_fields(panel) -> list[str]:
    fields = [name for name in panel.ROW_FIELDS if name not in ("utc", "session")]
    return fields + list(ROW_TAIL)


def harness_blob(root: Path) -> str:
    digest = hashlib.sha256()
    for rel in HARNESS_FILES:
        payload = (root / rel).read_bytes()
        digest.update(rel.encode())
        digest.update(b"\0")
        digest.update(payload)
    return digest.hexdigest()


def host_id() -> str:
    raw = Path("/etc/machine-id").read_bytes().strip()
    return hashlib.sha256(b"machine-id\0" + raw).hexdigest()


def thread_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("H2H_", "ANNEAL_")):
            del env[key]
    env.update(THREAD_ENV)
    return env


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def append_row(path: Path, fields: list[str], row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if fresh:
            writer.writeheader()
        writer.writerow({name: row[name] for name in fields})
        handle.flush()
        os.fsync(handle.fileno())


def run_command(
    command: list[str], cwd: Path, env: dict[str, str], timeout: int, log: Path, *, require_zero: bool = True
) -> str:
    log.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        command,
        cwd=cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        log.write_text(" ".join(command) + f"\ntimeout {timeout}\n")
        raise SystemExit(f"timeout {command[3:]} after {timeout}s")
    log.write_text(" ".join(command) + f"\nexit={proc.returncode}\n" + stdout[-20000:] + stderr[-5000:])
    if require_zero and proc.returncode != 0:
        raise SystemExit(f"exit {proc.returncode}: {command}")
    return stdout


def check_ancestor(root: Path) -> str:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", CODE_REVISION, head],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        raise SystemExit(f"HEAD {head} does not contain {CODE_REVISION}")
    diff = subprocess.run(
        ["git", "diff", "--stat", CODE_REVISION, "--", "src", "examples", "Cargo.toml", "Cargo.lock"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if diff:
        raise SystemExit(f"search sources differ from {CODE_REVISION}: {diff}")
    return head


def check_budgets(panel, bin_dir: Path, root: Path, systems: list[str]) -> None:
    for system in systems:
        budget, reference = panel.binary_budget(bin_dir, root, system)
        if budget != panel.PAPER_BUDGETS[system] or abs(reference - panel.spec(system)[3]) > 1e-8:
            raise SystemExit(f"{system}: binary budget {budget} reference {reference}")


def check_gmin_source(panel, source_path: Path) -> None:
    source = dict(
        line.split("=", 1) for line in source_path.read_text().splitlines() if "=" in line
    )
    if source.get("svnrev") != panel.GMIN_PIN:
        raise SystemExit(f"GMIN SVNREV {source.get('svnrev')} is not {panel.GMIN_PIN}")
    if source.get("tarball_sha256") != panel.GMIN_TARBALL_SHA256:
        raise SystemExit("GMIN tarball sha256 does not match the pin")


def run_anneal(panel, bin_dir: Path, root: Path, system: str, seed: int, log: Path, timeout: int) -> dict[str, str]:
    _driver, binary = panel.resolve_driver(bin_dir, root, system)
    budget = panel.PAPER_BUDGETS[system]
    command = ["taskset", "-c", "0", str(bin_dir / binary), "run", system, str(seed)]
    stdout = run_command(command, root, thread_env(), timeout, log)
    return panel.anneal_result(stdout, system, seed, budget)


def run_gmin(
    panel,
    bin_dir: Path,
    gmin_bin: Path,
    root: Path,
    system: str,
    seed: int,
    work: Path,
    log: Path,
    lj_data: str,
    morse_data: str,
    timeout: int,
) -> dict[str, str]:
    _driver, binary = panel.resolve_driver(bin_dir, root, system)
    budget = panel.PAPER_BUDGETS[system]
    reference = panel.spec(system)[3]
    work.mkdir(parents=True, exist_ok=True)
    for stale in ("row.txt", "coords", "data"):
        (work / stale).unlink(missing_ok=True)
    subprocess.run(
        ["taskset", "-c", "0", str(bin_dir / binary), "coords", system, str(seed), str(work / "coords")],
        cwd=root,
        env=thread_env(),
        capture_output=True,
        text=True,
        check=True,
    )
    (work / "data").write_text(panel.gmin_data(system, seed, lj_data, morse_data))
    env = thread_env()
    env.update(
        {
            "H2H_BUDGET": str(budget),
            "H2H_SYSTEM": system,
            "H2H_SEED": str(seed),
            "H2H_TARGET": f"{reference:.10f}",
            "H2H_TOL": f"{panel.HIT_TOL}",
            "H2H_ROW": str(work / "row.txt"),
        }
    )
    run_command(["taskset", "-c", "0", str(gmin_bin)], work, env, timeout, log, require_zero=False)
    row_path = work / "row.txt"
    if not row_path.is_file():
        raise SystemExit(f"gmin {system} seed {seed} wrote no row")
    return panel.gmin_result(row_path.read_text(), system, seed, budget)


def jobs(manifest: list[dict[str, str]], panel) -> list[tuple[str, int, str]]:
    systems = []
    for row in manifest:
        if row["status"] != "run":
            continue
        if row["system"] not in systems:
            systems.append(row["system"])
    out = []
    for system in systems:
        for seed in panel.SEEDS:
            for code in panel.arm_sequence("abba", seed):
                out.append((system, seed, code))
    return out


def main(argv: list[str]) -> int:
    os.environ.update(THREAD_ENV)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--bin", type=Path, required=True)
    parser.add_argument("--gmin", type=Path, required=True)
    parser.add_argument("--gmin-source", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=21600)
    args = parser.parse_args(argv[1:])
    root = args.root.resolve()
    if os.sched_getaffinity(0) != {0}:
        raise SystemExit(f"affinity is {sorted(os.sched_getaffinity(0))}, expected core 0")
    panel = load_panel(root)
    manifest = read_manifest(root / "scripts" / "paper_rerun" / "tuned_gmin" / "manifest.csv")
    run_rows = [row for row in manifest if row["status"] == "run"]
    systems = list(dict.fromkeys(row["system"] for row in run_rows))
    for system in systems:
        if panel.PAPER_BUDGETS[system] != int(next(row["budget"] for row in run_rows if row["system"] == system)):
            raise SystemExit(f"{system} manifest budget is not PAPER_BUDGETS")
        seeds = {int(row["seed_first"]) for row in run_rows if row["system"] == system}
        ends = {int(row["seed_last"]) for row in run_rows if row["system"] == system}
        if seeds != {panel.SEEDS[0]} or ends != {panel.SEEDS[-1]}:
            raise SystemExit(f"{system} seed span is not {panel.SEEDS[0]}-{panel.SEEDS[-1]}")
    head = check_ancestor(root)
    check_gmin_source(panel, args.gmin_source)
    check_budgets(panel, args.bin, root, systems)
    lj_data = (root / "examples" / "h2h_gmin_lj.data").read_text()
    morse_data = (root / "examples" / "h2h_gmin_morse.data").read_text()
    if not lj_data.endswith("\n"):
        lj_data += "\n"
    if not morse_data.endswith("\n"):
        morse_data += "\n"
    fields = row_fields(panel)
    blob = harness_blob(root)
    host = host_id()
    cpu = panel.cpu_model()
    stored = read_rows(args.rows)
    sessions = {row["session"] for row in stored if row.get("session")}
    if len(sessions) > 1:
        raise SystemExit("stored rows have more than one session")
    session = next(iter(sessions)) if sessions else time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    done = {
        (row["system"], row["code"], row["seed"], row["code_revision"], row["harness_revision"])
        for row in stored
    }
    print(f"session {session} head {head} code {CODE_REVISION} gmin {panel.GMIN_PIN} jobs {len(jobs(manifest, panel))}", flush=True)
    for system, seed, code in jobs(manifest, panel):
        code_revision = CODE_REVISION if code == "anneal" else panel.GMIN_PIN
        key = (system, code, str(seed), code_revision, head)
        if key in done:
            print(f"skip {system} {code} {seed}", flush=True)
            continue
        print(f"start {system} {code} {seed}", flush=True)
        log = args.logs / system / f"{code}-{seed}.log"
        if code == "anneal":
            result = run_anneal(panel, args.bin, root, system, seed, log, args.timeout)
            result.pop("head", None)
            driver = panel.resolve_driver(args.bin, root, system)[0]
        else:
            result = run_gmin(
                panel,
                args.bin,
                args.gmin,
                root,
                system,
                seed,
                args.work / system / str(seed),
                log,
                lj_data,
                morse_data,
                args.timeout,
            )
            driver = "GMIN"
        row = {
            "system": system,
            "code": code,
            "driver": driver,
            "seed": str(seed),
            "budget": str(panel.PAPER_BUDGETS[system]),
            **result,
            "code_revision": code_revision,
            "harness_revision": head,
            "host": host,
            "cpu": cpu,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "harness_blob": blob,
            "session": session,
        }
        append_row(args.rows, fields, row)
        done.add(key)
        print(
            f"done {system} {code} {seed} hit={row['hit']} calls={row['evals_to_hit']} wall_s={row['wall_s']}",
            flush=True,
        )
    print("campaign complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
