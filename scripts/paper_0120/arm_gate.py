# Copyright (c) anneal developers
"""The paper headline panel on two builds in one session, seed against seed.

Each ``--arm NAME=BIN_DIR@REVISION`` names a directory holding the
binaries ``run_panel.py`` runs (``h2h_gmin_proof``, ``h2h_gmin_proof_lj103``,
``h2h_metal``, ``h2h_morse``, ``h2h_seed``) and the revision they were built
at; its parent holds that revision's tree as ``src``, the working
directory of each run, as ``run_panel.py --tree``. Every system of ``run_panel.RUNS`` runs on every seed for every arm at
the paper budgets, one single-threaded process per core over ``--cores``
cores, the arms of one seed back to back. Rows are appended to ``--rows``
with the arm, the revision and the host, never rewritten. The panel is
untimed: wall columns are kept but compared only as a note.

``--summary`` reads the rows and, per system and driver, gives each arm's
hits and median calls to the minimum (misses at the budget), Fisher's
exact test on hits and the tie-corrected Mann-Whitney test on calls, and
the seeds on which the two arms agree in hit, calls to the minimum, calls
charged and best energy to 1e-8. An arm loses on a system when it hits
fewer times, or its calls are higher at p below 0.05.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import os
import queue
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_panel  # noqa: E402

FIELDS = ["arm", *run_panel.ROW_FIELDS[:-1]]


def arm(text: str) -> tuple[str, Path, str]:
    name, rest = text.split("=", 1)
    directory, revision = rest.rsplit("@", 1)
    return name, Path(directory).resolve(), revision


def run_one(bin_dir: Path, run: tuple, seed: int, core: int, log: Path) -> dict[str, str]:
    system, driver, binary, prefix = run
    budget = run_panel.PAPER_BUDGETS[system]
    with tempfile.TemporaryDirectory() as scratch:
        if driver == "h2h_seed":
            stop = run_panel.LJ75_REFERENCE + run_panel.SEED_HIT_TOL
            args = ["lj", "75", "0", str(budget), str(seed), "recommended", repr(stop), str(Path(scratch) / "run")]
        else:
            args = [*prefix, str(seed)]
        command = ["taskset", "-c", str(core), str(bin_dir / binary), *args]
        proc = subprocess.run(
            command, cwd=bin_dir.parent / "src", env=run_panel.single_thread_env(), capture_output=True, text=True, check=False
        )
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(" ".join(command) + "\n" + proc.stdout + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"{system} {driver} seed {seed} exited {proc.returncode}\n{proc.stderr[-2000:]}")
    if driver == "h2h_seed":
        return run_panel.seed_result(proc.stdout, seed, budget)
    return run_panel.csv_result(proc.stdout, system, seed, budget)


def run(args: argparse.Namespace) -> None:
    arms = [arm(text) for text in args.arm]
    runs = [r for r in run_panel.RUNS if not args.systems or r[0] in args.systems]
    seeds = range(args.seed0, args.seed0 + args.n)
    # Longest systems first, so the last jobs to start are short ones.
    order = sorted(
        itertools.product(runs, seeds), key=lambda job: -run_panel.PAPER_BUDGETS[job[0][0]]
    )
    jobs: queue.Queue = queue.Queue()
    for run_, seed in order:
        for name, directory, revision in arms if seed % 2 == 0 else arms[::-1]:
            jobs.put((run_, seed, name, directory, revision))
    host = socket.gethostname()
    cpu = run_panel.cpu_model()
    lock = threading.Lock()
    failures = []

    def worker(core: int) -> None:
        while True:
            try:
                run_, seed, name, directory, revision = jobs.get_nowait()
            except queue.Empty:
                return
            system, driver, _binary, _prefix = run_
            log = args.logs / name / f"{system}-{driver}-{seed}.txt"
            try:
                result = run_one(directory, run_, seed, core, log)
            except RuntimeError as error:
                with lock:
                    failures.append(str(error))
                print(error, flush=True)
                continue
            row = {
                "arm": name,
                "system": system,
                "driver": driver,
                "code": "anneal",
                "seed": str(seed),
                "budget": str(run_panel.PAPER_BUDGETS[system]),
                **{k: result[k] for k in ("hit", "evals_to_hit", "charged", "wall_s", "best", "pot_s", "outside_s")},
                "code_revision": revision,
                "host": host,
            }
            with lock:
                fresh = not args.rows.is_file() or args.rows.stat().st_size == 0
                with args.rows.open("a", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
                    if fresh:
                        writer.writeheader()
                    writer.writerow(row)
            print(",".join(row.values()), cpu, flush=True)

    threads = [threading.Thread(target=worker, args=(core,)) for core in range(args.cores)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise SystemExit(f"{len(failures)} runs failed")


def agree(a: dict[str, str], b: dict[str, str]) -> bool:
    return (
        a["hit"] == b["hit"]
        and a["evals_to_hit"] == b["evals_to_hit"]
        and a["charged"] == b["charged"]
        and f"{float(a['best']):.8f}" == f"{float(b['best']):.8f}"
    )


def summary(args: argparse.Namespace) -> None:
    with args.rows.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    base, other = args.arms
    lost = []
    for system, driver, _binary, _prefix in run_panel.RUNS:
        by_arm = {
            name: {int(r["seed"]): r for r in rows if r["arm"] == name and r["system"] == system and r["driver"] == driver}
            for name in (base, other)
        }
        if not by_arm[base] or not by_arm[other]:
            continue
        seeds = sorted(set(by_arm[base]) & set(by_arm[other]))
        b = [by_arm[base][s] for s in seeds]
        o = [by_arm[other][s] for s in seeds]
        hits_b = sum(r["hit"] == "1" for r in b)
        hits_o = sum(r["hit"] == "1" for r in o)
        calls_b = [float(r["evals_to_hit"]) for r in b]
        calls_o = [float(r["evals_to_hit"]) for r in o]
        same = sum(agree(x, y) for x, y in zip(b, o))
        fisher = run_panel.fisher(hits_o, len(o), hits_b, len(b))
        _u, p = run_panel.mann_whitney(calls_o, calls_b)
        print(
            f"{system} {driver}: {base} {hits_b}/{len(b)} median {statistics.median(calls_b):.0f}, "
            f"{other} {hits_o}/{len(o)} median {statistics.median(calls_o):.0f}, "
            f"Fisher p {fisher:.3f}, Mann-Whitney p {p:.3f}, same on {same}/{len(seeds)} seeds"
        )
        if hits_o < hits_b or (statistics.median(calls_o) > statistics.median(calls_b) and p < 0.05):
            lost.append(f"{system} {driver}")
    print(f"{other} loses: {', '.join(lost) if lost else 'nowhere'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--arm", action="append", default=[], help="NAME=BIN_DIR@REVISION")
    parser.add_argument("--systems", nargs="*")
    parser.add_argument("--seed0", type=int, default=2000)
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--cores", type=int, default=os.cpu_count())
    parser.add_argument("--logs", type=Path)
    parser.add_argument("--summary", nargs=2, metavar=("BASE", "OTHER"), dest="arms")
    args = parser.parse_args()
    if args.arms:
        summary(args)
        return
    if len(args.arm) != 2 or args.logs is None:
        raise SystemExit("--arm twice and --logs run the gate")
    run(args)


if __name__ == "__main__":
    main()
