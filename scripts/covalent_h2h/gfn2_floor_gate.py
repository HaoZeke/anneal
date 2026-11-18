#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Same-session gate of a GFN2 call-floor change against its base build.

``run`` takes two ``gfn2_covalent_h2h`` binaries, ``--arm NAME=BINARY@REV``
each, and runs every (system, seed, arm) at ``--budget`` calls, one job
per core from ``--first-core``, the arms of a seed back to back so all
see the same machine. The xtb engine comes from the environment
(``RGPOT_XTB_ENGINE``, ``LD_LIBRARY_PATH``) at ``RGPOT_XTB_ETEMP``. Each
job appends the example's final row, its arm, the arm's code revision and
the host to ``--rows``. A rows file keeps the header it was started with,
so a row with another column count, or a file with another header, is
refused rather than appended.

``summary`` reads those rows, from one file or several. Per system it
reports hits, calls to hit (a miss counts its budget), the SCC failure
rate and the best energy, with Fisher's exact test on hits and a
two-sided Mann-Whitney test on the rest. A metric is lost when the second
arm hits fewer times, or is worse with p below 0.05. A system is lost
when the two arms do not hold the same seeds.

The SCC failure rate is ``scc_failed_calls`` over ``charged`` where both
arms have that column. Rows from before it have only ``scc_rejected``,
the xtb evaluations whose SCC failed, and those builds ran two for each
failed call; a comparison that involves them uses ``scc_rejected``.
"""

import argparse
import csv
import itertools
import os
import queue
import socket
import statistics
import subprocess
import threading
from pathlib import Path

REFERENCE = {"si13": -599.34546694, "c20": -1148.85383633}
EXAMPLE_FIELDS = [
    "system", "code", "seed", "budget", "hit", "evals_to_hit", "charged",
    "probe_calls", "wall_s", "best", "pot_s", "outside_s", "pot_per_eval_s",
    "outside_per_eval_s", "scc_rejected", "geometry_rejected", "scc_failed_calls",
]
FIELDS = ["arm", *EXAMPLE_FIELDS, "code_revision", "host"]


def arm(text):
    name, rest = text.split("=", 1)
    binary, revision = rest.rsplit("@", 1)
    return name, Path(binary).resolve(), revision


def run_one(binary, system, seed, budget, core, where):
    where.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", RAYON_NUM_THREADS="1")
    with open(where / "stdout.txt", "w") as out:
        subprocess.run(
            ["taskset", "-c", str(core), str(binary), system, str(seed),
             str(budget), str(REFERENCE[system]), str(where / "best.xyz")],
            stdout=out, stderr=subprocess.STDOUT, env=env, check=False,
        )
    rows = [line for line in (where / "stdout.txt").read_text().splitlines()
            if line.startswith(f"{system},anneal-general,")]
    return rows[-1].split(",") if rows else None


def run(args):
    arms = [arm(text) for text in args.arm]
    jobs = queue.Queue()
    for system, seed in itertools.product(args.systems, range(args.seed0, args.seed0 + args.n)):
        order = arms if seed % 2 == 0 else arms[::-1]
        for name, binary, revision in order:
            jobs.put((system, seed, name, binary, revision))
    host = socket.gethostname()
    lock = threading.Lock()

    def worker(core):
        while True:
            try:
                system, seed, name, binary, revision = jobs.get_nowait()
            except queue.Empty:
                return
            where = args.out / system / name / str(seed)
            fields = run_one(binary, system, seed, args.budget, core, where)
            if fields is None:
                print(f"{system} {name} {seed}: no final row", flush=True)
                continue
            if len(fields) != len(EXAMPLE_FIELDS):
                print(f"{system} {name} {seed}: {len(fields)} fields, not "
                      f"{len(EXAMPLE_FIELDS)}", flush=True)
                continue
            with lock, open(args.rows, "a", newline="") as handle:
                csv.writer(handle, lineterminator="\n").writerow(
                    [name, *fields, revision, host])
            print(f"{system} {name} {seed}: {','.join(fields)}", flush=True)

    threads = [threading.Thread(target=worker, args=(core,))
               for core in range(args.first_core, args.first_core + args.cores)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def summary(args):
    rows = []
    for path in args.rows:
        with open(path) as handle:
            rows.extend(csv.DictReader(handle))
    base, other = args.arms
    lost = []
    for system in sorted({r["system"] for r in rows}, reverse=True):
        # A job with no final row appends nothing, so an arm can be short of
        # seeds; a comparison over what is left is not the gate.
        seeds = {name: sorted(int(r["seed"]) for r in rows if r["system"] == system and r["arm"] == name)
                 for name in (base, other)}
        if not seeds[base] or seeds[base] != seeds[other]:
            print(f"{system}: {base} has seeds {seeds[base]}, {other} has {seeds[other]}")
            lost.append(f"{system} seeds")
            continue
        runs = {name: [r for r in rows if r["system"] == system and r["arm"] == name]
                for name in (base, other)}
        per_call = all(r.get("scc_failed_calls") for name in runs for r in runs[name])
        scc_field = "scc_failed_calls" if per_call else "scc_rejected"
        metric = {}
        for name in (base, other):
            arm_runs = runs[name]
            scc = sum(int(r["scc_rejected"]) for r in arm_runs)
            charged = sum(int(r["charged"]) for r in arm_runs)
            metric[name] = {
                "n": len(arm_runs),
                "hits": sum(r["hit"] == "true" for r in arm_runs),
                "calls": [int(r["evals_to_hit"]) for r in arm_runs],
                "scc": [int(r[scc_field]) / int(r["charged"]) if int(r["charged"]) else 0.0
                        for r in arm_runs],
                "best": [float(r["best"]) for r in arm_runs],
            }
            m = metric[name]
            rate = f"{100 * scc / charged:.2f}%" if charged else "no calls"
            if all(r.get("scc_failed_calls") for r in arm_runs):
                failed = sum(int(r["scc_failed_calls"]) for r in arm_runs)
                failed_rate = f"{100 * failed / charged:.2f}%" if charged else "no calls"
                scc_text = (f"SCC-failed calls {failed}/{charged} ({failed_rate}), "
                            f"{scc} xtb evaluations")
            else:
                scc_text = f"SCC failures {scc}/{charged} ({rate})"
            print(f"{system} {name}: hits {m['hits']}/{m['n']}, median calls to hit "
                  f"{statistics.median(m['calls'])}, {scc_text}, "
                  f"median best {statistics.median(m['best']):.6f} eV, "
                  f"lowest {min(m['best']):.6f} eV")
        from scipy.stats import fisher_exact, mannwhitneyu

        b, o = metric[base], metric[other]
        p = fisher_exact([[b["hits"], b["n"] - b["hits"]], [o["hits"], o["n"] - o["hits"]]]).pvalue
        print(f"{system} hits: Fisher p = {p:.4f}")
        if o["hits"] / o["n"] < b["hits"] / b["n"]:
            lost.append(f"{system} hits")
        for key, label in (("calls", "calls to hit"), ("scc", "SCC failure rate"),
                           ("best", "best energy")):
            if sorted(b[key]) == sorted(o[key]):
                print(f"{system} {label}: the same in both arms")
                continue
            p = mannwhitneyu(o[key], b[key], alternative="two-sided").pvalue
            worse = statistics.median(o[key]) > statistics.median(b[key])
            print(f"{system} {label}: {other} median {'above' if worse else 'at or below'} "
                  f"{base}, Mann-Whitney p = {p:.4f}")
            if worse and p < 0.05:
                lost.append(f"{system} {label}")
    print(f"{other} loses: {', '.join(lost) if lost else 'nowhere'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    go = sub.add_parser("run")
    go.add_argument("--arm", action="append", required=True, help="NAME=BINARY@REVISION")
    go.add_argument("--systems", nargs="+", default=list(REFERENCE))
    go.add_argument("--seed0", type=int, default=2000)
    go.add_argument("--n", type=int, default=12)
    go.add_argument("--budget", type=int, default=100000)
    go.add_argument("--cores", type=int, default=os.cpu_count())
    go.add_argument("--first-core", type=int, default=0,
                    help="jobs run on cores first-core to first-core + cores - 1")
    go.add_argument("--out", type=Path, required=True)
    go.add_argument("--rows", type=Path, required=True)
    report = sub.add_parser("summary")
    report.add_argument("rows", type=Path, nargs="+")
    report.add_argument("--arms", nargs=2, default=["base", "floor"])
    args = parser.parse_args()
    if args.command == "run":
        if not args.rows.exists() or args.rows.stat().st_size == 0:
            with open(args.rows, "a", newline="") as handle:
                csv.writer(handle, lineterminator="\n").writerow(FIELDS)
        with open(args.rows, newline="") as handle:
            header = next(csv.reader(handle))
        if header != FIELDS:
            parser.error(f"{args.rows} has the header {','.join(header)}; "
                         f"start a new rows file for {','.join(FIELDS)}")
        run(args)
    else:
        summary(args)


if __name__ == "__main__":
    main()
