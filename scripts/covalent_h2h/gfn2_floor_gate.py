#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Same-session gate of a GFN2 call-floor change against its base build.

``run`` takes two ``gfn2_covalent_h2h`` binaries, ``--arm NAME=BINARY@REV``
each, and runs every (system, seed, arm) at ``--budget`` calls, one job
per core, the two arms of a seed back to back so both see the same
machine. The xtb engine comes from the environment
(``RGPOT_XTB_ENGINE``, ``LD_LIBRARY_PATH``) at ``RGPOT_XTB_ETEMP``. Each
job appends the example's final row, its arm, the arm's code revision and
the host to ``--rows``.

``--cores`` names the cores, ``1-7`` or ``1,3,5``, so a timed run can
keep one to itself.

``summary`` reads those rows. Per system it reports hits, calls to hit
(a miss counts its budget), the SCC failure rate (failed calls over
``charged``) and the best energy, with Fisher's exact test on hits and a
two-sided Mann-Whitney test on the rest, of every arm against the first.
A metric is lost when an arm hits fewer times, or is worse with p below
0.05. A system is lost when the arms do not hold the same seeds. A build
before 7136fcc8 ran each failed SCC twice and counted it twice;
``--doubled`` names those arms, whose counts are halved.
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

REFERENCE = {"si10": -460.68373809, "si13": -599.34546694, "c20": -1148.85383633}
EXAMPLE_FIELDS = [
    "system", "code", "seed", "budget", "hit", "evals_to_hit", "charged",
    "probe_calls", "wall_s", "best", "pot_s", "outside_s", "pot_per_eval_s",
    "outside_per_eval_s", "scc_rejected", "geometry_rejected",
]
FIELDS = ["arm", *EXAMPLE_FIELDS, "code_revision", "host"]


def arm(text):
    name, rest = text.split("=", 1)
    binary, revision = rest.rsplit("@", 1)
    return name, Path(binary).resolve(), revision


def cores(text):
    out = []
    for part in text.split(","):
        low, _, high = part.partition("-")
        out.extend(range(int(low), int(high or low) + 1))
    return out


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
            with lock, open(args.rows, "a", newline="") as handle:
                csv.writer(handle, lineterminator="\n").writerow(
                    [name, *fields, revision, host])
            print(f"{system} {name} {seed}: {','.join(fields)}", flush=True)

    threads = [threading.Thread(target=worker, args=(core,)) for core in args.cores]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def summary(args):
    with open(args.rows) as handle:
        rows = list(csv.DictReader(handle))
    base, others = args.arms[0], args.arms[1:]
    lost = {other: [] for other in others}

    def failed(row):
        count = int(row["scc_rejected"])
        if row["arm"] not in args.doubled:
            return count
        if count % 2:
            raise SystemExit(f"{row['arm']} {row['system']} {row['seed']}: odd doubled count")
        return count // 2

    for system in sorted({r["system"] for r in rows}, reverse=True):
        # A job with no final row appends nothing, so an arm can be short of
        # seeds; a comparison over what is left is not the gate.
        seeds = {name: sorted(int(r["seed"]) for r in rows if r["system"] == system and r["arm"] == name)
                 for name in args.arms}
        mismatched = [name for name in others if seeds[name] != seeds[base]]
        if not seeds[base] or mismatched:
            print(f"{system}: " + ", ".join(f"{name} has seeds {seeds[name]}" for name in args.arms))
            for name in mismatched:
                lost[name].append(f"{system} seeds")
            continue
        metric = {}
        for name in args.arms:
            runs = [r for r in rows if r["system"] == system and r["arm"] == name]
            scc = sum(failed(r) for r in runs)
            charged = sum(int(r["charged"]) for r in runs)
            metric[name] = {
                "n": len(runs),
                "hits": sum(r["hit"] == "true" for r in runs),
                "calls": [int(r["evals_to_hit"]) for r in runs],
                "scc": [failed(r) / int(r["charged"]) if int(r["charged"]) else 0.0 for r in runs],
                "best": [float(r["best"]) for r in runs],
            }
            m = metric[name]
            rate = f"{100 * scc / charged:.2f}%" if charged else "no calls"
            print(f"{system} {name}: hits {m['hits']}/{m['n']}, median calls to hit "
                  f"{statistics.median(m['calls'])}, SCC failures {scc}/{charged} "
                  f"({rate}), median best {statistics.median(m['best']):.6f} eV, "
                  f"lowest {min(m['best']):.6f} eV")
        from scipy.stats import fisher_exact, mannwhitneyu

        for other in others:
            b, o = metric[base], metric[other]
            p = fisher_exact([[b["hits"], b["n"] - b["hits"]],
                              [o["hits"], o["n"] - o["hits"]]]).pvalue
            print(f"{system} {other} hits: Fisher p = {p:.4f}")
            if o["hits"] / o["n"] < b["hits"] / b["n"]:
                lost[other].append(f"{system} hits")
            for key, label in (("calls", "calls to hit"), ("scc", "SCC failure rate"),
                               ("best", "best energy")):
                if sorted(b[key]) == sorted(o[key]):
                    print(f"{system} {other} {label}: the same as {base}")
                    continue
                p = mannwhitneyu(o[key], b[key], alternative="two-sided").pvalue
                worse = statistics.median(o[key]) > statistics.median(b[key])
                print(f"{system} {label}: {other} median {'above' if worse else 'at or below'} "
                      f"{base}, Mann-Whitney p = {p:.4f}")
                if worse and p < 0.05:
                    lost[other].append(f"{system} {label}")
    for other in others:
        print(f"{other} loses: {', '.join(lost[other]) if lost[other] else 'nowhere'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    go = sub.add_parser("run")
    go.add_argument("--arm", action="append", required=True, help="NAME=BINARY@REVISION")
    go.add_argument("--systems", nargs="+", default=list(REFERENCE))
    go.add_argument("--seed0", type=int, default=2000)
    go.add_argument("--n", type=int, default=12)
    go.add_argument("--budget", type=int, default=100000)
    go.add_argument("--cores", type=cores, default=list(range(os.cpu_count())))
    go.add_argument("--out", type=Path, required=True)
    go.add_argument("--rows", type=Path, required=True)
    report = sub.add_parser("summary")
    report.add_argument("rows", type=Path)
    report.add_argument("--arms", nargs="+", default=["base", "floor"])
    report.add_argument("--doubled", nargs="*", default=[])
    args = parser.parse_args()
    if args.command == "run":
        if not args.rows.exists() or args.rows.stat().st_size == 0:
            with open(args.rows, "a", newline="") as handle:
                csv.writer(handle, lineterminator="\n").writerow(FIELDS)
        run(args)
    else:
        summary(args)


if __name__ == "__main__":
    main()
