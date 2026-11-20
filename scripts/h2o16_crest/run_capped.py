#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Serial GFN2 (H2O)16 panel: anneal against CREST on one objective.

Both arms search the objective of ``examples/gfn2_water_h2h``: rgpot's
GFN2 handle (``libxtb_engine.so``, xtb 6.7.1) plus the one-sided O-H wall,
onset 1.15 angstrom, 200 eV per square angstrom. Anneal evaluates it in
process. CREST 3.0.2 reaches it through xtb's ``--driver``: every energy
and gradient xtb asks for goes to ``examples/gfn2_water_crest_driver``,
which holds the same handle and wall. Each seed's start is anneal's rigid
repack for that seed, written before any energy call, and both arms start
from it. CREST is

    crest start.xyz --gfn2 -T 1 --legacy -notopo -shake 1 -xnam <xtb --driver>

after one xtb ``--opt`` of the start through the same driver. ``-shake 1``
constrains every X-H bond, which for water is every covalent bond; the
driver supplies no Wiberg bond orders, which ``-shake 2`` reads. CREST's
first optimization and CREGEN compare a distance topology that the
driver's optimization changes, so the start is optimized first and CREGEN
runs with ``-notopo``.

Both arms are capped at ``BUDGET`` objective calls, the anneal (H2O)16
budget, and both stop at the first call at or below the reference plus
``HIT_TOL`` eV. A call is one energy and gradient of the objective. A miss
is stored as the full budget. Inside time is the time in the handle;
outside is the rest of the wall. The (seed, arm) pairs run in one shuffled
order. Rows are appended, never rewritten, with the anneal revision and
the host, and a pair already in the rows file at this budget is not run
again, so a stopped panel resumes in the same order. The summary covers
every row in the file at this budget for the chosen seeds and arms. This
process refuses to run unless the UTS hostname is bench.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import os
import random
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from scipy.stats import fisher_exact, mannwhitneyu

for _candidate in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):
    if (_candidate / "target_tol.py").is_file() and str(_candidate) not in sys.path:
        sys.path.insert(0, str(_candidate))
        break
from codata2018 import HARTREE_EV  # noqa: E402
from target_tol import DEFAULT_HIT_TOL  # noqa: E402

MOLECULES = 16
BUDGET = 280305
SEEDS = tuple(range(2000, 2012))
ARMS = ("anneal", "crest")
# wdbase-1, -81.396292669509 Eh, evaluated with the CODATA 2018 conversion.
REFERENCE_EV = -81.396292669509 * HARTREE_EV
HIT_TOL = DEFAULT_HIT_TOL
DRIVER_PATH_MAX = 64
SCHEDULE_SEED = 20261115

ROW_FIELDS = [
    "system",
    "arm",
    "seed",
    "budget",
    "hit",
    "evals_to_hit",
    "charged",
    "best_eV",
    "wall_s",
    "pot_s",
    "outside_s",
    "code_revision",
    "host",
    "date",
    "crest_sha256",
    "xtb_sha256",
    "engine_sha256",
    "rgpot_revision",
]
SUMMARY_FIELDS = [
    "system",
    "arm",
    "n",
    "hits",
    "budget",
    "seeds",
    "evals_median",
    "evals_iqr",
    "best_median_eV",
    "wall_median_s",
    "pot_median_s",
    "outside_median_s",
    "inside_ms_per_call",
    "outside_ms_per_call",
    "fisher_p",
    "mannwhitney_p",
    "mannwhitney_best_p",
    "code_revision",
    "host",
    "date",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_fields(path: Path) -> dict[str, str]:
    fields = {}
    for line in path.read_text().splitlines():
        key, sep, value = line.partition(":")
        if sep:
            fields[key.strip()] = value.strip()
    return fields


def type7(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    h = (len(ordered) - 1) * p
    low = int(h)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (h - low) * (ordered[high] - ordered[low])


def two_sided_mw(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return float("nan")
    if len(set(left) | set(right)) == 1:
        return 1.0
    return float(mannwhitneyu(left, right, alternative="two-sided").pvalue)


def run_anneal(
    binary: Path, seed: int, start: Path, out: Path, core: str, budget: int
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    target = f"{REFERENCE_EV + HIT_TOL:.10f}"
    env = dict(os.environ, H2H_START_XYZ=str(start))
    with (
        open(out / "stdout.txt", "w") as stdout,
        open(out / "stderr.txt", "w") as stderr,
    ):
        subprocess.run(
            [
                "taskset",
                "-c",
                core,
                str(binary),
                str(MOLECULES),
                str(budget),
                str(seed),
                str(out),
                target,
            ],
            env=env,
            stdout=stdout,
            stderr=stderr,
            check=True,
        )
    fields = read_fields(out / "run.txt")
    first = fields["first_hit_call"]
    hit = first != "none"
    charged = int(fields["energy_calls"])
    return {
        "arm": "anneal",
        "hit": "true" if hit else "false",
        "evals_to_hit": int(first) if hit else budget,
        "charged": charged,
        "best_eV": float(fields["objective_best_eV"]),
        "wall_s": float(fields["wall_s"]),
        "pot_s": float(fields["potential_s"]),
        "outside_s": float(fields["outside_s"]),
    }


def run_crest(
    crest: Path, xtb: Path, driver: Path, start: Path, out: Path, core: str, budget: int
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    served = out / "driver"
    served.mkdir(exist_ok=True)
    # xtb 6.7.1 aborts on a --driver path of about 75 characters, so the
    # wrappers and the socket sit in a short directory, copied into out.
    short = Path(tempfile.mkdtemp(prefix="crest-"))
    sock = short / "sock"
    target = f"{REFERENCE_EV + HIT_TOL:.10f}"
    call = short / "call"
    wrapper = short / "xtb"
    if len(str(call)) > DRIVER_PATH_MAX:
        raise SystemExit(f"{call} is longer than {DRIVER_PATH_MAX} characters")
    scripts = {
        call: f'#!/bin/sh\nexec "{driver}" call "{sock}"\n',
        wrapper: f'#!/bin/sh\nexec "{xtb}" "$@" --driver "{call}"\n',
    }
    for path, text in scripts.items():
        path.write_text(text)
        path.chmod(0o755)
        (out / f"driver-{path.name}").write_text(text)
    with open(out / "driver.out", "w") as log:
        server = subprocess.Popen(
            [
                "taskset",
                "-c",
                core,
                str(driver),
                "serve",
                str(sock),
                str(served),
                str(MOLECULES),
                str(budget),
                target,
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    while not (served / "ready").exists():
        if server.poll() is not None:
            raise RuntimeError(f"driver server exited {server.returncode}")
        time.sleep(0.05)
    preopt = out / "preopt"
    preopt.mkdir(exist_ok=True)
    (preopt / "start.xyz").write_bytes(start.read_bytes())
    search = out / "search"
    search.mkdir(exist_ok=True)
    command = [
        str(crest),
        "start.xyz",
        "--gfn2",
        "-T",
        "1",
        "--legacy",
        "-notopo",
        "-shake",
        "1",
        "-xnam",
        str(wrapper),
    ]
    (out / "command.txt").write_text(
        f"{wrapper} start.xyz --gfn 2 --opt\n" + " ".join(command) + "\n"
    )
    stop = served / "stop"
    began = time.monotonic()
    with open(preopt / "opt.out", "w") as stdout:
        subprocess.run(
            ["taskset", "-c", core, str(wrapper), "start.xyz", "--gfn", "2", "--opt"],
            cwd=preopt,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            check=False,
        )
    code = "preopt-failed"
    if (preopt / "xtbopt.xyz").is_file() and not stop.exists():
        (search / "start.xyz").write_bytes((preopt / "xtbopt.xyz").read_bytes())
        with (
            open(search / "crest.out", "w") as stdout,
            open(search / "crest.err", "w") as stderr,
        ):
            proc = subprocess.Popen(
                ["taskset", "-c", core] + command,
                cwd=search,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            while proc.poll() is None and not stop.exists():
                time.sleep(0.05)
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                code = f"stopped {stop.read_text().strip()}"
            else:
                code = f"exit {proc.returncode}"
    elif stop.exists():
        code = f"stopped {stop.read_text().strip()}"
    wall = time.monotonic() - began
    subprocess.run([str(driver), "stop", str(sock)], check=True)
    server.wait(timeout=60)
    shutil.rmtree(short, ignore_errors=True)
    (out / "exit.txt").write_text(code + "\n")
    fields = read_fields(served / "driver.txt")
    if int(fields["calls"]) == 0:
        raise RuntimeError(
            f"no call reached the driver ({code}), see {preopt / 'opt.out'}"
        )
    first = fields["first_hit_call"]
    hit = first != "none"
    inside = float(fields["inside_s"])
    return {
        "arm": "crest",
        "hit": "true" if hit else "false",
        "evals_to_hit": int(first) if hit else budget,
        "charged": int(fields["calls"]),
        "best_eV": float(fields["best_eV"]),
        "wall_s": wall,
        "pot_s": inside,
        "outside_s": max(wall - inside, 0.0),
    }


def append_rows(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.is_file() or path.stat().st_size == 0
    if not new:
        with open(path, newline="") as handle:
            header = next(csv.reader(handle))
        if header != fields:
            raise SystemExit(
                f"{path} has another header; rows are appended, never rewritten"
            )
    with open(path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if new:
            writer.writeheader()
        writer.writerows(rows)


def read_rows(path: Path, budget: int) -> list[dict]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with open(path, newline="") as handle:
        return [row for row in csv.DictReader(handle) if int(row["budget"]) == budget]


def summarise(
    rows: list[dict], revision: str, host: str, budget: int = BUDGET
) -> list[dict]:
    date = datetime.date.today().isoformat()
    by_arm = {arm: [row for row in rows if row["arm"] == arm] for arm in ARMS}
    anneal = by_arm["anneal"]
    out = []
    for arm in ARMS:
        group = by_arm[arm]
        if not group:
            continue
        evals = [float(row["evals_to_hit"]) for row in group]
        best = [float(row["best_eV"]) for row in group]
        seeds = sorted(int(row["seed"]) for row in group)
        hits = sum(row["hit"] == "true" for row in group)
        inside = [
            float(row["pot_s"]) / max(int(row["charged"]), 1) * 1e3 for row in group
        ]
        outside = [
            float(row["outside_s"]) / max(int(row["charged"]), 1) * 1e3 for row in group
        ]
        summary = {
            "system": f"h2o{MOLECULES}",
            "arm": arm,
            "n": len(group),
            "hits": hits,
            "budget": budget,
            "seeds": f"{seeds[0]}-{seeds[-1]}",
            "evals_median": f"{type7(evals, 0.5):.1f}",
            "evals_iqr": f"{type7(evals, 0.75) - type7(evals, 0.25):.1f}",
            "best_median_eV": f"{type7(best, 0.5):.8f}",
            "wall_median_s": f"{type7([float(row['wall_s']) for row in group], 0.5):.3f}",
            "pot_median_s": f"{type7([float(row['pot_s']) for row in group], 0.5):.3f}",
            "outside_median_s": f"{type7([float(row['outside_s']) for row in group], 0.5):.3f}",
            "inside_ms_per_call": f"{type7(inside, 0.5):.4f}",
            "outside_ms_per_call": f"{type7(outside, 0.5):.4f}",
            "fisher_p": "",
            "mannwhitney_p": "",
            "mannwhitney_best_p": "",
            "code_revision": revision,
            "host": host,
            "date": date,
        }
        if arm != "anneal" and anneal:
            anneal_hits = sum(row["hit"] == "true" for row in anneal)
            table = [
                [hits, len(group) - hits],
                [anneal_hits, len(anneal) - anneal_hits],
            ]
            summary["fisher_p"] = f"{fisher_exact(table).pvalue:.6g}"
            summary["mannwhitney_p"] = (
                f"{two_sided_mw(evals, [float(row['evals_to_hit']) for row in anneal]):.6g}"
            )
            summary["mannwhitney_best_p"] = (
                f"{two_sided_mw(best, [float(row['best_eV']) for row in anneal]):.6g}"
            )
        out.append(summary)
    return out


def self_test() -> None:
    if abs(type7([1.0, 2.0, 3.0, 4.0], 0.5) - 2.5) > 1e-12:
        raise SystemExit("type 7")
    if two_sided_mw([1.0] * 3, [1.0] * 3) != 1.0:
        raise SystemExit("tied mann-whitney")
    if abs(two_sided_mw([10.0, 11.0, 12.0], [1.0, 2.0, 3.0]) - 0.1) > 1e-12:
        raise SystemExit("exact mann-whitney")
    if abs(REFERENCE_EV + 2214.905861472502) > 1e-9:
        raise SystemExit("reference")
    with tempfile.TemporaryDirectory() as scratch:
        path = Path(scratch) / "rows.csv"
        row = {key: "x" for key in ROW_FIELDS}
        append_rows(path, ROW_FIELDS, [row])
        append_rows(path, ROW_FIELDS, [row])
        if len(path.read_text().splitlines()) != 3:
            raise SystemExit("append")
        path.write_text("system,arm\n")
        try:
            append_rows(path, ROW_FIELDS, [row])
        except SystemExit:
            pass
        else:
            raise SystemExit("header guard")
    rows = [
        {
            "arm": arm,
            "seed": seed,
            "hit": "false",
            "evals_to_hit": BUDGET,
            "charged": BUDGET,
            "best_eV": -2200.0 - seed % 3,
            "wall_s": 10.0,
            "pot_s": 4.0,
            "outside_s": 6.0,
        }
        for arm in ARMS
        for seed in SEEDS
    ]
    summary = summarise(rows, "rev", "bench")
    if summary[1]["fisher_p"] != "1" or summary[1]["mannwhitney_p"] != "1":
        raise SystemExit(f"summary {summary[1]}")
    print("ok")


def rgpot_revision_of(engine: Path, override: str | None = None) -> str:
    if override:
        return override.strip()
    env_rev = os.environ.get("RGPOT_REVISION", "").strip()
    if env_rev:
        return env_rev
    engine_dir = engine.resolve().parent
    for name in ("RGPOT_SOURCE_COMMIT", "rgpot.sha"):
        candidate = engine_dir / name
        if candidate.is_file():
            text = candidate.read_text().strip()
            if text:
                return text
    root = os.environ.get("RGPOT_ROOT", "").strip()
    if root:
        return subprocess.check_output(["git", "-C", root, "rev-parse", "HEAD"], text=True).strip()
    raise SystemExit("missing rgpot revision: pass --rgpot-revision, set RGPOT_REVISION/RGPOT_ROOT, or place rgpot.sha beside --engine")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--anneal", type=Path)
    ap.add_argument("--driver", type=Path)
    ap.add_argument("--crest", type=Path)
    ap.add_argument("--xtb", type=Path)
    ap.add_argument("--engine", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--rows", type=Path)
    ap.add_argument("--summary", type=Path)
    ap.add_argument("--head-file", type=Path)
    ap.add_argument("--rgpot-revision")
    ap.add_argument("--core", default="0")
    ap.add_argument("--budget", type=int, default=BUDGET)
    ap.add_argument("--seed", type=int, action="append")
    ap.add_argument("--arm", action="append", choices=ARMS)
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    for name in (
        "anneal",
        "driver",
        "crest",
        "xtb",
        "engine",
        "out",
        "rows",
        "head_file",
    ):
        if getattr(args, name) is None:
            raise SystemExit(f"--{name.replace('_', '-')} is required")
    if socket.gethostname() != "bench":
        raise SystemExit("hostname is not bench")
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["RGPOT_XTB_ENGINE"] = str(args.engine)
    revision = args.head_file.read_text().strip()
    host = socket.gethostname()
    seeds = tuple(args.seed) if args.seed else SEEDS
    arms = tuple(args.arm) if args.arm else ARMS
    pins = {
        "crest_sha256": sha256(args.crest),
        "xtb_sha256": sha256(args.xtb),
        "engine_sha256": sha256(args.engine),
        "rgpot_revision": rgpot_revision_of(args.engine, args.rgpot_revision),
    }
    starts = args.out / "starts"
    starts.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        where = starts / str(seed)
        subprocess.run(
            [str(args.anneal), str(MOLECULES), str(args.budget), str(seed), str(where)],
            env=dict(os.environ, H2H_START_ONLY="1"),
            stdout=subprocess.DEVNULL,
            check=True,
        )
    pairs = [(seed, arm) for seed in seeds for arm in arms]
    random.Random(SCHEDULE_SEED).shuffle(pairs)
    finished = {
        (int(row["seed"]), row["arm"]) for row in read_rows(args.rows, args.budget)
    }
    for seed, arm in pairs:
        if (seed, arm) in finished:
            print(f"skip {arm} {seed}, already in {args.rows}", flush=True)
            continue
        start = starts / str(seed) / "start.xyz"
        where = args.out / arm / str(seed)
        print(
            f"start {arm} {seed} {datetime.datetime.now().isoformat(timespec='seconds')}",
            flush=True,
        )
        if arm == "anneal":
            result = run_anneal(args.anneal, seed, start, where, args.core, args.budget)
        else:
            result = run_crest(
                args.crest, args.xtb, args.driver, start, where, args.core, args.budget
            )
        row = {
            "system": f"h2o{MOLECULES}",
            "seed": seed,
            "budget": args.budget,
            "code_revision": revision,
            "host": host,
            "date": datetime.date.today().isoformat(),
            **pins,
            **result,
        }
        append_rows(args.rows, ROW_FIELDS, [row])
        print(",".join(str(row[key]) for key in ROW_FIELDS), flush=True)
    if args.summary is not None:
        rows = [
            row
            for row in read_rows(args.rows, args.budget)
            if int(row["seed"]) in seeds and row["arm"] in arms
        ]
        summary = summarise(rows, revision, host, args.budget)
        append_rows(args.summary, SUMMARY_FIELDS, summary)
        for row in summary:
            print(",".join(str(row[key]) for key in SUMMARY_FIELDS), flush=True)


if __name__ == "__main__":
    main()
