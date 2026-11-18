#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Serial GFN2 (H2O)6 panel: anneal against CREST at one potential.

Both arms evaluate rgpot GFN2 (xtb 6.7.1, accuracy 0.01) plus the
one-sided O-H wall, onset 1.15 angstrom, 200 eV per square angstrom.
Anneal is ``examples/gfn2_water_h2h`` at ``BUDGET`` with the default
TIP4P surrogate. CREST 3.0.2 is

    crest start.xyz -T 1 --gfn2 --legacy -mquick -len <ps> -shake 1
        -notopo --noreftopo -xnam <xtb --acc 0.01 --driver>

``--legacy`` sends every energy and gradient to
``examples/gfn2_water_crest_driver``. The driver's gradient file is the
gradient that xtb expects on the same geometry, so the call script only
forwards the request. ``-mquick`` is CREST's shortest documented ensemble.
``-len`` is then the length. ``--noreftopo`` is the
switch CREST prints when the first optimisation changes a water
topology. ``-shake 1`` constrains X-H bonds. ``-notopo`` keeps CREGEN
from dropping a cluster whose contacts differ from the start.

The driver is served with ``BUDGET`` as its budget, so CREST cannot spend
more than 4000 served calls: after the 4000th the driver writes ``stop`` and
refuses every later request, and the row records ``budget`` as the stop
reason. CREST's own ``Total number of energy+grad calls`` line is read from
``crest.out`` and the row is charged the larger of that total and the served
count. Both counts, the stop reason and the log digest go to
``call-counts.csv``. A row is therefore never charged less than CREST says it
spent.

On this build the 4000-call ledger charges 3021 GFN2 evaluations on
every seed from 2000 to 2011. A quarter of the ledger is held for a
final polish and, with no quench inside the record gradient, that
quarter is not spent. Trial driver counts at ``-mquick`` were 2807 at
0.80 ps, 2867 and 2946 at 0.90 ps on seed 2000, 3338, 3731, 2941 and
2951 at 0.90 ps on seeds 2003, 2006, 2009 and 2011, 3148 and 4043 at
0.95 ps, and 3483 at 1.00 ps. The 0.90 ps draws sit around 3021 and
several of them fall outside 5 percent, so the panel records 0.90 ps
and also 0.11 ps. 0.11 ps is the shortest length that finished; 0.10 ps
stops in CREST with an empty ensemble. Anneal's median wall on the
twelve seeds is 11.3 s and the 0.11 ps wall on seed 2000 was 22.8 s.

A hit is a vtight polish (``--gfn 2 --opt vtight --acc 0.01``, 500
iterations) within 1e-7 Eh of the prism reference. The search does not
stop on that reference. Rows are appended, never rewritten. A second
header is refused. The summary is appended once per system, arm and
revision. This process runs the search only when the UTS hostname is
bench, on one core.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import gzip
import hashlib
import os
import random
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SYSTEM = "h2o6"
MOLECULES = 6
BUDGET = 4000
SEEDS = tuple(range(2000, 2012))
ARMS = ("anneal", "crest", "crest-wall")
CREST_LEN_PS = 0.90
CREST_WALL_LEN_PS = 0.11
LEN_PS = {"crest": CREST_LEN_PS, "crest-wall": CREST_WALL_LEN_PS}
CREST_FLAGS = "mquick,legacy,gfn2,T1,shake1,notopo,noreftopo,acc0.01"
# Prism, reoptimised with xtb 6.7.1 --gfn 2 --opt vtight --acc 0.01.
REFERENCE_EH = -30.496506792187
HIT_TOL_EH = 1e-7
AUTOEV = 27.21138505
DRIVER_PATH_MAX = 64
SCHEDULE_SEED = 20261118
POLISH_ARGS = ["--gfn", "2", "--opt", "vtight", "--acc", "0.01", "--iterations", "500"]
REFERENCE_XYZ = ROOT / "bench" / "crest-equal" / "reference-prism.xyz"
ENGRAD_LINE = "Total number of energy+grad calls:"

ROW_FIELDS = [
    "system",
    "arm",
    "seed",
    "budget",
    "hit",
    "charged",
    "polished_Eh",
    "gap_Eh",
    "engine_eV",
    "wall_s",
    "pot_s",
    "outside_s",
    "crest_len_ps",
    "crest_exit",
    "surrogate_calls",
    "crest_flags",
    "code_revision",
    "host",
    "date",
]
COUNT_FIELDS = [
    "system",
    "arm",
    "seed",
    "budget",
    "driver_calls",
    "driver_stop",
    "crest_log_calls",
    "charged",
    "crest_out_sha256",
    "code_revision",
    "host",
    "date",
]
SUMMARY_FIELDS = [
    "system",
    "arm",
    "n",
    "hits",
    "budget",
    "calls_median",
    "calls_iqr",
    "calls_min",
    "calls_max",
    "within_5pct",
    "polished_median_Eh",
    "wall_median_s",
    "pot_median_s",
    "outside_median_s",
    "fisher_p",
    "mannwhitney_calls_p",
    "mannwhitney_inside_p",
    "mannwhitney_outside_p",
    "mannwhitney_wall_p",
    "crest_len_ps",
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


def crest_log_calls(log: Path) -> int:
    """The integer on CREST's own energy+grad total."""
    if not log.is_file():
        raise SystemExit(f"{log} is missing")
    totals = []
    for line in log.read_text(errors="replace").splitlines():
        if ENGRAD_LINE in line:
            totals.append(int(line.split(ENGRAD_LINE, 1)[1].split()[0]))
    if len(totals) != 1:
        raise SystemExit(f"{log} has {len(totals)} energy+grad totals")
    return totals[0]


def charged_calls(driver_calls: int, log_calls: int) -> int:
    """Calls billed to a CREST row: never below what CREST says it spent."""
    if log_calls is None:
        raise SystemExit("CREST log has no energy+grad total")
    return max(driver_calls, log_calls)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def validate_header(path: Path, fields: list[str]) -> None:
    if path.is_file():
        existing = path.read_text().splitlines()
        if not existing or existing[0] != ",".join(fields):
            raise SystemExit(f"{path} header is not the current one")


def append_rows(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    validate_header(path, fields)
    existing_text = path.read_text() if path.is_file() else ""
    mode = path.stat().st_mode & 0o777 if path.is_file() else 0o644
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", newline="") as handle:
            if existing_text:
                handle.write(existing_text)
                if not existing_text.endswith("\n"):
                    handle.write("\n")
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
            if not existing_text:
                writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in fields})
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp_path, mode)
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def record_row(counts_path: Path, rows_path: Path, count_row: dict, row: dict) -> None:
    """Write the call-counts entry atomically before appending the result row."""
    validate_header(counts_path, COUNT_FIELDS)
    validate_header(rows_path, ROW_FIELDS)
    append_rows(counts_path, COUNT_FIELDS, [count_row])
    append_rows(rows_path, ROW_FIELDS, [row])


def quantile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    h = (len(ordered) - 1) * p
    low = int(h)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (h - low) * (ordered[high] - ordered[low])


def mann_whitney(left: list[float], right: list[float]) -> float:
    """Two-sided Mann-Whitney with the tie correction of the normal approximation."""
    if not left or not right:
        return float("nan")
    if len(set(left) | set(right)) == 1:
        return 1.0
    from scipy.stats import mannwhitneyu

    return float(
        mannwhitneyu(left, right, alternative="two-sided", method="asymptotic").pvalue
    )


def fisher(hits_left: int, n_left: int, hits_right: int, n_right: int) -> float:
    from scipy.stats import fisher_exact

    table = [
        [hits_left, n_left - hits_left],
        [hits_right, n_right - hits_right],
    ]
    return float(fisher_exact(table, alternative="two-sided")[1])


def fmt(value: float, digits: str) -> str:
    if value != value:
        return ""
    return format(value, digits)


def revision_of(head_file: Path | None) -> str:
    if head_file is not None:
        return head_file.read_text().strip()
    proc = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def polish(xtb: Path, xyz: Path, work: Path) -> tuple[str, str]:
    """Return (energy in Eh, converged) from one vtight optimisation."""
    work.mkdir(parents=True, exist_ok=True)
    (work / "in.xyz").write_text(xyz.read_text())
    proc = subprocess.run(
        ["taskset", "-c", os.environ.get("CREST_EQUAL_CORE", "0"), str(xtb), "in.xyz", *POLISH_ARGS],
        cwd=work,
        env=dict(os.environ, OMP_NUM_THREADS="1"),
        capture_output=True,
        text=True,
    )
    (work / "opt.out").write_text(proc.stdout + proc.stderr)
    energies = []
    for line in proc.stdout.splitlines():
        parts = line.split()
        if "TOTAL ENERGY" in line and "Eh" in parts:
            energies.append(float(parts[parts.index("Eh") - 1]))
    converged = proc.returncode == 0 and "GEOMETRY OPTIMIZATION CONVERGED" in proc.stdout
    if not energies:
        return "", "false"
    return f"{energies[-1]:.12f}", "true" if converged else "false"


def check_reference(xtb: Path, work: Path) -> None:
    energy, converged = polish(xtb, REFERENCE_XYZ, work)
    if converged != "true" or abs(float(energy) - REFERENCE_EH) > 1e-8:
        raise SystemExit(f"reference polish {energy} converged {converged}")


def run_anneal(binary: Path, seed: int, start: Path, out: Path, core: str) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["H2H_START_XYZ"] = str(start)
    env.pop("H2H_START_ONLY", None)
    env.pop("H2O_TARGET_EV", None)
    started = time.perf_counter()
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
                str(BUDGET),
                str(seed),
                str(out),
            ],
            env=env,
            stdout=stdout,
            stderr=stderr,
            check=True,
        )
    wall = time.perf_counter() - started
    fields = read_fields(out / "run.txt")
    return {
        "arm": "anneal",
        "charged": int(fields["energy_calls"]),
        "engine_eV": fields.get("engine_best_eV", ""),
        "wall_s": f"{wall:.6f}",
        "pot_s": fields["potential_s"],
        "outside_s": fields["outside_s"],
        "crest_len_ps": "",
        "crest_exit": "",
        "surrogate_calls": fields.get("surrogate_calls", "0"),
        "crest_flags": "",
        "best_xyz": out / "best.xyz",
    }


def driver_call_script(driver: Path, socket: Path) -> str:
    return f"""#!/bin/sh
"{driver}" call "{socket}"
"""


def run_crest(
    crest: Path,
    xtb: Path,
    driver: Path,
    start: Path,
    out: Path,
    core: str,
    arm: str,
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    served = out / "driver"
    served.mkdir(exist_ok=True)
    search = out / "search"
    search.mkdir(exist_ok=True)
    (search / "start.xyz").write_text(start.read_text())
    short = Path(tempfile.mkdtemp(prefix="ce"))
    socket_path = short / "s"
    call = short / "c"
    wrapper = short / "x"
    if len(str(call)) > DRIVER_PATH_MAX:
        raise SystemExit(f"driver path is longer than {DRIVER_PATH_MAX} characters")
    call.write_text(driver_call_script(driver, socket_path))
    call.chmod(0o755)
    wrapper.write_text(f'#!/bin/sh\nexec "{xtb}" "$@" --acc 0.01 --driver "{call}"\n')
    wrapper.chmod(0o755)
    server_log = open(served / "server.out", "w")
    server = subprocess.Popen(
        [
            "taskset",
            "-c",
            core,
            str(driver),
            "serve",
            str(socket_path),
            str(served),
            str(MOLECULES),
            str(BUDGET),
            "-1e9",
        ],
        stdout=server_log,
        stderr=subprocess.STDOUT,
    )
    for _ in range(400):
        if (served / "ready").exists():
            break
        time.sleep(0.05)
    else:
        server.kill()
        raise SystemExit("driver did not become ready")
    length = LEN_PS[arm]
    command = [
        "taskset",
        "-c",
        core,
        str(crest),
        "start.xyz",
        "-T",
        "1",
        "--gfn2",
        "--legacy",
        "-mquick",
        "-len",
        f"{length:.2f}",
        "-shake",
        "1",
        "-notopo",
        "--noreftopo",
        "-xnam",
        str(wrapper),
    ]
    started = time.perf_counter()
    with (
        open(search / "crest.out", "w") as stdout,
        open(search / "crest.err", "w") as stderr,
    ):
        proc = subprocess.run(command, cwd=search, stdout=stdout, stderr=stderr)
    wall = time.perf_counter() - started
    subprocess.run([str(driver), "stop", str(socket_path)], check=False, stdout=subprocess.DEVNULL)
    server.wait(timeout=60)
    server_log.close()
    fields = read_fields(served / "driver.txt")
    driver_calls = int(fields.get("calls", "0"))
    log_calls = crest_log_calls(search / "crest.out")
    inside = float(fields.get("inside_s", "nan"))
    outside = wall - inside if inside == inside else float("nan")
    best = search / "crest_best.xyz"
    if not best.is_file():
        best = served / "best.xyz"
    return {
        "arm": arm,
        "charged": charged_calls(driver_calls, log_calls),
        "driver_calls": driver_calls,
        "driver_stop": fields.get("stop", "none"),
        "crest_log_calls": log_calls,
        "crest_out": search / "crest.out",
        "engine_eV": fields.get("best_eV", ""),
        "wall_s": f"{wall:.6f}",
        "pot_s": f"{inside:.6f}" if inside == inside else "",
        "outside_s": f"{outside:.6f}" if outside == outside else "",
        "crest_len_ps": f"{length:.2f}",
        "crest_exit": str(proc.returncode),
        "surrogate_calls": "0",
        "crest_flags": CREST_FLAGS,
        "best_xyz": best,
    }


def keep_log(source: Path, logs: Path, arm: str, seed: int, revision: str) -> str:
    """Copy a CREST log gzipped beside the rows and return its sha256."""
    logs.mkdir(parents=True, exist_ok=True)
    target = logs / f"{arm}-{seed}-{revision[:8]}.out.gz"
    with source.open("rb") as raw, gzip.GzipFile(target, "wb", mtime=0) as packed:
        shutil.copyfileobj(raw, packed)
    return sha256(source)


def score(xtb: Path, result: dict, work: Path) -> dict:
    best = result.pop("best_xyz")
    polished, converged = ("", "false")
    if best.is_file():
        polished, converged = polish(xtb, best, work)
    gap = ""
    hit = "false"
    if polished and converged == "true":
        delta = float(polished) - REFERENCE_EH
        gap = f"{delta:.6e}"
        if abs(delta) <= HIT_TOL_EH:
            hit = "true"
    result["hit"] = hit
    result["polished_Eh"] = polished
    result["gap_Eh"] = gap
    return result


def summarise(
    rows: list[dict[str, str]],
    *,
    fisher_fn=None,
    mann_whitney_fn=None,
) -> list[dict[str, str]]:
    if fisher_fn is None:
        fisher_fn = fisher
    if mann_whitney_fn is None:
        mann_whitney_fn = mann_whitney
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault((row["system"], row["arm"], row["code_revision"]), []).append(row)
    out = []
    for (system, arm, revision), group in sorted(groups.items()):
        anneal = groups.get((system, "anneal", revision), [])
        charged = [float(row["charged"]) for row in group]
        walls = [float(row["wall_s"]) for row in group if row["wall_s"]]
        pots = [float(row["pot_s"]) for row in group if row["pot_s"]]
        outs = [float(row["outside_s"]) for row in group if row["outside_s"]]
        polished = [float(row["polished_Eh"]) for row in group if row["polished_Eh"]]
        hits = sum(row["hit"] == "true" for row in group)
        target = [float(row["charged"]) for row in anneal]
        center = quantile(target, 0.5) if target else float("nan")
        within = ""
        if center == center and center != 0.0:
            within = str(sum(abs(value - center) <= 0.05 * abs(center) for value in charged))
        fisher_p = calls_p = inside_p = outside_p = wall_p = ""
        if arm != "anneal" and anneal:
            fisher_p = f"{fisher_fn(sum(r['hit'] == 'true' for r in anneal), len(anneal), hits, len(group)):.6e}"
            calls_p = fmt(mann_whitney_fn(target, charged), ".6e")
            inside_p = fmt(
                mann_whitney_fn(
                    [float(r["pot_s"]) for r in anneal if r["pot_s"]],
                    pots,
                ),
                ".6e",
            )
            outside_p = fmt(
                mann_whitney_fn(
                    [float(r["outside_s"]) for r in anneal if r["outside_s"]],
                    outs,
                ),
                ".6e",
            )
            wall_p = fmt(
                mann_whitney_fn(
                    [float(r["wall_s"]) for r in anneal if r["wall_s"]],
                    walls,
                ),
                ".6e",
            )
        dates = [row["date"] for row in group if row["date"]]
        out.append(
            {
                "system": system,
                "arm": arm,
                "n": str(len(group)),
                "hits": str(hits),
                "budget": group[0]["budget"],
                "calls_median": fmt(quantile(charged, 0.5), ".6f"),
                "calls_iqr": fmt(quantile(charged, 0.75) - quantile(charged, 0.25), ".6f"),
                "calls_min": str(int(min(charged))),
                "calls_max": str(int(max(charged))),
                "within_5pct": within,
                "polished_median_Eh": fmt(quantile(polished, 0.5), ".12f") if polished else "",
                "wall_median_s": fmt(quantile(walls, 0.5), ".6f") if walls else "",
                "pot_median_s": fmt(quantile(pots, 0.5), ".6f") if pots else "",
                "outside_median_s": fmt(quantile(outs, 0.5), ".6f") if outs else "",
                "fisher_p": fisher_p,
                "mannwhitney_calls_p": calls_p,
                "mannwhitney_inside_p": inside_p,
                "mannwhitney_outside_p": outside_p,
                "mannwhitney_wall_p": wall_p,
                "crest_len_ps": group[0].get("crest_len_ps", ""),
                "code_revision": revision,
                "host": group[0]["host"],
                "date": max(dates) if dates else "",
            }
        )
    return out


def claim_sentence(summary: list[dict[str, str]]) -> str:
    """The sentence the README and the paper quote, built from the summary rows."""
    by_arm = {row["arm"]: row for row in summary}

    def calls(row: dict[str, str]) -> str:
        return format(float(row["calls_median"]), "g")

    anneal, crest, wall = by_arm["anneal"], by_arm["crest"], by_arm["crest-wall"]
    return (
        f"Anneal charges {calls(anneal)} calls and hits {anneal['hits']} of {anneal['n']}. "
        f"CREST at {crest['crest_len_ps']} ps has median {calls(crest)} calls and hits "
        f"{crest['hits']} of {crest['n']}. "
        f"CREST at {wall['crest_len_ps']} ps has median {calls(wall)} calls and hits "
        f"{wall['hits']} of {wall['n']}."
    )


def write_summary(path: Path, rows: list[dict[str, str]]) -> None:
    fresh = summarise(rows)
    have = {
        (row["system"], row["arm"], row["code_revision"]) for row in read_csv(path)
    }
    missing = [
        row
        for row in fresh
        if (row["system"], row["arm"], row["code_revision"]) not in have
    ]
    if missing or not path.is_file():
        append_rows(path, SUMMARY_FIELDS, missing)


def threads() -> None:
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "RAYON_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
        "GOTO_NUM_THREADS",
        "OMP_THREAD_LIMIT",
    ):
        os.environ[name] = "1"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--claim", action="store_true", help="print the sentence built from --summary")
    parser.add_argument("--anneal", type=Path)
    parser.add_argument("--driver", type=Path)
    parser.add_argument("--crest", type=Path)
    parser.add_argument("--xtb", type=Path)
    parser.add_argument("--engine", type=Path)
    parser.add_argument("--rows", type=Path, default=ROOT / "bench" / "crest-equal" / "rows.csv")
    parser.add_argument("--summary", type=Path, default=ROOT / "bench" / "crest-equal" / "summary.csv")
    parser.add_argument("--counts", type=Path, default=ROOT / "bench" / "crest-equal" / "call-counts.csv")
    parser.add_argument("--logs", type=Path, default=ROOT / "bench" / "crest-equal" / "logs")
    parser.add_argument("--stamp", type=Path)
    parser.add_argument("--head-file", type=Path)
    parser.add_argument("--core", default="0")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--arm", action="append", choices=ARMS)
    args = parser.parse_args()
    if args.claim:
        print(claim_sentence(read_csv(args.summary)))
        return
    if args.summarize and args.anneal is None:
        write_summary(args.summary, read_csv(args.rows))
        return
    for name in ("anneal", "driver", "crest", "xtb", "engine"):
        if getattr(args, name) is None:
            raise SystemExit(f"--{name} is required")
    if socket.gethostname() != "bench":
        raise SystemExit("hostname is not bench")
    threads()
    os.environ["RGPOT_XTB_ENGINE"] = str(args.engine)
    os.environ["CREST_EQUAL_CORE"] = args.core
    revision = revision_of(args.head_file)
    host = socket.gethostname()
    seeds = tuple(args.seed) if args.seed else SEEDS
    arms = tuple(args.arm) if args.arm else ARMS
    with tempfile.TemporaryDirectory(prefix="crest-equal-") as scratch:
        scratch_path = Path(scratch)
        check_reference(args.xtb, scratch_path / "reference")
        starts = scratch_path / "starts"
        for seed in seeds:
            where = starts / str(seed)
            where.mkdir(parents=True)
            subprocess.run(
                [
                    "taskset",
                    "-c",
                    args.core,
                    str(args.anneal),
                    str(MOLECULES),
                    str(BUDGET),
                    str(seed),
                    str(where),
                ],
                env=dict(os.environ, H2H_START_ONLY="1"),
                stdout=subprocess.DEVNULL,
                check=True,
            )
        done = {
            (int(row["seed"]), row["arm"], row["code_revision"])
            for row in read_csv(args.rows)
            if row.get("system") == SYSTEM and row.get("budget") == str(BUDGET)
        }
        pairs = [(seed, arm) for seed in seeds for arm in arms]
        random.Random(SCHEDULE_SEED).shuffle(pairs)
        for seed, arm in pairs:
            if (seed, arm, revision) in done:
                print(f"skip {arm} {seed}", flush=True)
                continue
            start = starts / str(seed) / "start.xyz"
            where = scratch_path / arm / str(seed)
            print(f"start {arm} {seed}", flush=True)
            if arm == "anneal":
                result = run_anneal(args.anneal, seed, start, where, args.core)
            else:
                result = run_crest(
                    args.crest, args.xtb, args.driver, start, where, args.core, arm
                )
            crest_out = result.pop("crest_out", None)
            count = {
                "arm": arm,
                "driver_calls": result.pop("driver_calls", result["charged"]),
                "driver_stop": result.pop("driver_stop", ""),
                "crest_log_calls": result.pop("crest_log_calls", ""),
                "charged": result["charged"],
                "crest_out_sha256": "",
            }
            if crest_out is not None and crest_out.is_file():
                count["crest_out_sha256"] = keep_log(crest_out, args.logs, arm, seed, revision)
            result = score(args.xtb, result, where / "polish")
            row = {
                "system": SYSTEM,
                "seed": seed,
                "budget": BUDGET,
                "code_revision": revision,
                "host": host,
                "date": datetime.datetime.now(datetime.timezone.utc).date().isoformat(),
                **result,
            }
            record_row(args.counts, args.rows, {**row, **count}, row)
            print(
                f"done {arm} {seed} hit {row['hit']} calls {row['charged']} "
                f"wall {row['wall_s']} polished {row['polished_Eh']}",
                flush=True,
            )
        write_summary(args.summary, read_csv(args.rows))
    if args.stamp is not None:
        args.stamp.parent.mkdir(parents=True, exist_ok=True)
        args.stamp.write_text(f"system {SYSTEM}\ncode_revision {revision}\nhost {host}\n")


if __name__ == "__main__":
    main()
