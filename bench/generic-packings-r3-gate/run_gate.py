#!/usr/bin/env python3
"""Run and summarize the r3 full-panel generic-packings gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEEDS = tuple(range(2000, 2012))
HIT_TOL = 1.0e-3
LAYOUT_ARMS = ("D", "D2", "L2", "L3", "L4", "L5")
CORE_ARMS = ("D", "D2", "G", "P")
REPLACEMENT_CAP = 2


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise SystemExit(f"cannot load {path.name}")
    spec.loader.exec_module(module)
    return module


panel = load("r3_gmin_panel", ROOT / "scripts/gmin_panel/run_panel.py")
morse = load("r3_morse_panel", ROOT / "scripts/morse_h2h/run_campaign.py")
sc = load("r3_sc_panel", ROOT / "scripts/metal_h2h/run_sc.py")

# (system, driver, budget, reference)
CELLS: list[tuple[str, str, int, float]] = []
for _name, _kind, _n, _ref, _rho, _driver, _binary in panel.SYSTEMS:
    CELLS.append((_name, _driver.split("+")[0], panel.PAPER_BUDGETS[_name], _ref))
for _name, _n, _rho, _budget, _ref in morse.SYSTEMS:
    if _name not in panel.PAPER_BUDGETS:
        CELLS.append((_name, "h2h_morse", _budget, _ref))
for _name, _budget in sc.PAPER_BUDGETS.items():
    CELLS.append((_name, "h2h_sc", _budget, sc.REFERENCES[_name]))
CELL = {cell[0]: cell for cell in CELLS}

THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "RAYON_NUM_THREADS",
    "TBB_NUM_THREADS",
)

ROW_FIELDS = [
    "session",
    "system",
    "driver",
    "arm",
    "seed",
    "slot",
    "attempt",
    "budget",
    "reference",
    "hit",
    "driver_hit",
    "evals_to_hit",
    "charged",
    "wall_s",
    "best",
    "pot_s",
    "outside_s",
    "outside_per_call_s",
    "rss_anon_self_peak_kb",
    "rss_anon_external_peak_kb",
    "vm_hwm_self_kb",
    "vm_hwm_external_kb",
    "ru_maxrss_kb",
    "packing_offered",
    "packing_hit",
    "packing_probe_calls",
    "code_revision",
    "code_tree",
    "binary_sha256",
    "text_sha256",
    "instrumentation_sha256",
    "harness_sha256",
    "machine",
    "utc",
]

METRIC = re.compile(
    r"R3_GATE_METRICS,"
    r"rss_anon_self_peak_kb=(\d+),"
    r"vm_hwm_kb=(\d+),"
    r"ru_maxrss_kb=(\d+),"
    r"packing_offered=(\d+),"
    r"packing_hit=(\d+),"
    r"packing_probe_calls=(\d+)"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_env(sc_lib: str | None) -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("H2H_", "ANNEAL_"))
    }
    for key in THREAD_VARS:
        env[key] = "1"
    if sc_lib:
        env["RGPOT_SC_LIB"] = sc_lib
    return env


def git(tree: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=tree,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def hits(best: float, reference: float) -> bool:
    return math.isfinite(best) and best <= reference + HIT_TOL


def arm_order(arms: list[str], seed: int) -> list[str]:
    return list(arms) if (seed - SEEDS[0]) % 2 == 0 else list(reversed(arms))


def external_memory_kb(pid: int) -> tuple[int, int]:
    try:
        text = Path(f"/proc/{pid}/status").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return 0, 0
    fields = {}
    for line in text.splitlines():
        if line.startswith(("RssAnon:", "VmHWM:")):
            key, value, _unit = line.split()
            fields[key.rstrip(":")] = int(value)
    return fields.get("RssAnon", 0), fields.get("VmHWM", 0)


def parse_result(text: str, system: str, seed: int, budget: int) -> dict[str, str]:
    headers = [line for line in text.splitlines() if line.startswith("system,code,seed,")]
    data = [line for line in text.splitlines() if line.startswith(f"{system},anneal,{seed},")]
    metrics = METRIC.findall(text)
    if not headers or len(data) != 1 or len(metrics) != 1:
        raise RuntimeError(f"{system} seed {seed}: incomplete output")
    row = dict(zip(headers[-1].split(","), data[0].split(",")))
    if int(float(row["budget"])) != budget:
        raise RuntimeError(f"{system} seed {seed}: wrong budget")
    metric = metrics[0]
    row.update(
        {
            "rss_anon_self_peak_kb": metric[0],
            "vm_hwm_self_kb": metric[1],
            "ru_maxrss_kb": metric[2],
            "packing_offered": metric[3],
            "packing_hit": metric[4],
            "packing_probe_calls": metric[5],
        }
    )
    return row


def check_budget(
    binary: Path,
    system: str,
    budget: int,
    reference: float,
    env: dict[str, str],
) -> None:
    proc = subprocess.run(
        [str(binary), "budget", system],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if proc.returncode != 0 or not lines:
        raise SystemExit(f"{binary.name} cannot report {system} budget")
    if int(float(lines[0])) != budget:
        raise SystemExit(f"{binary.name} has the wrong {system} budget")
    if len(lines) > 1 and abs(float(lines[1]) - reference) > 1.0e-6:
        raise SystemExit(f"{binary.name} has the wrong {system} reference")


def run_one(
    binary: Path,
    cell: tuple[str, str, int, float],
    seed: int,
    env: dict[str, str],
    log: Path,
) -> dict[str, str]:
    system, _driver, budget, reference = cell
    command = ["taskset", "-c", "0", str(binary), "run", system, str(seed)]
    proc = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    rss_peak = 0
    hwm_peak = 0
    while proc.poll() is None:
        rss, hwm = external_memory_kb(proc.pid)
        rss_peak = max(rss_peak, rss)
        hwm_peak = max(hwm_peak, hwm)
        time.sleep(0.002)
    stdout, stderr = proc.communicate()
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(f"run {system} {seed}\n{stdout}{stderr}")
    if proc.returncode != 0:
        raise RuntimeError(f"{system} seed {seed}: exit {proc.returncode}")
    raw = parse_result(stdout + stderr, system, seed, budget)
    best = float(raw["best"])
    hit = hits(best, reference)
    driver_hit = raw["hit"].strip().lower() in ("1", "true")
    evals = int(float(raw["evals_to_hit"]))
    charged = int(float(raw["charged"]))
    wall = float(raw["wall_s"])
    pot = float(raw["pot_s"])
    outside = float(raw.get("outside_s", max(wall - pot, 0.0)))
    if hit != driver_hit:
        raise RuntimeError(f"{system} seed {seed}: hit flag differs from final energy")
    return {
        "hit": "1" if hit else "0",
        "driver_hit": "1" if driver_hit else "0",
        "evals_to_hit": str(evals),
        "charged": str(charged),
        "wall_s": f"{wall:.6f}",
        "best": raw["best"],
        "pot_s": f"{pot:.6f}",
        "outside_s": f"{outside:.6f}",
        "outside_per_call_s": f"{outside / max(charged, 1):.9e}",
        "rss_anon_self_peak_kb": raw["rss_anon_self_peak_kb"],
        "rss_anon_external_peak_kb": str(rss_peak),
        "vm_hwm_self_kb": raw["vm_hwm_self_kb"],
        "vm_hwm_external_kb": str(hwm_peak),
        "ru_maxrss_kb": raw["ru_maxrss_kb"],
        "packing_offered": raw["packing_offered"],
        "packing_hit": raw["packing_hit"],
        "packing_probe_calls": raw["packing_probe_calls"],
    }


def append_csv(path: Path, fields: list[str], row: dict[str, str]) -> None:
    fresh = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        if fresh:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_text_manifest(path: Path) -> dict[tuple[str, str], str]:
    rows = read_rows(path)
    return {(row["arm"], row["driver"]): row["text_sha256"] for row in rows}


def run(args) -> int:
    if os.sched_getaffinity(0) != {0}:
        raise SystemExit("runner is not pinned to core 0")
    arms = [spec.split("=", 1) for spec in args.arm]
    names = [name for name, _source in arms]
    if len(set(names)) != len(names):
        raise SystemExit("arm names repeat")
    if tuple(name for name in names if name in CORE_ARMS) != CORE_ARMS:
        raise SystemExit("core arms must be ordered D,D2,G,P")
    sources = {name: Path(source) for name, source in arms}
    binaries = {name: Path(args.bin_root) / f"{name}-bin" for name in names}
    text_hashes = read_text_manifest(Path(args.text_manifest))
    env = run_env(args.sc_lib)
    systems = args.system or [cell[0] for cell in CELLS]
    provenance = {
        name: (
            git(source, "rev-parse", "HEAD"),
            git(source, "rev-parse", "HEAD^{tree}"),
        )
        for name, source in sources.items()
    }
    for name in names:
        for system in systems:
            _system, driver, budget, reference = CELL[system]
            check_budget(binaries[name] / driver, system, budget, reference, env)
            if (name, driver) not in text_hashes:
                raise SystemExit(f"missing text hash for {name} {driver}")
    harness = sha256_file(Path(__file__))
    instrumentation = sha256_file(Path(args.instrumentation))
    rows_path = Path(args.rows)
    existing = read_rows(rows_path)
    prior_sessions = {row["session"] for row in existing}
    if len(prior_sessions) > 1:
        raise SystemExit("journal contains more than one session")
    session = (
        next(iter(prior_sessions))
        if prior_sessions
        else time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    )
    done = {(row["system"], row["arm"], row["seed"]) for row in existing}
    replacements = {}
    for row in existing:
        replacements[row["system"]] = replacements.get(row["system"], 0) + int(
            row["attempt"]
        )
    for system in systems:
        cell = CELL[system]
        cell_invalid = False
        for seed in SEEDS:
            if cell_invalid:
                break
            for slot, name in enumerate(arm_order(names, seed)):
                if (system, name, str(seed)) in done:
                    continue
                attempt = 0
                while True:
                    try:
                        result = run_one(
                            binaries[name] / cell[1],
                            cell,
                            seed,
                            env,
                            Path(args.logs) / system / f"{name}-{seed}-{attempt}.log",
                        )
                        break
                    except Exception as error:
                        attempt += 1
                        replacements[system] = replacements.get(system, 0) + 1
                        append_csv(
                            Path(args.issues),
                            ["session", "system", "arm", "seed", "attempt", "error"],
                            {
                                "session": session,
                                "system": system,
                                "arm": name,
                                "seed": str(seed),
                                "attempt": str(attempt),
                                "error": str(error).replace("\n", " "),
                            },
                        )
                        if replacements[system] > REPLACEMENT_CAP:
                            cell_invalid = True
                            break
                if cell_invalid:
                    break
                row = {
                    "session": session,
                    "system": system,
                    "driver": cell[1],
                    "arm": name,
                    "seed": str(seed),
                    "slot": str(slot),
                    "attempt": str(attempt),
                    "budget": str(cell[2]),
                    "reference": f"{cell[3]:.8f}",
                    **result,
                    "code_revision": provenance[name][0],
                    "code_tree": provenance[name][1],
                    "binary_sha256": sha256_file(binaries[name] / cell[1]),
                    "text_sha256": text_hashes[(name, cell[1])],
                    "instrumentation_sha256": instrumentation,
                    "harness_sha256": harness,
                    "machine": args.machine_key,
                    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                append_csv(rows_path, ROW_FIELDS, row)
                print(
                    f"done {system} {name} {seed} hit={row['hit']} "
                    f"rss={row['rss_anon_self_peak_kb']}",
                    flush=True,
                )
    return 0


def quantiles(values: list[float]) -> tuple[float, float]:
    median = panel.type7(values, 0.5)
    iqr = panel.type7(values, 0.75) - panel.type7(values, 0.25)
    return median, iqr


def ratio(left: float, right: float) -> float:
    return left / right if right else math.inf


def ranked_calls(group: list[dict[str, str]]) -> list[float]:
    return [
        float(row["evals_to_hit"]) if row["hit"] == "1" else math.inf
        for row in group
    ]


def summarize(args) -> int:
    rows = read_rows(Path(args.rows))
    issues = read_rows(Path(args.issues))
    by = {}
    for row in rows:
        by.setdefault((row["system"], row["arm"]), {})[int(row["seed"])] = row
    fields = [
        "system",
        "status",
        "hits_D",
        "hits_D2",
        "hits_G",
        "hits_P",
        "hits_ok",
        "outside_median_G",
        "outside_iqr_G",
        "outside_median_D",
        "outside_iqr_D",
        "outside_ratio_G_D",
        "outside_band",
        "outside_mw_p",
        "outside_ok",
        "calls_median_G",
        "calls_iqr_G",
        "calls_median_D",
        "calls_iqr_D",
        "calls_ratio_G_D",
        "calls_mw_p",
        "rss_median_G_kb",
        "rss_iqr_G_kb",
        "rss_median_D_kb",
        "rss_iqr_D_kb",
        "rss_delta_kb",
        "rss_ratio_G_D",
        "rss_pooled_band",
        "rss_two_page_pass",
        "rss_ok",
        "offered_G",
        "packing_hits_G",
        "probe_calls_median_G",
        "vm_hwm_median_G_kb",
        "ru_maxrss_median_G_kb",
        "external_rss_median_G_kb",
        "replacements",
    ]
    output = []
    for system, _driver, _budget, _reference in CELLS:
        groups = {}
        for arm in (*CORE_ARMS, "L2", "L3", "L4", "L5"):
            indexed = by.get((system, arm), {})
            groups[arm] = [indexed[seed] for seed in SEEDS if seed in indexed]
        replacements = sum(
            int(row["attempt"]) for row in rows if row["system"] == system
        ) + len([row for row in issues if row["system"] == system])
        if any(len(group) != len(SEEDS) for group in groups.values()):
            output.append(
                {
                    "system": system,
                    "status": "INCONCLUSIVE",
                    "replacements": str(replacements),
                }
            )
            continue
        hits_by_arm = {
            arm: sum(row["hit"] == "1" for row in groups[arm]) for arm in CORE_ARMS
        }
        hit_spread = abs(hits_by_arm["D2"] - hits_by_arm["D"])
        hits_ok = hits_by_arm["G"] >= hits_by_arm["D"] - hit_spread
        outside = {
            arm: quantiles(
                [float(row["outside_per_call_s"]) for row in groups[arm]]
            )
            for arm in CORE_ARMS
        }
        outside_ratio = ratio(outside["G"][0], outside["D"][0])
        outside_aa = ratio(outside["D2"][0], outside["D"][0])
        outside_band = max(outside_aa, 1.0 / outside_aa)
        outside_ok = outside_ratio <= 1.10 or outside_ratio <= outside_band
        _u, outside_p = panel.mann_whitney(
            [float(row["outside_per_call_s"]) for row in groups["G"]],
            [float(row["outside_per_call_s"]) for row in groups["D"]],
        )
        calls = {
            arm: quantiles([float(row["evals_to_hit"]) for row in groups[arm]])
            for arm in CORE_ARMS
        }
        _u, calls_p = panel.mann_whitney(
            ranked_calls(groups["G"]), ranked_calls(groups["D"])
        )
        rss = {
            arm: quantiles(
                [float(row["rss_anon_self_peak_kb"]) for row in groups[arm]]
            )
            for arm in groups
        }
        rss_ratio = ratio(rss["G"][0], rss["D"][0])
        placement_ratios = [ratio(rss[arm][0], rss["D"][0]) for arm in LAYOUT_ARMS]
        pooled_band = max(placement_ratios)
        rss_delta = rss["G"][0] - rss["D"][0]
        two_page_pass = abs(rss_delta) <= 8.0
        rss_ok = rss_ratio <= pooled_band or two_page_pass
        offered = sum(row["packing_offered"] == "1" for row in groups["G"])
        packing_hits = sum(row["packing_hit"] == "1" for row in groups["G"])
        counters_ok = offered == len(SEEDS)
        valid_replacements = replacements <= REPLACEMENT_CAP
        status = (
            "PASS"
            if hits_ok and outside_ok and rss_ok and counters_ok and valid_replacements
            else "FAIL"
        )
        probes = quantiles(
            [float(row["packing_probe_calls"]) for row in groups["G"]]
        )
        vm_hwm = quantiles([float(row["vm_hwm_self_kb"]) for row in groups["G"]])
        ru_maxrss = quantiles([float(row["ru_maxrss_kb"]) for row in groups["G"]])
        external = quantiles(
            [float(row["rss_anon_external_peak_kb"]) for row in groups["G"]]
        )
        output.append(
            {
                "system": system,
                "status": status,
                "hits_D": str(hits_by_arm["D"]),
                "hits_D2": str(hits_by_arm["D2"]),
                "hits_G": str(hits_by_arm["G"]),
                "hits_P": str(hits_by_arm["P"]),
                "hits_ok": str(int(hits_ok)),
                "outside_median_G": f"{outside['G'][0]:.9e}",
                "outside_iqr_G": f"{outside['G'][1]:.9e}",
                "outside_median_D": f"{outside['D'][0]:.9e}",
                "outside_iqr_D": f"{outside['D'][1]:.9e}",
                "outside_ratio_G_D": f"{outside_ratio:.6f}",
                "outside_band": f"{outside_band:.6f}",
                "outside_mw_p": f"{outside_p:.6g}",
                "outside_ok": str(int(outside_ok)),
                "calls_median_G": f"{calls['G'][0]:.1f}",
                "calls_iqr_G": f"{calls['G'][1]:.1f}",
                "calls_median_D": f"{calls['D'][0]:.1f}",
                "calls_iqr_D": f"{calls['D'][1]:.1f}",
                "calls_ratio_G_D": f"{ratio(calls['G'][0], calls['D'][0]):.6f}",
                "calls_mw_p": f"{calls_p:.6g}",
                "rss_median_G_kb": f"{rss['G'][0]:.1f}",
                "rss_iqr_G_kb": f"{rss['G'][1]:.1f}",
                "rss_median_D_kb": f"{rss['D'][0]:.1f}",
                "rss_iqr_D_kb": f"{rss['D'][1]:.1f}",
                "rss_delta_kb": f"{rss_delta:.1f}",
                "rss_ratio_G_D": f"{rss_ratio:.6f}",
                "rss_pooled_band": f"{pooled_band:.6f}",
                "rss_two_page_pass": str(int(two_page_pass)),
                "rss_ok": str(int(rss_ok)),
                "offered_G": str(offered),
                "packing_hits_G": str(packing_hits),
                "probe_calls_median_G": f"{probes[0]:.1f}",
                "vm_hwm_median_G_kb": f"{vm_hwm[0]:.1f}",
                "ru_maxrss_median_G_kb": f"{ru_maxrss[0]:.1f}",
                "external_rss_median_G_kb": f"{external[0]:.1f}",
                "replacements": str(replacements),
            }
        )
    with Path(args.summary).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)
    if any(row["status"] == "INCONCLUSIVE" for row in output):
        verdict = "INCONCLUSIVE"
    elif all(row["status"] == "PASS" for row in output):
        verdict = "PASS"
    else:
        verdict = "FAIL"
    if args.readme:
        write_readme(Path(args.readme), rows, output, verdict)
    print(f"verdict {verdict} rows={len(rows)}")
    return 0


def write_readme(
    path: Path,
    rows: list[dict[str, str]],
    summary: list[dict[str, str]],
    verdict: str,
) -> None:
    sessions = sorted({row["session"] for row in rows})
    machines = sorted({row["machine"] for row in rows})
    lines = [
        "# generic-packings-r3 full-panel gate",
        "",
        "The r2 rows and verdict remain unchanged in their original directory.",
        "They are superseded because r2 polled RssAnon externally, used only",
        "the same-layout A/A band, and obtained packing counters by replay.",
        "",
        f"Verdict: **{verdict}**.",
        f"Rows: {len(rows)}. Sessions: {', '.join(sessions) if sessions else 'none'}.",
        f"App-keyed machine hash: {', '.join(machines) if machines else 'none'}.",
        "",
        "RssAnon is sampled by a thread inside each timed process. The gate",
        "uses its per-cell median. G passes when G/D is no greater than the",
        "largest ratio from five D layouts and D2, or when the median absolute",
        "difference is at most 8 kB. VmHWM, ru_maxrss, and external polling",
        "are report-only. A G row with offered=0 invalidates its cell.",
        "",
        "| cell | hits D/D2/G/P | outside G/D (band) | "
        "RssAnon G/D (band; delta kB) | offered/hit | status |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in summary:
        if row["status"] == "INCONCLUSIVE":
            lines.append(f"| {row['system']} | incomplete | - | - | - | INCONCLUSIVE |")
            continue
        lines.append(
            f"| {row['system']} | {row['hits_D']}/{row['hits_D2']}/"
            f"{row['hits_G']}/{row['hits_P']} | {row['outside_ratio_G_D']} "
            f"({row['outside_band']}) | {row['rss_ratio_G_D']} "
            f"({row['rss_pooled_band']}; {row['rss_delta_kb']}) | "
            f"{row['offered_G']}/{row['packing_hits_G']} | {row['status']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def self_test() -> None:
    assert len(CELLS) == 24
    assert CELL["lj38"][2] == 400_000
    assert CELL["lj98"][2] == 4_000_000
    assert arm_order(list(CORE_ARMS), 2000) == list(CORE_ARMS)
    assert arm_order(list(CORE_ARMS), 2001) == list(reversed(CORE_ARMS))
    assert hits(-1.0, -1.0005)
    assert not hits(-1.0, -1.002)
    print("ok")


def main(argv: list[str]) -> int:
    if argv[1:] == ["--self-test"]:
        self_test()
        return 0
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", required=True)
    parser.add_argument("--arm", action="append", default=[])
    parser.add_argument("--bin-root")
    parser.add_argument("--logs")
    parser.add_argument("--issues", required=True)
    parser.add_argument("--sc-lib")
    parser.add_argument("--system", action="append")
    parser.add_argument("--text-manifest")
    parser.add_argument("--instrumentation")
    parser.add_argument("--machine-key")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--summary")
    parser.add_argument("--readme")
    args = parser.parse_args(argv[1:])
    if args.summarize:
        if not args.summary:
            parser.error("--summary is required")
        return summarize(args)
    required = (
        args.arm,
        args.bin_root,
        args.logs,
        args.text_manifest,
        args.instrumentation,
        args.machine_key,
    )
    if not all(required):
        parser.error("run arguments are incomplete")
    if not re.fullmatch(r"[0-9a-f]{64}", args.machine_key):
        parser.error("--machine-key must be a sha256")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
