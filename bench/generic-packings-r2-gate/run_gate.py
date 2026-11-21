# Copyright (c) anneal developers
"""Four-arm same-session gate on the paper panel.

The panel is every cell of the paper head-to-head tables that the anneal
drivers run: the GMIN panel of ``scripts/gmin_panel/run_panel.py`` (LJ38 to
LJ150, Au55, M55 and M80 at rho 6 and 14), the Morse cells of
``scripts/morse_h2h/run_campaign.py`` not already in it (M38 at rho 14 and
6), and the twelve-seed Sutton-Chen cells of ``scripts/metal_h2h/run_sc.py``.
Budgets and references are those tables' ``PAPER_BUDGETS`` and reference
energies; each binary's own ``budget <system>`` must agree before the run.

Every arm is a release build of its own revision with its own examples:
``h2h_gmin_proof`` for Lennard-Jones, ``h2h_metal`` for Au55, ``h2h_morse``
for Morse and ``h2h_sc`` for Sutton-Chen. The command is ``run <system>
<seed>``; no ``H2H_*`` or ``ANNEAL_*`` variable reaches a run, so the budget
is the driver's paper budget and the only stop besides it is the target.

Seeds 2000-2011. For each cell and seed every arm runs once, in the arm
order given on even seed offsets and reversed on odd ones. One process,
``taskset -c 0``, every thread count 1. Rows are appended and fsynced after
every run. While each child is alive, the harness samples ``RssAnon`` and
``VmHWM`` from its process status. A row carries the session, a sha256 of
the machine id, the revision and tree of the arm's source, and the sha256
of the binary.

``--summarize`` reads the rows and writes the per-cell table.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


panel = load("gate_gmin_panel", ROOT / "scripts" / "gmin_panel" / "run_panel.py")
morse = load("gate_morse_h2h", ROOT / "scripts" / "morse_h2h" / "run_campaign.py")
sc = load("gate_metal_sc", ROOT / "scripts" / "metal_h2h" / "run_sc.py")

SEEDS = tuple(range(2000, 2012))
HIT_TOL = 1.0e-3

# (system, driver, budget, reference)
CELLS: list[tuple[str, str, int, float]] = []
for _name, _kind, _n, _ref, _rho, _driver, _binary in panel.SYSTEMS:
    CELLS.append((_name, _driver.split("+")[0], panel.PAPER_BUDGETS[_name], _ref))
for _name, _n, _rho, _budget, _ref in morse.SYSTEMS:
    if _name not in panel.PAPER_BUDGETS:
        CELLS.append((_name, "h2h_morse", _budget, _ref))
for _name, _budget in sc.PAPER_BUDGETS.items():
    CELLS.append((_name, "h2h_sc", _budget, sc.REFERENCES[_name]))
CELL = {c[0]: c for c in CELLS}

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
    "rss_anon_peak_kb",
    "vm_hwm_peak_kb",
    "packing_offered",
    "packing_hit",
    "packing_probe_calls",
    "charged_to_hit",
    "prior_core",
    "species",
    "code_revision",
    "code_tree",
    "binary_sha256",
    "harness_sha256",
    "machine",
    "utc",
]


def hits(best: float, reference: float) -> bool:
    return math.isfinite(best) and best <= reference + HIT_TOL


def run_env(sc_lib: str | None) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("H2H_", "ANNEAL_"))}
    for key in THREAD_VARS:
        env[key] = "1"
    if sc_lib:
        env["RGPOT_SC_LIB"] = sc_lib
    return env


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def machine_hash() -> str:
    for path in (Path("/etc/machine-id"), Path("/var/lib/dbus/machine-id")):
        if path.is_file() and path.read_text().strip():
            return hashlib.sha256(path.read_text().strip().encode()).hexdigest()
    raise SystemExit("no machine id")


def git(tree: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=tree, capture_output=True, text=True, check=True).stdout.strip()


def arm_order(arms: list[str], seed: int) -> list[str]:
    return list(arms) if (seed - SEEDS[0]) % 2 == 0 else list(reversed(arms))


def check_budget(binary: Path, system: str, budget: int, reference: float, env: dict[str, str]) -> None:
    proc = subprocess.run([str(binary), "budget", system], env=env, capture_output=True, text=True, check=False)
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if proc.returncode != 0 or not lines:
        raise SystemExit(f"{binary} budget {system} exited {proc.returncode}: {proc.stderr[-500:]}")
    if int(float(lines[0])) != budget:
        raise SystemExit(f"{binary} {system}: budget {lines[0]} is not {budget}")
    if len(lines) > 1 and abs(float(lines[1]) - reference) > 1e-6:
        raise SystemExit(f"{binary} {system}: reference {lines[1]} is not {reference}")


def parse_row(text: str, system: str, seed: int, budget: int) -> dict[str, str]:
    lines = [line for line in text.splitlines() if line.startswith("system,code,seed,")]
    data = [line for line in text.splitlines() if line.startswith(f"{system},anneal,{seed},")]
    if len(lines) < 1 or len(data) != 1:
        raise RuntimeError(f"{system} seed {seed}: expected one result row\n{text[-2000:]}")
    header = lines[-1].split(",")
    row = dict(zip(header, data[0].split(",")))
    if int(float(row["budget"])) != budget:
        raise RuntimeError(f"{system} seed {seed}: budget {row['budget']}")
    return row


def memory_kb(pid: int) -> tuple[int, int]:
    try:
        fields = {}
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith(("RssAnon:", "VmHWM:")):
                key, value, _unit = line.split()
                fields[key.rstrip(":")] = int(value)
        return fields.get("RssAnon", 0), fields.get("VmHWM", 0)
    except (FileNotFoundError, ProcessLookupError):
        return 0, 0


def run_one(binary: Path, cell: tuple[str, str, int, float], seed: int, env: dict[str, str], log: Path) -> dict[str, str]:
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
        rss, hwm = memory_kb(proc.pid)
        rss_peak = max(rss_peak, rss)
        hwm_peak = max(hwm_peak, hwm)
        time.sleep(0.002)
    stdout, stderr = proc.communicate()
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(" ".join(command) + "\n" + stdout + stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"{binary} {system} {seed} exited {proc.returncode}\n{stderr[-2000:]}")
    row = parse_row(stdout + stderr, system, seed, budget)
    best = float(row["best"])
    hit = hits(best, reference)
    driver_hit = row["hit"].strip().lower() in ("1", "true")
    evals = int(float(row["evals_to_hit"]))
    charged = int(float(row["charged"]))
    wall = float(row["wall_s"])
    pot = float(row["pot_s"])
    outside = float(row["outside_s"]) if "outside_s" in row else max(wall - pot, 0.0)
    if hit != driver_hit:
        raise RuntimeError(f"{system} {seed}: driver hit {driver_hit} but best {best} against {reference}")
    if hit and not 0 < evals <= charged <= budget:
        raise RuntimeError(f"{system} {seed}: hit with evals {evals} charged {charged}")
    if not hit and (evals != budget or charged != budget):
        raise RuntimeError(f"{system} {seed}: miss stopped at evals {evals} charged {charged} of {budget}")
    return {
        "hit": "1" if hit else "0",
        "driver_hit": "1" if driver_hit else "0",
        "evals_to_hit": str(evals),
        "charged": str(charged),
        "wall_s": f"{wall:.6f}",
        "best": row["best"],
        "pot_s": f"{pot:.6f}",
        "outside_s": f"{outside:.6f}",
        "outside_per_call_s": f"{outside / max(charged, 1):.6e}",
        "rss_anon_peak_kb": str(rss_peak),
        "vm_hwm_peak_kb": str(hwm_peak),
    }


def append_row(path: Path, row: dict[str, str]) -> None:
    fresh = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS, lineterminator="\n")
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


def run(args) -> int:
    if os.sched_getaffinity(0) != {0}:
        print(f"pinned to {sorted(os.sched_getaffinity(0))}, expected core 0", file=sys.stderr)
        return 2
    arms = [spec.split("=", 1) for spec in args.arm]
    names = [name for name, _ in arms]
    if len(set(names)) != len(names):
        raise SystemExit("arm names repeat")
    sources = {name: Path(src) for name, src in arms}
    bins = {name: Path(args.bin_root) / f"{name}-bin" for name in names}
    env = run_env(args.sc_lib)
    systems = args.system or [c[0] for c in CELLS]
    provenance = {}
    for name in names:
        tree = sources[name]
        if git(tree, "status", "--porcelain"):
            raise SystemExit(f"arm {name} source is not clean")
        provenance[name] = (git(tree, "rev-parse", "HEAD"), git(tree, "rev-parse", "HEAD^{tree}"))
        for system in systems:
            _s, driver, budget, reference = CELL[system]
            check_budget(bins[name] / driver, system, budget, reference, env)
    machine = machine_hash()
    harness = sha256_file(Path(__file__))
    session = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    done = {(r["system"], r["arm"], r["seed"]) for r in read_rows(Path(args.rows))}
    print(f"session {session} arms {' '.join(names)} systems {' '.join(systems)}", flush=True)
    for name in names:
        print(f"arm {name} revision {provenance[name][0]} tree {provenance[name][1]}", flush=True)
    for system in systems:
        cell = CELL[system]
        for seed in SEEDS:
            for slot, name in enumerate(arm_order(names, seed)):
                if (system, name, str(seed)) in done:
                    print(f"skip {system} {name} {seed}", flush=True)
                    continue
                binary = bins[name] / cell[1]
                print(f"start {system} {name} {seed} slot={slot}", flush=True)
                result = run_one(binary, cell, seed, env, Path(args.logs) / system / f"{name}-{seed}.log")
                row = {
                    "session": session,
                    "system": system,
                    "driver": cell[1],
                    "arm": name,
                    "seed": str(seed),
                    "slot": str(slot),
                    "budget": str(cell[2]),
                    "reference": f"{cell[3]:.8f}",
                    **result,
                    "packing_offered": "unknown" if name == args.candidate else "not-applicable",
                    "packing_hit": "unknown" if name == args.candidate else "not-applicable",
                    "packing_probe_calls": "",
                    "charged_to_hit": result["evals_to_hit"] if result["hit"] == "1" else "",
                    "prior_core": "removed",
                    "species": "none",
                    "code_revision": provenance[name][0],
                    "code_tree": provenance[name][1],
                    "binary_sha256": sha256_file(binary),
                    "harness_sha256": harness,
                    "machine": machine,
                    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                append_row(Path(args.rows), row)
                print(
                    f"done {system} {name} {seed} slot={slot} hit={row['hit']} calls={row['evals_to_hit']} "
                    f"wall_s={row['wall_s']} outside_s={row['outside_s']} best={row['best']}",
                    flush=True,
                )
    print("session done", flush=True)
    return 0


def ranked_calls(group: list[dict[str, str]]) -> list[float]:
    return [float(r["evals_to_hit"]) if r["hit"] == "1" else math.inf for r in group]


def quantiles(values: list[float]) -> tuple[float, float]:
    return panel.type7(values, 0.5), panel.type7(values, 0.75) - panel.type7(values, 0.25)


def ratio(left: float, right: float) -> float:
    return left / right if right else math.inf


def symmetric_band(value: float) -> float:
    return max(value, 1.0 / value) if value > 0 else math.inf


def mcnemar(left: list[dict[str, str]], right: list[dict[str, str]]) -> tuple[int, int, float]:
    gained = sum(a["hit"] == "1" and b["hit"] == "0" for a, b in zip(left, right))
    lost = sum(a["hit"] == "0" and b["hit"] == "1" for a, b in zip(left, right))
    discordant = gained + lost
    if discordant == 0:
        return gained, lost, 1.0
    tail = sum(math.comb(discordant, k) for k in range(min(gained, lost) + 1))
    return gained, lost, min(1.0, 2.0 * tail / (2**discordant))


def merge_probes(args) -> int:
    rows = read_rows(Path(args.rows))
    probes = read_rows(Path(args.probes))
    by = {(r["system"], r["seed"]): r for r in probes}
    seen = set()
    for row in rows:
        if row["arm"] != args.candidate:
            continue
        key = (row["system"], row["seed"])
        if key not in by:
            raise SystemExit(f"missing probe diagnostics for {key}")
        probe = by[key]
        seen.add(key)
        for field in ("packing_offered", "packing_hit", "packing_probe_calls"):
            row[field] = probe[field]
        row["charged_to_hit"] = row["evals_to_hit"] if row["hit"] == "1" else ""
    if len(seen) != len(CELLS) * len(SEEDS):
        raise SystemExit(f"merged {len(seen)} diagnostic rows")
    temporary = Path(f"{args.rows}.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, args.rows)
    return 0


def metric_fields(prefix: str) -> list[str]:
    return [f"{prefix}_median_{arm}" for arm in ("G", "D", "D2", "P")] + [
        f"{prefix}_iqr_{arm}" for arm in ("G", "D", "D2", "P")
    ]


def write_readme(path: Path, rows: list[dict[str, str]], summary: list[dict[str, str]], verdict: str) -> None:
    revisions = {}
    trees = {}
    for row in rows:
        revisions.setdefault(row["arm"], row["code_revision"])
        trees.setdefault(row["arm"], row["code_tree"])
    machine = {row["machine"] for row in rows}
    sessions = {row["session"] for row in rows}
    harnesses = {row["harness_sha256"] for row in rows}
    if len(machine) != 1:
        raise SystemExit("rows do not have one machine id")
    if len(sessions) != 1 or len(harnesses) != 1:
        raise SystemExit("rows do not have one session and harness")
    hit_fail = [r["system"] for r in summary if r["hits_ok"] == "0"]
    outside_fail = [r["system"] for r in summary if r["outside_ok"] == "0"]
    rss_fail = [r["system"] for r in summary if r["rss_anon_ok"] == "0"]
    lines = [
        "# generic-packings-r2 full-panel gate",
        "",
        f"Verdict: **{verdict}**.",
        "",
        "## Provenance",
        "",
        f"- Machine id: `{next(iter(machine))}`.",
        f"- Session: `{next(iter(sessions))}`; 1,152 rows.",
        f"- Harness sha256: `{next(iter(harnesses))}`.",
    ]
    for arm in ("D", "D2", "G", "P"):
        lines.append(f"- {arm}: revision `{revisions[arm]}`, tree `{trees[arm]}`.")
    lines.extend(
        [
            "",
            "The panel has 24 cells and seeds 2000 through 2011. Runs are serial",
            "on core 0 with thread counts fixed at one. Arm order is D, D2, G, P",
            "on even seed offsets and the reverse on odd offsets. Hits are",
            "re-derived from final energies with a tolerance of 0.001.",
            "Side-probe accounting is replayed from the same candidate source;",
            "it stops immediately after the deterministic initial probe and is",
            "not used for timing or final-energy rows.",
            "",
            "## Gate clauses",
            "",
            f"- Hit loss beyond A/A spread: {'FAIL' if hit_fail else 'PASS'}"
            + (f" ({', '.join(hit_fail)})" if hit_fail else "."),
            f"- Outside time per call: {'FAIL' if outside_fail else 'PASS'}"
            + (f" ({', '.join(outside_fail)})" if outside_fail else "."),
            f"- Peak RssAnon: {'FAIL' if rss_fail else 'PASS'}"
            + (f" ({', '.join(rss_fail)})" if rss_fail else "."),
            "",
            "VmHWM is reported but does not gate. Calls use the charged count to",
            "the hit; misses rank as positive infinity only in Mann-Whitney.",
            "The memory comparison uses per-cell medians of each process's",
            "sampled peak. Its A/A band is max(D2/D, D/D2).",
            "",
            "## Per-cell results",
            "",
            "| cell | hits D/D2/G/P | calls G med/IQR | outside G/D | "
            "MW calls/outside p | McNemar p | offered/hit | "
            "charged hit/probe med | RssAnon G/D | prior/species | cell |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in summary:
        lines.append(
            f"| {row['system']} | {row['hits_D']}/{row['hits_D2']}/"
            f"{row['hits_G']}/{row['hits_P']} | {row['calls_median_G']}/"
            f"{row['calls_iqr_G']} | {row['outside_ratio_G_D']} | "
            f"{row['mw_p_calls_G_D']}/{row['mw_p_outside_G_D']} | "
            f"{row['mcnemar_p_G_D']} | "
            f"{row['packing_offered_G']}/{row['packing_hits_G']} | "
            f"{row['charged_to_hit_median_G']}/{row['probe_calls_median_G']} | "
            f"{row['rss_anon_ratio_G_D']} | "
            f"{row['prior_core_G']}/{row['species_G']} | "
            f"{'pass' if row['cell_pass'] == '1' else 'FAIL'} |"
        )
    for named in ("lj38", "lj98"):
        row = next(item for item in summary if item["system"] == named)
        lines.extend(
            [
                "",
                f"{named.upper()}: G hits {row['hits_G']}/12; best energy "
                f"{row['best_G']}; charged-to-hit median "
                f"{row['charged_to_hit_median_G']}.",
            ]
        )
    path.write_text("\n".join(lines) + "\n")


def summarize(args) -> int:
    rows = read_rows(Path(args.rows))
    cand, base, aa, ref = args.candidate, args.baseline, args.aa, args.reference
    by: dict[tuple[str, str], dict[int, dict[str, str]]] = {}
    for r in rows:
        by.setdefault((r["system"], r["arm"]), {})[int(r["seed"])] = r
    fields = [
        "system", "budget", "hits_G", "hits_D", "hits_D2", "hits_P",
        "hit_aa_spread", "hits_ok", "best_G",
        *metric_fields("calls"),
        "calls_ratio_G_D", "calls_aa_ratio_D2_D", "mw_p_calls_G_D",
        *metric_fields("outside_per_call"),
        "outside_ratio_G_D", "outside_aa_ratio_D2_D", "mw_p_outside_G_D", "outside_ok",
        *metric_fields("rss_anon_peak_kb"),
        "rss_anon_ratio_G_D", "rss_anon_aa_ratio_D2_D", "rss_anon_ok",
        *metric_fields("vm_hwm_peak_kb"),
        "mcnemar_gained_G_D", "mcnemar_lost_G_D", "mcnemar_p_G_D",
        "packing_offered_G", "packing_hits_G", "probe_calls_median_G", "probe_calls_iqr_G",
        "charged_to_hit_median_G", "charged_to_hit_iqr_G",
        "prior_core_G", "species_G", "cell_pass",
    ]
    out = []
    for system, _driver, budget, _reference in CELLS:
        groups = {}
        for label, name in (("G", cand), ("D", base), ("D2", aa), ("P", ref)):
            group = by.get((system, name), {})
            groups[label] = [group[s] for s in SEEDS if s in group]
        if any(len(g) != len(SEEDS) for g in groups.values()):
            out.append({"system": system, "budget": str(budget), "cell_pass": "incomplete"})
            continue
        h = {k: sum(r["hit"] == "1" for r in g) for k, g in groups.items()}
        calls_g, calls_d = ranked_calls(groups["G"]), ranked_calls(groups["D"])
        _u, calls_p = panel.mann_whitney(calls_g, calls_d)
        metrics = {}
        for prefix, field in (
            ("calls", "evals_to_hit"),
            ("outside_per_call", "outside_per_call_s"),
            ("rss_anon_peak_kb", "rss_anon_peak_kb"),
            ("vm_hwm_peak_kb", "vm_hwm_peak_kb"),
        ):
            for arm, group in groups.items():
                med, iqr = quantiles([float(r[field]) for r in group])
                metrics[f"{prefix}_median_{arm}"] = med
                metrics[f"{prefix}_iqr_{arm}"] = iqr
        calls_ratio = ratio(metrics["calls_median_G"], metrics["calls_median_D"])
        calls_aa = ratio(metrics["calls_median_D2"], metrics["calls_median_D"])
        outside_ratio = ratio(
            metrics["outside_per_call_median_G"],
            metrics["outside_per_call_median_D"],
        )
        outside_aa = ratio(
            metrics["outside_per_call_median_D2"],
            metrics["outside_per_call_median_D"],
        )
        outside_g = [float(r["outside_per_call_s"]) for r in groups["G"]]
        outside_d = [float(r["outside_per_call_s"]) for r in groups["D"]]
        _u, outside_p = panel.mann_whitney(outside_g, outside_d)
        rss_ratio = ratio(
            metrics["rss_anon_peak_kb_median_G"],
            metrics["rss_anon_peak_kb_median_D"],
        )
        rss_aa = ratio(
            metrics["rss_anon_peak_kb_median_D2"],
            metrics["rss_anon_peak_kb_median_D"],
        )
        hit_spread = abs(h["D2"] - h["D"])
        hits_ok = h["G"] >= h["D"] - hit_spread
        outside_ok = outside_ratio <= 1.10 or outside_ratio <= symmetric_band(outside_aa)
        rss_ok = rss_ratio <= symmetric_band(rss_aa)
        gained, lost, mcnemar_p = mcnemar(groups["G"], groups["D"])
        probes = [float(r["packing_probe_calls"]) for r in groups["G"]]
        probe_med, probe_iqr = quantiles(probes)
        charged_hits = [float(r["charged_to_hit"]) for r in groups["G"] if r["charged_to_hit"]]
        charged_med, charged_iqr = quantiles(charged_hits) if charged_hits else (math.inf, math.inf)
        row = {
            "system": system,
            "budget": str(budget),
            "hits_G": str(h["G"]),
            "hits_D": str(h["D"]),
            "hits_D2": str(h["D2"]),
            "hits_P": str(h["P"]),
            "hit_aa_spread": str(hit_spread),
            "hits_ok": "1" if hits_ok else "0",
            "best_G": f"{min(float(r['best']) for r in groups['G']):.8f}",
            "calls_ratio_G_D": f"{calls_ratio:.4f}",
            "calls_aa_ratio_D2_D": f"{calls_aa:.4f}",
            "mw_p_calls_G_D": f"{calls_p:.6g}",
            "outside_ratio_G_D": f"{outside_ratio:.4f}",
            "outside_aa_ratio_D2_D": f"{outside_aa:.4f}",
            "mw_p_outside_G_D": f"{outside_p:.6g}",
            "outside_ok": "1" if outside_ok else "0",
            "rss_anon_ratio_G_D": f"{rss_ratio:.4f}",
            "rss_anon_aa_ratio_D2_D": f"{rss_aa:.4f}",
            "rss_anon_ok": "1" if rss_ok else "0",
            "mcnemar_gained_G_D": str(gained),
            "mcnemar_lost_G_D": str(lost),
            "mcnemar_p_G_D": f"{mcnemar_p:.6g}",
            "packing_offered_G": str(sum(r["packing_offered"] == "1" for r in groups["G"])),
            "packing_hits_G": str(sum(r["packing_hit"] == "1" for r in groups["G"])),
            "probe_calls_median_G": f"{probe_med:.1f}",
            "probe_calls_iqr_G": f"{probe_iqr:.1f}",
            "charged_to_hit_median_G": f"{charged_med:.1f}",
            "charged_to_hit_iqr_G": f"{charged_iqr:.1f}",
            "prior_core_G": ",".join(sorted({r["prior_core"] for r in groups["G"]})),
            "species_G": ",".join(sorted({r["species"] for r in groups["G"]})),
            "cell_pass": "1" if hits_ok and outside_ok and rss_ok else "0",
        }
        for key, value in metrics.items():
            row[key] = f"{value:.6g}"
        out.append(row)
    with Path(args.summary).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(out)
    complete = all(r["cell_pass"] in ("0", "1") for r in out)
    verdict = "PASS" if complete and all(r["cell_pass"] == "1" for r in out) else "FAIL"
    if args.readme:
        write_readme(Path(args.readme), rows, out, verdict)
    print(f"verdict {verdict} complete={int(complete)}")
    return 0


def self_test() -> None:
    assert hits(-1.0, -1.0005) and not hits(-1.0, -1.002) and not hits(float("nan"), 0.0)
    assert arm_order(["A", "B", "C", "D"], 2000) == ["A", "B", "C", "D"]
    assert arm_order(["A", "B", "C", "D"], 2001) == ["D", "C", "B", "A"]
    slots = {}
    for seed in SEEDS:
        for slot, name in enumerate(arm_order(["A", "B", "C", "D"], seed)):
            slots.setdefault(name, []).append(slot)
    assert all(sum(v) == sum(slots["A"]) for v in slots.values())
    assert len(CELLS) == len(CELL)
    assert CELL["m38r14"][2] == 400_000 and CELL["sc108-38"][2] == 400_000
    text = "system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,pot_s,outside_s\n"
    text += "m38r14,anneal,2000,400000,1,10,12,0.5,-144.32105392,0.25,0.2\n"
    row = parse_row(text, "m38r14", 2000, 400_000)
    assert row["outside_s"] == "0.2"
    left = [{"hit": "1"}, {"hit": "0"}]
    right = [{"hit": "0"}, {"hit": "0"}]
    assert mcnemar(left, right) == (1, 0, 1.0)
    print("ok")


def main(argv: list[str]) -> int:
    if argv[1:] == ["--self-test"]:
        self_test()
        return 0
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", required=True)
    parser.add_argument("--arm", action="append", default=[], help="name=source tree; run order on even seeds")
    parser.add_argument("--bin-root", help="directory holding <name>-bin/")
    parser.add_argument("--logs")
    parser.add_argument("--sc-lib")
    parser.add_argument("--system", action="append")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--merge-probes", action="store_true")
    parser.add_argument("--probes")
    parser.add_argument("--summary")
    parser.add_argument("--readme")
    parser.add_argument("--candidate", default="G")
    parser.add_argument("--baseline", default="D")
    parser.add_argument("--aa", default="Dp")
    parser.add_argument("--reference", default="P")
    args = parser.parse_args(argv[1:])
    if args.merge_probes:
        if not args.probes:
            parser.error("--probes is needed with --merge-probes")
        return merge_probes(args)
    if args.summarize:
        if not args.summary:
            parser.error("--summary is needed with --summarize")
        return summarize(args)
    if not (args.arm and args.bin_root and args.logs):
        parser.error("--arm, --bin-root and --logs are needed to run")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
