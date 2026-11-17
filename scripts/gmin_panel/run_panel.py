# Copyright (c) anneal developers
"""anneal against GMIN on the paper panel, both arms in one session.

LJ38, LJ55, LJ75, LJ98, LJ102, LJ103, LJ104 and LJ150 against the
Cambridge Cluster Database minima, Gupta Au55, and Morse M55 and M80 at
rho 6 and 14. Seeds 2000-2011, the paper call budgets in
``PAPER_BUDGETS``. A hit is a quenched energy at or below the reference
plus 0.001; a miss is entered at the budget. Both arms start from the
geometry the anneal example's ``coords`` writes for the seed.

anneal runs through the example its head-to-head rule runs:
``h2h_gmin_proof`` for Lennard-Jones, ``h2h_metal`` for Au55 and
``h2h_morse`` for Morse. ``h2h_gmin_proof`` has no LJ103 entry, so LJ103
runs on a second build of the same revision with
``bench/paper-0120/lj103-table.patch``, which adds the Cambridge energy
and the 4e6 budget and nothing else; its rows carry driver
``h2h_gmin_proof+lj103``. GMIN is one build with the evaluator cap and a
timer inside the Lennard-Jones, Morse and binary Gupta potentials. Its
inputs are ``examples/h2h_gmin_lj.data``, ``examples/h2h_gmin_morse.data``
plus ``MORSE <rho>``, and for Au55 the BGUPTAT gold block of
``scripts/metal_h2h/enter_gold.sh``.

One process, ``taskset -c 0``, one thread. For each system the seeds run
in order, anneal then GMIN. Rows are appended after every run and never
rewritten; a row already stored for the same system, code, seed, code
revision and harness revision is kept and its run skipped. ``code_revision``
is the anneal commit on anneal rows and the GMIN SVNREV on GMIN rows;
``harness_revision`` is the anneal commit whose examples wrote the starts.

``--summarize`` reads the rows and writes ``summary.csv`` and the table in
``bench/gmin-panel/README.md``; with ``--tex`` also table ``tab:gmin``
in ``paper/jctc/manuscript.tex``. The tables carry Fisher's exact test
on hits and the two-sided Mann-Whitney test, normal approximation with
the tie and continuity corrections, on calls to the minimum and on wall
time.
Medians and interquartile ranges are Hyndman-Fan type 7.

This process refuses to run unless the UTS hostname is bench and it is
pinned to core 0.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

SEEDS = tuple(range(2000, 2012))
HIT_TOL = 0.001
GMIN_REV = "aac1548a8aec13b53ad12a8d6904625081685d2e"
PAPER_BUDGETS = {
    "lj38": 400_000,
    "lj55": 1_000_000,
    "lj75": 4_000_000,
    "lj98": 4_000_000,
    "lj102": 4_000_000,
    "lj103": 4_000_000,
    "lj104": 4_000_000,
    "lj150": 4_000_000,
    "au55": 1_000_000,
    "m55r6": 1_000_000,
    "m55r14": 1_000_000,
    "m80r6": 4_000_000,
    "m80r14": 4_000_000,
}
# System, kind, atoms, reference, rho, anneal driver, anneal binary.
# Cambridge Cluster Database: Lennard-Jones, the Cu-Au Gupta archive
# (pure gold at 55) and the Doye-Wales Morse table.
SYSTEMS = (
    ("lj38", "lj", 38, -173.928427, 0.0, "h2h_gmin_proof", "h2h_gmin_proof"),
    ("lj55", "lj", 55, -279.248470, 0.0, "h2h_gmin_proof", "h2h_gmin_proof"),
    ("lj75", "lj", 75, -397.492331, 0.0, "h2h_gmin_proof", "h2h_gmin_proof"),
    ("lj98", "lj", 98, -543.665361, 0.0, "h2h_gmin_proof", "h2h_gmin_proof"),
    ("lj102", "lj", 102, -569.363652, 0.0, "h2h_gmin_proof", "h2h_gmin_proof"),
    ("lj103", "lj", 103, -575.766131, 0.0, "h2h_gmin_proof+lj103", "h2h_gmin_proof_lj103"),
    ("lj104", "lj", 104, -582.086642, 0.0, "h2h_gmin_proof", "h2h_gmin_proof"),
    ("lj150", "lj", 150, -893.310258, 0.0, "h2h_gmin_proof", "h2h_gmin_proof"),
    ("au55", "gupta", 55, -191.03085280, 0.0, "h2h_metal", "h2h_metal"),
    ("m55r6", "morse", 55, -250.286609, 6.0, "h2h_morse", "h2h_morse"),
    ("m55r14", "morse", 55, -220.646208, 14.0, "h2h_morse", "h2h_morse"),
    ("m80r6", "morse", 80, -378.333471, 6.0, "h2h_morse", "h2h_morse"),
    ("m80r14", "morse", 80, -340.811371, 14.0, "h2h_morse", "h2h_morse"),
)
# A, p, q, xi, r0: the gold row of scripts/metal_h2h/enter_gold.sh.
GOLD_ROW = "0.2061 10.229 4.036 1.79 2.884"
LABELS = {
    "lj38": ("LJ38", "LJ\\(_{38}\\)"),
    "lj55": ("LJ55", "LJ\\(_{55}\\)"),
    "lj75": ("LJ75", "LJ\\(_{75}\\)"),
    "lj98": ("LJ98", "LJ\\(_{98}\\)"),
    "lj102": ("LJ102", "LJ\\(_{102}\\)"),
    "lj103": ("LJ103", "LJ\\(_{103}\\)"),
    "lj104": ("LJ104", "LJ\\(_{104}\\)"),
    "lj150": ("LJ150", "LJ\\(_{150}\\)"),
    "au55": ("Au55", "Au\\(_{55}\\)"),
    "m55r6": ("M55 ρ=6", "M\\(_{55}\\), \\(\\rho=6\\)"),
    "m55r14": ("M55 ρ=14", "M\\(_{55}\\), \\(\\rho=14\\)"),
    "m80r6": ("M80 ρ=6", "M\\(_{80}\\), \\(\\rho=6\\)"),
    "m80r14": ("M80 ρ=14", "M\\(_{80}\\), \\(\\rho=14\\)"),
}
README = "bench/gmin-panel/README.md"
MANUSCRIPT = "paper/jctc/manuscript.tex"
README_MARKS = ("<!-- gmin-panel table: start -->", "<!-- gmin-panel table: end -->")
TEX_MARKS = ("% gmin-panel table: start", "% gmin-panel table: end")

ROW_FIELDS = [
    "system",
    "code",
    "driver",
    "seed",
    "budget",
    "hit",
    "evals_to_hit",
    "charged",
    "wall_s",
    "best",
    "pot_s",
    "outside_s",
    "code_revision",
    "harness_revision",
    "host",
    "cpu",
    "session",
    "utc",
]

SUMMARY_FIELDS = [
    "system",
    "budget",
    "reference",
    "anneal_hits",
    "anneal_n",
    "gmin_hits",
    "gmin_n",
    "fisher_p",
    "anneal_evals_median",
    "gmin_evals_median",
    "anneal_evals_iqr",
    "gmin_evals_iqr",
    "anneal_evals_median_to_hit",
    "gmin_evals_median_to_hit",
    "mannwhitney_u_calls",
    "mannwhitney_p_calls",
    "anneal_wall_median_s",
    "gmin_wall_median_s",
    "mannwhitney_u_wall",
    "mannwhitney_p_wall",
    "anneal_pot_median_s",
    "anneal_outside_median_s",
    "gmin_pot_median_s",
    "gmin_outside_median_s",
    "anneal_best_min",
    "gmin_best_min",
    "anneal_revision",
    "gmin_revision",
    "harness_revision",
    "host",
    "sessions",
    "seeds",
]


def spec(name: str) -> tuple:
    for row in SYSTEMS:
        if row[0] == name:
            return row
    raise SystemExit(f"unknown system {name}")


def type7(values: list[float], p: float) -> float:
    """Hyndman-Fan type 7."""
    ordered = sorted(values)
    count = len(ordered)
    if count < 1:
        raise ValueError("type 7 needs a sample")
    if count == 1:
        return ordered[0]
    position = 1.0 + (count - 1) * p
    lower = int(position)
    frac = position - lower
    if frac == 0.0:
        return ordered[lower - 1]
    return ordered[lower - 1] * (1.0 - frac) + ordered[lower] * frac


def fisher(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    """Two-sided exact p, summing the tables no more likely than the observed."""
    hits = hits_a + hits_b
    ways = math.comb(n_a + n_b, hits)

    def prob(k: int) -> float:
        return math.comb(n_a, k) * math.comb(n_b, hits - k) / ways

    observed = prob(hits_a)
    lo, hi = max(0, hits - n_b), min(n_a, hits)
    return min(1.0, sum(prob(k) for k in range(lo, hi + 1) if prob(k) <= observed * (1 + 1e-7)))


def mann_whitney(left: list[float], right: list[float]) -> tuple[float, float]:
    """U of ``left`` and the two-sided p, normal approximation with the tie
    and continuity corrections. Every value tied gives p = 1."""
    n1, n2 = len(left), len(right)
    pooled = sorted(left + right)
    n = n1 + n2
    rank: dict[float, float] = {}
    ties = 0.0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pooled[j + 1] == pooled[i]:
            j += 1
        rank[pooled[i]] = (i + j + 2) / 2.0
        t = j - i + 1
        ties += t**3 - t
        i = j + 1
    u = sum(rank[v] for v in left) - n1 * (n1 + 1) / 2.0
    var = n1 * n2 / 12.0 * ((n + 1) - ties / (n * (n - 1)))
    if var <= 0.0:
        return u, 1.0
    dev = max(abs(u - n1 * n2 / 2.0) - 0.5, 0.0)
    return u, min(1.0, math.erfc(dev / math.sqrt(var) / math.sqrt(2.0)))


def is_hit(text: str) -> bool:
    return text.strip().lower() in ("1", "true")


def cpu_model() -> str:
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip().replace(",", " ")
    return ""


def single_thread_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith("H2H_"):
            del env[key]
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "RAYON_NUM_THREADS"):
        env[key] = "1"
    return env


def binary_budget(bin_dir: Path, tree: Path, system: str) -> tuple[int, float]:
    binary = spec(system)[6]
    proc = subprocess.run(
        [str(bin_dir / binary), "budget", system],
        cwd=tree,
        env=single_thread_env(),
        capture_output=True,
        text=True,
        check=True,
    )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    return int(lines[0]), float(lines[1])


def anneal_result(text: str, system: str, seed: int, budget: int) -> dict[str, str]:
    """The result line of ``h2h_gmin_proof``, ``h2h_metal`` or ``h2h_morse``."""
    lines = [line for line in text.splitlines() if line.startswith("system,") or f",anneal,{seed}," in line]
    headers = [line for line in lines if line.startswith("system,")]
    data = [line for line in lines if not line.startswith("system,")]
    if len(headers) != 1 or len(data) != 1:
        raise RuntimeError(f"{system} seed {seed}: expected one header and one row\n{text[-2000:]}")
    row = dict(zip(headers[0].split(","), data[0].split(",")))
    if row["system"] != system or int(row["seed"]) != seed or int(row["budget"]) != budget:
        raise RuntimeError(f"{system} seed {seed}: row {data[0]}")
    reference = spec(system)[3]
    best = float(row["best"])
    hit = is_hit(row["hit"])
    if hit != (best <= reference + HIT_TOL):
        raise RuntimeError(f"{system} seed {seed}: hit {row['hit']} with best {best}")
    evals = int(row["evals_to_hit"])
    if not hit and evals != budget:
        raise RuntimeError(f"{system} seed {seed}: a miss reports {evals} calls")
    wall = float(row["wall_s"])
    pot = float(row["pot_s"])
    outside = float(row["outside_s"]) if "outside_s" in row else wall - pot
    return {
        "hit": "1" if hit else "0",
        "evals_to_hit": str(evals),
        "charged": str(int(row["charged"])),
        "wall_s": f"{wall:.6f}",
        "best": row["best"],
        "pot_s": f"{pot:.6f}",
        "outside_s": f"{outside:.6f}",
        "head": row.get("head", ""),
    }


def gmin_result(text: str, system: str, seed: int, budget: int) -> dict[str, str]:
    """The ``h2h_row`` line the GMIN cap writes."""
    lines = [line for line in text.splitlines() if line.startswith("h2h_row ")]
    if len(lines) != 1:
        raise RuntimeError(f"gmin {system} seed {seed}: expected one h2h_row\n{text[-2000:]}")
    fields = dict(part.split("=", 1) for part in lines[0].split()[1:] if "=" in part)
    if fields["system"] != system or int(fields["seed"]) != seed or int(fields["budget"]) != budget:
        raise RuntimeError(f"gmin {system} seed {seed}: row {lines[0]}")
    reference = spec(system)[3]
    best = float(fields["best"])
    hit = fields["hit"] == "1"
    if hit != (best <= reference + HIT_TOL):
        raise RuntimeError(f"gmin {system} seed {seed}: hit {fields['hit']} with best {best}")
    evals = int(fields["evals_to_hit"])
    if not hit and evals != budget:
        raise RuntimeError(f"gmin {system} seed {seed}: a miss reports {evals} calls")
    wall = float(fields["wall_s"])
    pot = float(fields["pot_s"])
    return {
        "hit": "1" if hit else "0",
        "evals_to_hit": str(evals),
        "charged": str(int(fields["charged"])),
        "wall_s": f"{wall:.6f}",
        "best": fields["best"],
        "pot_s": f"{pot:.6f}",
        "outside_s": f"{wall - pot:.6f}",
    }


def gmin_data(system: str, seed: int, lj_data: str, morse_data: str) -> str:
    _name, kind, n, _reference, rho, _driver, _binary = spec(system)
    if kind == "lj":
        return lj_data + f"RANSEED {seed}\n"
    if kind == "morse":
        return morse_data + f"MORSE {rho:.1f}\nRANSEED {seed}\n"
    return (
        f"BGUPTAT {n} {GOLD_ROW}\n"
        f"BGUPTATAB {GOLD_ROW}\n"
        f"BGUPTATBB {GOLD_ROW}\n"
        "SLOPPYCONV 5.0D-3\n"
        "TIGHTCONV 1.0D-3\n"
        "EDIFF 0.003\n"
        "MAXIT 600 500\n"
        "SAVE 2\n"
        "CENTRE\n"
        "RADIUS 50.0\n"
        "TEMPERATURE 1.0\n"
        "STEPS 90000000 1.0\n"
        "STEP 0.40 0.51\n"
        "NEWRESTART 40\n"
        f"RANSEED {seed}\n"
    )


def run_anneal(bin_dir: Path, tree: Path, system: str, seed: int, log: Path) -> dict[str, str]:
    binary = spec(system)[6]
    budget = PAPER_BUDGETS[system]
    command = ["taskset", "-c", "0", str(bin_dir / binary), "run", system, str(seed)]
    proc = subprocess.run(command, cwd=tree, env=single_thread_env(), capture_output=True, text=True, check=False)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(" ".join(command) + "\n" + proc.stdout + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"anneal {system} seed {seed} exited {proc.returncode}\n{proc.stderr[-2000:]}")
    return anneal_result(proc.stdout, system, seed, budget)


def run_gmin(
    bin_dir: Path, tree: Path, system: str, seed: int, work: Path, log: Path, lj_data: str, morse_data: str
) -> dict[str, str]:
    binary = spec(system)[6]
    budget = PAPER_BUDGETS[system]
    reference = spec(system)[3]
    work.mkdir(parents=True, exist_ok=True)
    for stale in ("row.txt", "coords", "data"):
        (work / stale).unlink(missing_ok=True)
    subprocess.run(
        [str(bin_dir / binary), "coords", system, str(seed), str(work / "coords")],
        cwd=tree,
        env=single_thread_env(),
        capture_output=True,
        text=True,
        check=True,
    )
    (work / "data").write_text(gmin_data(system, seed, lj_data, morse_data))
    env = single_thread_env()
    env.update(
        {
            "H2H_BUDGET": str(budget),
            "H2H_SYSTEM": system,
            "H2H_SEED": str(seed),
            "H2H_TARGET": f"{reference:.10f}",
            "H2H_TOL": f"{HIT_TOL}",
            "H2H_ROW": str(work / "row.txt"),
        }
    )
    command = ["taskset", "-c", "0", str(bin_dir / "GMIN")]
    proc = subprocess.run(command, cwd=work, env=env, capture_output=True, text=True, check=False)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(" ".join(command) + f"\nexit={proc.returncode}\n" + proc.stdout[-20000:] + proc.stderr[-5000:])
    row = work / "row.txt"
    if not row.is_file():
        raise RuntimeError(f"gmin {system} seed {seed} exited {proc.returncode} with no row\n{proc.stdout[-2000:]}")
    return gmin_result(row.read_text(), system, seed, budget)


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def append_row(path: Path, row: dict[str, str]) -> None:
    fresh = not path.is_file() or path.stat().st_size == 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROW_FIELDS, lineterminator="\n")
        if fresh:
            writer.writeheader()
        writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())


def row_key(row: dict[str, str]) -> tuple[str, str, str, str, str]:
    return (row["system"], row["code"], row["seed"], row["code_revision"], row["harness_revision"])


def panel_groups(rows: list[dict[str, str]], revision: str) -> dict[tuple[str, str], list[dict[str, str]]]:
    """The first stored row per seed of each system and code at this revision."""
    wanted = {"anneal": revision, "gmin": GMIN_REV}
    groups: dict[tuple[str, str], dict[int, dict[str, str]]] = {}
    for row in rows:
        if row["harness_revision"] != revision or wanted.get(row["code"]) != row["code_revision"]:
            continue
        groups.setdefault((row["system"], row["code"]), {}).setdefault(int(row["seed"]), row)
    return {key: [group[s] for s in sorted(group)] for key, group in groups.items()}


def fmt(value: float | None, digits: int = 6) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def summarise(rows: list[dict[str, str]], revision: str) -> list[dict[str, str]]:
    out = []
    groups = panel_groups(rows, revision)
    for system, _kind, _n, reference, _rho, _driver, _binary in SYSTEMS:
        anneal = groups.get((system, "anneal"), [])
        gmin = groups.get((system, "gmin"), [])
        if not anneal or not gmin:
            continue

        def col(group: list[dict[str, str]], key: str) -> list[float]:
            return [float(r[key]) for r in group]

        def to_hit(group: list[dict[str, str]]) -> str:
            hits = [float(r["evals_to_hit"]) for r in group if r["hit"] == "1"]
            return f"{type7(hits, 0.5):.2f}" if hits else ""

        ah = sum(r["hit"] == "1" for r in anneal)
        gh = sum(r["hit"] == "1" for r in gmin)
        u_calls, p_calls = mann_whitney(col(anneal, "evals_to_hit"), col(gmin, "evals_to_hit"))
        u_wall, p_wall = mann_whitney(col(anneal, "wall_s"), col(gmin, "wall_s"))
        both = anneal + gmin
        out.append(
            {
                "system": system,
                "budget": str(PAPER_BUDGETS[system]),
                "reference": f"{reference:.8f}",
                "anneal_hits": str(ah),
                "anneal_n": str(len(anneal)),
                "gmin_hits": str(gh),
                "gmin_n": str(len(gmin)),
                "fisher_p": f"{fisher(ah, len(anneal), gh, len(gmin)):.6g}",
                "anneal_evals_median": f"{type7(col(anneal, 'evals_to_hit'), 0.5):.2f}",
                "gmin_evals_median": f"{type7(col(gmin, 'evals_to_hit'), 0.5):.2f}",
                "anneal_evals_iqr": f"{type7(col(anneal, 'evals_to_hit'), 0.75) - type7(col(anneal, 'evals_to_hit'), 0.25):.2f}",
                "gmin_evals_iqr": f"{type7(col(gmin, 'evals_to_hit'), 0.75) - type7(col(gmin, 'evals_to_hit'), 0.25):.2f}",
                "anneal_evals_median_to_hit": to_hit(anneal),
                "gmin_evals_median_to_hit": to_hit(gmin),
                "mannwhitney_u_calls": f"{u_calls:.1f}",
                "mannwhitney_p_calls": f"{p_calls:.6g}",
                "anneal_wall_median_s": fmt(type7(col(anneal, "wall_s"), 0.5)),
                "gmin_wall_median_s": fmt(type7(col(gmin, "wall_s"), 0.5)),
                "mannwhitney_u_wall": f"{u_wall:.1f}",
                "mannwhitney_p_wall": f"{p_wall:.6g}",
                "anneal_pot_median_s": fmt(type7(col(anneal, "pot_s"), 0.5)),
                "anneal_outside_median_s": fmt(type7(col(anneal, "outside_s"), 0.5)),
                "gmin_pot_median_s": fmt(type7(col(gmin, "pot_s"), 0.5)),
                "gmin_outside_median_s": fmt(type7(col(gmin, "outside_s"), 0.5)),
                "anneal_best_min": f"{min(col(anneal, 'best')):.8f}",
                "gmin_best_min": f"{min(col(gmin, 'best')):.8f}",
                "anneal_revision": revision,
                "gmin_revision": GMIN_REV,
                "harness_revision": revision,
                "host": " ".join(sorted({r["host"] for r in both})),
                "sessions": " ".join(sorted({r["session"] for r in both})),
                "seeds": " ".join(r["seed"] for r in anneal),
            }
        )
    return out


def show_p(p: float, tex: bool) -> str:
    """Two significant figures; 1 only when p is exactly 1."""
    if p >= 1.0:
        return "1"
    if p >= 0.01:
        text = f"{p:.2f}" if p >= 0.1 else f"{p:.3f}"
        return f"{p:.3f}" if text == "1.00" else text
    mantissa, exponent = f"{p:.1e}".split("e")
    if tex:
        return f"\\({mantissa}\\times 10^{{{int(exponent)}}}\\)"
    return f"{mantissa}e{int(exponent)}"


def show_calls(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def show_budget(budget: int, tex: bool) -> str:
    mantissa, exponent = f"{budget:.0e}".split("e")
    if not tex or int(mantissa) * 10 ** int(exponent) != budget:
        return str(budget)
    return f"\\({mantissa}\\times 10^{{{int(exponent)}}}\\)"


def show_s(value: float) -> str:
    return f"{value:.3f}" if value < 100.0 else f"{value:.1f}"


def table_lines(summary: list[dict[str, str]], tex: bool) -> list[list[str]]:
    """One line per system and arm. Fisher and Mann-Whitney sit on the anneal line."""
    out = []
    for line in summary:
        name = LABELS[line["system"]][1 if tex else 0]
        for arm in ("anneal", "gmin"):
            label = ("this work" if tex else "anneal") if arm == "anneal" else "GMIN"
            first = arm == "anneal"
            out.append(
                [
                    name if first else "",
                    show_budget(int(line["budget"]), tex) if first else "",
                    label,
                    f"{line[f'{arm}_hits']}/{line[f'{arm}_n']}",
                    show_calls(float(line[f"{arm}_evals_median"])),
                    show_calls(float(line[f"{arm}_evals_iqr"])),
                    show_s(float(line[f"{arm}_wall_median_s"])),
                    show_s(float(line[f"{arm}_pot_median_s"])),
                    show_s(float(line[f"{arm}_outside_median_s"])),
                    show_p(float(line["fisher_p"]), tex) if first else "",
                    show_p(float(line["mannwhitney_p_calls"]), tex) if first else "",
                    show_p(float(line["mannwhitney_p_wall"]), tex) if first else "",
                ]
            )
    return out


HEADER = ["System", "Budget", "Arm", "Hits", "Median calls", "IQR", "Wall (s)", "Inside (s)", "Outside (s)",
          "Fisher p", "MW p calls", "MW p wall"]


def readme_table(summary: list[dict[str, str]]) -> str:
    rows = ["| " + " | ".join(HEADER) + " |", "| " + " | ".join("---" for _ in HEADER) + " |"]
    rows += ["| " + " | ".join(cells) + " |" for cells in table_lines(summary, tex=False)]
    return "\n".join(rows)


def tex_table(summary: list[dict[str, str]], revision: str) -> str:
    body = [" & ".join(cells) + " \\\\" for cells in table_lines(summary, tex=True)]
    return "\n".join(
        [
            "\\begin{table}",
            "\\caption{This work against GMIN on one host in one session.",
            f"Revisions: anneal \\texttt{{{revision[:8]}}}, GMIN \\texttt{{{GMIN_REV[:8]}}}.",
            "Seeds 2000--2011, one core, budget in force calls.",
            "Calls and IQR are the median and the interquartile range of the",
            "calls to the minimum, with a miss at the full budget.",
            "Wall, In and Out are the median seconds of the run, inside the",
            "potential and outside it.",
            "\\(p_\\mathrm{F}\\) is the two-sided Fisher exact test on hits;",
            "\\(p_\\mathrm{calls}\\) and \\(p_\\mathrm{wall}\\) are the two-sided",
            "Mann--Whitney test on calls and on wall.}",
            "\\label{tab:gmin}",
            "\\centering",
            "\\scriptsize",
            "\\setlength{\\tabcolsep}{2pt}",
            "\\begin{tabular}{@{}lrlrrrrrrrrr@{}}",
            "\\toprule",
            "Cluster & Budget & Arm & Hits & Calls & IQR & Wall & In & Out & \\(p_\\mathrm{F}\\) & \\(p_\\mathrm{calls}\\) & \\(p_\\mathrm{wall}\\) \\\\",
            "\\midrule",
            *body,
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )


def splice(path: Path, marks: tuple[str, str], block: str) -> None:
    text = path.read_text()
    start, end = marks
    if text.count(start) != 1 or text.count(end) != 1:
        raise SystemExit(f"{path} needs one '{start}' and one '{end}'")
    head, rest = text.split(start, 1)
    _old, tail = rest.split(end, 1)
    path.write_text(f"{head}{start}\n{block}\n{end}{tail}")


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_tables(root: Path, rows: list[dict[str, str]], revision: str, tex: bool) -> None:
    summary = summarise(rows, revision)
    write_csv(root / "bench" / "gmin-panel" / "summary.csv", SUMMARY_FIELDS, summary)
    splice(root / README, README_MARKS, readme_table(summary))
    if tex:
        splice(root / MANUSCRIPT, TEX_MARKS, tex_table(summary, revision))


def self_test() -> None:
    sample = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    if abs(type7(sample, 0.25) - 2.25) > 1e-12 or abs(type7(sample, 0.5) - 3.5) > 1e-12:
        raise SystemExit("type 7")
    if [row[0] for row in SYSTEMS] != list(PAPER_BUDGETS) or set(LABELS) != set(PAPER_BUDGETS):
        raise SystemExit("every system has a budget and a label")
    if len(SEEDS) != 12 or SEEDS[0] != 2000:
        raise SystemExit("seeds 2000-2011")
    if abs(fisher(12, 12, 0, 12) - 2.0 / math.comb(24, 12)) > 1e-15:
        raise SystemExit("fisher")
    if fisher(6, 12, 6, 12) != 1.0:
        raise SystemExit("fisher of equal counts")
    u, p = mann_whitney([4e6] * 12, [4e6] * 12)
    if u != 72.0 or p != 1.0:
        raise SystemExit(f"mann whitney of ties {u} {p}")
    try:
        from scipy.stats import fisher_exact, mannwhitneyu
    except ImportError:
        pass
    else:
        left = [9.0, 4e6, 4e6, 120.0, 55.0, 4e6, 7.0, 3.0]
        right = [4e6, 4e6, 4e6, 900.0, 4e6, 4e6, 4e6, 4e6]
        want = mannwhitneyu(left, right, alternative="two-sided", method="asymptotic", use_continuity=True)
        u, p = mann_whitney(left, right)
        if abs(u - want.statistic) > 1e-9 or abs(p - want.pvalue) > 1e-12:
            raise SystemExit(f"mann whitney {u} {p} against scipy {want}")
        if abs(fisher(9, 12, 2, 12) - fisher_exact([[9, 3], [2, 10]])[1]) > 1e-12:
            raise SystemExit("fisher against scipy")
    text = "system,code,seed,budget,hit,evals_to_hit,charged,wall_s,best,head,tree,cpu,pins,pot_s\n"
    text += "lj38,anneal,2000,400000,true,9000,9100,0.5,-173.92842700,abc,t,c,0,0.25\n"
    row = anneal_result(text, "lj38", 2000, 400_000)
    if row["hit"] != "1" or row["outside_s"] != "0.250000" or row["head"] != "abc":
        raise SystemExit(f"anneal row {row}")
    line = ("h2h_row system=m80r6 code=gmin seed=2001 budget=4000000 hit=0 evals_to_hit=4000000 "
            "charged=4000000 best=-378.000000000 wall_s=150.5 pot_s=140.25\n")
    row = gmin_result(line, "m80r6", 2001, 4_000_000)
    if row["hit"] != "0" or row["outside_s"] != "10.250000":
        raise SystemExit(f"gmin row {row}")
    if "MORSE 14.0\nRANSEED 2003\n" not in gmin_data("m80r14", 2003, "", "x\n"):
        raise SystemExit("morse data")
    if not gmin_data("au55", 2000, "", "").startswith(f"BGUPTAT 55 {GOLD_ROW}\n"):
        raise SystemExit("gupta data")
    if show_p(7.39602e-07, tex=False) != "7.4e-7" or show_p(1.0, tex=True) != "1" or show_p(0.2105, False) != "0.21" \
            or show_p(0.0341, False) != "0.034":
        raise SystemExit("p format")
    if show_budget(400_000, tex=True) != "\\(4\\times 10^{5}\\)" or show_budget(1_000_000, tex=False) != "1000000":
        raise SystemExit("budget format")
    print("ok")


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[1] == "--self-test":
        self_test()
        return 0
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--bin", type=Path)
    parser.add_argument("--tree", type=Path)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--logs", type=Path)
    parser.add_argument("--gmin-source", type=Path)
    parser.add_argument("--lj-data", type=Path)
    parser.add_argument("--morse-data", type=Path)
    parser.add_argument("--system", action="append")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--tex", action="store_true", help="with --summarize, also rewrite tab:gmin")
    args = parser.parse_args(argv[1:])
    if args.summarize:
        write_tables(args.root, read_rows(args.rows), args.revision, args.tex)
        return 0
    if socket.gethostname() != "bench":
        print(f"uts hostname is {socket.gethostname()}, expected bench", file=sys.stderr)
        return 2
    if os.sched_getaffinity(0) != {0}:
        print(f"pinned to {sorted(os.sched_getaffinity(0))}, expected core 0", file=sys.stderr)
        return 2
    needed = (args.bin, args.tree, args.work, args.logs, args.gmin_source, args.lj_data, args.morse_data)
    if any(value is None for value in needed):
        parser.error("--bin, --tree, --work, --logs, --gmin-source, --lj-data and --morse-data are needed to run")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=args.tree, capture_output=True, text=True, check=True
    ).stdout.strip()
    if head != args.revision:
        raise SystemExit(f"{args.tree} is at {head}, not {args.revision}")
    source = dict(
        line.split("=", 1) for line in args.gmin_source.read_text().splitlines() if "=" in line
    )
    if source.get("svnrev") != GMIN_REV:
        raise SystemExit(f"GMIN SVNREV {source.get('svnrev')} is not {GMIN_REV}")
    systems = [row[0] for row in SYSTEMS if not args.system or row[0] in args.system]
    for system in systems:
        budget, reference = binary_budget(args.bin, args.tree, system)
        if budget != PAPER_BUDGETS[system] or abs(reference - spec(system)[3]) > 1e-8:
            raise SystemExit(f"{spec(system)[6]} {system}: budget {budget} reference {reference}")
    lj_data = args.lj_data.read_text()
    morse_data = args.morse_data.read_text()
    lj_data += "" if lj_data.endswith("\n") else "\n"
    morse_data += "" if morse_data.endswith("\n") else "\n"
    host = socket.gethostname()
    cpu = cpu_model()
    session = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    done = {row_key(r) for r in read_rows(args.rows)}
    seeds = tuple(args.seed) if args.seed else SEEDS
    print(f"session {session} revision {args.revision} gmin {GMIN_REV}", flush=True)
    for system in systems:
        driver = spec(system)[5]
        for seed in seeds:
            for code in ("anneal", "gmin"):
                code_revision = args.revision if code == "anneal" else GMIN_REV
                key = (system, code, str(seed), code_revision, args.revision)
                if key in done:
                    print(f"skip {system} {code} {seed}", flush=True)
                    continue
                print(f"start {system} {code} {seed}", flush=True)
                log = args.logs / system / f"{code}-{seed}.log"
                if code == "anneal":
                    result = run_anneal(args.bin, args.tree, system, seed, log)
                    if result.pop("head") not in ("", args.revision):
                        raise SystemExit(f"{system} {seed}: binary reports another head")
                else:
                    result = run_gmin(
                        args.bin, args.tree, system, seed, args.work / system / str(seed), log, lj_data, morse_data
                    )
                row = {
                    "system": system,
                    "code": code,
                    "driver": driver if code == "anneal" else "GMIN",
                    "seed": str(seed),
                    "budget": str(PAPER_BUDGETS[system]),
                    **result,
                    "code_revision": code_revision,
                    "harness_revision": args.revision,
                    "host": host,
                    "cpu": cpu,
                    "session": session,
                    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                }
                append_row(args.rows, row)
                done.add(key)
                print(
                    f"done {system} {code} {seed} hit={row['hit']} calls={row['evals_to_hit']} "
                    f"wall_s={row['wall_s']} pot_s={row['pot_s']} outside_s={row['outside_s']} best={row['best']}",
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
