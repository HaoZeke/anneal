# Copyright (c) anneal developers
"""anneal against GMIN on Doye-Meyer binary Lennard-Jones clusters.

Seven clusters from the Cambridge Cluster Database: sigma_BB/sigma_AA of
1.05, 1.15 and 1.30 at N=38 and N=55, and 1.30 at N=98. Seeds 2000-2011,
the call budgets of the matching sizes on the GMIN panel. A hit is a
quenched energy at or below the structure's header energy plus 0.001; a
miss is entered at the budget. Both arms start from the geometry
``h2h_blj coords`` writes for the seed.

anneal runs ``Config::recommended`` with two species labels and no
diameters, through ``h2h_blj``. GMIN is one build with the evaluator cap
and a timer inside ``BLJ_CLUST``. Its input is ``examples/h2h_gmin_lj.data``
plus ``BLJCLUSTER_NOCUT`` and ``RANSEED``.

One process, ``taskset -c 0``, one thread. For each system the seeds run
in order, anneal then GMIN. Rows are appended after every run and never
rewritten. ``code_revision`` is the anneal commit on anneal rows and the
GMIN SVNREV on GMIN rows; ``harness_revision`` is the anneal commit whose
example wrote the starts.

``--summarize`` reads the rows and writes ``summary.csv``, the table in
``bench/blj-gmin/README.md`` and, with ``--tex``, table ``tab:blj`` and
the results sentences. Medians and interquartile ranges are Hyndman-Fan
type 7. Fisher is the two-sided exact test on hits. Mann-Whitney is
two-sided, the normal approximation with the tie and continuity
corrections, on calls to the minimum and on wall time.

This process refuses to run unless the UTS hostname is bench and it is
pinned to core 0.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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
    "blj105n38": 400_000,
    "blj105n55": 1_000_000,
    "blj115n38": 400_000,
    "blj115n55": 1_000_000,
    "blj130n38": 400_000,
    "blj130n55": 1_000_000,
    "blj130n98": 4_000_000,
}
# System, atoms, species A, sigma_BB, header energy of the stored structure.
SYSTEMS = (
    ("blj105n38", 38, 8, 1.05, -177.2090846748),
    ("blj105n55", 55, 13, 1.05, -283.1545313810),
    ("blj115n38", 38, 9, 1.15, -187.3676867453),
    ("blj115n55", 55, 16, 1.15, -289.2931853058),
    ("blj130n38", 38, 13, 1.30, -191.1840192602),
    ("blj130n55", 55, 19, 1.30, -306.7337643721),
    ("blj130n98", 98, 41, 1.30, -590.7874132760),
)
LABELS = {
    "blj105n38": ("BLJ38 σ=1.05", "BLJ\\(_{38}\\) at \\(\\sigma_{BB}=1.05\\)"),
    "blj105n55": ("BLJ55 σ=1.05", "BLJ\\(_{55}\\) at \\(\\sigma_{BB}=1.05\\)"),
    "blj115n38": ("BLJ38 σ=1.15", "BLJ\\(_{38}\\) at \\(\\sigma_{BB}=1.15\\)"),
    "blj115n55": ("BLJ55 σ=1.15", "BLJ\\(_{55}\\) at \\(\\sigma_{BB}=1.15\\)"),
    "blj130n38": ("BLJ38 σ=1.30", "BLJ\\(_{38}\\) at \\(\\sigma_{BB}=1.30\\)"),
    "blj130n55": ("BLJ55 σ=1.30", "BLJ\\(_{55}\\) at \\(\\sigma_{BB}=1.30\\)"),
    "blj130n98": ("BLJ98 σ=1.30", "BLJ\\(_{98}\\) at \\(\\sigma_{BB}=1.30\\)"),
}
README = "bench/blj-gmin/README.md"
MANUSCRIPT = "paper/jctc/manuscript.tex"
README_MARKS = ("<!-- blj-gmin table: start -->", "<!-- blj-gmin table: end -->")
TEX_MARKS = ("% blj-gmin table: start", "% blj-gmin table: end")
README_PROSE = ("<!-- blj-gmin prose: start -->", "<!-- blj-gmin prose: end -->")
TEX_PROSE = ("% blj-gmin prose: start", "% blj-gmin prose: end")
CCD = "bench/blj-gmin/ccd"

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


def sigma_ab(sigma_bb: float) -> float:
    return 0.5 * (1.0 + sigma_bb)


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
    proc = subprocess.run(
        [str(bin_dir / "h2h_blj"), "budget", system],
        cwd=tree,
        env=single_thread_env(),
        capture_output=True,
        text=True,
        check=True,
    )
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    return int(lines[0]), float(lines[1])


def anneal_result(text: str, system: str, seed: int, budget: int) -> dict[str, str]:
    lines = [line for line in text.splitlines() if line.startswith("system,") or f",anneal,{seed}," in line]
    headers = [line for line in lines if line.startswith("system,")]
    data = [line for line in lines if not line.startswith("system,")]
    if len(headers) != 1 or len(data) != 1:
        raise RuntimeError(f"{system} seed {seed}: expected one header and one row\n{text[-2000:]}")
    row = dict(zip(headers[0].split(","), data[0].split(",")))
    if row["system"] != system or int(row["seed"]) != seed or int(row["budget"]) != budget:
        raise RuntimeError(f"{system} seed {seed}: row {data[0]}")
    reference = spec(system)[4]
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
    lines = [line for line in text.splitlines() if line.startswith("h2h_row ")]
    if len(lines) != 1:
        raise RuntimeError(f"gmin {system} seed {seed}: expected one h2h_row\n{text[-2000:]}")
    fields = dict(part.split("=", 1) for part in lines[0].split()[1:] if "=" in part)
    if fields["system"] != system or int(fields["seed"]) != seed or int(fields["budget"]) != budget:
        raise RuntimeError(f"gmin {system} seed {seed}: row {lines[0]}")
    reference = spec(system)[4]
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


def gmin_data(system: str, seed: int, lj_data: str) -> str:
    _name, _n, n_a, sigma_bb, _reference = spec(system)
    return (
        lj_data
        + f"BLJCLUSTER_NOCUT {n_a} 1.0 1.0 {sigma_ab(sigma_bb):.4f} {sigma_bb:.4f}\n"
        + f"RANSEED {seed}\n"
    )


def run_anneal(bin_dir: Path, tree: Path, system: str, seed: int, log: Path) -> dict[str, str]:
    budget = PAPER_BUDGETS[system]
    command = ["taskset", "-c", "0", str(bin_dir / "h2h_blj"), "run", system, str(seed)]
    proc = subprocess.run(command, cwd=tree, env=single_thread_env(), capture_output=True, text=True, check=False)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(" ".join(command) + "\n" + proc.stdout + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"anneal {system} seed {seed} exited {proc.returncode}\n{proc.stderr[-2000:]}")
    return anneal_result(proc.stdout, system, seed, budget)


def run_gmin(bin_dir: Path, tree: Path, system: str, seed: int, work: Path, log: Path, lj_data: str) -> dict[str, str]:
    budget = PAPER_BUDGETS[system]
    reference = spec(system)[4]
    work.mkdir(parents=True, exist_ok=True)
    for stale in ("row.txt", "coords", "data"):
        (work / stale).unlink(missing_ok=True)
    subprocess.run(
        [str(bin_dir / "h2h_blj"), "coords", system, str(seed), str(work / "coords")],
        cwd=tree,
        env=single_thread_env(),
        capture_output=True,
        text=True,
        check=True,
    )
    (work / "data").write_text(gmin_data(system, seed, lj_data))
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
    for system, _n, _n_a, _sigma, reference in SYSTEMS:
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
                "reference": f"{reference:.10f}",
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


HEADER = [
    "System",
    "Budget",
    "Arm",
    "Hits",
    "Median calls",
    "IQR",
    "Wall (s)",
    "Inside (s)",
    "Outside (s)",
    "Fisher p",
    "MW p calls",
    "MW p wall",
]


def readme_table(summary: list[dict[str, str]]) -> str:
    rows = ["| " + " | ".join(HEADER) + " |", "| " + " | ".join("---" for _ in HEADER) + " |"]
    rows += ["| " + " | ".join(cells) + " |" for cells in table_lines(summary, tex=False)]
    return "\n".join(rows)


def tex_table(summary: list[dict[str, str]], revision: str) -> str:
    body = [" & ".join(cells) + " \\\\" for cells in table_lines(summary, tex=True)]
    return "\n".join(
        [
            "\\begin{table}",
            "\\caption{This work against GMIN on the Doye--Meyer binary Lennard-Jones clusters.",
            f"Revisions: anneal \\texttt{{{revision[:8]}}}, GMIN \\texttt{{{GMIN_REV[:8]}}}.",
            "Seeds 2000--2011, one core, budget in force calls.",
            "Calls and IQR are the median and the interquartile range of the",
            "calls to the minimum, with a miss at the full budget.",
            "Wall, In and Out are the median seconds of the run, inside the",
            "potential and outside it.",
            "\\(p_\\mathrm{F}\\) is the two-sided Fisher exact test on hits;",
            "\\(p_\\mathrm{calls}\\) and \\(p_\\mathrm{wall}\\) are the two-sided",
            "Mann--Whitney test on calls and on wall.}",
            "\\label{tab:blj}",
            "\\centering",
            "\\footnotesize",
            "\\setlength{\\tabcolsep}{3pt}",
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


def english_join(items: list[str]) -> str:
    if len(items) <= 1:
        return items[0] if items else ""
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def result_sentences(groups: dict[tuple[str, str], list[dict[str, str]]], tex: bool) -> list[str]:
    """Hit counts, the ahead/behind split, and Fisher's test, from the rows."""
    systems = [row[0] for row in SYSTEMS]

    def hits(group: list[dict[str, str]]) -> int:
        return sum(r["hit"] == "1" for r in group)

    lines = []
    ahead, behind, tied = [], [], []
    for system in systems:
        anneal = groups[(system, "anneal")]
        gmin = groups[(system, "gmin")]
        ah, gh = hits(anneal), hits(gmin)
        label = LABELS[system][1 if tex else 0]
        lines.append(f"{label} hits on {ah} of {len(anneal)} against GMIN on {gh} of {len(gmin)}.")
        (ahead if ah > gh else behind if ah < gh else tied).append(system)
    total = len(systems)
    lines.append(f"This setting hits more seeds than GMIN on {len(ahead)} of the {total} clusters.")
    if behind:
        detail = ", and on ".join(
            f"{LABELS[s][1 if tex else 0]}, {hits(groups[(s, 'gmin')])} of {len(groups[(s, 'gmin')])} "
            f"against {hits(groups[(s, 'anneal')])} of {len(groups[(s, 'anneal')])}"
            for s in behind
        )
        lines.append(f"GMIN hits more on {detail}.")
    if tied:
        lines.append(f"The two tie on {english_join([LABELS[s][1 if tex else 0] for s in tied])}.")
    significant = [
        s
        for s in systems
        if fisher(hits(groups[(s, "anneal")]), len(groups[(s, "anneal")]), hits(groups[(s, "gmin")]), len(groups[(s, "gmin")]))
        < 0.05
    ]
    ours = [s for s in significant if s in ahead]
    theirs = [s for s in significant if s in behind]
    if theirs:
        tail = f": {len(ours)} for this setting and {english_join([LABELS[s][1 if tex else 0] for s in theirs])} for GMIN"
    elif ours:
        tail = f": {len(ours)} for this setting"
    else:
        tail = ""
    lines.append(f"Fisher's test on hits is below 0.05 on {len(significant)} of the {total} clusters{tail}.")
    return lines


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
    groups = panel_groups(rows, revision)
    missing = [system for system, *_rest in SYSTEMS if (system, "anneal") not in groups or (system, "gmin") not in groups]
    if missing:
        raise SystemExit(f"rows at {revision} do not cover {', '.join(missing)}")
    summary = summarise(rows, revision)
    write_csv(root / "bench" / "blj-gmin" / "summary.csv", SUMMARY_FIELDS, summary)
    splice(root / README, README_MARKS, readme_table(summary))
    splice(root / README, README_PROSE, "\n".join(result_sentences(groups, tex=False)))
    if tex:
        splice(root / MANUSCRIPT, TEX_MARKS, tex_table(summary, revision))
        splice(root / MANUSCRIPT, TEX_PROSE, "\n".join(result_sentences(groups, tex=True)))


def ccd_header(path: Path) -> tuple[int, int, float, float]:
    lines = path.read_text().splitlines()
    n = int(lines[0].split()[0])
    tokens = lines[1].split()
    energy = float(tokens[tokens.index("1=") + 1])
    index = tokens.index("(NA,SIG_BB)")
    return n, int(tokens[index + 1]), float(tokens[index + 2]), energy


def self_test() -> None:
    sample = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    if abs(type7(sample, 0.25) - 2.25) > 1e-12 or abs(type7(sample, 0.5) - 3.5) > 1e-12:
        raise SystemExit("type 7")
    if [row[0] for row in SYSTEMS] != list(PAPER_BUDGETS) or set(LABELS) != set(PAPER_BUDGETS):
        raise SystemExit("every system has a budget and a label")
    if len(SEEDS) != 12 or SEEDS[0] != 2000 or SEEDS[-1] != 2011:
        raise SystemExit("seeds 2000-2011")
    for name, n, n_a, sigma, reference in SYSTEMS:
        if PAPER_BUDGETS[name] != {38: 400_000, 55: 1_000_000, 98: 4_000_000}[n]:
            raise SystemExit(f"{name} budget")
        if float(f"{sigma_ab(sigma):.4f}") != sigma_ab(sigma) or float(f"{sigma:.4f}") != sigma:
            raise SystemExit(f"{name} sigma text")
        if not (reference < 0.0 and n_a < n):
            raise SystemExit(name)
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
    text += "blj105n38,anneal,2000,400000,true,9000,9100,0.5,-177.20908467,abc,t,c,0,0.25\n"
    row = anneal_result(text, "blj105n38", 2000, 400_000)
    if row["hit"] != "1" or row["outside_s"] != "0.250000" or row["head"] != "abc":
        raise SystemExit(f"anneal row {row}")
    line = (
        "h2h_row system=blj130n98 code=gmin seed=2001 budget=4000000 hit=0 evals_to_hit=4000000 "
        "charged=4000000 best=-588.000000000 wall_s=150.5 pot_s=140.25\n"
    )
    row = gmin_result(line, "blj130n98", 2001, 4_000_000)
    if row["hit"] != "0" or row["outside_s"] != "10.250000":
        raise SystemExit(f"gmin row {row}")
    data = gmin_data("blj105n38", 2003, "STEPS 1\n")
    if "BLJCLUSTER_NOCUT 8 1.0 1.0 1.0250 1.0500\nRANSEED 2003\n" not in data:
        raise SystemExit(f"blj data {data}")
    if "BLJCLUSTER_NOCUT 41 1.0 1.0 1.1500 1.3000\n" not in gmin_data("blj130n98", 2000, ""):
        raise SystemExit("blj 98 data")
    if show_p(7.39602e-07, tex=False) != "7.4e-7" or show_p(1.0, tex=True) != "1" or show_p(0.2105, False) != "0.21" \
            or show_p(0.0341, False) != "0.034":
        raise SystemExit("p format")
    if show_budget(400_000, tex=True) != "\\(4\\times 10^{5}\\)" or show_budget(1_000_000, tex=False) != "1000000":
        raise SystemExit("budget format")
    root = Path(__file__).resolve().parents[2]
    sources = list(csv.DictReader((root / CCD / "sources.csv").open(newline="")))
    by_file = {row["file"]: row for row in sources}
    for name, n, n_a, sigma, reference in SYSTEMS:
        path = root / CCD / name
        got_n, got_a, got_sigma, got_energy = ccd_header(path)
        if (got_n, got_a) != (n, n_a) or abs(got_sigma - sigma) > 1e-12 or abs(got_energy - reference) > 1e-12:
            raise SystemExit(f"{name} header {got_n} {got_a} {got_sigma} {got_energy}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if by_file[name]["sha256"] != digest or by_file[name]["role"] != "reference":
            raise SystemExit(f"{name} checksum")
    original = root / CCD / "blj130n98-original"
    if hashlib.sha256(original.read_bytes()).hexdigest() != by_file["blj130n98-original"]["sha256"]:
        raise SystemExit("original 98 checksum")
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
    parser.add_argument("--system", action="append")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--summarize", action="store_true")
    parser.add_argument("--tex", action="store_true", help="with --summarize, also rewrite tab:blj")
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
    needed = (args.bin, args.tree, args.work, args.logs, args.gmin_source, args.lj_data)
    if any(value is None for value in needed):
        parser.error("--bin, --tree, --work, --logs, --gmin-source and --lj-data are needed to run")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=args.tree, capture_output=True, text=True, check=True
    ).stdout.strip()
    if head != args.revision:
        raise SystemExit(f"{args.tree} is at {head}, not {args.revision}")
    source = dict(line.split("=", 1) for line in args.gmin_source.read_text().splitlines() if "=" in line)
    if source.get("svnrev") != GMIN_REV:
        raise SystemExit(f"GMIN SVNREV {source.get('svnrev')} is not {GMIN_REV}")
    systems = [row[0] for row in SYSTEMS if not args.system or row[0] in args.system]
    for system in systems:
        budget, reference = binary_budget(args.bin, args.tree, system)
        if budget != PAPER_BUDGETS[system] or abs(reference - spec(system)[4]) > 1e-8:
            raise SystemExit(f"h2h_blj {system}: budget {budget} reference {reference}")
    lj_data = args.lj_data.read_text()
    lj_data += "" if lj_data.endswith("\n") else "\n"
    host = socket.gethostname()
    cpu = cpu_model()
    session = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    done = {row_key(r) for r in read_rows(args.rows)}
    seeds = tuple(args.seed) if args.seed else SEEDS
    print(f"session {session} revision {args.revision} gmin {GMIN_REV}", flush=True)
    for system in systems:
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
                        args.bin, args.tree, system, seed, args.work / system / str(seed), log, lj_data
                    )
                row = {
                    "system": system,
                    "code": code,
                    "driver": "h2h_blj" if code == "anneal" else "GMIN",
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
