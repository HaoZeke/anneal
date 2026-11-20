#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Write the head-to-head tables of the manuscript and the SI from the committed summaries.

Five blocks, each between a pair of marker comments:

    tab:gofee       si.tex          bench/ase-gofee-h2h/lj-large-seeds.csv
                                    or summary.csv when that file is absent
    tab:tip4p-gmin  si.tex          bench/tip4p-h2h/rows-h2h-r2.csv, wall-r2.csv
    tab:gofee-si    si.tex          written only when asg2-summary.csv names a held commit
    tab:sc          manuscript.tex  bench/metal-cluster-h2h/sc-n12/results.csv, pvalues.csv
    tab:sc-si       si.tex          bench/metal-cluster-h2h/sc-n12/results.csv

    python3 paper/jctc/scripts/h2h_tables.py            # rewrite the blocks
    python3 paper/jctc/scripts/h2h_tables.py --check    # exit 1 if one is stale
"""

from __future__ import annotations

import csv
import math
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "tip4p_h2h"))
sys.path.insert(0, str(ROOT / "bench" / "tip4p-h2h"))
from run_h2h_r2 import mann_whitney_u_two_sided_exact  # noqa: E402

MANUSCRIPT = ROOT / "paper" / "jctc" / "manuscript.tex"
SI = ROOT / "paper" / "jctc" / "si.tex"
GOFEE = ROOT / "bench" / "ase-gofee-h2h"
TIP4P = ROOT / "bench" / "tip4p-h2h" / "rows-h2h-r2.csv"
TIP4P_WALL = ROOT / "bench" / "tip4p-h2h" / "wall-r2.csv"
SC = ROOT / "bench" / "metal-cluster-h2h" / "sc-n12" / "results.csv"
SC_SEEDS = ROOT / "bench" / "metal-cluster-h2h" / "sc-n12" / "seeds.csv"
SC_P = ROOT / "bench" / "metal-cluster-h2h" / "sc-n12" / "pvalues.csv"
GMIN_TUNED = ROOT / "bench" / "gmin-tuned" / "rows-shard2.csv"
ABCLUSTER = ROOT / "bench" / "abcluster-template-free" / "rerun-20261119" / "rows.csv"
CREST_EQUAL = ROOT / "bench" / "crest-equal"
CREST_ARMS = ("anneal", "crest", "crest-wall")
# The call tolerance of the panel: run_panel.py counts a CREST run as at the hop arm's calls
# when it lies within 5 percent of them.
CREST_CALL_TOLERANCE = 0.05

MARKS = {
    "gofee": ("% gofee table: start", "% gofee table: end"),
    "tip4p": ("% tip4p-gmin table: start", "% tip4p-gmin table: end"),
    "tip4p-prose": ("% tip4p-gmin prose: start", "% tip4p-gmin prose: end"),
    "gofee-si": ("% gofee-si table: start", "% gofee-si table: end"),
    "sc-prose": ("% sc prose: start", "% sc prose: end"),
    "sc": ("% sc table: start", "% sc table: end"),
    "sc-si": ("% sc-si table: start", "% sc-si table: end"),
    "gmin-tuned": ("% gmin-tuned table: start", "% gmin-tuned table: end"),
    "abcluster": ("% abcluster table: start", "% abcluster table: end"),
}
SC_ORDER = ("sc126-38", "sc108-38", "sc96-38", "sc126-55", "sc108-55", "sc96-55", "sc126-80", "sc108-80", "sc96-80")
SC_FORM = {"126": "12-6", "108": "10-8", "96": "9-6"}
ARM_LABEL = {"paper": "This work", "ase_bh": "ASE BH", "ase_mh": "ASE MH", "agox_gofee": "GOFEE"}
LJ_LABEL = {"lj38": "LJ\\(_{38}\\)", "lj55": "LJ\\(_{55}\\)", "lj75": "LJ\\(_{75}\\)"}
LARGE_LABEL = {
    "lj98": "LJ\\(_{98}\\)", "lj102": "LJ\\(_{102}\\)", "lj103": "LJ\\(_{103}\\)",
    "lj104": "LJ\\(_{104}\\)", "lj150": "LJ\\(_{150}\\)",
}
LARGE_SYSTEMS = tuple(LARGE_LABEL)
# The later AGOX seeds name this harness head. It is not in the repository, and
# bench/unreachable-revisions.csv lists the rows. The table still counts them.
UNHELD_AGOX = "89de725a337b6e37daef05877796ae8192fbf8e1"
ANNEAL_SOURCE = "786c1a3b87fc948be5823927bee83027570e3223"
AGOX_HEADS = (
    "4508a253b8bd41dbba19ccf61278b3080249e400",
    ANNEAL_SOURCE,
    UNHELD_AGOX,
)
OTHER_LABEL = {"au55": "Au\\(_{55}\\)", "cu38": "Cu\\(_{38}\\)", "cu12au26": "Cu\\(_{12}\\)Au\\(_{26}\\)",
               "sw10": "Si\\(_{10}\\)", "tc10": "C\\(_{10}\\)", "tip4p8": "(H\\(_2\\)O)\\(_8\\)"}
TIP4P_LABEL = {"tip4p8": "(H\\(_2\\)O)\\(_8\\)", "tip4p15": "(H\\(_2\\)O)\\(_{15}\\)",
               "tip4p20": "(H\\(_2\\)O)\\(_{20}\\)", "tip4p21": "(H\\(_2\\)O)\\(_{21}\\)",
               "tip4p25": "(H\\(_2\\)O)\\(_{25}\\)"}


def commit_held(revision: str) -> bool:
    """The revision is an ancestor of HEAD."""
    if not revision or revision in {"unknown", "latest"}:
        return False
    return subprocess.run(
        ["git", "merge-base", "--is-ancestor", revision, "HEAD"],
        cwd=ROOT, capture_output=True,
    ).returncode == 0


# Pins of other projects. They are not commits of this repository.
EXTERNAL_PINS = (
    "aac1548a8aec13b53ad12a8d6904625081685d2e",
    "9cbec1a446352bc9ef064bd2b1a5767194655919",
    "8efdd246322d1f32bc1bcbd7e5dfdd7a4237960e",
    "71185263eb6a8130d1262fd76d90109f5fabdcf5",
    "fa8a4416e8fe093d0075bc10ac875494c2a449a9",
)


def revision_reachable(revision: str) -> bool:
    """The revision is a commit here, or one of the external pins."""
    revision = (revision or "").strip()
    if commit_held(revision):
        return True
    if not re.fullmatch(r"[0-9a-f]{7,40}", revision):
        return False
    return any(
        pin == revision or pin.startswith(revision) or revision.startswith(pin)
        for pin in EXTERNAL_PINS
    )


def show_s(value: float) -> str:
    if value < 100.0:
        return f"{value:.3f}"
    return f"{value:.1f}"


def show_calls(value: str) -> str:
    return f"{float(value):.0f}" if value else "--"


def type7_median(values: list[float]) -> float:
    ordered = sorted(values)
    h = 0.5 * (len(ordered) - 1)
    lo = int(h)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (h - lo) * (ordered[hi] - ordered[lo])


def show_calls_median(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".")


def show_p(value: float) -> str:
    """Two significant figures; 1 only when p is exactly 1."""
    if value >= 1.0:
        return "1"
    if value >= 0.01:
        return f"{value:.2f}" if value >= 0.1 else f"{value:.3f}"
    mantissa, exponent = f"{value:.1e}".split("e")
    return f"\\({mantissa}\\times 10^{{{int(exponent)}}}\\)"


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def budget_tex(budget: str) -> str:
    mantissa, exponent = f"{int(budget):.0e}".split("e")
    return f"\\({mantissa}\\times 10^{{{int(exponent)}}}\\)"


def show_sci(value: float) -> str:
    text = f"{value:.3g}"
    if "e" not in text:
        return text
    mantissa, exponent = text.split("e")
    return f"\\({mantissa}\\times 10^{{{int(exponent)}}}\\)"


def fisher_p(hits_a: int, n_a: int, hits_b: int, n_b: int) -> float:
    """Two-sided Fisher exact p, the same sum the audit uses."""
    hits = hits_a + hits_b
    ways = math.comb(n_a + n_b, hits)

    def prob(k: int) -> float:
        return math.comb(n_a, k) * math.comb(n_b, hits - k) / ways

    observed = prob(hits_a)
    lo, hi = max(0, hits - n_b), min(n_a, hits)
    return min(1.0, sum(prob(k) for k in range(lo, hi + 1) if prob(k) <= observed * (1 + 1e-7)))


def large_seeds() -> list[dict[str, str]] | None:
    """The LJ98--LJ150 seed rows, or nothing when a quoted commit is not the recorded set."""
    path = GOFEE / "lj-large-seeds.csv"
    if not path.is_file():
        return None
    rows = read(path)
    for system in LARGE_SYSTEMS:
        for arm in ("anneal", "agox_gofee"):
            group = [row for row in rows if row["system"] == system and row["arm"] == arm]
            if len(group) != 12:
                return None
    if any(row["code_revision"] != ANNEAL_SOURCE for row in rows if row["arm"] == "anneal"):
        return None
    if any(row["code_revision"] not in AGOX_HEADS for row in rows if row["arm"] == "agox_gofee"):
        return None
    if any(not commit_held(row["code_revision"]) for row in rows if row["code_revision"] != UNHELD_AGOX):
        return None
    if len({row["budget"] for row in rows}) != 1:
        return None
    return rows


def held_large_seeds() -> list[dict[str, str]] | None:
    """Held-harness AGOX rows and the matching this-work seeds."""
    rows = large_seeds()
    if not rows:
        return None
    agox = [
        row for row in rows
        if row["arm"] == "agox_gofee" and row["code_revision"] != UNHELD_AGOX
    ]
    held = {(row["system"], row["seed"]) for row in agox}
    anneal = [
        row for row in rows
        if row["arm"] == "anneal" and (row["system"], row["seed"]) in held
    ]
    return anneal + agox


def gofee_hits_sentence() -> str:
    """The hit sentence for held-harness seed pairs of the five large clusters."""
    rows = held_large_seeds()
    if not rows:
        return ""
    bits = []
    for system in LARGE_SYSTEMS:
        group = [row for row in rows if row["system"] == system and row["arm"] == "anneal"]
        hits = sum(int(row["hit"]) for row in group)
        bits.append(f"{LARGE_LABEL[system]} on {hits} of {len(group)}")
    return (
        "This setting hits " + bits[0] + " seeds, " + ", ".join(bits[1:-1])
        + " and " + bits[-1] + ", and GOFEE hits none."
    )


def gofee_wall_sentence() -> str:
    """The wall-cap sentence, or nothing when AGOX did not stop on that cap."""
    rows = held_large_seeds()
    if not rows:
        return ""
    agox = [row for row in rows if row["arm"] == "agox_gofee"]
    if any(row["stop"] != "capped" for row in agox):
        return ""
    wall = type7_median([float(row["wall_s"]) for row in agox])
    if abs(wall - 7200) > 60:
        return ""
    calls = type7_median([float(row["charged"]) for row in agox])
    budget = budget_tex(rows[0]["budget"])
    return (
        "AGOX stopped on the 2~h wall cap at a median of "
        f"{show_sci(calls)} calls, against this work's {budget}-call budget."
    )


def gofee_large_block() -> str:
    """Held-harness LJ98--LJ150 rows against AGOX GOFEE."""
    rows = held_large_seeds()
    sentence = gofee_wall_sentence()
    if not rows or not sentence:
        return ""
    all_rows = large_seeds() or []
    missing_rows = [
        row for row in all_rows
        if row["arm"] == "agox_gofee" and row["code_revision"] == UNHELD_AGOX
    ]
    body = []
    for system in LARGE_SYSTEMS:
        anneal = [row for row in rows if row["system"] == system and row["arm"] == "anneal"]
        agox = [row for row in rows if row["system"] == system and row["arm"] == "agox_gofee"]
        hits_a = sum(int(row["hit"]) for row in anneal)
        hits_g = sum(int(row["hit"]) for row in agox)
        body.append(" & ".join([
            LARGE_LABEL[system],
            f"{hits_a}/{len(anneal)}",
            f"{hits_g}/{len(agox)}",
            show_p(fisher_p(hits_a, len(anneal), hits_g, len(agox))),
        ]) + " \\\\")
    missing_body = [
        " & ".join([
            LARGE_LABEL[row["system"]], row["seed"],
        ]) + " \\\\"
        for row in sorted(missing_rows, key=lambda row: (row["system"], int(row["seed"])))
    ]
    return "\n".join([
        gofee_hits_sentence(),
        "",
        "\\begin{table}",
        "\\caption{This work against AGOX GOFEE on LJ\\(_{98}\\), LJ\\(_{102}\\),",
        "LJ\\(_{103}\\), LJ\\(_{104}\\) and LJ\\(_{150}\\).",
        "Only seed pairs whose AGOX harness commit is held are included; one core.",
        f"The budget is {budget_tex(rows[0]['budget'])} force calls.",
        sentence,
        "\\(p\\) is the two-sided Fisher exact test on the hits.",
        "The anneal rows are from \\texttt{786c1a3b}.",
        "The AGOX harness heads this repository holds are \\texttt{786c1a3b} and \\texttt{4508a253}.",
        f"The {len(missing_rows)} AGOX rows are excluded.}}",
        "\\label{tab:gofee}",
        "\\centering",
        "\\small",
        "\\begin{tabular}{@{}lrrr@{}}",
        "\\toprule",
        "Cluster & This work & GOFEE & \\(p\\) \\\\",
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
        "\\begin{table}",
        "\\caption{AGOX rows excluded from Table~\\ref{tab:gofee}.}",
        "\\label{tab:gofee-untraceable}",
        "\\centering",
        "\\small",
        "\\begin{tabular}{@{}lr@{}}",
        "\\toprule",
        "Cluster & Seed \\\\",
        "\\midrule",
        *missing_body,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])


def gofee_block() -> str:
    """The large-cluster GOFEE table, or the three-cluster table when those rows are what is quotable."""
    large = gofee_large_block()
    if large:
        return large
    return gofee_small_block()


def gofee_small_block() -> str:
    """The Lennard-Jones GOFEE table, or nothing when a quoted row cites an absent commit."""
    seeds = [row for row in read(GOFEE / "results.csv") if row["system"] in LJ_LABEL]
    if not seeds or any(not commit_held(row["head"]) for row in seeds):
        return ""
    rows = read(GOFEE / "summary.csv")
    body = []
    for system in LJ_LABEL:
        for arm in ARM_LABEL:
            row = next(r for r in rows if r["system"] == system and r["arm"] == arm)
            body.append(" & ".join([
                LJ_LABEL[system] if arm == "paper" else "", ARM_LABEL[arm],
                budget_tex(row["budget"]) if arm == "paper" else "",
                f"{row['hits']}/{row['n']}", row["capped"], show_calls(row["calls_to_hit_median"]),
                show_s(float(row["wall_median_s"])),
            ]) + " \\\\")
        if system != "lj75":
            body.append("\\addlinespace")
    return "\n".join([
        "\\begin{table}",
        "\\caption{This work against ASE basin hopping (BH), ASE minima hopping (MH)",
        "and AGOX GOFEE on the same three Lennard-Jones clusters.",
        "Seeds 2000--2005, one core, budget in force calls.",
        "Capped counts the runs stopped by a wall cap of 2 hours, which are misses.",
        "Calls is the median force-call count of the runs that hit, and Wall is the median",
        "wall in seconds of all six runs.}",
        "\\label{tab:gofee}",
        "\\centering",
        "\\small",
        "\\begin{tabular}{@{}llrrrrr@{}}",
        "\\toprule",
        "Cluster & Arm & Budget & Hits & Capped & Calls & Wall (s) \\\\",
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])


def tip4p_groups() -> dict[int, tuple[list[dict[str, str]], list[dict[str, str]]]]:
    rows = read(TIP4P)
    sizes = sorted({int(row["molecules"]) for row in rows})
    groups = {}
    for size in sizes:
        anneal = [row for row in rows if row["molecules"] == str(size) and row["arm"] == "anneal"]
        gmin = [row for row in rows if row["molecules"] == str(size) and row["arm"] == "gmin"]
        if len(anneal) != 12 or len(gmin) != 12:
            raise ValueError(f"TIP4P {size}: expected 12 rows per arm")
        a_starts = {row["seed"]: row["start_id"] for row in anneal}
        g_starts = {row["seed"]: row["start_id"] for row in gmin}
        if a_starts != g_starts:
            raise ValueError(f"TIP4P {size}: arm starts differ")
        groups[size] = (anneal, gmin)
    return groups


def tip4p_wall() -> dict[int, dict[str, str]]:
    rows = {int(row["molecules"]): row for row in read(TIP4P_WALL)}
    if set(rows) != {8, 15, 20, 21, 25}:
        raise ValueError("TIP4P wall summary has the wrong sizes")
    if any(row["mwu_method"] != "exact-rank-enumeration"
           for row in rows.values()):
        raise ValueError("TIP4P wall summary is not exact")
    raw = read(TIP4P)
    for size, row in rows.items():
        anneal = [
            float(item["wall_s"]) for item in raw
            if item["molecules"] == str(size) and item["arm"] == "anneal"
        ]
        gmin = [
            float(item["wall_s"]) for item in raw
            if item["molecules"] == str(size) and item["arm"] == "gmin"
        ]
        p_value, method = mann_whitney_u_two_sided_exact(anneal, gmin)
        if method != row["mwu_method"] or abs(
            p_value - float(row["wall_mwu_p"])
        ) > 5.0e-13:
            raise ValueError(f"TIP4P {size}: stale wall summary")
    return rows


def tip4p_prose() -> str:
    groups = tip4p_groups()
    wall = tip4p_wall()
    seeds = sorted({int(row["seed"]) for arms in groups.values() for arm in arms for row in arm})
    hit = {
        size: tuple(sum(int(row["hit"]) for row in arm) for arm in arms)
        for size, arms in groups.items()
    }
    all_rows = [row for arms in groups.values() for arm in arms for row in arm]
    if len({row["machine_id"] for row in all_rows}) != 1:
        raise ValueError("the TIP4P ABBA comparison used more than one machine")
    if {row["core"] for row in all_rows} != {"0"}:
        raise ValueError("the TIP4P ABBA comparison was not pinned to core 0")
    expected_order = ("anneal,gmin", "gmin,anneal")
    for arms in groups.values():
        rows_by_seed = {row["seed"]: row for row in arms[0]}
        orders = [
            rows_by_seed[str(seed)]["arm_order"]
            for seed in seeds
        ]
        if any(order != expected_order[index % 2]
               for index, order in enumerate(orders)):
            raise ValueError("the TIP4P comparison does not use ABBA arm order")
    shown = {
        size: show_p(float(row["wall_mwu_p"]))
        .removeprefix("\\(").removesuffix("\\)")
        for size, row in wall.items() if size != 25
    }
    repeated = set(shown.values())
    if len(repeated) != 1:
        raise ValueError("TIP4P scored sizes do not share one wall-time p")
    exhausted_cost = {
        size: (
            float(wall[size]["anneal_wall_median_s"]) * 1.0e6
            / float(wall[size]["anneal_calls_median"]),
            float(wall[size]["gmin_wall_median_s"]) * 1.0e6
            / float(wall[size]["gmin_calls_median"]),
        )
        for size in (21,)
    }
    unscored = wall[25]
    return "\n".join([
        "Table~\\ref{tab:tip4p-gmin} puts this setting against GMIN on rigid TIP4P",
        "clusters of 8, 15, 20, 21, and 25 molecules.",
        "The hit tolerance for the scored sizes is 0.001~kJ/mol.",
        f"The 120-run ABBA comparison uses seeds {seeds[0]}--{seeds[-1]} on one machine pinned to core 0,",
        "with the arm order alternating across seeds.",
        f"This setting hits \\((\\mathrm{{H_2O}})_{{8}}\\) on {hit[8][0]} of 12 against GMIN on {hit[8][1]} of 12,",
        f"\\((\\mathrm{{H_2O}})_{{15}}\\) on {hit[15][0]} of 12 against {hit[15][1]} of 12, and",
        f"\\((\\mathrm{{H_2O}})_{{20}}\\) on {hit[20][0]} of 12 against {hit[20][1]} of 12.",
        "Neither arm hits \\((\\mathrm{H_2O})_{21}\\).",
        "Size 25 has no independent reference and is unscored.",
        "This setting's best and median final energies are "
        f"{unscored['anneal_energy_best_kj_mol']} and "
        f"{unscored['anneal_energy_median_kj_mol']}~kJ/mol, and its median wall "
        f"is {unscored['anneal_wall_median_s']}~s.",
        "GMIN's corresponding values are "
        f"{unscored['gmin_energy_best_kj_mol']} and "
        f"{unscored['gmin_energy_median_kj_mol']}~kJ/mol and "
        f"{unscored['gmin_wall_median_s']}~s.",
        "Median walls for this setting and GMIN are",
        f"{float(wall[8]['anneal_wall_median_s']):.6f} and "
        f"{float(wall[8]['gmin_wall_median_s']):.6f}~s at 8 molecules,",
        f"{float(wall[15]['anneal_wall_median_s']):.6f} and "
        f"{float(wall[15]['gmin_wall_median_s']):.6f}~s at 15 molecules, and",
        f"{float(wall[20]['anneal_wall_median_s']):.6f} and "
        f"{float(wall[20]['gmin_wall_median_s']):.6f}~s at 20 molecules.",
        "At 21 molecules every run exhausts the two-million-call budget;",
        "the median costs per call for this setting and GMIN are",
        f"{exhausted_cost[21][0]:.6f} and {exhausted_cost[21][1]:.6f}~"
        "\\(\\mu\\)s at 21 molecules.",
        "The two-sided exact rank-enumeration Mann--Whitney test on wall time",
        f"gives \\(p={next(iter(repeated))}\\) at sizes 8, 15, 20, and 21.",
    ])


def tip4p_block() -> str:
    body = []
    wall = tip4p_wall()
    for size, (anneal, gmin) in tip4p_groups().items():
        if size == 25:
            continue
        a_best = [float(row["best_energy"]) for row in anneal]
        g_best = [float(row["best_energy"]) for row in gmin]
        hits = (
            f"{sum(int(row['hit']) for row in anneal)}/12",
            f"{sum(int(row['hit']) for row in gmin)}/12",
        )
        body.append(" & ".join([
            TIP4P_LABEL[f"tip4p{size}"],
            *hits,
            show_calls_median(type7_median([float(row["calls"]) for row in anneal])),
            show_calls_median(type7_median([float(row["calls"]) for row in gmin])),
            f"{type7_median(a_best):.6f}",
            f"{type7_median(g_best):.6f}",
            show_p(float(wall[size]["wall_mwu_p"])),
        ]) + " \\\\")
    return "\n".join([
        "\\begin{table}",
        "\\caption{This work against GMIN on rigid TIP4P water clusters.",
        "The 120-run ABBA comparison uses seeds 2000--2011 on one machine pinned to core 0,",
        "with the arm order alternating across seeds. Calls and Best are the medians over the 12 runs.",
        "\\(p_\\mathrm{wall}\\) is the two-sided exact rank-enumeration",
        "Mann--Whitney test on the per-run wall times.}",
        "\\label{tab:tip4p-gmin}",
        "\\centering",
        "\\scriptsize",
        "\\begin{tabular}{@{}lrrrrrrr@{}}",
        "\\toprule",
        " & \\multicolumn{2}{c}{Hits} & \\multicolumn{2}{c}{Calls} & "
        "\\multicolumn{2}{c}{Best energy (kJ/mol)} & \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
        "Cluster & This work & GMIN & This work & GMIN & This work & GMIN & "
        "\\(p_\\mathrm{wall}\\) \\\\",
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])


def gofee_si_block() -> str:
    """The other-panel GOFEE table, or nothing when its rows name no held commit."""
    rows = [r for r in read(GOFEE / "asg2-summary.csv") if r["source"] == "measured"]
    revision_keys = ("code_revision", "head", "anneal_sha", "harness_revision")
    if not rows or not any(key in rows[0] for key in revision_keys):
        return ""
    if any(not commit_held(row.get(key, "")) for row in rows for key in revision_keys if row.get(key)):
        return ""
    body = []
    for system in OTHER_LABEL:
        first = True
        for arm in ("ase_bh", "ase_mh", "agox_gofee"):
            row = next(r for r in rows if r["system"] == system and r["arm"] == arm)
            body.append(" & ".join([
                OTHER_LABEL[system] if first else "", ARM_LABEL[arm],
                budget_tex(row["budget"]) if first else "",
                f"{row['hits']}/{row['n']}", row["capped"], show_calls(row["calls_to_hit_median"]),
                show_s(float(row["wall_median_s"])),
            ]) + " \\\\")
            first = False
        if system != "tip4p8":
            body.append("\\addlinespace")
    return "\n".join([
        "\\begin{table}",
        "\\caption{ASE basin hopping (BH), ASE minima hopping (MH) and AGOX GOFEE on the",
        "non-Lennard-Jones panel systems, on the potential of this code.",
        "Six seeds, one core, budget in force calls, and a wall cap of 2 hours per run.",
        "Capped counts the runs stopped by the cap, which are misses.",
        "Calls is the median force-call count of the runs that hit, and Wall is the median",
        "wall in seconds of all six runs.}",
        "\\label{tab:gofee-si}",
        "\\centering",
        "\\small",
        "\\begin{tabular}{@{}llrrrrr@{}}",
        "\\toprule",
        "System & Arm & Budget & Hits & Capped & Calls & Wall (s) \\\\",
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])


def sc_label(system: str) -> str:
    form, size = system.removeprefix("sc").split("-")
    return f"{SC_FORM[form]}, {size}"


def sc_pairs() -> list[tuple[dict[str, str], dict[str, str]]]:
    """The anneal row of each system and the GMIN row of the corrected data file.

    The corrected file is the GMIN harness that differs from the anneal one;
    the anneal harness's own GMIN rows ran with Lennard-Jones step and
    temperature and are not shown."""
    rows = read(SC)
    pairs = []
    for system in SC_ORDER:
        anneal_rows = [r for r in rows if r["system"] == system and r["arm"] == "anneal"]
        anneal = anneal_rows[-1]
        anneal_harnesses = {r["harness_revision"] for r in anneal_rows}
        gmin = [r for r in rows if r["system"] == system and r["arm"] == "gmin"
                and r["harness_revision"] not in anneal_harnesses]
        if not gmin:
            raise SystemExit(f"{system}: expected a corrected GMIN row")
        pairs.append((anneal, gmin[-1]))
    return pairs


def sc_tests() -> dict[str, dict[str, str]]:
    """The p-values of each system against its corrected GMIN rows, from pvalues.csv."""
    tests: dict[str, dict[str, str]] = {}
    for anneal, gmin in sc_pairs():
        rows = [r for r in read(SC_P) if r["system"] == anneal["system"]
                and r["code_revision"] == anneal["code_revision"]
                and r["harness_revision"] == gmin["harness_revision"]]
        if len(rows) != 1:
            raise SystemExit(f"{anneal['system']}: expected one pvalues.csv row, found {len(rows)}")
        tests[anneal["system"]] = rows[0]
    return tests


def sc_gmin_quotable() -> bool:
    """Every printed arm names a harness this repository holds, or an external pin."""
    pairs = sc_pairs()
    if len(pairs) != len(SC_ORDER):
        return False
    for anneal, gmin in pairs:
        for revision in (
            anneal["harness_revision"], anneal["code_revision"],
            gmin["harness_revision"], gmin["code_revision"],
        ):
            if not revision_reachable(revision):
                return False
    return True


def sc_prose() -> str:
    """Plain-language comparison recomputed from the latest template-free rows."""
    if not sc_gmin_quotable():
        return ""
    pairs = sc_pairs()
    tests = sc_tests()
    ahead = [a["system"] for a, g in pairs if int(a["hits"]) > int(g["hits"])]
    tied = [a["system"] for a, g in pairs if int(a["hits"]) == int(g["hits"])]
    behind = [a["system"] for a, g in pairs if int(a["hits"]) < int(g["hits"])]
    significant = [
        a["system"] for a, _g in pairs if float(tests[a["system"]]["fisher_p"]) < 0.05
    ]
    gmin_wall = sum(float(g["wall_median_s"]) < float(a["wall_median_s"]) for a, g in pairs)
    inside_ratios = [float(a["pot_per_eval_s"]) / float(g["pot_per_eval_s"]) for a, g in pairs]

    def names(systems: list[str]) -> str:
        labels = [sc_label(system) for system in systems]
        return ", ".join(labels[:-1]) + (", and " if len(labels) > 2 else " and ") + labels[-1]

    return "\n".join([
        f"This work has more hits on {len(ahead)} of the {len(pairs)} systems,",
        f"the arms tie on {len(tied)}, and GMIN has more hits on {len(behind)}.",
        f"The two-sided Fisher test is below 0.05 on {len(significant)} systems:",
        f"{names(significant)}; all {len(significant)} favor this work.",
        f"GMIN nevertheless has the lower median raw wall time on {gmin_wall} of {len(pairs)} systems.",
        "Per call, this work spends "
        f"{min(inside_ratios):.1f} to {max(inside_ratios):.1f} times as long "
        "inside the potential as GMIN.",
    ])


def sc_block() -> str:
    if not sc_gmin_quotable():
        return _sc_anneal_block()
    body = []
    tests = sc_tests()
    seed_rows = read(SC_SEEDS)
    for anneal, gmin in sc_pairs():
        system = anneal["system"]
        test = tests[system]
        a_seeds = [float(r["evals_to_hit"]) for r in seed_rows
                   if r["system"] == system and r["arm"] == "anneal"
                   and r["harness_revision"] == anneal["harness_revision"]]
        g_seeds = [float(r["evals_to_hit"]) for r in seed_rows
                   if r["system"] == system and r["arm"] == "gmin"
                   and r["harness_revision"] == gmin["harness_revision"]]
        if len(a_seeds) != 12 or len(g_seeds) != 12:
            raise SystemExit(f"{system}: expected 12 seed rows per arm, found {len(a_seeds)} and {len(g_seeds)}")
        a_med = type7_median(a_seeds)
        g_med = type7_median(g_seeds)
        body.append(" & ".join([
            sc_label(system), budget_tex(anneal["budget"]),
            anneal["hits_over_n"], gmin["hits_over_n"],
            show_calls_median(a_med), show_calls_median(g_med),
            show_s(float(anneal["wall_median_s"])), show_s(float(gmin["wall_median_s"])),
            show_p(float(test["fisher_p"])), show_p(float(test["mannwhitney_p_evals"])),
            show_p(float(test["mannwhitney_p_wall"])),
        ]) + " \\\\")
    return "\n".join([
        "\\begin{table}",
        "\\caption{This work against GMIN on reduced Sutton--Chen clusters at equal workload.",
        "Form \\(n\\)-\\(m\\) and size \\(N\\); seeds 2000--2011; the same start file and",
        "the same force-call budget for both arms; one run at a time on one core.",
        "Calls is the median force-call count with a miss at the budget, and Wall the",
        "median wall in seconds over the 12 runs.",
        "\\(p_\\mathrm{F}\\) is the two-sided Fisher exact test on hits, and \\(p_\\mathrm{U}\\)",
        "the two-sided continuity-corrected asymptotic Mann--Whitney test on calls and on wall",
        "against the Doye--Wales Sutton--Chen GMIN rows, as",
        "\\texttt{scripts/metal\\_h2h/sc\\_n12\\_pvalues.py} writes them to",
        "\\texttt{bench/metal-cluster-h2h/sc-n12/pvalues.csv}.}",
        "\\label{tab:sc}",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.5pt}%",
        "\\begin{tabular}{@{}lrrrrrrrrrr@{}}",
        "\\toprule",
        " & & \\multicolumn{2}{c}{Hits} & \\multicolumn{2}{c}{Calls} & \\multicolumn{2}{c}{Wall (s)} & & "
        "\\multicolumn{2}{c}{\\(p_\\mathrm{U}\\)} \\\\",
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}\\cmidrule(lr){10-11}",
        "Form, \\(N\\) & Budget & This work & GMIN & This work & GMIN & This work & GMIN & "
        "\\(p_\\mathrm{F}\\) & Calls & Wall \\\\",
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])


def _sc_anneal_block() -> str:
    """This setting only. The GMIN arm of the comparison cites a commit this repository does not hold."""
    body = []
    seed_rows = read(SC_SEEDS)
    harnesses = set()
    for anneal, _gmin in sc_pairs():
        if not commit_held(anneal["harness_revision"]):
            continue
        system = anneal["system"]
        harnesses.add(anneal["harness_revision"][:8])
        calls = [float(row["evals_to_hit"]) for row in seed_rows
                 if row["system"] == system and row["arm"] == "anneal"
                 and row["harness_revision"] == anneal["harness_revision"]]
        if len(calls) != 12:
            raise SystemExit(f"{system}: expected 12 held anneal seed rows, found {len(calls)}")
        body.append(" & ".join([
            sc_label(system), budget_tex(anneal["budget"]), anneal["hits_over_n"],
            show_calls_median(type7_median(calls)), show_s(float(anneal["wall_median_s"])),
        ]) + " \\\\")
    if not body:
        return ""
    shown = ", ".join(f"\\texttt{{{item}}}" for item in sorted(harnesses))
    return "\n".join([
        "\\begin{table}",
        "\\caption{This setting on reduced Sutton--Chen clusters.",
        "Form \\(n\\)-\\(m\\) and size \\(N\\); seeds 2000--2011; one run at a time on one core.",
        "Calls is the median force-call count with a miss at the budget, and Wall the",
        f"median wall in seconds over the 12 runs. Every row is harness {shown}.}}",
        "\\label{tab:sc}",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.5pt}%",
        "\\begin{tabular}{@{}lrrrr@{}}",
        "\\toprule",
        "Form, \\(N\\) & Budget & Hits & Calls & Wall (s) \\\\",
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])


def _sc_anneal_si_block() -> str:
    body = []
    harnesses = set()
    for anneal, _gmin in sc_pairs():
        if not commit_held(anneal["harness_revision"]):
            continue
        harnesses.add(anneal["harness_revision"][:8])
        body.append(" & ".join([
            sc_label(anneal["system"]), f"\\({anneal['reference']}\\)",
            f"{1e6 * float(anneal['pot_per_eval_s']):.2f}",
            f"{1e6 * float(anneal['outside_per_eval_s']):.2f}",
        ]) + " \\\\")
    if not body:
        return ""
    shown = ", ".join(f"\\texttt{{{item}}}" for item in sorted(harnesses))
    return "\n".join([
        "\\begin{table}",
        "\\caption{Reference energies and the cost per force call of this setting on Sutton--Chen.",
        "The reference is the Doye--Wales global minimum in reduced units.",
        "Inside is the median time per call inside the energy and gradient, and",
        "Outside the median time per call spent elsewhere, in microseconds.",
        f"Every row is harness {shown}.}}",
        "\\label{tab:sc-si}",
        "\\centering",
        "\\small",
        "\\begin{tabular}{@{}lrrr@{}}",
        "\\toprule",
        "Form, \\(N\\) & Reference & Inside (\\(\\mu\\)s) & Outside (\\(\\mu\\)s) \\\\",
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])


def sc_si_block() -> str:
    if not sc_gmin_quotable():
        return _sc_anneal_si_block()
    body = []
    pairs = sc_pairs()
    for anneal, gmin in pairs:
        body.append(" & ".join([
            sc_label(anneal["system"]), f"\\({anneal['reference']}\\)",
            f"{1e6 * float(anneal['pot_per_eval_s']):.2f}", f"{1e6 * float(gmin['pot_per_eval_s']):.2f}",
            f"{1e6 * float(anneal['outside_per_eval_s']):.2f}", f"{1e6 * float(gmin['outside_per_eval_s']):.2f}",
            gmin["harness_revision"][:8],
        ]) + " \\\\")
    return "\n".join([
        "\\begin{table}",
        "\\caption{Reference energies and the cost per force call of the Sutton--Chen panel.",
        "The reference is the Doye--Wales global minimum in reduced units.",
        "Inside is the median time per call inside the energy and gradient, and",
        "Outside the median time per call spent elsewhere, in microseconds.",
        "Harness is the revision of the GMIN Sutton--Chen data file; every anneal row is",
        f"harness \\texttt{{{pairs[0][0]['harness_revision'][:8]}}}.}}",
        "\\label{tab:sc-si}",
        "\\centering",
        "\\small",
        "\\begin{tabular}{@{}lrrrrrl@{}}",
        "\\toprule",
        " & & \\multicolumn{2}{c}{Inside (\\(\\mu\\)s)} & \\multicolumn{2}{c}{Outside (\\(\\mu\\)s)} & \\\\",
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
        "Form, \\(N\\) & Reference & This work & GMIN & This work & GMIN & Harness \\\\",
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])


def tuned_gmin_summary() -> list[dict[str, object]]:
    """Recompute tuned-GMIN shard 2 directly from its per-seed rows."""
    rows = read(GMIN_TUNED)
    summary = []
    for system in ("lj98", "lj104"):
        arms = {
            arm: [row for row in rows if row["system"] == system and row["code"] == arm]
            for arm in ("anneal", "gmin")
        }
        if any(len(group) != 12 for group in arms.values()):
            raise SystemExit(f"{GMIN_TUNED}: {system} needs 12 rows per arm")
        ah = sum(int(row["hit"]) for row in arms["anneal"])
        gh = sum(int(row["hit"]) for row in arms["gmin"])
        fisher = fisher_p(ah, 12, gh, 12)
        summary.append({
            "system": system,
            "anneal_hits": ah,
            "gmin_hits": gh,
            "anneal_calls": type7_median([float(row["calls_to_hit"]) for row in arms["anneal"]]),
            "gmin_calls": type7_median([float(row["calls_to_hit"]) for row in arms["gmin"]]),
            "anneal_wall": type7_median([float(row["wall_s"]) for row in arms["anneal"]]),
            "gmin_wall": type7_median([float(row["wall_s"]) for row in arms["gmin"]]),
            "fisher": fisher,
            "fisher_two": min(1.0, 2.0 * fisher),
        })
    return summary


def tuned_gmin_harness_verified(row: dict[str, str]) -> bool:
    """Verify a tuned-GMIN row: reachable code_revision, exact data_blob and tuned.py in HEAD."""
    system = (row.get("system") or "").strip()
    data_blob = (row.get("data_blob") or "").strip()
    harness = (row.get("harness_commit") or "").strip()
    code_rev = (row.get("code_revision") or "").strip()
    if system not in {"lj98", "lj104"} or not data_blob or not harness:
        return False
    if not revision_reachable(code_rev):
        return False
    if revision_reachable(harness):
        return True
    def git_out(*args: str) -> tuple[int, str]:
        proc = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)
        return proc.returncode, proc.stdout.strip()
    rc_data, head_data = git_out("rev-parse", f"HEAD:bench/gmin-tuned/inputs/{system}.data")
    rc_tuned, head_tuned = git_out("rev-parse", "HEAD:scripts/gmin_panel/tuned.py")
    if rc_data != 0 or head_data != data_blob or rc_tuned != 0:
        return False
    if git_out("cat-file", "-e", f"{harness}^{{commit}}")[0] != 0:
        return False
    rc_rd, ref_data = git_out("rev-parse", f"{harness}:bench/gmin-tuned/inputs/{system}.data")
    rc_rt, ref_tuned = git_out("rev-parse", f"{harness}:scripts/gmin_panel/tuned.py")
    if rc_rd != 0 or ref_data != data_blob or rc_rt != 0 or ref_tuned != head_tuned:
        return False
    rc_base, base = git_out("merge-base", "HEAD", harness)
    if rc_base != 0 or not base:
        return False
    rc_diff, diff_names = git_out("diff", "--name-only", base, harness)
    if rc_diff != 0:
        return False
    if any(
        name.startswith(("src/", "examples/")) or name in {"Cargo.toml", "Cargo.lock"}
        for name in diff_names.splitlines()
    ):
        return False
    return True


def tuned_gmin_block() -> str:
    used = [
        row for row in read(GMIN_TUNED)
        if row["system"] in {"lj98", "lj104"} and row["code"] in {"anneal", "gmin"}
    ]
    for row in used:
        if not tuned_gmin_harness_verified(row):
            return ""
    summary = tuned_gmin_summary()
    body = []
    for row in summary:
        body.append(" & ".join([
            f"LJ\\(_{{{str(row['system']).removeprefix('lj')}}}\\)",
            f"{row['anneal_hits']}/12", f"{row['gmin_hits']}/12",
            show_calls_median(float(row["anneal_calls"])),
            show_calls_median(float(row["gmin_calls"])),
            show_s(float(row["anneal_wall"])), show_s(float(row["gmin_wall"])),
            show_p(float(row["fisher"])), show_p(float(row["fisher_two"])),
        ]) + " \\\\")
    lj98, lj104 = summary
    return "\n".join([
        f"On LJ\\(_{{98}}\\), this work hits {lj98['anneal_hits']} of 12 and tuned GMIN",
        f"hits {lj98['gmin_hits']} of 12; the two-test-corrected Fisher value is",
        f"\\({show_p(float(lj98['fisher_two']))}\\),",
        "which is not below 0.05.",
        f"On LJ\\(_{{104}}\\), tuned GMIN hits {lj104['gmin_hits']} of 12 against",
        f"{lj104['anneal_hits']} of 12 for this work; the corrected value is",
        f"\\({show_p(float(lj104['fisher_two']))}\\).",
        "This is a GMIN advantage in the observed counts, is not significant,",
        "and is not a win for this work.",
        "",
        "\\begin{table}",
        "\\caption{This work against tuned GMIN in shard 2.",
        "Seeds 2000--2011; equal force-call budgets; one core.",
        "Calls and wall are medians over 12 runs, with misses charged at the budget.",
        "\\(p_\\mathrm{F}\\) is the two-sided Fisher exact test on hits and",
        "\\(p_{\\mathrm{F},2}\\) its Bonferroni value for these two clusters.}",
        "\\label{tab:gmin-tuned}",
        "\\centering",
        "\\small",
        "\\begin{tabular}{@{}lrrrrrrrr@{}}",
        "\\toprule",
        " & \\multicolumn{2}{c}{Hits} & \\multicolumn{2}{c}{Calls} & "
        "\\multicolumn{2}{c}{Wall (s)} & & \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
        "Cluster & This work & GMIN & This work & GMIN & This work & GMIN & "
        "\\(p_\\mathrm{F}\\) & \\(p_{\\mathrm{F},2}\\) \\\\",
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])


def abcluster_block() -> str:
    """Au55 recommended preset against ABCluster, excluding the same-code arm."""
    rows = read(ABCLUSTER)
    groups = {
        arm: [row for row in rows if row["system"] == "au55" and row["arm"] == arm]
        for arm in ("anneal-recommended", "abcluster")
    }
    if any(len(group) != 12 for group in groups.values()):
        raise SystemExit(f"{ABCLUSTER}: Au55 needs 12 recommended and 12 ABCluster rows")
    ah = sum(int(row["hit"]) for row in groups["anneal-recommended"])
    bh = sum(int(row["hit"]) for row in groups["abcluster"])
    p_value = fisher_p(ah, 12, bh, 12)
    body = []
    for arm, label in (("anneal-recommended", "This work"), ("abcluster", "ABCluster")):
        group = groups[arm]
        body.append(" & ".join([
            label, f"{sum(int(row['hit']) for row in group)}/12",
            show_calls_median(type7_median([float(row["calls_to_hit"]) for row in group])),
            show_s(type7_median([float(row["wall_s"]) for row in group])),
        ]) + " \\\\")
    return "\n".join([
        f"This work hits the Au\\(_{{55}}\\) target on {ah} of 12 seeds and ABCluster",
        f"on {bh} of 12.  The two-sided Fisher value is \\({show_p(p_value)}\\),",
        "so the observed advantage is not significant.",
        "The same-code \\texttt{anneal-cluster} arm in the rows is a preset diagnostic,",
        "not a comparison with another code, and is excluded.",
        "",
        "\\begin{table}",
        "\\caption{This work against ABCluster on Au\\(_{55}\\).",
        "Seeds 2000--2011; one-million-call budget; one core.",
        "Calls and wall are medians over the 12 runs, with misses charged at the budget.}",
        "\\label{tab:abcluster}",
        "\\centering",
        "\\small",
        "\\begin{tabular}{@{}lrrr@{}}",
        "\\toprule",
        "Arm & Hits & Calls & Wall (s) \\\\",
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
    ])


def crest_equal_summary() -> dict[str, dict[str, str]]:
    """The summary row of each arm at the revision of the latest rows, checked against the rows."""
    rows = read(CREST_EQUAL / "rows.csv")
    revision = rows[-1]["code_revision"]
    rows = [r for r in rows if r["code_revision"] == revision]
    summary = {r["arm"]: r for r in read(CREST_EQUAL / "summary.csv") if r["code_revision"] == revision}
    if set(summary) != set(CREST_ARMS):
        raise SystemExit(f"crest-equal: summary arms at {revision[:8]} are {sorted(summary)}")
    for arm, row in summary.items():
        runs = [r for r in rows if r["arm"] == arm]
        hits = sum(r["hit"] == "true" for r in runs)
        if (len(runs), hits) != (int(row["n"]), int(row["hits"])):
            raise SystemExit(f"crest-equal {arm}: rows give {hits}/{len(runs)}, summary {row['hits']}/{row['n']}")
    return summary


def crest_p(value: str) -> str:
    """Three decimals down to 0.001, then two significant figures."""
    if not value:
        return "--"
    number = float(value)
    if number >= 0.999:
        return "1"
    if number >= 0.001:
        return f"{number:.3f}"
    return show_p(number)


def crest_equal_block() -> str:
    summary = crest_equal_summary()
    label = {"anneal": "This work", "crest": "CREST", "crest-wall": "CREST"}
    body = []
    for arm in CREST_ARMS:
        row = summary[arm]
        body.append(" & ".join([
            label[arm], row["crest_len_ps"] or "--", f"{row['hits']}/{row['n']}",
            f"{float(row['calls_median']):.1f}".removesuffix(".0"), f"{row['calls_min']}--{row['calls_max']}",
            row["within_5pct"], f"{float(row['wall_median_s']):.1f}", f"{float(row['pot_median_s']):.2f}",
            f"{float(row['outside_median_s']):.2f}", crest_p(row["fisher_p"]),
            crest_p(row["mannwhitney_calls_p"]), crest_p(row["mannwhitney_inside_p"]),
            crest_p(row["mannwhitney_wall_p"]),
        ]) + " \\\\")
    return "\n".join([
        "\\begin{table*}",
        "\\caption{This work against CREST 3.0.2 on the water hexamer at equal GFN2 calls.",
        "Both arms evaluate GFN2-xTB through xtb 6.7.1 at accuracy 0.01 plus the one-sided O--H",
        "wall; seeds 2000--2011; one run at a time on one core.",
        "Length is the \\texttt{-mquick} length of CREST in picoseconds.",
        "Calls is the GFN2 energy and gradient count served by the driver, the median",
        "and the range; the CREST rows had a driver ceiling of 50000 calls, and their",
        "4000 budget field is a label rather than a cap.",
        "\\(\\pm5\\%\\)",
        f"counts the runs within {100 * CREST_CALL_TOLERANCE:.0f}\\% of the {float(summary['anneal']['calls_median']):.0f} "
        "calls of this work.",
        "Wall, Inside, and Outside are medians in seconds of the whole run, of the time inside the",
        "potential, and of the rest.",
        "A hit is a vtight polish of the best structure within \\(10^{-7}\\)~Eh of the prism.",
        "\\(p_\\mathrm{F}\\) is the two-sided Fisher exact test on hits and \\(p_\\mathrm{U}\\) the",
        "two-sided Mann--Whitney test, each against this work, as",
        "\\texttt{scripts/crest\\_equal/run\\_panel.py} writes them to",
        "\\texttt{bench/crest-equal/summary.csv}.}",
        "\\label{tab:crest-equal}",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{3.5pt}%",
        "\\begin{tabular}{@{}lrrrrrrrrrrrr@{}}",
        "\\toprule",
        " & & & \\multicolumn{3}{c}{Calls} & & & & & \\multicolumn{3}{c}{\\(p_\\mathrm{U}\\)} \\\\",
        "\\cmidrule(lr){4-6}\\cmidrule(lr){11-13}",
        "Arm & Length & Hits & Median & Range & \\(\\pm5\\%\\) & Wall & Inside & Outside & "
        "\\(p_\\mathrm{F}\\) & Calls & Inside & Wall \\\\",
        "\\midrule",
        *body,
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table*}",
    ])


BLOCKS = {
    "gofee": (SI, gofee_block),
    "tip4p-prose": (SI, tip4p_prose),
    "tip4p": (SI, tip4p_block),
    "gofee-si": (SI, gofee_si_block),
    "sc-prose": (MANUSCRIPT, sc_prose),
    "sc": (MANUSCRIPT, sc_block),
    "sc-si": (SI, sc_si_block),
    "gmin-tuned": (SI, tuned_gmin_block),
    "abcluster": (SI, abcluster_block),
}


def splice(text: str, marks: tuple[str, str], block: str) -> str:
    """Put ``block`` between ``marks``. An empty block clears the interior and keeps the marks."""
    start, end = marks
    if text.count(start) != 1 or text.count(end) != 1:
        if not block and start not in text and end not in text:
            return text
        raise SystemExit(f"{start!r}: the file needs exactly one block")
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    middle = f"\n{block}\n" if block else "\n"
    return f"{head}{start}{middle}{end}{tail}"


def refresh(names: tuple[str, ...]) -> None:
    """Rewrite the named blocks in place. Other blocks are left as they are."""
    texts: dict[Path, str] = {}
    for name in names:
        path, make = BLOCKS[name]
        text = texts.get(path, path.read_text())
        texts[path] = splice(text, MARKS[name], make())
    for path, text in texts.items():
        path.write_text(text)


def main(argv: list[str]) -> int:
    stale = False
    override = os.environ.get("H2H_TABLES_OUTPUT", "")
    checking = "--check" in argv
    generated: dict[Path, str] = {}
    for name, (path, make) in BLOCKS.items():
        target = Path(override) / path.name if override else path
        if not checking and target.exists() and target not in generated:
            raise SystemExit(f"refusing to overwrite {target}")
        text = generated.get(target)
        if text is None:
            source = target if target.exists() else path
            text = source.read_text()
        updated = splice(text, MARKS[name], make())
        if checking:
            if updated != text:
                print(f"the {name} table in {target.name} is stale", file=sys.stderr)
                stale = True
        else:
            generated[target] = updated
    if not checking:
        for target, text in generated.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
    return 1 if stale else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
