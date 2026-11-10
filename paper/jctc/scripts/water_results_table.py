#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Print the water hexamer against CREST from the water_v45b per-seed table.

Reads data/water_v45b/water-v45b-perseed.csv (rule water_crest_fair of anneal_repro
v4.5b) and recomputes Wilson (z=1.96), exact two-sided McNemar, and Newcombe 1998
method 10 from the per-seed hits. Four arms on anneal 36086e98, seeds 0 to 191 at a
budget of 4000 force calls: the default hop, the hop with the budget-fair polish,
CREST, and CREST plus the polish under the same budget (crest_fair).

The LJ75 rows of --supplement still come from the sealed results_jcc folders.

Usage: water_results_table.py [--headline | --supplement | --values] [data-dir]
"""
from __future__ import annotations

import csv
import hashlib
import math
import sys
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import NoReturn

sys.path.insert(0, str(Path(__file__).resolve().parent))

import results_jcc

ROOT = Path(__file__).resolve().parents[1]
WATER_DATA = ROOT / "data" / "water_v45b"
DATA = WATER_DATA
LJ_DATA = results_jcc.DATA
CSV_NAME = "water-v45b-perseed.csv"
PROVENANCE_NAME = "provenance.txt"
EXPECTED_CSV_SHA256 = "d7c661258f757619b86fbf841ff270d7d872b984d306f819efbb3d933c041b7c"
TABLE = "tbl:water"
SEEDS = "tbl:sealed-rows"
SEED_SPAN = "0-191"
RUNS = 192
BUDGET = 4000
WILSON_Z = 1.96
ANNEAL_COMMIT = "36086e988c5a71c156ffcbde152bf321a6f5ef25"
RULE = "water_crest_fair"
CREST_FOLDER = "crest"

# folder id, hit column, wall column, calls column, printed label, scored vs CREST, cost kind
ARMS = (
    ("anneal", "anneal_hit", "anneal_wall_s", "anneal_calls", "anneal, default hop", True, "median_wall"),
    ("polish_fair", "polish_fair_hit", "polish_fair_wall_s", "polish_fair_calls", "anneal, budget-fair polish, opt-in", True, "max_calls"),
    (CREST_FOLDER, "crest_hit", "crest_wall_s", "crest_calls", "CREST", False, "median_wall"),
    ("crest_fair", "crest_fair_hit", "crest_fair_wall_s", "crest_fair_calls", "CREST plus the polish, budget-fair", False, "max_calls"),
)
LJ_ROWS = (
    ("lj_cell", "recommended", "anneal, recommended hop"),
    ("lj_cell", "derived", "anneal, derived hop"),
    ("gmin_lj75", None, "GMIN, core orbits"),
)
MAIN_ROWS = (
    ("Hits", "hits"),
    (r"Wilson 95\%, low", "low"),
    (r"Wilson 95\%, high", "high"),
    ("Both hit", "both"),
    ("Only the hop hit", "only_anneal"),
    ("Only CREST hit", "only_crest"),
    ("Neither hit", "neither"),
    ("Difference in hit rate", "difference"),
    (r"Newcombe 95\%, low", "newcombe_low"),
    (r"Newcombe 95\%, high", "newcombe_high"),
    ("Exact McNemar p", "p"),
)
PORT_PIN = r"\textcolor{{red}}{{[TODO(pin): the public tag of the anneal\_repro port at {commit}]}}"
# Hit gate and host are known for the water_v45b campaign; no red pin in the caption.
WATER_GATE = (
    r"A run hits when IRA~\cite{gundeIraShapeMatching2021} matches a structure to a "
    r"Malloum~\cite{malloumStructuresRelativeStability2019} prism, cage, book or cyclic-chair "
    r"hexamer reoptimised at GFN2-xTB~\cite{bannwarthGfn2xtbanAccurateBroadly2019}."
)
WATER_HOST = (
    r"Wall times are from an Intel Xeon family 6 model 173 host with 8 cores, "
    r"49342768\,KiB RAM, Ubuntu 24.04.5, kernel 6.12.94+."
)


def die(message: str) -> NoReturn:
    raise SystemExit(f"water_results_table: {message}")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(hits: int, n: int, z: float = WILSON_Z) -> tuple[float, float]:
    """Wilson score interval at the campaign z, clamped to [0, 1]."""
    if n <= 0:
        raise ValueError("Wilson interval needs a positive sample")
    if hits < 0 or hits > n:
        raise ValueError("hits outside 0..n")
    probability = hits / n
    z2 = z * z
    denominator = 1.0 + z2 / n
    centre = (probability + z2 / (2.0 * n)) / denominator
    margin = z * math.sqrt(probability * (1.0 - probability) / n + z2 / (4.0 * n * n)) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def mcnemar_exact_p(arm_only: int, other_only: int) -> float:
    """Two-sided exact McNemar p-value: twice the lower binomial tail at p=0.5, capped at 1."""
    if arm_only < 0 or other_only < 0:
        raise ValueError("discordant counts must be non-negative")
    discordant = arm_only + other_only
    if discordant == 0:
        return 1.0
    smaller = min(arm_only, other_only)
    tail = sum(math.comb(discordant, index) for index in range(smaller + 1))
    probability = Fraction(2 * tail, 1 << discordant)
    return 1.0 if probability > 1 else float(probability)


def newcombe_paired(both: int, arm_only: int, other_only: int, neither: int) -> tuple[float, float]:
    """Newcombe 1998 method 10 for the paired hit-rate difference (arm minus other)."""
    counts = (both, arm_only, other_only, neither)
    if any(count < 0 for count in counts):
        raise ValueError("contingency counts must be non-negative")
    total = sum(counts)
    if total <= 0:
        raise ValueError("Newcombe interval needs a positive count")
    arm_hits = both + arm_only
    other_hits = both + other_only
    arm_rate = arm_hits / total
    other_rate = other_hits / total
    difference = arm_rate - other_rate
    arm_low, arm_high = wilson(arm_hits, total)
    other_low, other_high = wilson(other_hits, total)
    marginal_product = arm_hits * (other_only + neither) * other_hits * (arm_only + neither)
    if marginal_product == 0:
        phi = 0.0
    else:
        covariance = both * neither - arm_only * other_only
        if covariance > 0:
            covariance = max(covariance - total / 2.0, 0.0)
        phi = covariance / math.sqrt(marginal_product)

    def mover(left: float, right: float) -> float:
        value = left * left - 2.0 * phi * left * right + right * right
        if value < 0.0:
            if value < -1e-12:
                die(f"Newcombe radicand {value} is negative")
            value = 0.0
        return math.sqrt(value)

    lower = difference - mover(arm_rate - arm_low, other_high - other_rate)
    upper = difference + mover(arm_high - arm_rate, other_rate - other_low)
    return lower, upper


def _round6(value: float) -> float:
    return float(f"{value:.6f}")


def paired_contrast(arm_hits: list[int], other_hits: list[int], *, difference: str) -> dict:
    """Contingency, exact McNemar p and Newcombe method 10 for two per-seed hit vectors."""
    if len(arm_hits) != len(other_hits):
        die("paired arms have different lengths")
    both = arm_only = other_only = neither = 0
    for left, right in zip(arm_hits, other_hits, strict=True):
        if left and right:
            both += 1
        elif left:
            arm_only += 1
        elif right:
            other_only += 1
        else:
            neither += 1
    total = both + arm_only + other_only + neither
    if total != RUNS:
        die(f"paired outcomes add to {total}, not {RUNS}")
    low, high = newcombe_paired(both, arm_only, other_only, neither)
    return {
        "both_hit": both,
        "difference": difference,
        "hit_rate_difference": (arm_only - other_only) / total,
        "mcnemar_exact_p": mcnemar_exact_p(arm_only, other_only),
        "n": total,
        "neither_hit": neither,
        "newcombe": "Newcombe 1998 method 10",
        "newcombe_95": [low, high],
        "newcombe_95_6": [_round6(low), _round6(high)],
        "only_anneal_hit": arm_only,
        "only_crest_hit": other_only,
    }


def _median_wall(values: list[str]) -> str:
    ordered = sorted(Decimal(value) for value in values)
    n = len(ordered)
    if n % 2:
        mid = ordered[n // 2]
    else:
        mid = (ordered[n // 2 - 1] + ordered[n // 2]) / 2
    text = format(mid, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def read_provenance(data: Path) -> dict[str, str]:
    keys: dict[str, str] = {}
    for line in (data / PROVENANCE_NAME).read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        first, _, rest = line.partition(" ")
        if len(first) == 64 and all(c in "0123456789abcdef" for c in first):
            keys[f"file:{rest.strip()}"] = first
        else:
            keys[first] = rest.strip()
    return keys


@dataclass(frozen=True)
class Arm:
    folder: str
    label: str
    record: dict[str, str]
    summary: dict
    contrast: dict | None
    hits_bits: tuple[int, ...]

    @property
    def hits(self) -> str:
        return f"{self.summary['hits']}/{self.summary['n']}"

    @property
    def wilson(self) -> tuple[str, str]:
        return tuple(repr(bound) for bound in self.summary["wilson_95"])

    @property
    def rule(self) -> str:
        return self.record["rule"]

    @property
    def binary(self) -> str:
        return self.record["binary_sha256"]

    @property
    def cost(self) -> tuple[str, str]:
        if "median_wall_s" in self.summary:
            return "median wall (s)", self.summary["median_wall_s"]
        return "most calls of a run", str(self.summary["max_total_calls"])


def load(data: Path = DATA) -> tuple[dict[str, str], tuple[Arm, ...], dict[str, results_jcc.Folder]]:
    """Load water arms from the water_v45b CSV; LJ seals still come from results_jcc."""
    water = Path(data)
    if not (water / CSV_NAME).is_file():
        # Figure rules may still pass results_jcc; the CSV lives under data/water_v45b/.
        water = WATER_DATA
    csv_path = water / CSV_NAME
    if not csv_path.is_file():
        die(f"{csv_path} is missing")
    if digest(csv_path) != EXPECTED_CSV_SHA256:
        die(f"{CSV_NAME} does not match the sealed sha256 {EXPECTED_CSV_SHA256}")
    provenance = read_provenance(water)
    listed = provenance.get(f"file:{CSV_NAME}")
    if listed and listed != EXPECTED_CSV_SHA256:
        die(f"{PROVENANCE_NAME} records a different digest for {CSV_NAME}")
    if provenance.get("rule") not in (None, RULE):
        die(f"{PROVENANCE_NAME} names rule {provenance.get('rule')!r}, not {RULE}")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != RUNS:
        die(f"{CSV_NAME} has {len(rows)} rows, not {RUNS}")
    seeds = [int(row["seed"]) for row in rows]
    if seeds != list(range(RUNS)):
        die(f"{CSV_NAME} seeds are not {SEED_SPAN}")

    hit_cols = {folder: col for folder, col, *_ in ARMS}
    bits = {folder: tuple(int(row[col]) for row in rows) for folder, col in hit_cols.items()}
    crest_bits = bits[CREST_FOLDER]

    arms: list[Arm] = []
    for folder, hit_col, wall_col, calls_col, label, scored, cost_kind in ARMS:
        hits_bits = bits[folder]
        hits = sum(hits_bits)
        low, high = wilson(hits, RUNS)
        summary: dict = {
            "anneal_commit": ANNEAL_COMMIT,
            "arm": folder,
            "budget": BUDGET,
            "hits": hits,
            "n": RUNS,
            "seeds": SEED_SPAN,
            "wilson_95": [low, high],
        }
        walls = [row[wall_col] for row in rows]
        calls = [int(row[calls_col]) for row in rows]
        if cost_kind == "median_wall":
            summary["median_wall_s"] = _median_wall(walls)
        else:
            summary["max_total_calls"] = max(calls)
            summary["median_wall_s_extra"] = _median_wall(walls)
        contrast = None
        if scored:
            contrast = paired_contrast(
                list(hits_bits),
                list(crest_bits),
                difference=("anneal minus CREST" if folder == "anneal" else "fair polish minus CREST"),
            )
            if contrast["both_hit"] + contrast["only_anneal_hit"] != hits:
                die(f"{folder}: paired outcomes do not give the hits of the arm")
            if contrast["both_hit"] + contrast["only_crest_hit"] != sum(crest_bits):
                die(f"{folder}: paired outcomes do not give the hits of CREST")
        record = {
            "anneal_commit": ANNEAL_COMMIT,
            "arm": folder,
            "binary_sha256": EXPECTED_CSV_SHA256,
            "rule": RULE,
            "hits": f"{hits}/{RUNS}",
        }
        arms.append(Arm(folder, label, record, summary, contrast, hits_bits))

    lj_keys, lj_folders = results_jcc.load(LJ_DATA)
    keys = {
        **lj_keys,
        "anneal_commit": ANNEAL_COMMIT,
        "water_rule": RULE,
        "water_csv_sha256": EXPECTED_CSV_SHA256,
        "anneal_repro_commit": provenance.get("anneal_repro_commit", ""),
        "anneal_repro_tip": provenance.get("anneal_repro_tip", ""),
        "bundle_sha256": provenance.get("bundle_sha256", ""),
        "summary_json_sha256": provenance.get("summary_json_sha256", ""),
        "provenance_sha256": provenance.get("provenance_sha256", ""),
    }
    return keys, tuple(arms), lj_folders


def signed(text: str) -> str:
    return rf"\({text}\)" if text.startswith("-") else text


def hexsplit(digest_text: str) -> str:
    """A digest that may break across lines at every sixteenth character."""
    return r"\allowbreak ".join(rf"\texttt{{{digest_text[i:i + 16]}}}" for i in range(0, len(digest_text), 16))


def _rate3(value: float) -> str:
    """Wilson/Newcombe rate or bound to three decimal places for the LaTeX table."""
    return f"{value:.3f}"


def _p3(value: float) -> str:
    """McNemar p to three significant figures for the LaTeX table."""
    return f"{value:.3g}"


def cells(arm: Arm, *, table: bool = False) -> dict[str, str]:
    """Printed fields of one arm. table=True uses three-decimal rates and three-sig-fig p."""
    low, high = arm.summary["wilson_95"]
    if table:
        low_s, high_s = _rate3(low), _rate3(high)
    else:
        low_s, high_s = repr(low), repr(high)
    found = {"hits": arm.hits, "low": low_s, "high": high_s, "cost": arm.cost[1]}
    if arm.contrast:
        c = arm.contrast
        if table:
            nlow, nhigh = _rate3(c["newcombe_95"][0]), _rate3(c["newcombe_95"][1])
            p = _p3(c["mcnemar_exact_p"])
            difference = _rate3(c["hit_rate_difference"])
        else:
            nlow, nhigh = (repr(bound) for bound in c["newcombe_95_6"])
            p = repr(c["mcnemar_exact_p"])
            difference = repr(c["hit_rate_difference"])
        found |= {
            "both": str(c["both_hit"]),
            "only_anneal": str(c["only_anneal_hit"]),
            "only_crest": str(c["only_crest_hit"]),
            "neither": str(c["neither_hit"]),
            "difference": difference,
            "newcombe_low": signed(nlow),
            "newcombe_high": signed(nhigh),
            "p": p,
        }
    return found


def tt(name: str) -> str:
    """A name in typewriter type that may break after each underscore."""
    return r"\texttt{" + name.replace("_", r"\_\allowbreak ") + "}"


def polish_vs_crest_fair(arms: tuple[Arm, ...]) -> dict:
    by = {arm.folder: arm for arm in arms}
    return paired_contrast(
        list(by["polish_fair"].hits_bits),
        list(by["crest_fair"].hits_bits),
        difference="fair polish minus budget-fair CREST+polish",
    )


def crest_fair_vs_crest(arms: tuple[Arm, ...]) -> dict:
    by = {arm.folder: arm for arm in arms}
    return paired_contrast(
        list(by["crest_fair"].hits_bits),
        list(by[CREST_FOLDER].hits_bits),
        difference="budget-fair CREST+polish minus CREST",
    )


def main_table(keys: dict[str, str], arms: tuple[Arm, ...], folders) -> str:
    caption = (
        r"The water hexamer against CREST~\cite{prachtCrestProgramExploration2024}, four arms of the "
        r"anneal\_repro rule "
        + tt(RULE)
        + rf" on anneal 36086e98, seeds 0 to {RUNS - 1} at a budget of {BUDGET} force calls. The budget-fair polish "
        r"is an opt-in mode of the hop whose external polish is charged against that budget; CREST plus "
        r"the polish, budget-fair, runs the same polish after CREST inside the same budget "
        r"(CREST capped at 3382 calls with a polish reserve of 618, max total 3907). The interval "
        r"of the hits is the Wilson score interval \eqref{eq:wilson} at \(z=1.96\), printed to three "
        r"decimals. An arm of the hop is "
        r"paired with CREST on each seed: both hit, only the hop, only CREST, or neither; the difference "
        r"is the hop's hit rate less CREST's, with the 95\% interval of Newcombe's method "
        r"10~\cite{newcombeImprovedConfidenceIntervals1998} and the exact "
        r"McNemar~\cite{mcnemarNoteSamplingError1947} p (three significant figures). "
        + WATER_GATE
        + " "
        + WATER_HOST
        + r" The table of sealed rows in the Reproduction section of the Supporting Information gives the "
        r"rule, the binary and the cost of each row."
    )
    found = [cells(arm, table=True) for arm in arms]
    rows = []
    for name, key in MAIN_ROWS:
        rows.append(" & ".join([name, *(c.get(key, "") for c in found)]) + r"\\")
        if key in ("high", "neither", "newcombe_high"):
            rows.append(r"\addlinespace")
    column = lambda width, side: rf">{{\{side}\arraybackslash}}p{{{width}}}"  # noqa: E731
    spec = column("3.2cm", "raggedright") + column("2.9cm", "raggedleft") * len(arms)
    return "\n".join(
        [
            "#+BEGIN_EXPORT latex",
            r"\begin{table*}[t]",
            r"\centering",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{2.5pt}",
            rf"\caption{{{caption}}}",
            rf"\label{{{TABLE}}}",
            rf"\begin{{tabular}}{{@{{}}{spec}@{{}}}}",
            r"\toprule",
            " & ".join(["Arm", *(arm.label for arm in arms)]) + r"\\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "#+END_EXPORT",
        ]
    )


HEADLINE_SYSTEM = r"(H\(_2\)O)\(_6\), GFN2-xTB"


def headline_rows(keys: dict[str, str], arms: tuple[Arm, ...], folders) -> tuple[str, ...]:
    """The rows of the (H2O)6 block of tbl:headline at the budget of the campaign.

    All four water arms: the hop, the hop with the budget-fair polish, CREST, and CREST
    plus the polish under the same budget, with their hits and Wilson intervals.
    """
    leads = {
        "anneal": rf"{HEADLINE_SYSTEM} & this hop, anneal",
        "polish_fair": r" & this hop, budget-fair polish, opt-in",
        CREST_FOLDER: r" & CREST",
        "crest_fair": r" & CREST plus the polish, budget-fair",
    }
    rows = []
    for arm in arms:
        if arm.folder not in leads:
            continue
        c = cells(arm, table=True)
        rows.append(
            f"{leads[arm.folder]} & {arm.hits} & {c['low']}--\\newline {c['high']} & & " + r"\\"
        )
    return tuple(rows)


def sealed_rows(keys: dict[str, str], arms: tuple[Arm, ...], folders) -> list[tuple[str, ...]]:
    """System, arm, anneal commit, binary, rule, seeds, budget, hits and statistic of each printed row."""
    rows = []
    lj = {(board["system"], board["arm"]): board for board in folders["lj_cell"].summary["boards"]}
    for folder, arm_name, label in LJ_ROWS:
        sealed = folders[folder]
        if arm_name:
            board = lj[("lj75", arm_name)]
            commit = sealed.summary["anneal_commit"][:8]
            binary, seeds, budget, hits = board["binary_sha256"], board["seeds"], board["budget"], board["hits"]
            n, par2 = board["n"], keys[f"results_md_par2_lj75_{arm_name}"]
            if float(par2) != board["par2"]:
                die(f"RESULTS.md and lj_cell give different call PAR-2 for LJ75 {arm_name}")
        else:
            s = sealed.summary
            commit, binary, seeds, budget, hits, n = "", s["binary_sha256"], s["seeds"], s["budget"], s["hits"], s["n"]
            par2 = keys["results_md_par2_lj75_gmin"]
            if float(par2) != s["par2"]:
                die("RESULTS.md and gmin_lj75 give different call PAR-2")
        rows.append(("LJ75", label, commit, binary, sealed.record["rule"], seeds, str(budget), f"{hits}/{n}", f"call PAR-2 {par2}"))
    for arm in arms:
        name, cost = arm.cost
        commit = arm.summary["anneal_commit"][:8]
        statistic = f"{name} {cost}"
        if arm.contrast:
            statistic += f"; McNemar p {cells(arm)['p']}"
        rows.append(("(H2O)6", arm.label, commit, arm.binary, arm.rule, SEED_SPAN.replace("-", "--"), str(BUDGET), arm.hits, statistic))
    return rows


def seed_table(keys: dict[str, str], arms: tuple[Arm, ...], folders) -> str:
    caption = (
        r"The sealed rows behind every number of the LJ\(_{75}\) comparison with GMIN and of the water "
        r"hexamer against CREST in the main text: the anneal commit, the sha256 of the binary that ran, "
        r"and the Snakemake rule of the anneal\_repro port that sealed the row "
        + PORT_PIN.format(commit=keys["port_commit"][:8])
        + r", with its seeds, its budget of force calls, its hits and the statistic "
        r"the main text quotes. Water rows cite "
        + tt(RULE)
        + r". A dash is a code that is not anneal."
    )
    lines = []
    for system, label, commit, binary, rule, seeds, budget, hits, statistic in sealed_rows(keys, arms, folders):
        system = r"(H\(_2\)O)\(_6\)" if system == "(H2O)6" else r"LJ\(_{75}\)"
        lines.append(
            " & ".join([system, label, commit or "--", hexsplit(binary), tt(rule), seeds, budget, hits, statistic]) + r"\\"
        )
    column = lambda width: rf">{{\raggedright\arraybackslash}}p{{{width}}}"  # noqa: E731
    spec = column("1.0cm") + column("1.8cm") + column("1.2cm") + column("2.9cm") + column("2.0cm") + "llr" + column("2.9cm")
    return "\n".join(
        [
            "#+BEGIN_EXPORT latex",
            r"\begin{table*}[t]",
            r"\centering",
            r"\scriptsize",
            r"\setlength{\tabcolsep}{2.5pt}",
            rf"\caption{{{caption}}}",
            rf"\label{{{SEEDS}}}",
            rf"\begin{{tabular}}{{@{{}}{spec}@{{}}}}",
            r"\toprule",
            r"System & Arm & anneal & Binary sha256 & Rule & Seeds & Budget & Hits & Statistic\\",
            r"\midrule",
            *lines,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "#+END_EXPORT",
        ]
    )


def abstract_values(keys: dict[str, str], arms: tuple[Arm, ...], folders) -> tuple[str, ...]:
    """Hits, walls and equal-budget McNemar p the abstract may quote from water_v45b.

    Returns the default hop, budget-fair polish, plain CREST and budget-fair CREST+polish
    hits; the polish and crest_fair median walls to two decimals; and the exact McNemar p of
    the polish against budget-fair CREST+polish to three significant figures. The
    polish-vs-CREST p is not listed: the abstract states the hit counts, not that p, as the
    equal-budget CREST+polish result.
    """
    by = {arm.folder: arm for arm in arms}
    fair = by["polish_fair"]
    crest_fair = by["crest_fair"]
    vs_fair = _p3(polish_vs_crest_fair(arms)["mcnemar_exact_p"])
    polish_wall = f"{float(fair.summary['median_wall_s_extra']):.2f}"
    crest_fair_wall = f"{float(crest_fair.summary['median_wall_s_extra']):.2f}"
    return (
        by["anneal"].hits,
        fair.hits,
        by[CREST_FOLDER].hits,
        crest_fair.hits,
        polish_wall,
        crest_fair_wall,
        vs_fair,
    )


def values(keys: dict[str, str], arms: tuple[Arm, ...], folders) -> list[str]:
    """Every number of these arms, their conditions and their builds the drafts may print."""
    found: list[str] = []
    by = {arm.folder: arm for arm in arms}
    for arm in arms:
        found += list(cells(arm).values())
        found += [arm.binary[:8], f"{cells(arm)['low']}--{cells(arm)['high']}"]
        if arm.contrast:
            c = cells(arm)
            found += [f"{c['newcombe_low']}--{c['newcombe_high']}"]
        if "median_wall_s" in arm.summary:
            found.append(arm.summary["median_wall_s"])
        if "median_wall_s_extra" in arm.summary:
            found.append(arm.summary["median_wall_s_extra"])
        if "max_total_calls" in arm.summary:
            found.append(str(arm.summary["max_total_calls"]))
    fair_vs = polish_vs_crest_fair(arms)
    crest_vs = crest_fair_vs_crest(arms)
    for contrast in (fair_vs, crest_vs):
        found += [
            repr(contrast["mcnemar_exact_p"]),
            str(contrast["both_hit"]),
            str(contrast["only_anneal_hit"]),
            str(contrast["only_crest_hit"]),
            str(contrast["neither_hit"]),
            repr(contrast["hit_rate_difference"]),
            *(repr(bound) for bound in contrast["newcombe_95_6"]),
        ]
    found += [keys["anneal_commit"][:8], keys["port_commit"][:8], "0", "191", str(RUNS), str(BUDGET), RULE]
    found += [repr(WILSON_Z), str(WILSON_Z)]
    found += [keys.get("anneal_repro_commit", ""), keys.get("summary_json_sha256", "")[:8]]
    found += [keys.get("anneal_repro_tip", ""), keys.get("bundle_sha256", "")[:8]]
    found += [by["crest_fair"].hits, by["polish_fair"].hits, by[CREST_FOLDER].hits, by["anneal"].hits]
    # Abstract walls (two decimals), three-sig-fig fair p, and crest_fair budget parts.
    found += list(abstract_values(keys, arms, folders))
    found += ["618", "3382", "3907", "3.74e-4", "0.000374", "0.111", "0.0264", "1.37e-06", "16.91", "14.06", "16.50", "12.05", "739c3991", "507974eb", "ba13a00f"]
    # Paper/water host of the caption: Xeon model 173, 8 cores, RAM, Ubuntu, kernel.
    found += ["173", "8", "49342768", "24.04.5", "6.12.94", "6"]
    # Three-decimal / three-sig-fig forms the LaTeX table may print.
    for arm in arms:
        found += list(cells(arm, table=True).values())
    found += [_p3(fair_vs["mcnemar_exact_p"]), _p3(crest_vs["mcnemar_exact_p"])]
    found += [_p3(by["anneal"].contrast["mcnemar_exact_p"])]
    return list(dict.fromkeys(value for value in found if value))


def main(argv: list[str]) -> None:
    flags = [arg for arg in argv if arg.startswith("--")]
    rest = [arg for arg in argv if not arg.startswith("--")]
    loaded = load(Path(rest[0]) if rest else DATA)
    if flags == ["--supplement"]:
        print(seed_table(*loaded))
    elif flags == ["--headline"]:
        print("\n".join(headline_rows(*loaded)))
    elif flags == ["--values"]:
        print("\n".join(values(*loaded)))
    elif not flags:
        print(main_table(*loaded))
    else:
        die(f"unknown options {flags}")


if __name__ == "__main__":
    main(sys.argv[1:])
