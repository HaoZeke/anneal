# Copyright (c) anneal developers
"""Tables for the native-unit ASE head-to-head, read from the recorded rows."""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from mapping import (
    ARMS,
    ASE_ARMS,
    BASE_SHA,
    ASE_RELAX_STEPS,
    GTOL,
    HIT_TOL,
    OLD_BH_DR,
    OLD_BH_FMAX,
    OLD_BH_OPTIMIZER,
    OLD_BH_TEMPERATURE_OVER_KB,
    OLD_MAXSTEP,
    OLD_MH_EDIFF,
    OLD_MH_FMAX,
    OLD_MH_MDMIN,
    OLD_MH_MINIMA_THRESHOLD,
    OLD_MH_OPTIMIZER,
    OLD_MH_T0_KELVIN,
    OLD_MH_TIMESTEP_FS,
    SYSTEMS,
    fmax_from_infinity,
    fisher_exact_two_sided,
    mann_whitney_exact,
    median,
)

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "bench" / "ase-native"
ROWS = DATA / "rows.csv"
SESSIONS = DATA / "sessions.csv"


def load_rows() -> list[dict[str, str]]:
    with ROWS.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_sessions() -> list[dict[str, str]]:
    if not SESSIONS.is_file():
        return []
    with SESSIONS.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _float(text: str) -> float:
    if text == "" or text is None:
        return float("nan")
    return float(text)


def _censored_evals(row: dict[str, str]) -> float:
    """Evaluations to the hit, or the paper budget when the run missed."""
    if row.get("hit") == "1" and row.get("evals_to_hit"):
        return float(row["evals_to_hit"])
    return float(row["budget"])


def _fmt(value: float) -> str:
    if not math.isfinite(value):
        return "inf"
    return f"{value:.6g}"


def old_versus_new() -> list[str]:
    fmax = fmax_from_infinity()
    return [
        "What changed relative to the previous ASE panel",
        "",
        "The previous runs (scripts/ase_panel and scripts/ase_nonlj) evaluated",
        "reduced-unit potentials with ASE 3.29 constructor defaults and stopped",
        "ASE, and the recommended anneal preset, on a wall-clock cap. Basin",
        f"hopping used {OLD_BH_OPTIMIZER}, fmax {OLD_BH_FMAX}, dr {OLD_BH_DR}, and",
        f"temperature {OLD_BH_TEMPERATURE_OVER_KB:g} k_B (about 0.008617 eV).",
        f"Minima hopping used {OLD_MH_OPTIMIZER}, fmax {OLD_MH_FMAX}, T0 {OLD_MH_T0_KELVIN:g} K,",
        f"Ediff0 {OLD_MH_EDIFF} eV, timestep {OLD_MH_TIMESTEP_FS} fs, mdmin {OLD_MH_MDMIN},",
        f"and a minima threshold of {OLD_MH_MINIMA_THRESHOLD} angstrom. The local",
        f"step cap stayed at the optimiser default of {OLD_MAXSTEP} angstrom.",
        "The wall cap was three times the slowest anneal seed of that system.",
        "",
        "This run keeps the same systems, the same paper evaluation budgets,",
        "the same hit targets, and the anneal for_cluster preset. It drops the",
        "wall cap. Potentials are parameterised in eV and angstroms (Lennard-Jones",
        "epsilon = 1 eV, sigma = 1 angstrom; Morse epsilon = 1 eV, r0 = 1 angstrom;",
        "Gupta gold already in eV and angstroms). ASE's fmax is",
        f"sqrt(3) * {GTOL:.0e} = {fmax:.6e} eV/angstrom, the Euclidean per-atom",
        "norm that meets anneal's infinity-norm quench tolerance on the vector",
        "(a, a, a). Basin-hopping temperature and step are 0.8 * energy_scale",
        "and 0.38 * length_scale. Minima-hopping temperature and Ediff0 use that",
        "same energy, the timestep is the preset 0.005 time unit at mass 1 u,",
        "and the local optimiser is L-BFGS with 8 pairs, Armijo 1e-4, curvature",
        "0.9, and a per-atom cap of min(1, length_scale). Anneal stops a quench",
        "after 200 WarmLbfgs iterations. Those iterations are not ASE's, so an",
        f"ASE quench stops on fmax or after {ASE_RELAX_STEPS} ASE iterations, whichever is",
        "first. Basin hopping restores the centre of mass after its kick;",
        "anneal's all-points move does not.",
        "A hit is a recorded quench at or below the reference plus",
        f"{HIT_TOL:.0e}. A miss is censored at the paper budget.",
        "",
    ]


def render() -> str:
    rows = load_rows()
    sessions = load_sessions()
    lines = [
        "Native-unit ASE head-to-head",
        "",
        *old_versus_new(),
    ]
    if sessions:
        done = [row for row in sessions if row.get("event") == "end"]
        source = done[-1] if done else sessions[-1]
        lines.extend(
            [
                "Recorded stack",
                "",
                f"ASE {source.get('ase_version', '')}, numpy {source.get('numpy_version', '')}.",
                f"Anneal base {source.get('anneal_sha', '')}.",
                f"Anneal tree {source.get('anneal_tree', '')}.",
                f"Core {source.get('core', '')}. Machine {source.get('machine_id', '')}.",
                "",
            ]
        )
    lines.extend(
        [
            "Fisher is the two-sided exact test on the hit table.",
            "Mann-Whitney is the exact two-sided permutation test on midranks.",
            "The energy comparison uses every run. The evaluation comparison",
            "uses evaluations consumed: evaluations to the hit when the run",
            "hit, and the paper budget when it did not.",
            "Medians put a non-finite best energy after every finite one and,",
            "for an even count, average the two central values.",
            "",
        ]
    )
    for system, spec in SYSTEMS.items():
        grouped = {arm: [] for arm in ARMS}
        for row in rows:
            if row["system"] == system and row["arm"] in grouped:
                grouped[row["arm"]].append(row)
        lines.append(f"## {system}")
        lines.append("")
        lines.append(
            f"Budget {spec['budget']}. Reference {spec['reference']}. "
            f"Tolerance {HIT_TOL:.0e}."
        )
        lines.append("")
        lines.append(
            "| arm | hits | n | median best | median evals | "
            "Fisher p | MW p energy | MW p evals | median wall s |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        anneal = grouped["anneal"]
        for arm in ARMS:
            sample = grouped[arm]
            n = len(sample)
            hits = sum(int(row["hit"]) for row in sample)
            bests = [_float(row["best_energy"]) for row in sample]
            evals = [_censored_evals(row) for row in sample]
            walls = [_float(row["wall_s"]) for row in sample]
            if arm == "anneal" or not anneal or not sample:
                fisher = ""
                energy_p = ""
                eval_p = ""
            else:
                fisher = _fmt(
                    fisher_exact_two_sided(
                        sum(int(row["hit"]) for row in anneal),
                        len(anneal) - sum(int(row["hit"]) for row in anneal),
                        hits,
                        n - hits,
                    )
                )
                energy_p = _fmt(mann_whitney_exact(bests, [_float(row["best_energy"]) for row in anneal])[0])
                eval_p = _fmt(
                    mann_whitney_exact(evals, [_censored_evals(row) for row in anneal])[0]
                )
            lines.append(
                f"| {arm} | {hits} | {n} | {_fmt(median(bests)) if sample else ''} | "
                f"{_fmt(median(evals)) if sample else ''} | {fisher} | {energy_p} | "
                f"{eval_p} | {_fmt(median(walls)) if sample else ''} |"
            )
        lines.append("")
        missing = [arm for arm in ARMS if not grouped[arm]]
        if missing:
            lines.append(f"Missing arms: {', '.join(missing)}.")
            lines.append("")
        n_used = sorted({len(grouped[arm]) for arm in ARMS if grouped[arm]})
        if n_used and n_used != [12]:
            lines.append(
                f"Seed count for this system is {n_used[0]}. "
                "Twelve repetitions of this budget were not run."
            )
            lines.append("")
    lines.append(
        "Comparisons are against the anneal arm of the same system. "
        "ASE arms are " + ", ".join(ASE_ARMS) + "."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    destination = DATA / "report.md"
    text = render()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text)
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
