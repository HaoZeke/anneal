# Copyright (c) anneal developers
"""Pre-registered cell list for the budget-stopped G2 ASE campaign.

The cap lives here. The Snakemake rule loads this module and rejects a
config value, so a command-line default cannot replace it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

CPU_CAP_S = 43_200
SYSTEMS = ("lj38", "lj55", "lj98", "lj104", "au55")
PAPER_BUDGETS = {
    "lj38": 400_000,
    "lj55": 1_000_000,
    "lj98": 4_000_000,
    "lj104": 4_000_000,
    "au55": 1_000_000,
}
SEEDS = tuple(range(2000, 2012))
ARMS = ("ase_bh", "ase_mh", "anneal")
ANNEAL_PRESET = "for_cluster"
CAPPED_SYSTEMS = frozenset({"lj98", "lj104", "au55"})
CAPPED_ARMS = frozenset({"ase_mh", "anneal"})
# Amendment A1, after preregistration 1308d867. These systems are not
# part of SYSTEMS. Budgets and references are the values already scored
# in this repository. lj75 comes from scripts/ase_lj75/run_one.py
# (PAPER_BUDGETS and REFERENCE), which scripts/ase_nonlj/cpu_cap_spec.py
# loads for the capped-shard harness. lj150 comes from
# scripts/ase_panel/run_one.py SYSTEMS. Both references are the Cambridge
# Cluster Database values listed in scripts/gmin_panel/run_panel.py.
AMENDMENT_ID = "A1"
AMENDMENT_SYSTEMS = ("lj75", "lj150")
AMENDMENT_ARM = "ase_mh"
AMENDMENT_BUDGETS = {
    "lj75": 4_000_000,
    "lj150": 4_000_000,
}
AMENDMENT_REFERENCES = {
    "lj75": -397.492331,
    "lj150": -893.310258,
}
AMENDMENT_ATOMS = {
    "lj75": 75,
    "lj150": 150,
}
CORES = 1
SERIAL = True
ONE_PER_VM = True
WALL_CAP_S = None
ROWS_NAME = "rows-ljfix-budget.csv"
SHAPES_NAME = "shapes-ljfix-budget.jsonl"
LOCKED_CONFIG_KEYS = (
    "cpu_cap_s",
    "cpu_cap",
    "cap_s",
    "g2_cpu_cap_s",
    "g2_budget_cpu_cap_s",
)


@dataclass(frozen=True)
class Cell:
    system: str
    arm: str
    seed: int
    budget: int
    stop_rule: str
    cpu_cap_s: int | None
    preset: str | None
    cores: int
    wall_cap_s: None


def stop_rule(system: str, arm: str) -> str:
    if system in AMENDMENT_SYSTEMS:
        if arm != AMENDMENT_ARM:
            raise ValueError(
                f"amendment {AMENDMENT_ID} accepts {system} only on {AMENDMENT_ARM}"
            )
        return "cap"
    if system not in PAPER_BUDGETS:
        raise ValueError(f"unknown system {system}")
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm}")
    if system in CAPPED_SYSTEMS and arm in CAPPED_ARMS:
        return "cap"
    return "budget"


def cpu_cap_s(system: str, arm: str) -> int | None:
    if stop_rule(system, arm) == "cap":
        return CPU_CAP_S
    return None


def cells() -> tuple[Cell, ...]:
    found = []
    for system in SYSTEMS:
        for arm in ARMS:
            rule = stop_rule(system, arm)
            for seed in SEEDS:
                found.append(Cell(
                    system,
                    arm,
                    seed,
                    PAPER_BUDGETS[system],
                    rule,
                    CPU_CAP_S if rule == "cap" else None,
                    ANNEAL_PRESET if arm == "anneal" else None,
                    CORES,
                    WALL_CAP_S,
                ))
    return tuple(found)


def amendment_cells() -> tuple[Cell, ...]:
    found = []
    for system in AMENDMENT_SYSTEMS:
        rule = stop_rule(system, AMENDMENT_ARM)
        for seed in SEEDS:
            found.append(Cell(
                system,
                AMENDMENT_ARM,
                seed,
                AMENDMENT_BUDGETS[system],
                rule,
                CPU_CAP_S if rule == "cap" else None,
                None,
                CORES,
                WALL_CAP_S,
            ))
    return tuple(found)


def find_cell(system: str, arm: str, seed: int) -> Cell:
    for cell in cells():
        if cell.system == system and cell.arm == arm and cell.seed == seed:
            return cell
    for cell in amendment_cells():
        if cell.system == system and cell.arm == arm and cell.seed == seed:
            return cell
    raise ValueError(f"unknown cell {system} {arm} {seed}")


def _arms_for(system: str) -> tuple[str, ...]:
    if system in AMENDMENT_SYSTEMS:
        return (AMENDMENT_ARM,)
    if system not in PAPER_BUDGETS:
        raise ValueError(f"unknown system {system}")
    return ARMS


def pairs(system: str) -> tuple[tuple[str, str], ...]:
    arms = _arms_for(system)
    found = []
    for index, left in enumerate(arms):
        for right in arms[index + 1:]:
            if stop_rule(system, left) == stop_rule(system, right):
                found.append((left, right))
    return tuple(found)


def assert_pair(system: str, left: str, right: str) -> None:
    allowed = pairs(system)
    if (left, right) not in allowed and (right, left) not in allowed:
        raise ValueError(
            f"{system} {left} is {stop_rule(system, left)} and "
            f"{right} is {stop_rule(system, right)}"
        )


def _require_system(system: str) -> None:
    if system not in PAPER_BUDGETS and system not in AMENDMENT_SYSTEMS:
        raise ValueError(f"unknown system {system}")


def rows_path(root: Path, system: str) -> Path:
    _require_system(system)
    return Path(root) / "bench" / f"ase-{system}" / ROWS_NAME


def shapes_path(root: Path, system: str) -> Path:
    _require_system(system)
    return Path(root) / "bench" / f"ase-{system}" / SHAPES_NAME


def is_scope1_rows(path: Path) -> bool:
    name = Path(path).name
    return name.startswith("rows-ljfix") and name.endswith(".csv") and name != ROWS_NAME


def reject_cap_override(config) -> None:
    locked = {key.lower() for key in LOCKED_CONFIG_KEYS}
    present = [str(key) for key in config if str(key).lower() in locked]
    if present:
        names = ", ".join(present)
        raise ValueError(
            f"CPU cap is the pre-registered constant CPU_CAP_S={CPU_CAP_S} "
            f"and cannot be set from the CLI ({names})"
        )


def cell_record(cell: Cell) -> dict:
    record = {
        "system": cell.system,
        "arm": cell.arm,
        "seed": cell.seed,
        "budget": cell.budget,
        "stop_rule": cell.stop_rule,
        "cpu_cap_s": cell.cpu_cap_s,
        "preset": cell.preset,
        "cores": cell.cores,
        "wall_cap_s": cell.wall_cap_s,
        "serial": SERIAL,
        "one_per_vm": ONE_PER_VM,
        "rows": f"bench/ase-{cell.system}/{ROWS_NAME}",
        "shapes": f"bench/ase-{cell.system}/{SHAPES_NAME}",
    }
    if cell.system in AMENDMENT_SYSTEMS:
        record["amendment"] = AMENDMENT_ID
    return record
