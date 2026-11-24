# Copyright (c) anneal developers
"""Preregistered cells for the PR 171 r-next2 gate."""

from __future__ import annotations

from dataclasses import dataclass

BASE = "364679f060386949f805a72dea9d7d63b7e9ea4f"
BASE_TREE = "79610c980d39f19a2d383ffe3c53e4505ddb9b6c"
PR_TREE = "c3a236ac7e28f07bb8ed29fd1436295ba44d1745"
SEEDS = tuple(range(2000, 2012))
WATER_SEEDS = tuple(range(2000, 2024))
BOOT_SEED = 1712026
BOOT_DRAWS = 10_000
TIME_CAP = 1.10
RSS_CAP = 1.10
HANG_S = 14_400
TARGET_CELL = "m38r14d"
MOLECULAR_CELL = "tip4p6"


@dataclass(frozen=True)
class Cell:
    name: str
    driver: str
    model: str
    n: int
    rho: float
    budget: int
    reference: float
    seeds: tuple[int, ...] = SEEDS


CELLS = (
    Cell("m38r14d", "h2h_seed", "morse", 38, 14.0, 400_000, -144.321054),
    Cell("lj75", "h2h_gmin_proof", "panel", 75, 0.0, 4_000_000, -397.492331),
    Cell("lj98", "h2h_gmin_proof", "panel", 98, 0.0, 4_000_000, -543.665361),
    Cell(
        "tip4p6",
        "h2h_seed",
        "tip4p",
        6,
        0.0,
        400_000,
        -197.7805260034,
        WATER_SEEDS,
    ),
)


def arm_order(seed: int) -> tuple[str, str, str, str]:
    """ABBA with the independent base build alternating by seed."""
    if seed % 2:
        return ("D2", "G2", "G", "D")
    return ("D", "G", "G2", "D2")


def stop_level(cell: Cell) -> float:
    return cell.reference + 0.001


def self_test() -> None:
    assert len(SEEDS) == 12
    assert len(WATER_SEEDS) == 24
    assert [len(cell.seeds) for cell in CELLS] == [12, 12, 12, 24]
    assert len({cell.name for cell in CELLS}) == len(CELLS)
    assert arm_order(2000) == ("D", "G", "G2", "D2")
    assert arm_order(2001) == ("D2", "G2", "G", "D")
    assert next(cell for cell in CELLS if cell.name == MOLECULAR_CELL).model == "tip4p"
    print("cells ok")


if __name__ == "__main__":
    self_test()
