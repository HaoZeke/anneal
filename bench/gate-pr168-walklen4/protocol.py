# Copyright (c) anneal developers
"""LJ98 walklen4 gate, shard A. Constants and the ABBA schedule.

Arm A is develop 4feb0699. Arm B is the merge-tree of that commit and
ddd53f2f ("fix: quench past the quiet record"). The paper budget is
4_000_000 energy calls. A miss is scored as budget + 1. The hit level is
the known LJ98 minimum plus the repository hit tolerance.
"""

from __future__ import annotations

ARM_A_REV = "4feb0699d4e61c9531283e82196fcb60baf66932"
ARM_A_TREE = "788d36860fb10d5863299acef0d0538618267d6d"
ARM_B_TIP = "ddd53f2f60d53aebdf2e97f03d3ade07c2905738"
ARM_B_TREE = "2fb57f0db81b5c3d1ec78077881b2982b8b78a55"
ARM_B_PARENT = "f75836be832c0f953f99c26efb9cd5c4b87565b0"

REFERENCE = -543.665361
HIT_TOL = 1.0e-3
STOP = "-543.664361"
BUDGET = 4_000_000
MISS_CALLS = BUDGET + 1
SEEDS = tuple(range(2048, 2084))
CORE = 0
N_SEEDS = 36

ALPHA = 0.05
OUTSIDE_CAP = 1.10
BOOT_SEED = 20261123
BOOT_DRAWS = 10_000

assert len(SEEDS) == N_SEEDS
assert SEEDS[0] == 2048 and SEEDS[-1] == 2083
assert abs((REFERENCE + HIT_TOL) - float(STOP)) < 1e-12
assert MISS_CALLS == 4_000_001


def schedule(seeds: tuple[int, ...] = SEEDS) -> list[dict]:
    """ABBA: even seeds run A then B, odd seeds run B then A."""
    jobs = []
    for index, seed in enumerate(seeds):
        order = ("A", "B") if index % 2 == 0 else ("B", "A")
        for arm in order:
            jobs.append({"seq": len(jobs), "arm": arm, "seed": int(seed)})
    return jobs


def revision_of(arm: str) -> tuple[str, str, str]:
    """Return (code_revision, code_tree, patch_tip) for an arm."""
    if arm == "A":
        return ARM_A_REV, ARM_A_TREE, ""
    if arm == "B":
        return ARM_B_TIP, ARM_B_TREE, ARM_B_TIP
    raise ValueError(arm)
