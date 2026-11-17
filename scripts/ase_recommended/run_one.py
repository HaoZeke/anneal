#!/usr/bin/env python3
# Copyright (c) anneal developers
"""Paper budgets for the recommended arm of the ASE panel.

The search is scripts/ase_panel/run_one.py. It writes code_revision and host.
Lennard-Jones uses Config::recommended. Au55 uses Config::recommended_with_scales.
"""

PAPER_BUDGETS = {
    38: 400_000,
    55: 1_000_000,
    98: 4_000_000,
    104: 4_000_000,
    150: 4_000_000,
    "au55": 1_000_000,
}
SEEDS = tuple(range(2000, 2012))
