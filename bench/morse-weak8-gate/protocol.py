# Copyright (c) anneal developers
"""Shared protocol for the weak8 paired gate.

Eight cells, the recommended preset, and the paper budgets. Twelve seeds,
2000 through 2011. Three arms run in a balanced order-flipped sequence.
Statistics are two-sided Fisher exact tests on hits and tie-corrected
Mann-Whitney tests on calls to hit.
"""

from __future__ import annotations

import math

ARMS = ("A", "AP", "B")
ARM_LABEL = {"A": "A", "AP": "A'", "B": "B"}

# display, kind, atoms, rho, Cambridge minimum, paper budget.
# Budgets are the Lennard-Jones paper list used by PAPER_BUDGETS and by
# h2h_gmin_proof / h2h_morse: 4e5 up to 38 atoms, 1e6 below 75, 4e6 from 75.
CELLS = {
    "lj75": ("LJ75", "lj", 75, "na", -397.492331, 4_000_000),
    "lj98": ("LJ98", "lj", 98, "na", -543.665361, 4_000_000),
    "m75r6": ("M75 rho 6", "morse", 75, "6.0", -351.472365, 4_000_000),
    "m75r14": ("M75 rho 14", "morse", 75, "14.0", -318.407330, 4_000_000),
    "m80r6": ("M80 rho 6", "morse", 80, "6.0", -378.333471, 4_000_000),
    "m80r14": ("M80 rho 14", "morse", 80, "14.0", -340.811371, 4_000_000),
    "lj38": ("LJ38", "lj", 38, "na", -173.928427, 400_000),
    "m38r10": ("m38r10", "morse", 38, "10.0", -145.849817, 400_000),
}
CELL_ORDER = tuple(CELLS)

BASE_REVISION = "5eb79cdcb0a986984dc7b487e5a3cc93224246f9"
BASE_TREE = "7a2ba49b37241a8fe42b2c45c226374de6d92274"
B_TREE = "e45fa8e672477cad76454b61c36498ee027d7340"

THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
}
RUN_ENV = {**THREAD_ENV, "LJOS_SEAT": "benchbot"}
MIN_ROUNDS = 12
SEED0 = 2000
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20261119
MASK64 = (1 << 64) - 1
PINNED_CORE = 0

# Rotations of A, A', B, then each of those orders reversed. Twelve seeds
# repeat the six-order block twice, so every (slot, arm) pair occurs equally
# often.
_FORWARD = (
    ("A", "AP", "B"),
    ("AP", "B", "A"),
    ("B", "A", "AP"),
)

DRIVER_FIELDS = (
    "cell",
    "mode",
    "seed",
    "n",
    "kind",
    "rho",
    "budget",
    "reference",
    "replicas",
    "ledger",
    "hit",
    "evals_to_hit",
    "charged",
    "best",
    "best_bits",
    "wall_s",
    "pot_s",
    "outside_s",
    "cpu_ns",
    "wait_ns",
    "vol_ctxt",
    "invol_ctxt",
    "threads",
    "core_start",
    "core_end",
    "cpus_allowed",
    "cpus_allowed_list",
    "start_sha256",
    "config_sha256",
)
ROW_FIELDS = (
    "session",
    "sequence",
    "cell",
    "arm",
    "mode",
    "round",
    "slot",
    "order",
    "seed",
    "n",
    "kind",
    "rho",
    "budget",
    "reference",
    "replicas",
    "ledger",
    "hit",
    "evals_to_hit",
    "charged",
    "best",
    "best_bits",
    "wall_s",
    "pot_s",
    "outside_s",
    "proc_s",
    "cpu_ns",
    "wait_ns",
    "vol_ctxt",
    "invol_ctxt",
    "threads",
    "core_pinned",
    "core_busy_ticks",
    "core_steal_ticks",
    "core_start",
    "core_end",
    "cpus_allowed",
    "cpus_allowed_list",
    "start_sha256",
    "config_sha256",
    "source_revision",
    "source_tree",
    "binary_sha256",
    "host_id",
    "utc_start",
    "utc_end",
)
INT_FIELDS = (
    "sequence",
    "round",
    "slot",
    "seed",
    "n",
    "budget",
    "replicas",
    "hit",
    "evals_to_hit",
    "charged",
    "cpu_ns",
    "wait_ns",
    "vol_ctxt",
    "invol_ctxt",
    "threads",
    "core_pinned",
    "core_busy_ticks",
    "core_steal_ticks",
    "core_start",
    "core_end",
)
FLOAT_FIELDS = ("reference", "wall_s", "pot_s", "outside_s", "proc_s")
HOST_FIELDS = (
    "host_id",
    "cpu_model",
    "logical_cpus",
    "mem_total_kb",
    "arch",
    "kernel_release",
    "os_release",
)

# Code fact, checked against every recorded row. Config::recommended leaves
# the population search off (one replica; on the patched tree adaptive_search
# and stream_hold stay false and out of the resolved preset). The search forks
# a second stream only when both of those are on, and it quenches any replica
# id other than the incumbent only when more than one replica is configured.
# Each gate run constructs Ledger::new, so the boundary list is empty and the
# spent count is zero: the walk is not a resumed ledger. Every cell uses that
# preset, so every cell is a fresh-ledger default trajectory. The replica-id
# filter belongs to the adaptive preset, which this gate does not run.
FRESH_LEDGER_RULE = (
    "Config::recommended is one replica with the population search off, and "
    "each run starts from Ledger::new. All eight cells use that preset, so "
    "all eight are fresh-ledger default trajectories. Only an adaptive, "
    "replica-id filtered search is allowed to move."
)


class ProtocolError(Exception):
    """The declared protocol cannot continue."""


def paper_budget(n):
    if n <= 38:
        return 400_000
    if n < 75:
        return 1_000_000
    return 4_000_000


def arm_order(round_index):
    block = round_index % 6
    order = _FORWARD[block % 3]
    if block >= 3:
        return tuple(reversed(order))
    return order


def build_schedule(cells, rounds, seed0):
    entries = []
    sequence = 0
    for cell in cells:
        for round_index in range(rounds):
            order = arm_order(round_index)
            for slot, arm in enumerate(order):
                entries.append(
                    {
                        "sequence": sequence,
                        "cell": cell,
                        "round": round_index,
                        "slot": slot,
                        "arm": arm,
                        "seed": seed0 + round_index,
                        "order": ">".join(order),
                    }
                )
                sequence += 1
    return entries


def type7(sorted_values, p):
    if not sorted_values:
        raise ValueError("a quantile needs a sample")
    h = (len(sorted_values) - 1) * p
    lo = math.floor(h)
    hi = math.ceil(h)
    return sorted_values[lo] + (h - lo) * (sorted_values[hi] - sorted_values[lo])


def median(values):
    return type7(sorted(values), 0.5)


def fisher_two_sided(hits_a, n_a, hits_b, n_b):
    hits = hits_a + hits_b

    def weight(x):
        return math.comb(n_a, x) * math.comb(n_b, hits - x)

    observed = weight(hits_a)
    mass = sum(
        weight(x)
        for x in range(max(0, hits - n_b), min(n_a, hits) + 1)
        if weight(x) <= observed
    )
    return mass / math.comb(n_a + n_b, hits)


def mann_whitney_two_sided(x, y):
    """Tie-corrected two-sided Mann-Whitney, normal approximation with continuity correction."""
    if not x or not y:
        return None
    n1, n2 = len(x), len(y)
    n = n1 + n2
    pooled = list(x) + list(y)
    order = sorted(range(n), key=lambda i: pooled[i])
    ranks = [0.0] * n
    tie_term = 0
    i = 0
    while i < n:
        j = i
        while j + 1 < n and pooled[order[j + 1]] == pooled[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        size = j - i + 1
        tie_term += size**3 - size
        i = j + 1
    u1 = sum(ranks[:n1]) - n1 * (n1 + 1) / 2
    u = max(u1, n1 * n2 - u1)
    variance = n1 * n2 / 12 * ((n + 1) - tie_term / (n * (n - 1)))
    if variance <= 0:
        return 1.0
    z = (u - n1 * n2 / 2 - 0.5) / math.sqrt(variance)
    return min(1.0, math.erfc(z / math.sqrt(2)))


class SplitMix64:
    def __init__(self, seed):
        self.state = seed & MASK64

    def next_u64(self):
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
        return z ^ (z >> 31)

    def below(self, bound):
        limit = (1 << 64) - ((1 << 64) % bound)
        while True:
            value = self.next_u64()
            if value < limit:
                return value % bound


def bootstrap_medians(values, draws, rng):
    if not values:
        return []
    n = len(values)
    result = []
    for _ in range(draws):
        result.append(median([values[rng.below(n)] for _ in range(n)]))
    result.sort()
    return result


def median_ci(values, seed):
    if not values:
        return None
    draws = bootstrap_medians(values, BOOTSTRAP_DRAWS, SplitMix64(seed))
    return {
        "median": median(values),
        "bootstrap_95_ci": [type7(draws, 0.025), type7(draws, 0.975)],
        "n": len(values),
    }


def aa_ratio_band(values_a, values_ap, seed):
    if not values_a or not values_ap:
        return None
    rng = SplitMix64(seed)
    ratios = []
    for _ in range(BOOTSTRAP_DRAWS):
        ma = median([values_a[rng.below(len(values_a))] for _ in values_a])
        map_ = median([values_ap[rng.below(len(values_ap))] for _ in values_ap])
        if ma > 0:
            ratios.append(map_ / ma)
    ratios.sort()
    return [type7(ratios, 0.025), type7(ratios, 0.975)]


def slot_arm_counts(rows):
    counts = {(slot, arm): 0 for slot in range(3) for arm in ARMS}
    for row in rows:
        key = (int(row["slot"]), row["arm"])
        counts[key] = counts.get(key, 0) + 1
    return counts


def selftest():
    ok = True

    def expect(name, condition):
        nonlocal ok
        print(("pass " if condition else "FAIL ") + name)
        ok = ok and condition

    def close(a, b, tol=1e-9):
        return abs(a - b) <= tol * max(1.0, abs(a), abs(b))

    expect("type7 median", median([1, 2, 3, 4, 5, 6]) == 3.5)
    expect("balanced Fisher", close(fisher_two_sided(5, 10, 5, 10), 1.0))
    expect(
        "complete-split Fisher",
        close(fisher_two_sided(0, 12, 12, 12), 2 / math.comb(24, 12)),
    )
    expect(
        "Mann-Whitney separated",
        mann_whitney_two_sided([1, 2, 3, 4], [5, 6, 7, 8]) < 0.05,
    )
    tied = mann_whitney_two_sided([1, 1, 2], [1, 2, 2])
    expect("Mann-Whitney ties", tied is not None and tied > 0.2)
    rng = SplitMix64(0)
    expect(
        "SplitMix64 stream",
        [rng.next_u64() for _ in range(3)]
        == [0xE220A8397B1DCDAF, 0x6E789E6AA1B965F4, 0x06C45D188009454F],
    )
    expect(
        "paper budgets",
        all(CELLS[name][5] == paper_budget(CELLS[name][2]) for name in CELLS)
        and paper_budget(38) == 400_000
        and paper_budget(75) == 4_000_000
        and paper_budget(80) == 4_000_000
        and paper_budget(98) == 4_000_000,
    )
    schedule = build_schedule(list(CELL_ORDER), 12, 2000)
    expect("schedule length", len(schedule) == 8 * 12 * 3)
    counts = {(slot, arm): 0 for slot in range(3) for arm in ARMS}
    orders_ok = True
    for entry in schedule:
        counts[(entry["slot"], entry["arm"])] += 1
        orders_ok = orders_ok and entry["order"] == ">".join(arm_order(entry["round"]))
    expect("orders match the flip block", orders_ok)
    expect("balanced slot arms", set(counts.values()) == {32})
    per_cell = {(slot, arm): 0 for slot in range(3) for arm in ARMS}
    for entry in schedule:
        if entry["cell"] == "lj75":
            per_cell[(entry["slot"], entry["arm"])] += 1
    expect("balanced within a cell", set(per_cell.values()) == {4})
    seeds = {entry["seed"] for entry in schedule if entry["cell"] == "lj38"}
    expect("seeds 2000-2011", seeds == set(range(2000, 2012)))
    same = True
    for round_index in range(12):
        group = [entry for entry in schedule if entry["round"] == round_index and entry["cell"] == "m80r14"]
        same = same and len({entry["seed"] for entry in group}) == 1
    expect("one seed per round", same)
    flipped = arm_order(0) == tuple(reversed(arm_order(3)))
    expect("second half reverses the first", flipped and arm_order(1) == tuple(reversed(arm_order(4))))
    expect("driver fields unique", len(set(DRIVER_FIELDS)) == len(DRIVER_FIELDS))
    expect("row fields cover driver", set(DRIVER_FIELDS) <= set(ROW_FIELDS))
    return 0 if ok else 1
