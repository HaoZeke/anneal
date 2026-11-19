# Copyright (c) anneal developers
"""Unit map and convergence map for the native-unit ASE head-to-head.

Unit choice
-----------
The potentials are parameterised in ASE's native electronvolts and
angstroms. Reduced Lennard-Jones is epsilon = 1 eV and sigma = 1 angstrom,
with no cutoff and no shift::

    E_eV = epsilon * E_reduced
    r_A = sigma * r_reduced
    F_eV_per_A = (epsilon / sigma) * F_reduced

With those parameters the map is the identity, so the published reduced
energies stay the numbers the paper rows use. Morse uses epsilon = 1 eV
and r0 = 1 angstrom, again with the tail left untruncated. Gupta gold is
already the Cleri-Rosato potential in eV and angstroms, so its map is the
identity as well.

Anneal's preset multiplies geometric controls by ``length_scale`` and
energetic controls by ``energy_scale``. For Lennard-Jones both are 1.
For Morse, ``length_scale`` is the pair minimum divided by 2^(1/6),
which is 2^(-1/6), and ``energy_scale`` is 1. For gold they are the
pair-minimum probe in ``examples/ase_panel.rs``: the dimer separation
of lowest energy, divided by 2^(1/6), and the depth of that well.

Convergence
-----------
Anneal's local optimiser is WarmLbfgs, which stops on the infinity norm
of the Cartesian gradient::

    max |dE/dr_{i,alpha}| < gtol,    gtol = 1e-6

in the potential's native energy per length. ASE stops on ``fmax``, the
maximum over atoms of the Euclidean force norm::

    max_i ||F_i||_2 < fmax

On one atom, ||v||_2 <= sqrt(3) ||v||_inf, and the bound is sharp for
v = (a, a, a). The thresholds therefore meet on that vector when

    fmax = sqrt(3) * gtol

A structure is recorded, and can count as a hit, when every Cartesian
component is below the preset record tolerance

    record_gradient = 1e-3 * energy_scale / length_scale

which is also an infinity norm (``gradient_is_converged``). The local
optimiser still targets the tighter fmax.

The other local-optimiser settings that anneal exposes, and the ASE
controls they replace:

    memory = 8                         WarmLbfgs / rgmin Lbfgs pairs
    alpha = 1                          initial Hessian; ASE stores 1/alpha
                                       as the inverse-Hessian diagonal,
                                       and rgmin's inverse_curvature is 1
    c1 = 1e-4, c2 = 0.9                Armijo and Wolfe curvature
    relax_steps = 200                  anneal's WarmLbfgs iteration cap
    maxstep = min(1, length_scale)     per-atom Euclidean cap. ASE's
                                       LBFGS rejects a cap above 1 angstrom.
                                       Anneal's L-BFGS has no per-atom cap;
                                       the cap used here is the preset
                                       length, clipped to that ceiling.

Global search
-------------
Basin hopping displaces every Cartesian component by Uniform(-dr, dr)
and then restores the centre of mass, which is the ASE 3.29 move.
Anneal's all-points move uses the same uniform box and does not restore
the centre of mass. The half-width is

    dr = 0.38 * length_scale

The Metropolis temperature is an energy, kT in eV:

    kT = 0.8 * energy_scale

Minima hopping's initial temperature is that energy expressed in kelvin,

    T0 = kT / k_B

and its initial acceptance window is the same energy,

    Ediff0 = kT

The MD timestep is Config.md_escape_dt = 0.005 in the unit system where
the mass is 1 u and the coordinates and energies are the native angstroms
and electronvolts. ASE multiplies a femtosecond keyword by ``units.fs``,
so

    timestep_fs = 0.005 / units.fs

and every mass is set to 1 u. Two potential minima end an escape
(``mdmin = 2``), the same count as Config.md_escape_minima. Two quenched
geometries are treated as the same minimum inside

    minima_threshold = 0.7 * length_scale

which is the preset merge radius. ASE's comparison is a coordinate
distance; anneal's merge radius is a fingerprint distance. The length is
scaled; the metric is the one ASE implements.

Dimensionless minima-hopping feedback (beta1, beta2, beta3, alpha1,
alpha2) stays at the ASE 3.29 defaults. Those factors do not carry a unit.

Budgets are the paper evaluation caps. A miss is censored at that cap.
There is no wall-clock cap.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE_SHA = "5eb79cdcb0a986984dc7b487e5a3cc93224246f9"

# Preset coefficients. These are the values Config::with_scales and
# WarmLbfgs::default actually install; the formulas below are the map.
TEMPERATURE = 0.8
ALL_POINTS_STEP = 0.38
RECORD_GRADIENT_COEFFICIENT = 1.0e-3
GTOL = 1.0e-6
RELAX_STEPS = 200
# WarmLbfgs iterations are not ASE LBFGS iterations. On seed 2000, ASE
# reached fmax in 290 (lj38), 831 (lj98), 1418 (lj150), 553 (m38r14),
# 710 (m55r14) and 267 (au55) iterations, and after 200 iterations the
# LJ150 infinity norm was still above the record tolerance. The ASE
# quench stops on fmax, with this ceiling so a quench that can meet
# fmax is not cut off first. The ceiling sits above those counts.
ASE_RELAX_STEPS = 4000
LBFGS_MEMORY = 8
ARMIJO = 1.0e-4
WOLFE_CURVATURE = 0.9
INVERSE_CURVATURE = 1.0
ASE_MAXSTEP_CEILING = 1.0
MD_TIMESTEP = 0.005
MD_MINIMA = 2
MERGE_RADIUS = 0.7
HIT_TOL = 1.0e-3
MIN_SEP = 0.85
DENSITY = 0.7
N_SEEDS = 12
SEED0 = 2000

# ASE 3.29 constructor defaults the previous panel left in place.
OLD_BH_TEMPERATURE_OVER_KB = 100.0
OLD_BH_DR = 0.1
OLD_BH_FMAX = 0.1
OLD_BH_OPTIMIZER = "FIRE"
OLD_MH_T0_KELVIN = 1000.0
OLD_MH_EDIFF = 0.5
OLD_MH_TIMESTEP_FS = 1.0
OLD_MH_FMAX = 0.05
OLD_MH_OPTIMIZER = "QuasiNewton"
OLD_MH_MDMIN = 2
OLD_MH_MINIMA_THRESHOLD = 0.5
OLD_MAXSTEP = 0.2

SYSTEMS = {
    "lj38": {"n": 38, "kind": "lj", "budget": 400_000, "reference": -173.928427, "rho": 0.0},
    "lj55": {"n": 55, "kind": "lj", "budget": 1_000_000, "reference": -279.248470, "rho": 0.0},
    "lj98": {"n": 98, "kind": "lj", "budget": 4_000_000, "reference": -543.665361, "rho": 0.0},
    "lj104": {"n": 104, "kind": "lj", "budget": 4_000_000, "reference": -582.086642, "rho": 0.0},
    "lj150": {"n": 150, "kind": "lj", "budget": 4_000_000, "reference": -893.310258, "rho": 0.0},
    "au55": {"n": 55, "kind": "gupta", "budget": 1_000_000, "reference": -191.03085280, "rho": 0.0},
    "m38r6": {"n": 38, "kind": "morse", "budget": 400_000, "reference": -157.477108, "rho": 6.0},
    "m38r14": {"n": 38, "kind": "morse", "budget": 400_000, "reference": -144.321054, "rho": 14.0},
    "m55r6": {"n": 55, "kind": "morse", "budget": 1_000_000, "reference": -250.286609, "rho": 6.0},
    "m55r14": {"n": 55, "kind": "morse", "budget": 1_000_000, "reference": -220.646208, "rho": 14.0},
}

PANEL_SYSTEMS = ("lj38", "lj55", "lj98", "lj104", "lj150", "au55")
NONLJ_SYSTEMS = ("m38r6", "m38r14", "m55r6", "m55r14")
ARMS = ("anneal", "ase_bh", "ase_mh")
ASE_ARMS = ("ase_bh", "ase_mh")

# Twelve seeds. A core-0 timing probe, before any recorded row, measured
# about 2.5e3 minima-hopping evaluations per second at N=150 and about
# 7e3 basin-hopping evaluations per second. A 4e6 budget is then roughly
# half an hour per minima-hopping seed. That is slow and still runnable,
# so no system is reduced to the floor of six.
SEEDS_PER_SYSTEM = {name: N_SEEDS for name in SYSTEMS}


def morse_length_scale() -> float:
    """Pair minimum of Morse (r0 = 1) divided by 2^(1/6)."""
    return 2.0 ** (-1.0 / 6.0)


def fmax_from_infinity(gtol: float = GTOL) -> float:
    """Euclidean per-atom threshold that meets the infinity norm at (a, a, a)."""
    return math.sqrt(3.0) * gtol


def record_gradient(energy_scale: float, length_scale: float) -> float:
    return RECORD_GRADIENT_COEFFICIENT * energy_scale / length_scale


def metropolis_kT(energy_scale: float) -> float:
    return TEMPERATURE * energy_scale


def basin_dr(length_scale: float) -> float:
    return ALL_POINTS_STEP * length_scale


def initial_hessian_alpha() -> float:
    """ASE's alpha is the Hessian guess. The inverse diagonal is 1/alpha."""
    return 1.0 / INVERSE_CURVATURE


def local_maxstep(length_scale: float) -> float:
    return min(ASE_MAXSTEP_CEILING, length_scale)


def minima_temperature_kelvin(energy_scale: float, k_b: float) -> float:
    return metropolis_kT(energy_scale) / k_b


def minima_ediff(energy_scale: float) -> float:
    return metropolis_kT(energy_scale)


def minima_timestep_fs(fs_per_time_unit: float) -> float:
    """Femtosecond keyword whose product with ``units.fs`` is the preset dt."""
    return MD_TIMESTEP / fs_per_time_unit


def minima_threshold(length_scale: float) -> float:
    return MERGE_RADIUS * length_scale


def lj_map(epsilon: float = 1.0, sigma: float = 1.0) -> tuple[float, float, float]:
    """Return (energy factor, length factor, force factor) from reduced units."""
    return epsilon, sigma, epsilon / sigma


def energy_hits(energy: float, target: float, tol: float = HIT_TOL) -> bool:
    if not (math.isfinite(energy) and math.isfinite(target) and math.isfinite(tol)):
        return False
    return tol >= 0.0 and energy <= target + tol


def seeds_for(system: str) -> tuple[int, ...]:
    n = SEEDS_PER_SYSTEM[system]
    if n < 6 or n > N_SEEDS:
        raise RuntimeError(f"{system} seed count {n} is outside 6..{N_SEEDS}")
    return tuple(range(SEED0, SEED0 + n))


def schedule(systems: tuple[str, ...] | None = None) -> list[dict[str, int | str]]:
    """Interleave arms inside each system and rotate the arm order by seed.

    Consecutive slots of one system cycle through the arms. The rotation
    by seed index makes each arm equally often the one in a given
    ``slot % len(arms)`` residue.
    """
    chosen = tuple(SYSTEMS) if systems is None else systems
    jobs: list[dict[str, int | str]] = []
    slot = 0
    n_arms = len(ARMS)
    for system in chosen:
        for index, seed in enumerate(seeds_for(system)):
            start = index % n_arms
            order = ARMS[start:] + ARMS[:start]
            for arm in order:
                jobs.append({"slot": slot, "system": system, "arm": arm, "seed": seed})
                slot += 1
    return jobs


def balance_counts(jobs: list[dict[str, int | str]]) -> dict[tuple[int, str], int]:
    counts: dict[tuple[int, str], int] = {}
    n_arms = len(ARMS)
    for job in jobs:
        key = (int(job["slot"]) % n_arms, str(job["arm"]))
        counts[key] = counts.get(key, 0) + 1
    return counts


def balance_ok(jobs: list[dict[str, int | str]]) -> bool:
    """Each system gives every arm the same number of seeds, in every slot residue."""
    n_arms = len(ARMS)
    grouped: dict[str, list[dict[str, int | str]]] = {}
    for job in jobs:
        grouped.setdefault(str(job["system"]), []).append(job)
    if not grouped:
        return False
    for system, rows in grouped.items():
        expected = len(seeds_for(system))
        if expected % n_arms != 0 or len(rows) != expected * n_arms:
            return False
        arm_counts = {arm: 0 for arm in ARMS}
        residue_counts: dict[tuple[int, str], int] = {}
        for local, job in enumerate(rows):
            arm = str(job["arm"])
            if arm not in arm_counts:
                return False
            arm_counts[arm] += 1
            key = (local % n_arms, arm)
            residue_counts[key] = residue_counts.get(key, 0) + 1
        if any(count != expected for count in arm_counts.values()):
            return False
        for arm in ARMS:
            for residue in range(n_arms):
                if residue_counts.get((residue, arm), 0) != expected // n_arms:
                    return False
    return True


def _assignment(path: Path, name: str) -> ast.AST:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return node.value
    raise RuntimeError(f"{path} has no {name}")


def historical_budgets() -> dict[str, int]:
    """Paper budgets copied out of the existing panel scripts."""
    panel = ast.literal_eval(ast.unparse(_assignment(ROOT / "scripts/ase_panel/run_one.py", "SYSTEMS")))
    nonlj = ast.literal_eval(ast.unparse(_assignment(ROOT / "scripts/ase_nonlj/run_one.py", "SYSTEMS")))
    found = {name: int(spec["budget"]) for name, spec in panel.items()}
    found.update({name: int(spec["budget"]) for name, spec in nonlj.items()})
    return found


def historical_arms() -> tuple[tuple[str, ...], tuple[str, ...]]:
    panel = ast.literal_eval(ast.unparse(_assignment(ROOT / "scripts/ase_panel/run_one.py", "ARMS")))
    nonlj = ast.literal_eval(ast.unparse(_assignment(ROOT / "scripts/ase_nonlj/run_one.py", "ARMS")))
    return tuple(panel), tuple(nonlj)


def fisher_exact_two_sided(hits_a: int, miss_a: int, hits_b: int, miss_b: int) -> float:
    """Two-sided Fisher exact p, summing tables no more likely than the observed one."""
    a, b, c, d = hits_a, miss_a, hits_b, miss_b
    n = a + b + c + d
    if n == 0:
        return 1.0
    row1 = a + b
    row2 = c + d
    col1 = a + c
    if row1 == 0 or row2 == 0 or col1 == 0 or (b + d) == 0:
        return 1.0

    def table_p(x: int) -> float:
        return math.comb(row1, x) * math.comb(row2, col1 - x) / math.comb(n, col1)

    observed = table_p(a)
    total = 0.0
    for x in range(max(0, col1 - row2), min(row1, col1) + 1):
        px = table_p(x)
        if px <= observed * (1.0 + 1.0e-12):
            total += px
    return min(1.0, total)


def mann_whitney_exact(left: list[float], right: list[float]) -> tuple[float, float]:
    """Exact two-sided Mann-Whitney p and the rank-sum U of ``left``.

    Ties use midranks. The p-value is the permutation probability of a
    rank sum at least as far from its null mean as the observed sum.
    """
    n1 = len(left)
    n2 = len(right)
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    pooled = [(float(value), index) for index, value in enumerate(list(left) + list(right))]
    pooled.sort(key=lambda item: item[0])
    ranks = [0.0] * (n1 + n2)
    rank_at = 0
    while rank_at < len(pooled):
        end = rank_at
        while end + 1 < len(pooled) and pooled[end + 1][0] == pooled[rank_at][0]:
            end += 1
        mid = 0.5 * ((rank_at + 1) + (end + 1))
        for offset in range(rank_at, end + 1):
            ranks[pooled[offset][1]] = mid
        rank_at = end + 1
    half = [int(round(rank * 2.0)) for rank in ranks]
    counts: list[dict[int, int]] = [{0: 1}] + [{} for _ in range(n1)]
    for item in half:
        for taken in range(n1, 0, -1):
            destination = counts[taken]
            for total, ways in counts[taken - 1].items():
                updated = total + item
                destination[updated] = destination.get(updated, 0) + ways
    ways = counts[n1]
    total_ways = sum(ways.values())
    observed = sum(half[:n1])
    centre = n1 * (n1 + n2 + 1)
    distance = abs(observed - centre)
    extreme = sum(weight for total, weight in ways.items() if abs(total - centre) >= distance)
    u = sum(ranks[:n1]) - n1 * (n1 + 1) / 2.0
    return extreme / total_ways, u


def median(values: list[float]) -> float:
    finite = sorted(value for value in values if math.isfinite(value))
    if not finite:
        return float("inf")
    # Non-finite energies sort as worse than every finite energy. They stay
    # in the sample: an infinite best is a run that never recorded a quench.
    sample = sorted(values, key=lambda value: (math.isinf(value), value))
    mid = len(sample) // 2
    if len(sample) % 2 == 1:
        return sample[mid]
    return 0.5 * (sample[mid - 1] + sample[mid])
