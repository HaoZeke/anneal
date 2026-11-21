# Copyright (c) anneal developers
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from anneal.device import DeviceHistory, EnsembleHistory

__version__: str
DEFAULT_HIT_TOL: float
SEARCH_HIT_TOL: float
COPPER_Z: int
GOLD_Z: int
PLATINUM_Z: int

Preset = Boltzmann | Fast | Gsa

class BasinBias:
    def __init__(
        self,
        n_points: int,
        merge_radius: float = 0.01,
        w0: float = 0.25,
        gamma: float = 5.0,
    ) -> None: ...
    def potential(self, x: npt.NDArray[np.float64]) -> float: ...
    def deposit(self, x: npt.NDArray[np.float64], temp: float) -> None: ...
    @property
    def n_basins(self) -> int: ...
    @property
    def deepest(self) -> float: ...
    @property
    def missing_mass(self) -> float: ...

class Boltzmann:
    t_init: float
    sigma: float
    def __init__(self, t_init: float = 1.0, sigma: float = 0.5) -> None: ...

class Fast:
    t_init: float
    gamma: float
    def __init__(self, t_init: float = 1.0, gamma: float = 0.5) -> None: ...

class Gsa:
    t_init: float
    q_v: float
    q_a: float
    def __init__(
        self,
        t_init: float = 1.0,
        q_v: float = 2.62,
        q_a: float = 1.7,
    ) -> None: ...

class Bounds:
    def __init__(
        self,
        low: npt.NDArray[np.float64],
        high: npt.NDArray[np.float64],
        slack: float = 1e-6,
    ) -> None: ...
    @property
    def dims(self) -> int: ...
    @property
    def slack(self) -> float: ...
    def contains(self, x: npt.NDArray[np.float64]) -> bool: ...
    def clip(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...

class PyObjective:
    def __init__(
        self,
        fn_: Callable[[npt.NDArray[np.float64]], float],
        bounds: Bounds,
        grad_fn: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]] | None = None,
    ) -> None: ...
    @property
    def dim(self) -> int: ...
    @property
    def bounds(self) -> Bounds: ...
    def eval(self, x: npt.NDArray[np.float64]) -> float: ...
    def grad(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...

class PairPotential:
    def __init__(
        self,
        n_points: int,
        rho: float | None = None,
        extent: float | None = None,
    ) -> None: ...
    @staticmethod
    def lennard_jones(n_points: int) -> PairPotential: ...
    @staticmethod
    def morse(n_points: int, rho: float) -> PairPotential: ...
    @staticmethod
    def ccd_morse_minimum(n_points: int, rho: float) -> float | None: ...
    @property
    def n_points(self) -> int: ...
    @property
    def dim(self) -> int: ...
    @property
    def kind(self) -> str: ...
    @property
    def rho(self) -> float | None: ...
    @property
    def r_min(self) -> float: ...
    @property
    def length_scale(self) -> float: ...
    @property
    def second_neighbour_energy(self) -> float: ...
    @property
    def bounds(self) -> Bounds: ...
    def eval(self, x: npt.NDArray[np.float64]) -> float: ...
    def grad(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
    def value_and_gradient(
        self,
        x: npt.NDArray[np.float64],
    ) -> tuple[float, npt.NDArray[np.float64]]: ...
    def to_objective(self) -> PyObjective: ...
    def __call__(self, x: npt.NDArray[np.float64]) -> float: ...

class BinaryLennardJones:
    def __init__(
        self,
        n_a: int,
        n_b: int,
        epsilon_aa: float = 1.0,
        epsilon_ab: float = 1.0,
        epsilon_bb: float = 1.0,
        sigma_aa: float = 1.0,
        sigma_ab: float = 1.0,
        sigma_bb: float = 1.0,
    ) -> None: ...
    @staticmethod
    def doye_meyer(n_a: int, n_b: int, sigma_bb: float) -> BinaryLennardJones: ...
    @property
    def n_a(self) -> int: ...
    @property
    def n_b(self) -> int: ...
    @property
    def n_points(self) -> int: ...
    @property
    def dim(self) -> int: ...
    @property
    def epsilon_aa(self) -> float: ...
    @property
    def epsilon_ab(self) -> float: ...
    @property
    def epsilon_bb(self) -> float: ...
    @property
    def sigma_aa(self) -> float: ...
    @property
    def sigma_ab(self) -> float: ...
    @property
    def sigma_bb(self) -> float: ...
    @property
    def bounds(self) -> Bounds: ...
    def eval(self, x: npt.NDArray[np.float64]) -> float: ...
    def grad(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
    def value_and_gradient(
        self,
        x: npt.NDArray[np.float64],
    ) -> tuple[float, npt.NDArray[np.float64]]: ...
    def to_objective(self) -> PyObjective: ...
    def __call__(self, x: npt.NDArray[np.float64]) -> float: ...

class SuttonChen:
    def __init__(self, n_points: int, family: str) -> None: ...
    @staticmethod
    def sc_12_6(n_points: int) -> SuttonChen: ...
    @staticmethod
    def sc_9_6(n_points: int) -> SuttonChen: ...
    @staticmethod
    def sc_10_8(n_points: int) -> SuttonChen: ...
    @property
    def n_points(self) -> int: ...
    @property
    def dim(self) -> int: ...
    @property
    def family(self) -> str: ...
    @property
    def n_exp(self) -> int: ...
    @property
    def m_exp(self) -> int: ...
    @property
    def c(self) -> float: ...
    @property
    def basin_hopping_temperature(self) -> float: ...
    @property
    def bounds(self) -> Bounds: ...
    def eval(self, x: npt.NDArray[np.float64]) -> float: ...
    def grad(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
    def value_and_gradient(
        self,
        x: npt.NDArray[np.float64],
    ) -> tuple[float, npt.NDArray[np.float64]]: ...
    def to_objective(self) -> PyObjective: ...
    def __call__(self, x: npt.NDArray[np.float64]) -> float: ...

class Gupta:
    @staticmethod
    def zinc(n_points: int) -> Gupta: ...
    @staticmethod
    def platinum(n_points: int) -> Gupta: ...
    @staticmethod
    def copper_gold(n_copper: int, n_gold: int) -> Gupta: ...
    @staticmethod
    def platinum_copper(n_platinum: int, n_copper: int) -> Gupta: ...
    @staticmethod
    def elements(labels: Sequence[int]) -> Gupta: ...
    @staticmethod
    def copper_gold_species(n_copper: int, n_gold: int) -> list[int]: ...
    @staticmethod
    def platinum_copper_species(n_platinum: int, n_copper: int) -> list[int]: ...
    @staticmethod
    def homonuclear_r0(label: int) -> float | None: ...
    @property
    def n_points(self) -> int: ...
    @property
    def dim(self) -> int: ...
    @property
    def bounds(self) -> Bounds: ...
    def eval(self, x: npt.NDArray[np.float64]) -> float: ...
    def grad(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
    def value_and_gradient(
        self,
        x: npt.NDArray[np.float64],
    ) -> tuple[float, npt.NDArray[np.float64]]: ...
    def to_objective(self) -> PyObjective: ...
    def __call__(self, x: npt.NDArray[np.float64]) -> float: ...

class Tip4pCluster:
    def __init__(self, n_molecules: int) -> None: ...
    @staticmethod
    def ccd_minimum(n_molecules: int) -> float | None: ...
    @property
    def n_molecules(self) -> int: ...
    @property
    def dim(self) -> int: ...
    @property
    def bounds(self) -> Bounds: ...
    def molecule_sites(
        self,
        x: npt.NDArray[np.float64],
        index: int,
    ) -> npt.NDArray[np.float64]: ...
    def fold_rotations(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
    def eval(self, x: npt.NDArray[np.float64]) -> float: ...
    def grad(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
    def value_and_gradient(
        self,
        x: npt.NDArray[np.float64],
    ) -> tuple[float, npt.NDArray[np.float64]]: ...
    def to_objective(self) -> PyObjective: ...
    def __call__(self, x: npt.NDArray[np.float64]) -> float: ...

class GminTip4p:
    def __init__(self, n_molecules: int) -> None: ...
    @property
    def n_molecules(self) -> int: ...
    @property
    def dim(self) -> int: ...
    @property
    def bounds(self) -> Bounds: ...
    def eval(self, x: npt.NDArray[np.float64]) -> float: ...
    def grad(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
    def value_and_gradient(
        self,
        x: npt.NDArray[np.float64],
    ) -> tuple[float, npt.NDArray[np.float64]]: ...
    def to_objective(self) -> PyObjective: ...
    def __call__(self, x: npt.NDArray[np.float64]) -> float: ...

NativeClusterPotential = (
    PairPotential
    | BinaryLennardJones
    | SuttonChen
    | Gupta
    | Tip4pCluster
    | GminTip4p
)

class EpochLine:
    @property
    def epoch(self) -> int: ...
    @property
    def temp(self) -> float: ...
    @property
    def accepted(self) -> int: ...
    @property
    def rejected(self) -> int: ...
    @property
    def best_val(self) -> float: ...

class History:
    @property
    def epochs(self) -> list[EpochLine]: ...
    @property
    def best_pos(self) -> list[float]: ...
    @property
    def best_val(self) -> float: ...
    @property
    def stationarity_flags(self) -> list[bool]: ...
    @property
    def total_accepted(self) -> int: ...
    @property
    def total_rejected(self) -> int: ...

class LandfoldMap:
    def project(self, rows: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
    @property
    def landmark_xy(self) -> npt.NDArray[np.float64]: ...
    @property
    def weights(self) -> npt.NDArray[np.float64]: ...

class Config:
    @staticmethod
    def recommended(n: int) -> Config: ...
    @staticmethod
    def communicating(n: int) -> Config: ...
    @staticmethod
    def derived(n: int) -> Config: ...
    @staticmethod
    def for_cluster(n: int) -> Config: ...
    @staticmethod
    def communicating_tip4p(n: int) -> Config: ...
    @staticmethod
    def recommended_tip4p(n: int) -> Config: ...
    @staticmethod
    def validate_json(text: str) -> None: ...
    @staticmethod
    def validate_toml(text: str) -> None: ...
    @property
    def n_points(self) -> int: ...
    @property
    def rigid_body(self) -> bool: ...
    @property
    def screen_margin(self) -> float: ...
    @property
    def calibrated_screen(self) -> bool: ...
    @property
    def finish_improving(self) -> bool: ...
    @property
    def burst_moves(self) -> bool: ...
    @property
    def allocate_moves(self) -> bool: ...
    @property
    def depth_reward(self) -> bool: ...
    @property
    def orbit_complete_on_new(self) -> bool: ...
    @property
    def shared_visit_policy(self) -> str: ...
    @property
    def tabu_on_stall(self) -> bool: ...
    @property
    def bayes_threshold(self) -> float: ...
    @property
    def budget_window(self) -> bool: ...

class Ledger:
    def __init__(self, budget: int) -> None: ...
    @property
    def budget(self) -> int: ...
    @property
    def spent(self) -> int: ...
    @property
    def remaining(self) -> int: ...
    @property
    def best(self) -> float: ...

def energy_hits(energy: float, target: float, tol: float = ...) -> bool: ...
def rigid_from_ohh(cartesian: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
def energy_ohh(cartesian: npt.NDArray[np.float64]) -> float: ...
def block_from_interleaved_gmin(
    interleaved: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]: ...
def interleaved_gmin_from_block(
    block: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]: ...
def gmin_tip4p_sites() -> npt.NDArray[np.float64]: ...
def low_discrepancy_points(
    low: npt.NDArray[np.float64],
    high: npt.NDArray[np.float64],
    n: int,
    skip: int = 1,
) -> list[list[float]]: ...
def osa_acceptance_rate(
    delta: float,
    temp: float,
    sigma: float,
    trials: int = 20000,
    c_star: float = 0.0,
    seed: int = 0,
) -> tuple[float, float]: ...
def pilot_draws_qmc(n: int, seed: int = 42) -> list[tuple[float, float, float]]: ...
def run(
    obj_fn: Callable[[npt.NDArray[np.float64]], float],
    low: npt.NDArray[np.float64],
    high: npt.NDArray[np.float64],
    preset: Preset,
    n_epochs: int = 100,
    steps_per_epoch: int = 200,
    seed: int = 42,
    x0: npt.NDArray[np.float64] | None = None,
    max_evals: int | None = None,
) -> History: ...
def run_qmc(
    obj_fn: Callable[[npt.NDArray[np.float64]], float],
    low: npt.NDArray[np.float64],
    high: npt.NDArray[np.float64],
    preset: Preset,
    n_starts: int = 8,
    n_epochs: int = 100,
    steps_per_epoch: int = 200,
    seed: int = 42,
    x0: npt.NDArray[np.float64] | None = None,
    max_evals: int | None = None,
) -> History: ...
def run_hmc(
    obj_fn: Callable[[npt.NDArray[np.float64]], float],
    grad_fn: Callable[[npt.NDArray[np.float64]], Sequence[float] | npt.NDArray[np.float64]],
    low: npt.NDArray[np.float64],
    high: npt.NDArray[np.float64],
    t_init: float = 5.0,
    epsilon: float = 0.05,
    l_steps: int = 5,
    q: float = 1.0,
    n_epochs: int = 100,
    steps_per_epoch: int = 50,
    seed: int = 42,
    x0: npt.NDArray[np.float64] | None = None,
) -> History: ...
def _run_device(
    obj_fn: Callable[[Any], Any],
    low: Any,
    high: Any,
    preset: Preset,
    *,
    n_epochs: int = 100,
    steps_per_epoch: int = 200,
    seed: int = 42,
    start: Any | None = None,
) -> DeviceHistory: ...
def _run_device_ensemble(
    obj_fn: Callable[[Any], Any],
    low: Any,
    high: Any,
    preset: Preset,
    *,
    n_chains: int,
    n_epochs: int = 100,
    steps_per_epoch: int = 200,
    seed: int = 42,
) -> EnsembleHistory: ...
def polish(
    obj_fn: Callable[[npt.NDArray[np.float64]], float],
    grad_fn: Callable[[npt.NDArray[np.float64]], Sequence[float] | npt.NDArray[np.float64]],
    low: npt.NDArray[np.float64],
    high: npt.NDArray[np.float64],
    x0: npt.NDArray[np.float64],
    max_fevals: int = 200,
    step0: float = 1.0,
    grad_tol: float = 1e-8,
) -> dict[str, Any]: ...
def qmc_polish(
    obj_fn: Callable[[npt.NDArray[np.float64]], float],
    grad_fn: Callable[[npt.NDArray[np.float64]], Sequence[float] | npt.NDArray[np.float64]],
    low: npt.NDArray[np.float64],
    high: npt.NDArray[np.float64],
    n_starts: int,
    max_fevals_per_start: int,
    seed: int = 0,
    step0: float = 1.0,
    grad_tol: float = 1e-8,
    top_k: int = 0,
) -> dict[str, Any]: ...
def qmc_polish_objective(
    objective: PyObjective,
    n_starts: int,
    max_fevals_per_start: int,
    seed: int = 0,
    step0: float = 1.0,
    grad_tol: float = 1e-8,
    top_k: int = 0,
) -> dict[str, Any]: ...
def qmc_best1bin_scout(
    obj_fn: Callable[[npt.NDArray[np.float64]], float],
    low: npt.NDArray[np.float64],
    high: npt.NDArray[np.float64],
    max_evals: int,
    seed: int = 0,
    population_size: int = 30,
    weight_min: float = 0.5,
    weight_span: float = 0.5,
    crossover_rate: float = 0.7,
) -> dict[str, Any]: ...
def qmc_best1bin_scout_objective(
    objective: PyObjective,
    max_evals: int,
    seed: int = 0,
    population_size: int = 30,
    weight_min: float = 0.5,
    weight_span: float = 0.5,
    crossover_rate: float = 0.7,
) -> dict[str, Any]: ...
def qmc_gsa_global_search(
    obj_fn: Callable[[npt.NDArray[np.float64]], float],
    low: npt.NDArray[np.float64],
    high: npt.NDArray[np.float64],
    max_evals: int,
    seed: int = 0,
    n_chains: int = 30,
    t_init: float = 1.0,
    q_v: float = 2.62,
    q_a: float = 1.7,
    x0: npt.NDArray[np.float64] | None = None,
) -> dict[str, Any]: ...
def qmc_gsa_global_search_objective(
    objective: PyObjective,
    max_evals: int,
    seed: int = 0,
    n_chains: int = 30,
    t_init: float = 1.0,
    q_v: float = 2.62,
    q_a: float = 1.7,
    x0: npt.NDArray[np.float64] | None = None,
) -> dict[str, Any]: ...
def qmc_trust_region_poll(
    obj_fn: Callable[[npt.NDArray[np.float64]], float],
    low: npt.NDArray[np.float64],
    high: npt.NDArray[np.float64],
    center: npt.NDArray[np.float64],
    max_evals: int,
    seed: int = 0,
    radius_fraction: float = 0.0,
    n_levels: int = 3,
    points_per_level: int = 0,
) -> dict[str, Any]: ...
def qmc_trust_region_poll_objective(
    objective: PyObjective,
    center: npt.NDArray[np.float64],
    max_evals: int,
    seed: int = 0,
    radius_fraction: float = 0.0,
    n_levels: int = 3,
    points_per_level: int = 0,
) -> dict[str, Any]: ...
def shifted_qmc_polish(
    obj_fn: Callable[[npt.NDArray[np.float64]], float],
    grad_fn: Callable[[npt.NDArray[np.float64]], Sequence[float] | npt.NDArray[np.float64]],
    low: npt.NDArray[np.float64],
    high: npt.NDArray[np.float64],
    n_starts: int,
    max_fevals_per_start: int,
    seed: int = 0,
    n_replicates: int = 1,
    step0: float = 1.0,
    grad_tol: float = 1e-8,
    top_k: int = 0,
) -> dict[str, Any]: ...
def additive_independence(
    obj_fn: Callable[[npt.NDArray[np.float64]], float],
    low: npt.NDArray[np.float64],
    high: npt.NDArray[np.float64],
    max_fevals: int,
    seed: int = 0,
    degree: int = 8,
    grid_m: int = 65,
    local_frac: float = 0.2,
    n_epochs: int = 40,
    n_pilot: int = 0,
) -> dict[str, Any]: ...
def estimate_gle_omega0(
    obj_fn: Callable[[npt.NDArray[np.float64]], float],
    grad_fn: Callable[[npt.NDArray[np.float64]], Sequence[float] | npt.NDArray[np.float64]],
    low: npt.NDArray[np.float64],
    high: npt.NDArray[np.float64],
) -> float: ...
def gle_langevin(
    obj_fn: Callable[[npt.NDArray[np.float64]], float],
    grad_fn: Callable[[npt.NDArray[np.float64]], Sequence[float] | npt.NDArray[np.float64]],
    low: npt.NDArray[np.float64],
    high: npt.NDArray[np.float64],
    max_fevals: int,
    seed: int = 0,
    omega0: float | None = None,
    dt: float = 0.2,
    n_epochs: int = 40,
    x0: npt.NDArray[np.float64] | None = None,
) -> dict[str, Any]: ...
def gle_langevin_objective(
    objective: PyObjective,
    max_fevals: int,
    seed: int = 0,
    omega0: float | None = None,
    dt: float = 0.2,
    n_epochs: int = 40,
    x0: npt.NDArray[np.float64] | None = None,
) -> dict[str, Any]: ...
def gle_langevin_preconditioned(
    obj_fn: Callable[[npt.NDArray[np.float64]], float],
    grad_fn: Callable[[npt.NDArray[np.float64]], Sequence[float] | npt.NDArray[np.float64]],
    low: npt.NDArray[np.float64],
    high: npt.NDArray[np.float64],
    max_fevals: int,
    seed: int = 0,
    omega0: float | None = None,
    dt: float = 0.2,
    n_epochs: int = 40,
    x0: npt.NDArray[np.float64] | None = None,
    preconditioner_probes: int | None = None,
) -> dict[str, Any]: ...
def gle_langevin_preconditioned_objective(
    objective: PyObjective,
    max_fevals: int,
    seed: int = 0,
    omega0: float | None = None,
    dt: float = 0.2,
    n_epochs: int = 40,
    x0: npt.NDArray[np.float64] | None = None,
    preconditioner_probes: int | None = None,
) -> dict[str, Any]: ...
def dmc_population_optimize(
    obj_fn: Callable[[npt.NDArray[np.float64]], float],
    low: npt.NDArray[np.float64],
    high: npt.NDArray[np.float64],
    budget: int,
    seed: int = 0,
    grad_fn: Callable[[npt.NDArray[np.float64]], Sequence[float] | npt.NDArray[np.float64]] | None = None,
    target_n: int = 16,
    steps_per_control: int = 4,
    x0: npt.NDArray[np.float64] | None = None,
) -> dict[str, Any]: ...
def gpmd_optimize(
    obj_fn: Callable[[npt.NDArray[np.float64]], float],
    low: npt.NDArray[np.float64],
    high: npt.NDArray[np.float64],
    budget: int,
    seed: int = 0,
    grad_fn: Callable[[npt.NDArray[np.float64]], Sequence[float] | npt.NDArray[np.float64]] | None = None,
    x0: npt.NDArray[np.float64] | None = None,
) -> dict[str, Any]: ...
def amsa_optimize(
    obj_fn: Callable[[npt.NDArray[np.float64]], float],
    low: npt.NDArray[np.float64],
    high: npt.NDArray[np.float64],
    budget: int,
    seed: int = 0,
    grad_fn: Callable[[npt.NDArray[np.float64]], Sequence[float] | npt.NDArray[np.float64]] | None = None,
    x0: npt.NDArray[np.float64] | None = None,
) -> dict[str, Any]: ...
def bfwt_optimize(
    obj_fn: Callable[[npt.NDArray[np.float64]], float],
    low: npt.NDArray[np.float64],
    high: npt.NDArray[np.float64],
    budget: int,
    seed: int = 0,
    barrier_hat: float = 0.0,
    grad_fn: Callable[[npt.NDArray[np.float64]], Sequence[float] | npt.NDArray[np.float64]] | None = None,
    x0: npt.NDArray[np.float64] | None = None,
) -> dict[str, Any]: ...
def box_ensemble_optimize(
    obj_fn: Callable[[npt.NDArray[np.float64]], float],
    low: npt.NDArray[np.float64],
    high: npt.NDArray[np.float64],
    budget: int,
    seed: int = 0,
    grad_fn: Callable[[npt.NDArray[np.float64]], Sequence[float] | npt.NDArray[np.float64]] | None = None,
    x0: npt.NDArray[np.float64] | None = None,
    replicas: int = 4,
    history: str = "shared",
    membership: str = "accepted",
) -> dict[str, Any]: ...
def ensemble_optimize(
    obj_fn: Callable[[npt.NDArray[np.float64]], float],
    low: npt.NDArray[np.float64],
    high: npt.NDArray[np.float64],
    budget: int,
    seed: int = 0,
    grad_fn: Callable[[npt.NDArray[np.float64]], Sequence[float] | npt.NDArray[np.float64]] | None = None,
    x0: npt.NDArray[np.float64] | None = None,
    replicas: int = 4,
    history: str = "shared",
    membership: str = "accepted",
    *,
    coverage_shared: bool | None = None,
    coverage_radius: float | None = None,
    coverage_neighbors: int | None = None,
) -> dict[str, Any]: ...
def global_optimize(
    obj_fn: Callable[[npt.NDArray[np.float64]], float],
    low: npt.NDArray[np.float64],
    high: npt.NDArray[np.float64],
    budget: int,
    seed: int = 0,
    grad_fn: Callable[[npt.NDArray[np.float64]], Sequence[float] | npt.NDArray[np.float64]] | None = None,
    noise_sigma: float | None = None,
    policy: str = "auto",
    x0: npt.NDArray[np.float64] | None = None,
    *,
    replicas: int = 1,
    coverage_shared: bool = True,
    coverage_radius: float = 0.05,
    coverage_neighbors: int = 0,
) -> dict[str, Any]: ...
def global_optimize_objective(
    objective: PyObjective,
    budget: int,
    seed: int = 0,
    use_gradient: bool = True,
    noise_sigma: float | None = None,
    x0: npt.NDArray[np.float64] | None = None,
) -> dict[str, Any]: ...
def cluster_search(
    obj_fn: Callable[[npt.NDArray[np.float64]], float] | NativeClusterPotential,
    grad_fn: Callable[[npt.NDArray[np.float64]], Sequence[float] | npt.NDArray[np.float64]] | None,
    n: int,
    budget: int,
    seed: int = 0,
    recommended: bool = True,
    derived: bool = False,
    communicating: bool = False,
    *,
    ras: bool = False,
    start: npt.NDArray[np.float64] | None = None,
    length_scale: float | None = None,
    species: list[int] | None = None,
    sigmas: list[float] | None = None,
    rigid: bool = False,
    layout: str = "block",
    target: float | None = None,
    water_sites: npt.NDArray[np.float64] | None = None,
    groups: list[list[int]] | None = None,
    local_step: str | None = None,
    ring_moves: bool | None = None,
    hold_stereocentres: bool | None = None,
    measured_energy_scale: bool | None = None,
) -> dict[str, Any]: ...
def quench(
    fg_fn: Callable[[npt.NDArray[np.float64]], tuple[float, Sequence[float]]] | NativeClusterPotential,
    x0: npt.NDArray[np.float64],
    gtol: float,
    max_iter: int = 10000,
    method: str = "lbfgs",
    species: list[int] | None = None,
) -> dict[str, Any]: ...
def landfold(
    rows: npt.NDArray[np.float64],
    *,
    landmarks: int = ...,
    seed: int = 0,
    sigma: float | None = None,
    sigma_scale: float = ...,
    sigma_low: float | None = None,
    high: tuple[float, float] = ...,
    low: tuple[float, float] = ...,
    grid: int = ...,
    iterations: int = ...,
    mix: list[float] | None = None,
    weight_power: float = ...,
    metric: str = "l1",
    threads: int | None = None,
) -> dict[str, Any]: ...
def landfold_book(
    paths: list[str],
    *,
    landmarks: int = ...,
    seed: int = 0,
    sigma: float | None = None,
    sigma_scale: float = ...,
    sigma_low: float | None = None,
    high: tuple[float, float] = ...,
    low: tuple[float, float] = ...,
    grid: int = ...,
    iterations: int = ...,
    mix: list[float] | None = None,
    weight_power: float = ...,
    threads: int | None = None,
) -> dict[str, Any]: ...
