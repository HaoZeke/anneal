# Copyright (c) anneal developers
"""Type stubs for the ``anneal._core`` extension module.

``pytest/test_stubs.py`` compares this file with the compiled module, so a
binding added to ``src/python.rs`` fails the test until it is described here.
"""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Final, Self, final

import numpy as np
from numpy.typing import ArrayLike, NDArray

_Vector = NDArray[np.float64]
_Objective = Callable[[_Vector], float]
_Gradient = Callable[[_Vector], ArrayLike]
_Result = dict[str, Any]

__all__ = [
    "additive_independence",
    "amsa_optimize",
    "AnnealError",
    "BasinBias",
    "bfwt_optimize",
    "Boltzmann",
    "Bounds",
    "BoundsError",
    "box_ensemble_optimize",
    "cluster_search",
    "Config",
    "ConfigError",
    "DEFAULT_HIT_TOL",
    "dmc_population_optimize",
    "energy_hits",
    "ensemble_optimize",
    "EpochLine",
    "estimate_gle_omega0",
    "Fast",
    "FeatureUnavailableError",
    "gle_langevin",
    "gle_langevin_objective",
    "gle_langevin_preconditioned",
    "gle_langevin_preconditioned_objective",
    "global_optimize",
    "global_optimize_objective",
    "gpmd_optimize",
    "Gsa",
    "History",
    "InvalidArgumentError",
    "landfold",
    "landfold_book",
    "LandfoldMap",
    "Ledger",
    "low_discrepancy_points",
    "NoFiniteStartError",
    "osa_acceptance_rate",
    "pilot_draws_qmc",
    "polish",
    "PyObjective",
    "qmc_best1bin_scout",
    "qmc_best1bin_scout_objective",
    "qmc_gsa_global_search",
    "qmc_gsa_global_search_objective",
    "qmc_polish",
    "qmc_polish_objective",
    "qmc_trust_region_poll",
    "qmc_trust_region_poll_objective",
    "quench",
    "RemovedConfigKeyError",
    "run",
    "_run_device",
    "_run_device_ensemble",
    "run_hmc",
    "run_qmc",
    "SEARCH_HIT_TOL",
    "shifted_qmc_polish",
    "__version__",
]

__version__: Final[str]
DEFAULT_HIT_TOL: Final[float]
SEARCH_HIT_TOL: Final[float]

class AnnealError(ValueError): ...
class InvalidArgumentError(AnnealError): ...
class BoundsError(InvalidArgumentError): ...
class ConfigError(AnnealError):
    key: str

class RemovedConfigKeyError(ConfigError): ...
class FeatureUnavailableError(AnnealError): ...

class NoFiniteStartError(AnnealError):
    attempts: int

@final
class Boltzmann:
    def __new__(cls, t_init: float = 1.0, sigma: float = 0.5) -> Self: ...
    @property
    def t_init(self) -> float: ...
    @property
    def sigma(self) -> float: ...

@final
class Fast:
    def __new__(cls, t_init: float = 1.0, gamma: float = 0.5) -> Self: ...
    @property
    def t_init(self) -> float: ...
    @property
    def gamma(self) -> float: ...

@final
class Gsa:
    def __new__(cls, t_init: float = 1.0, q_v: float = 2.62, q_a: float = 1.7) -> Self: ...
    @property
    def t_init(self) -> float: ...
    @property
    def q_v(self) -> float: ...
    @property
    def q_a(self) -> float: ...

@final
class EpochLine:
    @property
    def epoch(self) -> int: ...
    @property
    def temp(self) -> float: ...
    @property
    def best_val(self) -> float: ...
    @property
    def accepted(self) -> int: ...
    @property
    def rejected(self) -> int: ...

@final
class History:
    @property
    def best_pos(self) -> list[float]: ...
    @property
    def best_val(self) -> float: ...
    @property
    def epochs(self) -> list[EpochLine]: ...
    @property
    def stationarity_flags(self) -> list[bool]: ...
    @property
    def total_accepted(self) -> int: ...
    @property
    def total_rejected(self) -> int: ...

@final
class BasinBias:
    def __new__(
        cls,
        n_points: int,
        merge_radius: float = 1e-2,
        w0: float = 0.25,
        gamma: float = 5.0,
    ) -> Self: ...
    def potential(self, x: _Vector) -> float: ...
    def deposit(self, x: _Vector, temp: float) -> None: ...
    @property
    def n_basins(self) -> int: ...
    @property
    def deepest(self) -> float: ...
    @property
    def missing_mass(self) -> float: ...

@final
class Bounds:
    def __new__(cls, low: _Vector, high: _Vector, slack: float = 1e-6) -> Self: ...
    def clip(self, x: _Vector) -> _Vector: ...
    def contains(self, x: _Vector) -> bool: ...
    @property
    def dims(self) -> int: ...
    @property
    def slack(self) -> float: ...

@final
class PyObjective:
    def __new__(
        cls,
        fn_: _Objective,
        bounds: Bounds,
        grad_fn: _Gradient | None = None,
    ) -> Self: ...
    def eval(self, x: _Vector) -> float: ...
    def grad(self, x: _Vector) -> _Vector: ...
    @property
    def dim(self) -> int: ...
    @property
    def bounds(self) -> Bounds: ...

@final
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
    def for_tip4p(n: int) -> Config: ...
    @staticmethod
    def with_scales(n: int, length_scale: float, energy_scale: float) -> Config: ...
    @staticmethod
    def recommended_with_scales(n: int, length_scale: float, energy_scale: float) -> Config: ...
    @staticmethod
    def for_cluster_with_scales(n: int, length_scale: float, energy_scale: float) -> Config: ...
    @staticmethod
    def recommended_for_pair(n: int, pair: str, rho: float | None = None) -> Config: ...
    @staticmethod
    def for_cluster_dual_core(n: int) -> Config: ...
    @staticmethod
    def packing_superbasin(n: int) -> Config: ...
    @staticmethod
    def for_molecular(
        species: Sequence[int],
        groups: Sequence[Sequence[int]],
        energy_scale: float = 1.0,
    ) -> Config: ...
    @staticmethod
    def recommended_molecular(
        species: Sequence[int],
        groups: Sequence[Sequence[int]],
        energy_scale: float = 1.0,
    ) -> Config: ...
    def with_dual_orbit_core(self) -> Config: ...
    def with_species(
        self,
        species: Sequence[int],
        sigma: Sequence[float] | None = None,
    ) -> Config: ...
    def with_flexible_molecules(self, start: _Vector) -> Config | None: ...
    @staticmethod
    def field_names() -> list[str]: ...
    def get(self, name: str) -> Any: ...
    def to_dict(self) -> dict[str, Any]: ...
    def replace(self, **fields: Any) -> Config: ...
    def resolved_json(self) -> str: ...
    def resolved_sha256(self) -> str: ...
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

@final
class Ledger:
    def __new__(cls, budget: int) -> Self: ...
    @property
    def budget(self) -> int: ...
    @property
    def spent(self) -> int: ...
    @property
    def remaining(self) -> int: ...
    @property
    def best(self) -> float: ...

@final
class LandfoldMap:
    def project(self, rows: ArrayLike) -> NDArray[np.float64]: ...
    @property
    def landmark_xy(self) -> NDArray[np.float64]: ...
    @property
    def weights(self) -> _Vector: ...

def energy_hits(energy: float, target: float, tol: float = ...) -> bool: ...
def low_discrepancy_points(
    low: _Vector, high: _Vector, n: int, skip: int = 1
) -> list[list[float]]: ...
def pilot_draws_qmc(n: int, seed: int = 42) -> list[list[float]]: ...
def osa_acceptance_rate(
    delta: float,
    temp: float,
    sigma: float,
    trials: int = 20000,
    c_star: float = 0.0,
    seed: int = 0,
) -> tuple[float, float]: ...
def run(
    obj_fn: _Objective,
    low: _Vector,
    high: _Vector,
    preset: Boltzmann | Fast | Gsa,
    n_epochs: int = 100,
    steps_per_epoch: int = 200,
    seed: int = 42,
    x0: _Vector | None = None,
    max_evals: int | None = None,
) -> History: ...
def run_qmc(
    obj_fn: _Objective,
    low: _Vector,
    high: _Vector,
    preset: Boltzmann | Fast | Gsa,
    n_starts: int = 8,
    n_epochs: int = 100,
    steps_per_epoch: int = 200,
    seed: int = 42,
    x0: _Vector | None = None,
    max_evals: int | None = None,
) -> History: ...
def run_hmc(
    obj_fn: _Objective,
    grad_fn: _Gradient,
    low: _Vector,
    high: _Vector,
    t_init: float = 5.0,
    epsilon: float = 0.05,
    l_steps: int = 5,
    q: float = 1.0,
    n_epochs: int = 100,
    steps_per_epoch: int = 50,
    seed: int = 42,
    x0: _Vector | None = None,
) -> History: ...
def _run_device(
    obj_fn: Any,
    low: Any,
    high: Any,
    preset: Boltzmann | Fast | Gsa,
    *,
    n_epochs: int = 100,
    steps_per_epoch: int = 200,
    seed: int = 42,
    start: Any | None = None,
) -> Any: ...
def _run_device_ensemble(
    obj_fn: Any,
    low: Any,
    high: Any,
    preset: Boltzmann | Fast | Gsa,
    *,
    n_chains: int,
    n_epochs: int = 100,
    steps_per_epoch: int = 200,
    seed: int = 42,
) -> Any: ...
def polish(
    obj_fn: _Objective,
    grad_fn: _Gradient,
    low: _Vector,
    high: _Vector,
    x0: _Vector,
    max_fevals: int = 200,
    step0: float = 1.0,
    grad_tol: float = 1e-8,
) -> _Result: ...
def qmc_polish(
    obj_fn: _Objective,
    grad_fn: _Gradient,
    low: _Vector,
    high: _Vector,
    n_starts: int,
    max_fevals_per_start: int,
    seed: int = 0,
    step0: float = 1.0,
    grad_tol: float = 1e-8,
    top_k: int = 0,
) -> _Result: ...
def qmc_polish_objective(
    objective: PyObjective,
    n_starts: int,
    max_fevals_per_start: int,
    seed: int = 0,
    step0: float = 1.0,
    grad_tol: float = 1e-8,
    top_k: int = 0,
) -> _Result: ...
def shifted_qmc_polish(
    obj_fn: _Objective,
    grad_fn: _Gradient,
    low: _Vector,
    high: _Vector,
    n_starts: int,
    max_fevals_per_start: int,
    seed: int = 0,
    n_replicates: int = 1,
    step0: float = 1.0,
    grad_tol: float = 1e-8,
    top_k: int = 0,
) -> _Result: ...
def qmc_best1bin_scout(
    obj_fn: _Objective,
    low: _Vector,
    high: _Vector,
    max_evals: int,
    seed: int = 0,
    population_size: int = 30,
    weight_min: float = 0.5,
    weight_span: float = 0.5,
    crossover_rate: float = 0.7,
) -> _Result: ...
def qmc_best1bin_scout_objective(
    objective: PyObjective,
    max_evals: int,
    seed: int = 0,
    population_size: int = 30,
    weight_min: float = 0.5,
    weight_span: float = 0.5,
    crossover_rate: float = 0.7,
) -> _Result: ...
def qmc_gsa_global_search(
    obj_fn: _Objective,
    low: _Vector,
    high: _Vector,
    max_evals: int,
    seed: int = 0,
    n_chains: int = 30,
    t_init: float = 1.0,
    q_v: float = 2.62,
    q_a: float = 1.7,
    x0: _Vector | None = None,
) -> _Result: ...
def qmc_gsa_global_search_objective(
    objective: PyObjective,
    max_evals: int,
    seed: int = 0,
    n_chains: int = 30,
    t_init: float = 1.0,
    q_v: float = 2.62,
    q_a: float = 1.7,
    x0: _Vector | None = None,
) -> _Result: ...
def qmc_trust_region_poll(
    obj_fn: _Objective,
    low: _Vector,
    high: _Vector,
    center: _Vector,
    max_evals: int,
    seed: int = 0,
    radius_fraction: float = 0.0,
    n_levels: int = 3,
    points_per_level: int = 0,
) -> _Result: ...
def qmc_trust_region_poll_objective(
    objective: PyObjective,
    center: _Vector,
    max_evals: int,
    seed: int = 0,
    radius_fraction: float = 0.0,
    n_levels: int = 3,
    points_per_level: int = 0,
) -> _Result: ...
def additive_independence(
    obj_fn: _Objective,
    low: _Vector,
    high: _Vector,
    max_fevals: int,
    seed: int = 0,
    degree: int = 8,
    grid_m: int = 65,
    local_frac: float = 0.2,
    n_epochs: int = 40,
    n_pilot: int = 0,
) -> _Result: ...
def estimate_gle_omega0(
    obj_fn: _Objective, grad_fn: _Gradient, low: _Vector, high: _Vector
) -> float: ...
def gle_langevin(
    obj_fn: _Objective,
    grad_fn: _Gradient,
    low: _Vector,
    high: _Vector,
    max_fevals: int,
    seed: int = 0,
    omega0: float | None = None,
    dt: float = 0.2,
    n_epochs: int = 40,
    x0: _Vector | None = None,
) -> _Result: ...
def gle_langevin_objective(
    objective: PyObjective,
    max_fevals: int,
    seed: int = 0,
    omega0: float | None = None,
    dt: float = 0.2,
    n_epochs: int = 40,
    x0: _Vector | None = None,
) -> _Result: ...
def gle_langevin_preconditioned(
    obj_fn: _Objective,
    grad_fn: _Gradient,
    low: _Vector,
    high: _Vector,
    max_fevals: int,
    seed: int = 0,
    omega0: float | None = None,
    dt: float = 0.2,
    n_epochs: int = 40,
    x0: _Vector | None = None,
    preconditioner_probes: int | None = None,
) -> _Result: ...
def gle_langevin_preconditioned_objective(
    objective: PyObjective,
    max_fevals: int,
    seed: int = 0,
    omega0: float | None = None,
    dt: float = 0.2,
    n_epochs: int = 40,
    x0: _Vector | None = None,
    preconditioner_probes: int | None = None,
) -> _Result: ...
def dmc_population_optimize(
    obj_fn: _Objective,
    low: _Vector,
    high: _Vector,
    budget: int,
    seed: int = 0,
    grad_fn: _Gradient | None = None,
    target_n: int = 16,
    steps_per_control: int = 4,
    x0: _Vector | None = None,
) -> _Result: ...
def gpmd_optimize(
    obj_fn: _Objective,
    low: _Vector,
    high: _Vector,
    budget: int,
    seed: int = 0,
    grad_fn: _Gradient | None = None,
    x0: _Vector | None = None,
) -> _Result: ...
def amsa_optimize(
    obj_fn: _Objective,
    low: _Vector,
    high: _Vector,
    budget: int,
    seed: int = 0,
    grad_fn: _Gradient | None = None,
    x0: _Vector | None = None,
) -> _Result: ...
def bfwt_optimize(
    obj_fn: _Objective,
    low: _Vector,
    high: _Vector,
    budget: int,
    seed: int = 0,
    barrier_hat: float = 0.0,
    grad_fn: _Gradient | None = None,
    x0: _Vector | None = None,
) -> _Result: ...
def box_ensemble_optimize(
    obj_fn: _Objective,
    low: _Vector,
    high: _Vector,
    budget: int,
    seed: int = 0,
    grad_fn: _Gradient | None = None,
    x0: _Vector | None = None,
    replicas: int = 4,
    history: str = "shared",
    membership: str = "accepted",
    *,
    identity_tol: float | None = None,
    shared_deposits: int | None = None,
    shared_visit_policy: str | None = None,
    escape: str | None = None,
    langevin_steps: int | None = None,
    langevin_omega0: float | None = None,
    langevin_dt: float | None = None,
    langevin_friction: float | None = None,
    coverage_shared: bool | None = None,
    coverage_radius: float | None = None,
    coverage_neighbors: int | None = None,
    coverage_height: float | None = None,
    coverage_well_tempering: float | None = None,
    coverage_peer_weight: float | None = None,
) -> _Result: ...
def ensemble_optimize(
    obj_fn: _Objective,
    low: _Vector,
    high: _Vector,
    budget: int,
    seed: int = 0,
    grad_fn: _Gradient | None = None,
    x0: _Vector | None = None,
    replicas: int = 4,
    history: str = "shared",
    membership: str = "accepted",
    *,
    coverage_shared: bool | None = None,
    coverage_radius: float | None = None,
    coverage_neighbors: int | None = None,
    identity_tol: float | None = None,
    shared_deposits: int | None = None,
    shared_visit_policy: str | None = None,
    escape: str | None = None,
    langevin_steps: int | None = None,
    langevin_omega0: float | None = None,
    langevin_dt: float | None = None,
    langevin_friction: float | None = None,
    coverage_height: float | None = None,
    coverage_well_tempering: float | None = None,
    coverage_peer_weight: float | None = None,
) -> _Result: ...
def global_optimize(
    obj_fn: _Objective,
    low: _Vector,
    high: _Vector,
    budget: int,
    seed: int = 0,
    grad_fn: _Gradient | None = None,
    noise_sigma: float | None = None,
    policy: str = "auto",
    x0: _Vector | None = None,
    *,
    replicas: int = 1,
    coverage_shared: bool = True,
    coverage_radius: float = 0.05,
    coverage_neighbors: int | str = 0,
) -> _Result: ...
def global_optimize_objective(
    objective: PyObjective,
    budget: int,
    seed: int = 0,
    use_gradient: bool = True,
    noise_sigma: float | None = None,
    x0: _Vector | None = None,
) -> _Result: ...
def cluster_search(
    obj_fn: _Objective,
    grad_fn: _Gradient,
    n: int,
    budget: int,
    seed: int = 0,
    recommended: bool = True,
    derived: bool = False,
    communicating: bool = False,
    *,
    ras: bool = False,
    start: _Vector | None = None,
    length_scale: float | None = None,
    species: Sequence[int] | None = None,
    sigmas: Sequence[float] | None = None,
    rigid: bool = False,
    layout: str = "block",
    target: float | None = None,
    water_sites: _Vector | None = None,
    groups: Sequence[Sequence[int]] | None = None,
    local_step: str | None = None,
    ring_moves: bool | None = None,
    hold_stereocentres: bool | None = None,
    measured_energy_scale: bool | None = None,
    config: Config | None = None,
    full: bool = False,
) -> _Result: ...
def quench(
    fg_fn: Callable[[_Vector], tuple[float, ArrayLike]],
    x0: _Vector,
    gtol: float,
    max_iter: int = 10000,
    method: str = "lbfgs",
    species: Sequence[int] | None = None,
) -> _Result: ...
def landfold(
    rows: ArrayLike,
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
    mix: Sequence[float] | None = None,
    weight_power: float = ...,
    metric: str = "l1",
    threads: int | None = None,
) -> _Result: ...
def landfold_book(
    paths: Sequence[str | Path],
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
    mix: Sequence[float] | None = None,
    weight_power: float = ...,
    threads: int | None = None,
) -> _Result: ...
