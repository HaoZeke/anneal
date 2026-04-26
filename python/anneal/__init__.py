"""anneal: simulated annealing components on the eindir typed primitives.

Public API:
  - Boltzmann(t_init, sigma): logarithmic cooling + Gaussian + Metropolis.
  - Fast(t_init, gamma): reciprocal cooling + Cauchy + Metropolis.
  - Gsa(t_init, q_v, q_a): Tsallis cooling + Tsallis visit + Tsallis accept.
  - run(obj_fn, low, high, preset, n_epochs, steps_per_epoch, seed): SA loop.
  - History, EpochLine: returned by `run`.

The IISE-manuscript composition laws L1-L4 are enforced inside the Rust
SaVariant::checked constructor; preset constructors call it under the hood.
"""

from anneal._core import (
    Boltzmann,
    EpochLine,
    Fast,
    Gsa,
    History,
    __version__,
    run,
    run_hmc,
)

__all__ = [
    "Boltzmann",
    "EpochLine",
    "Fast",
    "Gsa",
    "History",
    "__version__",
    "run",
    "run_hmc",
]
