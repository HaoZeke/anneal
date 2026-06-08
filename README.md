<p align="center">
  <img src="./branding/logo/anneal_logo.png" alt="Anneal" width="280">
</p>

# Anneal

Simulated-annealing components on the eindir typed primitives. One surface, many drivers: classical presets, Bayesian pilot+mixer, GLE colored noise, rank-1 additive independence, QMC polish, device/ensemble scale. All obey the same five-component algebra (Obj / Cool / Neigh / Move / Accept) and four laws.

[![Documentation](https://img.shields.io/badge/docs-anneal.rgoswami.me-blue)](https://anneal.rgoswami.me)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/597892274.svg)](https://zenodo.org/doi/10.5281/zenodo.10672746)

## Install

```bash
pip install anneal
```

Full stack (pinned Rust + Python + docs):

```bash
pixi install
```

## 8-line modern example (additive independence + GLE polish)

```python
import numpy as np
from anneal import additive_independence, gle_langevin, qmc_polish

def rastrigin(x):
    return 10.0 * len(x) + np.sum(x*x - 10.0 * np.cos(2.0 * np.pi * x))

def grad_rastrigin(x):
    return 2.0 * x + 20.0 * np.pi * np.sin(2.0 * np.pi * x)

low = np.full(5, -5.0)
high = np.full(5, 5.0)

# Values-only rank-1 independence (no gradient)
res = additive_independence(rastrigin, low, high, max_fevals=3000, seed=7)
x0 = res["best_pos"]

# Polish with gradient (or use gle_langevin directly if grad available)
refined = qmc_polish(rastrigin, grad_rastrigin, low, high,
                     n_starts=32, max_fevals_per_start=50, seed=0, top_k=1)
print(refined["best_val"])
```

Full docs, tutorials (classical, Bayesian pilot+mixer with Beta trace, GLE, polish+device), algebra, how-tos, and reference at https://anneal.rgoswami.me .

## Development

```bash
pixi install
pixi run -e python python-test
pixi run -e docs docs-export
pixi run -e docs docs-build
```

See `pixi.toml` and `docs/export.el` (modeled on rgpycrumbs/rsx-rs patterns).

## License and citation

MIT. See `LICENSE.txt` and `CITATION.cff`. The reference publication is the IISE/INFORMS Journal on Computing paper (typed component algebras, mechanized equivalences, TLA+ spec). Reproducibility package: HaoZeke/anneal_repro.
