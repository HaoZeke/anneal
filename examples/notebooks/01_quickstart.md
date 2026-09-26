---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# anneal quickstart

Minimal path for new users. Install with `pip install anneal` (or use the
project `pixi` environment). Full docs: https://anneal.rgoswami.me

This notebook exercises the **stand-alone** tools: classical presets and the
budget-only portfolio `global_optimize`. No paper proofs required.

```{code-cell} ipython3
import numpy as np
import anneal
print("anneal", anneal.__version__)
```

## 1. Portfolio optimizer (recommended entry)

One argument besides the objective and box: a work-unit budget (objective +
gradient evaluations share the counter).

```{code-cell} ipython3
from anneal import global_optimize

def rastrigin(x):
    return 10.0 * len(x) + np.sum(x * x - 10.0 * np.cos(2.0 * np.pi * x))

low = np.full(5, -5.0)
high = np.full(5, 5.0)
out = global_optimize(rastrigin, low, high, budget=2000, seed=0)
print("best_val", float(out["best_val"]))
print("best_pos", np.asarray(out["best_pos"]))
print("n_evals", out.get("n_evals"), "n_grads", out.get("n_grads"))
```

## 2. Classical presets (Boltzmann / Fast / GSA)

Same `run` driver; only the preset (Cool / Move / Accept) changes.

```{code-cell} ipython3
from anneal import Boltzmann, Fast, Gsa, run

def rosenbrock(x):
    return (1.0 - x[0]) ** 2 + 100.0 * (x[1] - x[0] ** 2) ** 2

low2 = np.array([-5.0, -5.0])
high2 = np.array([5.0, 5.0])

for name, preset in [
    ("Boltzmann", Boltzmann(t_init=5.0, sigma=0.5)),
    ("Fast", Fast(t_init=5.0, gamma=0.5)),
    ("Gsa", Gsa(t_init=5.0, q_v=2.5, q_a=1.5)),
]:
    h = run(rosenbrock, low2, high2, preset, n_epochs=25, steps_per_epoch=40, seed=42)
    print(f"{name:10s} best_val={h.best_val:.6f}")
```

## 3. Optional: values-only additive independence arm

Rank-1 surrogate independence proposals; no gradient required.

```{code-cell} ipython3
from anneal import additive_independence

res = additive_independence(rastrigin, low, high, max_fevals=1500, seed=7, n_epochs=20)
print("additive best_val", float(res["best_val"]))
```

## Next steps

- Website tutorials: classical, Bayesian pilot+mixer, GLE, polish+device
- Reproducibility (paper tables/figures): https://github.com/HaoZeke/anneal_repro
- Zenodo archive: https://doi.org/10.5281/zenodo.20672620
