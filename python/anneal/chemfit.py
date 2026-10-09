"""ChemFit adapter for anneal.

Allows running anneal directly on a ChemFit Fitter instance with proper
ask/tell lifecycle, initial parameters preservation, and bounds enforcement.
"""

from __future__ import annotations

from typing import Any
import numpy as np


def run_fitter(
    fitter: Any,
    budget: int,
    method: str = "global_optimize",
    preset: Any = None,
    seed: int = 42,
    box_constrained: bool = True,
    **kwargs,
) -> dict[str, Any]:
    """Run an optimization with anneal on a ChemFit Fitter.

    Args:
        fitter: chemfit.Fitter instance with initial_parameters and bounds.
        budget: Total objective evaluations allowed.
        method: "global_optimize" (SOTA portfolio) or "sa" (preset simulated annealing).
        preset: Preset instance for method="sa" (e.g. anneal.Boltzmann()).
        seed: Random seed.
        box_constrained: Whether to reflect proposals inside parameter bounds.
        **kwargs: Additional parameters passed to anneal.global_optimize or anneal.run.

    Returns:
        dict[str, Any]: Optimized parameters matching fitter structure.
    """
    if hasattr(fitter, "fit_anneal"):
        return fitter.fit_anneal(
            budget=budget,
            method=method,
            preset=preset,
            seed=seed,
            box_constrained=box_constrained,
            **kwargs,
        )

    # Standalone adapter if fit_anneal is not yet monkeypatched or on older ChemFit versions
    from pydictnest import flatten_dict, unflatten_dict
    import anneal

    flat_params = flatten_dict(fitter.initial_parameters)
    flat_bounds = flatten_dict(fitter.bounds)

    keys = list(flat_params.keys())
    x0 = np.array([flat_params[k] for k in keys], dtype=np.float64)

    low_list = []
    high_list = []
    for k in keys:
        bound_val = flat_bounds.get(k, None)
        if bound_val is not None and isinstance(bound_val, (tuple, list)) and len(bound_val) == 2:
            lo, hi = bound_val
        else:
            lo, hi = None, None
        low_list.append(-1e4 if lo is None else float(lo))
        high_list.append(1e4 if hi is None else float(hi))

    low = np.array(low_list, dtype=np.float64)
    high = np.array(high_list, dtype=np.float64)

    fitter.init()

    def obj(x: np.ndarray) -> float:
        p = unflatten_dict(dict(zip(keys, x)), dict_factory=dict[str, Any])
        loss = fitter.ask(p)
        assert isinstance(loss, float)
        fitter.tell(fitter.contexts[0].n_evals)
        return loss

    if method == "global_optimize":
        result = anneal.global_optimize(
            obj,
            low,
            high,
            budget=budget,
            seed=seed,
            **kwargs,
        )
        best_pos = result["best_pos"]
    elif method == "sa":
        sa_preset = anneal.Boltzmann() if preset is None else preset
        n_epochs = kwargs.pop("n_epochs", max(1, budget // 100))
        steps_per_epoch = kwargs.pop("steps_per_epoch", min(100, budget))
        history = anneal.run(
            obj,
            low,
            high,
            preset=sa_preset,
            n_epochs=n_epochs,
            steps_per_epoch=steps_per_epoch,
            seed=seed,
            x0=x0,
            box_constrained=box_constrained,
            **kwargs,
        )
        best_pos = np.asarray(history.best_pos, dtype=np.float64)
    else:
        msg = f"Unknown anneal method {method!r}. Choose 'global_optimize' or 'sa'."
        raise ValueError(msg)

    opt_params = dict(zip(keys, best_pos))
    opt_params = unflatten_dict(opt_params)
    return fitter.finish(opt_params)
