"""GPU vs CPU benchmark for batched ensemble simulated annealing.

Runs ``anneal.run_ensemble`` over an ensemble of independent chains on the CPU
(NumPy) and, when available, on the GPU (CuPy), and reports wall time, the
best objective reached across the ensemble, and the GPU speedup. The chains
share one batched device kernel, so the GPU is saturated by the ensemble width
rather than starved by a single chain. The objectives are namespace-agnostic
(they read the Array API namespace of their argument), so the same callable
runs on either backend.

Schema: backend, objective, dim, n_chains, n_epochs, steps_per_epoch,
wall_time_s, global_best, known_min.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import array_api_compat
import numpy as np

from anneal import Boltzmann, Fast, Gsa
from anneal.device import run_ensemble

# The three algebra presets, all driven GPU-resident through the one device
# backend: BSA (Gaussian/log), FSA (Cauchy/reciprocal), GSA (Tsallis).
PRESETS = {
    "bsa": Boltzmann(t_init=5.0, sigma=0.5),
    "fsa": Fast(t_init=5.0, gamma=0.5),
    "gsa": Gsa(t_init=5.0, q_v=2.62, q_a=1.7),
}


def styb_tang(x):
    xp = array_api_compat.array_namespace(x)
    return 0.5 * xp.sum(x**4 - 16.0 * x**2 + 5.0 * x, axis=-1)


def rastrigin(x):
    xp = array_api_compat.array_namespace(x)
    n = x.shape[-1]
    return 10.0 * n + xp.sum(x**2 - 10.0 * xp.cos(2.0 * np.pi * x), axis=-1)


def ackley(x):
    xp = array_api_compat.array_namespace(x)
    n = x.shape[-1]
    s1 = xp.sum(x**2, axis=-1)
    s2 = xp.sum(xp.cos(2.0 * np.pi * x), axis=-1)
    return -20.0 * xp.exp(-0.2 * xp.sqrt(s1 / n)) - xp.exp(s2 / n) + 20.0 + np.e


OBJECTIVES = {
    "styb_tang": (styb_tang, -5.0, 5.0, lambda d: -39.16599 * d),
    "rastrigin": (rastrigin, -5.12, 5.12, lambda d: 0.0),
    "ackley": (ackley, -32.768, 32.768, lambda d: 0.0),
}


def _backends(requested):
    backends = {"numpy": np}
    try:
        import cupy

        cupy.zeros(1)  # force a context; raises if no usable GPU
        backends["cupy"] = cupy
    except Exception as exc:  # noqa: BLE001
        print(f"[gpu-bench] CuPy unavailable, CPU only: {exc}", file=sys.stderr)
    if requested == "all":
        return backends
    return {requested: backends[requested]} if requested in backends else {}


def _sync(xp):
    if xp.__name__.startswith("cupy"):
        xp.cuda.Device().synchronize()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--objective", default="styb_tang", choices=list(OBJECTIVES))
    p.add_argument("--dims", default="2,10,50")
    p.add_argument("--chains", default="256,1024,4096,16384")
    p.add_argument("--n-epochs", type=int, default=40)
    p.add_argument("--steps-per-epoch", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", default="all", choices=["all", "numpy", "cupy"])
    p.add_argument("--presets", default="bsa,fsa,gsa")
    args = p.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    obj_fn, lo, hi, known = OBJECTIVES[args.objective]
    dims = [int(d) for d in args.dims.split(",")]
    chains = [int(c) for c in args.chains.split(",")]
    presets = [p.strip() for p in args.presets.split(",")]
    backends = _backends(args.backend)
    if not backends:
        raise SystemExit("no usable backend")

    rows = []
    timing = {}
    for preset_name in presets:
        preset = PRESETS[preset_name]
        for dim in dims:
            for n_chains in chains:
                for name, xp in backends.items():
                    low = xp.asarray(np.full(dim, lo), dtype=xp.float64)
                    high = xp.asarray(np.full(dim, hi), dtype=xp.float64)
                    _sync(xp)
                    t0 = time.perf_counter()
                    h = run_ensemble(
                        obj_fn,
                        low,
                        high,
                        preset,
                        n_chains=n_chains,
                        n_epochs=args.n_epochs,
                        steps_per_epoch=args.steps_per_epoch,
                        seed=args.seed,
                    )
                    best = float(h.global_best_val)
                    _sync(xp)
                    wall = time.perf_counter() - t0
                    timing[(preset_name, dim, n_chains, name)] = wall
                    rows.append(
                        dict(
                            preset=preset_name,
                            backend=name,
                            objective=args.objective,
                            dim=dim,
                            n_chains=n_chains,
                            n_epochs=args.n_epochs,
                            steps_per_epoch=args.steps_per_epoch,
                            wall_time_s=f"{wall:.4f}",
                            global_best=f"{best:.6f}",
                            known_min=f"{known(dim):.6f}",
                        )
                    )
                    print(
                        f"  {preset_name} {name:6s} obj={args.objective} dim={dim:3d} "
                        f"chains={n_chains:6d} wall={wall:8.3f}s best={best:.4f}"
                    )
                if "numpy" in backends and "cupy" in backends:
                    sp = (
                        timing[(preset_name, dim, n_chains, "numpy")]
                        / timing[(preset_name, dim, n_chains, "cupy")]
                    )
                    print(
                        f"    -> {preset_name} GPU speedup x{sp:.1f} "
                        f"at dim={dim} chains={n_chains}"
                    )

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    sys.exit(main())
