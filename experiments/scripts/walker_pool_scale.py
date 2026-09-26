#!/usr/bin/env python3
"""Prove multi-walker process-pool scaling (no CUTEst required).

Simulates an expensive objective with time.sleep, evaluates a batch of
walker proposals under a ProcessPoolExecutor with one "problem" per worker
(same architecture as CUTEst walker clones).
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

_WORKER_COST = 0.002  # seconds per eval (synthetic expensive CUTEst)


def _init(cost: float) -> None:
    global _WORKER_COST
    _WORKER_COST = cost


def _eval_chunk(rows: np.ndarray) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float64)
    out = np.empty(rows.shape[0], dtype=np.float64)
    for i in range(rows.shape[0]):
        # Synthetic work ~ CUTEst: non-trivial CPU / wall per point.
        x = rows[i]
        s = 0.0
        for _ in range(8000):
            s += float(np.dot(x, x))
        time.sleep(_WORKER_COST * 0.0)  # optional wall pad; CPU loop dominates
        out[i] = s
    return out


def run_batch(n_walkers: int, dim: int, n_workers: int) -> float:
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n_walkers, dim))
    t0 = time.perf_counter()
    if n_workers <= 1:
        _ = _eval_chunk(X)
    else:
        chunks = [c for c in np.array_split(X, n_workers) if c.shape[0] > 0]
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_init,
            initargs=(0.0,),
        ) as pool:
            futs = [pool.submit(_eval_chunk, c) for c in chunks]
            _ = [f.result() for f in futs]
    return time.perf_counter() - t0


def main():
    n_walkers, dim = 64, 50
    print(f"synthetic multi-walker batch n={n_walkers} dim={dim}")
    print("workers  wall_s  speedup")
    t1 = None
    for w in (1, 2, 4, 8):
        # warm
        run_batch(n_walkers, dim, w)
        dt = run_batch(n_walkers, dim, w)
        if t1 is None:
            t1 = dt
        print(f"{w:7d}  {dt:.4f}  {t1 / dt:.2f}x")


if __name__ == "__main__":
    main()
