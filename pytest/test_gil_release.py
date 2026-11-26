# Copyright (c) anneal developers
"""Native potential kernels release the GIL for the pair walk."""

from __future__ import annotations

import os
import threading
import time

import numpy as np

import anneal


def _ratio(work) -> float:
    work()
    started = time.perf_counter()
    work()
    work()
    serial = time.perf_counter() - started

    def parallel() -> float:
        threads = [threading.Thread(target=work) for _ in range(2)]
        started = time.perf_counter()
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        return time.perf_counter() - started

    samples = sorted(parallel() for _ in range(3))
    return samples[1] / serial


def test_pair_eval_and_grad_release_the_gil():
    """Two threads overlap a long pair walk when the GIL is released.

    A held GIL keeps the parallel run at about the serial run. Releasing
    it around the kernel drops the median parallel/serial ratio below 0.75
    on two or more cores. n=1200 and 8 repeats sit near one second of
    serial work, which is long enough that thread startup does not hide
    the difference.
    """
    cores = os.cpu_count() or 1
    assert cores >= 2, f"need two cores to see a released GIL, found {cores}"
    n = 1200
    reps = 8
    x = np.random.default_rng(0).standard_normal(3 * n)

    def evals() -> None:
        pot = anneal.PairPotential.lennard_jones(n)
        for _ in range(reps):
            pot.eval(x)

    def grads() -> None:
        pot = anneal.PairPotential.lennard_jones(n)
        for _ in range(reps):
            pot.grad(x)

    eval_ratio = _ratio(evals)
    grad_ratio = _ratio(grads)
    assert eval_ratio < 0.75, f"eval parallel/serial = {eval_ratio:.3f}"
    assert grad_ratio < 0.75, f"grad parallel/serial = {grad_ratio:.3f}"
