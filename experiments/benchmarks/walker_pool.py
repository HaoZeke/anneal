"""Process-pool multi-walker evaluation for non-reentrant objectives (CUTEst).

Each worker process loads its own CUTEst problem instance so Fortran/global
state is not shared. The parent process only ships proposal matrices and
aggregates energies — this is the scaling path for dmc_pop on SOTA/CUTEst.
"""

from __future__ import annotations

import atexit
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

# Per-worker problem callable (set by initializer).
_WORKER_FN = None


def _config_from_paths(paths: dict[str, str]):
    from experiments.benchmarks.cutest_runner import CutestConfig

    return CutestConfig(
        bench_dir=paths["bench_dir"],
        cache_dir=paths["cache_dir"],
        myarch=paths.get("myarch", "pc64.lnx.gfo"),
    )


def _worker_init(name: str, sif_params: dict | None, paths: dict[str, str]) -> None:
    """Load one CUTEst clone per worker. Serialize compile with a file lock."""
    global _WORKER_FN
    import fcntl
    from pathlib import Path

    from experiments.benchmarks.cutest_runner import load

    config = _config_from_paths(paths)
    config.validate()
    lock_path = Path(paths["cache_dir"]) / f".cutest_load_{name}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+", encoding="utf-8") as lockf:
        fcntl.flock(lockf.fileno(), fcntl.LOCK_EX)
        try:
            prob = load(name, sif_params=sif_params, config=config)
            _WORKER_FN = prob.fn
        finally:
            fcntl.flock(lockf.fileno(), fcntl.LOCK_UN)


def _worker_eval_chunk(rows: np.ndarray) -> np.ndarray:
    """Evaluate a (k, dim) chunk inside a worker process."""
    global _WORKER_FN
    if _WORKER_FN is None:
        raise RuntimeError("walker worker not initialized")
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    out = np.empty(rows.shape[0], dtype=np.float64)
    fn = _WORKER_FN
    for i in range(rows.shape[0]):
        out[i] = float(fn(rows[i]))
    return out


def config_paths(config) -> dict[str, str]:
    return {
        "bench_dir": str(config.bench_dir),
        "cache_dir": str(config.cache_dir),
        "myarch": str(config.myarch),
    }


@dataclass
class ProcessWalkerPool:
    """One persistent process pool for a single CUTEst problem name."""

    name: str
    n_workers: int
    sif_params: dict | None
    paths: dict[str, str]
    _pool: Any = None

    @classmethod
    def create(
        cls,
        name: str,
        *,
        n_workers: int,
        config,
        sif_params: dict | None = None,
    ) -> "ProcessWalkerPool":
        n_workers = max(1, int(n_workers))
        paths = config_paths(config)
        pool = ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_worker_init,
            initargs=(name, sif_params, paths),
        )
        inst = cls(
            name=name,
            n_workers=n_workers,
            sif_params=sif_params,
            paths=paths,
            _pool=pool,
        )
        atexit.register(inst.shutdown)
        return inst

    def eval_matrix(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        n = int(X.shape[0])
        if n == 0:
            return np.zeros(0, dtype=np.float64)
        if self.n_workers <= 1 or n == 1:
            return _worker_eval_chunk(X) if _WORKER_FN is not None else self._eval_local_fallback(X)

        n_chunks = min(self.n_workers, n)
        chunks = [c for c in np.array_split(X, n_chunks) if c.shape[0] > 0]
        futs = [self._pool.submit(_worker_eval_chunk, c) for c in chunks]
        parts = [f.result() for f in futs]
        return np.concatenate(parts, axis=0)

    def _eval_local_fallback(self, X: np.ndarray) -> np.ndarray:
        # Parent-side serial (should not be used when pool is live).
        from experiments.benchmarks.cutest_runner import load

        config = _config_from_paths(self.paths)
        prob = load(self.name, sif_params=self.sif_params, config=config)
        return np.asarray([float(prob.fn(X[i])) for i in range(X.shape[0])], dtype=np.float64)

    def shutdown(self) -> None:
        pool = self._pool
        self._pool = None
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)


def default_worker_count() -> int:
    """ANNEAL_WALKER_WORKERS, default 1 (safe). Set to nproc for scaling runs."""
    raw = os.environ.get("ANNEAL_WALKER_WORKERS", "1")
    try:
        return max(1, int(raw))
    except ValueError:
        return 1
