"""Panel and row schema for the paired PR 176 follow-up."""

from __future__ import annotations

from dataclasses import dataclass

BASE_COMMIT = "e25221f3a0bf4f78a67d4b7877d38f813b002ef5"
PR_HEAD = "6b12b9aab1e78589f0634413aae8c12611104607"
BASE_TREE = "cc5c9c3c4062db5e4afaaf4c007cabc8965851cc"
EXPECTED_B_TREE = "c9f9914d243a9b362513f49ffae749f4ef980e79"
SESSION = "pr176-followup-n24"
SEEDS: tuple[int, ...] = tuple(range(2000, 2024))
HIT_TOL = 1e-3
TIMED_PAD_BYTES = 1

ARM_ROTATION: tuple[tuple[str, str, str], ...] = (
    ("A", "B", "A2"),
    ("B", "A2", "A"),
    ("A2", "A", "B"),
    ("B", "A", "A2"),
    ("A", "A2", "B"),
    ("A2", "B", "A"),
)

THREAD_ENV: dict[str, str] = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "RAYON_NUM_THREADS": "1",
    "BLIS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "JULIA_NUM_THREADS": "1",
    "GOTO_NUM_THREADS": "1",
    "TBB_NUM_THREADS": "1",
    "TOKIO_WORKER_THREADS": "1",
    "POLARS_MAX_THREADS": "1",
    "OMP_DYNAMIC": "FALSE",
    "MKL_DYNAMIC": "FALSE",
}


@dataclass(frozen=True)
class CellSpec:
    cell: str
    driver: str
    budget: int
    reference: float
    n_molecules: int = 0


CELLS: tuple[CellSpec, ...] = (
    CellSpec("tip4p8", "h2h_seed", 2_000_000, -305.5183221676, 8),
    CellSpec("lj38", "h2h_gmin_proof", 400_000, -173.928427),
)
CELL_BY_NAME = {spec.cell: spec for spec in CELLS}
DRIVERS = tuple(sorted({spec.driver for spec in CELLS}))

ROW_COLUMNS: tuple[str, ...] = (
    "cell", "arm", "seed", "slot", "hit", "best", "best_bits",
    "charged", "evals_to_hit", "budget", "wall_s", "pot_s",
    "outside_s", "outside_per_charged_s", "core",
    "machine_id_sha256", "arm_commit", "arm_tree", "binary_sha256",
    "session_id", "status", "attempt", "exit_code", "rss_anon_kb",
    "vm_hwm_kb", "ru_maxrss_kb", "basis_walks", "basis_evidence",
    "basis_claims", "basis_probes", "basis_candidates", "basis_spent",
    "pad_bytes", "log_sha256",
)
