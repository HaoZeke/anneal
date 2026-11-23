"""Cell panel and row schema for the lazy-basin gate."""

from __future__ import annotations

from dataclasses import dataclass

BASE_COMMIT = "364679f060386949f805a72dea9d7d63b7e9ea4f"
BASE_TREE = "79610c980d39f19a2d383ffe3c53e4505ddb9b6c"
PR_HEAD = "6b12b9aab1e78589f0634413aae8c12611104607"
LAZY_COMMIT = "5bf74404ebc21751e95ac3d5efe1d74f3f58da04"
LAZY_TREE = "0467d399182a809d06661f1ae1903553833b1e8e"
B_TREE = "9d40e137e4f20cd0a8401b40b0b2f06f7521716e"
B0_TREE = "be6101476d3999e9232e71016b7871c17ca52a4d"
SESSION = "lazy-basin-gate"
HIT_TOL = 1e-3
TIMED_PAD_BYTES = 1
OUTSIDE_BAR = 1.10
BOOTSTRAP_DRAWS = 100_000

# Six-seed block. A and B alternate in ABBA order across seed pairs.
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
    n_seeds: int
    n_molecules: int = 0

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(range(2000, 2000 + self.n_seeds))


# Fast cells first so a partial seal still contains finished cells.
# lj100 is not a harness cell. LJ from lj75 through lj150 is the six
# cells the harness panel actually contains.
CELLS: tuple[CellSpec, ...] = (
    CellSpec("tip4p8", "h2h_seed", 2_000_000, -305.5183221676, 24, 8),
    CellSpec("lj38", "h2h_gmin_proof", 400_000, -173.928427, 24),
    CellSpec("m38r14", "h2h_morse", 400_000, -144.321054, 12),
    CellSpec("lj75", "h2h_gmin_proof", 4_000_000, -397.492331, 12),
    CellSpec("lj98", "h2h_gmin_proof", 4_000_000, -543.665361, 12),
    CellSpec("lj102", "h2h_gmin_proof", 4_000_000, -569.363652, 12),
    CellSpec("lj103", "h2h_gmin_proof", 4_000_000, -575.766131, 12),
    CellSpec("lj104", "h2h_gmin_proof", 4_000_000, -582.086642, 12),
    CellSpec("lj150", "h2h_gmin_proof", 4_000_000, -893.310258, 12),
    CellSpec("m80r10", "h2h_morse", 4_000_000, -344.911381, 12),
    CellSpec("m80r14", "h2h_morse", 4_000_000, -340.811371, 12),
)
CELL_BY_NAME = {spec.cell: spec for spec in CELLS}
DRIVERS = ("h2h_gmin_proof", "h2h_morse", "h2h_seed")
PANEL_ARMS = ("A", "A2", "B")
IDENTITY_ARM = "B0"

ROW_COLUMNS: tuple[str, ...] = (
    "cell", "arm", "seed", "slot", "hit", "best", "best_bits",
    "charged", "evals_to_hit", "budget", "wall_s", "pot_s",
    "outside_s", "outside_per_charged_s", "core",
    "machine_id_sha256", "arm_commit", "arm_tree", "binary_sha256",
    "session_id", "status", "attempt", "exit_code", "rss_anon_kb",
    "vm_hwm_kb", "ru_maxrss_kb", "basis_walks", "basis_evidence",
    "basis_claims", "basis_probes", "basis_candidates", "basis_spent",
    "pad_bytes", "log_sha256", "struct_sha256", "final_struct_sha256",
    "final_energy_bits",
)
