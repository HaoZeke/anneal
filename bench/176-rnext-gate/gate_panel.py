# Copyright (c) anneal developers
"""Cell panel and row schema for the 176 r-next default-on gate."""

from __future__ import annotations

from dataclasses import dataclass

BASE_COMMIT = "e25221f3a0bf4f78a67d4b7877d38f813b002ef5"
BASE_TREE = "cc5c9c3c4062db5e4afaaf4c007cabc8965851cc"
EXPECTED_B_TREE = "c9f9914d243a9b362513f49ffae749f4ef980e79"
SESSION_R1 = "gate-176-rnext-s1"
SESSION_R2 = "gate-176-rnext-r2"
SEEDS: tuple[int, ...] = tuple(range(2000, 2012))
PLACEMENT_SEED = 2000
HIT_TOL = 1e-3
REPLACEMENT_CAP = 2
RSS_ABS_ALLOW_KB = 8
OUTSIDE_FAIL_RATIO = 1.10
OUTSIDE_AP_LO = 0.95
OUTSIDE_AP_HI = 1.05
MW_ALPHA = 0.05
TIMED_PAD_BYTES = 1

# Six-seed block repeated twice across 2000..2011.
# Every arm is in slots 1, 2, and 3 four times per cell.
# Consecutive seeds alternate A before B, then B before A.
ARM_ROTATION: tuple[tuple[str, str, str], ...] = (
    ("A", "B", "A'"),
    ("B", "A'", "A"),
    ("A'", "A", "B"),
    ("B", "A", "A'"),
    ("A", "A'", "B"),
    ("A'", "B", "A"),
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

# A rebuilt five times. Only the linked padding changes.
PLACEMENT_PADS: tuple[tuple[str, int], ...] = (
    ("P1", 4096),
    ("P2", 8192),
    ("P3", 16384),
    ("P4", 32768),
    ("P5", 65536),
)

DRIVERS: tuple[str, ...] = (
    "h2h_gmin_proof",
    "h2h_morse",
    "h2h_sc",
    "h2h_metal",
    "h2h_seed",
    "h2h_blj",
)


@dataclass(frozen=True)
class CellSpec:
    cell: str
    family: str
    driver: str
    budget: int
    reference: float
    n_molecules: int = 0


CELLS: tuple[CellSpec, ...] = (
    CellSpec("lj38", "lj", "h2h_gmin_proof", 400_000, -173.928427),
    CellSpec("lj55", "lj", "h2h_gmin_proof", 1_000_000, -279.248470),
    CellSpec("lj75", "lj", "h2h_gmin_proof", 4_000_000, -397.492331),
    CellSpec("lj98", "lj", "h2h_gmin_proof", 4_000_000, -543.665361),
    CellSpec("lj102", "lj", "h2h_gmin_proof", 4_000_000, -569.363652),
    CellSpec("lj103", "lj", "h2h_gmin_proof", 4_000_000, -575.766131),
    CellSpec("lj104", "lj", "h2h_gmin_proof", 4_000_000, -582.086642),
    CellSpec("lj150", "lj", "h2h_gmin_proof", 4_000_000, -893.310258),
    CellSpec("blj105n38", "blj", "h2h_blj", 400_000, -177.2090846748),
    CellSpec("blj105n55", "blj", "h2h_blj", 1_000_000, -283.1545313810),
    CellSpec("blj115n38", "blj", "h2h_blj", 400_000, -187.3676867453),
    CellSpec("blj115n55", "blj", "h2h_blj", 1_000_000, -289.2931853058),
    CellSpec("m30r6", "morse", "h2h_morse", 400_000, -118.432844),
    CellSpec("m30r14", "morse", "h2h_morse", 400_000, -106.835790),
    CellSpec("m38r6", "morse", "h2h_morse", 400_000, -157.477108),
    CellSpec("m38r14", "morse", "h2h_morse", 400_000, -144.321054),
    CellSpec("m55r6", "morse", "h2h_morse", 1_000_000, -250.286609),
    CellSpec("m55r14", "morse", "h2h_morse", 1_000_000, -220.646208),
    CellSpec("m70r3", "morse", "h2h_morse", 1_000_000, -577.739782),
    CellSpec("m70r6", "morse", "h2h_morse", 1_000_000, -325.887749),
    CellSpec("m70r10", "morse", "h2h_morse", 1_000_000, -296.412149),
    CellSpec("m70r14", "morse", "h2h_morse", 1_000_000, -292.462856),
    CellSpec("m80r3", "morse", "h2h_morse", 4_000_000, -690.577890),
    CellSpec("m80r6", "morse", "h2h_morse", 4_000_000, -378.333471),
    CellSpec("m80r10", "morse", "h2h_morse", 4_000_000, -344.911381),
    CellSpec("m80r14", "morse", "h2h_morse", 4_000_000, -340.811371),
    CellSpec("sc126-38", "sc", "h2h_sc", 400_000, -35419.9804),
    CellSpec("sc126-55", "sc", "h2h_sc", 1_000_000, -52884.6806),
    CellSpec("au55", "metal", "h2h_metal", 1_000_000, -191.03085280),
    CellSpec("tip4p8", "tip4p", "h2h_seed", 2_000_000, -305.5183221676, 8),
    CellSpec("tip4p12", "tip4p", "h2h_seed", 2_000_000, -492.9082357122, 12),
    CellSpec("tip4p15", "tip4p", "h2h_seed", 2_000_000, -628.3722525077, 15),
    CellSpec("tip4p16", "tip4p", "h2h_seed", 2_000_000, -681.1925945898, 16),
    CellSpec("tip4p20", "tip4p", "h2h_seed", 2_000_000, -872.9887544952, 20),
    CellSpec("tip4p21", "tip4p", "h2h_seed", 2_000_000, -916.7074302940, 21),
)

CELL_BY_NAME: dict[str, CellSpec] = {spec.cell: spec for spec in CELLS}

# Fastest round-1 median wall first. Tables stay in CELLS order.
RUN_ORDER: tuple[str, ...] = (
    "tip4p8",
    "m30r6",
    "lj55",
    "tip4p12",
    "sc126-55",
    "m55r6",
    "m38r6",
    "lj38",
    "blj115n38",
    "sc126-38",
    "m30r14",
    "m38r14",
    "m70r6",
    "m80r6",
    "blj105n55",
    "blj105n38",
    "m55r14",
    "m70r14",
    "tip4p15",
    "m70r3",
    "tip4p16",
    "tip4p20",
    "tip4p21",
    "m70r10",
    "m80r3",
    "blj115n55",
    "au55",
    "lj75",
    "lj150",
    "lj98",
    "lj102",
    "lj103",
    "lj104",
    "m80r10",
    "m80r14",
)

ROW_COLUMNS: tuple[str, ...] = (
    "cell",
    "arm",
    "seed",
    "slot",
    "hit",
    "best",
    "best_bits",
    "charged",
    "evals_to_hit",
    "budget",
    "wall_s",
    "pot_s",
    "outside_s",
    "outside_per_charged_s",
    "core",
    "machine_id_sha256",
    "arm_commit",
    "arm_tree",
    "binary_sha256",
    "session_id",
    "status",
    "superseded",
    "attempt",
    "exit_code",
    "rss_anon_kb",
    "vm_hwm_kb",
    "ru_maxrss_kb",
    "basis_walks",
    "basis_evidence",
    "basis_claims",
    "basis_probes",
    "basis_candidates",
    "basis_spent",
    "pad_bytes",
    "log_sha256",
)

SUMMARY_COLUMNS: tuple[str, ...] = (
    "cell",
    "family",
    "budget",
    "hits_a",
    "hits_ap",
    "hits_b",
    "fisher_p",
    "fisher_loss_p",
    "hit_change",
    "median_calls_a",
    "median_calls_ap",
    "median_calls_b",
    "mw_p",
    "mw_worse_p",
    "calls_b_a",
    "calls_ap_a",
    "outside_med_a_s",
    "outside_med_ap_s",
    "outside_med_b_s",
    "outside_b_a",
    "outside_ap_a",
    "rss_med_a_kb",
    "rss_med_ap_kb",
    "rss_med_b_kb",
    "rss_delta_kb",
    "rss_aa_kb",
    "rss_place_kb",
    "rss_band_kb",
    "rss_excess_kb",
    "rss_two_sided",
    "vm_hwm_med_a_kb",
    "vm_hwm_med_b_kb",
    "ru_maxrss_med_a_kb",
    "ru_maxrss_med_b_kb",
    "basis_walks_min_b",
    "basis_evidence_min_b",
    "basis_claims_med_b",
    "basis_probes_med_b",
    "basis_candidates_med_b",
    "basis_spent_med_b",
    "replacements",
    "verdict",
    "reason",
)

TIMED_ARMS: tuple[str, ...] = ("A", "A'", "B")


def pad_bytes_for(arm: str) -> int:
    if arm in ("A", "A'", "B"):
        return TIMED_PAD_BYTES
    for name, size in PLACEMENT_PADS:
        if name == arm:
            return size
    raise KeyError(arm)
