# ASE and AGOX Lennard-Jones board.
# One process: snakemake -s workflow/lj_board.smk -j 1 lj_numbers
# Sizes default to the paper set. Seeds are the frozen held-out list.
# The budget for a size is the paper ceiling inside run_arm.py.
# This kit does not run the anneal or GMIN arms.

import pathlib

_SEED_FILE = pathlib.Path("experiments/lj_held_out_seeds.txt")
LJ_SEEDS = []
for _line in _SEED_FILE.read_text().splitlines():
    _line = _line.split("#", 1)[0].strip()
    if _line:
        LJ_SEEDS.append(_line)

LJ_SIZES = [
    item.strip()
    for item in str(config.get("sizes", "38,55,75,98,104")).split(",")
    if item.strip()
]
LJ_CODES = [
    item.strip()
    for item in str(config.get("codes", "ase,agox")).split(",")
    if item.strip()
]


rule lj_inventory:
    output:
        "results/lj-board/provenance/inventory.tsv",
    params:
        sizes=LJ_SIZES,
        seeds=LJ_SEEDS,
        codes=LJ_CODES,
    shell:
        """
        mkdir -p results/lj-board/provenance
        printf 'code\\tsize\\tseed\\tbudget_source\\n' > {output}
        for code in {params.codes}; do
          for size in {params.sizes}; do
            for seed in {params.seeds}; do
              printf '%s\\t%s\\t%s\\tpaper\\n' "$code" "$size" "$seed" >> {output}
            done
          done
        done
        """


rule lj_one:
    input:
        "results/lj-board/provenance/inventory.tsv",
    output:
        "results/lj-board/{code}/lj{n}-seed{seed}.log",
    threads: 1
    shell:
        """
        set -euo pipefail
        export OMP_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1
        export NUMBA_NUM_THREADS=1
        export VECLIB_MAXIMUM_THREADS=1
        export BLIS_NUM_THREADS=1
        export OMP_PROC_BIND=TRUE
        export OMP_PLACES=cores
        export LJ_BOARD_ROOT=results/lj-board
        python3 scripts/lj_board/run_arm.py {wildcards.code} {wildcards.n} {wildcards.seed}
        """


rule lj_numbers:
    input:
        expand(
            "results/lj-board/{code}/lj{n}-seed{seed}.log",
            code=LJ_CODES,
            n=LJ_SIZES,
            seed=LJ_SEEDS,
        ),
    output:
        "results/lj-board/NUMBERS.tsv",
    shell:
        "python3 scripts/lj_board/summarize.py {input} > {output}"
