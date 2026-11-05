# Weak scaling for parallel tempering and the population methods.
# Run one cell at a time so the threads are not sharing the host:
#   snakemake -s workflow/weak_scale.smk -j 1 --cores 1

FAMILIES = ["pt", "population", "dmc"]
WORKERS = [1, 2, 4, 8, 16]
SEEDS = list(range(6))


rule all:
    input:
        "results/weak_scale/table.md",
        "results/weak_scale/summary.csv",
        "results/weak_scale/asv/tracks.csv",


rule build_weak_scale:
    output:
        "../tmp/anneal-target/release/examples/weak_scale",
    shell:
        "cargo +1.88.0 build --release --example weak_scale"


rule weak_scale_cell:
    input:
        "../tmp/anneal-target/release/examples/weak_scale",
    output:
        "results/weak_scale/raw/{family}/w{workers}/seed{seed}.log",
    resources:
        slots=1,
    shell:
        "bash scripts/weak_scale_cell.sh {wildcards.family} {wildcards.workers} {wildcards.seed} {output}"


rule weak_scale_table:
    input:
        expand(
            "results/weak_scale/raw/{family}/w{workers}/seed{seed}.log",
            family=FAMILIES,
            workers=WORKERS,
            seed=SEEDS,
        ),
    output:
        "results/weak_scale/table.md",
        "results/weak_scale/summary.csv",
    shell:
        "python3 scripts/weak_scale_summarize.py"


rule weak_scale_asv:
    input:
        "results/weak_scale/summary.csv",
    output:
        "results/weak_scale/asv/tracks.csv",
    shell:
        "python3 scripts/weak_scale_asv_deposit.py"
