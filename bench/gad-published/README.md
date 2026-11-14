# Deposited Guided Adaptive Diffusion on LJ31 and LJ38

The GAD arm is the Lennard-Jones evolution files in Zenodo [10.5281/zenodo.19001618](https://doi.org/10.5281/zenodo.19001618), the deposit for Adel, Szmitek, Hartl, Wanzenböck, and Madsen, Journal of Chemical Information and Modeling 66, 7414–7425 (2026), doi:10.1021/acs.jcim.6c00843. Those files are figures from the authors' runs. This bench does not rerun the diffusion search, and it does not train or load a model. The zip is downloaded by the Snakemake rule into `/work` and is not committed. Its md5 is `e140fde0c62c71f08b326b12ac50d77e`.

## Counting

Each of `lj-31/lj-31_evolution.h5` and `lj-38/lj-38_evolution.h5` contains one run, `run_0`, with generations `gen_0` through `gen_999`. A generation stores `fitness` for 64 samples, the coordinates in `samples`, and `model_loss`. `model_loss` is the diffusion training loss. It is not an energy evaluation. No dataset records how many FIRE steps a sample took. The run attribute `es_config` records the relaxer as `AtomsRelaxer(..., steps=1000, ...)`, and the objective says `maximize` is true, so the potential energy is the negation of `fitness`.

An energy evaluation in these files is one step of that recorded cap:

    generations × samples × recorded relaxation steps
    = generations × 64 × 1000

The generations in that product are those up to and including the first generation whose best energy is within 0.001 of the Cambridge value. The best energy is the lowest energy stored in the file. The same 0.001 band is the hit rule for anneal and GMIN.

Anneal is `Config::recommended` from `examples/gad_published`. GMIN is Wales's public tarball at `aac1548a8aec13b53ad12a8d6904625081685d2e`, with the evaluator cap in `bench/lj104-lattice/v2/h2h-gmin.patch`. Both count one entry to the potential, including every evaluation inside a local relaxation. Both stop at 400000 calls. That ceiling is the LJ38 entry of the elja map. LJ31 is not on the map; this bench uses the same ceiling so the two sizes share one force-call budget. A rerun that does not reach the Cambridge energy is stored as 400000 calls. Seeds are 2000 through 2011, one process at a time, under `sudo unshare --uts` with hostname `bench`, pinned to core 0.

`reached` counts runs whose best energy is inside the 0.001 band. For the deposit that can take more calls than the ceiling. `hits` counts runs that are inside the band at or before 400000 calls. Fisher's exact test is two-sided on `hits` against the anneal arm. Mann–Whitney is two-sided on the evaluations to the Cambridge energy. Medians and the interquartile range are Hyndman–Fan type 7. Inside microseconds are potential time divided by charged calls. Outside microseconds are the rest of the wall on the same denominator. The deposit has no wall clock, so those columns are empty. Model inference is not in the deposit and is not added to the evaluation count.

Cambridge energies are −133.586422 for LJ31 and −173.928427 for LJ38.

## Results

The table is `summary.csv`, written by `scripts/gad_published/enter_bench.sh`.
