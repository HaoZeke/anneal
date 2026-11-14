# Deposited Guided Adaptive Diffusion on LJ31 and LJ38

The GAD arm is the Lennard-Jones evolution files in Zenodo [10.5281/zenodo.19001618](https://doi.org/10.5281/zenodo.19001618), the deposit for Adel, Szmitek, Hartl, Wanzenböck, and Madsen, Journal of Chemical Information and Modeling 66, 7414–7425 (2026), doi:10.1021/acs.jcim.6c00843. Those files are figures from the authors' runs. This bench does not rerun the diffusion search, and it does not train or load a model. The zip is downloaded by the Snakemake rule into `/work` and is not committed. Its md5 is `e140fde0c62c71f08b326b12ac50d77e`.

## Counting

Each of `lj-31/lj-31_evolution.h5` and `lj-38/lj-38_evolution.h5` contains one run, `run_0`, with generations `gen_0` through `gen_999`. A generation stores `fitness` for 64 samples, the coordinates in `samples`, and `model_loss`. The objective says `maximize` is true, so the potential energy is the negation of `fitness`. `model_loss` is the diffusion training loss. It is not an energy evaluation.

The HDF5 files do not record how many FIRE steps a sample took, and the Lennard-Jones XYZ files are one frame each with no step count. The only step number in the archive is the FIRE cap in the relaxer string, `AtomsRelaxer(..., steps=1000, ...)`. Charging that cap for every sample is an upper bound. It overstates the cost whenever FIRE stopped early. The lower bound charges one evaluation per sample and treats the relaxation as free. Through the generation that first reaches the Cambridge energy, including that generation,

    lower = generations × 64
    upper = generations × 64 × 1000

The same 0.001 band is the hit rule for anneal and GMIN.

Anneal is `Config::recommended` from `examples/gad_published`. GMIN is Wales's public tarball at `aac1548a8aec13b53ad12a8d6904625081685d2e`, with the evaluator cap in `bench/lj104-lattice/v2/h2h-gmin.patch`. Both count one entry to the potential, including every evaluation inside a local relaxation. Both stop at 400000 calls. That ceiling is the LJ38 entry of the elja map. LJ31 is not on the map; this bench uses the same ceiling so the two sizes share one force-call budget. A rerun that does not reach the Cambridge energy is stored as 400000 calls. Seeds are 2000 through 2011, one process at a time, under `sudo unshare --uts` with hostname `bench`, pinned to core 0.

`reached` counts runs whose best energy is inside the 0.001 band. Anneal and GMIN `hits` count runs that reach it at or before 400000 calls. A miss is stored as 400000. The GAD rows have no hit column. They report the generation index at which the deposit first reaches Cambridge, and both evaluation bounds. Fisher's exact test for a GAD bound is two-sided on one question: the bound is at or below 400000 and the run reached Cambridge, against the rerun's hits. Mann–Whitney for that bound is two-sided on the bound against the rerun's evaluation counts. The comparison is exact when the pooled sample has no ties, and the tie-corrected normal approximation otherwise. The columns `fisher_p` and `mannwhitney_p` on the GMIN row compare GMIN's own call counts with anneal. They do not use a GAD bound. Medians and the interquartile range are Hyndman–Fan type 7. Inside microseconds are potential time divided by charged calls. Outside microseconds are the rest of the wall on the same denominator. The deposit has no wall clock, so those columns are empty.

Cambridge energies are −133.586422 for LJ31 and −173.928427 for LJ38.

## Results

One deposited run per size. Anneal and GMIN are twelve seeds, 2000–2011. Anneal is `Config::recommended` built on `f2a628c0157bd004c997549f15ba06c0b023abb3`. Generation indexes count from zero. LJ31 reaches Cambridge at generation 140, which is 141 generations, so the bounds are 141 × 64 = 9024 and 141 × 64 × 1000 = 9024000. LJ38 reaches it at generation 482, which is 483 generations, so the bounds are 483 × 64 = 30912 and 483 × 64 × 1000 = 30912000. The lower-bound tests use 9024 and 30912. The upper-bound tests use 9024000 and 30912000. Empty cells have no clock.

| Size | Arm | Source | Reached | Generation | Lower bound | Upper bound | Hits | Median evaluations | IQR | Best energy | Median wall (s) | Inside µs | Outside µs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LJ31 | gad | deposited | 1/1 | 140 | 9024 | 9024000 | | | | −133.58642190 | | | |
| LJ31 | anneal | rerun | 12/12 | | | | 12/12 | 51089.5 | 50873.5 | −133.58642192 | 0.711720 | 8.2749 | 5.1841 |
| LJ31 | gmin | rerun | 12/12 | | | | 12/12 | 87151 | 94385.5 | −133.58641251 | 0.637458 | 4.9851 | 2.2646 |
| LJ38 | gad | deposited | 1/1 | 482 | 30912 | 30912000 | | | | −173.92842658 | | | |
| LJ38 | anneal | rerun | 12/12 | | | | 12/12 | 13469.5 | 3990.75 | −173.92842659 | 0.224667 | 11.9589 | 4.8309 |
| LJ38 | gmin | rerun | 7/12 | | | | 7/12 | 389399 | 173127.75 | −173.92838636 | 3.792015 | 7.1525 | 2.5612 |

Fisher and Mann–Whitney for the deposited bounds:

| Size | Bound | Against | Fisher p | Mann–Whitney p |
| --- | --- | --- | --- | --- |
| LJ31 | lower, 9024 | anneal | 1 | 0.307692 |
| LJ31 | upper, 9024000 | anneal | 0.0769231 | 0.153846 |
| LJ31 | lower, 9024 | gmin | 1 | 0.461538 |
| LJ31 | upper, 9024000 | gmin | 0.0769231 | 0.153846 |
| LJ38 | lower, 30912 | anneal | 1 | 0.153846 |
| LJ38 | upper, 30912000 | anneal | 0.0769231 | 0.153846 |
| LJ38 | lower, 30912 | gmin | 1 | 0.216034 |
| LJ38 | upper, 30912000 | gmin | 0.461538 | 0.130518 |

GMIN against anneal, on the call counts the reruns recorded: LJ31 Fisher p = 1, Mann–Whitney p = 0.218921. LJ38 Fisher p = 0.0372671, Mann–Whitney p = 3.38156e-05.

`summary.csv` is the same table. The anneal binary was built at `238f35825457b10d839e57d40ade9a93c7482ce7`. GMIN is revision `aac1548a8aec13b53ad12a8d6904625081685d2e`.
