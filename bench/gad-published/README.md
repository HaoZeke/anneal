# Guided Adaptive Diffusion on LJ31 and LJ38

GAD is Adel, Szmitek, Hartl, Wanzenböck, and Madsen, Journal of Chemical Information and Modeling 66, 7414–7425 (2026), doi:10.1021/acs.jcim.6c00843, with its deposit at Zenodo [10.5281/zenodo.19001618](https://doi.org/10.5281/zenodo.19001618). The bench has two GAD arms. `gad` reruns the search at twelve seeds with `scripts/gad_published/gad_rerun.py`. `gad_deposit` reads the single run in the deposit. The zip is downloaded by the Snakemake rule into `/work` and is not committed. Its md5 is `e140fde0c62c71f08b326b12ac50d77e`.

## Rerun

`gad_rerun.py` drives CondEvo's CHARLES at `ae76bf0462de63fa6bf3bde97df7cc00237b8d1d` (github.com/bhartl/CondEvo) with the settings recorded in the deposit's run attributes:

- population 64, initial sigma 1.5;
- DDIM with 5000 steps on an MLP with 96 hidden units, 8 layers, LeakyReLU and 2 conditions;
- genetic selection with pressure 20 (adaptive), elite 0.15, crossover 0.125, mutation 0.05;
- Adam, learning rate 0.01, weight decay 1e-5, batch 256, 1000 epochs per generation, buffer 1000;
- an origin condition with radius 1.0016399 times the founder's largest atom distance from the origin, and a 10-nearest-neighbour novelty condition (beta 10, weighted by fitness);
- FIRE relaxation of every sample with fmax 0.001 and at most 1000 steps, the relaxed coordinates told back to the search.

The potential is the untruncated Lennard-Jones in reduced units. Every energy and force evaluation is one call. The founder is the seeded start that `examples/gad_published coords` writes, the same start anneal and GMIN use. `scripts/gad_published/build_gad.sh` builds the pinned venv (torch 2.15.0+cpu, ase 3.29.0, numpy 2.5.4) and the CondEvo checkout.

## Deposit counting

Each of `lj-31/lj-31_evolution.h5` and `lj-38/lj-38_evolution.h5` contains one run, `run_0`, with generations `gen_0` through `gen_999`. A generation stores `fitness` for 64 samples, the coordinates in `samples`, and `model_loss`. The objective says `maximize` is true, so the potential energy is the negation of `fitness`. `model_loss` is the diffusion training loss. It is not an energy evaluation.

The HDF5 files do not record how many FIRE steps a sample took, and the Lennard-Jones XYZ files are one frame each with no step count. The only step number in the archive is the FIRE cap in the relaxer string, `AtomsRelaxer(..., steps=1000, ...)`. Charging that cap for every sample is an upper bound. It overstates the cost whenever FIRE stopped early. The lower bound charges one evaluation per sample and treats the relaxation as free. Through the generation that first reaches the Cambridge energy, including that generation,

    lower = generations × 64
    upper = generations × 64 × 1000

The same 0.001 band is the hit rule for the GAD rerun, anneal and GMIN.

Anneal is `Config::recommended` from `examples/gad_published`. GMIN is Wales's public tarball at `aac1548a8aec13b53ad12a8d6904625081685d2e`, with the evaluator cap in `bench/lj104-lattice/v2/h2h-gmin.patch`. Both, like the GAD rerun, count one entry to the potential, including every evaluation inside a local relaxation. All three stop at 400000 calls. That ceiling is the LJ38 entry of the elja map. LJ31 is not on the map; this bench uses the same ceiling so the two sizes share one force-call budget. A rerun that does not reach the Cambridge energy is stored as 400000 calls. Seeds are 2000 through 2011, one process at a time, under `sudo unshare --uts` with hostname `bench`, pinned to one core (`GAD_BENCH_CORE`, core 4 for the rows below).

In `rows.csv`, `hits` counts reruns that reach the 0.001 band at or before 400000 calls, and a miss is stored as 400000. The `gad_deposit` rows have no hit column. They report the generation index at which the deposit first reaches Cambridge, and both evaluation bounds. On the `gmin` and `gad` rows, `fisher_p` is two-sided on hits against anneal and `mannwhitney_p` is two-sided on calls to the hit against anneal, exact when the pooled sample has no ties and the tie-corrected normal approximation otherwise. Medians and the interquartile range are Hyndman–Fan type 7. Inside microseconds are potential time divided by charged calls. Outside microseconds are the rest of the wall on the same denominator. The deposit has no wall clock, so those columns are empty. `summary.csv` keeps the earlier bound tests for the deposit.

Cambridge energies are −133.586422 for LJ31 and −173.928427 for LJ38.

## Results

Twelve seeds, 2000–2011, for anneal, GMIN and the GAD rerun, all at 400000 calls, serial on core 4 under `sudo unshare --uts` with hostname `bench`. Anneal is `Config::recommended` built at `48ecd4c0f348bc40bb6120a92a9e5bc5b124e142`, whose Rust sources are those of `91f835fe951d5bcaa13e43c1a3fb98d6887da9e0`. A miss counts as 400000 calls. Fisher is two-sided on hits against anneal, Mann–Whitney two-sided on calls to the hit against anneal. Pot is median wall inside the potential, outside is the rest of the wall.

| Size | Arm | Hits | Median calls to hit | IQR | Best (median) | Wall (s) | Pot (s) | Outside (s) | Inside µs/call | Outside µs/call | Fisher p | Mann–Whitney p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| LJ31 | anneal | 12/12 | 18666 | 29896.8 | −133.58642192 | 0.107647 | 0.032629 | 0.073946 | 1.5603 | 3.4399 | | |
| LJ31 | gmin | 12/12 | 90210.5 | 102713.8 | −133.58640476 | 0.486330 | 0.307743 | 0.187484 | 3.1514 | 1.7074 | 1 | 0.0172712 |
| LJ31 | gad | 2/12 | 400000 | 0 | −133.10462015 | 187.522859 | 23.479254 | 163.718770 | 59.3887 | 411.3383 | 6.73038e-05 | 1.83067e-05 |
| LJ38 | anneal | 12/12 | 10615.5 | 3591.2 | −173.92842659 | 0.069455 | 0.029011 | 0.042200 | 2.4370 | 3.2839 | | |
| LJ38 | gmin | 7/12 | 329662 | 326680 | −173.92837772 | 1.802020 | 1.276786 | 0.525234 | 3.8734 | 1.5935 | 0.0372671 | 0.000693135 |
| LJ38 | gad | 0/12 | 400000 | 0 | −171.77778794 | 195.144155 | 31.069388 | 164.491350 | 77.6735 | 411.2284 | 7.39602e-07 | 1.02689e-05 |

The per-seed rerun log is `seed-reruns.csv`, recorded at the verified source alias `407731c5c3983cc927966cd308cbf23975bb0ffb` with hostname `bench`. The two LJ31 GAD hits are seed 2000 at 226282 calls and seed 2006 at 114143 calls. Within the budget GAD completed a median of 13 generations on LJ31 (2–20) and 15.5 on LJ38 (12–24).

The deposited run reaches Cambridge at generation 140 for LJ31 (bounds 9024 and 9024000 calls) and generation 482 for LJ38 (bounds 30912 and 30912000 calls).

`rows.csv` holds these rows with `code_revision`, `host`, the GMIN revision `aac1548a8aec13b53ad12a8d6904625081685d2e` and the CondEvo commit. The runner appends to it and refuses a file with another header. `summary.csv` is the earlier deposit-only table, built at `238f35825457b10d839e57d40ade9a93c7482ce7`.
