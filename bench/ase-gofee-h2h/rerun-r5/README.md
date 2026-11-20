# GOFEE rerun

This directory records a new run of the two GOFEE tables. The search
code is commit `3d005d2485dc5c7ad91b3bbd4854040cf2b38aca`, tree
`b12293b9d637dd32417be3aa482302e0f30dd594`. The rules below were fixed
before any seed of this run.

## Systems, arms, and seeds

Nine systems, three arms, seeds 2000 through 2011.

| System | Id | Budget | Reference |
| --- | --- | --- | --- |
| LJ38 | lj38 | 400000 | -173.928427 |
| LJ55 | lj55 | 1000000 | -279.248470 |
| LJ75 | lj75 | 4000000 | -397.492331 |
| Au55 | au55 | 1000000 | -191.03085280 |
| Cu38 | cu38 | 400000 | -108.96759055 |
| Cu12Au26 | cu12au26 | 400000 | -127.21076866 |
| Si10 | sw10 | 100000 | -13.79658026 |
| C10 | tc10 | 100000 | -55.08979464 |
| (H2O)8 | tip4p8 | 2000000 | -305.5183221676 |

LJ38, LJ55, LJ75, and Au55 take the budgets in
`scripts/gmin_panel/run_panel.py`. The other five are not in that map.
They take the budgets printed by the evaluator and recorded in
`bench/ase-gofee-h2h/benchmarks/ase_gofee_h2h.py`: 400000 for Cu38 and
Cu12Au26, 100000 for Si10 and C10, and 2000000 for (H2O)8.

The arms are `anneal`, `ase_mh` (ASE minima hopping at the 3.29.0
constructor defaults), and `agox_gofee` (AGOX 3.11.1, the search in
`scripts/ase_gofee_h2h`). Basin hopping is not an arm of this run.
Versions are ASE 3.29.0, AGOX 3.11.1, numpy 2.2.6, and numba 0.61.2.
`pip-freeze.txt` is the environment that ran.

Jobs run one at a time, in system order as in the table, and within a
system in seed order. The three arms of a seed run in an order rotated
by the seed: seed 2000 starts at anneal, seed 2001 at minima hopping,
seed 2002 at GOFEE, and then the rotation repeats. The process is
pinned to cpu 0. `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, `RAYON_NUM_THREADS`, `NUMBA_NUM_THREADS`,
`VECLIB_MAXIMUM_THREADS`, and `BLIS_NUM_THREADS` are 1. The search
library can start a helper process on that same core. The model is
built with its helper flag off, `RAY_AUTH_MODE` is `disabled`, and the
helper's temporary files stay in the run directory. The helper is shut
down after the row is written.

## Hit rule

A hit is a quenched energy at or below the reference plus 0.001.
Non-finite energies are misses. The comparison is
`scripts/target_tol.py`.

## Stop rules

Anneal runs until a hit or until the call budget is spent. It has no
wall cap. Minima hopping and GOFEE on every system, including LJ38 and
LJ55, also stop if the wall reaches 7200 s. That stop is recorded as
`wall`. A wall stop is a miss: `hit` is 0, `evals_to_hit` is blank, and
the row keeps the calls completed, the best energy, and the wall.
`budget` is a miss that spent the call budget. `hit` is a stop on the
hit rule. The budget column stays the call budget in every case.

## What each arm runs

Lennard-Jones anneal is `h2h_gmin_proof`. Au55 anneal is `h2h_metal`.
Cu38 anneal is `h2h_metal` on the composition key `cu38au0`, which is
`Gupta::copper_gold` for 38 copper atoms. The named `cu38` entry of
that example is a different potential, so it is not used. Its compiled
reference is unset, so that one process is given `H2H_REFERENCE` equal
to -108.96759055. Cu12Au26 anneal is `h2h_metal` on `cu12au26` with
`H2H_REFERENCE` equal to -127.21076866, because that composition's
compiled reference is unset. No other `H2H_` or `ANNEAL_` variable is
set. Si10 and C10 anneal are `h2h_covalent` with the recommended arm.
(H2O)8 anneal is `h2h_seed` on rigid TIP4P, eight molecules, the
recommended arm, and a stop at the reference plus 0.001. The stored
best for that arm is the search outcome's best, and the run is kept
only when that value and the example's own hit flag agree.

Minima hopping and GOFEE call the functions in
`scripts/ase_gofee_h2h`. Lennard-Jones energies are the Python pair
checked against `h2h_gmin_proof`. The other six systems evaluate
through `asg_eval serve`. The covalent library is the one built by
`scripts/covalent_h2h/build_rgpot_covalent.sh` at rgpot
`20c58fe5a2859cd0a79a3e06d64d567f96d0ccbe`.

For Lennard-Jones, gold, copper-gold, silicon, and carbon, the anneal
`coords` command writes one start per system and seed. Minima hopping
reads that file. Those starts match `asg_eval start` on every seed
from 2000 to 2011 for Au55, Cu38, Cu12Au26, Si10, and C10. GOFEE, as
that search is written, draws its own candidates and does not read the
file. (H2O)8 anneal draws its start inside `h2h_seed`, which has no
coordinate argument. Minima hopping for (H2O)8 reads `asg_eval start`.
Both use the same seed formula.

## Rows and tests

`rows.csv` gains one row after each job and is not rewritten. The
columns are `system`, `method`, `seed`, `budget`, `hit`, `best`,
`evals_to_hit`, `charged`, `wall_s`, `stop`, `revision`,
`binary_sha256`, `machine_id_sha256`. `revision` is the code commit
above. `evals_to_hit` is blank on a miss. `binary_sha256` is the anneal
binary that searched, or, for minima hopping and GOFEE, the binary
that defines the potential (`h2h_gmin_proof` on Lennard-Jones,
`asg_eval` otherwise).

`summarize.py` writes `summary.csv` from the rows. For each system and
each of the two other arms it reports Fisher's exact test on the hit
counts against anneal, and the Mann-Whitney test on calls to the hit.
A miss is ranked at positive infinity. Both tests are the functions in
`scripts/gmin_panel/run_panel.py`: the two-sided exact Fisher test, and
the two-sided Mann-Whitney normal approximation with the tie and
continuity corrections. The Mann-Whitney U is the other arm's U against
anneal. Medians are Hyndman-Fan type 7. The calls median is over hits
only. The best and wall medians are over all twelve seeds.

`hosts.csv` is the machine record: the sha256 of the machine id, the
architecture, the kernel, the processor, the cpu count, and `MemTotal`.
It does not name the machine.

## Binaries

| Binary | sha256 |
| --- | --- |
| h2h_gmin_proof | `aa42b5d60c1de031a5fdce8f620a9ae52ded46acf8ce35d6f776e70dd5891d8f` |
| h2h_morse | `e3ec53040b4e8df5977d08de4add6c452c70b17abe6433a666e45db4754e9cba` |
| h2h_metal | `f78d5045be0d2f1d4694f7f1fa5077f47c6328058aa95505c95b8b16f3a08f66` |
| h2h_covalent | `57d6454d3e112373beb8c5e04716a314f853db9652619f53d913dd91456427d5` |
| h2h_seed | `83e9b9a0ceceb68eddf6d89bb6d1e5d9e432f29e11b5c3a652fec9643885844d` |
| asg_eval | `a73bf4f900cb0be6f48b6723634c091afe61eebd9f42c4dc35fd63ff2042a2aa` |

`h2h_morse` is not used by these nine systems. It is recorded because
the build that produced the other binaries built it as well. The runner
refuses a binary whose sha256 is not in this table.
