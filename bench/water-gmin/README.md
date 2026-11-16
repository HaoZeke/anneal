# (H₂O)ₙ, anneal against GMIN, twelve seeds

Anneal develop `5d44d218`, default preset (`recommended`), against Wales's
GMIN rigid-body TIP4P basin-hopping at n = 8, 12, 16, 20 and 21. GMIN is
`scripts/gmin_h2h/build_gmin.sh` at revision `aac1548a`, NEWTIP, TIPID 4,
temperature 2 kJ/mol and step 0.40 / 0.51 held fixed, the Lennard-Jones
container (`examples/h2h_gmin_tip4p.data`), NEWTIP timed by
`scripts/tip4p_h2h/newtip-timer.patch`. Seeds 2000–2011, 2_000_000 charged
evaluations for both codes at every size, serial, `taskset -c 0`, one session
under `sudo unshare --uts` with hostname `bench`, working directory `/work`.
The driver is `scripts/tip4p_h2h/enter_water_gmin.sh`, the Snakemake rule is
`water_gmin` and the asv suite is `benchmarks/water_gmin.py`.

Both arms open the same molecular centres: `h2h_seed` writes the start it
draws for a seed and GMIN reads it as its `coords`; every row carries the
start's sha1. One charged evaluation is one entry to the potential. A run
stops at the Cambridge Cluster Database header plus 0.001 kJ/mol or at the
budget. Anneal scores the lowest energy it evaluates, GMIN the lowest quench.

`rows.csv` is appended to, one row per code and seed, with `code_revision`
(the anneal commit, or `gmin-` and the GMIN revision) and `host`. A stored
anneal row is reused only at the same `code_revision`; a GMIN row only at the
same revision and start sha1. No earlier anneal water rows are from
`5d44d218`, so every row here is from this session. `summary.csv` is built
from the current revisions' rows. Develop `92bc5805` changes the cluster
hopping search after these rows were taken; `water_gmin` at that or a later
commit runs the anneal arm again under its own `code_revision`.

## Both codes reproduce the Cambridge headers

Each code takes a single point at the published minimum, written as a NEWTIP
centre/angle-axis block. Rows are `verify.csv`.

| Size | Cambridge header (kJ/mol) | anneal − header | GMIN − header |
| --- | --- | --- | --- |
| (H₂O)₈ | −305.5183221676 | +1.5e-05 | −6.1e-04 |
| (H₂O)₁₂ | −492.9082357122 | −1.9e-05 | −1.0e-03 |
| (H₂O)₁₆ | −681.1925945898 | −5.0e-05 | −1.4e-03 |
| (H₂O)₂₀ | −872.9887544952 | −1.1e-05 | −1.8e-03 |
| (H₂O)₂₁ | −916.7074302940 | −9.7e-05 | −1.9e-03 |

GMIN's offset is its `DEFTIP4` Coulomb prefactor rounding and grows with the
pair count; it is below the 0.1 kJ/mol gap to the next minimum at every size.

## Panel

`calls` is the charged evaluation that first reached the target, or the
budget for a miss. The first calls column takes all twelve seeds, the second
hits only. `p(F)` is the two-sided Fisher exact test on hits, `p(U)` the
two-sided Mann–Whitney U test on calls. Times are medians in seconds;
`pot` is inside the potential, `out` the rest of the wall.

| Size | anneal | GMIN | anneal calls (all / hits) | GMIN calls (all / hits) | wall a / g | anneal pot / out | GMIN pot | p(F) | p(U) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| (H₂O)₈ | 12/12 | 8/12 | 6_339 / 6_339 | 240_163 / 90_822 | 0.020 / 1.65 | 0.012 / 0.008 | 1.32 | 0.093 | 3.1e-05 |
| (H₂O)₁₂ | 12/12 | 5/12 | 41_418.5 / 41_418.5 | 2_000_000 / 976_528 | 0.229 / 26.9 | 0.150 / 0.078 | 23.6 | 0.0046 | 7.0e-05 |
| (H₂O)₁₆ | 2/12 | 0/12 | 2_000_000 / 275_150.5 | 2_000_000 / – | 15.9 / 45.2 | 11.7 / 4.16 | 41.2 | 0.48 | 0.15 |
| (H₂O)₂₀ | 6/12 | 0/12 | 1_686_249.5 / 556_785 | 2_000_000 / – | 18.8 / 69.2 | 14.7 / 4.09 | 64.6 | 0.014 | 0.0063 |
| (H₂O)₂₁ | 0/12 | 0/12 | 2_000_000 / – | 2_000_000 / – | 24.8 / 75.4 | 18.9 / 5.88 | 70.5 | 1.000 | 1.000 |

Anneal loses no size. It has at least GMIN's hits everywhere and fewer median
calls at 8, 12 and 20. At 16 and 21 both all-seed medians are the 2 M budget,
and at 21 neither code reaches the minimum in 2 M calls.
There anneal's lowest energy is below GMIN's on all twelve paired seeds,
median −914.368 against −895.593 kJ/mol. The hit counts separate at 12 and 20
(p(F) ≤ 0.014); at 8 and 16 they do not at twelve seeds. Over the twelve
seeds anneal spends 73–77% of its wall inside the potential from 16 up, and
58–61% at 8 and 12, where the median run stops within 0.25 s.

At (H₂O)₁₆ anneal's ten misses end in a few recurring basins: −676.186 kJ/mol
on four seeds and −674.006 on three, 5.0 and 7.2 kJ/mol above the header.
GMIN's misses end 12.7 to 19.0 kJ/mol above it. Because anneal does not lose
the size, those basins were not diagnosed in this run.
