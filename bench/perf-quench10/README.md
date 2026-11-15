# M80 ρ=6, where the 12/12 went

Seeds 2000–2011, budget 4e6, one core, hostname `bench`. A hit is within
0.001 of −378.333471. Medians are Hyndman–Fan type 7. `priors=off` is
`H2H_PRIORS=0`: the fivefold SOAP move, the icosahedral reseed, lattice
regrowth, the labelled core, the motif reseed, stall symmetrisation and
the continuous-symmetry reseed are off. Unset, they stay at the preset.

| revision | priors | hits | median calls | median wall (s) | median inside (s) | median outside (s) |
| --- | --- | --- | --- | --- | --- | --- |
| `fef007ed` | on | 12/12 | 1,066,991 | 43.695 | 28.762 | 15.033 |
| `fef007ed` | off | 12/12 | 422,432 | 15.469 | 11.439 | 4.029 |
| `beb89163` | on | 2/12 | 4,000,000 | 163.071 | 107.358 | 55.577 |
| `beb89163` | off | 12/12 | 993,596 | 36.020 | 26.395 | 9.625 |

Fisher exact on the hits, priors off against priors on: `fef007ed` 1.00,
`beb89163` 0.0001. Mann–Whitney on the calls: 0.10 and 0.0001.

`fef007ed` is develop immediately before `9abb056e` merged
`fix/recommended-quench-cap`. `beb89163` is current develop. The
structure priors are already on at `fef007ed`, and that commit hits
12/12. They are not what arrived between the two.

What arrived, for an unlabelled Morse cluster, is `3c45b899`: the
recommended full quench runs 1000 steps instead of 200. On the merge
itself (`9abb056e`, still pinned to rgpot `1c1b4ed`), seed 2000 misses
inside 4e6 calls at 1000 steps. Putting that one assignment back to 200,
priors left on, returns the previous hit at 1,402,197 calls with the
same arm draws. The same seed on `beb89163` with the cap at 200 does
too. The rgpot pin `8efdd246` is not in that pair. Between `1c1b4ed`
and `8efdd246` the only C change is `CppCore/rgpot/Gupta/gupta.c`, the
thread-local buffers, not the Morse kernel.

The 1000-step cap changes the endpoint of a quench that has not met the
tolerance, and the chain accepts that endpoint. With the structure
priors on, the longer endpoint is a different walk and ρ=6 falls to
2/12. With those priors off, `beb89163` is back to 12/12, so the cap
does not lose the minimum by itself. The prior-side bugs (the
icosahedral test using a reduced-unit cutoff on an ångström cluster,
and the stall clock) are left as they are.
