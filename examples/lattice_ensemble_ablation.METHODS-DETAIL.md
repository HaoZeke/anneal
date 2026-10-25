# Lattice ensemble ablation

Driver: `examples/lattice_ensemble_ablation.rs`.

One arm changes one published difference. The full arm is `Plan::default`: shared bank, one slot, splice 0.2, retirement after 20 unimproved draws, radial resemblance, merge distance held at D_ave/2, bank on, `StartRule::PresetSphere`, lattice lengths from `Lattice::for_pair` (multiples of `r_min`). The force ledger is accounting. It is not an arm.

| arm | field | full | published setting |
| --- | --- | --- | --- |
| `private` | `Plan.sharing` | `Shared`, slots 1 | `Private`, slots 2 |
| `splice_off` | `Plan.splice` | 0.2 | 0.0, shared bank kept, slots stay 1 |
| `retire_off` | `Plan.retire` | 20 | 0, Lee's bank with a held cutoff |
| `lee_distance` | `Plan.resemblance` | `Radial` | `LeeCsa`, eq. (2) below |
| `lee_anneal` | `Plan.cutoff` | `LinearBudget` with merge `(0.5, 0.5)` | `LeeFixedRatio` |
| `shao_lattice` | `Plan.banked` | true | false: fresh lattice descents, no bank, no splice |

`private` uses two slots because a splice needs a donor. `choose` draws one only when the bank already holds two members. Slots 2 is the smallest such bank. `splice_off` is the next arm.

Measured on the unit test `a_one_slot_private_bank_does_not_splice_and_two_slots_do` (LJ, n = 13, 4 chains, budget 600, fresh 0, splice 1, quench tolerance 1e6 so a trial finishes): slots 1 spliced 0 trials, slots 2 spliced 11 trials.

## Lee distance

Lee, Lee and Lee, Phys. Rev. Lett. 91, 080201 (2003), eq. (2), arXiv:cond-mat/0307690.

Shells are spheres of radius 1.35 and 1.70 in the paper's reduced units (sigma = 1). `H(1, n)` is the number of atoms with `n` neighbours in the first sphere, `H(2, n)` the same for the second. A neighbour on the boundary counts. The second sphere includes the first.

```
D(k, k') = sum_n n * (2 |H(1,n) - H'(1,n)| + |H(2,n) - H'(2,n)|)
```

On Lennard-Jones the radii are exactly 1.35 and 1.70. On any other pair they are multiplied by `r_min(kind) / r_min(Lennard-Jones)`, which is 1 for Lennard-Jones.

## Lee cutoff

Same paper: D_cut starts at D_ave/2, where D_ave is the average distance in the first bank. It is reduced by a fixed ratio so that it equals D_ave/5 after 10000 local minimizations, then held. The fraction implemented is `(1/2) * (2/5) ^ min(m / 10000, 1)`, with `m` the number of converged trial quenches already finished on that bank. `Plan.merge` is not read on this arm.

## Landscapes

Registered seeds, from `paper/seeds.toml`. Success is an energy at most 1e-4 above the reference. The search does not read the reference.

| name | n | pair | seeds | full budget | reference |
| --- | --- | --- | --- | --- | --- |
| lj38 | 38 | Lennard-Jones | 3100-3199 (100) | 400000 | -173.928427 |
| lj75 | 75 | Lennard-Jones | 3000-3029 (30) | 4000000 | -397.492331 |
| lj98 | 98 | Lennard-Jones | 3000-3029 (30) | 4000000 | -543.665361 |
| morse80 | 80 | Morse rho 14 | 2200-2247 (48) | 400000 | -340.811371 |

Lennard-Jones references are the values `examples/lj_lattice_ensemble.rs` already scores against (Cambridge Cluster Database). The Morse value is structure 80G in the rho = 14 column of Doye and Wales' table, the fourth energy column (rho = 3, 6, 10, 14), global minimum for rho at and above 11.36. https://doye.chem.ox.ac.uk/jon/structures/Morse/tables.html

LJ38 and the Morse budgets are the lattice harness default of 400000 force calls. LJ75 and LJ98 use 4000000. No separate Morse lattice budget is registered. Full-run hit rates are TBD-from-run on the v0.12.0 tag. They are not produced here.

The quench and the lattice site energy follow `PairKind`. A Morse run is Morse, not Lennard-Jones geometry on a Lennard-Jones potential. The default quench kind remains Lennard-Jones, and that gradient path is unchanged.

## Columns

Hits n/N with a Wilson 95% interval (z = 1.96). Median force calls to the hit, PAR-2 on a miss (twice the budget), and the interquartile range. The same for wall time. Force calls per second. Best energy when no run hits. Paired difference of the hit indicators against `full` on the same seeds, arm minus full, with a Wald 95% interval. Also the median of the paired PAR-2 force differences.

## Reproduction

```
lattice_ensemble_ablation all
lattice_ensemble_ablation lj38
```

Smoke, not a paper table:

```
PAPER_SMOKE=1 PAPER_OUT=/tmp/lattice-ablation-SMOKE lattice_ensemble_ablation all
```

`PAPER_SMOKE=1` uses the first two registered seeds, four chains and a budget of 8000, and writes `lattice-ablation-SMOKE.txt`. Hardware, cores, threads, rustc and the commit are printed in that file.
