# Covalent and many-body clusters

Stillinger-Weber Si10, Si13 and Si20 and Tersoff C10 and C20. GMIN is the
comparison on silicon. GMIN has no Tersoff, so ASE basin hopping is the
comparison on carbon. The ASE rows on silicon are a cross-check.

Three anneal arms run:

- R is `Config::recommended(n)`.
- P is R with `H2H_PRIORS=0`. Orbit completion, stall symmetrisation, the
  continuous-symmetry reseed, the Lennard-Jones energy lattice and the
  two-phase surface are off.
- G is P on scales read off the potential's own dimer. The length is the bond
  length over \(2^{1/6}\) and the energy is the well depth. The probe is
  charged to the ledger.

## Potentials

Both potentials are in rgpot 3.6.0 at `20c58fe5`. anneal has neither.
`rgpot::StillingerWeberSi` is Stillinger and Weber, Phys. Rev. B 31, 5262
(1985). `rgpot::TersoffSiC` is Tersoff, Phys. Rev. B 39, 5566 (1989), with the
erratum, Phys. Rev. B 41, 3248 (1990). Both are periodic C++ classes and have
no C entry. `scripts/covalent_h2h/rgpot_covalent.cc` is a shim of about a
hundred lines. Each connected part of the cluster goes in its own vacuum cell,
so no periodic image is inside a cutoff.

Silicon is in reduced units, with \(\sigma = 2.0951\) Å and
\(\epsilon = 2.1672\) eV. GMIN's `SW` keyword uses the same units. Carbon is in
Å and eV.

## References

| system | energy | source |
|---|---|---|
| Si10 | -13.79658026 ε | lowest of reference.csv |
| Si13 | -18.50399759 ε | lowest of reference.csv |
| Si20 | -30.18725174 ε | Yoo and Zeng, J. Chem. Phys. 119, 1442 (2003), Table II, -1.5094 ε per atom |
| C10 | -55.08979464 eV | lowest of reference.csv |
| C20 | -120.69889105 eV | relaxed corannulene bowl, `tc20-bowl.xyz` |

`reference.csv` holds unstopped 1e6-call runs at seeds 9000-9002 for every code.
On Si10, Si13 and Si20 every code reaches the same minimum. On C10 every anneal
arm does, and ASE stops at -51.30 eV. I did not find a published Tersoff value
for C10 or C20.

On C20, no unbiased run reaches the bowl. The best in `reference.csv` is
-119.1536 eV. The lowest unbiased minimum I saw is a planar flake at
-119.6919 eV, `tc20-flake.xyz`, from an exploratory G run with a wider
container. `scripts/covalent_h2h/bowl.py` builds the
corannulene skeleton and the dodecahedral cage and relaxes both. The bowl
relaxes to -120.6989 eV and the cage to -115.5898 eV. Three ASE basin-hopping
runs of 150k calls each, started from the bowl, found nothing lower. Four runs
started from the flake never reached the bowl. Ring, bowl and cage are the C20
isomers long studied with first-principles methods (Grossman, Mitas and
Raghavachari, Phys. Rev. Lett. 75, 3870 (1995)).

## Method

The protocol follows the earlier head-to-heads:

- Six seeds, 2000-2005.
- All runs serial, in one session, under `sudo unshare --uts` with hostname
  `bench`, pinned by `taskset -c 0`, in an order shuffled by
  `Random(20261113)`.
- Every code gets the same budget: 1e5 calls at 10 and 13 atoms, 4e5 at 20.
- A hit is a recorded energy at or below the reference plus 0.001 in the
  system's unit.
- `pot_s` is the time inside the potential library. `outside_s` is the wall
  time minus `pot_s`.

Configuration of the comparators:

- GMIN is the public Wales tree at `aac1548a`, with `SWTWO` timed by
  `sw-timer.patch`. `h2h_gmin_sw.data` uses `TEMPERATURE 0.2` (from Yoo and
  Zeng) and `STEP 0.40`. `SLOPPYCONV 5e-3` is GMIN's RMS-gradient quench
  convergence, so GMIN's hits sit about 3e-5 ε above the reference while
  still counting. `RADIUS` is set to anneal's container. With GMIN's default
  radius, Si atoms evaporated past the cutoff.
- ASE is `ase.optimize.basin.BasinHopping` with LBFGS at `fmax=1e-3`, at
  temperature \(0.2\,\epsilon\) and step \(0.4\,\sigma\) in the system's
  scales. It calls the same shim through ctypes. Its `outside_s` is mostly
  Python.

The before panel's anneal binary is `93e43c60`. Its rows were re-summarised
with `fbec7090`. When R reaches the target through orbit completion, the hit
has no row in the improvement list, so the run printed the budget. The ledger
had stopped at the target, so the charged count is the real hit count.

The before panel shared the machine with exploratory runs on cores 4-7. The
after panel ran alone. Call counts are unaffected by this, but before-panel
wall times may be slightly high.

## Before

| system | code | hits | median calls | best | pot_s | outside_s |
|---|---|---|---|---|---|---|
| Si10 | anneal R | 6/6 | 812 | -13.796580 | 0.015 | 0.001 |
| Si10 | anneal P | 6/6 | 792 | -13.796580 | 0.015 | 0.001 |
| Si10 | anneal G | 6/6 | 941 | -13.796580 | 0.015 | 0.001 |
| Si10 | GMIN | 6/6 | 898 | -13.796577 | 0.006 | 0.002 |
| Si10 | ASE | 5/6 | 33,946 | -13.796580 | 1.565 | 19.508 |
| Si13 | anneal R | 6/6 | 1,154 | -18.503998 | 0.025 | 0.002 |
| Si13 | anneal P | 6/6 | 1,432 | -18.503998 | 0.030 | 0.002 |
| Si13 | anneal G | 6/6 | 2,267 | -18.503998 | 0.049 | 0.003 |
| Si13 | GMIN | 6/6 | 612 | -18.503987 | 0.005 | 0.001 |
| Si13 | ASE | 1/6 | 100,000 | -18.503998 | 4.386 | 52.945 |
| Si20 | anneal R | 6/6 | 3,546 | -30.187252 | 0.123 | 0.008 |
| Si20 | anneal P | 6/6 | 95,232 | -30.187252 | 3.096 | 0.194 |
| Si20 | anneal G | 6/6 | 58,011 | -30.187252 | 1.972 | 0.110 |
| Si20 | GMIN | 6/6 | 9,481 | -30.187223 | 0.165 | 0.017 |
| Si20 | ASE | 1/6 | 400,000 | -30.187251 | 22.895 | 224.873 |
| C10 | anneal R | 6/6 | 26,721 | -55.089795 | 0.533 | 0.028 |
| C10 | anneal P | 6/6 | 32,758 | -55.089795 | 0.605 | 0.030 |
| C10 | anneal G | 5/6 | 74,712 | -55.089795 | 1.280 | 0.062 |
| C10 | ASE | 0/6 | 100,000 | -54.279110 | 4.012 | 41.923 |
| C20 | anneal R | 0/6 | 400,000 | -119.568269 | 13.331 | 1.016 |
| C20 | anneal P | 0/6 | 400,000 | -119.326297 | 13.919 | 1.140 |
| C20 | anneal G | 0/6 | 400,000 | -118.244824 | 12.395 | 0.965 |
| C20 | ASE | 0/6 | 400,000 | -114.398096 | 25.043 | 210.452 |

Median calls count a miss as the full budget. All time columns are per-run
medians in seconds.

## Where anneal loses

R loses to GMIN on Si13 (1,154 calls against 612) and is level on Si10. On
Si20, R takes 3,546 calls against GMIN's 9,481. P and G need 58k-95k. Removing
one prior at a time from R on Si20, over seeds 2000-2005, shows where that
gap comes from. With orbit completion off, the median goes from 3.5k to 47k.
Stall symmetrisation, the surface, the lattice and the continuous-symmetry
reseed each change it by less than seed noise. So orbit completion is not a
Lennard-Jones prior: it helps on a covalent cluster too.

The temperature is the other Si20 lever. Over the same seeds, P at
\(0.2\,\epsilon\) instead of \(0.8\,\epsilon\) reaches the minimum in about
a quarter of the calls. R does
not improve. A temperature read off the run itself, `statistical_temperature`,
did not separate from noise either. I left the temperature alone.

On Si20 the walk stalls 0.004 ε above the minimum, at -30.18328 ε. In the
sorted pair-distance key, that trap and the minimum are 0.58 apart, inside the
0.7 merge radius, so the return screen can call a hop from the trap into the
minimum a return. On P over seeds 2000-2003, turning the screen off, polishing
returns, a radius of 0.35 and the calibrated radius all stayed within seed
noise.

On carbon anneal beats ASE in every arm: ASE misses C10 on all six seeds.
G is the weakest anneal arm on C10. In a G walk on C10, 65 of 138 accepted
hops held a piece with no neighbour inside 2.1 Å. That is Tersoff's cutoff,
so the piece feels no force at all. Of 89 such hops, 52 had shed a single
atom. No quench can bring such an atom back, and the walk goes on moving it
on a flat surface. Among recorded minima, 30-50 per cent are in pieces for
G, 5-11 per cent for R with its smaller container, and 1-3 per cent on
Si20.

No code reaches the C20 bowl. Flake-to-bowl is a pentagon closure that a
random Cartesian step does not make.

## Fix

`945f1575` refuses a quench that has shed a free atom. A free atom has no
neighbour inside the configuration's neighbour cutoff, and its gradient is
exactly zero. The gradient is the one the record test already charges, so the
check costs no extra calls. The fix adds no constant. If the chain itself
already holds a free atom, which a random start can, it keeps moving
normally.

Lennard-Jones and Morse tails never reach zero, so no atom meets the test and
their trajectories are unchanged. Rigid bodies and labelled species are not
judged. A first version tested geometry alone, and it moved the ρ=14 Morse
rows at 55 atoms. Testing the force exactly is what keeps that row fixed.

On 96 C10 seeds, 3000-3095, run four at a time on cores 4-7:

| arm | hits before | hits after | median calls before | median calls after | mean before | mean after | Mann-Whitney p |
|---|---|---|---|---|---|---|---|
| R | 87/96 | 91/96 | 30,642 | 32,432 | 39,086 | 40,111 | 0.71 |
| P | 84/96 | 87/96 | 42,196 | 37,728 | 48,209 | 45,176 | 0.50 |
| G | 77/96 | 84/96 | 74,739 | 62,112 | 62,320 | 55,133 | 0.10 |

Hits rise in every arm. Call counts on the hits do not change for R, and they
fall for G. The rows are in `c10-96/`.

Not implemented:

- G's container is too large for carbon. 0.8 of `0.9 σ n^{1/3}` gives G 48/48
  hits on C10, with a mean of 31k calls instead of 55k. The right size depends
  on the potential's range, which the configuration does not know, and any
  rule that shrinks the container moves the Lennard-Jones runs.
- G's temperature is 0.8 of the C2 well depth, 4.1 eV. Lowering it alone did
  not help on C10.

## After

| system | code | hits | median calls | best | pot_s | outside_s |
|---|---|---|---|---|---|---|
| Si10 | anneal R | 6/6 | 812 | -13.796580 | 0.014 | 0.001 |
| Si10 | anneal P | 6/6 | 792 | -13.796580 | 0.014 | 0.001 |
| Si10 | anneal G | 6/6 | 941 | -13.796580 | 0.015 | 0.001 |
| Si13 | anneal R | 6/6 | 1,154 | -18.503998 | 0.024 | 0.002 |
| Si13 | anneal P | 6/6 | 1,559 | -18.503998 | 0.033 | 0.002 |
| Si13 | anneal G | 6/6 | 2,267 | -18.503998 | 0.050 | 0.003 |
| Si20 | anneal R | 6/6 | 2,772 | -30.187252 | 0.100 | 0.006 |
| Si20 | anneal P | 6/6 | 73,372 | -30.187252 | 2.549 | 0.171 |
| Si20 | anneal G | 6/6 | 30,875 | -30.187252 | 1.112 | 0.062 |
| C10 | anneal R | 6/6 | 11,002 | -55.089795 | 0.216 | 0.012 |
| C10 | anneal P | 4/6 | 86,186 | -55.089795 | 1.491 | 0.076 |
| C10 | anneal G | 6/6 | 55,243 | -55.089795 | 0.982 | 0.051 |
| C20 | anneal R | 0/6 | 400,000 | -119.689551 | 13.757 | 1.078 |
| C20 | anneal P | 0/6 | 400,000 | -119.370222 | 13.235 | 1.047 |
| C20 | anneal G | 0/6 | 400,000 | -117.337739 | 13.282 | 1.040 |

The anneal binary is `8f627b2f`. GMIN and ASE are as before. At six seeds,
C10 P dropping to 4/6 is within the spread of the 96-seed table.

The first panel's binaries were built before its commits were rebased onto
develop, so `93e43c60`, `fbec7090` and `8f627b2f` name the pre-merge trees.

## Second panel

The second panel targets three gaps without system constants: P and G on
Si20, the quench tolerance on Si13, and C10 and C20 without the 0.8
container. P and G now take three general mechanisms. R is unchanged, and
so are the Lennard-Jones, Morse, Gupta and TIP4P pins.

### Mechanisms

- Orbit completion as an allocator arm (`Config::orbit_arm`,
  `ClusterMove::OrbitComplete`). R completes orbits on every new basin,
  and that is what makes R fast on Si20. P and G offer the same
  symmetrise-and-quench move to the depth-reward Thompson allocator, which
  draws it as often as it pays. When no orbit group moves an atom, the arm
  takes a single-point step.
- The budget-window temperature (`budget_window`) replaces the 0.8 ε
  Lennard-Jones calibration in P and G. It clamps between an escape floor
  and a descent ceiling. The floor is the barrier the chain failed to cross
  over the log of the remaining budget, and the ceiling is twice the gap to
  the incumbent per degree of freedom.
- The measured pair reach (`Config::pair_range`, `PairRange::measure`).
  The harness bisects the separation past which the dimer's energy and
  force are exactly zero. The probe is charged and timed. When that reach is
  shorter than the container's diameter, a hop gathers each part of the
  trial back within reach of the largest part, one bond away, instead of
  pressing it into the sphere. The reaches are the cutoffs, 1.8 σ for SW and
  2.1 Å for Tersoff C. Both are inside the container's diameter at every
  size here, so gathering replaces the sphere on every P and G row.
  Lennard-Jones and Morse have no reach. No preset sets the field.

`ANNEAL_ADAPTIVE_CORE` is not on develop and was not tried.

### Method

The protocol is that of the first panel, with these changes:

- Twelve seeds (2000-2011) per code. ASE has six (2000-2005) because a
  4e5-call ASE miss takes about 3.5 minutes.
- R, P and G come from `f235b67d`, on develop `2ea8d834`. "P v1" and "G v1"
  are develop's own P and G at `2ea8d834`. R is the same build in both.
- The quench tolerance is reported at two settings. anneal quenches to a
  largest gradient component of 1e-6. The "5e-3" rows set `H2H_GTOL=5e-3`,
  which loosens anneal's quench and record gate to GMIN's `SLOPPYCONV`.
  "GMIN 1e-6" sets `SLOPPYCONV` and `TIGHTCONV` to 1e-6 (RMS).
- 510 runs, serial on core 0, in one session with nothing else on the
  machine, under `sudo unshare --uts` with hostname `bench`.
- p-values are two-sided: Mann-Whitney U on calls to the hit, with a miss
  counted as the budget, and Fisher's exact test on hits. They come from
  `scripts/covalent_h2h/tables.py panel`.

| system | code | hits | median calls | IQR | best | pot_s | outside_s |
|---|---|---|---|---|---|---|---|
| Si10 | anneal R | 12/12 | 758 | 378 | -13.796580 | 0.012 | 0.001 |
| Si10 | anneal P v1 | 12/12 | 966 | 948 | -13.796580 | 0.016 | 0.001 |
| Si10 | anneal P | 12/12 | 1,782 | 1,282 | -13.796580 | 0.028 | 0.002 |
| Si10 | anneal G v1 | 12/12 | 1,336 | 1,050 | -13.796580 | 0.020 | 0.001 |
| Si10 | anneal G | 12/12 | 1,599 | 1,126 | -13.796580 | 0.024 | 0.001 |
| Si10 | anneal R 5e-3 | 12/12 | 667 | 558 | -13.796579 | 0.011 | 0.001 |
| Si10 | anneal P 5e-3 | 12/12 | 879 | 634 | -13.796580 | 0.012 | 0.001 |
| Si10 | anneal G 5e-3 | 12/12 | 802 | 639 | -13.796580 | 0.011 | 0.001 |
| Si10 | GMIN 5e-3 | 12/12 | 624 | 673 | -13.796577 | 0.003 | 0.001 |
| Si10 | GMIN 1e-6 | 12/12 | 1,118 | 1,343 | -13.796580 | 0.006 | 0.001 |
| Si10 | ASE BH | 5/6 | 33,946 | 45,537 | -13.796580 | 1.258 | 16.863 |
| Si13 | anneal R | 12/12 | 1,166 | 908 | -18.503998 | 0.024 | 0.002 |
| Si13 | anneal P v1 | 12/12 | 1,467 | 1,430 | -18.503998 | 0.030 | 0.002 |
| Si13 | anneal P | 12/12 | 1,006 | 516 | -18.503998 | 0.018 | 0.001 |
| Si13 | anneal G v1 | 12/12 | 2,234 | 1,901 | -18.503998 | 0.043 | 0.003 |
| Si13 | anneal G | 12/12 | 1,016 | 477 | -18.503998 | 0.018 | 0.001 |
| Si13 | anneal R 5e-3 | 12/12 | 636 | 732 | -18.503996 | 0.013 | 0.001 |
| Si13 | anneal P 5e-3 | 12/12 | 837 | 521 | -18.503997 | 0.014 | 0.001 |
| Si13 | anneal G 5e-3 | 12/12 | 850 | 552 | -18.503997 | 0.015 | 0.001 |
| Si13 | GMIN 5e-3 | 12/12 | 683 | 332 | -18.503990 | 0.005 | 0.001 |
| Si13 | GMIN 1e-6 | 12/12 | 2,225 | 2,950 | -18.503998 | 0.014 | 0.003 |
| Si13 | ASE BH | 1/6 | 100,000 | 0 | -18.503998 | 3.958 | 48.986 |
| Si20 | anneal R | 12/12 | 4,454 | 8,120 | -30.187252 | 0.140 | 0.010 |
| Si20 | anneal P v1 | 12/12 | 105,620 | 92,918 | -30.187252 | 3.260 | 0.216 |
| Si20 | anneal P | 12/12 | 9,230 | 8,470 | -30.187252 | 0.279 | 0.015 |
| Si20 | anneal G v1 | 12/12 | 59,803 | 92,803 | -30.187252 | 1.827 | 0.107 |
| Si20 | anneal G | 12/12 | 8,207 | 5,942 | -30.187252 | 0.250 | 0.015 |
| Si20 | anneal R 5e-3 | 12/12 | 7,256 | 4,132 | -30.187249 | 0.234 | 0.017 |
| Si20 | anneal P 5e-3 | 12/12 | 6,142 | 7,934 | -30.187248 | 0.189 | 0.011 |
| Si20 | anneal G 5e-3 | 12/12 | 7,276 | 8,440 | -30.187249 | 0.231 | 0.013 |
| Si20 | GMIN 5e-3 | 12/12 | 5,900 | 8,598 | -30.187223 | 0.097 | 0.010 |
| Si20 | GMIN 1e-6 | 12/12 | 9,601 | 8,902 | -30.187252 | 0.148 | 0.016 |
| Si20 | ASE BH | 1/6 | 400,000 | 0 | -30.187251 | 20.426 | 206.492 |
| C10 | anneal R | 12/12 | 16,696 | 22,517 | -55.089795 | 0.249 | 0.013 |
| C10 | anneal P v1 | 9/12 | 62,863 | 66,196 | -55.089795 | 0.927 | 0.044 |
| C10 | anneal P | 12/12 | 13,788 | 12,108 | -55.089795 | 0.205 | 0.010 |
| C10 | anneal G v1 | 11/12 | 55,243 | 43,848 | -55.089795 | 0.798 | 0.041 |
| C10 | anneal G | 12/12 | 15,504 | 14,944 | -55.089795 | 0.231 | 0.011 |
| C10 | ASE BH | 0/6 | 100,000 | 0 | -54.279110 | 3.589 | 36.895 |
| C20 | anneal R | 0/12 | 400,000 | 0 | -119.689551 | 11.663 | 0.861 |
| C20 | anneal P v1 | 0/12 | 400,000 | 0 | -119.370222 | 11.711 | 0.896 |
| C20 | anneal P | 4/12 | 400,000 | 35,812 | -120.698891 | 11.335 | 0.648 |
| C20 | anneal G v1 | 0/12 | 400,000 | 0 | -119.762363 | 11.231 | 0.813 |
| C20 | anneal G | 3/12 | 400,000 | 30,903 | -120.698891 | 11.258 | 0.657 |
| C20 | ASE BH | 0/6 | 400,000 | 0 | -114.398096 | 19.906 | 173.299 |

| system | comparison | median calls | hits | Mann-Whitney p | Fisher p |
|---|---|---|---|---|---|
| Si10 | anneal P vs anneal P v1 | 1,782 vs 966 | 12/12 vs 12/12 | 0.29 | 1 |
| Si10 | anneal G vs anneal G v1 | 1,599 vs 1,336 | 12/12 vs 12/12 | 0.8 | 1 |
| Si10 | anneal P vs anneal R | 1,782 vs 758 | 12/12 vs 12/12 | 0.019 | 1 |
| Si10 | anneal G vs anneal R | 1,599 vs 758 | 12/12 vs 12/12 | 0.019 | 1 |
| Si10 | anneal R vs GMIN 1e-6 | 758 vs 1,118 | 12/12 vs 12/12 | 0.1 | 1 |
| Si10 | anneal P vs GMIN 1e-6 | 1,782 vs 1,118 | 12/12 vs 12/12 | 0.67 | 1 |
| Si10 | anneal G vs GMIN 1e-6 | 1,599 vs 1,118 | 12/12 vs 12/12 | 0.75 | 1 |
| Si10 | anneal R 5e-3 vs GMIN 5e-3 | 667 vs 624 | 12/12 vs 12/12 | 0.84 | 1 |
| Si10 | anneal P 5e-3 vs GMIN 5e-3 | 879 vs 624 | 12/12 vs 12/12 | 0.29 | 1 |
| Si10 | anneal G 5e-3 vs GMIN 5e-3 | 802 vs 624 | 12/12 vs 12/12 | 0.31 | 1 |
| Si13 | anneal P vs anneal P v1 | 1,006 vs 1,467 | 12/12 vs 12/12 | 0.14 | 1 |
| Si13 | anneal G vs anneal G v1 | 1,016 vs 2,234 | 12/12 vs 12/12 | 0.019 | 1 |
| Si13 | anneal P vs anneal R | 1,006 vs 1,166 | 12/12 vs 12/12 | 1 | 1 |
| Si13 | anneal G vs anneal R | 1,016 vs 1,166 | 12/12 vs 12/12 | 0.98 | 1 |
| Si13 | anneal R vs GMIN 1e-6 | 1,166 vs 2,225 | 12/12 vs 12/12 | 0.11 | 1 |
| Si13 | anneal P vs GMIN 1e-6 | 1,006 vs 2,225 | 12/12 vs 12/12 | 0.13 | 1 |
| Si13 | anneal G vs GMIN 1e-6 | 1,016 vs 2,225 | 12/12 vs 12/12 | 0.13 | 1 |
| Si13 | anneal R 5e-3 vs GMIN 5e-3 | 636 vs 683 | 12/12 vs 12/12 | 0.98 | 1 |
| Si13 | anneal P 5e-3 vs GMIN 5e-3 | 837 vs 683 | 12/12 vs 12/12 | 0.11 | 1 |
| Si13 | anneal G 5e-3 vs GMIN 5e-3 | 850 vs 683 | 12/12 vs 12/12 | 0.11 | 1 |
| Si20 | anneal P vs anneal P v1 | 9,230 vs 105,620 | 12/12 vs 12/12 | 0.00012 | 1 |
| Si20 | anneal G vs anneal G v1 | 8,207 vs 59,803 | 12/12 vs 12/12 | 0.00031 | 1 |
| Si20 | anneal P vs anneal R | 9,230 vs 4,454 | 12/12 vs 12/12 | 0.11 | 1 |
| Si20 | anneal G vs anneal R | 8,207 vs 4,454 | 12/12 vs 12/12 | 0.29 | 1 |
| Si20 | anneal R vs GMIN 1e-6 | 4,454 vs 9,601 | 12/12 vs 12/12 | 0.11 | 1 |
| Si20 | anneal P vs GMIN 1e-6 | 9,230 vs 9,601 | 12/12 vs 12/12 | 0.84 | 1 |
| Si20 | anneal G vs GMIN 1e-6 | 8,207 vs 9,601 | 12/12 vs 12/12 | 0.84 | 1 |
| Si20 | anneal R 5e-3 vs GMIN 5e-3 | 7,256 vs 5,900 | 12/12 vs 12/12 | 0.89 | 1 |
| Si20 | anneal P 5e-3 vs GMIN 5e-3 | 6,142 vs 5,900 | 12/12 vs 12/12 | 0.8 | 1 |
| Si20 | anneal G 5e-3 vs GMIN 5e-3 | 7,276 vs 5,900 | 12/12 vs 12/12 | 0.75 | 1 |
| C10 | anneal P vs anneal P v1 | 13,788 vs 62,863 | 12/12 vs 9/12 | 0.00089 | 0.22 |
| C10 | anneal G vs anneal G v1 | 15,504 vs 55,243 | 12/12 vs 11/12 | 0.0051 | 1 |
| C10 | anneal P vs anneal R | 13,788 vs 16,696 | 12/12 vs 12/12 | 0.58 | 1 |
| C10 | anneal G vs anneal R | 15,504 vs 16,696 | 12/12 vs 12/12 | 0.98 | 1 |
| C20 | anneal P vs anneal P v1 | 400,000 vs 400,000 | 4/12 vs 0/12 | 0.037 | 0.093 |
| C20 | anneal G vs anneal G v1 | 400,000 vs 400,000 | 3/12 vs 0/12 | 0.079 | 0.22 |
| C20 | anneal P vs anneal R | 400,000 vs 400,000 | 4/12 vs 0/12 | 0.037 | 0.093 |
| C20 | anneal G vs anneal R | 400,000 vs 400,000 | 3/12 vs 0/12 | 0.079 | 0.22 |

### Findings

- Si20: P and G go from 106k and 60k calls to 9.2k and 8.2k, p ≤ 3e-4.
  GMIN takes 9.6k at 1e-6 (p = 0.84) and 5.9k at 5e-3, against 6.1k-7.3k
  for anneal at 5e-3 (p ≥ 0.75). R stays faster at 4.5k, but the gap to R
  is not significant at n=12 (p = 0.11 and 0.29).
- Si13: at the same tolerance, the codes are level. At 5e-3, R takes 636
  calls and GMIN 683 (p = 0.98). At 1e-6, R takes 1,166 and GMIN 2,225
  (p = 0.11). The first panel's 1,154 against 612 compared two different
  quench tolerances. G improves from 2,234 to 1,016 (p = 0.019).
- Si10: P rises from 966 to 1,782 calls. That change is not significant
  (p = 0.29), but P and G at 1e-6 are now behind R (p = 0.019).
- C10: P goes from 9/12 hits at 63k calls to 12/12 at 13.8k, and G from
  11/12 at 55k to 12/12 at 15.5k. Both are level with R. No container
  constant is involved.
- C20: P reaches the corannulene bowl, -120.698891 eV, in 4 of 12 runs,
  and G in 3. No other arm reaches it: R, P v1 and G v1 go 0/12 and ASE
  0/6. No run went below the bowl. The best energy per arm is R
  -119.689551, P v1 -119.370222, G v1 -119.762363, P and G -120.698891, and
  ASE -114.398096 eV. Against develop's P, the Fisher p is 0.093 for P and
  0.22 for G.

### Exploration

These runs used seeds 2000-2011, or 2000-2047 where marked, on cores 2-7 in
parallel. They count calls only; the numbers are not from the timed panel.

- Si20, P, without each mechanism and with each alone (48 seeds): base
  90k; orbit arm 25.5k; window 19.9k; both 11.0k; both plus gather 8.9k.
  For G: base 82k; orbit arm 20.3k; window 17.6k; both 12.3k; both plus
  gather 11.1k. Taking the energy lattice out of R moves its median from
  4.5k to 7.8k.
- C10 container, G with the old flags (24 seeds): 1.0x 50k, 22/24 hits;
  0.8x 22.7k, 24/24; gather into the measured reach 31.1k, 24/24. With the
  orbit arm and the window, every container from 0.7x to 1.2x lands at
  15k-22k calls, so the constant no longer matters.
- C20, what was blocking the bowl: with the orbit arm and window but inside
  the sphere, P goes 0/6. The sphere was the obstacle. Gathering alone, or
  gathering with the window, reaches it in at most 1 of 12 runs, and the
  orbit arm is needed with it.
- C20, population: CSA (`csa_cluster`, 8 or 16 members, 20k or 50k calls
  per member) gave P 0-2/12 and G 2-3/12, no better than one walk (4/12
  and 3/12).
- C20, plateau restarts (`restart_on_stall` with `escape_on_stall` off, at
  2e4 or 6e4 stalled calls): P 1-2/12 and G 5/12, against 4/12 and 3/12
  without restarts. The G gain is within noise (Fisher p = 0.67), so
  restarts are not on.
- C20, R with gather: 1/12.

### Pins and tests

`scripts/covalent_h2h/pins.sh` prints sha1
`419b06def1a0e01a152c13ccdb24cbd9919551cc` on develop `2ea8d834` and on
`f235b67d`. `tests/resolved_config.rs` is unchanged: the new fields leave
the resolved JSON when unset.

On `f235b67d`, `cargo test --release` passes everything except three tests in
`tests/feedback_identity_namespaces.rs` and
`tests/feedback_namespace_controls.rs`, which fail identically on develop
`2ea8d834`. `pytest pytest/` gives 1306 passed and 19 skipped. With
ChemFit 4.0.0, `pytest pytest/test_chemfit*.py` gives 1005 passed and 4
skipped, and `examples/chemfit_anneal.py` prints best loss
2.347665788572983.

## Pins

`scripts/covalent_h2h/pins.sh` prints, with every clock column removed:

- trace and lowest-geometry hashes of whole-budget `h2h_seed` runs: LJ13,
  LJ38, LJ55 and LJ75, ρ=14 Morse 38, and TIP4P 8 and 12;
- `h2h_morse` rows for ρ=6 and ρ=14 at 38 and 55 atoms;
- `h2h_metal` rows for Zn38, Cu19Au19 and Pt13.

Before and after the fix, the output's sha1 is
`419b06def1a0e01a152c13ccdb24cbd9919551cc`. `tests/lj_trajectory_pin.rs` passes.

## Files

- `before/`, `after/`: per-seed rows and summaries of the first panel.
- `v2/`: per-seed rows and summaries of the second panel.
- `reference.csv`: the unstopped reference runs.
- `tc20-bowl.xyz`, `tc20-flake.xyz`: the C20 bowl and the best unbiased flake.
- `c10-96/`: the 96-seed C10 rows.
