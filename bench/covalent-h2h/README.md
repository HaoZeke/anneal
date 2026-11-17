# Covalent and many-body clusters

Stillinger-Weber Si10, Si13 and Si20 and Tersoff C10 and C20. In the first
two panels GMIN is the comparison on silicon and ASE basin hopping the one
on carbon, because GMIN has no Tersoff keyword. The third panel adds GMIN on
carbon through its `USERPOT` keyword and runs ASE on every system.

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

`reference.csv` holds unstopped runs at seeds 9000-9002: 1e6 calls for each
anneal arm, 4e5 for ASE and, on silicon, 2e6 for GMIN. On Si10 every code
reaches the same minimum, within the 0.001 hit band. On Si13 and Si20 every
anneal arm and GMIN do; ASE stops at -17.19 ε on Si13 seed 9000, and its best
on Si20 is -29.51 ε. On C10 every anneal arm reaches it, and ASE does on seeds
9001 and 9002 and stops at -51.30 eV on seed 9000. I did not find a published
Tersoff value for C10 or C20.

On C20, no unbiased run reaches the bowl. The best in `reference.csv` is
-119.6919 eV, anneal G on seed 9001. The lowest unbiased minimum I saw is a planar flake at
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

- Si20: P and G go from 106k and 60k calls to 9.2k and 8.2k, p ≤ 3.1e-4.
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

## Third panel

The third panel targets P and G against R on Si10 and the C20 bowl. It adds
GMIN on carbon and runs ASE on all twelve seeds. Every change is general
search machinery: no Lennard-Jones prior, no motif and no size constant. P
and G behave the same way on every potential. R is unchanged, and so are the
Lennard-Jones, Morse, Gupta and TIP4P pins.

### Mechanisms

- The measured reach bounds a ranged trial on its own
  (`contain_trial`, `f03d770e`). In the second panel the sphere came back
  when the reach was at or past the container's diameter. Now any
  `Config::pair_range` replaces the sphere, and the two rigid side walks
  gather the same way. `PairRange::measure` bisects to a relative 1e-6,
  finer than any connectivity cut the reach sets.
- The dimer probe works on the bond force (`examples/h2h_covalent`,
  `4ba4f6ed`). The old probe scanned at 5% a step and ran golden-section
  search at two calls a step, about 160 charged calls. On Si10, where the
  answer costs under a thousand calls, that was a fifth of every P and G
  run. A 1.2x scan finds the sign change of the bond force inside the well,
  and Illinois false position narrows it. Bond and depth agree with the old
  probe to 1e-9. The dimer and reach probes together went from 198 calls
  to 57.
- P and G complete orbits on every new basin, as R does, and keep the
  allocator's orbit arm (`25838776`). The five prior gates develop added
  (fivefold SOAP, icosahedral reseed, lattice regrowth, labelled core and
  motif reseed) are off in P and G.
- GMIN on carbon (`9631d65a`). `scripts/covalent_h2h/tersoff_userpot.f90`
  takes the place of GMIN's `dummy_userpot.f90` in a second build
  (`build_gmin_tersoff.sh`). It calls the same `librgpot_covalent.so` the
  anneal example and ASE load, and it is timed the way `SW.f90` is.
  `examples/h2h_gmin_tc.data` is `h2h_gmin_sw.data` on the carbon dimer
  scales, 1.2892326 Å and 5.1661751 eV. That gives `TEMPERATURE 1.0332`,
  `STEP 0.51569 0.65751`, `SLOPPYCONV 2.0036D-2`, `TIGHTCONV 4.0072D-3`,
  `EDIFF 0.015499` and `AVOID 5.1569`. `RADIUS` is anneal's container, as on
  silicon. "GMIN 1e-6" sets both convergences to 1e-6.

The start of each seed is drawn on the dimer scale. The new probe's scale
differs from the old one in the tenth digit, so every code starts from
slightly different coordinates than in the second panel, and R's chaotic
runs land on different call counts. R's code is unchanged. Between the
panels, R's samples differ at Mann-Whitney p ≥ 0.11 on every system: Si10
0.67, Si13 0.11, Si20 0.31, C10 0.13, and at 5e-3 0.71-0.98.

### Method

The protocol is that of the second panel, with these changes:

- Twelve seeds (2000-2011) for every code, ASE included.
- The paper budgets: 1e5 calls for Si10, Si13 and C10, and 4e5 for Si20
  and C20. `run_campaign.py` refuses a binary whose budgets differ.
- R, P and G come from `9631d65a`, on develop `7da10659`. "P v2" and "G v2"
  are develop's P and G at `7da10659`. They reproduce the second panel's P
  and G call for call.
- 588 runs, serial on core 0, in one session with nothing else on the
  machine, under `sudo unshare --uts` with hostname `bench`.

| system | code | hits | median calls | IQR | best | pot_s | outside_s |
|---|---|---|---|---|---|---|---|
| Si10 | anneal R | 12/12 | 597 | 203 | -13.796580 | 0.010 | 0.001 |
| Si10 | anneal P v2 | 12/12 | 1,782 | 1,282 | -13.796580 | 0.028 | 0.002 |
| Si10 | anneal P | 12/12 | 734 | 866 | -13.796580 | 0.011 | 0.001 |
| Si10 | anneal G v2 | 12/12 | 1,599 | 1,126 | -13.796580 | 0.024 | 0.002 |
| Si10 | anneal G | 12/12 | 990 | 878 | -13.796580 | 0.015 | 0.001 |
| Si10 | anneal R 5e-3 | 12/12 | 567 | 546 | -13.796580 | 0.009 | 0.001 |
| Si10 | anneal P 5e-3 | 12/12 | 502 | 390 | -13.796580 | 0.008 | 0.000 |
| Si10 | anneal G 5e-3 | 12/12 | 472 | 262 | -13.796580 | 0.007 | 0.000 |
| Si10 | GMIN 5e-3 | 12/12 | 550 | 547 | -13.796578 | 0.003 | 0.001 |
| Si10 | GMIN 1e-6 | 12/12 | 986 | 728 | -13.796580 | 0.005 | 0.001 |
| Si10 | ASE BH | 10/12 | 16,162 | 27,727 | -13.796580 | 0.600 | 7.938 |
| Si13 | anneal R | 12/12 | 769 | 78 | -18.503998 | 0.015 | 0.001 |
| Si13 | anneal P v2 | 12/12 | 1,006 | 516 | -18.503998 | 0.017 | 0.001 |
| Si13 | anneal P | 12/12 | 640 | 500 | -18.503998 | 0.012 | 0.001 |
| Si13 | anneal G v2 | 12/12 | 1,016 | 477 | -18.503998 | 0.018 | 0.001 |
| Si13 | anneal G | 12/12 | 658 | 824 | -18.503998 | 0.013 | 0.001 |
| Si13 | anneal R 5e-3 | 12/12 | 591 | 596 | -18.503997 | 0.012 | 0.001 |
| Si13 | anneal P 5e-3 | 12/12 | 424 | 489 | -18.503997 | 0.008 | 0.001 |
| Si13 | anneal G 5e-3 | 12/12 | 424 | 468 | -18.503997 | 0.008 | 0.001 |
| Si13 | GMIN 5e-3 | 12/12 | 784 | 950 | -18.503993 | 0.006 | 0.001 |
| Si13 | GMIN 1e-6 | 12/12 | 1,164 | 1,332 | -18.503998 | 0.008 | 0.002 |
| Si13 | ASE BH | 6/12 | 86,979 | 48,512 | -18.503998 | 3.434 | 41.456 |
| Si20 | anneal R | 12/12 | 8,956 | 11,392 | -30.187252 | 0.287 | 0.020 |
| Si20 | anneal P v2 | 12/12 | 9,230 | 8,470 | -30.187252 | 0.279 | 0.015 |
| Si20 | anneal P | 12/12 | 10,866 | 6,833 | -30.187252 | 0.339 | 0.015 |
| Si20 | anneal G v2 | 12/12 | 8,207 | 5,942 | -30.187252 | 0.248 | 0.015 |
| Si20 | anneal G | 12/12 | 9,662 | 10,260 | -30.187252 | 0.295 | 0.014 |
| Si20 | anneal R 5e-3 | 12/12 | 3,266 | 10,629 | -30.187250 | 0.100 | 0.010 |
| Si20 | anneal P 5e-3 | 12/12 | 7,831 | 10,173 | -30.187247 | 0.242 | 0.012 |
| Si20 | anneal G 5e-3 | 12/12 | 8,674 | 8,222 | -30.187248 | 0.273 | 0.013 |
| Si20 | GMIN 5e-3 | 12/12 | 12,573 | 16,588 | -30.187225 | 0.214 | 0.022 |
| Si20 | GMIN 1e-6 | 12/12 | 20,931 | 36,124 | -30.187252 | 0.325 | 0.034 |
| Si20 | ASE BH | 0/12 | 400,000 | 0 | -29.577777 | 20.221 | 210.394 |
| C10 | anneal R | 12/12 | 27,496 | 61,182 | -55.089795 | 0.405 | 0.020 |
| C10 | anneal P v2 | 12/12 | 13,788 | 12,108 | -55.089795 | 0.204 | 0.011 |
| C10 | anneal P | 12/12 | 15,038 | 37,727 | -55.089795 | 0.220 | 0.010 |
| C10 | anneal G v2 | 12/12 | 15,504 | 14,944 | -55.089795 | 0.231 | 0.012 |
| C10 | anneal G | 12/12 | 11,256 | 20,038 | -55.089795 | 0.165 | 0.008 |
| C10 | GMIN 5e-3 | 4/12 | 100,000 | 51,402 | -55.089745 | 1.605 | 0.126 |
| C10 | GMIN 1e-6 | 5/12 | 100,000 | 19,376 | -55.089795 | 1.583 | 0.117 |
| C10 | ASE BH | 2/12 | 100,000 | 0 | -55.089795 | 3.511 | 36.925 |
| C20 | anneal R | 0/12 | 400,000 | 0 | -119.306982 | 11.695 | 0.883 |
| C20 | anneal P v2 | 4/12 | 400,000 | 35,812 | -120.698891 | 11.294 | 0.651 |
| C20 | anneal P | 5/12 | 400,000 | 210,607 | -120.698891 | 11.187 | 0.555 |
| C20 | anneal G v2 | 3/12 | 400,000 | 30,903 | -120.698891 | 11.191 | 0.657 |
| C20 | anneal G | 4/12 | 400,000 | 82,088 | -120.698891 | 11.189 | 0.570 |
| C20 | GMIN 5e-3 | 0/12 | 400,000 | 0 | -118.064758 | 11.310 | 0.633 |
| C20 | GMIN 1e-6 | 0/12 | 400,000 | 0 | -119.153594 | 11.063 | 0.603 |
| C20 | ASE BH | 0/12 | 400,000 | 0 | -115.532884 | 19.933 | 171.666 |

| system | comparison | median calls | hits | Mann-Whitney p | Fisher p |
|---|---|---|---|---|---|
| Si10 | anneal P vs anneal P v2 | 734 vs 1,782 | 12/12 vs 12/12 | 0.089 | 1 |
| Si10 | anneal G vs anneal G v2 | 990 vs 1,599 | 12/12 vs 12/12 | 0.13 | 1 |
| Si10 | anneal P vs anneal R | 734 vs 597 | 12/12 vs 12/12 | 0.51 | 1 |
| Si10 | anneal G vs anneal R | 990 vs 597 | 12/12 vs 12/12 | 0.31 | 1 |
| Si10 | anneal R vs GMIN 1e-6 | 597 vs 986 | 12/12 vs 12/12 | 0.13 | 1 |
| Si10 | anneal P vs GMIN 1e-6 | 734 vs 986 | 12/12 vs 12/12 | 0.62 | 1 |
| Si10 | anneal G vs GMIN 1e-6 | 990 vs 986 | 12/12 vs 12/12 | 0.71 | 1 |
| Si10 | anneal R 5e-3 vs GMIN 5e-3 | 567 vs 550 | 12/12 vs 12/12 | 1 | 1 |
| Si10 | anneal P 5e-3 vs GMIN 5e-3 | 502 vs 550 | 12/12 vs 12/12 | 0.71 | 1 |
| Si10 | anneal G 5e-3 vs GMIN 5e-3 | 472 vs 550 | 12/12 vs 12/12 | 0.51 | 1 |
| Si10 | anneal P vs ASE BH | 734 vs 16,162 | 12/12 vs 10/12 | 0.00016 | 0.48 |
| Si10 | anneal G vs ASE BH | 990 vs 16,162 | 12/12 vs 10/12 | 0.00012 | 0.48 |
| Si13 | anneal P vs anneal P v2 | 640 vs 1,006 | 12/12 vs 12/12 | 0.03 | 1 |
| Si13 | anneal G vs anneal G v2 | 658 vs 1,016 | 12/12 vs 12/12 | 0.11 | 1 |
| Si13 | anneal P vs anneal R | 640 vs 769 | 12/12 vs 12/12 | 0.14 | 1 |
| Si13 | anneal G vs anneal R | 658 vs 769 | 12/12 vs 12/12 | 0.37 | 1 |
| Si13 | anneal R vs GMIN 1e-6 | 769 vs 1,164 | 12/12 vs 12/12 | 0.01 | 1 |
| Si13 | anneal P vs GMIN 1e-6 | 640 vs 1,164 | 12/12 vs 12/12 | 0.069 | 1 |
| Si13 | anneal G vs GMIN 1e-6 | 658 vs 1,164 | 12/12 vs 12/12 | 0.13 | 1 |
| Si13 | anneal R 5e-3 vs GMIN 5e-3 | 591 vs 784 | 12/12 vs 12/12 | 0.71 | 1 |
| Si13 | anneal P 5e-3 vs GMIN 5e-3 | 424 vs 784 | 12/12 vs 12/12 | 0.24 | 1 |
| Si13 | anneal G 5e-3 vs GMIN 5e-3 | 424 vs 784 | 12/12 vs 12/12 | 0.14 | 1 |
| Si13 | anneal P vs ASE BH | 640 vs 86,979 | 12/12 vs 6/12 | 3.2e-05 | 0.014 |
| Si13 | anneal G vs ASE BH | 658 vs 86,979 | 12/12 vs 6/12 | 3.2e-05 | 0.014 |
| Si20 | anneal P vs anneal P v2 | 10,866 vs 9,230 | 12/12 vs 12/12 | 0.98 | 1 |
| Si20 | anneal G vs anneal G v2 | 9,662 vs 8,207 | 12/12 vs 12/12 | 0.89 | 1 |
| Si20 | anneal P vs anneal R | 10,866 vs 8,956 | 12/12 vs 12/12 | 1 | 1 |
| Si20 | anneal G vs anneal R | 9,662 vs 8,956 | 12/12 vs 12/12 | 0.93 | 1 |
| Si20 | anneal R vs GMIN 1e-6 | 8,956 vs 20,931 | 12/12 vs 12/12 | 0.061 | 1 |
| Si20 | anneal P vs GMIN 1e-6 | 10,866 vs 20,931 | 12/12 vs 12/12 | 0.14 | 1 |
| Si20 | anneal G vs GMIN 1e-6 | 9,662 vs 20,931 | 12/12 vs 12/12 | 0.069 | 1 |
| Si20 | anneal R 5e-3 vs GMIN 5e-3 | 3,266 vs 12,573 | 12/12 vs 12/12 | 0.061 | 1 |
| Si20 | anneal P 5e-3 vs GMIN 5e-3 | 7,831 vs 12,573 | 12/12 vs 12/12 | 0.67 | 1 |
| Si20 | anneal G 5e-3 vs GMIN 5e-3 | 8,674 vs 12,573 | 12/12 vs 12/12 | 0.31 | 1 |
| Si20 | anneal P vs ASE BH | 10,866 vs 400,000 | 12/12 vs 0/12 | 1e-05 | 7.4e-07 |
| Si20 | anneal G vs ASE BH | 9,662 vs 400,000 | 12/12 vs 0/12 | 1e-05 | 7.4e-07 |
| C10 | anneal P vs anneal P v2 | 15,038 vs 13,788 | 12/12 vs 12/12 | 0.37 | 1 |
| C10 | anneal G vs anneal G v2 | 11,256 vs 15,504 | 12/12 vs 12/12 | 0.54 | 1 |
| C10 | anneal P vs anneal R | 15,038 vs 27,496 | 12/12 vs 12/12 | 0.13 | 1 |
| C10 | anneal G vs anneal R | 11,256 vs 27,496 | 12/12 vs 12/12 | 0.035 | 1 |
| C10 | anneal R vs GMIN 1e-6 | 27,496 vs 100,000 | 12/12 vs 5/12 | 0.0078 | 0.0046 |
| C10 | anneal P vs GMIN 1e-6 | 15,038 vs 100,000 | 12/12 vs 5/12 | 0.0018 | 0.0046 |
| C10 | anneal G vs GMIN 1e-6 | 11,256 vs 100,000 | 12/12 vs 5/12 | 0.00041 | 0.0046 |
| C10 | anneal P vs ASE BH | 15,038 vs 100,000 | 12/12 vs 2/12 | 0.00023 | 6.7e-05 |
| C10 | anneal G vs ASE BH | 11,256 vs 100,000 | 12/12 vs 2/12 | 0.00011 | 6.7e-05 |
| C20 | anneal P vs anneal P v2 | 400,000 vs 400,000 | 5/12 vs 4/12 | 0.57 | 1 |
| C20 | anneal G vs anneal G v2 | 400,000 vs 400,000 | 4/12 vs 3/12 | 0.86 | 1 |
| C20 | anneal P vs anneal R | 400,000 vs 400,000 | 5/12 vs 0/12 | 0.016 | 0.037 |
| C20 | anneal G vs anneal R | 400,000 vs 400,000 | 4/12 vs 0/12 | 0.037 | 0.093 |
| C20 | anneal R vs GMIN 1e-6 | 400,000 vs 400,000 | 0/12 vs 0/12 | tied | 1 |
| C20 | anneal P vs GMIN 1e-6 | 400,000 vs 400,000 | 5/12 vs 0/12 | 0.016 | 0.037 |
| C20 | anneal G vs GMIN 1e-6 | 400,000 vs 400,000 | 4/12 vs 0/12 | 0.037 | 0.093 |
| C20 | anneal P vs ASE BH | 400,000 vs 400,000 | 5/12 vs 0/12 | 0.016 | 0.037 |
| C20 | anneal G vs ASE BH | 400,000 vs 400,000 | 4/12 vs 0/12 | 0.037 | 0.093 |

### Findings

- Si10: P goes from 1,782 calls to 734 and G from 1,599 to 990 (p = 0.089
  and 0.13 against v2). Against R's 597, p is 0.51 for P and 0.31 for G,
  where the second panel had p = 0.019 for both. At 5e-3, P takes 502 calls
  and G 472, against 567 for R and 550 for GMIN (p ≥ 0.50).
- Si13: P takes 640 calls and G 658, against 1,006 and 1,016 in v2 (p =
  0.03 and 0.11) and 769 for R (p = 0.14 and 0.37). GMIN takes 1,164 at
  1e-6 and 784 at 5e-3.
- Si20: P takes 10.9k and G 9.7k, against 9.2k and 8.2k in v2 (p ≥ 0.88)
  and 9.0k for R (p ≥ 0.93). GMIN takes 20.9k at 1e-6 and 12.6k at 5e-3.
  ASE hits 0 of 12.
- C10: every anneal arm hits 12/12. G takes 11.3k calls against R's 27.5k
  (p = 0.035), and P takes 15.0k (p = 0.13). GMIN hits 5/12 at 1e-6 and
  4/12 at 5e-3, and ASE 2/12. Against GMIN at 1e-6, R, P and G have
  Fisher p = 0.0046.
- C20: P reaches the bowl, -120.698891 eV, in 5 of 12 runs, and G in 4,
  against 4 and 3 in v2 (p = 1). R, both GMIN codes and ASE go 0/12. The
  best energy per code is R -119.306982, GMIN 5e-3 -118.064758, GMIN 1e-6
  -119.153594 and ASE -115.532884 eV. Against R, GMIN 1e-6 and ASE, the
  Fisher p is 0.037 for P and 0.093 for G.

### Exploration

These runs used seeds 2000-2047 on cores 2-7 in parallel. They count calls
only; the numbers are not from the timed panel.

- The final P and G against R, 48 seeds:

  | system | P | G | R | P vs R p | G vs R p |
  |---|---|---|---|---|---|
  | Si10 | 652 | 662 | 750 | 0.25 | 0.41 |
  | Si13 | 1,220 | 1,226 | 758 | 0.04 | 0.03 |
  | Si20 | 11,162 | 10,908 | 8,454 | 0.08 | 0.22 |
  | C10 | 20,087 | 23,764 | 43,802 (45/48) | 0.002 | 0.008 |

  On Si13 at 48 seeds, P and G stay behind R. Twelve seeds do not show it.
- Si13, what R has that P lacks: taking the energy lattice out of R gives
  862 calls, taking the five prior gates out gives 854, and running R on
  the budget-window temperature gives 1,084. Adding the two-phase surface,
  stall symmetrisation, the continuous-inversion reseed or a temperature
  scale to P did not help it.
- A budget window counted in hops instead of calls gave no gain on Si10 and
  was worse on Si20 and C10. A bias-corrected barrier average hurt C10 P
  (p = 0.046) and helped nowhere significantly. Neither is in.
- C20 at 48 seeds: P and G both reach the bowl in 15 of 48 runs. The misses
  end at -120.131 and -120.107 eV. None of the following raised the hit
  rate significantly, so none is in: a temperature multiplier, pair
  rotations, the plane move with hole regrowth, replicas, soft-mode moves,
  and moves towards novel basins.
- ASE minima hopping (`ase.optimize.minimahopping`) reached C10 in 6,168
  calls on seed 2000, but a C20 seed took over 22 minutes and was at
  -114 eV after 741 hops. It is not in the panel.

### Pins and tests

`scripts/covalent_h2h/pins.sh` prints sha1
`419b06def1a0e01a152c13ccdb24cbd9919551cc` on `9631d65a`. On `9631d65a`,
`cargo test` passes everything. `cargo clippy --all-targets -D warnings`
is clean with and without `rgpot-ex`. `pytest pytest/` gives 1306 passed
and 19 skipped. With ChemFit 4.0.0, `pytest pytest/test_chemfit*.py` gives
1005 passed and 4 skipped, and `examples/chemfit_anneal.py` prints best
loss 2.347665788572983.

## GFN2-xTB against CREST

GFN-FF runs on both clusters, but it is a bonded force field whose
topology is fixed from the input geometry, so a search on it cannot
rearrange the cluster. GFN2-xTB has no topology. Both codes search it with
real xtb 6.7.1 (conda-forge build): anneal through rgpot's
`libxtb_engine.so` at the pinned engine revision, and CREST 3.0.2 through
its legacy xtb driver. Silicon clusters have no gap and the SCC fails at
300 K, so both run at an electronic temperature of 3000 K
(`RGPOT_XTB_ETEMP`, 300 K when unset).

- Anneal is the panel's G arm (`examples/gfn2_covalent_h2h`, the
  `general` configuration of `examples/h2h_covalent.rs` on develop
  `bb486bfd`) on scales from the GFN2 dimer, which costs 46 calls on
  silicon and 17 on carbon. It has no pair-range cut, because dispersion
  has no finite reach. The budget is the paper's 1e5 calls, and the run
  stops at the reference.
- CREST is `crest start.xyz --gfn2 -T 1 --legacy -notopo -nocross -shake 0
  -tstep 2`, given the same start after one xtb `--opt` of it. With its
  defaults it does not search a cluster. On the start it stops because its
  first optimisation changed the topology. From the optimised start,
  CREGEN removed 36 of 49 structures as topology changes and the genetic
  crossing crashed on the one left, at -20.73008 Eh against the ring's
  -20.99572. Its default SHAKE also holds every bond during the
  metadynamics. `-shake 0` with a 1 fs step reached -566.88 eV on seed 2000
  in 146,476 calls, and 2 fs is the step the panel uses.
- A CREST call is an MD step (the `max steps` of each xtb MD) or an
  optimisation cycle (each `* total energy` line). `inside` is the summed
  wall of the xtb processes for CREST (from `/proc/uptime` for rows.csv,
  which stops with the machine), and the time in the handle for anneal.
- The reference is the lowest minimum any run reached: -460.68373809 eV
  for Si10, and for C10 -571.32273783 eV, the cumulenic ring with every
  bond at 1.285 Å. A hit is within 1e-3 eV. Seeds 2000-2011 on C10 for
  both codes. On Si10, anneal runs all twelve seeds and CREST, at about
  45 minutes a run, seeds 2000-2003 on core 0 and seeds 2004-2011 in
  `gfn2/rows.csv` on core 4 (`e71e5de9`). Serial, hostname `bench`.

| system | code | hits | median calls | IQR | best / eV | median wall / s | inside / call | outside / call |
|---|---|---|---|---|---|---|---|---|
| Si10 | anneal G | 12/12 | 666 | 268 | -460.683738 | 33.2 | 58.43 ms | 0.010 ms |
| Si10 | CREST | 11/12 | 97,521 | 8,524 | -460.683718 | 2646.6 | 26.72 ms | 0.244 ms |
| C10 | anneal G | 11/12 | 6,350 | 6,077 | -571.322738 | 14.6 | 1.18 ms | 0.002 ms |
| C10 | CREST | 4/12 | 78,367 | 2,880 | -571.322711 | 439.5 | 5.10 ms | 0.288 ms |

CREST's calls are the calls of its whole run, which does not stop at the
reference.

- Si10, with the anneal rows of `seeds.csv` (the paper's Si10 number,
  from the anneal rerun beside CREST, is below): anneal reaches the
  minimum on 12 of 12 seeds and CREST on 11 (Fisher p = 1). On seeds 2000-2003 anneal takes 620, 505, 520 and 2,581
  calls, and CREST 108,264, 96,365, 102,236 and 94,277. On seeds
  2004-2011 CREST takes 103,294, 93,076, 98,677, 94,481, 100,309, 0,
  103,479 and 92,415 calls. On seed 2009 xtb's SCC does not converge on
  the start, with 250 or 1000 iterations, so the optimisation before
  CREST stops and CREST makes no call. Mann-Whitney p = 3.7e-5 over the
  twelve seeds of each code, a miss ranked after every hit
  (`gfn2/summaries.csv`). Anneal's calls cost more inside xtb, 58 ms
  against 27 ms, because its quenches start from random clusters where
  the SCC struggles.
- C10: anneal reaches the ring on 11 of 12 seeds and CREST on 4 (Fisher p
  = 0.0094, Mann-Whitney p = 0.0005). Seven CREST runs end in a compact
  cage with four three-coordinated atoms at -566.877 eV, and its eighth
  miss, seed 2004, in the linear chain at -568.818 eV. Anneal's one miss,
  seed 2003 at the full 1e5 calls, ends in a bicyclic ring with two
  three-coordinated atoms at -566.959 eV. 97,170 of its calls are trial
  geometries with two atoms inside 0.90 Å, which the handle refuses
  before xtb. The example prints the handle's refusals as
  `geometry_rejected`, two to a call because the energy and the gradient
  each reach the handle.

On the machine that ran CREST seeds 2004-2011, anneal does not take the
calls of `seeds.csv`: only seed 2002 matches. Anneal's Si10 seeds
2000-2011 ran again there at `f3d6a050`, serial on core 4 with hostname
`bench`, after the CREST seeds, and are the anneal rows of
`gfn2/rows.csv` at that revision. The summary's `--anneal-revision`
takes only them in `gfn2/summaries.csv`, against the same twelve CREST
rows.

Seed 2009's hit is not an energy of its geometry. Anneal stops at call
745 on -492.59511796 eV, an SCC solution that rgpot's xtb engine reaches
from a reused wavefunction; the same geometry alone in the engine, and
xtb 6.7.1 `--sp --etemp 3000`, give -449.050334 eV, 11.6 eV above the
reference (`gfn2/scc-false-minimum/`, which reproduces it in three
calls). `gfn2/false-hits.csv` names the row, and `--false-hits` counts it
as a miss at the full budget, ranked after every hit. The Si10 number
for the paper is that one:

| system | code | hits | median calls | IQR | best / eV | median wall / s | inside / call | outside / call |
|---|---|---|---|---|---|---|---|---|
| Si10 | anneal G, `f3d6a050` | 11/12 | 943.5 | 1,006 | -460.683738 | 44.9 | 45.06 ms | 0.011 ms |
| Si10 | CREST | 11/12 | 97,521 | 8,524 | -460.683718 | 2646.6 | 26.79 ms | 0.243 ms |

- Anneal reaches the minimum on 11 of 12 seeds and CREST on 11 (Fisher p
  = 1). Mann-Whitney p = 5.3e-4 on calls to the hit, both misses ranked
  after every hit. Anneal's median counts seed 2009 at 1e5 calls.
- Anneal takes 1,191, 1,512, 520, 618, 560, 1,723, 556, 539, 2,344, 745
  (the false hit), 696 and 1,475 calls on seeds 2000-2011, against 620,
  505, 520, 2,581, 801, 778, 306, 712, 601, 728, 1,619 and 302 in
  `seeds.csv`.
- Counting seed 2009 as the hit it records, anneal has 12/12, median
  720.5 calls and IQR 925, and CREST has Fisher p = 1 and Mann-Whitney p =
  3.7e-5, as against `seeds.csv`: every anneal row is below every CREST
  row. Both lines are in `gfn2/summaries.csv`, the one with the false hit
  at `8aeb3b0b` and the one without at `b22eb285`.
- The best geometries of seeds 2002 and 2008 give -460.683718 eV under
  the same xtb single point.

`scripts/covalent_h2h/gfn2_enter_bench.sh` builds the example and runs
both codes (`GFN2_CODES` picks them, `GFN2_CORE` the core). New rows are
appended to `gfn2/rows.csv` with the anneal revision, the host and the
CREST, xtb and engine sha256, and each run appends one line per system
and code to `gfn2/summaries.csv` over `seeds.csv` and `rows.csv`, with
Fisher's exact test on hits and a two-sided Mann-Whitney on calls to the
hit against anneal, a miss ranked after every hit. Neither file is rewritten, and the Snakemake rules
`covalent_gfn2_crest` and `covalent_gfn2_crest_si10_n12` output a stamp
under `/work`. The CREST rows were measured on `8b4a2592` from starts that
are identical byte for byte on `bb486bfd`. On `f2a628c0` anneal takes
the same calls on Si10 seeds 2002, 2006 and 2011 and C10 seeds 2000,
2002 and 2004. A tree with uncommitted changes to the build is refused,
so the revision names the binary.

### With the call floor

The rgpot handle refuses a call with a pair inside 0.90 Å. With the
call floor that limit reaches the search: a trial pair inside it is
opened to 0.945 Å before the quench, and a probe the handle would refuse
is not charged. Anneal's Si10 and C10 rows with it were measured on
`ce1e7a46` (the fix on `f2a628c0`, a local build head), serial on core 0
with hostname `bench`, and are the anneal rows of `gfn2/rows.csv` at
that revision. CREST's rows are those of the first table, from starts
that are identical byte for byte.

| system | code | hits | median calls | IQR | best / eV | median wall / s | inside / call | outside / call |
|---|---|---|---|---|---|---|---|---|
| Si10 | anneal G, `ce1e7a46` | 12/12 | 665.5 | 296 | -460.683738 | 36.1 | 70.88 ms | 0.044 ms |
| Si10 | CREST | 11/12 | 97,521 | 8,524 | -460.683718 | 2646.6 | 26.72 ms | 0.244 ms |
| C10 | anneal G, `ce1e7a46` | 12/12 | 2,516.5 | 1,908 | -571.322738 | 10.9 | 4.43 ms | 0.005 ms |
| C10 | CREST | 4/12 | 78,367 | 2,880 | -571.322711 | 439.5 | 5.10 ms | 0.288 ms |

- Si10: anneal reaches the minimum on 12 of 12 seeds and CREST on 11
  (Fisher p = 1, Mann-Whitney p = 3.7e-5, a miss ranked after every
  hit). On seeds 2000-2003 anneal takes 619, 563, 520 and 2,611 calls.
- C10: anneal reaches the ring on 12 of 12 seeds and CREST on 4 (Fisher
  p = 0.0013, Mann-Whitney p = 2.6e-5). Seed 2003, the miss of the first
  table, takes 3,123 calls. There its refused calls were the screen and
  the quench each starting from a trial with a pair at 0.6-0.8 Å, drawn
  by every move kind. No call is refused on either system
  (`geometry_rejected` is 0 on every seed). A refused call took no time
  in xtb, so the inside time per call is higher than in the first table.
- The summary lines are in `gfn2/summaries.csv`, with
  `--anneal-revision ce1e7a46…`.

### C20 and Si13

The rows are in `gfn2/rows.csv` with the anneal revision in the code
cell. Anneal ran through engine `04aceaa7`, whose sha256 the rows give.
Both codes ran seeds 2000-2011 on C20 and 2000-2005 on Si13, serial on
core 0 with hostname `bench`. On Si13 every anneal call starts its SCC
from xtb's own guess (the engine's warm move set to zero, `ab93b60b`).
The Si10, C10 and C20 rows were measured before that, with the engine
reusing the previous call's wavefunction; their minima give the same
energy from a fresh SCC to 1e-10 eV, but Si13 and Si20 minima do not
(below).

- The C20 reference, -1148.85383633 eV, is a cage with every atom
  three-coordinated and bonds from 1.449 Å. One run found it: anneal
  without the call floor (`91a4aa70`, develop `f2a628c0` with these
  systems), seed 2006 at 1e5 calls, in an untimed comparison. No other run
  of either code has reached it: the 24 timed runs below, and six untimed
  anneal runs on seeds 3000-3005, three at 1e5 calls and three at 3e5.
  xtb 6.7.1 gives the cage the same energy from a fresh SCC, to 1e-9 Eh.
- The Si13 reference, -599.34546694 eV, is the compact minimum anneal
  reached with a fresh SCC on every call on four of six untimed seeds
  (3002-3005 at 30000 calls; 3000 and 3001 were stopped before the end).
  xtb 6.7.1 gives it -22.025539659 Eh.

| system | code | hits | median calls | IQR | best / eV | median wall / s | inside / call | outside / call |
|---|---|---|---|---|---|---|---|---|
| C20 | anneal G, `5351f272` | 0/12 | 100,000 | 0 | -1148.292908 | 1866.6 | 18.54 ms | 0.005 ms |
| C20 | CREST | 0/12 | 78,581.5 | 4,776.8 | -1146.770080 | 1545.1 | 19.30 ms | 0.307 ms |
| Si13 | anneal G, `c1c02d16` | 6/6 | 1,974.5 | 825.8 | -624.704463 | 264.6 | 222.19 ms | 0.031 ms |
| Si13 | CREST | 4/6 | 103,804.5 | 89,756 | -599.345440 | 4191.4 | 47.01 ms | 0.282 ms |

- C20: neither code reaches the cage. Anneal's best is lower than
  CREST's on 134 of the 144 pairs of seeds (Mann-Whitney p = 0.0004 on
  the best energies): -1148.293 eV at best (seed 2000), and -1146.550 eV
  on five seeds, against CREST's -1146.770 eV at best (seed 2004) and
  median -1144.664. No anneal call was refused or failed its SCC. The
  rows were measured on `5351f272` (develop `eae26977` with the call
  floor and these systems, a local build head; `b4ddad0d` on
  `bench/covalent-cov6b` has its tree) with untimed searches on the other
  cores. The example of this series ends seeds 2005 and 2007 at the same
  energies after the same 1e5 calls, and repeats seeds 2000 and 2001 over
  their first 5000 calls.
- Si13: anneal reaches the reference on all six seeds, in 683 to 2,487
  calls, and CREST on four, in 95,213 to 143,630 (Fisher p = 0.45,
  Mann-Whitney p = 0.005, a miss ranked after every hit). CREST's other
  two seeds, 2000 and 2005, end before any search: the SCC of the xtb
  `--opt` of the start does not converge in xtb's 250 iterations, and
  their calls are 0. Measured on `c1c02d16`, serial on core 0 with
  nothing else running. An anneal call costs 222 ms inside xtb against
  CREST's 47 ms: from a fresh SCC, 0.5% to 15% of anneal's calls on these
  seeds (366 of 2,487 on seed 2001) are SCCs that do not converge in 500
  iterations, each about 2 s.
- Anneal stops at the first energy at or below the reference, and seeds
  2003 and 2005 stop below it, at -624.70446339 and -617.18016907 eV. The
  first is an open ring in which every atom has one or two neighbours,
  and xtb 6.7.1 gives it -22.957465589557 Eh from a fresh start. It was
  the lowest Si13 minimum any run had reached until the floor gate below,
  recorded in `gfn2/minima.csv` with its structure in
  `gfn2/minima/si13-ring.xyz`.
  Against it CREST hits on none of its six seeds (`gfn2/rescore.csv`,
  from `--rescore si13 -624.70446339`). Anneal's rows are not rescored:
  each stopped at the reference it ran with.
- Si20 has no panel reference yet. With the engine reusing the previous
  wavefunction, the Si13 and Si20 minima had energies a fresh SCC does
  not reproduce: an Si13 minimum at -601.12421 eV is -593.51232 eV fresh,
  and Si20 minima near -979 eV are near -868 eV. One Si13 search recorded
  a planar sheet at -640.659 eV whose SCC does not converge from a fresh
  start. Before the element floor, 74% of anneal's first 3,000 Si20 calls
  from a fresh SCC were SCCs that do not converge, nearly all on trials
  with a pair at 0.90-0.92 Å, just past the single floor.
- Si20 reference search (`gfn2/si20-capped.csv`): the example on
  `a52726af`, the element floor of the gate below, seeds 3000-3007 at
  1e5 calls with no target, one seed per core on cores 0-7 at once, each
  stopped by a four-hour wall cap (`H2H_WALL_S=14400`). It is untimed and
  not a panel row. Every seed reached the cap, after 4,231 to 14,673
  calls. The lowest energy, -922.98227162 eV, is reached by seeds 3003
  and 3006, the same structure to 2.4e-5 Å in every pair distance: a
  twenty-membered ring, every atom with two neighbours within 2.7 Å and
  no pair below 2.23 Å. Seeds 3000 and 3007 share the next,
  -922.64013443 eV, and the other four end between -922.51325303 and
  -921.99400132 eV. xtb 6.7.1 from a fresh start (`--sp --etemp 3000
  --acc 0.01`) gives every best geometry its energy to 4.1e-5 eV, the
  same offset on all eight. The ring is in `gfn2/minima.csv` and
  `gfn2/minima/si20-ring.xyz`. Two of eight seeds within four hours is
  not yet a reference the panel can score against. The handle counted
  32,704 SCCs that do not converge over the eight runs, at a median
  shortest pair of 2.15 Å, a third of them below 1.6 Å.

### The element call floor

From `95aed412` the example names its element, and each pair's call
floor is 0.70 of the summed covalent radii and never below 0.90 Å:
1.55 Å for Si–Si and 1.06 Å for C–C. A trial pair inside it is opened to
1.05 times that floor. The rule has no per-system constant, and an
objective that names no elements keeps the single floor, so the pins do
not change. `95aed412` is the floor on `c1c02d16`, pushed as `eac995e8`
on `bench/cov-floor2`; this series carries the same change on this base.

- It was gated against `c1c02d16` on seeds 2000-2011 of Si13 and C20 at
  1e5 calls, both arms in one untimed session with eight runs at a time
  (`gfn2/floor_gate.csv`). On Si13 both arms hit on all twelve seeds.
  1.44% of the floor arm's calls are SCCs that do not converge, against
  3.02%; the rows count each such call twice, 2.89% and 6.05%, as every
  build before `7136fcc8` does: it ran a failed fresh SCC a second time.
  Its median calls to hit is 2,207 against 1,897 (Mann-Whitney p = 0.45).
  On C20 neither arm reaches the cage, no call fails its SCC in either, and the median best is -1146.550 eV in both (p = 0.86). The
  `c1c02d16` arm repeats the timed Si13 calls to hit seed for seed. The
  Si10, C10, Si13 and C20 rows above are without it.
- The same gate against develop `3c189e92` (`floor-gate/`, untimed,
  eight runs at a time), on this series as it was based on `3c189e92`
  (`bench/cov-floor-develop`). Develop's example has no Si13 or C20, so
  the GFN2 arms are `98a69388`, develop with the series up to the Si13
  rows, and `a52726af`, with the floor (`floor-gate/gfn2.csv`), pushed
  as `51abe20f` and `c3566383` with the same trees. Both arms repeat the
  `c1c02d16` gate seed for seed: on Si13 12/12 hits in each, SCC failures
  1.44% of calls against 3.02%, median calls to hit 2,207 against 1,897
  (p = 0.45). The floor arm takes more calls on six seeds, fewer on five
  and the same on one. On C20 the two arms are 0/12 with no SCC failure,
  median best -1146.550 eV (p = 0.86). The paper headline panel
  (`scripts/paper_0120/arm_gate.py`, LJ38 to LJ150, Au55 and M80 at ρ=6
  and 14, seeds 2000-2011, the paper budgets) ran on `3c189e92` and
  `a52726af`: every one of the 144 seed pairs has the same hit, calls to
  the minimum, calls charged and best energy (`floor-gate/panel.csv`).
  The pins (`floor-gate/pins-a52726af.txt`, sha1 `a22093e7…`) and the
  Au55 pins (`floor-gate/au55-pins-a52726af.txt`) are byte for byte
  those of `bench/paper-0120` on develop.

### The floor gate on `c2de652d`

On this series rebased on `c2de652d`, three example builds ran in one
untimed session at 1e5 calls, seven runs at a time on cores 1-7 while
core 0 ran CREST (`gfn2/floor-gate-c2de652d.csv`, one row per arm and
seed): `3be9554e`, the series up to the Si13 rows, with no element
floor; `ccd69256`, the element floor; and `e5e21199`, the floor with the
line-search spacing below. `c2de652d` itself has no Si13 or C20 in the
example. Each build before `7136fcc8` counts a failed SCC twice, and the
counts below are halved for them. In short: the floor takes 16% more
median Si13 calls at twelve seeds and 9% more at sixty, neither
significant, and with the spacing the example takes as many Si13 calls
as the base, fails a quarter as many SCCs, and is the floor on Si10 and
C20.

- Si13, seeds 2000-2011: both arms hit on all twelve. The floor's median
  calls to hit is 2,206.5 against 1,896.5 (Mann-Whitney p = 0.45), and
  1.44% of its calls fail their SCC against 3.02% (p = 0.30 over the
  per-seed rates), the numbers of `gfn2/floor_gate.csv`.
- Why the floor took more calls there, from builds that trace every
  trial: the two arms part at the first hop or the second on eleven of
  the twelve seeds, at the first trial with a pair inside 1.55 Å, which
  the floor opens and the base hands to xtb as drawn; seed 2009 is the
  same in both. From there each seed's calls to hit is a separate draw.
  The base's hit came from such a trial on seeds 2000, 2004 and 2010,
  shortest pair 1.135, 1.014 and 1.081 Å, and the floor opened each of
  those. The floor's own hit came from a trial it opened on six seeds.
  The floor refuses trials that lead the base to the hit and opens others
  that lead it there.
- Seeds 2012-2059 added to both arms give 60 of 60 hits in each, a median
  of 1,665.5 calls against 1,525 (p = 0.76), and SCC failures in 1.76% of
  calls against 2.85% (p = 0.035). The 16% at twelve seeds is the spread
  of those draws. The halving at twelve was mostly three base seeds, 2008,
  2001 and 2011, with 319, 183 and 143 of its 823 failed SCCs.
- Where the failures are: of the base's failed Si13 SCCs on 2000-2011, 28%
  had their shortest pair at 0.90-1.55 Å, which the floor keeps out of a
  trial's start, and 62% at 2.0 Å or more, where no floor acts. Of the
  failed SCCs of all Si10 and Si13 runs, 63% had a shortest pair within
  0.002 Å of the failure before them: a line search bisecting towards
  its accepted step, each probe an SCC that fails. On Si10 seed 2000 the
  base took 16,819 calls and five hours to hit, 4,528 of them failed
  SCCs, two thirds of those so; the floor hit in 1,019 calls with 7.
- `0ebf7300` gives `WarmLbfgs` a refusal spacing, and `e5e21199` sets it
  to 0.01 Å in the example: a line search asks no probe within 0.01 Å, in
  every coordinate, of a probe just refused, and ends at its accepted
  step. On Si13 over seeds 2000-2059 it hits on 60 of 60, median 1,517
  calls against the base's 1,525 (p = 0.92), and 0.66% of its calls fail
  their SCC against 2.85% (p = 0.0011) and the floor's 1.76%. Its rows are
  the floor's on 42 of the 60 seeds, the same calls, hit, energy and
  halved failed SCCs; on the other 18 it has fewer failed SCCs on 15 and
  as many on 3, never more. On Si10 and C20, seeds 2000-2011, every row
  is the floor's.
- The floor and the spacing both stop Si13 seed 2031 at -632.49855758
  eV, 7.8 eV below the ring, on the same geometry: a branched chain of
  thirteen atoms with one to three neighbours within 2.7 Å and a shortest
  pair of 1.63 Å. A fresh xtb 6.7.1 SCC at 3000 K does not converge there in its
  default 250 iterations; with the engine's 500 it converges at the
  265th to -23.243893267921 Eh, -632.49852977 eV, so the hit is the
  geometry's energy. It is in `gfn2/minima.csv` and
  `gfn2/minima/si13-2031.xyz`. A fresh single point of every other best
  geometry in the gate gives its recorded energy to 0.01 eV.
- Si10: every arm hits on 12 of 12. The floor and the spacing take a
  median of 1,094.5 calls against the base's 1,001.5 (p = 0.86), and
  0.33% of their calls fail their SCC against 15.81%, 1.00% without seed
  2000 (p = 0.023 over the per-seed rates). C20: no arm reaches the cage, no call fails its SCC,
  and the median best is -1146.550 eV in all three (floor against base p
  = 0.86).
- A third arm opened only pairs inside 0.90 Å, to 1.05 times the element
  floor, and left 0.90-1.55 Å to xtb. Over seeds 2000-2011 it took a
  median of 2,541.5 Si13 calls with 3.39% failing, and is not kept; its
  rows are not in the file, as its build is not on this branch.

`scripts/covalent_h2h/gfn2_floor_gate.py` runs a same-session gate of
any number of example builds on named cores and summarises every arm
against the first.

### The Si20 panel

The panel is anneal against CREST on Si20 at the paper budget, 4e5
calls, serial on core 0 under the `bench` hostname, CREST first, then
anneal with `H2H_WALL_S` at three times CREST's slowest wall
(`gfn2/si20-panel.csv`). It is not finished.

- The seeds. On seven of 2000-2011, 2000, 2001, 2002, 2004, 2006, 2007
  and 2011, xtb 6.7.1 at 3000 K does not converge the start's SCC in its
  default 250 iterations, and CREST, whose xtb runs with that default,
  ends with no call. `--check-starts` writes a fresh single point of each
  start (`gfn2/si20-starts.csv`, seeds 2000-2021), and the panel takes the
  first twelve seeds whose start converges: 2003, 2005, 2008, 2009, 2010,
  2012, 2013, 2014, 2018, 2019, 2020 and 2021. The rows of the other
  seeds stay in the file and the summary leaves them out.
- CREST stopped seeds 2008 and 2010 after 219 calls and 2012 after 245 on
  a topology change: a random start is not a bonded molecule, and its
  first optimisation forms bonds the start did not have. From `d09da9d9`
  CREST runs with `-noreftopo`, which in CREST 3.0.2 only makes that
  check adopt the new topology and go on. `gfn2/superseded.csv` names the
  three rows; they are run again, and the summary and the wall cap leave
  the old rows out.
- Seed 2009: no hit after 183,839 calls and 15,311 s, best -922.038 eV,
  above the ring.
- Seed 2003 is censored, not a miss or a timeout. CREST was still
  running after 17,402 s, but had made no xtb call for 52 minutes and was
  using one CPU to write binary process memory to
  `crest_rotamers_0.xyz`; the file was 174,915,166,208 bytes when the
  process was stopped to recover the full disk. Its xtb log held 1,356
  records. No final energy or exit code exists, so
  `gfn2/si20-censoring.csv` records neither.
- Seed 2005 records a hit at call 108,256 (13,710 s) at -959.85629642
  eV, 36.9 eV below the ring, on a planar flake with a shortest pair of
  1.846 Å. xtb at 3000 K does not converge that geometry's SCC from a
  fresh start in 250 or 500 iterations, and with 1,000 its Broyden step
  fails; at xtb's default 300 K it converges to -31.934707740950 Eh,
  -868.99 eV. The energy is that of CREST's optimisation, whose SCC
  starts from the step before. `gfn2/false-hits.csv` names the row, and
  the summary counts it as a miss at its budget.
- Both arms stop at the ring, -922.98227162 eV
  (`gfn2/minima/si20-ring.xyz`). It is the panel's reference only if
  anneal or CREST reaches it on more seeds than the two of the reference
  search; otherwise the panel reports best energies, not hits.

## Pins

`scripts/covalent_h2h/pins.sh` prints, with every clock column removed:

- trace and lowest-geometry hashes of whole-budget `h2h_seed` runs: LJ13,
  LJ38, LJ55 and LJ75, ρ=14 Morse 38, and TIP4P 8 and 12;
- `h2h_morse` rows for ρ=6 and ρ=14 at 38 and 55 atoms;
- `h2h_metal` rows for Zn38, Cu19Au19 and Pt13.

Before and after the fix, the output's sha1 is
`419b06def1a0e01a152c13ccdb24cbd9919551cc`. `tests/lj_trajectory_pin.rs` passes.

On develop `35115269` the sha1 is `97677e2820a32d62d21c5b45e3a62138c69ec3ce`.
Three first-parent commits changed it:

- `b73ff6c3` (`7a2bfeb1…`): the ρ=14 Morse 55 hit on seed 2000 is
  recorded at the charged stop, 255670 calls in place of 1000000.
- `bb486bfd` (`254b635a…`, the squash of #59): every `h2h_seed` trace
  and the Lennard-Jones and Morse lowest geometries change; TIP4P's do
  not. On the branch, `9d670da6` charges each quench point once, and
  `72e8d0d8`, `f70b5354` and `bc50572f` move the seeded draws and
  rotations to portable libm. ρ=14 Morse 55 drops from 2 to 1 hit in 3
  (seed 2000 misses) and Zn38 from 2 to 0.
- `3c45b899` (`97677e28…`, merged in `9abb056e`): the recommended full
  quench runs 1000 steps. ρ=14 Morse 55 hits on all 3 seeds. Zn38 stays
  at 0 of 3.

## Files

- `before/`, `after/`: per-seed rows and summaries of the first panel.
- `v2/`: per-seed rows and summaries of the second panel.
- `v3/`: per-seed rows and summaries of the third panel.
- `reference.csv`: the unstopped reference runs.
- `tc20-bowl.xyz`, `tc20-flake.xyz`: the C20 bowl and the best unbiased flake.
- `c10-96/`: the 96-seed C10 rows.
- `gfn2/`: per-seed rows and summaries of GFN2-xTB Si10, C10, Si13 and
  C20, anneal against CREST, CREST's rescore against the Si13 ring, the
  lowest minima with their xtb energies, the element-floor gates on
  `c1c02d16` and `c2de652d`, the capped Si20 reference search with a
  fresh xtb single point of each best geometry, and the Si20 panel's rows
  so far, its start check, and the rows it supersedes or counts as false
  hits.
- `floor-gate/`: the element-floor gate against develop, GFN2 and the
  paper headline panel, with the pins of the floor build.
