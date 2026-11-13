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

`0555fd23` refuses a quench that has shed a free atom. A free atom has no
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

## Pins

`scripts/covalent_h2h/pins.sh` prints, with every clock column removed:

- trace and lowest-geometry hashes of whole-budget `h2h_seed` runs: LJ13,
  LJ38, LJ55 and LJ75, ρ=14 Morse 38, and TIP4P 8 and 12;
- `h2h_morse` rows for ρ=6 and ρ=14 at 38 and 55 atoms;
- `h2h_metal` rows for Zn38, Cu19Au19 and Pt13.

Before and after the fix, the output's sha1 is
`419b06def1a0e01a152c13ccdb24cbd9919551cc`. `tests/lj_trajectory_pin.rs` passes.

## Files

- `before/`, `after/`: per-seed rows and summaries.
- `reference.csv`: the unstopped reference runs.
- `tc20-bowl.xyz`, `tc20-flake.xyz`: the C20 bowl and the best unbiased flake.
- `c10-96/`: the 96-seed C10 rows.
