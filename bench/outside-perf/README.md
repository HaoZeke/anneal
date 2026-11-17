# Outside-potential performance

This directory compares `0ae5bfe0efb874a718f8e9632d18ff5fbc29f782`
with the optimized revision. It does not replace or modify the GMIN-panel
measurements.

## Profiling

Callgrind profiles used seed 2000 and a 5,000-call diagnostic budget for
LJ150, LJ104, M80 at rho=6, and Au55. Each release binary ran on one core
with all thread-count variables set to one. The profiles showed that the
important bookkeeping costs were:

- the dense shortest-path assignment scan for LJ150;
- Kuhn-Munkres phase-delta scans for M80 at rho=6;
- continuous-symmetry projection and orbit completion for LJ150 and LJ104;
- minimum-cost assignment for Au55.

SOAP cost construction was visible but smaller than assignment on the large
Lennard-Jones systems. Archive, bias, fingerprint, cloning, and allocation
costs were secondary. In particular, the earlier M80 timing regression came
from changing descriptor handling despite assignment being its dominant
bookkeeping path. That change is absent here.

The final instruction counts were lower on every diagnostic:

| system | base instructions | new instructions | change |
|---|---:|---:|---:|
| LJ150 | 3,402,849,428 | 3,355,184,454 | -1.401% |
| LJ104 | 1,872,317,042 | 1,839,334,796 | -1.762% |
| M80 rho=6 | 2,525,513,706 | 2,458,181,555 | -2.666% |
| Au55 | 2,322,867,950 | 2,291,786,367 | -1.338% |

The optimized assignment code reuses solver storage, avoids materializing a
negated cost matrix, skips fully used AVX2 lanes, rejects four-lane blocks
that cannot lower the current shortest-path minimum, and avoids second scans
of strict row minima. Orbit completion avoids square roots except at the
rounding boundary, where it retains the original comparison. The final
profiles and timings retain the original floating-point operation order at
all decisions that can affect a trajectory.

## Reproduction

Run:

```console
snakemake -s workflow/Snakefile outside_perf
```

The rule builds the base and current release examples on the same machine,
then enters a UTS namespace named `bench`. It runs base then new on CPU 0 for
every seed from 2000 through 2011, one process at a time. The outputs are:

- `rows.csv`: raw same-session inside/outside timings and binary identities;
- `summary.csv`: medians, relative changes, and tie-corrected two-sided
  Mann-Whitney results;
- `trajectory-diff.csv`: exact comparisons of `evals_to_hit`, `charged`, and
  `best`.
