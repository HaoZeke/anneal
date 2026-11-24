# Timing gate protocol

This statement was written before any timing row. Later rows are appended
and are not rewritten. A partial seal reports the row count of each table.

## Arms

- A is one release build of the base revision.
- A2 is a second independent release build of the same base sources.
- B is a release build of the candidate revision.
- Each arm has its own source tree and build directory.
- Long-range support and the benchmark targets are enabled in every build.
- The DSF harness source and the compile settings match in every arm.
- Fortran pots and the fast pair list are disabled in every build.
- The measurement is one session on one machine. Builds finish before the
  first timed round.

## Machine controls

- The process is pinned to logical CPUs 0 through 7 for the primary series.
- Dynamic OpenMP teams are disabled.
- Thread counts are 1, 2, 4, and 8.
- Every BLAS thread limit that is set is 1.
- Primary OpenMP wait policy is ACTIVE.
- Primary OpenMP process binding is close, with places set to cores.
- The machine table records the core count, whether the avx2 flag is
  present, the wait policy, the process binding, and the pin set.
- It does not record a hostname.
- There is no quick mode. Every timed point uses 25 repetitions.

## Workload

- Cells are Ewald, PME, and DSF.
- Sizes are 128, 256, 512, and 1024 atoms.
- Ewald and PME use the long-range physical benchmark on the strong ladder
  only, with its default tolerance and with no matched-tolerance column.
- DSF uses the DSF harness and times energy plus forces.
- Each visit of an arm runs every cell once, in size order and then in
  thread order, Ewald before PME, and DSF after both.
- A round is one visit. Each arm is visited 10 times.
- Visit order is the palindrome A, B, A2, A2, B, A, repeated five times.
- That block is itself a palindrome, and the five repeats stay one.
- Raw rows keep every measured wall time in microseconds, with the energy
  of that point.
- Geometries use the benchmark seeds and the benchmark charge pattern.
- Five warmups inside the long-range benchmark, and five inside the DSF
  harness, stay out of the 25 repetitions.

## Stalls

A repetition is an outlier when it is at least 8 times the median of its
own arm, round, and cell, and at least 4 ms above that median. A round
stalls when 5 or more repetitions are outliers.

A round also stalls when its median is at least 8 times the paired round
median on A, at least 4 ms above that median, and its interquartile range
is at most a quarter of its own median. Arm A uses only the outlier rule.

A cell has a B-only stall when some B round stalls and no A round of that
cell stalls.

If a B-only stall appears, that cell is rerun for 10 rounds and 25
repetitions in the same palindrome, twice:

- passive: wait policy PASSIVE, still pinned the same way
- free: wait policy ACTIVE, process not pinned, places not bound, and
  per-thread affinity requests succeed without pinning

A stall counts for the verdict only when the B-only stall remains in the
passive rerun and in the free rerun. If either rerun removes it, the cell
is an environment tick. The five-percent test for that cell then uses the
passive rows when those have no B-only stall, and otherwise the free rows.
Every other cell uses the primary rows.

## Correctness

For every method, size, and thread count, energies and force components
from B are compared with A at full precision. The cell agrees when the
absolute energy difference is at most 1e-4 eV and every force component
differs by at most 1e-5 eV/A. Differences at 1e-9 and below are agreement.
Correctness does not change the timing verdict.

## Statistics

Ratios pair the k-th round and the r-th repetition of B with the same
indexes of A, and likewise for A2 against A. The reported effect is the
geometric mean of those ratios. Each cell draws 10,000 bootstrap resamples
of the paired indexes with a generator reseeded to 20261123, and reports
the 2.5 and 97.5 percentiles.

A cell is a real regression only when the B/A lower limit is above both
1 and the A2/A upper limit.

The timing verdict is HOLD when any Ewald or PME cell is a real regression
whose geometric-mean B/A is above 1.05, or when any stall counts under the
rule above. Otherwise the verdict is PASS.

DSF is not part of that HOLD rule. For DSF at 2, 4, and 8 threads the
speedup is one minus the B/A geometric mean, listed against the claim of
17 to 39 percent faster.
