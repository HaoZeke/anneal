# PR 372 timing gate protocol

The protocol was fixed before collecting timing data.

## Arms

- A is the requested base revision.
- A2 is a second independent release build of the base revision.
- B is the five-patch change applied to the base revision.
- Every arm has its own source tree and build directory.
- Long-range support and benchmark targets are enabled in every build.
- The DSF harness source and compile settings are identical in every arm.

## Machine controls

- All runs use the same eight logical CPUs.
- Process affinity is fixed to CPUs 0 through 7.
- Dynamic OpenMP teams are disabled.
- OpenMP thread counts are 1, 2, 4, and 8.
- BLAS implementations are limited to one thread.
- The machine metadata records the processor model, CPU count, and core set.

## Workload

- Ewald and PME use the existing long-range physical benchmark.
- DSF uses `dsf_phys_bench.cc` and times energy plus forces.
- Strong-scaling cells contain 128, 256, and 512 atoms.
- Each point has 25 measured repetitions after five warmups.
- Full rounds follow the order A, B, A2, B, A, A2.
- Raw data retain every measured wall time in microseconds.
- Geometries use the benchmark's deterministic seeds and charge pattern.

## Correctness

For each method and cell, B and A energies and force vectors are compared.
Energy and force differences are reported at full precision. Differences
below normal floating-point reduction noise are treated as agreement, not
as physical changes.

## Statistics

Each cell reports the median for A, A2, and B. Ratios pair observations by
round position and repetition, then use the geometric mean. Confidence
intervals use 10,000 paired bootstrap resamples with seed 20261123. The
Mann-Whitney U test is two-sided.

A cell is a real regression only when the lower confidence limit for B/A
is above both 1.0 and the upper confidence limit for A2/A. The overall
verdict is FIX only when an Ewald or PME real regression exceeds 5%;
otherwise it is PASS.
