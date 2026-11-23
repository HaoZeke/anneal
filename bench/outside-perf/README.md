# Outside-potential performance

This directory compares develop `61fa67cf380e46876b569da45e7cc091647a775b`
with the candidate and with a second separate build of that same develop
source. The repeated build is the A/A control.

## Reproduction

Run:

```console
snakemake -s workflow/Snakefile outside_perf
```

The rule builds develop twice and builds the checked-out candidate once.
It then runs one serial session on core 0. Thread-count variables are set
to one. There is no warm-up run. Seeds 2000 through 2011 are used for
LJ150, LJ104, M80 at rho 6, M80 at rho 14, Au55, BLJ55 sigma 1.30, and
BLJ38 sigma 1.05.

Within every block of four seeds the arm order is forward, reversed,
reversed, forward. The next block starts on the next arm. Each arm
therefore occupies each position four times. The outputs are:

- `rows.csv`: raw inside and outside timings, core, affinity, and the
  sha256 digest of the machine id;
- `summary.csv`: outside-time quartiles, relative changes, and
  tie-corrected two-sided Mann-Whitney results for the candidate and the
  repeated develop build;
- `trajectory-diff.csv`: hit, calls, charged count, and minimum for every
  seed.

The five non-BLJ systems use their paper stopping rule and budget. Both
BLJ systems use the committed reference minima and paper budgets, while
timing forces every arm through 500,000 charged calls. Their outside
result is seconds per 100,000 calls.
