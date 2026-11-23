# Trail-arm default-on gate

This protocol is committed before the harness and every result row.

## Base and arms

The delivery base is commit
`8868e7f0f5b2c1f47ed4809b06650b7f768a9a8d`, tree
`b545d16ae0cebadf84aaf6e7ad4bc805472874ec`.

- A is `364679f060386949f805a72dea9d7d63b7e9ea4f`, tree
  `79610c980d39f19a2d383ffe3c53e4505ddb9b6c`.
- A2 is an independent release build of A in another worktree and
  target directory.
- Canonical B is A plus the validated trail-arm patch, commit
  `7b20cd761457c3644c424a75a58e9d535e18aa63`, tree
  `34c4de6ad5d38a06b30f0c59cd2baae36c1fa613`.
- Measured B adds only a gate counter, commit
  `a9d29043288d5c38609df2d145f4a7521121f659`, tree
  `3e7a8865393be494fd84bea68ee1011798b2ea0c`.

The counter uses thread-local integer cells. It counts entries to the
stall bandit and idle trail redraws, then prints the totals during
process teardown when `ANNEAL_TRAIL_ARM_GATE=1`. It does not consume
random numbers or change control flow.

All three arms are separate worktrees. The gate harness is a fourth
worktree. Only the anneal implementations are run. No GMIN, ASE, or
other competitor program is built, launched, downloaded, or read.

## Panel

The 28 cells are:

- tip4p9 through tip4p21, and tip4p25;
- lj38, lj55, lj75, lj98, lj102, lj103, lj104, and lj150;
- lj75-seed;
- m55r6, m55r14, m80r6, and m80r14;
- au55-seed.

Every cell uses seeds 2000 through 2011. TIP4P cells use 2,000,000
calls. LJ38 uses 400,000 calls. LJ55, both Morse-55 cells, and
au55-seed use 1,000,000 calls. The remaining cells use 4,000,000
calls. References are fixed in the harness.

Runs are serial release builds pinned to one available core. All known
thread-count variables are one. There is no quick mode, retry,
exclusion, or replacement. Starts are shared by seed.

The six seed-wise orders are `A-B-A2`, `A2-B-A`, `B-A2-A`,
`A-A2-B`, `A2-A-B`, and `B-A-A2`, repeated once. Thus each arm takes
each position four times in every cell.

Rows are append-only, flushed, and synced individually. Each records
the arm, canonical and measured trees, seed, position, outside time,
calls, hit, final energy bits, in-process `RssAnon`, stall triggers,
and idle trail redraws. A preloaded teardown probe reads
`/proc/self/status` from inside the measured process.

## Analysis

For each cell, misses are scored as `budget + 1` calls. The report
includes hit counts, a one-sided Fisher exact probability for A having
more hits than B, and a one-sided Mann-Whitney probability for B using
more calls.

Paired seed bootstraps use 10,000 deterministic resamples and type-7
2.5% and 97.5% quantiles. They report median-ratio intervals for raw
outside time and in-process `RssAnon`, both for B/A and A2/A.

The only verdict rules are:

- no cell may lose hits from A to B;
- every B/A outside-time interval upper bound must be at most 1.10;
- memory fails only when the B/A interval lower bound exceeds both
  1.10 and the A2/A interval upper bound.

The time result is also reported against
`max(1.10, A2/A upper)` as a diagnostic, not another verdict rule.

For each cell, the report counts seeds with zero stall-bandit triggers.
For those seeds it checks that A and B have identical final-energy
bits and calls.
