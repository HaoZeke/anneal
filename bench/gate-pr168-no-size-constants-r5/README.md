# Gate for the no-size-constants tree

Rules below were fixed before any timed run. The verdict section is
the only part filled in afterwards.

## What reaches the changed code

Compared at the base tree `cff796c507a8fa1d0bfb28888c3570b61988055e`
and the candidate tree `113a3420d7770508519da9b5257d519d3b0fb829`.

The three stock examples call `run_with_gradient` and pass their own
L-BFGS relax. They do not read `cfg.surfaces`.

- `examples/h2h_gmin_proof.rs:327`
- `examples/h2h_morse.rs:295`
- `examples/h2h_metal.rs:536`

`run_with_gradient` is `src/methods/cluster_hopping.rs:2675` on the
candidate. The new ledger flags `block_opened`, `revert_unrestrained`
and `suspend_record` are written only from
`src/methods/cluster_search.rs:910`, `:913` and `:1194`. Nothing on
the stock path sets them, so those branches stay false.

That part of the probe is right for these three examples, and wrong
as a claim about the whole diff. The hop function changed rigid-body
acceptance at `src/methods/cluster_hopping.rs:8117` on the candidate.
`cluster_search` does not call `run_with_gradient`, so that edit is
not on the `search` or `search_from` path. No cell below enters it.
The stock examples are atomic. `h2h_seed`'s water arm
(`examples/h2h_seed.rs:158` on both trees) uses `recommended_tip4p`,
which starts from `for_tip4p` and sets a rigid library there
(`config.rs:2314` and `:2321` on the base tree, `:2311` and `:2318`
on the candidate) without calling `with_rigid_bodies`. `with_scales`
leaves `surfaces` empty and `two_phase` unset (`config.rs:2273` and
`:2269` on the base tree, `:2270` and `:2266` on the candidate), so
those cells take the plain descent in `cluster_search.rs:879` on the
candidate. `with_rigid_bodies` does clear the surface
(`config.rs:2424` on the base tree, `:2421` on the candidate), and
this gate does not call it. Bit identity of the atomic stock cells is
still measured, not assumed.

The size gate is the part the stock examples cannot see. On the base
tree `Config::recommended_with_scales` and `Config::communicating`
install the two-phase surface only when `n_points < 98`
(`src/methods/cluster_hopping/config.rs:1849` and `:1897`). On the
candidate the same assignments are unconditional
(`src/methods/cluster_hopping/config.rs:1852` and `:1896`). Only
`cluster_search::search` (`src/methods/cluster_search.rs:1685`) and
`search_from` (`:1777`) read that portfolio.

Callers of that entry on the base tree include
`examples/h2h_seed.rs:265` and `:276`, `examples/paper_controls.rs:371`,
`:375` and `:377`, `examples/lj_digest.rs:54`,
`examples/morse_table.rs:137`, `examples/ccd_hits.rs:239`, and
`src/python.rs:3579` (the calls at `:3879` through `:3886`). Other
examples call the same entry and are not cells of this gate.

## Cells

Seeds are 2000 through 2023, twenty-four seeds, on every measured
cell. Arms are A, A' and B. A' is recorded as `Ap`. A and A' are two
separate release builds of the base tree. B is a release build of the
candidate tree above. The candidate is checked by that tree and not
by a commit id.

A hit is a finite best at or below the published reference plus
0.001. Budgets and references come from the panel harness that names
the cell. No `H2H_*` variable is set. The build is
`cargo build --release --locked` with toolchain 1.88.0.

Unreached guards use the stock example. Reached guards and the added
cells use `h2h_seed` with arm `recommended`. The stop passed to
`h2h_seed` is the reference plus 0.001, the same addition the panel
harness uses. `guard_trace.rs` is the copied helper; its hexamer
budget is 500000 and its reference is -197.7805260034, and the
hexamer cell uses those values through `h2h_seed` because the helper
does not report time inside the potential.

Unreached guards, stock driver, atomic, role `guard`:

- Lennard-Jones from `scripts/gmin_panel`: lj38, lj55, lj75, lj98,
  lj102, lj103, lj104, lj150, through `h2h_gmin_proof`. The panel's
  binary name for lj103 is `h2h_gmin_proof_lj103`, the label of a
  patch used when the example had no lj103 entry. At this tree the
  entry is `examples/h2h_gmin_proof.rs:256`, with the panel budget, so
  the guard runs that example. The budget command has to match the
  panel or the cell is missing.
- au55 from that panel, through `h2h_metal`.
- Every Morse cell in `scripts/morse_gmin`: m30r6, m30r14, m38r6,
  m38r14, m55r6, m55r14, m80r3, m80r6, m80r14, through `h2h_morse`.

Reached guards, role `guard`, because the driver is `search_from`:

- hexamer, as above.
- Every TIP4P size named by a committed panel harness with a published
  reference: tip4p4, tip4p8, tip4p12, tip4p15, tip4p16, tip4p20,
  tip4p21. The budget is the harness budget, 2000000.

Added cells, role `reached`, `h2h_seed` arm `recommended`:

- lj75-seed, the `scripts/paper_0120` LJ75 run.
- lj103-seed, m38r6-seed, m55r14-seed, at the panel budget and
  reference of those systems.

`lj103-seed` is the size the base preset leaves without a two-phase
surface and the candidate preset equips. `lj75-seed`, `m38r6-seed`
and `m55r14-seed` already have that surface on both trees. The stock
lj75, lj103, m38r6 and m55r14 rows stay on the unreached drivers.

Not run, recorded as missing:

- au55-seed. `h2h_seed` accepts lj, morse or tip4p and has no gold
  potential, on both trees.
- tip4p25. `scripts/tip4p_h2h/run_h2h.py` lists it and scores the hit
  against the session minimum, not a published reference.
- Doye-Meyer binary clusters. They are fixture checks, not cells of a
  committed panel harness.

## Design

Within each cell and seed the arms run in order A, B, Ap. The next
seed reverses that order. A later slot of the same cell and seed does
not start until the earlier slot has a row. Distinct cells may run
together.

At most three lanes. Lane k is pinned with `taskset -c k`. A row
whose recorded affinity is not that core fails its cell. Every thread
count is 1. If available memory is under 8 GiB the cap is two lanes;
under 4 GiB it is one. The cap is read again before each launch.
There is no warm-up and no quick mode. A job that is still running
after 7200 s is a failure with exit code 124. A job killed by signal
9 is exit code 137. The peak resident set is the largest `VmHWM`
observed for that process, in KiB.

An unparsed or nonzero-exit row is kept. Its hit is a miss. A cell
with any such row does not pass. Error rows are not deleted and not
repeated. Hit counts, including the A/A hit band and McNemar, count
that miss. Charged-call medians and outside-time medians use only
rows that exited 0 and parsed.

## Pass rule

A measured cell passes only when all three hold.

1. Exact two-sided McNemar on the paired hits, the binomial test on
   the discordant pairs, does not show B worse at p < 0.05. B is worse
   on that test when A-only pairs outnumber B-only pairs and the
   exact p is below 0.05. Also, B's hit count is not below A's by more
   than A' differs from A in any measured cell. That allowance is the
   maximum of the absolute A' minus A hit counts.
2. The ratio of median charged calls, B over A, is not above the A/A
   band unless B has strictly more hits than A. The band is the
   maximum over measured cells of max(r, 1/r), where r is the ratio of
   median charged calls, A' over A. A zero denominator is an infinite
   ratio when the numerator is positive, and 1 when both are zero.
3. The ratio of the median of outside time per charged call, B over
   A, is at most 1.10. The per-call value is outside_s divided by
   charged. Rows with no positive charged count are left out of that
   median.

Misses are hits of 0. For the Wilcoxon signed-rank report, a miss is
ranked worse than every hit by using budget + 1 in place of its
charged count. That test is reported and is not a pass condition.
The signed-rank p is the exact two-sided midrank distribution under
independent signs. All-zero differences have p = 1.

Medians are the ordinary even-sample median. Bootstrap intervals are
percentile intervals of the paired median ratio, 10000 draws, indexes
int(9999 * 0.025) and int(9999 * 0.975) on the sorted draws. The
generator starts at 20261120 and advances one cell at a time in the
cell order above. Intervals are reported for median calls, median
wall time, and median outside time per charged call, for B/A and
A'/A. They are not pass conditions.

Bit identity is whether, on every seed, the printed best, the charged
count and the calls count of A and B are the same. `best_bits` is the
IEEE-754 hex of that printed best, not a digit the driver withheld.
For the unreached guards this is the check that B matches A. It is
not a pass condition. The pass rule is the three tests above, on
every measured cell, guards included.

## Provenance

Pin checks and the tree check run before any timed search. The
machine id stored with the rows is the SHA-256 of the machine id
file. Rows record the tree of the build, the SHA-256 of the binary,
the core, the affinity, the peak resident set and the exit code.

<!-- verdict -->

## Verdict

Not yet run.
