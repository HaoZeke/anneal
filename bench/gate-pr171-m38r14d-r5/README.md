# Gate for morse m38 rho 14, round 5

BASE is 61fa67cf380e46876b569da45e7cc091647a775b,
tree cff796c507a8fa1d0bfb28888c3570b61988055e.
Arms A and A2 are separate release builds of BASE.
Arms B and B2 use
633d5ce5e8176ebd99cfe13f779a10752a059910,
tree a5a1423fb2c00475962770dd792f37944cf61174.
The harness stops before measurement if these ids
or trees differ.

## Superseded round-four invariants

Round four remains FAIL under its pre-registered
rule. Its 864 rows are immutable. Round five does
not relabel or rescore that verdict.

Three round-four additions are superseded here.
Comparing ledger spent with observed calls mixed
two counters. Printed best can lag seen.lowest when
a stop fires during a quench. Requiring B search
identity with A also rejects the target cell by
construction. Round five reports those differences
but does not bind on them.

## Cells and protocol

The nine cells, budgets, references, and seeds
2000 through 2023 are unchanged. Each seed runs
A, B, B2, A2 on core 0. Release binaries are used.
Every declared thread-count variable is one. There
is no quick mode, warm-up, retuning, rescuing,
hand-picking, or relabeling.

Rows are append-only. A non-zero or incomplete
attempt remains in the file. At most two replacement
attempts may follow for a cell and arm row. The first
conclusive attempt is selected by sequence, not by
its result. If a cell needs more than two replacement
attempts, that cell is INCONCLUSIVE.

Preflight runs one production cell-seed for every
arm and driver. It uses seed 1999, outside the
campaign, and requires zero exits plus populated
timing and memory fields.

## Binding rule

The original three clauses and their definitions
remain unchanged.

1. B has no disallowed hit loss under the paired
   McNemar and A/A2 hit-allowance rule.
2. The ranked median charged ratio B/A does not
   exceed the global A/A2 band, unless hits improve.
3. The median outside time per charged call has
   B/A at most 1.10.

The hit-accounting check now compares like counters.
For h2h_seed, the top-level seen calls are compared
with the row evals_to_hit, and top-level ledger spent
is compared with the row charged value. Panel rows
carry the same source counter into both named fields.
Only completed hits enter the mismatch rate. B's
rate must not exceed max(A rate, A2 rate) in that
cell.

PASS requires all three original clauses and the
like-counter rate check. Search identity and printed
out.best differences are reported only. Scoring uses
seen.lowest. out.best is recorded separately.

The B path counter counts target-cell B rows whose
seen.lowest reaches the m38 rho-14 reference window.
It is an observable path-outcome counter, not a
binding clause, and must be non-zero.

## Memory and logs

The process is launched through the affinity command,
which then execs the release binary. The harness
samples RssAnon from that exec pid. Peak RssAnon is
the only memory gate. ru_maxrss and VmHWM are
reported but never gate.

Logs are written with a .tmp suffix, synced, moved,
synced again, and read back for sha256. The archive
uses numeric owner zero. Build logs are scrubbed.
The pre-archive scan rejects long environment values,
environment dumps, private paths, and host patterns.

## Separate base issue

Both LJ75 drivers use the same seeded start and
recommended configuration, but they stop and score
at different boundaries. h2h_seed observes every
potential invocation and raises its shared stop when
an intermediate evaluation reaches the target. It
scores seen.lowest. h2h_gmin_proof scores only a
quenched improvement recorded in Outcome.

Round five reports the resulting hit and call-count
difference. It is not a gate clause.

## Verdict

Filled only after the fresh session is complete.
