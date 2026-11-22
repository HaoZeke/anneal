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

PASS. The fresh session contains 864 initial attempts
and no replacements. Every attempt exited zero, was
conclusive, and ran on core 0. All nine cells pass the
three original clauses and the corrected like-counter
rule.

The statistics output uses W = W+ for Wilcoxon. For
each cell, the code draws six paired bootstrap ratios:
charged B/A, charged A2/A, calls B/A, wall B/A,
outside B/A, and outside A2/A.

| cell / driver | hits A/A2/B | charged B/A (band) | outside B/A | verdict |
| --- | ---: | ---: | ---: | --- |
| lj75 / gmin proof | 7/7/7 | 1.000000 (1.000000) | 1.014735 | PASS |
| lj98 / gmin proof | 16/16/16 | 1.000000 (1.000000) | 0.998008 | PASS |
| m55r14 / morse | 18/18/18 | 1.000000 (1.000000) | 1.042822 | PASS |
| m80r14 / morse | 15/15/15 | 1.000000 (1.000000) | 1.030170 | PASS |
| m30r14 / morse | 17/17/17 | 1.000000 (1.000000) | 1.035224 | PASS |
| m38r6 / morse | 24/24/24 | 1.000000 (1.000000) | 1.006777 | PASS |
| m38r14 / morse | 11/11/24 | 0.106436 (1.000000) | 0.950888 | PASS |
| m38r14 / seed | 24/24/24 | 0.849222 (1.000000) | 0.996084 | PASS |
| lj75 / seed | 24/24/24 | 1.000000 (1.000000) | 1.028189 | PASS |

Every like-counter mismatch rate is zero, so every
cell passes the zero A/A2 rate band. The target-cell B
path counter is 24 and is non-zero.

The report-only search mismatch counts are two for
m55r14 morse, 19 for m38r14 morse, nine for m38r14
seed, and three for lj75 seed. All other cells have
zero. Printed-best scoring differs from seen.lowest
on five m38r14 seed rows and all 24 lj75 seed rows.

For the separate base issue, seen.lowest gives the
LJ75 seed driver 24/24 hits in 2,345 through 143,083
calls. Printed out.best gives 0/24. The proof driver
gets 7/24 with its four-million-call cap.

For m80r14 seed 2006, all four arms hit with calls and
charged both equal to 3,802,858. This confirms the
reported claim.

Peak RssAnon is 286,019,584 bytes and is the gated
memory value. Peak ru_maxrss and VmHWM are both
291,057,664 bytes and are report-only.
