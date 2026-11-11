# Lattice v3.2

The development comparison is a tie with the reference.
Seeds 3000-3029 are the development set.
Seeds 5000-5029 stay the registered test, outside this seal.

The charge and its fail-before test are one commit, eaf8665c.
That commit is stamped 2026-11-10T16:46:36-0600.
logs/fail-before-charge.txt names base 0a79aadb.
The panic line reads spent 653.
receipts/tests.txt records the pass at the tip.

## Disposition

Seal.
REPORT, SHA256SUMS, and OUTER fall in later seconds.
The bundle touch and STORE_READY follow them.
STORE_READY lists OUTER, the tip, and the tree.
It also lists BUNDLE_SHA256 and the stat times.
It also lists the full refs/heads name.

History.
The replay script is replay.py and its log is replay.log.
Shared seconds are real.
Archive refs are never pushed.
The bundle carries refs/archive for 0a79aadb and e5037f12.
It also carries the v3.2-draft e947d022.

Paths.
The fail-before log names the base SHA and the panic line.
The compile lines use a neutral mount.

Serial runner.
run_development.sh runs seeds 3000-3029 one cell at a time on cpu 0.
Time T is the first start, on or after the charge commit.

Identity.
receipts/identity.txt names the engine commit.
It also names the tree that produced the binary.

Reference logs.
gmin/seed holds row.txt, stdout.txt, stderr.txt, and lowest.
The rows match the sealed reference table.
Hash 0e343f9f37c25d4567a586817710aa64baa2b308b7ced17a783949d13d06d2cc.

One threshold.
Both sides use a re-quenched energy at or below -582.086641.
The reference re-quench stays 19 of 30.

Uniform cut.
The tie pool keeps exact float equality.
These energies leave a one-member pool.

Charge.
The development cuts below come from the serial run.
That tip charges the atom pair sum.

Propose arm.
The exchange propose match stays closed.
energy_lattice_propose_is_unreachable passes.

Fail before.
the_atom_pair_sum_is_charged fails on 0a79aadb.
It passes on the charge commit.

Message.
The lattice-search commit names every file it adds.
It also names the five files it changes.
The hit flag is the energy test.

Copyright.
The developer line is on dls_published.rs and lj_dls.rs.
It is also on gmin_run.sh, gmin_sha.sh, and h2h_cap.f90.

Prose.
proseguard: NOT_RUN (binary absent).
receipts/lint.txt holds the line-width check.
It also holds the banned-token check.

## Development recount

T is 2026-11-10T16:47:33-0600.
The charge commit eaf8665c is 57 seconds earlier.
Lowest-index hits are 20 of 30.
The median charged cost of a hit is 941310.
The wall sum is 2263.458 seconds.
The lowest-index window opens at 2026-11-10T16:47:33-0600.
It closes at 2026-11-10T17:38:59-0600.
Uniform hits are 20 of 30.
The median charged cost of a hit is 532260.5.
The wall sum is 1956.674 seconds.
The uniform window opens at 2026-11-10T17:38:59-0600.
It closes at 2026-11-10T18:26:25-0600.
recount.py rebuilds both cut tables from these raw logs.
It also rebuilds charged-diff.tsv.
The one-minute load at the first start is 1.89.
The one-minute load at the last uniform seed is 1.00.
Fisher exact on the 20 and 19 hit table is 1.
The comparison is a tie.

## Clock

Receipt times are -0600.
A paired reading was 20:01:32 +0000 and 14:01:32 -0600.
The two readings are six hours apart in the same second.
The runner TZ is America/Chicago.

## Fork

This clone holds 87a1c409 in place of ed5849f6.
Both name tree b8c8d0bc0b0684229c1c7b5862b15e983ffc16c9.
The lattice-search commit follows the replayed v1 tip.

## Rebase

The 30-seed panel was measured at 28ece242.
Its tree is 011a2b3e13379b9b6344b51f8b5ae40ca607a8b9.
Both cuts are 20 of 30, as the v3.2 kit reports.
The rebase onto 6c048e30 changes the base only.
That develop commit is an ancestor of this tip.
No rebase conflict touched src/ lattice code.
range-diff marks each of the 34 patches equal.
