Copyright (c) anneal developers

# G3: where searches spend their time on double-funnel landscapes

LJ38, LJ75, LJ98, LJ150 and Morse M80 at rho 6, seeds 2000-2011, the paper
budgets of `scripts/gmin_panel/run_panel.py` (400,000 calls for LJ38,
4,000,000 for the rest) and its references. A hit is a finite best at most
0.001 above the reference; error rows count as misses.

One session, `scripts/g3_funnel/run_session.py`: serial, `taskset -c 0`,
every thread variable at 1, no `H2H_` or `ANNEAL_` variable, release builds,
no warm-up run. Per (system, seed) the method order alternates (ABBA).

## Methods

- **anneal**: `h2h_gmin_proof` (LJ) and `h2h_morse` (M80), the head-to-head
  drivers, with `--trace`. The trace is the opt-in accepted-minimum sink of
  `cluster_hopping::Ledger::trace_accepted`, off by default.
- **ASE minima hopping**: the committed ASE harness unchanged (start from its
  `place`, its metered calculators, `MinimaHopping` with the ASE 3.29
  defaults, the wall cap stored on the committed ASE rows: three times the
  slowest anneal wall per system). Accepted minima are the ones minima
  hopping writes to its own `minima.traj`; each is matched bit for bit to
  the force call that returned its energy. Every minima hopping move is an
  MD escape (`md`).
- **GMIN-1 / GMIN-CSM: not run.** See "Not available".

## Session

120 rows (5 systems x 12 seeds x 2 methods), 0 error rows, one session,
code revision 0cc3814c053c2f5a709f0f375ebae07e5373c171, machine
(sha256 of the machine id) 3b11c160088e7af5c4f7ebd9da95e0bfe019a34698416701bb057c15fea10873.
Release builds of the examples (Rust 1.88.0):
`h2h_gmin_proof` 0b168ed5581ae0b41e5dda5f41c88ad50781c851f8e9f242c4df977b5a158876,
`h2h_morse` 95e1445358876494aa3cc0b5232bf04432dcb508b9f4b17e9e3b505e4c7bcc31.
Python 3.12.3, ASE 3.29.0, numba 0.68.1 (its cache built before the
session), numpy 2.5.5, OVITO 3.17.2.

Minima hopping wall caps (s): lj38 4.653972, lj75 167.806728, lj98 246.546693,
lj150 544.418154 (the committed ASE rows), m80r6 534.808788 (three times
178.269596, the slowest anneal wall in `bench/gmin-panel/rows.csv`). Every
minima hopping row stopped at its cap, so its calls are right-censored at
`charged`, far below the budget.

## Trace identity

`tests/accept_trace_identity.rs` runs each system on seeds 2000-2002 with the
sink off and on and asserts the best energy bits, the charged calls, the hops
and the best state bits are identical. The short tests (40,000 calls) run by
default; the ignored one repeats it at the paper budgets with the
`reference + 0.001` stop of the harness. Both passed (15/15 paper-budget
pairs identical; 0 unattributed minima).

## Files

- `rows.csv`: one row per run.
- `minima.csv.zst` (sha256 43dc009c61fbe22bdc9482786c8ed981148c09c3b14127d76aaffe3f600fd35d):
  every accepted minimum with its run, force-call index, move, energy, CNA
  shares and funnel.
- `occupancy.csv`, `transitions.csv`, `misses.csv`: written by
  `scripts/g3_funnel/derive.py` from the rows and the classified traces.
- `classify-rules.md`: cutoffs and thresholds, committed before any trace.
- The raw traces (extended XYZ, one frame per accepted minimum), the per-run
  logs, minima hopping's own `hop.log` and `minima.traj`, the per-run
  classified CSVs and the identity-test log are in the logs archive, sha256
  d987a03518eb4cc53dafca8fcc2fd2b940cc35a29841fd6ab189c694618c2d41
  (635,852,172 bytes). Each trace's sha256 is in `rows.csv`.

## Reading the traces

The trace records accepted minima. When a quench reaches the reference the
run stops at once, so the hitting minimum appears in the trace only if it
was accepted before the stop: 15 of the 44 anneal hits (column "trace has
the hit minimum"). Occupancy and transitions are therefore those of the
search before the stop. For misses the terminal minimum is the lowest
accepted minimum of the trace.

The polyicosahedral, disordered minima early in a run have few 421 and 422
bonds and fall under `other` by rule 1; the LJ98 Leary tetrahedron is
labelled `icosahedral` (see `classify-rules.md`).

## Hits

| method | system | hits | median charged calls | trace has the hit minimum |
|---|---|---|---|---|
| anneal | lj38 | 12/12 | 127839.5 | 3/12 |
| anneal | lj75 | 5/12 | 4000000 | 4/5 |
| anneal | lj98 | 7/12 | 3166811.5 | 7/7 |
| anneal | lj150 | 8/12 | 1129381.5 | 1/8 |
| anneal | m80r6 | 12/12 | 62677 | 0/12 |
| ase_mh | lj38 | 0/12 | 23986.5 | 0/0 |
| ase_mh | lj75 | 0/12 | 67568.5 | 0/0 |
| ase_mh | lj98 | 0/12 | 60025 | 0/0 |
| ase_mh | lj150 | 0/12 | 52695.5 | 0/0 |
| ase_mh | m80r6 | 0/12 | 23662 | 0/0 |

## Occupancy

Shares of accepted minima / of charged force calls per funnel, all 12 seeds pooled.

| method | system | icosahedral | decahedral | fcc | other | before first | inter-funnel transitions |
|---|---|---|---|---|---|---|---|
| anneal | lj38 | 0.826 / 0.833 | 0.044 / 0.040 | 0.028 / 0.025 | 0.102 / 0.100 | 0.002 | 1084 |
| anneal | lj75 | 0.943 / 0.956 | 0.036 / 0.028 | 0.021 / 0.016 | 0.000 / 0.000 | 0.000 | 399 |
| anneal | lj98 | 0.730 / 0.768 | 0.248 / 0.211 | 0.022 / 0.021 | 0.000 / 0.000 | 0.000 | 1942 |
| anneal | lj150 | 0.297 / 0.292 | 0.340 / 0.427 | 0.363 / 0.280 | 0.000 / 0.000 | 0.000 | 1601 |
| anneal | m80r6 | 0.239 / 0.300 | 0.394 / 0.344 | 0.367 / 0.354 | 0.000 / 0.000 | 0.002 | 627 |
| ase_mh | lj38 | 0.690 / 0.627 | 0.103 / 0.063 | 0.000 / 0.000 | 0.207 / 0.254 | 0.057 | 11 |
| ase_mh | lj75 | 0.781 / 0.798 | 0.205 / 0.163 | 0.000 / 0.000 | 0.013 / 0.008 | 0.030 | 7 |
| ase_mh | lj98 | 1.000 / 0.959 | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 | 0.041 | 0 |
| ase_mh | lj150 | 0.992 / 0.943 | 0.008 / 0.003 | 0.000 / 0.000 | 0.000 / 0.000 | 0.054 | 2 |
| ase_mh | m80r6 | 0.904 / 0.921 | 0.038 / 0.014 | 0.057 / 0.042 | 0.000 / 0.000 | 0.022 | 4 |

## Moves that cross funnels

The five moves with the most transitions per method and system (all funnel pairs).

| method | system | transitions | top moves (count) |
|---|---|---|---|
| anneal | lj38 | 1084 | dls-kernel (274), orbit-completion (255), sym (167), single (121), soap (87) |
| anneal | lj75 | 399 | dls-kernel (122), burst (108), orbit-completion (88), stall-climb (31), surface (25) |
| anneal | lj98 | 1942 | dls-kernel (958), burst (546), stall-climb (166), surface (90), orbit-completion (57) |
| anneal | lj150 | 1601 | burst (564), dls-kernel (452), orbit-completion (350), surface (90), sym (66) |
| anneal | m80r6 | 627 | dls-kernel (331), surface (84), sym (67), burst (49), stall-climb (46) |
| ase_mh | lj38 | 11 | md (11) |
| ase_mh | lj75 | 7 | md (7) |
| ase_mh | lj98 | 0 |  |
| ase_mh | lj150 | 2 | md (2) |
| ase_mh | m80r6 | 4 | md (4) |

## Misses: funnel of the terminal lowest accepted minimum

| method | system | misses | icosahedral | decahedral | fcc | other | energies of the lowest accepted minima |
|---|---|---|---|---|---|---|---|
| anneal | lj38 | 0 | | | | | |
| anneal | lj75 | 7 | 7 | 0 | 0 | 0 | -396.2822 to -396.2822 |
| anneal | lj98 | 5 | 5 | 0 | 0 | 0 | -543.6430 to -543.6430 |
| anneal | lj150 | 4 | 0 | 3 | 1 | 0 | -884.9486 to -881.4081 |
| anneal | m80r6 | 0 | | | | | |
| ase_mh | lj38 | 12 | 9 | 1 | 0 | 2 | -172.2307 to -168.3453 |
| ase_mh | lj75 | 12 | 9 | 3 | 0 | 0 | -396.2385 to -385.6678 |
| ase_mh | lj98 | 12 | 12 | 0 | 0 | 0 | -540.2478 to -524.2014 |
| ase_mh | lj150 | 12 | 12 | 0 | 0 | 0 | -877.6731 to -847.5678 |
| ase_mh | m80r6 | 12 | 10 | 1 | 1 | 0 | -375.5013 to -366.8128 |

## Not available

- **GMIN-1**: not run. The pinned GMIN snapshot (SVN 9cbec1a4, tarball
  sha256 ec5730d5e27fd5c1ac142360278abc62bd3c7dbea8686c9a2dc9de817b58b776)
  could not be obtained: the public tarball now has sha256
  b3995a554ca5c423329ceaa66675531b5c31956d94875c538951e3aabf90c3f5 (a newer
  revision), no archived copy matches, and `build_gmin.sh` refuses any
  other tarball. No other revision was substituted.
- **GMIN-CSM**: not run, for the same reason. At 55f9bfcf132b703a00812d7ffdb1f8905b753013
  of the private repro branch the only published per-size input is LJ75
  (`workflow/lj_csm/gmin/lj75/data`, sha256
  0be0216d19761517af47b0b1227353804cf54182596479248a063c06a7d69512); LJ38,
  LJ98, LJ150 and M80 rho 6: no published input.
