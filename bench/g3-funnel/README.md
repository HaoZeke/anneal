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
pairs identical; 0 unattributed minima). The test now runs the traced arm
with the incumbent sink (`Ledger::trace_incumbent`) installed as well and
checks that its last frame carries the best's energy and state bits; it
passed again, short and paper-budget (15/15).

## Replay

`scripts/g3_funnel/replay.py` reran all 120 rows after the session:
anneal with the same seeds, minima hopping with `run_mh.py --calls
<charged>`. Replays are deterministic and untimed, so they ran in parallel.
Every replay reproduced its row: the accepted-minimum trace has the sha256
stored on the row and the printed best is the same text (120/120;
`replay_trace_match`, `replay_best_match`). The rows did not store energy
bits, so the check is that bit-identical trace plus the printed best;
`best_bits` is the replay's best, the last incumbent frame. The examples
were rebuilt for the replay (`h2h_gmin_proof`
8414a65e2a0fefed5272626fd92ca77bea745def6b383aa8e414f15584bbbc73,
`h2h_morse` 1d604b7bd694df4e02f27c81397ad9155ff67ad4d07dd8e4a63510aa9ad465c2,
`xyz_to_con` 87f887c1188f32b62ac009ecaa890ddbae8ad80c56ec976168aff0ef84d42a4c).

## Files

- `rows.csv`: one row per run, with `best_bits`, `hop_first_best` and the
  replay checks appended.
- `minima.csv.zst` (sha256 43dc009c61fbe22bdc9482786c8ed981148c09c3b14127d76aaffe3f600fd35d):
  every accepted minimum with its run, force-call index, move, energy, CNA
  shares and funnel.
- `occupancy.csv`, `transitions.csv`: written by
  `scripts/g3_funnel/derive.py` from the rows and the classified traces.
- `misses/`: the final structure of every miss as `.con` (readcon-core).
- `misses.csv`, `traps.csv`: written by `scripts/g3_funnel/misses.py` from
  those files: point group, CNA class, miss class and structure per miss,
  and per system and miss class the counts per method.
- `classify-rules.md`: cutoffs and thresholds, committed before any trace.
- `misses-rules.md`: point group tolerances, structure identity, the
  dominant-trap and `hop_first_best` rules, committed before any miss
  structure was written.
- The raw traces (extended XYZ, one frame per accepted minimum), the per-run
  logs, minima hopping's own `hop.log` and `minima.traj`, the per-run
  classified CSVs, the replay's incumbent traces and logs and both
  identity-test logs are in the logs archive, sha256
  2b0ef3b9b0088f12c52617163450c0635c3b00d7a5f0060cb7637eef974d33a5
  (645,769,319 bytes). Each trace's sha256 is in `rows.csv`.

## Plain funnel tables

Regenerate the uncompressed tables from the repository root with Python 3
and the `zstandard` package:

```console
python3 bench/g3-funnel/derive_plain_csv.py
```

The command reads `rows.csv`, `minima.csv.zst`, and the existing classified
`misses.csv`. It writes these deterministic, LF-terminated files:

- `funnel_occupancy_runs.csv`: one row per method, system, and seed.
- `funnel_occupancy_summary.csv`: the run table summed per method and system.
- `funnel_transitions_runs.csv`: directional crossing counts per method,
  system, seed, producing move, source funnel, and destination funnel.
- `funnel_transitions_summary.csv`: the crossing table summed per method and
  system.
- `funnel_misses.csv`: one row per miss with its final energy, point group,
  CNA class, combined miss class, structure identifier, and committed
  `.con` path.

In the occupancy tables, `accepted_minima_*` counts trace frames in each
funnel. `force_calls_*` assigns the interval from an accepted minimum's
force-call index to the next accepted minimum, or to the run's `charged`
total for the last minimum. `force_calls_before_first` is the unlabelled
opening interval. `observed_hop_steps_*` assigns differences between
successive accepted-minimum hop indices to the earlier minimum's funnel;
`observed_hop_steps_before_first` is the first accepted minimum's hop index.
The corresponding `*_total` columns include all reported funnels and the
opening interval.

Transition tables contain one zero-count row with blank crossing fields when
a run or method-system panel has no inter-funnel crossing. This keeps every
run and every summary panel explicit without inventing a funnel or move.

The trace has no elapsed-time field and `rows.csv` has no final hop count.
Elapsed-time occupancy is therefore omitted, and observed hop-step occupancy
ends at the last accepted minimum rather than filling an unknown terminal
interval. No value is inferred for either missing quantity.

## Reading the traces

The trace records accepted minima. When a quench reaches the reference the
run stops at once, so the hitting minimum appears in the trace only if it
was accepted before the stop: 15 of the 44 anneal hits (column "trace has
the hit minimum"). Occupancy and transitions are therefore those of the
search before the stop.

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

## hop_first_best

The hop at which each run first reached the basin of its final best
(`misses-rules.md`): completed hops for anneal, minima hopping steps for
minima hopping. Medians and ranges over 12 seeds; hits and misses apart.

| method | system | median (all) | hits | misses |
|---|---|---|---|---|
| anneal | lj38 | 1118 | 1118 (351-4334) |  |
| anneal | lj75 | 1871 | 26204 (2113-33908) | 927 (318-2051) |
| anneal | lj98 | 13189.5 | 13945 (666-38435) | 12434 (517-18275) |
| anneal | lj150 | 1652 | 884.5 (37-31500) | 10245.5 (312-36658) |
| anneal | m80r6 | 450 | 450 (39-30289) |  |
| ase_mh | lj38 | 31 |  | 31 (18-35) |
| ase_mh | lj75 | 60 |  | 60 (47-99) |
| ase_mh | lj98 | 57 |  | 57 (51-66) |
| ase_mh | lj150 | 48.5 |  | 48.5 (31-56) |
| ase_mh | m80r6 | 94.5 |  | 94.5 (48-118) |

anneal's LJ75 misses reached their final basin by hop 2,051 and stayed
there; seeds 2000 and 2002, both misses, ran 39,993 and 38,760 hops in all
(identity-test log).

## Traps: miss classes

Every miss's final structure, classified by point group and CNA class
(`misses-rules.md`). `structures` counts distinct structures by the
distance-list test; energies are the rows' best, shown for context only.
The dominant trap is the most frequent class pooled over methods.

| system | miss class | misses | anneal | minima hopping | structures | best range | dominant |
|---|---|---|---|---|---|---|---|
| lj150 | icosahedral C1 | 12 | 0 | 12 | 12 | -877.6731 to -849.0784 | yes |
| lj150 | decahedral Cs | 3 | 3 | 0 | 2 | -884.9486 to -884.6407 |  |
| lj150 | fcc C3h | 1 | 1 | 0 | 1 | -881.4081 to -881.4081 |  |
| lj38 | icosahedral C1 | 6 | 0 | 6 | 6 | -171.1804 to -168.8617 | yes |
| lj38 | other C1 | 3 | 0 | 3 | 3 | -170.7131 to -168.3455 |  |
| lj38 | icosahedral Cs | 2 | 0 | 2 | 2 | -173.1343 to -169.9556 |  |
| lj38 | decahedral C1 | 1 | 0 | 1 | 1 | -169.7357 to -169.7357 |  |
| lj75 | icosahedral C1 | 15 | 7 | 8 | 9 | -396.2822 to -385.6678 | yes |
| lj75 | decahedral C1 | 3 | 0 | 3 | 3 | -391.7770 to -388.9632 |  |
| lj75 | icosahedral Cs | 1 | 0 | 1 | 1 | -396.2385 to -396.2385 |  |
| lj98 | icosahedral C1 | 12 | 0 | 12 | 12 | -540.2478 to -524.7604 | yes |
| lj98 | icosahedral Cs | 5 | 5 | 0 | 1 | -543.6430 to -543.6430 |  |
| m80r6 | icosahedral C1 | 10 | 0 | 10 | 10 | -375.5013 to -366.8880 | yes |
| m80r6 | fcc C1 | 1 | 0 | 1 | 1 | -371.0573 to -371.0573 |  |
| m80r6 | decahedral C1 | 1 | 0 | 1 | 1 | -368.4206 to -368.4206 |  |

Structures shared by more than one miss:

| system | structure | misses | methods | miss class | best |
|---|---|---|---|---|---|
| lj75 | S1 | 7 | anneal | icosahedral C1 | -396.28224883 |
| lj98 | S1 | 5 | anneal | icosahedral Cs | -543.64295724 |
| lj150 | S2 | 2 | anneal | decahedral Cs | -884.64066360 |

Every other miss is a structure of its own. All 7 anneal misses at LJ75 end
in one structure (S1, icosahedral C1) and all 5 at LJ98 in one structure (S1,
icosahedral Cs). Minima hopping's 12 misses per system are 12 distinct
structures. At LJ98 the pooled rule therefore makes minima hopping's 12
distinct C1 endpoints the dominant class, while the most frequent single
structure is anneal's S1.

## Saddle path from the dominant trap to the global minimum

Not computed. The lowest saddle path from an LJ75 icosahedral trap to the
Marks decahedron, or from the LJ98 icosahedral trap to the Leary
tetrahedron, runs through many intermediate minima; finding the lowest one
takes a search over a database of minima and transition states (discrete
path sampling), which nothing in this panel's stack does in-run. One
nudged elastic band or dimer chain between the two endpoints, the tool that
is available (rgsaddle 0.3.0), returns one connected path whose highest
saddle bounds the barrier from above; it is not the lowest path and its
length depends on the interpolation, so it would be an estimate. The
permutational alignment that chain needs is IRA (anneal's `ira` feature),
whose library is not installed here. No barrier or path length is reported.

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
- Miss structures, traps and `hop_first_best` exist for anneal and minima
  hopping only; the two GMIN arms have no rows.
