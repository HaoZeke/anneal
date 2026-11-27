# Attribution of the template-removal commit

Parent `628773d6837ea2d959a27256ff4163b21162fd4d` (`ci: gate macOS Python matrix by relevant paths`) is the first parent of `1c1b12384683485cf820d3a244151b48973dc0f7` (`fix: remove structure templates from cluster search`). That parent is an ancestor of the removal commit, and the removal commit is an ancestor of develop `61fa67cf380e46876b569da45e7cc091647a775b`.

The removal commit is where these four cells take on the develop rows. The child release binary matches the kept develop rows on hit, best, and calls for all 48 seeds. The parent still hits every seed.

## What the commit deletes

`git show --stat 1c1b1238` is 34 files, 938 insertions, 3164 deletions. The search sources are `examples/h2h_morse.rs` (11 lines), `src/methods/cluster_hopping.rs` (1327), `src/methods/cluster_hopping/binary_geometry.rs` (825), `src/methods/cluster_hopping/mackay_cap_tests.rs` (deleted, 345), `src/methods/cluster_hopping/moves.rs` (60), `src/methods/cluster_hopping/packing_bank.rs` (143), and `src/soap.rs` (3). The rest of the stat is bench tables, paper text, and panel scripts.

The deleted code builds generic packings. It is parameterized by atom count and a nearest-neighbour length. It is not a stored coordinate table for one system. `src/structure.rs:545` still defines `Template` on both sides of the commit. The parent builders are:

- `src/methods/cluster_hopping/binary_geometry.rs:125` `fcc_cut`
- `src/methods/cluster_hopping/binary_geometry.rs:193` `polyicosahedral_cut`
- `src/methods/cluster_hopping/binary_geometry.rs:261` `mackay_cut`
- `src/methods/cluster_hopping/binary_geometry.rs:270` `mackay_cap`
- `src/methods/cluster_hopping/binary_geometry.rs:356` `axial_spine_cut`

Those functions, and the tests under them, occupy parent lines 108-745 and 1048-1173. The search quenches them from `src/methods/cluster_hopping.rs:5045-5070` (one Mackay cap when `long_ranged` is set), `5091-5565`, `6853-7121`, `9811-10066`, and `10189-10261`. `src/methods/cluster_hopping/mackay_cap_tests.rs` is removed as a whole. The builders have a zero-size guard and no branch on a particular cluster size.

`examples/h2h_morse.rs:149-153` on the parent sets `cfg.long_ranged` when the second-neighbour energy is below `-0.05`. That is rho 6. Rho 14 leaves it false. The child keeps the field (`src/methods/cluster_hopping/config.rs:1175`, written false at line 1957) and the search no longer reads it.

## How the rows were run

Release binaries of each revision. Seeds 2000-2011, budget 400000, one serial session, core 0, every thread count set to 1. Even offsets run the parent, then the child. Odd offsets run the child, then the parent. The command is `run <system> <seed>`. The budget is the only search setting passed in. There is no quick flag and no warm start. A hit is a finite best at or below the reference plus 0.001.

References: m38r6 `-157.477108`, m38r14 `-144.321054`, sc126-38 `-35419.9804`, sc108-38 `-10117.2454`.

The median is Hyndman-Fan type 7 of `evals_to_hit`, with a miss stored as the budget. Mann-Whitney ranks a miss worse than every hit, with the tie correction and the continuity correction. Fisher is the two-sided exact test on the hit counts. The machine id on every row is `3b11c160088e7af5c4f7ebd9da95e0bfe019a34698416701bb057c15fea10873`.

Driver blobs:

| role | blob | where it comes from |
| --- | --- | --- |
| parent Morse release | `dfd32871b747b48de31e97180c5863267d120e4f` | same blob as `0ef6ba4e:examples/h2h_morse.rs` |
| child Morse release | `97c5b883648052a43035dbb51abc99e23ceb3489` | the removal commit's example |
| both SC release binaries | `44aa5e7b1d09311e93147b256368be6a001652d8` | unchanged by the removal commit |
| old Morse, built on the child | `dfd32871b747b48de31e97180c5863267d120e4f` | `0ef6ba4e:examples/h2h_morse.rs`, unchanged |
| old SC, built on both | `033ff7f87497559f7e96c13e43f8413a0974c04e` | `e3ed7dad:examples/h2h_sc.rs`, unchanged |

The old Morse file compiled on the child. Its 24 rows match the child release rows on hit, best, charged, and calls. The old SC file compiled on both revisions. Its 48 rows match the release rows of the revision they were built on. The example edits do not move these cells. The library deletion does.

The develop examples do not compile against either revision, and they were not patched. `61fa67cf:examples/h2h_morse.rs` (`5d3532d21caa2b424a9200ee6ef7237193aeb108`) is missing `DEFAULT_HIT_TOL`, `energy_hits`, `Config::check_removed_env`, `Config::recommended_for_pair`, `Config::parse_prior_switches`, `branch_on_stall`, and `plateau_heat`. `61fa67cf:examples/h2h_sc.rs` (`d9e57aa36c8cdc38a7e7119bc2cf3cfed901aa31`) is missing `DEFAULT_HIT_TOL` and `energy_hits`. The same errors occur on the parent and on the child. The runner here is `run_cells.py`. It calls the same `run <system> <seed>` command.

## Release comparison

| cell | parent hits | parent median | child hits | child median | Fisher p | Mann-Whitney p |
| --- | --- | --- | --- | --- | --- | --- |
| m38r14 | 12/12 | 16733.5 | 4/12 | 400000 | 0.00134608 | 3.37232e-05 |
| m38r6 | 12/12 | 432.5 | 12/12 | 49438.5 | 1 | 3.65846e-05 |
| sc126-38 | 12/12 | 9861 | 12/12 | 113504.5 | 1 | 4.1301e-05 |
| sc108-38 | 12/12 | 9762.5 | 7/12 | 361031.5 | 0.0372671 | 3.38156e-05 |

Parent Morse uses blob `dfd32871b747b48de31e97180c5863267d120e4f`. Child Morse uses blob `97c5b883648052a43035dbb51abc99e23ceb3489`. Both SC binaries use blob `44aa5e7b1d09311e93147b256368be6a001652d8`.

The child medians are the kept develop medians: m38r6 49438.5, m38r14 400000, sc126-38 113504.5, sc108-38 361031.5. Parent m38r6 matches the kept `0ef6ba4e` rows on all 12 seeds, median 432.5. Parent m38r14 still hits 12/12 with the same bests as those older rows; its median is 16733.5 against the kept median 17762.5. Both parent SC cells still hit 12/12 with the same bests as the kept `e3ed7dad` rows; the medians are 9861 and 9762.5 against the kept medians 10344.5 and 11193.

## Per-seed release rows

Parent is `628773d6837ea2d959a27256ff4163b21162fd4d`. Child is `1c1b12384683485cf820d3a244151b48973dc0f7`.

### m38r14

| seed | parent hit | parent calls | parent best | child hit | child calls | child best |
| --- | --- | --- | --- | --- | --- | --- |
| 2000 | 1 | 20496 | -144.32105392 | 0 | 400000 | -142.71594037 |
| 2001 | 1 | 12583 | -144.32105392 | 1 | 219638 | -144.32105392 |
| 2002 | 1 | 18128 | -144.32105392 | 0 | 400000 | -142.68775213 |
| 2003 | 1 | 21042 | -144.32105392 | 0 | 400000 | -142.69789782 |
| 2004 | 1 | 14062 | -144.32105392 | 0 | 400000 | -142.69789782 |
| 2005 | 1 | 15339 | -144.32105392 | 0 | 400000 | -142.71594037 |
| 2006 | 1 | 25873 | -144.32105392 | 0 | 400000 | -142.71594037 |
| 2007 | 1 | 14738 | -144.32105392 | 0 | 400000 | -142.69779698 |
| 2008 | 1 | 19548 | -144.32105392 | 1 | 44357 | -144.32105392 |
| 2009 | 1 | 8840 | -144.32105392 | 1 | 115381 | -144.32105392 |
| 2010 | 1 | 12375 | -144.32105392 | 0 | 400000 | -142.71594037 |
| 2011 | 1 | 18572 | -144.32105392 | 1 | 25004 | -144.32105392 |

### m38r6

| seed | parent hit | parent calls | parent best | child hit | child calls | child best |
| --- | --- | --- | --- | --- | --- | --- |
| 2000 | 1 | 423 | -157.47710811 | 1 | 67722 | -157.47710811 |
| 2001 | 1 | 431 | -157.47710811 | 1 | 247613 | -157.47699843 |
| 2002 | 1 | 454 | -157.47710811 | 1 | 19477 | -157.47710811 |
| 2003 | 1 | 407 | -157.47710811 | 1 | 30269 | -157.47710811 |
| 2004 | 1 | 412 | -157.47710811 | 1 | 68845 | -157.47710811 |
| 2005 | 1 | 759 | -157.47710811 | 1 | 17111 | -157.47710811 |
| 2006 | 1 | 424 | -157.47710811 | 1 | 86018 | -157.47710811 |
| 2007 | 1 | 434 | -157.47710811 | 1 | 82982 | -157.47710811 |
| 2008 | 1 | 547 | -157.47710811 | 1 | 10585 | -157.47710811 |
| 2009 | 1 | 465 | -157.47710811 | 1 | 144964 | -157.47710811 |
| 2010 | 1 | 631 | -157.47710811 | 1 | 17637 | -157.47710811 |
| 2011 | 1 | 347 | -157.47710811 | 1 | 31155 | -157.47710811 |

### sc126-38

| seed | parent hit | parent calls | parent best | child hit | child calls | child best |
| --- | --- | --- | --- | --- | --- | --- |
| 2000 | 1 | 7488 | -35419.98044844 | 1 | 256599 | -35419.98044844 |
| 2001 | 1 | 17080 | -35419.98044844 | 1 | 89394 | -35419.98044844 |
| 2002 | 1 | 7243 | -35419.98044844 | 1 | 105311 | -35419.98044844 |
| 2003 | 1 | 9212 | -35419.98044844 | 1 | 107334 | -35419.98044844 |
| 2004 | 1 | 7708 | -35419.98044844 | 1 | 19820 | -35419.98044844 |
| 2005 | 1 | 6736 | -35419.98044844 | 1 | 143552 | -35419.98044844 |
| 2006 | 1 | 13942 | -35419.98044844 | 1 | 167761 | -35419.98044844 |
| 2007 | 1 | 7163 | -35419.98044844 | 1 | 119675 | -35419.98044844 |
| 2008 | 1 | 11463 | -35419.98044844 | 1 | 124239 | -35419.98044844 |
| 2009 | 1 | 18718 | -35419.98044844 | 1 | 18718 | -35419.98044844 |
| 2010 | 1 | 12758 | -35419.98044844 | 1 | 198816 | -35419.98044844 |
| 2011 | 1 | 10510 | -35419.98044844 | 1 | 57774 | -35419.98044844 |

### sc108-38

| seed | parent hit | parent calls | parent best | child hit | child calls | child best |
| --- | --- | --- | --- | --- | --- | --- |
| 2000 | 1 | 10445 | -10117.24540728 | 1 | 244800 | -10117.24540728 |
| 2001 | 1 | 15110 | -10117.24540728 | 0 | 400000 | -10108.55984979 |
| 2002 | 1 | 9080 | -10117.24540728 | 0 | 400000 | -10108.55984979 |
| 2003 | 1 | 12089 | -10117.24540728 | 1 | 113633 | -10117.24540728 |
| 2004 | 1 | 10953 | -10117.24540728 | 1 | 382791 | -10117.24540728 |
| 2005 | 1 | 8535 | -10117.24540728 | 0 | 400000 | -10108.55984979 |
| 2006 | 1 | 13462 | -10117.24540728 | 1 | 137127 | -10117.24540728 |
| 2007 | 1 | 7217 | -10117.24540728 | 1 | 339272 | -10117.24540728 |
| 2008 | 1 | 8796 | -10117.24540728 | 1 | 25206 | -10117.24540728 |
| 2009 | 1 | 7009 | -10117.24540728 | 1 | 245822 | -10117.24540728 |
| 2010 | 1 | 8122 | -10117.24540728 | 0 | 400000 | -10108.55984979 |
| 2011 | 1 | 14438 | -10117.24540728 | 0 | 400000 | -10108.55984979 |

