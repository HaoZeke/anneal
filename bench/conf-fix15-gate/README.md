Copyright (c) anneal developers

# conf-fix15 same-session A/B gate

Overall verdict: **FAIL**.

A and B are release, locked builds from the recorded commits and trees.
A-prime is a second A build. Five byte-identical A artifact placements
measure the cluster wall-time placement band. All runs are serial on core
zero with numerical runtime thread counts set to one, seeds 2000--2011,
no warm-up, and repository paper budgets. Pair order is balanced ABBA;
`schedule-counts.csv` verifies every `(slot, arm)` count.

The family-wise threshold is 0.05/13 = 0.00384615385. A cell fails when B
has significantly fewer hits, significantly larger calls, or significantly
larger wall time beyond its measured A/A or placement band. A cluster cell
also fails on any unexplained bit-identity break in best energy, calls, or
final-coordinate bits.

The external CREST columns reproduce the available repository harness rows.
Those rows use that campaign's six-seed 0.1 kcal/mol energy-hit summary;
a dash means the repository has no completed CREST row for that molecule.

## Molecular conformer panel

| System | A hits | B hits | CREST hits | A calls (IQR) | B calls (IQR) | CREST calls | A outside/call (s) | B outside/call (s) | CREST outside/call (s) | Fisher p | MW p calls | MW p wall | Wall band | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| rq3 | 12/12 | 12/12 | - | 2273.500 (5630.750) | 1304.000 (1673.500) | - | 0.000038295889 | 0.000588621130 | - | 1 | 0.214493808 | 3.65845535e-05 | 1.013129 | FAIL |
| 915 | 7/12 | 10/12 | - | 10165.500 (7999.000) | 5376.000 (4058.250) | - | 0.000033802681 | 0.000406486775 | - | 0.370709382 | 0.0957426593 | 3.65845535e-05 | 1.006551 | FAIL |
| xld | 10/12 | 7/12 | 3/6 | 9706.500 (10497.000) | 14997.000 (15194.000) | 628752.000000 | 0.000047009978 | 0.001239753808 | 0.000603680 | 0.370709382 | 0.465000684 | 3.65845535e-05 | 1.002733 | FAIL |
| fluvastatin | 5/12 | 9/12 | 3/6 | 21629.000 (11694.000) | 6539.000 (7541.000) | 1051333.500000 | 0.000045453747 | 0.000819009053 | 0.000833867 | 0.213756899 | 0.168122015 | 3.65845535e-05 | 1.000145 | FAIL |
| z34 | 1/12 | 4/12 | 2/6 | 23013.000 (0.000) | 23013.000 (11390.750) | 753368.000000 | 0.000052540983 | 0.001163347297 | 0.000698811 | 0.316770186 | 0.112900305 | 3.65845535e-05 | 1.002472 | FAIL |
| amprenavir | 1/12 | 4/12 | 0/6 | 19577.000 (0.000) | 19577.000 (4553.500) | 2635655.000000 | 0.000067637765 | 0.001394651458 | 0.001208708 | 0.316770186 | 0.179790679 | 3.65845535e-05 | 1.008574 | FAIL |
| lopinavir | 0/12 | 2/12 | 2/6 | 22168.000 (0.000) | 22168.000 (0.000) | 6728488.000000 | 0.000110862793 | 0.002753688633 | 0.002265667 | 0.47826087 | 0.166224591 | 3.65845535e-05 | 1.006928 | FAIL |
| lorlatinib | 4/12 | 2/12 | - | 21838.000 (6583.750) | 21838.000 (0.000) | - | 0.000041732611 | 0.000585709058 | - | 0.640405361 | 0.569210963 | 0.0463865933 | 1.021151 | PASS |

## Cluster identity panel

| System | Identity A/B | A hits | B hits | A calls (IQR) | B calls (IQR) | A outside/call (s) | B outside/call (s) | Fisher p | MW p calls | MW p wall | Placement band | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| lj38 | 12/12 | 12/12 | 12/12 | 6357.000 (23774.250) | 6357.000 (23774.250) | 0.000004414265 | 0.000004227437 | 1 | 1 | 0.795012172 | 1.028037 | PASS |
| lj75 | 12/12 | 12/12 | 12/12 | 44176.000 (26570.250) | 44176.000 (26570.250) | 0.000011022131 | 0.000011053109 | 1 | 1 | 1 | 1.018411 | PASS |
| m80r6 | 12/12 | 12/12 | 12/12 | 238328.000 (213084.500) | 238328.000 (213084.500) | 0.000011639703 | 0.000011535217 | 1 | 1 | 0.976970255 | 1.045837 | PASS |
| au38 | 12/12 | 12/12 | 12/12 | 18365.500 (37982.250) | 18365.500 (37982.250) | 0.000007010968 | 0.000006951372 | 1 | 1 | 0.930987447 | 1.006425 | PASS |
| tip4p20 | 12/12 | 6/12 | 6/12 | 1686249.500 (1376627.500) | 1686249.500 (1376627.500) | 0.000002723896 | 0.000002695691 | 1 | 1 | 0.976970255 | 1.013461 | PASS |

## Provenance

- Input kit SHA-256: `e8dc8bcd08bc2b325df211c995b1109ee35690b5321b165ff77b1acc0c9c625d`
- B source tree: `fe5d02843a76c6e04d934844d6c7600ee098839d`
- Session: `conf-fix15-gate-41d9`
- Hashed machine id: `9eb296774121518feb1b187b0f6a109a42de7a5dd39e62ccd37d6e7d329604b8`
- Per-run source commits, source trees, artifact hashes, and harness
  blob ids are in `rows.csv`; build/session records are in `sessions.csv`.
