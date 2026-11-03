# Reconcile

The Wales-group basin-hopping program (GMIN) is the external code in the earlier tables.
The disclosure sentence is the generic American Chemical Society (ACS) wording.
The GMIN tables sit beside one measured head.
That head is commit `90c898dd2971c8753268b153d96b9b54429b081c` on `lib/recommended-no-size-switches`.
`Config::recommended` on that head has no atom-count branch.
The starts are the seeded draws, six seeds, ceilings 1.2e6, 1e5, 4e5, 1e6 and 1e6.
The 38-atom claims in the abstract, the results and the conclusions are those six runs.

## Cited commits

The article cites one commit, the library head above.
`git merge-base --is-ancestor` exits 0 when the first commit is an ancestor of the second, and that includes a commit checked against itself.

```
git merge-base --is-ancestor 90c898dd2971c8753268b153d96b9b54429b081c 46c8c867d23d6dba75a878a7614144687fb31542
echo lib_head_ancestor_of_main:$?
```

`lib_head_ancestor_of_main:1`

```
git merge-base --is-ancestor 90c898dd2971c8753268b153d96b9b54429b081c 90c898dd2971c8753268b153d96b9b54429b081c
echo lib_head_ancestor_of_lib_head:$?
```

`lib_head_ancestor_of_lib_head:0`

The cited commit is reachable from the library head.
It is not reachable from main `46c8c867d23d6dba75a878a7614144687fb31542`.
Main is not cited in the article.
The article names main only as the line that still has both size tests.

## Commits not cited

Three commits that an earlier paper text mapped onto are not ancestors of main and not ancestors of the library head.
The article does not cite them.
`645bc9386838e4951dd3f90bf63f010103ff8c5d` is only on `fix/paper-controls-warm-relax`.

```
git merge-base --is-ancestor b7f4b6e1acbc02cf7d11fc38cbc2cda91bb78f37 46c8c867d23d6dba75a878a7614144687fb31542
echo b7f4b6e1_ancestor_of_main:$?
git merge-base --is-ancestor b7f4b6e1acbc02cf7d11fc38cbc2cda91bb78f37 90c898dd2971c8753268b153d96b9b54429b081c
echo b7f4b6e1_ancestor_of_lib_head:$?
git merge-base --is-ancestor 19523548a2fb43c4c0ad7ee53db0a2fdf6fe193d 46c8c867d23d6dba75a878a7614144687fb31542
echo 19523548_ancestor_of_main:$?
git merge-base --is-ancestor 19523548a2fb43c4c0ad7ee53db0a2fdf6fe193d 90c898dd2971c8753268b153d96b9b54429b081c
echo 19523548_ancestor_of_lib_head:$?
git merge-base --is-ancestor 645bc9386838e4951dd3f90bf63f010103ff8c5d 46c8c867d23d6dba75a878a7614144687fb31542
echo 645bc938_ancestor_of_main:$?
git merge-base --is-ancestor 645bc9386838e4951dd3f90bf63f010103ff8c5d 90c898dd2971c8753268b153d96b9b54429b081c
echo 645bc938_ancestor_of_lib_head:$?
```

`b7f4b6e1_ancestor_of_main:1`
`b7f4b6e1_ancestor_of_lib_head:1`
`19523548_ancestor_of_main:1`
`19523548_ancestor_of_lib_head:1`
`645bc938_ancestor_of_main:1`
`645bc938_ancestor_of_lib_head:1`

`e7d59c15` is not an object in this clone.
The disclosure in the article is that ACS sentence.
That sentence names no model, vendor or company.

## LJ55 and LJ98 seed table

`data/lj55_lj98/seeds.csv` in the retired paper tree wrote `develop_sha` as `f29dedc0e379564ac31caf8eb2c6c9cb94f515eb` on all 512 rows.
The same value is on the GMIN rows, so it is not a per-run revision.
No reader in this archive uses the column.
The file carried here drops the column.
The row digest is `json_sha256` of the campaign record.

## LJ98 hits

The two deposited LJ98 tables are earlier measurements.
They are not the measured head.

The referee report and `logs/paired_bootstrap_lj98.txt` record a hop at 24 of 64 and GMIN at 62 of 64, with penalised-cost medians 2000000 and 158540.5.
The paired-bootstrap interval of the median ratio is 12.62 with endpoints 8.52 and 15.22, from 2000 resamples.
The report quotes Holm 3.1e-8 for that table.
This archive does not recompute that probability.

`data/campaign-scores.csv` records `anneal-rec` at 36 of 64, median 824807, and `gmin-bh` at 61 of 64, median 321201.5.
The recommended rows of that file were scored while the preset still branched on the atom count.

## TIP4P20 seeds

The campaign file scores TIP4P20 on 16 seeds: `anneal-rec` 0 of 16, `gmin-bh` 8 of 16, `gmin-tbp` 10 of 16.
The referee report states GMIN at 37 of 64 against a hop at 0 of 64.
The seed counts differ.
The article reports no TIP4P20 hit count.

## TIP4P10 medians

The referee report gives a GMIN median first hit of 9562 against a hop median of 19487.5.
The campaign file's penalised-cost medians are 10773 for `anneal-rec` and 11616 for `gmin-bh`.
The lower median is not the same pair of counters.
The article reports no TIP4P10 hit count.
`logs/tip4p_constants.txt` records the pair coefficients.
GMIN matches Jorgensen's A and C.
A rounded well depth and a rounded sigma build the anneal pair.
That anneal pair does not match those two coefficients.

## LJ38 call medians

The campaign file gives `anneal-rec` 722.5, `anneal-lean` 5332, `gmin-bh` 3638 and `gmin-sym` 21647, each on 64 seeds.
The previous rewrite recorded calls 51657, 29087, 26179, 677, 1322 and 1464.
Those calls are the lean library on frozen starts.
The compacted surface was still installed.
The six runs in the article are a seeded draw of the preset with both branches removed.
No probability is reported for the lean library against `gmin-sym`.
The campaign file stores medians and does not store the paired sample that a test would need.

## Claims removed

The previous manuscript stated a Holm-adjusted Wilcoxon probability of 0.0054 for paired calls at 75 atoms.
The campaign file stores medians.
It does not store that paired sample.
The article drops the probability.

The campaign file stores no core-second test.
The article removes those probabilities.

Holm-adjusted probabilities are not reported for the six runs.
The rerun plan names Holm for a family written down before a later campaign.

The campaign file does not name a GMIN revision.
The previous manuscript named GMIN revision 26635.
No log in this archive records that revision string, and the article removes the claim.

The previous manuscript called a GMIN method at 104 atoms a genetic algorithm.
`data/literature.tsv` places Oakley's genetic-algorithm means at 75 and 98 atoms.
The article removes the 104-atom wording.

The previous manuscript said that no single-chain code hits the 104-atom minimum.
The campaign file records OGOLEM at 6 of 64 on that size.
The article removes the sentence.
The citation for OGOLEM is Dieterich and Hartke, Mol. Phys. 108 (2010) 279.
The citation for ABCluster, which the same file also records, is Zhang and Dolg, Phys. Chem. Chem. Phys. 17 (2015) 24173.

The OGOLEM citation and the ABCluster citation sit in the campaign file.
We compare hit counts with Fisher's exact test.
The Mann-Whitney test is not the test for a hit count.

## Oakley means

`data/literature.tsv` quotes Oakley, Johnston and Wales, Phys. Chem. Chem. Phys. 15 (2013) 3965, Table 1.
The continuous-symmetry basin-hopping mean at 38 atoms is 4369 evaluations over 100 successful runs.
The core-orbit basin-hopping means are 50229 at 75 atoms and 109753 at 98 atoms.
Each of those two means also comes from 100 runs that all succeeded.
The article cites those means and does not divide them into the six-run call counts.
The counters and the starts differ.
