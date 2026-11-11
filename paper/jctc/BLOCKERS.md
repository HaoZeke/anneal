# BLOCKERS

No `XXX` token is in `manuscript.tex`, `si.tex`, or `cover-letter.tex`.
`si-six.tex` is removed. Its values 65670 and 30.042525 had no raw log.
The hexamer energy `-197.7750` kJ/mol is removed. The committed TIP4P-6 geometry with the anneal pair constants evaluates to -197.7751 kJ/mol, not -197.7750.

## A. Waiting on running science

The 192-seed water rates and the McNemar values are removed from the manuscript.
`develop` at `91b15a02` (`kit/water-crest-h2h-v2.6.3`) has no 192-seed stdout.
The water archive in this tree has seeds 0 through 5.
`origin` has no `water-v45b` ref.
The CSV provenance names `anneal_repro` bundle `v4.5/port-v4.5b.bundle`, and that bundle is not on this machine.
`HaoZeke/anneal_repro` returns 404 to this token.
`figs/water_crest.pdf` is the six-seed CREST counter chart from `logs/water/crest-hexamer-seedN.log`.
It does not draw the 192-seed rates, PAR-2, or McNemar values.
`water-v45b-perseed.csv` is not evidence for any number in the paper.

The LJ98 rates (24 of 64, the PAR-2 medians, the ratio, and the 2000 resamples) are removed.
`data/lj55_lj98/seeds.csv` is a per-seed table. The tree has no raw stdout for those 64 seeds.
A rerun of that campaign does not fit in 30 minutes.

These values were not put back. No source file, config line, or run log states them.

- LJ75 seeds 0 through 199. The sealed encounter curves are 200 runs, seeds 2000 through 2199, in `data/lj75_gmin/lj75-recommended-seeds-2000-2199.csv`.
- The 30-start count.
- The copper bound 1e-6 eV. The recorded gaps start at 8e-6 eV, and the script tolerance is 1e-3 eV.
- The archive marker #56. The archive contains `bias.rs`, `neighbors.rs`, and `potentials.rs`, and nothing counts 56 copies.

These campaigns are named and not sealed. No count is guessed.

- LJ75 seed sweep. The manuscript states that no hit count is in the sealed record. The seed bounds are not printed, because that run is unsealed.
- ASE/AGOX v5.
- armD-v1.4, the LJ104 arm D versus GMIN head-to-head.
- H2O16 v9, the water 16-mer against CREST.

`paper/jctc/manuscript.tex:350`

```
No hit count for those seeds is in the sealed record.
```

`paper/jctc/manuscript.tex:522`

```
The water 16-mer count is separate open work.
```

`paper/jctc/manuscript.tex:523`

```
Apart from the hexamer campaign.
```

`paper/jctc/manuscript.tex:524`

```
This comparison leaves out the Atomic Simulation Environment.
```

`paper/jctc/manuscript.tex:525`

```
The atomistic global optimization framework (AGOX).
```

`paper/jctc/manuscript.tex:533`

```
No hit count for the unfinished LJ\(_{75}\) seed sweep is in the sealed record.
```


## B. Rohit-only decisions

`SURF B.V., Amsterdam` stays. It is Rohit's ruling for the affiliation.
proseguard 0.1.0 flags `B.V.` under `rgoswami.AbbreviationStyle` when that style pack is loaded.
This tree has no style pack and no allowlist key for that one string, so the exception is recorded here.

The AI-drafting sentence is unchanged.

`paper/jctc/manuscript.tex:557`

```
Generative AI tools assisted with drafting and editing the text.
```

`paper/jctc/manuscript.tex:558`

```
The authors reviewed and edited all content.
```

`paper/jctc/manuscript.tex:559`

```
And take full responsibility for it.
```


The submission DOI is not assigned.

`paper/jctc/manuscript.tex:582`

```
The Zenodo digital object identifier (DOI) for this submission is not assigned.
```


Cover-letter telephone:

`paper/jctc/cover-letter.tex:13`

```
Telephone: \todo Rohit
```


Cover-letter suggested reviewers:

`paper/jctc/cover-letter.tex:43`

```
Suggested reviewers follow.
```

`paper/jctc/cover-letter.tex:44`

```
\todo Rohit \todo Rohit \todo Rohit
```


Version 0.12.0 has no release tag.

`paper/jctc/manuscript.tex:583`

```
The crate records version 0.12.0.
```

`paper/jctc/manuscript.tex:584`

```
The release tag for version 0.12.0 is not assigned.
```


USPEX and CALYPSO. The paper sources contain no sentence on either licence.

Co-author emails remain incomplete. The manuscript email is Debabrata Goswami. Rohit's email is on the cover letter.

`paper/jctc/manuscript.tex:36`

```
\email{dgoswami@iitk.ac.in}
```

`paper/jctc/cover-letter.tex:12`

```
Email: \url{rgoswami@ieee.org}\\
```


## Head-to-head tables

`tab:crest-counters` does not carry hardware, wall time, n, median/IQR, or the exact command in the table.
`tab:hexamer-gaps` does not carry hardware, wall time, n, median/IQR, or the exact command in the table.
The 192-seed rates are removed. The CSV remains in the tree and is not a raw stdout log.
`si.tex` has no table.

## Format not met

The abstract is four sentences. Supporting information is `si.pdf`. The heading is Data Availability Statement. The table-of-contents graphic is on the last page.
No separate 300 dpi TIF or RGB EPS graphic was written. The achemso frame sits outside the 3.25 in by 1.75 in box.
The cover letter has no telephone number, no street address, and no reviewers.

## Number trace

284 numbers. 283 match a committed script and a raw log, a per-seed table, a source line, or the layout whitelist. 0 are unmatched. 1 has no source. No number was filled in.

The one missing number is the water 16-mer, waiting on H2O16 v9. The 192-seed rates, the McNemar values, and the LJ98 rates from seeds.csv are removed. ASE/AGOX v5 and armD v1.4 have no count in the text.

paper/jctc/manuscript.tex:522	16	raw_log=	script=	match=no	no raw log
