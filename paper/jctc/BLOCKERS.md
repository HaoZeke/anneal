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

These campaigns are named and not sealed. No count is guessed.

- LJ75 seed sweep. The manuscript states that no hit count is in the sealed record. The seed bounds are not printed, because that run is unsealed.
- ASE/AGOX v5.
- armD-v1.4, the LJ104 arm D versus GMIN head-to-head.
- H2O16 v9, the water 16-mer against CREST.

`paper/jctc/manuscript.tex:350`

```
No hit count for those seeds is in the sealed record.
```

`paper/jctc/manuscript.tex:523`

```
The water 16-mer count is separate open work.
```

`paper/jctc/manuscript.tex:524`

```
Apart from the hexamer campaign.
```

`paper/jctc/manuscript.tex:525`

```
This comparison leaves out the Atomic Simulation Environment.
```

`paper/jctc/manuscript.tex:526`

```
The atomistic global optimization framework (AGOX).
```

`paper/jctc/manuscript.tex:534`

```
No hit count for the unfinished LJ\(_{75}\) seed sweep is in the sealed record.
```


## B. Rohit-only decisions

The AI-drafting sentence is unchanged.

`paper/jctc/manuscript.tex:558`

```
Generative AI tools assisted with drafting and editing the text.
```

`paper/jctc/manuscript.tex:559`

```
The authors reviewed and edited all content.
```

`paper/jctc/manuscript.tex:560`

```
And take full responsibility for it.
```


The submission DOI is not assigned.

`paper/jctc/manuscript.tex:583`

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

`paper/jctc/manuscript.tex:584`

```
The crate records version 0.12.0.
```

`paper/jctc/manuscript.tex:585`

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

259 numbers. 258 match a committed script and a raw or per-seed log. 0 have a log and no committed script. 1 has no log. No number was filled in.

The one missing number is the water 16-mer, waiting on H2O16 v9. The 192-seed rates and the McNemar values are removed. ASE/AGOX v5 and armD v1.4 have no count in the text.

paper/jctc/manuscript.tex:523	16	raw_log=	script=	match=no	no raw log
