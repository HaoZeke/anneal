# BLOCKERS

No `XXX` token is in `manuscript.tex`, `si.tex`, or `cover-letter.tex`.
`si-six.tex` is removed. Its values 65670 and 30.042525 had no raw log.
The hexamer energy `-197.7750` kJ/mol is removed. The committed TIP4P-6 geometry with the anneal pair constants evaluates to -197.7751 kJ/mol, not -197.7750.

## A. Waiting on running science

These campaigns are named and not sealed. No count is guessed.

- LJ75 seed sweep. The manuscript states that no hit count is in the sealed record. The seed bounds are not printed, because that run is unsealed.
- ASE/AGOX v5.
- armD-v1.4, the LJ104 arm D versus GMIN head-to-head.
- H2O16 v9, the water 16-mer against CREST.

`paper/jctc/manuscript.tex:350`

```
No hit count for those seeds is in the sealed record.
```

`paper/jctc/manuscript.tex:554`

```
The water 16-mer count is separate open work.
```

`paper/jctc/manuscript.tex:555`

```
Apart from the hexamer campaign.
```

`paper/jctc/manuscript.tex:556`

```
This comparison leaves out the Atomic Simulation Environment.
```

`paper/jctc/manuscript.tex:557`

```
The atomistic global optimization framework (AGOX).
```

`paper/jctc/manuscript.tex:568`

```
No hit count for the unfinished LJ\(_{75}\) seed sweep is in the sealed record.
```


## B. Rohit-only decisions

The AI-drafting sentence is unchanged.

`paper/jctc/manuscript.tex:592`

```
Generative AI tools assisted with drafting and editing the text.
```

`paper/jctc/manuscript.tex:593`

```
The authors reviewed and edited all content.
```

`paper/jctc/manuscript.tex:594`

```
And take full responsibility for it.
```


The submission DOI is not assigned.

`paper/jctc/manuscript.tex:617`

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

`paper/jctc/manuscript.tex:618`

```
The crate records version 0.12.0.
```

`paper/jctc/manuscript.tex:619`

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
The 192-seed rates are backed by `paper/jctc/data/water_v45b/water-v45b-perseed.csv` and `paper/jctc/scripts/water_results_table.py`.
`si.tex` has no table.

## Format not met

The abstract is four sentences. Supporting information is `si.pdf`. The heading is Data Availability Statement. The table-of-contents graphic is on the last page.
No separate 300 dpi TIF or RGB EPS graphic was written. The achemso frame sits outside the 3.25 in by 1.75 in box.
The cover letter has no telephone number, no street address, and no reviewers.

## Number trace

293 numbers. 292 match a committed script and a raw or per-seed log. 0 have a log and no committed script. 1 has no log. No number was filled in.

The one missing number is the water 16-mer, waiting on H2O16 v9. ASE/AGOX v5 and armD v1.4 have no count in the text.

paper/jctc/manuscript.tex:554	16	raw_log=	script=	match=no	no raw log
