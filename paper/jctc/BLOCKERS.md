# BLOCKERS

No `XXX` token is in `manuscript.tex`, `si.tex`, or `cover-letter.tex`.

## A. Waiting on running science

The paper sources have no sentence for the LJ104 arm D versus GMIN head-to-head on seeds 5000–5029.

The water 16-mer is named. The sentence does not name CREST v9.

`paper/jctc/manuscript.tex:551`

```
The water 16-mer count is separate open work.
```

`paper/jctc/manuscript.tex:552`

```
Apart from the hexamer campaign.
```

ASE, AGOX, and GMIN are named. The lines do not say ASE/AGOX v5.

`paper/jctc/manuscript.tex:553`

```
This comparison leaves out the Atomic Simulation Environment.
```

`paper/jctc/manuscript.tex:554`

```
The atomistic global optimization framework (AGOX).
```

`paper/jctc/manuscript.tex:555`

```
And the Wales-group basin-hopping program (GMIN).
```

`paper/jctc/manuscript.tex:556`

```
\todo{ASE, AGOX and GMIN counts are not in a sealed kit through v1.30.}
```

LJ75, at the lines that were near 358 before this assembly:

`paper/jctc/manuscript.tex:348`

```
The LJ\(_{75}\) seeds 0 through 199 run serially at a cap of.
```

`paper/jctc/manuscript.tex:349`

```
\(4\times 10^{6}\) force calls.
```

`paper/jctc/manuscript.tex:350`

```
No hit count is reported until that run finishes.
```

`paper/jctc/manuscript.tex:351`

```
\todo{The LJ75 hit count is not in a sealed kit through v1.30.}
```

## B. Rohit-only decisions

Cover-letter telephone:

`paper/jctc/cover-letter.tex:13`

```
Telephone: \todo{Rohit}
```

Cover-letter suggested reviewers. Three placeholders, and no names, affiliations, or emails:

`paper/jctc/cover-letter.tex:43`

```
Suggested reviewers follow.
```

`paper/jctc/cover-letter.tex:44`

```
\todo{Rohit} \todo{Rohit} \todo{Rohit}
```

AI disclosure. The manuscript states a use and leaves the tool sentence for Rohit. This is the only AI string in the paper sources. No Cursor or Grok string is in the TeX or BibTeX.

`paper/jctc/manuscript.tex:591`

```
Generative AI tools assisted with drafting and editing the text.
```

`paper/jctc/manuscript.tex:592`

```
The authors reviewed and edited all content.
```

`paper/jctc/manuscript.tex:593`

```
And take full responsibility for it.
```

`paper/jctc/manuscript.tex:594`

```
% OPTION for Rohit. This comment is not the submitted disclosure.
```

`paper/jctc/manuscript.tex:595`

```
% The journal asks for one sentence that names the tool,
```

`paper/jctc/manuscript.tex:596`

```
% the version, the purpose,
```

`paper/jctc/manuscript.tex:597`

```
% and the portions of the manuscript. A tool is not an author.
```

`paper/jctc/manuscript.tex:598`

```
% Rohit fills or deletes the sentence before submission.
```

`paper/jctc/manuscript.tex:599`

```
% OPTION: <tool> <version> assisted with <purpose> in <portions>.
```

Co-author list. The manuscript email is Debabrata Goswami only. Rohit's email is on the cover letter. The other authors have no email in the sources.

`paper/jctc/manuscript.tex:15`

```
\author{Rohit Goswami}
```

`paper/jctc/manuscript.tex:16`

```
\affiliation{SURF B.V., Amsterdam}
```

`paper/jctc/manuscript.tex:20`

```
\author{Ruhila Goswami}
```

`paper/jctc/manuscript.tex:23`

```
\author{Amrita Goswami}
```

`paper/jctc/manuscript.tex:25`

```
\author{Moritz Sallermann}
```

`paper/jctc/manuscript.tex:27`

```
\author{Sonaly Goswami}
```

`paper/jctc/manuscript.tex:30`

```
\author{Debabrata Goswami}
```

`paper/jctc/manuscript.tex:36`

```
\email{dgoswami@iitk.ac.in}
```

`paper/jctc/cover-letter.tex:12`

```
Email: \url{rgoswami@ieee.org}\\
```

Version 0.12.0. The tag is not assigned. No release was created here.

`paper/jctc/manuscript.tex:618`

```
The newest released tag is v0.11.1.
```

`paper/jctc/manuscript.tex:619`

```
The crate records version 0.12.0.
```

`paper/jctc/manuscript.tex:620`

```
The release tag for version 0.12.0 is not assigned.
```

Zenodo DOI. The concept record is cited. The submission DOI is a placeholder. No upload was made.

`paper/jctc/manuscript.tex:614`

```
The archive of the software line is the Zenodo concept record.
```

`paper/jctc/manuscript.tex:615`

```
\url{https://doi.org/10.5281/zenodo.10672746}.
```

`paper/jctc/manuscript.tex:616`

```
The Zenodo digital object identifier (DOI) for this submission is.
```

`paper/jctc/manuscript.tex:617`

```
\todo{DOI}.
```

USPEX and CALYPSO. The paper sources contain no sentence on either licence.

Funding:

`paper/jctc/manuscript.tex:601`

```
\section*{Funding}
```

`paper/jctc/manuscript.tex:602`

```
This research received no specific grant.
```

`paper/jctc/manuscript.tex:603`

```
No funding agency in the public, commercial.
```

`paper/jctc/manuscript.tex:604`

```
Or not-for-profit sectors supported it.
```

Acknowledgements:

`paper/jctc/manuscript.tex:584`

```
\section*{Acknowledgments}
```

`paper/jctc/manuscript.tex:585`

```
Rohit Goswami thanks his pets Ar\'{\i}, Yoda, Jude, Crystee, Thor, and.
```

`paper/jctc/manuscript.tex:586`

```
Loki, and the garden cats.
```

`paper/jctc/manuscript.tex:587`

```
In memoriam TuiTui, Snoopy, and Cosmo.
```

`paper/jctc/manuscript.tex:588`

```
He thanks his cactus Ami and his in-laws Nagaselvamani and.
```

`paper/jctc/manuscript.tex:589`

```
Sankaranarayan.
```

`paper/jctc/manuscript.tex:590`

```
Ruhila Goswami set the figure colours.
```

## Head-to-head tables

`tab:crest-counters`, caption at `paper/jctc/manuscript.tex:459`. Columns are seed, old counter, and energy+gradient. The table does not carry hardware, wall time, n, median/IQR, or the exact command.

`tab:hexamer-gaps`, caption at `paper/jctc/manuscript.tex:499`. Columns are seed, two gaps, and two deviations. The table does not carry hardware, wall time, n, median/IQR, or the exact command.

The 192-seed comparison is prose, not a table. It states n, the call PAR-2 median and interquartile range, and the budget. It does not state hardware or the exact command.

`si.tex` has no table. `si-six.tex` is not included.

## Format not met

The abstract is four sentences. The supporting information is `si.pdf`. The data-availability heading is "Data Availability Statement". The table-of-contents graphic is on the last manuscript page, labeled "For Table of Contents Only", and the artwork file is 3.25 in by 1.75 in.

These items are still open:

- The cover letter has no telephone number and no street postal address. The affiliation on the letter is SURF B.V., Amsterdam.
- The three suggested reviewers are placeholders.
- Co-author emails are missing except the two addresses quoted above.
- The AI sentence does not name the tool, the version, the purpose, and the portions.
- The submission Zenodo DOI is a placeholder, and version 0.12.0 has no release tag.
- No USPEX or CALYPSO licence sentence is in the sources.
- The guidelines ask for a separate TOC graphic as a 300 dpi TIF or an RGB EPS. The manuscript embeds the PDF. No TIF or EPS was written. The achemso class draws a frame outside the 3.25 in by 1.75 in box.
- `\todo` marks remain in the PDF because the items above are open. The journal asks for a file free of annotations.

## Number trace

421 numbers. 38 match a committed script and a raw log. 10 have a raw log and no committed script. 373 have no raw log. No number was filled in.

paper/jctc/manuscript.tex:33	02139	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:42	98	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:42	0.7	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:44	3.0.2	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:44	1\times 10^{-7}	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:44	1	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:44	1.35\times 10^{-7}	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:67	192	raw_log=	script=paper/jctc/scripts/water_results_table.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:74	98	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:75	0.7	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:75	1	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:75	98	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:85	1	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:88	0.8	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:89	1	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:96	0.001	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:99	-2	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:108	1	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:125	98	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:128	0.7	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:128	1	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:129	0.7	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:131	1	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:134	98	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:140	38	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:140	55	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:141	75	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:142	38	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:145	38	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:150	0	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:173	0.8	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:207	1	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:207	1	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:214	0	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:214	-1	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:216	2	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:219	-1	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:219	2	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:219	-1	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:226	0.8	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:226	0.6	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:226	0.4	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:239	0.8	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:308	75	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:320	4	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:323	2	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:331	38	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:331	75	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:331	98	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:332	104	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:333	4	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:333	10	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:333	15	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:333	20	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:334	2	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:339	75	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:340	75	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:341	-2	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:345	200	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:348	75	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:348	0	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:348	199	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:349	4\times 10^{6}	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:351	75	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:351	1.30	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:353	75	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:353	200	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:357	38	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:357	75	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:358	98	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:359	38	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:360	75	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:361	-396.282249	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:361	1.210	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:362	-397.492331	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:363	98	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:363	0.022	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:364	-543.665361	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:365	38	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:371	4	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:376	38	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:376	75	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:377	98	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:388	38	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:389	0.15	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:398	4	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:399	38	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:400	75	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:400	200	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:406	192	raw_log=	script=paper/jctc/scripts/water_results_table.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:408	0	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:408	191	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:408	4000	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:417	2	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:419	121	raw_log=	script=paper/jctc/scripts/water_results_table.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:419	192	raw_log=	script=paper/jctc/scripts/water_results_table.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:420	-2	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:420	4000	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:420	-7	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:421	4000	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:422	97	raw_log=	script=paper/jctc/scripts/water_results_table.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:422	192	raw_log=	script=paper/jctc/scripts/water_results_table.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:423	-2	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:423	4000	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:423	-7	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:424	4000	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:426	0.0264	raw_log=	script=paper/jctc/scripts/water_results_table.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:427	148	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:428	192	raw_log=	script=paper/jctc/scripts/water_results_table.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:429	-2	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:429	4000	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:429	-7	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:429	0	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:430	114	raw_log=	script=paper/jctc/scripts/water_results_table.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:430	192	raw_log=	script=paper/jctc/scripts/water_results_table.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:431	-2	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:431	3560	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:431	-7	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:432	4546.25	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:434	1.37\times 10^{-6}	raw_log=	script=paper/jctc/scripts/water_results_table.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:435	0.000374	raw_log=	script=paper/jctc/scripts/water_results_table.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:439	3.0.2	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:445	30	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:448	0.001	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:449	2	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:450	0.01	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:450	500	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:451	1\times 10^{-7}	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:478	4000	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:478	4000	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:483	4000	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:491	1\times 10^{-7}	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:492	1	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:492	1.3509\times 10^{-7}	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:493	4000	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:494	1.3505\times 10^{-7}	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:496	0.00212	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:508	4000	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:510	4000	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:513	0	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:513	-1.1214\times 10^{-8}	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:514	-1.1210\times 10^{-8}	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:514	0.001004	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:514	0.000987	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:515	1	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:515	-1.3509\times 10^{-7}	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:516	-1.3505\times 10^{-7}	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:516	0.002023	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:516	0.001964	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:517	2	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:517	-2.5460\times 10^{-9}	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:518	-2.5850\times 10^{-9}	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:518	0.000532	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:518	0.000477	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:519	3	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:519	-4.1510\times 10^{-9}	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:520	-4.1430\times 10^{-9}	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:520	0.000322	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:520	0.000308	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:521	4	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:521	-6.8980\times 10^{-8}	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:522	-6.9022\times 10^{-8}	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:522	0.002116	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:522	0.002068	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:523	5	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:523	-2.1900\times 10^{-9}	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:524	-2.2000\times 10^{-9}	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:524	0.000157	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:524	0.000134	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:529	6	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:529	207	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:532	544.0	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:532	18.8	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:532	4000	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:533	195.5	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:535	18	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:542	192	raw_log=	script=paper/jctc/scripts/water_results_table.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:545	1998	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:547	1947	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:551	16	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:556	1.30	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:559	192	raw_log=	script=paper/jctc/scripts/water_results_table.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:559	121	raw_log=	script=paper/jctc/scripts/water_results_table.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:559	192	raw_log=	script=paper/jctc/scripts/water_results_table.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:560	97	raw_log=	script=paper/jctc/scripts/water_results_table.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:560	192	raw_log=	script=paper/jctc/scripts/water_results_table.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:561	148	raw_log=	script=paper/jctc/scripts/water_results_table.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:561	192	raw_log=	script=paper/jctc/scripts/water_results_table.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:562	114	raw_log=	script=paper/jctc/scripts/water_results_table.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:562	192	raw_log=	script=paper/jctc/scripts/water_results_table.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:563	1\times 10^{-7}	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:565	1	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:566	75	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:566	0	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:566	199	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:618	0.11.1	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:619	0.12.0	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:620	0.12.0	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:624	0.001	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:626	118	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:626	119	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:627	1e-3	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:628	-415.971529	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:629	1e-6	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:631	0.001	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:632	0.01	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:632	0.01	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:635	256	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:642	3.25	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:642	1.75	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:27	02139	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:34	0.001	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:36	-173.928427	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:36	38	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:37	-279.248470	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:37	55	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:37	-397.492331	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:37	75	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:38	98	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:38	-543.665361	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:38	104	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:38	-582.086642	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:39	0.8	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:41	8	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:43	0.7	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:47	38	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:49	0	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:49	30000	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:50	38	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:50	0	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:51	-173.928427	raw_log=paper/jctc/logs/lj38-plain-seed0.log	script=	match=no	log has the value but no committed script derives it
paper/jctc/si.tex:52	30000	raw_log=paper/jctc/logs/lj38-plain-seed0.log	script=	match=no	log has the value but no committed script derives it
paper/jctc/si.tex:55	1	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:57	38	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:57	4369	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:57	100	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:58	2	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:59	75	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:59	98	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:60	104	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:62	4	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:63	4	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:65	12	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:65	-1	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:66	6	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:66	-1	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:68	6	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:68	12	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:70	6	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:70	12	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:71	4	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:72	2	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:75	64	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:76	4	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:76	20	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:76	16	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:79	55	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:79	98	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:79	98	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:81	24	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:81	64	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:82	-2	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:82	2000000	raw_log=paper/jctc/logs/paired_bootstrap_lj98.txt	script=	match=no	log has the value but no committed script derives it
paper/jctc/si.tex:82	-7	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:83	1513982.75	raw_log=paper/jctc/logs/paired_bootstrap_lj98.txt	script=	match=no	log has the value but no committed script derives it
paper/jctc/si.tex:84	64	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:84	64	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:85	-2	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:85	158540.5	raw_log=paper/jctc/logs/paired_bootstrap_lj98.txt	script=	match=no	log has the value but no committed script derives it
paper/jctc/si.tex:85	-7	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:86	201463.5	raw_log=paper/jctc/logs/paired_bootstrap_lj98.txt	script=	match=no	log has the value but no committed script derives it
paper/jctc/si.tex:87	98	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:88	24	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:88	-64	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:88	12.62	raw_log=paper/jctc/logs/paired_bootstrap_lj98.txt	script=	match=no	log has the value but no committed script derives it
paper/jctc/si.tex:89	8.52	raw_log=paper/jctc/logs/paired_bootstrap_lj98.txt	script=	match=no	log has the value but no committed script derives it
paper/jctc/si.tex:89	15.22	raw_log=paper/jctc/logs/paired_bootstrap_lj98.txt	script=	match=no	log has the value but no committed script derives it
paper/jctc/si.tex:89	2000	raw_log=paper/jctc/logs/paired_bootstrap_lj98.txt	script=	match=no	log has the value but no committed script derives it
paper/jctc/si.tex:90	75	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:90	200	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:91	6	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:92	64	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:92	104	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:95	98	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:99	75	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:99	75	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:100	75	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:103	200	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:107	56	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:110	4	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:111	56	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:113	75	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:118	200	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:119	-2	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:120	75	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:120	-37535	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:120	1	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:122	2023	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:123	75	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:126	218	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:127	2	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:127	4	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:127	93	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:127	15	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:128	52	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:128	14	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:128	1	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:129	4	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:129	1	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:130	55	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:131	75	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:137	256	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:145	38	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:147	75	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:148	98	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:161	4	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:162	2	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:162	6	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:162	-197.7805	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:164	4	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:164	6	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:164	12	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:168	6	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:168	12	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:169	-197.7750	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:174	0.48	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:174	38	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:175	38	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:177	600	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:184	0.48	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:184	38	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:185	38	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:187	2.429	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:188	3.051	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:193	0.48	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:193	75	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:194	75	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:196	1.210	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:201	0.48	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:201	98	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:202	98	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:204	0.022	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:205	0.119	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:210	0.48	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:211	38	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:211	75	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:211	98	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:218	0.48	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:218	38	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:219	38	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:222	2.429	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:227	0.48	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:227	75	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:228	75	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:231	1.210	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:236	0.48	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:236	98	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:237	98	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:240	0.022	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:241	0.119	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:246	0.001	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:248	118	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:248	119	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:249	1e-3	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:250	-415.971529	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:251	8e-6	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:251	3.68e-4	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:252	1e-6	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:254	0.001	raw_log=	script=	match=no	no raw log
paper/jctc/cover-letter.tex:2	11	raw_log=	script=	match=no	no raw log
paper/jctc/cover-letter.tex:3	1	raw_log=	script=	match=no	no raw log
paper/jctc/cover-letter.tex:5	1	raw_log=	script=	match=no	no raw log
paper/jctc/cover-letter.tex:5	1	raw_log=	script=	match=no	no raw log
paper/jctc/cover-letter.tex:15	1	raw_log=	script=	match=no	no raw log
paper/jctc/cover-letter.tex:19	1	raw_log=	script=	match=no	no raw log
paper/jctc/cover-letter.tex:24	2023	raw_log=	script=	match=no	no raw log
paper/jctc/cover-letter.tex:28	98	raw_log=	script=	match=no	no raw log
paper/jctc/cover-letter.tex:32	1\times 10^{-7}	raw_log=	script=	match=no	no raw log
paper/jctc/cover-letter.tex:32	1	raw_log=	script=	match=no	no raw log
paper/jctc/cover-letter.tex:33	4000	raw_log=	script=	match=no	no raw log
paper/jctc/cover-letter.tex:39	2	raw_log=	script=	match=no	no raw log
paper/jctc/cover-letter.tex:41	3.25	raw_log=	script=	match=no	no raw log
paper/jctc/cover-letter.tex:41	1.75	raw_log=	script=	match=no	no raw log
