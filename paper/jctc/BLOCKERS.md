# BLOCKERS

No `XXX` token is in `manuscript.tex`, `si.tex`, or `cover-letter.tex`.
`si-six.tex` is removed. Its values 65670 and 30.042525 had no raw log.
The hexamer energy `-197.7750` kJ/mol is removed. The committed TIP4P-6 geometry with the anneal pair constants evaluates to -197.7751 kJ/mol, not -197.7750.

## A. Waiting on running science

These campaigns are named and not sealed. No count is guessed.

- LJ75 seeds 0 through 199. The manuscript states that no hit count is in the sealed record.
- ASE/AGOX v5.
- armD-v1.4, the LJ104 arm D versus GMIN head-to-head.
- H2O16 v9, the water 16-mer against CREST.

`paper/jctc/manuscript.tex:350`

```
No hit count for those seeds is in the sealed record.
```

`paper/jctc/manuscript.tex:550`

```
The water 16-mer count is separate open work.
```

`paper/jctc/manuscript.tex:551`

```
Apart from the hexamer campaign.
```

`paper/jctc/manuscript.tex:552`

```
This comparison leaves out the Atomic Simulation Environment.
```

`paper/jctc/manuscript.tex:553`

```
The atomistic global optimization framework (AGOX).
```

`paper/jctc/manuscript.tex:564`

```
No hit count for the LJ\(_{75}\) seeds 0 through 199 is in the sealed record.
```


## B. Rohit-only decisions

The AI-drafting sentence is unchanged.

`paper/jctc/manuscript.tex:588`

```
Generative AI tools assisted with drafting and editing the text.
```

`paper/jctc/manuscript.tex:589`

```
The authors reviewed and edited all content.
```

`paper/jctc/manuscript.tex:590`

```
And take full responsibility for it.
```


The submission DOI is not assigned.

`paper/jctc/manuscript.tex:613`

```
The Zenodo digital object identifier (DOI) for this submission is not assigned.
```


Cover-letter telephone:

`paper/jctc/cover-letter.tex:13`

```
Telephone: \todo{Rohit}
```


Cover-letter suggested reviewers:

`paper/jctc/cover-letter.tex:43`

```
Suggested reviewers follow.
```

`paper/jctc/cover-letter.tex:44`

```
\todo{Rohit} \todo{Rohit} \todo{Rohit}
```


Version 0.12.0 has no release tag.

`paper/jctc/manuscript.tex:615`

```
The crate records version 0.12.0.
```

`paper/jctc/manuscript.tex:616`

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

345 numbers. 64 match a committed script and a raw or per-seed log. 10 have a log and no committed script. 271 have no log. No number was filled in.

paper/jctc/manuscript.tex:33	02139	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:42	98	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:42	0.7	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:44	3.0.2	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:44	1\times 10^{-7}	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:44	1	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:44	1.35\times 10^{-7}	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:74	98	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:75	0.7	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:75	1	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:75	98	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:85	1	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:88	0.8	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:89	1	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:96	0.001	raw_log=	script=	match=no	no raw log
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
paper/jctc/manuscript.tex:216	2	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:219	2	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:226	0.8	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:226	0.6	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:226	0.4	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:239	0.8	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:308	75	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:331	38	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:331	75	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:331	98	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:332	104	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:333	10	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:333	15	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:333	20	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:340	75	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:345	200	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:348	75	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:348	0	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:348	199	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:349	4\times 10^{6}	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:352	75	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:352	200	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:356	38	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:356	75	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:357	98	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:358	38	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:359	75	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:360	-396.282249	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:360	1.210	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:361	-397.492331	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:362	98	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:362	0.022	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:363	-543.665361	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:364	38	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:375	38	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:375	75	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:376	98	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:387	38	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:388	0.15	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:398	38	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:399	75	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:399	200	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:407	0	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:407	191	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:407	4000	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:419	4000	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:420	4000	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:422	4000	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:423	4000	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:428	4000	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:428	0	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:430	3560	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:431	4546.25	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:438	3.0.2	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:444	30	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:447	0.001	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:449	0.01	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:449	500	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:450	1\times 10^{-7}	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:477	4000	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:477	4000	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:482	4000	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:490	1\times 10^{-7}	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:491	1	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:491	1.3509\times 10^{-7}	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:492	4000	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:493	1.3505\times 10^{-7}	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:495	0.00212	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:507	4000	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:509	4000	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:512	0	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:512	-1.1214\times 10^{-8}	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:513	-1.1210\times 10^{-8}	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:513	0.001004	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:513	0.000987	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:514	1	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:514	-1.3509\times 10^{-7}	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:515	-1.3505\times 10^{-7}	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:515	0.002023	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:515	0.001964	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:516	2	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:516	-2.5460\times 10^{-9}	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:517	-2.5850\times 10^{-9}	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:517	0.000532	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:517	0.000477	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:518	3	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:518	-4.1510\times 10^{-9}	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:519	-4.1430\times 10^{-9}	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:519	0.000322	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:519	0.000308	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:520	4	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:520	-6.8980\times 10^{-8}	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:521	-6.9022\times 10^{-8}	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:521	0.002116	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:521	0.002068	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:522	5	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:522	-2.1900\times 10^{-9}	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:523	-2.2000\times 10^{-9}	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:523	0.000157	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:523	0.000134	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:528	6	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:528	207	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:531	544.0	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:531	18.8	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:531	4000	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:532	195.5	raw_log=	script=paper/jctc/scripts/hexamer_numbers.py	match=no	committed script reproduces the value from a table that is not a raw log
paper/jctc/manuscript.tex:534	18	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:550	16	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:561	1\times 10^{-7}	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:563	1	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:564	75	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:564	0	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:564	199	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:614	0.11.1	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:615	0.12.0	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:616	0.12.0	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:620	0.001	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:622	118	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:622	119	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:623	1e-3	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:624	-415.971529	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:625	1e-6	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:627	0.001	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:628	0.01	raw_log=	script=	match=no	no raw log
paper/jctc/manuscript.tex:628	0.01	raw_log=	script=	match=no	no raw log
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
paper/jctc/si.tex:65	12	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:65	-1	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:66	6	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:66	-1	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:68	6	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:68	12	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:70	6	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:70	12	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:75	64	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:76	20	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:76	16	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:79	98	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:81	24	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:81	64	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:82	2000000	raw_log=paper/jctc/logs/paired_bootstrap_lj98.txt	script=	match=no	log has the value but no committed script derives it
paper/jctc/si.tex:83	1513982.75	raw_log=paper/jctc/logs/paired_bootstrap_lj98.txt	script=	match=no	log has the value but no committed script derives it
paper/jctc/si.tex:84	64	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:84	64	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:85	158540.5	raw_log=paper/jctc/logs/paired_bootstrap_lj98.txt	script=	match=no	log has the value but no committed script derives it
paper/jctc/si.tex:86	201463.5	raw_log=paper/jctc/logs/paired_bootstrap_lj98.txt	script=	match=no	log has the value but no committed script derives it
paper/jctc/si.tex:88	24	raw_log=	script=	match=no	no raw log
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
paper/jctc/si.tex:103	200	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:107	56	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:111	56	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:118	200	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:122	2023	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:126	218	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:127	2	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:127	93	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:127	15	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:128	52	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:128	14	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:128	1	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:129	4	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:129	1	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:130	55	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:131	75	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:145	38	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:147	75	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:148	98	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:162	2	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:162	6	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:173	0.48	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:174	38	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:176	600	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:183	0.48	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:184	38	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:186	2.429	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:187	3.051	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:192	0.48	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:193	75	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:195	1.210	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:200	0.48	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:201	98	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:203	0.022	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:204	0.119	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:209	0.48	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:210	38	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:210	75	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:210	98	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:217	0.48	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:218	38	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:221	2.429	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:226	0.48	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:227	75	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:230	1.210	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:235	0.48	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:236	98	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:239	0.022	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:240	0.119	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:245	0.001	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:247	118	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:247	119	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:248	1e-3	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:249	-415.971529	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:251	1e-6	raw_log=	script=	match=no	no raw log
paper/jctc/si.tex:253	0.001	raw_log=	script=	match=no	no raw log
paper/jctc/cover-letter.tex:5	1	raw_log=	script=	match=no	no raw log
paper/jctc/cover-letter.tex:5	1	raw_log=	script=	match=no	no raw log
paper/jctc/cover-letter.tex:24	2023	raw_log=	script=	match=no	no raw log
paper/jctc/cover-letter.tex:28	98	raw_log=	script=	match=no	no raw log
paper/jctc/cover-letter.tex:32	1\times 10^{-7}	raw_log=	script=	match=no	no raw log
paper/jctc/cover-letter.tex:32	1	raw_log=	script=	match=no	no raw log
paper/jctc/cover-letter.tex:33	4000	raw_log=	script=	match=no	no raw log
paper/jctc/cover-letter.tex:41	3.25	raw_log=	script=	match=no	no raw log
paper/jctc/cover-letter.tex:41	1.75	raw_log=	script=	match=no	no raw log
