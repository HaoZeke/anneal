# Claims at 3a8ef25e, kept beside the v2 edit

The Wales-group program (GMIN) is the external basin-hopping code named below.
This inventory is the v3 draft at commit 3a8ef25e.
Each numbered line is one softening sentence in that draft.
The edit changes those sentences and leaves the title, the author block, the Metropolis step, and the bibliography in place.

## Sentences

1. `manuscript.tex:52` refuses to order the Oakley table against the force-call ledger.
2. `manuscript.tex:55` describes size tests that remain on main.
3. `manuscript.tex:56` points at a local head instead of the result.
4. `manuscript.tex:74` uses the modal "would" for an unconverged quench.
5. `manuscript.tex:98` prints 0.7 with no derivation in the article.
6. `manuscript.tex:118` defers hit tests to a later campaign.
7. `manuscript.tex:119` defers paired cost tests to a later campaign.
8. `manuscript.tex:120` states that no second code was run.
9. `manuscript.tex:153` calls the Oakley means a different protocol.
10. `manuscript.tex:154` lists experiments a later campaign still has to run.
11. `manuscript.tex:185` says the measured commit is absent from main.
12. `si.tex:31` leaves the density 0.7 open.
13. `si.tex:37` says main still contains both atom-count tests.
14. `si.tex:38` contrasts the six runs with an earlier campaign row.
15. `si.tex:65` says the article reports no water hit count.
16. `si.tex:66` assigns a later water campaign.
17. `si.tex:80` says the eight experiments were not run.
18. `si.tex:81` describes the multi-replica experiment.
19. `si.tex:82` describes the rival-code experiment, including GMIN.
20. `si.tex:83` describes the ablation experiment.
21. `si.tex:84` describes the shared hit-rule experiment.
22. `si.tex:85` describes the held-out size experiment.
23. `si.tex:86` describes the water-coefficient experiment.
24. `si.tex:87` describes the external-potential experiment.
25. `si.tex:88` describes a conformer experiment or dropping that section.
26. `manuscript.tex:110` sets a different call ceiling at each size.

The draft contains no fix-me marker and no placeholder token.
Sentence 12 is the open derivation.
The before count is 26 sentences.

## Code problem on 90c898dd

Ceiling 109753, miss charge 219506, six seeds, commit 90c898dd.
These counts are a code problem for the anneal head.
They are not the finding of the article.
The article keeps the six-run table as a placeholder for the head that replaces this one.

LJ38: 5/6, median PAR-2 67570.5, type-7 interquartile range 28016.25.
Sorted PAR-2: 10249, 55324, 65670, 69471, 91412, 219506.
Seed 4 misses at energy -173.252378.
LJ55: 6/6, median 7164, type-7 interquartile range 8714.25.
LJ75, LJ98, and LJ104: 0/6, median 219506, interquartile range 0.

## Head-to-head at the same ceiling

GMIN, ABCluster, and OGOLEM, six seeds, 109753 potential calls.
GMIN hits 0/6 at every size.
ABCluster hits 1/6 at LJ55 (seed 5) and 0/6 elsewhere.
The call index of that LJ55 minimum was not written.
OGOLEM hits 6/6 at LJ55, median penalised cost 53055, type-7 interquartile range 8067, and 0/6 elsewhere.
Logs: `/src/anneal_repro/logs/competitors/`.
