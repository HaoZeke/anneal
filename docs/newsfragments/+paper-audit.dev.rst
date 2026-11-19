``scripts/paper_audit.py`` recomputes the numbers in the manuscript and in the bench tables from the committed rows.
It checks each row file for a code revision and a host, and checks that the revision each row cites is in the history or in ``bench/revision-aliases.csv``.
``bench/unreachable-revisions.csv`` lists the rows that cite each unpushed revision, and ``--write-revision-map`` rewrites it.
The audit exits 1 on drift, and 2 when the clone is shallow or lacks the commits it needs to answer.
``pytest/test_paper_audit.py`` runs every check and feeds the audit drifted cells to see that it reports them.
The lint job of CI runs the audit as its last step.
