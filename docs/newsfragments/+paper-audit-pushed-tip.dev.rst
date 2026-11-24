The revision tests in ``pytest/test_paper_audit.py`` take their tip from the branches the audit counts as pushed: ``origin/develop``, then ``origin/main``, then the rest of the audited set.
A clone that holds other remote branches, as the CI checkout does, no longer picks the alphabetically first ref, and the helper fails with a message when no audited branch is present.
``scripts/paper_audit.py`` tests that a commit exists with one ``git cat-file --batch-check`` helper everywhere.
