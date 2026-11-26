``scripts/paper_audit.py`` fails when a cross-document label file differs from the output of ``cross_labels.py``, or when ``paper/jctc/logs/pdf-length.txt`` is missing, lacks a count, or holds a sha256 that does not match its source.
A rendered length outside the limits of ``pdf_length.py`` is reported as expected and does not fail the run.
``pytest/test_paper_audit.py`` copies the paper sources to a scratch root and checks that an edited ``si.tex`` or ``macros.tex`` fails the audit.
