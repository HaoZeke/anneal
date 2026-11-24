``scripts/paper_audit.py`` runs on a Python without numpy.
``scripts/sc96_78_common.py`` imports numpy inside ``write_start``, the only function that draws starts, so the audit, which loads ``scripts/sc96_78_hits.py`` for the Sutton-Chen step, no longer pulls it in.
``pytest/test_paper_audit.py`` runs the Sutton-Chen check with numpy blocked.
