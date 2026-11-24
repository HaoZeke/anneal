``scripts/paper_audit.py`` runs on a Python without numpy.
``scripts/sc96_78_common.py`` imports numpy inside ``write_start``, the only function that draws starts, so the audit, which loads ``scripts/sc96_78_hits.py`` for the Sutton-Chen step, no longer pulls it in.
``pytest/test_paper_audit.py`` runs the whole audit entry point with numpy, scipy and matplotlib refused by an import hook, requires exit status 0 and that no blocked import was even attempted, and checks that the same harness fails for a script that imports numpy.
