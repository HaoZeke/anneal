``anneal.chemfit.FEATURES`` is a frozen set of what a caller of the ChemFit
bridges may test for before using it, rather than a version or a source
commit. It holds ``"batch_size"``, which ``fit_anneal``, ``fit_chemfit`` and
``run_benchmark`` take as a named parameter, so a check of
``run_benchmark``'s signature gives the same answer. The ChemFit how-to
shows the check.
