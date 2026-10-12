``anneal.chemfit.run_fitter`` called a fitter that has its own ``fit_anneal``
with its own method names, with ``budget`` by keyword, and with ``method``,
``preset`` and ``seed`` whatever the signature of ``fit_anneal``. So
``run_fitter(fitter, budget)`` failed on a ``fit_anneal`` that runs
``run_benchmark`` and takes that function's names, ``"portfolio"``,
``"boltzmann"``, ``"fast"`` and ``"gsa"``, as did ``method="sa"``; a
``fit_anneal`` without a ``preset`` parameter refused every call, as did one
that takes its budget only by position, and one without ``method`` refused
the default call too. ``run_fitter`` now passes ``"portfolio"`` for
``"global_optimize"`` and the preset's driver, or ``"boltzmann"`` without
one, for ``"sa"``, the drivers it runs for those names on a fitter without
``fit_anneal``, and passes only what the signature of ``fit_anneal`` takes:
the budget, by position when its first parameter is a positional-only
``budget``, the method and ``seed`` when it has those parameters, a preset
only when one is given, and each keyword as it is. The default method and
seed stay out of a ``fit_anneal`` without such a parameter, so one without
``method`` runs as it stands. A method or seed other than the defaults, a
preset, or a keyword that ``fit_anneal`` has no parameter for, and that it
cannot take through ``**kwargs``, or that it takes only by position, is a
TypeError naming it and the fitter's ``fit_anneal``, raised before anything
is evaluated. So is every call to a ``fit_anneal`` whose positional-only
``budget`` comes after another parameter, and the TypeError says which
parameter a budget passed by position would fill. As on a fitter without
``fit_anneal``, an unknown method is refused before ``fit_anneal`` is
called, and a preset that the driver passed as ``method`` does not run is
ignored with a FutureWarning. Where ``fit_anneal`` takes no ``preset``, or
takes it only by position, that warning says so in place of advising
``method="sa"``, under which ``run_fitter`` would refuse the preset.
