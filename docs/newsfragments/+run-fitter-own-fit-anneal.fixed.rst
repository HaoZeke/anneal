``anneal.chemfit.run_fitter`` passed its own method names to a fitter that
has its own ``fit_anneal``, so its default ``method="global_optimize"``, and
``method="sa"``, failed on a ``fit_anneal`` that runs ``run_benchmark`` and
takes that function's names: ``"portfolio"``, ``"boltzmann"``, ``"fast"`` and
``"gsa"``. ``run_fitter`` now passes ``"portfolio"`` for
``"global_optimize"`` and the preset's driver, or ``"boltzmann"`` without
one, for ``"sa"``, the drivers it runs for those names on a fitter without
``fit_anneal``. As there, an unknown method is refused before ``fit_anneal``
is called, and a preset that the method's driver does not run is ignored
with a FutureWarning.
