New ``anneal.chemfit.fit_chemfit`` and ``ChemFitVector`` flatten a fitter
with dotted keys, build a finite box from ``fitter.bounds`` (unbounded
parameters fall back to ``init +/- default_span``), and close the run with
``fitter.finish``. ``run_benchmark`` accepts the same session when the fitter
speaks ``evaluate`` / ``step`` instead of ``ask`` / ``tell``.
