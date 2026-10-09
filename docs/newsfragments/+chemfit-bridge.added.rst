New ``anneal.chemfit`` bridge for gradient-free ChemFit fitting:
``fit_chemfit`` drives the ``init`` / ``ask`` / ``tell`` / ``finish``
protocol from ChemFit initial parameters and bounds (unbounded parameters
fall back to ``init +/- default_span``), defaults to the Thompson-allocated
portfolio driver, and offers the bound-respecting classical chains for
ablations. See ``examples/chemfit_anneal.py``.
