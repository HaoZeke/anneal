``anneal.chemfit.fit_anneal`` and ``fit_chemfit`` drive ChemFit 4 fitters,
which name the loss call ``evaluate`` and ``step``. They called ``ask`` and
``tell`` only, so with a ChemFit 4 fitter every evaluation raised
``AttributeError``, ChemFit counted none, and the fit returned the starting
parameters with a warning. All three bridges call ``evaluate`` / ``step`` when
the fitter has them and ``ask`` / ``tell`` otherwise. The classical drivers of
``fit_anneal``, ``fit_chemfit`` and ``run_benchmark`` spend exactly ``budget``
evaluations, the start included, over ``ceil(budget / steps_per_epoch)``
epochs of the cooling schedule; they made
``1 + steps_per_epoch * (budget // steps_per_epoch)`` calls, one more than a
budget divisible by ``steps_per_epoch``. ``fit_chemfit`` starts the portfolio
at the fitter's initial parameters, as the other bridges do.
