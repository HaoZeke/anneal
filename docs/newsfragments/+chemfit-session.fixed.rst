The ChemFit bridges ``fit_anneal``, ``fit_chemfit``, ``run_benchmark`` and
``run_fitter`` drive ``evaluate`` / ``step`` when the fitter has them and
``ask`` / ``tell`` otherwise, and refuse a fitter with neither pair before
``init``; only ``run_benchmark`` still drives ``evaluate`` or ``ask`` alone,
with a ``FutureWarning``. On current ChemFit, ``fit_anneal``,
``fit_chemfit`` and ``run_fitter`` called the missing ``ask``, scored every
candidate as ``+inf`` with the ``AttributeError`` swallowed, and handed
``finish`` a point that was never evaluated. The first exception the fitter
raises, or a loss that is not a real number, now ends the fit: the fitter is
not called again, ``finish`` is skipped, and the exception reaches the
caller.
