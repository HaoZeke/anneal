The ChemFit bridges ``fit_anneal``, ``fit_chemfit``, ``run_benchmark`` and
``run_fitter`` drive ``evaluate`` / ``step`` when the fitter has them and
``ask`` / ``tell`` otherwise, and refuse a fitter with neither pair before
``init``; only ``run_benchmark`` still drives ``evaluate`` or ``ask`` without
its partner, with a ``FutureWarning``. On current ChemFit, ``fit_anneal``,
``fit_chemfit`` and ``run_fitter`` called the missing ``ask``, scored every
candidate as ``+inf`` with the ``AttributeError`` swallowed, and handed
``finish`` a point that was never evaluated. The first exception the fitter
raises, or a loss that is not one real number, now ends the fit: the fitter is
not called again, ``finish`` is skipped, and the exception reaches the caller.
anneal 0.10.0 read a numeric string, a bool or the real part of a NumPy
complex number as the loss, and scored a loss it could not read, such as an
array of two values, as the worst value. Every bridge reads a one-element
list, tuple or array of any shape as its element, which 0.10.0 read only as an
array, with NumPy's ``DeprecationWarning``, or as a list from ``ask`` in
``fit_chemfit`` and ``run_benchmark``.
