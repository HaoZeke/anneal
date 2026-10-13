The notes of ``anneal.chemfit``, the docstrings of ``fit_anneal``,
``fit_chemfit`` and ``run_benchmark``, the README, the Quickstart, the
ChemFit Integration tutorial and the Choose a Driver how-to said that the
first exception the fitter raises ends the fit without ``finish``. When
nothing raised before it, an exception from ``finish`` itself is that first
one, and it reaches the caller as it is once ``finish`` has been called, a
single time and as the fitter's last call. These texts now limit ``finish``
being skipped to an exception raised before it, and say what becomes of one
that ``finish`` raises.
