The Quickstart said that the ChemFit bridges raise an exception from
``evaluate`` that its own loop over ``anneal.run`` scores as the worst value,
and the notes of ``anneal.chemfit`` said that the first exception the fitter
raises ends the drive and reaches the caller without ``finish``. That holds
for the calls that drive the session themselves, ``run_fitter`` among them
only on a fitter without a ``fit_anneal`` method of its own. ``run_fitter``
leaves an exception raised inside a fitter's own ``fit_anneal`` to that
method: one over ``run_benchmark`` raises it, and one that runs its own loop
over ``anneal.minimize`` scores it as the worst value with a
``RuntimeWarning`` and goes on. Both texts now limit the raise to the calls
that drive the session themselves, and say that ``run_fitter`` leaves such
an exception to a fitter's own ``fit_anneal``.
