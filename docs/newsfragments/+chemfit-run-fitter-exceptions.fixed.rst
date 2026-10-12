The Quickstart said that the ChemFit bridges raise an exception from
``evaluate`` that its own loop over ``anneal.run`` scores as the worst value,
and the notes of ``anneal.chemfit`` said that the first exception the fitter
raises, or a loss that is not a real number, ends the drive, and that the
exception reaches the caller without ``finish``. That holds for the calls
that drive the session themselves, ``run_fitter`` among them only on a
fitter without a ``fit_anneal`` method of its own. ``run_fitter`` leaves
what becomes of either inside a fitter's own ``fit_anneal`` to that method:
one over ``run_benchmark`` ends the drive the same way, and one that runs its
own loop over ``anneal.minimize`` scores an exception the fitter raises as
the worst value with a ``RuntimeWarning`` and goes on. Both texts now limit
the raise to the calls that drive the session themselves, and say that
``run_fitter`` leaves such an exception to a fitter's own ``fit_anneal``;
the notes say so of such a loss as well.
