The Quickstart said that the ChemFit bridges raise an exception from
``evaluate`` that its own loop over ``anneal.run`` scores as the worst value,
and the notes of ``anneal.chemfit`` said that the first exception the fitter
raises, or a loss that is not a real number, ends the drive, and that the
exception reaches the caller without ``finish``. That holds for the calls
that drive the session themselves, ``run_fitter`` among them only on a
fitter without a ``fit_anneal`` method of its own. Both texts now limit the
raise to those calls, and say that ``run_fitter`` leaves what becomes of
such an exception inside a fitter's own ``fit_anneal`` to that method; the
notes say so of such a loss as well.
