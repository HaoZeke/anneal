The README, the ChemFit Integration tutorial and the notes of
``anneal.chemfit`` said that ``run_fitter``, like ``fit_anneal``,
``fit_chemfit`` and ``run_benchmark``, refuses a fitter with neither session
pair before ``init``, and the README said so of any invalid argument too;
the Quickstart, the Choose a Driver how-to and the package docstring said
that all four drive the fitter's session. ``run_fitter`` does so only on a
fitter without a ``fit_anneal`` method of its own. It hands a fitter with one
to that method without checking which pair it has, and refuses only what it
cannot pass to it, so what that method does with the session, ``init``
included, decides the rest: a fitter with neither pair whose ``fit_anneal``
drives no session is not refused, and a ``fit_anneal`` that calls ``init``
before it refuses a budget of 0 refuses it after ``init``. These texts now
limit the feature check, and the refusals before ``init``, to the calls that
drive the session themselves, and say that ``run_fitter`` hands a fitter
with its own ``fit_anneal`` to that method without them.
