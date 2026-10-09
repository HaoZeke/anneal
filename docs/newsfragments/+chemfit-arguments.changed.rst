The ChemFit bridges now raise, before ``fitter.init()``, on arguments anneal
0.10.0 coerced or ignored without a word. A fractional or string ``budget``
or ``seed`` raises instead of passing through ``int()`` (a budget of 60.5
ran 60 evaluations), and a negative seed, or one of ``2**64`` or more,
raises ``ValueError`` where it raised ``OverflowError`` after ``init``.
``tell_every`` and ``steps_per_epoch`` must be positive whole numbers, where
0 ran as 1 and 2.5 as 2; ``steps_per_epoch`` is checked under the portfolio
too, which does not use it. ``fit_chemfit`` raises ``TypeError`` on a
keyword no preset takes, such as ``t_inti`` or ``tell_evry``, which it
ignored, and on a preset keyword of its method that is not a number, which
it passed through ``float()``. A ``preset_kwargs`` that is not a dict, and a
``preset`` that is not ``Boltzmann()``, ``Fast()`` or ``Gsa()``, raise
``TypeError`` under the portfolio too. ``run_benchmark`` raises
``ValueError`` on an unknown method, where it ran the preset it was given or
raised ``KeyError`` after ``init``. A bounds entry that is not one
``(lower, upper)`` pair, such as a 3-tuple or a list of three per-element
pairs, raises in ``fit_chemfit`` and ``chemfit_box``, which ignored it in
favour of ``default_span``; a two-item list is that pair, each side a scalar
or an array of the leaf's shape. A parameter that is not finite raises in
``ChemFitVector`` and in ``fit_chemfit``, whose portfolio ignored it; a
complex leaf raises instead of losing its imaginary part; and
``resolve_bounds`` raises on a lower bound above the upper instead of
returning the inverted box. ``fit_chemfit`` and ``run_fitter`` now read
method names in any case, as ``fit_anneal`` and ``run_benchmark`` did.
