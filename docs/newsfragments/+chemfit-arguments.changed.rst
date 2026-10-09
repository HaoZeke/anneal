The ChemFit bridges now raise, before ``fitter.init()``, on arguments anneal
0.10.0 coerced or ignored without a word. A fractional ``budget`` or ``seed``
raises instead of passing through ``int()`` (a budget of 60.5 ran 60
evaluations), and a negative seed, or one of ``2**64`` or more, raises
``ValueError`` where it raised ``OverflowError`` after ``init``. A string or
bool ``budget``, ``seed``, ``steps_per_epoch``, ``tell_every``, ``bound_span``
or ``default_span`` raises ``TypeError``, where 0.10.0 passed it through
``int()`` or ``float()``, reading ``True`` as 1, or did not read it.
``tell_every`` and ``steps_per_epoch`` must be positive whole numbers under
every method: where the method uses one, 0 ran as 1 and 2.5 as 2, and where it
does not (``tell_every`` under the classical methods, ``steps_per_epoch``
under the portfolio), any value ran. ``bound_span`` must be positive and
finite beside ``low`` and ``high`` too, and ``default_span`` when the fitter
bounds every parameter, where 0.10.0 did not read them. ``fit_chemfit`` raises
``TypeError`` on a keyword no preset takes, such as ``t_inti`` or
``tell_evry``, which it ignored, and on a preset keyword of its method that is
not a number, such as ``"2.0"`` or ``True``, which it passed through
``float()``. A ``preset_kwargs`` that is not a dict, and a ``preset`` that is
not ``Boltzmann()``, ``Fast()`` or ``Gsa()``, raise ``TypeError`` under the
portfolio too. ``run_benchmark`` raises ``ValueError`` on an unknown method,
where it ran the preset it was given or raised ``KeyError`` after ``init``. A
bounds entry that is not one ``(lower, upper)`` pair, such as a 3-tuple or a
list of three per-element pairs, or that holds a non-numeric string, raises in
``fit_chemfit`` and ``chemfit_box``, which ignored it in favour of
``default_span``; a two-item list is that pair, each side a scalar or an array
of the leaf's shape. A parameter that is not finite raises in
``ChemFitVector`` and in ``fit_chemfit``, whose portfolio ignored it, and a
complex leaf raises instead of losing its imaginary part. ``resolve_bounds``
and ``bounds_from_fitter`` raise on an infinite or NaN bound, a box too wide
for a float, or a lower bound above the upper, and ``chemfit_box`` on an
infinite bound or a box too wide for a float, where each returned the box.
``fit_chemfit`` and ``run_fitter`` now read method names in any case, as
``fit_anneal`` and ``run_benchmark`` did.
