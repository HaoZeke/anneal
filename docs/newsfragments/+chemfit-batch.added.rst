``fit_anneal``, ``fit_chemfit`` and ``run_benchmark`` take ``batch_size``.
Above one, the portfolio runs that many batch replicas of
``anneal.minimize`` sharing one archive and incumbent: the fitter is
initialised with ``init(batch_size=batch_size)``, and each batch of
candidates, at most ``batch_size`` of them, goes to one ``Fitter.evaluate``
of a list and one ``step``, so a fitter whose schedule runs candidates
concurrently evaluates them together. The start is the first evaluation, as
a batch of one, and the budget stays the total number of evaluations.
``fit_chemfit`` then steps the fitter once per batch instead of every
``tell_every`` evaluations. A batch needs the portfolio and a fitter with
``evaluate`` and ``step``; anything else is refused before ``init``. The
default of one keeps the single-candidate path.
