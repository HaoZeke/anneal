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
``evaluate``, ``step`` and an ``init`` that takes ``batch_size``. When every
other check passes, a batch under another driver is refused before ``init``,
by a message naming the driver, and under the portfolio a fitter without
``evaluate``, ``step`` or ``init``, or whose ``init`` has a signature that can
be read and cannot take ``batch_size``, is refused before ``init``, by a
message naming what it lacks. Any other ``init``, such as one whose signature
cannot be read or one behind a wrapper of ``(*args, **kwargs)`` written
without ``functools.wraps``, is not named there and meets ``batch_size`` only
at its call, where what it raises ends the fit without ``finish``.
No ChemFit release has all three yet: ChemFit 3.1.1 has ``ask`` and ``tell``,
ChemFit's source has ``evaluate`` and ``step`` from commit c57d6f5, and it
takes batches from commit 3b65fe0 on. Whether ``evaluate`` takes a list
shows only when it is called: one that takes a single candidate gets the
start as a list of one, after ``init``, and what it raises, or a TypeError
when its reply is not one real loss per candidate, ends the fit without
``finish``. The default of one keeps the single-candidate path.
