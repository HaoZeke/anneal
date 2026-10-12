The docstrings of ``anneal.chemfit.fit_anneal`` and ``fit_chemfit``, and of
the ChemFit example's ``run_anneal``, named arms that the default portfolio
does not run as the bridges call it, without a gradient, such as parallel
tempering, and left out arms that it does run then, such as CMA-ES and the
quasi-Newton descent. They now name no arm and point to
``anneal.global_optimize``, whose documentation names the arms of the
values-only loop that the bridges run.
