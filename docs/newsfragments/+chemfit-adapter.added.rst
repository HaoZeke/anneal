New ``anneal.chemfit.fit_anneal`` drives the gradient-free optimizers from a
ChemFit fitter through ``init`` / ``ask`` / ``tell`` / ``finish``. The default
driver is the Thompson-allocated portfolio. ``boltzmann``, ``fast``, and
``gsa`` are the single-chain ablations. Nested parameter dictionaries are
flattened to vectors and rebuilt on return. The fitter's initial parameters
are the start unless ``x0`` is passed, and every evaluation stays in the box.
