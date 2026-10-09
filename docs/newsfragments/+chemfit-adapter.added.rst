New ``anneal.chemfit`` module with ``fit_anneal``: drives the
gradient-free optimizers (the Thompson-allocated ``portfolio`` SOTA driver
by default, or single-chain ``boltzmann`` / ``fast`` / ``gsa`` ablations)
from a ChemFit ``Fitter`` through its user-driven ``init`` / ``ask`` /
``tell`` / ``finish`` protocol. Nested parameter dicts (scalars and arrays
such as per-atom positions) are flattened to vectors and rebuilt on return;
bounds come from the fitter's ``bounds`` dict or explicit ``low`` / ``high``
vectors, and the fitter's ``initial_params`` seed the start by default.
Includes ``examples/chemfit_positions.py`` with the corrected
review-response driver.
