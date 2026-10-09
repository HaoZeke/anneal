A driver whose objective was called and never returned a number (NaN
counts as one), or whose only callback, the gradient, never returned a
usable gradient, re-raises the first
exception of that callback when it returns instead of reporting it as a
``RuntimeWarning``: nothing it would have returned was measured. A gradient
that returns does not stand in for an objective that never does. A ChemFit
fitter driven through the wrong lifecycle, an objective with a typo, or a
missing import now raise when the driver returns, after the calls it makes
before stopping. A run in which the objective returned at least once, or a
gradient-only run in which a gradient did, keeps the warning, so budget
counters can still stop a driver by raising.
