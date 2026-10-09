A driver whose objective was called and never returned a usable value, or
whose only callback, the gradient, never returned one, re-raises the first
exception of that callback when it returns instead of reporting it as a
``RuntimeWarning``: nothing it would have returned was measured. A gradient
that returns does not stand in for an objective that never does. A ChemFit
fitter driven through the wrong lifecycle, an objective with a typo, or a
missing import now raise when the driver returns, after it has spent its
budget on the failing calls. A run in which some calls of each callback
return keeps the warning, so budget counters can still stop a driver by
raising.
