When no objective call of a driver returns a number and at least one raised
an ordinary exception, the driver re-raises the first exception when it
returns instead of reporting it as a ``RuntimeWarning``: nothing it would
have returned was measured. A ChemFit fitter driven through the wrong
lifecycle, an objective with a typo, or a missing import now fail at once. A
run in which some calls return keeps the warning, so budget counters can
still stop a driver by raising.
