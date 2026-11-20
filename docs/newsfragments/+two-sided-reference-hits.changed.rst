Reference-hit scoring and audits now use ``energy_matches_reference``, a
two-sided finite check within ``tol`` of the declared reference energy,
so non-finite or off-reference energies are never reported as hits.
