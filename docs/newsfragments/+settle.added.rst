``cluster_search::settle`` descends a minimum past ``Config::record_gradient`` on the caller's ledger.
It returns the lowest evaluated point whose gradient passes ``Config::record_gradient``.
A caller that reports energies can use calls kept back from the search to settle the best minimum.
