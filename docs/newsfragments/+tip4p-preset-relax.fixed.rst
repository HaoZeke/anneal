``water_tip4p`` relaxes with ``cluster_search::warm_relax`` on
``Config::for_tip4p`` and folds the rotation vectors after it. The descent
stops at ``record_gradient`` and each full quench ends with the charged
convergence check, as in ``cluster_search::search_from``. It descended to
a 1e-6 gradient before, so its hops and first hits move.
