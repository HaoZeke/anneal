``Config::entropy_stop`` ends a run with ``HopStop::EntropyConverged`` after at least three rounds, GOAT's three global cycles (de Souza 2025).
It stops at the end of a round that finds no lower minimum and moves the conformer entropy by less than 0.0503 k_B.
The tolerance is ORCA's ``MINDELS`` default of 0.1 cal/(mol K) for GOAT-ENTROPY, stated in units of k_B.
``Config::entropy_round`` sets a round to one GOAT global cycle, four workers each with the optimisations of the paper's eq. (4).
The stop turns on ``Config::ensemble_report``.
``lj_cluster_search`` and ``water_tip4p`` take ``estop``.
``goat::EntropyStop`` applies the stop outside the hop loop.
The flag is off by default, so seeded searches follow the same trajectories as before.
