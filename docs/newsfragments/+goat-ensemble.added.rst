``Config::ensemble_report`` reports the Boltzmann conformer ensemble of the unique minima a run found, as GOAT does (de Souza 2025).
``Outcome::conformers`` gives the members within 10.13 k_B T of the lowest, their entropy in units of k_B, and their free energy.
``Config::conformer_temperature`` gives k_B T at GOAT's 298.15 K in the run's unit, and every degeneracy is one.
The window is ORCA's ``MAXEN`` default of 6 kcal/mol for GOAT, stated in k_B T at 298.15 K.
``lj_cluster_search`` and ``water_tip4p`` take ``ensemble``.
``goat::ensemble_entropy`` gives the member count, the entropy and the free energy of a list of energies.
The flag is off by default, so seeded searches follow the same trajectories as before.
