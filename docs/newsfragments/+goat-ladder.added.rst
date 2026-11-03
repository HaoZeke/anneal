``Config::with_goat_ladder`` sets GOAT's four rungs, which never exchange states (de Souza 2025).
Each rung starts at the temperature where its rise of 0.5, 1, 2 or 4 kcal/mol is accepted half the time.
``Config::kcal_per_mol`` gives a kcal/mol in the run's ``Config::energy_unit``, and ``Config::for_tip4p`` declares kJ/mol.
A run in reduced units reads a kcal/mol as one ``Config::energy_scale``.
The ladder turns minima hopping off and the self-regulating temperature on.
``EnsembleConfig::validate`` refuses a ladder, so the ladder cannot run on communicating chains over a shared history.
``lj_cluster_search`` and ``water_tip4p`` take ``goat`` and run the ladder as one process.
The new ``goat`` module collects GOAT's rules, and ``goat::ladder_temperatures`` gives the rung temperatures.
