``ttt_lj38`` and the ``serial`` and ``indep`` arms of
``lj_lattice_ensemble`` run ``cluster_search``, so they walk the preset
``Config::recommended`` names, surfaces included. ``lj_lattice_ensemble``
no longer replays ``lj_ensemble_splice`` chains bit for bit, and those arms
print different energies and first hits.
