``Config::plateau_heat`` doubles the Metropolis temperature and the
lattice-exchange temperature on each escape plateau, up to four times
the preset, and returns to the preset when a new lowest minimum is
recorded. The proposal step stays the configured length. Off leaves the
preset trajectory. ``h2h_gmin_proof``, ``h2h_morse`` and ``h2h_metal``
read ``H2H_PLATEAU_HEAT=1``.
