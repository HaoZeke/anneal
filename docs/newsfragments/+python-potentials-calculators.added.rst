``PairPotential``, ``BinaryLennardJones``, ``SuttonChen``, ``Gupta``,
``Tip4pCluster``, ``GminTip4p``, the TIP4P chart helpers
(``rigid_from_ohh``, ``energy_ohh``, ``block_from_interleaved_gmin``,
``interleaved_gmin_from_block``, ``gmin_tip4p_sites``), and
``anneal.ase.PotentialCalculator`` are exposed to Python. ``cluster_search``
and ``quench`` accept native cluster potentials directly without per-call
Python callbacks.
