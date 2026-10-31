``Config::with_dual_orbit_core`` and the ``Config::for_cluster_dual_core`` preset complete each new basin on two cores and offer the lower quench (``OrbitCore::Both``).
One core is a shell core about a weighted centre, after the ``SYMMETRY`` routine of the Cambridge global minimisation program (GMIN), and the other is the fraction core.
They also restart a chain from a random cluster after 4000 calls without a new best.
Both are opt-in.
The fraction core is left out of the resolved JSON, so ``for_cluster``, ``recommended`` and every other preset resolve to the same configuration as before.
