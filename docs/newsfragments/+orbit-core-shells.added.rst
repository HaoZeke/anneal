``Config::orbit_core`` chooses the core that orbit completion reads its point group from.
``OrbitCore::Fraction``, the default, is the innermost ``Config::symmetrise_core_fraction`` of the atoms about the centroid, as before.
``OrbitCore::Shells(ShellCore)`` grows a core shell by shell until the detector finds an operation on it.
The shells are about a centroid weighted by ``exp(-decay r)``, and the core has at most ``max_points`` atoms.
It refines each operation to the orthogonal fit of the core onto its images and closes the set with products equal within ``closure``.
Completion then acts about that core's centre.
``OrbitCore::Both`` relaxes the shell candidate and the fraction candidate and offers the lower.
``ShellCore::scaled`` takes the ``SYMMETRY`` settings of the global optimisation program (GMIN) at 75 atoms.
Core symmetrisation and the continuous-symmetry reseed keep the fraction core.
``graph_split`` takes ``GRAPH_ORBIT_CORE=fraction|shells|both``.
