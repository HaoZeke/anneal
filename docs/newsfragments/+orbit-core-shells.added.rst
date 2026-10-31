``Config::orbit_core`` chooses the core orbit completion reads its point group from.
The default, ``OrbitCore::Fraction``, is the innermost ``Config::symmetrise_core_fraction`` of the atoms about the centroid, as before.
``OrbitCore::Shells(ShellCore)`` builds the core one shell at a time about a centroid weighted by ``exp(-decay r)``.
The core grows until the detector finds an operation on it or until it reaches ``max_points`` atoms.
Each operation found is refined to the orthogonal fit of the core onto its images, and products equal within ``closure`` close the set.
Completion then acts about that core's centre.
``OrbitCore::Both`` relaxes the shell candidate and the fraction candidate and offers the lower.
``ShellCore::scaled`` takes the ``SYMMETRY`` settings of Wales's global minimisation program (GMIN) at 75 atoms.
Core symmetrisation and the continuous-symmetry reseed keep the fraction core.
``graph_split`` takes ``GRAPH_ORBIT_CORE=fraction|shells|both``.
