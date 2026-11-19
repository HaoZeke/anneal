``Config::for_cluster_with_scales(n, length, energy)`` is ``Config::for_cluster`` at caller-supplied physical scales.
It keeps the whole cluster preset, including the scaled shell core of ``orbit_core``, and changes only the length and energy.
``for_cluster_with_scales(n, 1.0, 1.0)`` resolves to the same configuration as ``for_cluster(n)``.
Use it for a potential that is not Lennard-Jones, such as the Gupta potential.
