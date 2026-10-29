``Plan::start`` is ``StartRule::PresetSphere`` or ``StartRule::Density``.
The default is the preset sphere of ``Config::recommended``, radius
``0.9 * L0 * N^(1/3)`` with ``L0`` the preset length scale. A fresh
trial used to be ``random_cluster`` at density 0.7; that draw is
``StartRule::Density(0.7)``. The serial arm at seed ``s`` quenches to
the same first energy, bit for bit, as the recommended chain at that
seed. Changing the default does change existing output. The 13-atom
ensemble tests still pass and still finish on the icosahedron, but
the best-energy bits and the trace are not the density-0.7 values.
``lj_lattice_ensemble`` prints different energies unless ``DENSITY``
is set, which selects ``StartRule::Density``.
