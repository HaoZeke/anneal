``potentials::BinaryLennardJones`` is a two-species Lennard-Jones cluster with no cutoff.
Its first ``n_a`` atoms are species A and the rest are species B, with an energy and a diameter for each of the AA, AB and BB pairs.
It evaluates through the ``binary_lj`` kernel of rgpot when the process has installed the kernels, and through a local pair loop otherwise.
``ClusterKernels`` has a ``binary_lj`` entry for that kernel, and ``install_rgpot_kernels!`` fills it.
