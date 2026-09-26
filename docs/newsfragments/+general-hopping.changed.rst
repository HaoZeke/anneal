Cluster proposal libraries implement the general ``MoveKernel`` interface and
``Config::proposal_kernel`` plugs them into the same ``HoppingSampler`` used by
ordinary bounded objectives. Length- and energy-bearing preset values,
including restart and rigid-group repacking geometry, derive from declared
``length_scale`` and ``energy_scale`` values rather than Lennard-Jones units.
