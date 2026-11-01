``Config::orbit_memory`` lets a chain keep the minima its orbit completions start from and reach, and adopt a completion only downhill onto a minimum not kept before.
``OrbitMemory::Off``, the default, keeps nothing, and no preset changes.
``OrbitMemory::Energy`` tells minima apart by energy within 1e-6 of ``energy_scale`` and does not start a completion from a kept energy.
``OrbitMemory::Structure`` also compares the history bias's basin descriptor within ``merge_radius`` and quenches every completion.
``Outcome::orbits_skipped`` counts the completions not started.
The arm-3 head-to-head ran on Lennard-Jones (LJ) clusters against the Wales group's global optimisation program (GMIN).
Both memories hit more often than the default from LJ102 up and less often on LJ76, LJ77 and LJ98.
So neither is on by default.
