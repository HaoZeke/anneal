``ase_panel`` runs one seed of ``Config::for_cluster`` or ``Config::recommended`` on a Lennard-Jones cluster of 38, 55, 98, 104 or 150 atoms or on a gold cluster of 55 atoms, with an optional wall cap.
``ase_nonlj`` does the same on the Cambridge Morse clusters of 38 and 55 atoms at ``rho = 6`` and ``rho = 14``.
Both print one line with the hit flag, the best energy, the charged calls and the wall time inside and outside the potential.
``h2h_morse`` refuses the removed ``PRIOR_*`` and ``H2H_PRIORS`` settings on its ``coords`` command as it does on ``run``.
