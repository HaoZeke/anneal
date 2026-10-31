``Config::for_cluster`` sets ``Config::orbit_core`` to ``OrbitCore::Both(ShellCore::scaled(length_scale))``.
Each orbit completion then relaxes a candidate from the shell core and one from the fraction core and offers the lower.
The setting carries over to every caller built on ``for_cluster``, such as ``packing_superbasin``.
``recommended``, ``communicating`` and ``derived`` keep the fraction core.
``for_cluster`` is not the base of the presets for the four-point transferable intermolecular potential (TIP4P) water model.
