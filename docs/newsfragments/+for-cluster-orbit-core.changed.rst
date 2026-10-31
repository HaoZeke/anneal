``Config::for_cluster`` sets ``Config::orbit_core`` to ``OrbitCore::Both(ShellCore::scaled(length_scale))``.
Each orbit completion relaxes a candidate from the shell core and one from the fraction core and offers the lower.
``packing_superbasin`` and other callers built on ``for_cluster`` inherit it.
``recommended``, ``communicating`` and ``derived`` keep the fraction core, and the TIP4P presets do not build on ``for_cluster``.
