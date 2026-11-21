Added opt-in stall recovery controls on ``Config``: ``branch_on_stall``,
``plateau_heat`` (doubling Metropolis and lattice-exchange temperatures up to
4x on escape plateaus), ``retry_noop_stall_leave``, and
``align_directed_stall_source``; all default off so preset digests stay
unchanged.
