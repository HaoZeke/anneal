The energy_lattice option also enables the generic packing probe for
atomic, non-rigid, non-frozen searches. The probe is offered at hop 36 of
the walk by default, which is Config.packing_deferral (four) times nine,
and one hop below max_hops at the latest in a capped walk. A walk that hits
earlier does not pay for it. Every potential call of the probe is charged.
