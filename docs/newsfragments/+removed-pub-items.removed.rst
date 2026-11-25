Breaking removal of items that v0.12.0 still defines.
``anneal_core::soap::fivefold_plateau``,
``anneal_core::soap::extended_scale`` and
``anneal_core::soap::motif_fractions`` are gone. Delete those calls.
Fraction triples that remain go through
``anneal_core::structure::named_triplet_fractions`` with the cutoff
the caller chooses. There is no replacement for the plateau test or
the extended-scale test.
The private ``packing_bank`` module is gone. That removes
``PackingBank``, ``PackingBank::new``, ``PackingBank::order``,
``PackingBank::offer``, ``PackingBank::better_other`` and the
constant ``GAP``.
These ``pub(crate)`` helpers in
``anneal_core::methods::cluster_hopping::binary_geometry`` are gone:
``FccCentre``, ``fcc_cut``, ``IcosahedronJoin``,
``polyicosahedral_cut``, ``mackay_cut``, ``mackay_cap``,
``axial_spine_cut`` and ``quench_as_placed``.
