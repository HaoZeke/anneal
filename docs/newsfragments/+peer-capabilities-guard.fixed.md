`rgpot.h` declares `rgpot_potential_set_peer_capabilities` only when
`RGPOT_HAS_SCHEMA` is defined, which the meson build does with RPC on. Before,
the header declared it in builds whose library did not define it, so a call
failed at link time.
