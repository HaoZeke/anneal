The C ABI minor of `rgpot-core` is now 10, in `RGPOT_ABI_VERSION_MINOR`, for the
additive `rgpot_rpc_client_select_orbitals` RPC export. A caller built against
minor 0 to 9 still matches; a caller built against minor 10 is refused by an
older library.
