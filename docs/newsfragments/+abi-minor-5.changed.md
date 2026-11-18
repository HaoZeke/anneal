The C ABI minor of `rgpot-core` is now 5, in `RGPOT_ABI_VERSION_MINOR`, for
the additive engine and RPC Hessian-vector exports. A caller built against
minor 0, 1, 2, 3 or 4 still matches; a caller built against minor 5 is refused
by an older library.
