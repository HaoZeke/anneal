The C ABI minor of `rgpot-core` is now 6, in `RGPOT_ABI_VERSION_MINOR`, for the
additive `rgpot_potential_new_eam_table` constructor and the four
`RGPOT_EAM_TABLE_*` format codes. A caller built against minor 0 to 5 still
matches; a caller built against minor 6 is refused by an older library.
