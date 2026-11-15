`XTBPot` and `XTBDlopen` no longer start an SCC from the iterate of one
that failed, and a session's GFN2 energy on hot Si clusters matches a
fresh start and standalone xtb. The new `XTBConfig::warm_min_gap` (1 eV)
uses xtb's own guess when the HOMO-LUMO gap is too small for a unique
SCC solution. The engine exports `rgpot_xtb_set_warm_min_gap` and
`rgpot_xtb_warm_min_gap`. `TBLitePot` drops the wavefunction of a failed
SCF.
