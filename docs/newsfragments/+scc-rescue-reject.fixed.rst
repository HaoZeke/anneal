A GFN2 call that returns success with ``rgpot_xtb_last_eval_flags`` bit 0
set (``RGPOT_XTB_EVAL_SCC_RESCUED``) is an unconverged SCC. The worker
rejects it, so the energy is not finite and that geometry cannot be
stored as a target, a best, or an answer. A dropped status leaves a zero
force buffer; a non-finite energy is not accepted as that force.
rgpot-core stays crates.io 3.5.1. The flag word is read with ``dlsym``
when the loaded engine exports it. That symbol is the rgpot 3.8.0
engine ABI (``ac6f6f5f``), which is not on crates.io. An engine without
the symbol is unchanged.
