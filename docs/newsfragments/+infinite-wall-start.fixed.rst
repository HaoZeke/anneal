Without ``x0``, ``run_rs_variant`` and the other single-chain Rust runners
start an axis with an infinite wall on its finite wall, or at 0 when both
walls are infinite, where the uniform start draw used to panic. Python's
``run`` refuses such bounds and is unaffected.
