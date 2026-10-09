On a Rust box whose width ``high - low`` overflows, such as
``[-1e308, 1e308]``, the single-chain runners draw the start at half scale
instead of panicking, and the QMC runners scale their starts at half size
instead of putting every chain but ``x0`` on ``high``. Python refuses such
bounds and is unaffected.
