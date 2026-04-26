"""Publication plots for the IISE manuscript benchmarks.

Imports here force the chemparseplot ruhi theme to register its colormaps
on first use, so callers can rely on `cmap="ruhi_diverging"` without an
explicit setup call.
"""

from chemparseplot.plot.theme import (  # noqa: F401  (registers ruhi cmap on import)
    RUHI_COLORS,
    get_theme,
    setup_publication_theme,
)
