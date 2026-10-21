Binary Lennard-Jones and rigid TIP4P on the stock cluster presets.
``cluster_search`` takes ``species``, ``sigmas``, ``rigid``, ``layout``,
and ``target``. Two or more species labels add a swap move. A wide
diameter ratio leaves the occupied basin by one softened microcanonical
step and one inner-outer exchange, then quenches that point in full.
A rigid chart orients every body and keeps the centres fixed. The
acceptance temperature of a rigid chart is scaled inside the hop. A
chart with more than ten bodies holds that factor at 4 and scales the
orientation amplitude toward an acceptance ratio of one half. A chart
of ten bodies or fewer is unchanged. Preset temperature, screen length,
relax length, and record gradient stay on the preset. No parameter was
added, so the C header is unchanged.
