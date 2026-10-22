``cluster_search`` keeps both chart keyword sets on one call. ``species``,
``sigmas``, ``rigid``, ``layout`` and ``target`` select a stock preset and
then a binary or rigid chart. ``water_sites`` selects the TIP4P presets and
a caller body frame. The two sets exclude each other. A callback that raises
``KeyboardInterrupt`` still stops the run. The C header is unchanged.
