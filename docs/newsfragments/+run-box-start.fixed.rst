The Boltzmann, Fast and GSA presets mirror-reflect every proposal into the
objective's box, so ``run`` and ``run_qmc`` only evaluate points inside
``[low, high]``. Equal bounds pin a coordinate, and an empty, non-finite or
inverted box raises ``ValueError``. Both drivers accept an optional start point
``x0``, which is the first point evaluated.
