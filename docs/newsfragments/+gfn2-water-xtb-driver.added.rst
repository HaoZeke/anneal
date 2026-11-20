``gfn2_water_crest_driver`` serves the ``gfn2_water_h2h`` objective,
including the O-H wall, to ``xtb --driver``. It counts every call, stops
past the cap or at the first call on the target, and writes
``driver.txt`` and ``best.xyz`` on stop.
