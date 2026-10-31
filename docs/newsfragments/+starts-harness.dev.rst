``organic_bench`` prints ``separate_start_molecules`` and
``START_SEPARATION none|unresolved|k K s S``. Under ``WATER_GAP_ARM``,
``water_hexamer`` prints ``start_separation`` and the charged calls at the
first record per seed, and with ``WATER_GEOMETRY_DIR`` set it writes a
separated start as ``separated_s{seed}.xyz``. ``experiments/starts_rerun.py``
runs the campaign's jobs, reproduces the separation in Python and applies the
decision rule; ``experiments/starts_rerun.sh`` runs the jobs.
