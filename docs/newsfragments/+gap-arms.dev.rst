``organic_bench`` takes the arms ``loose``, ``chains``, ``tkick``, ``combo``
and ``chains4``, and prints ``REJECTED_TRIALS``, ``REFINES``, ``ROTORS`` and
``CHAINS``. ``water_hexamer`` applies the same arms under ``WATER_GAP_ARM``
and then prints the refine and rejected-trial counts per seed; unset, the
board is unchanged. ``experiments/gap_rerun.py`` runs and scores them.
