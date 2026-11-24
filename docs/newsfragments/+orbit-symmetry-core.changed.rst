``orbit_complete_core`` now ranks higher rotational orders with a
cheap outer-first sample pass and verifies only top candidates with a
relaxed majority-exchange threshold. Every distinct candidate direction
is scored, and the lowest residual of a near-parallel set is kept.
``csm::propose`` scores the ``C_i``, ``C_5`` and ``D_5h`` library and the
group closed from detected operations by one measure, and the detected
group replaces the library choice only when its measure is lower.
