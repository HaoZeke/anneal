``orbit_complete_core`` now ranks higher rotational orders with a
cheap outer-first sample pass and verifies only top candidates with a
relaxed majority-exchange threshold, while ``csm::propose`` skips
redundant inertia orientations on already-framed detected operations.
