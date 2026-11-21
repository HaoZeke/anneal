A regression test now guards the cluster-hopping non-finite proposal path:
non-finite trial coordinates are refused before relaxation, so the objective
callback never receives invalid coordinates.
