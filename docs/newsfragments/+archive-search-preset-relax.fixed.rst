The Python ``cluster_archive_search`` relaxes with
``cluster_search::warm_relax``, the closure ``cluster_search::search``
uses. Its quench now reads the preset's surfaces, stops at
``record_gradient``, and screens with the predictor. It ran a bare L-BFGS
to a 1e-6 gradient on the plain surface before. ``warm_relax`` is public,
so a driver that calls the chain itself can take the same relaxation.
