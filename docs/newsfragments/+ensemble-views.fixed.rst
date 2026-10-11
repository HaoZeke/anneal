``ensemble_optimize`` and ``box_ensemble_optimize`` take ``low``, ``high``
and ``x0`` as any array-like of one shape, strided views such as the columns
of a ``(dim, 2)`` array included, flattened in C order as ``global_optimize``
takes them, and ``cluster_search`` takes such a ``start``. A non-contiguous
array had raised ``TypeError``.
