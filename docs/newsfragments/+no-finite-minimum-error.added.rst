Added ``SearchError::NoFiniteMinimum``, returned by ``cluster_search``
drivers when a run spends its evaluations without recording a finite
minimum, and surfaced as an explicit failure by the benchmark drivers.
