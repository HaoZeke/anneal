Classical ``run`` and ``run_qmc`` reflect every trial into ``[low, high]``
and accept a starting point ``x0``. ``anneal.chemfit.run_benchmark``
drives a ChemFit session from ``initial_params`` inside that box.
