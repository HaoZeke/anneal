<!-- Copyright 2023--present rgpot developers -->

`rgpot.lammps.FixExternal` now imports `mpi4py` through `importlib.import_module` when LAMMPS runs on multiple ranks and verifies the broadcast error report type, so `mypy --strict` and `pyright` pass without `mpi4py` installed.
