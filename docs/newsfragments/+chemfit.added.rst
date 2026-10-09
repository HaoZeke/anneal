``anneal.chemfit.fit`` fits a ChemFit ``Fitter`` with the portfolio,
Boltzmann, Fast, GSA, QMC or DMC driver, starting at its initial parameters,
evaluating only points inside its bounds and stopping at a hard evaluation
budget. Parameter trees may hold scalars and arrays of any shape, such as
``(n, 3)`` positions, and ``anneal.chemfit.objective`` exposes the flat view
that the drivers search. ChemFit remains an optional dependency.
