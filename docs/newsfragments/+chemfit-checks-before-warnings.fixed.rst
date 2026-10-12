The ChemFit bridges gave their ``FutureWarning`` for a call anneal 0.10.0
took, such as a preset under the portfolio or a fitter without ``step``,
before checking the rest of the call. A call refused for another argument,
such as an ``x0`` of the wrong size, empty bounds or a budget of 0, so
warned first about a fit that never ran, and under
``warnings.simplefilter("error")`` raised the warning in place of the error.
``fit_anneal``, ``fit_chemfit``, ``run_benchmark`` and ``run_fitter`` now
check every argument they refuse before giving any such warning: a refused
call raises its error alone, and a call that runs warns as before.
