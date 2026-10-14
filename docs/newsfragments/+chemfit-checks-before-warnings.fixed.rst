The ChemFit bridges gave their ``FutureWarning`` for a call anneal 0.10.0
took, such as a preset under the portfolio or a fitter without ``step``,
before checking the rest of the call. A call refused for another argument,
such as an ``x0`` of the wrong size, empty bounds or a budget of 0, so
warned first about a fit that never ran, and under
``warnings.simplefilter("error")`` raised the warning in place of the error.
``fit_anneal``, ``fit_chemfit``, ``run_benchmark`` and ``run_fitter`` now
check every argument they refuse before giving any such warning, so a
refused call raises its error alone, and a call that runs warns as before
where it drives the fitter's session itself. Where ``run_fitter`` hands a
fitter to its own ``fit_anneal``, it refuses only what it cannot pass to
that method, and the budget, the seed, the
fitter's bounds and the rest are the method's to check; the warning of a
preset that ``run_fitter`` ignores waits for the calls the method makes to
``fit_anneal``, ``fit_chemfit`` and ``run_benchmark``, and comes once, from
the first call whose checks pass, before ``init``. So a refusal there, such
as ``run_benchmark``'s of a budget of 0 or a seed of -1, raises its error
alone too, and a call made after the refusal gets the warning once its
checks pass. A call the method makes on another thread with the fitter, or
with an object that holds the fitter or one of the fitter's bound methods,
such as a session that wraps the fitter, gives the warning on that thread,
before its ``init``, so under ``warnings.simplefilter("error")`` it is
raised there in place of the fit. The fitter or the method is found in the
object's slots, in the dictionary of its other attributes, whatever its
class makes of the name ``__dict__``, among its items, as a namedtuple holds
its fields, or in a dictionary held in any of these, at any depth, without
running any code of the object's classes, such as a property or a
``__getattr__``. A call on another thread given any other object, such as
one that keeps the fitter only in a closure, or in a list, a tuple or
another object that it holds, does not, and the warning then comes once the
method returns, after the fit, as it does when no call passes its checks; it
does not come when the method raises. In each case it points at the line
that called ``run_fitter``, in a module, a script or a Jupyter cell alike.
