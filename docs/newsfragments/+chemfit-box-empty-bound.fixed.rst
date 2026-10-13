``anneal.chemfit.chemfit_box`` gave its ``FutureWarning`` for bounds read
from numeric strings before it refused a bound with no width, so such a call
warned about a box it never built, and under
``warnings.simplefilter("error")`` raised the warning in place of the
``ValueError`` naming the parameter. It now refuses that bound first, so the
refusal comes alone, and it warns of strings only when it returns the box.
