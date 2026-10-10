The ChemFit bridges evaluate the start first, bit for bit, count it against
the budget, and never exceed the budget. Every parameter leaf comes back in
its own type: a Python number, a ``Decimal`` included, as a ``float``, a NumPy
scalar or array with its shape and floating dtype (integer, bool and object
leaves as float64), and a list or tuple as a list or tuple nested the same
way. The bounds of a ``float32`` or ``float16`` leaf are rounded inward so the
cast candidate stays inside them, and a floating leaf wider than float64, such
as an x86 ``longdouble``, raises ``TypeError`` naming it. Bounds may be NumPy
pairs or per-element arrays, and a bound that is not finite, is inverted, or
is too wide for a float raises ``ValueError`` naming the parameter element. A
parameter whose two bounds are equal is held fixed, and every argument is
checked before ``fitter.init()``.
