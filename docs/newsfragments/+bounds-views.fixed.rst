``Bounds`` takes ``low`` and ``high``, and its ``contains`` and ``clip``
take their point, as any array-like of one size, strided views such as the
columns of a ``(dim, 2)`` array and transposes included, flattened in C order
into float64 copies. A non-contiguous array had raised ``TypeError``.
``Bounds`` now also shows ``low`` and ``high``, has a ``repr`` that reads
back, pickles and copies, and raises ``ValueError`` for a malformed box or a
point of the wrong size for ``clip``, where the extension had panicked. The
box ``PyObjective.bounds`` returns is still the extension's own, which counts
as a ``Bounds`` but reads only contiguous points.
