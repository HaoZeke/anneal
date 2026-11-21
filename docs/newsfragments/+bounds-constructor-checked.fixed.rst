``anneal.Bounds(low, high, slack)`` raises ``BoundsError`` for sides of different lengths, an inverted pair, a coordinate that is not a number and a negative or NaN ``slack``.
``eindir-core`` 0.6.0 asserts on the first, second and fourth of these, which reached Python as a ``PanicException`` that ``except Exception`` does not catch, and it accepted the third.
``anneal.Bounds`` is now a class whose call checks the arrays and returns the ``eindir-core`` box, so ``isinstance`` and ``PyObjective`` work as before.
The new ``anneal._core.make_bounds`` does the check.
