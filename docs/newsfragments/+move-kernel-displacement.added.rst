``MoveKernel::displacement`` returns the step ``d`` of a kernel whose proposal
is ``i + d``, drawn with the same calls on the random number generator as
``propose``; the default, ``None``, draws nothing. ``Gaussian`` and ``Cauchy``
implement it. Where the sum ``i + d`` overflows ``f64``, so that the proposal
alone reaches the reflection as ``+-inf``, ``Reflected`` uses the step to
mirror the sum into the box, with an infinite wall standing at ``+-MAX``. A
symmetric kernel of one's own gets the same fold by implementing
``displacement``.
