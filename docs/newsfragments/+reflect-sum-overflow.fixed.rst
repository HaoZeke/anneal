The ``Boltzmann`` and ``Fast`` presets mirror a step whose sum ``x + step``
overflows, in a box with a wall near ``+-MAX`` such as ``[-1.7e308, -1e308]``,
where the step used to stop on a wall. A Cauchy step that is itself infinite
leaves that coordinate where it was.
