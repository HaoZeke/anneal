``repr(Config)`` guessed the preset from a flag, so a TIP4P configuration
printed as ``Config.recommended(n)`` or ``Config.for_cluster(n)``, and the
class docstring named three constructors of many. The ``repr`` now records
the call chain that built the ``Config``, and the docstring points to
``field_names``, ``get`` and ``replace``.
