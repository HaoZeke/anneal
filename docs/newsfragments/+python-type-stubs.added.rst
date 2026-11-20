``anneal._core`` now ships type stubs (``_core.pyi``) and a ``py.typed``
marker, so a type checker sees the classes and function signatures of the
compiled module. ``pytest/test_stubs.py`` compares the stubs with the
module, and a new Python API reference page describes them.
