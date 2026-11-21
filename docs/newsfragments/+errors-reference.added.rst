The Errors section of the Python API page now covers ``MemoryError`` (the builtin, which is not an ``anneal.AnnealError``), says that a state length past a machine word is an ``InvalidArgumentError``, and has a table of the errors that each call raises, with an example of each.
It also says that ``anneal._core.Bounds`` called directly is not checked, and what that does with ``eindir-core`` 0.6.0.
``pytest/test_errors_page.py`` fails when the page lists a class the module lacks or omits one it has, gives a class the wrong parent, or gives a call errors that its stub docstring does not name.
