- Fixed building the Python extension when `with_python` is enabled: the
  `has_analytic_hvp()` query no longer calls a non-const thread-slot accessor
  on a const handle (the same `primary().caps()` path as `caps()`).