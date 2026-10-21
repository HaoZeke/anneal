The JCC LJ causal pair runner runs under bash 3.2, the bash GitHub puts
on macOS images. Protocol values are exported without ``declare -g``,
replica indexes are walked without ``seq``, the dynamic-loader check is
skipped where ``ldd`` is absent, and run artifacts are sealed without
``mapfile`` or ``sort -z``.
