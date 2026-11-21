No ``PanicException`` is reachable from the Python bindings through a bad
argument, and the C ABI refuses what it read as undefined.
``anneal.Bounds`` and ``PyObjective`` are anneal's own classes: a bad box,
a NaN or negative slack, a ``clip`` of the wrong length, or a callback
that raises gives a typed error, where the eindir classes panicked. Every
box goes through one rule and raises ``BoundsError``: equal lengths, at
least one axis, finite sides with a finite width and sum of widths, and
order. The device drivers raise the same error as the host ones. A start
that is not finite raises ``InvalidArgumentError``. A scalar guard tests
finiteness first, so a NaN or infinite ``step0``, ``grad_tol``, ``t_init``,
``epsilon``, ``q``, GLE ``dt``, ``local_frac`` and the rest raise
``InvalidArgumentError``, and a count too large to allocate raises
``MemoryError``. ``Config`` validates at construction and in ``replace``:
``delayed_acceptance`` with more than one replica, and a molecular group
that is empty, out of range or not strictly increasing, raise
``InvalidArgumentError``, and the Rust search returns
``SearchError::InvalidConfig`` for the same settings.
``anneal_box_options_default`` writes only the caller's ``struct_size``
bytes and refuses a size below the oldest layout, so callers set
``struct_size`` first. The C box search refuses an overflowing width, a
start that is not finite, a negative DLPack shape, misaligned data and an
``escape`` outside ``anneal_box_escape_t``.
