``Ledger`` exposes the accounting of the Rust ledger: ``best_state``,
``encounters``, ``target``, ``target_reached``, ``below_target``,
``stop_below``, ``charge``, ``charge_many``, ``charge_frac``, ``record``,
``replace_best``, ``record_quench_boundary``, ``quench_boundaries``, ``grant``
and ``slice``. ``stop_below`` and
``replace_best`` raise ``InvalidArgumentError`` for a value that Rust
would ignore silently. The ``budget`` docstring said the creation value
and now says it includes what ``grant`` added. ``pytest/test_ledger.py``
fails when a public method of the Rust ledger is neither exposed nor named
as belonging to a search driver.
