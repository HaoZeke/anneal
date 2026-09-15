Python ``ensemble_optimize`` is the box search entry: with a
gradient it hops and quenches; without one it is the values-only
portfolio. Each Python eval and grad callback spends one ledger unit.
The design vector is never treated as a point set.
