Mirror reflection folds a step past ``high`` back off ``high`` in a box that
lies far from zero, such as ``[-0.6 * MAX, -0.1 * MAX]``. Where ``x - low``
overflowed, the step landed on ``low``, the far wall, and so did a proposal
that overflowed to ``+inf``; that one now stops on ``high``.
