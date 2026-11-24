The GFN2 examples built on ``examples/common/rgpot_eindir.rs`` descend
on the energy again. ``rgpot_xtb_force`` already writes the force,
-dE/dx in eV/Å, and the callback negated it a second time before rgpot
negated it to make the objective gradient, so every quench climbed.
The callback now publishes the engine force unchanged. The unreleased
``anneal_core::xtb_gradient`` module, which held that extra negation,
is removed. ``tests/rgpot_xtb_gradient.rs`` checks the objective
gradient against a central difference of the engine energy.
