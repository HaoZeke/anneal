The GFN2 example objective published the xtb gradient as the rgpot force.
xtb reports the energy slope and rgpot negates the force to build the objective gradient, so every descent step walked uphill.
``xtb_gradient::store_as_force`` converts the buffer, and ``xtb_gradient::objective_component`` gives the component rgpot then sees.
