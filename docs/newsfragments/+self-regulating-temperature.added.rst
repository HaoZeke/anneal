``Config::self_regulating_temperature`` moves each chain's temperature by GOAT's factor of 1.05 per outcome (de Souza 2025).
The temperature never falls below the chain's base.
An outcome is a quenched trial that reached another basin, so returns and screened trials leave the temperature alone.
The flag needs the Metropolis rule and does nothing under minima hopping.
``Outcome::temperature_factors`` reports each rung's factor at the end of a run.
``lj_cluster_search`` and ``water_tip4p`` take ``selft``, and every chain prints a ``goat:`` line with its factors.
``goat::SelfRegulatingTemperature`` applies the rule outside the hop loop.
The flag is off by default, so seeded searches follow the same trajectories as before.
