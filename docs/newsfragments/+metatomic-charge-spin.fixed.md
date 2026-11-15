`rgpot_engine_set_charge_spin` on the Metatomic engine reaches the model: each system carries the `rgpot::charge`, `rgpot::spin_multiplicity` and `rgpot::spin` data the model declares in `requested_inputs`, and the engine reads `charge`, `spin` and `attachSystemExtras` from `MetatomicParams`.
It used to return 0 and drop the values.
For a value the model does not read, other than charge 0 or multiplicity 1, `rgpot_engine_set_charge_spin` now returns 3 and `rgpot_engine_create` fails.
