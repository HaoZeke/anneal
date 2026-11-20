Added ``Config::call_floor`` and ``Config::call_floor_elements`` (along with
``CallFloor``, ``clears_call_floor``, ``element_call_floor``, and
``open_call_floor``) so trials inside an objective's minimum pair distance are
opened before charging and zero-step reads or invalid floors are rejected
without charge.
