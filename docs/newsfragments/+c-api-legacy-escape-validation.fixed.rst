``anneal_box_config_t.escape`` now uses ``int32_t`` semantics at the C ABI
boundary and is validated before callbacks run. Unknown escape values return
``ANNEAL_INVALID_PARAMETER`` with an explanatory ``anneal_last_error``
message instead of relying on enum layout assumptions.
