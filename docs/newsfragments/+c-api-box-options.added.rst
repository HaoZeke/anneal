The C ABI reaches every setting of the box ensemble. ``anneal_box_options_t``
adds the history, the membership rule, the identity tolerance, the shared
deposits, the visit policy, the Langevin friction and the coverage sharing,
peer weight and neighbourhood to what ``anneal_box_config_t`` has.
``anneal_box_options_default`` fills it with the Rust defaults and
``anneal_box_minimize_ex`` runs the ensemble and fills an optional
``anneal_box_stats_t`` with the history and coverage counters. Both
structures start with ``struct_size``, so fields can be appended later. An
unknown enum value or an out-of-range setting returns
``ANNEAL_INVALID_PARAMETER`` with the reason in ``anneal_last_error`` before
any callback runs. The change is additive: ``anneal_box_config_t``,
``anneal_box_report_t`` and ``anneal_box_minimize`` are as they were, and
give the same result as before. ``tests/ffi_smoke.c`` now exercises the box
ensemble through the generated header, and ``scripts/ffi_smoke.sh`` builds
and runs it from a checkout.
