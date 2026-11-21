The C ABI reaches every setting of the box ensemble. ``anneal_box_options_t``
adds the history, the membership rule, the identity tolerance, the shared
deposits, the visit policy, the Langevin friction and the coverage sharing,
peer weight and neighbourhood to what ``anneal_box_config_t`` has.
``anneal_box_options_default`` fills it with the Rust defaults and
``anneal_box_minimize_ex`` runs the ensemble and fills an optional
``anneal_box_stats_t`` with the history and coverage counters. Both structures
start with ``struct_size``, so fields can be appended later. An unknown enum
value or an out-of-range setting returns ``ANNEAL_INVALID_PARAMETER`` with the
reason in ``anneal_last_error`` before any callback runs.
``anneal_box_report_t`` is as it was. Two things differ for a caller of
``anneal_box_config_t`` and ``anneal_box_minimize``. The field ``escape`` of
``anneal_box_config_t`` is now an ``int32_t`` holding an
``anneal_box_escape_t`` value, so an unknown value is refused with
``ANNEAL_INVALID_PARAMETER`` instead of being read as an enum. It has the same
size as the enum on a target that gives enums four bytes, and the
``ANNEAL_BOX_ESCAPE_*`` constants still compile in an assignment or a
comparison. And ``anneal_box_config_default`` now takes its budget from the
Rust default, 8000 work units instead of 400. A caller that keeps the default
now runs twenty times the budget. A caller that sets ``budget`` itself gets
the budget it set. See the ``c-api-default-box-budget-parity`` and
``c-api-legacy-escape-validation`` entries. ``tests/ffi_smoke.c`` now
exercises the box ensemble through the generated header, and
``scripts/ffi_smoke.sh`` builds and runs it from a checkout.
