Engine-specific ``CPMDC_CONFIG`` and ``NWCHEMC_CONFIG`` environment
variables now take precedence over the generic ``POTENTIAL_CONFIG``
fallback in ``examples/common/profile_engine.rs`` and
``examples/water_hexamer.rs``, preventing ambient ``POTENTIAL_CONFIG``
values from shadowing generated engine decks.
