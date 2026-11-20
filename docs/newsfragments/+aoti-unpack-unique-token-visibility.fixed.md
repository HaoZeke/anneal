`aoti_unpack::detail::unique_token` now has hidden symbol visibility so `librgpot` does not export its function-local static symbols (`counter`, `seed` and its guard variable) in Metatomic builds.
