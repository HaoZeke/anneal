``Config::branch_on_stall`` runs budget-aware fresh-start side chains
after a measured stall. Off preserves the original walk. Side-chain work
is bounded by observed walk progress, the record rate, and a
five-percent share of walk calls.
