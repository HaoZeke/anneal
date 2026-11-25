The generic packing probe relaxes its candidates as diagnostic quenches.
Through ``search_from`` it used the walk's own quench, whose surface
portfolio counted each probe screening pass as a hop of the held block
and each probe minimum as a block result. That moved the block boundary
and the reward of every later block, so the walk after the probe was not
the walk without it. Morse 55 at rho 14, seed 2001 under ``h2h_seed``,
reached the reference at 884178 charged calls without the probe and
missed it at 1000000 with it. The ``search_from`` quench now honours
``Ledger::is_diagnostic_quench``: a diagnostic quench descends on the
plain surface and leaves the portfolio and the local-step decision to
the walk. That seed now hits at 884563, the walk's 884178 plus the
probe's 385 calls.
