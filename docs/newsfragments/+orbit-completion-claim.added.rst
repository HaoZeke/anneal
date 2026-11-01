A shared minimum history can claim the orbit completion of each minimum.
``HistoryHook::claim_completion`` returns true the first time a chain asks for a minimum and false after that.
Its default is true, so a chain without a history completes as before.
Each completion product is now reported to the history, so other chains recognise it.
When a second orbit candidate is identical to the first, it is skipped before its quench.
``Outcome::orbit_duplicates`` counts those skips.
