``Config::retry_noop_stall_leave`` retries a directed leave on the next hop when its activation produced no candidate.
By default an attempted leave uses up its plateau window whether or not it moved the chain.
``Config::align_directed_stall_source`` starts a directed leave from the recorded best rather than from the live walker, which may have accepted an uphill hop into another basin since the call clock was armed.
Both are off by default and are left out of the resolved configuration while off, so a stock preset keeps its digest.
