``Config::retry_noop_stall_leave`` retries a call-clock leave when
activation produced no candidate. It is off by default, so an attempted
leave still consumes its plateau window.
