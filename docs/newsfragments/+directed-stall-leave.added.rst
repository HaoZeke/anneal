``Config::retry_noop_stall_leave`` and ``Config::align_directed_stall_source``
gate two directed-leave behaviours. The first consumes the call-clock
plateau only when a stall response runs, and retries a leave that produced
no candidate. The second starts that leave from the recorded best rather
than the live walker. Both default off. ``lj_cluster_search`` names them
``retrynoop`` and ``alignstall``.
