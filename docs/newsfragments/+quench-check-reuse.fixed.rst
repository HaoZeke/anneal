A cluster-search quench no longer pays twice for one point.
The record check reuses the descent's last evaluation, and a descent that starts where the last one ended reuses that one too.
A record polish is now a descent to the solver's default tolerance rather than to ``Config::record_gradient``, so the best a run reports is settled.
``Config::polish_records`` sets the length of the polish, and zero turns it off.
