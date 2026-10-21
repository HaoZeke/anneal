The cluster-search quench stops at gradient 1e-5, where the line search
stops accepting steps, rather than at ``Config::record_gradient``. A
minimum stays recordable at ``record_gradient``. The default rgmin handle
still requests 1e-6.
