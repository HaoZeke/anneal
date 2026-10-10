A ``cma`` arm for values-only portfolio runs (``policy="auto"``, no
``grad_fn``, no ``noise_sigma``): resumable (mu/mu_w, lambda)-CMA-ES with
cumulative step-size adaptation, rank-one plus rank-mu covariance updates, and
BIPOP restarts. Every run is centred on the incumbent, which is ``x0`` until a
lower point is found; the first run is local, and the first restart after it
has the default population and a box-wide step size. A run ends once its
recent best values span less than the portfolio's success threshold in force
when the arm last took a slice, which ``CmaEs::set_tol_fun_hist`` sets between
generations (``CmaEs::tol_fun_hist`` reads it). Above 100 dimensions, or when
the budget is shorter than the covariance's learning horizon, runs keep only
the diagonal (sep-CMA-ES), so a candidate costs linear time and memory.
Candidates are mirror-reflected into the box and each one is a single charged
evaluation.
