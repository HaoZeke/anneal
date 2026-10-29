A search whose first quench is not finite returns
``SearchError::NoFiniteStart``. The ledger redraws the start, up to
eight attempts, and records each redraw. A finite first quench is
unchanged. A caller stop during that opening quench returns the
outcome recorded so far.
