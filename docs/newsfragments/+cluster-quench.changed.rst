A cluster quench stops at gradient 1e-5, where the line search stops
accepting steps, and reuses that evaluation's gradient when the quenched
point is the one the convergence check would read. Sorted pair-spectrum
distances are scanned from the tail.
