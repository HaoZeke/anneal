``global_optimize`` now clips an ``x0`` start onto the box bounds, matching
the documented contract and the Rust core behavior. Non-finite starts and
length mismatches are still rejected.
