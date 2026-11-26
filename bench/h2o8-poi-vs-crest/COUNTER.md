# Counter audit

Engine commit `b35601b7b7b2325c28f20ac604d267965cd64648` (`enh: polish a new incumbent in the hop`). `feat/polish-on-improve` is not a remote branch. Line numbers under `src/` are from that commit. The harness files cited below are the copies on `42e708289a718d3d916b46e5ae43f26cf0a677c1`.

`git fetch origin feat/polish-on-improve` failed (`fatal: couldn't find remote ref`); `b35601b7` was already a local object (`git cat-file -t b35601b7` printed `commit`), `git merge-base --is-ancestor b35601b7 HEAD` exits 1, and `git branch -a --contains b35601b7` lists `origin/develop`.

## Finding

Polish-on-improve local-polish objective evaluations are counted. A finite polish evaluation increments the hop ledger and the objective counter that the harness stores as `calls_to_hit`. The stored column is not adjusted.

## What produces calls_to_hit

`bench/octamer-matched/h2o8/PROTOCOL.md:50-53` installs reference plus tolerance with `Ledger::stop_below` and defines `calls_to_hit` as the first objective evaluation at or below that threshold.

`bench/octamer-matched/h2o8/run.py:235-248` copies `first_hit_call` from `run.txt` into `calls_to_hit` when the seed hits.

`examples/gfn2_mol_h2h.rs:258-263` sets that field. After a finite `value_and_gradient`, it increments `OBJECTIVE_CALLS` and, on the first energy for which `energy_hits(energy, target, 0.0)` holds, stores that index in `FIRST_HIT_CALL`. `examples/gfn2_mol_h2h.rs:567-568` is the `stop_below` call. A non-finite result returns at `examples/gfn2_mol_h2h.rs:255-256` and does not increment the counter.

## Polish uses that objective and that ledger

`src/methods/cluster_hopping/config.rs:1364-1370` says each polish-on-improve evaluation is charged on the hop's own ledger, with nothing held back and nothing granted, and that a polished point at or below the stop target ends the search on that evaluation.

`src/methods/cluster_search.rs:1618-1619`: `search_from` with the flag set calls `search_from_polishing` on the same objective and ledger. `src/methods/cluster_search.rs:1677-1679` and `:1781-1785` bind `polish_improved_answer` to that ledger and pass it into the hop.

`src/methods/cluster_search.rs:1994-2022`: the polish stops on the first evaluation at or below the ledger target. The open wall uses `polish_answer_at`. Any other wall uses `polish_answer_walled_at`.

Open wall, `src/methods/cluster_search.rs:1977-1984`: `ledger.charge()` then `evaluated_pair`. If the energy is at or below the target, that evaluation is the hit. `src/methods/cluster_search.rs:317` is `objective.value_and_gradient`, the method that increments `OBJECTIVE_CALLS`.

Walled path, `src/methods/cluster_search.rs:600` charges the same ledger inside `charge_or_refuse_overlap`, then evaluates the objective. `src/methods/cluster_search.rs:1928` stops when `ledger.below_target` is true. An overlap refusal at `src/methods/cluster_search.rs:586-589` does not call the objective and is not an objective evaluation.

`src/methods/cluster_hopping.rs:880-885`: `charge` increments `spent`. `src/methods/cluster_hopping.rs:5545-5546` records a lower polished point on that ledger. `src/methods/cluster_hopping.rs:993` stores the encounter as `(spent, value)`.

`src/methods/cluster_hopping.rs:809-811`: `below_target` is `energy_hits(energy, target, 0.0)`. `src/target_tol.rs:14` and `:24` define that as `energy <= target + tol`. The harness target is already reference plus tolerance, and this comparison uses tolerance 0, so the rule is the first objective evaluation at or below reference plus tolerance.

`src/methods/cluster_search.rs:3192-3208` defines the first-hit call as that encounter index and checks it equals `ledger.spent()`. `src/methods/cluster_hopping.rs:13363-13394` checks that the polish charge's `spent` is the encounter of the target crossing.
