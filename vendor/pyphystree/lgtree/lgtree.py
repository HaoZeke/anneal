from typing import List, Optional, Tuple, Dict, Any
import numpy as np
import logging

from .MCTS import MCTS
from .utils import _latin, adaptive_window_scaling  # kept for compatibility


class MCTSBatch:
    def __init__(
        self,
        objfunc: Any,
        config_dict: Dict[int, Dict[str, Any]],
        logger, 
        nplayouts: int = 5,
        explore_constant_max: float = 1e20,
        explore_constant_min: float = 1e-20,
        max_depth: int = 12,
        niterations: int = 5,
        nexpand: int = 2,
        nsimulate: int = 1,
        patience: int = 3,
        score_threshold_iterations: int = 2,
        verbose: bool = True,
        sampling_mode: str = "hypersphere",
        logistic_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Batch wrapper around the MCTS implementation.

        Parameters
        ----------
        objfunc : callable
            Objective function: x -> (x_relaxed, score).
        config_dict : dict
            Per-tree configuration:
                {
                    tree_id: {
                        "a": float,
                        "b": float,
                        "root_data": np.ndarray,
                        "head_data": Any or None,
                        "lower_bounds": np.ndarray,
                        "upper_bounds": np.ndarray,
                    },
                    ...
                }
        sampling_mode : {"hypersphere", "logistic"}
            Sampling mode forwarded to MCTS (and internally to Perturbate).
        logistic_kwargs : dict or None
            Extra kwargs for logistic mode; forwarded to MCTS, which passes
            them into the unified Perturbate (where the directional surrogate
            is managed).
        """
        self.objfunc = objfunc
        self.config_dict = config_dict
        self.logger = logger

        # MCTS hyperparameters
        self.nplayouts = nplayouts
        self.explore_constant_max = explore_constant_max
        self.explore_constant_min = explore_constant_min
        self.max_depth = max_depth
        self.niterations = niterations
        self.nexpand = nexpand
        self.nsimulate = nsimulate
        self.patience = patience
        self.score_threshold_iterations = score_threshold_iterations
        self.verbose = verbose

        # Sampling mode and logistic surrogate config
        self.sampling_mode = sampling_mode
        self.logistic_kwargs: Dict[str, Any] = logistic_kwargs or {}

        self.best_result: Optional[Tuple[float, List[float]]] = None

    def run(self) -> Tuple[Dict[int, Dict[str, Any]], int]:
        """
        Run an ensemble of MCTS trees (one per config in config_dict).

        Returns
        -------
        best_stats : dict
            Per-tree best info:
                {
                    tree_id: {
                        "a": float,
                        "b": float,
                        "score": float,
                        "depth": int or None,
                        "parameter": List[float],
                        "lb": np.ndarray,
                        "ub": np.ndarray,
                    },
                    ...
                }
        total_calculations : int
            Total number of objective evaluations across all trees.
        """
        best_stats: Dict[int, Dict[str, Any]] = {}
        total_calculations: int = 0
        best_result: Optional[Tuple[float, List[float]]] = None

        for i, config in self.config_dict.items():
            root_data = np.asarray(config["root_data"], dtype=float)
            head_data = config["head_data"]
            lb = np.asarray(config["lower_bounds"], dtype=float)
            ub = np.asarray(config["upper_bounds"], dtype=float)
            a = float(config["a"])
            b = float(config["b"])

            self.logger.info(f"Running MCTS tree id={i} with a = {a}, b = {b}")
            self.logger.info(f"Root data: {root_data}")
            self.logger.info(f"Head data: {head_data}")
            self.logger.info(f"Bounds: lower = {lb}, upper = {ub}")
            self.logger.info(f"Sampling mode: {self.sampling_mode}")

            # MCTS instance
            mcts = MCTS(
                root_data=root_data,
                lower_bound=lb,
                upper_bound=ub,
                evaluate=self.objfunc,
                nplayouts=self.nplayouts,
                explore_constant_max=self.explore_constant_max,
                explore_constant_min=self.explore_constant_min,
                max_depth=self.max_depth,
                a=a,
                b=b,  # depth / window scaling parameter
                logger=self.logger,
                sampling_mode=self.sampling_mode,
                logistic_kwargs=self.logistic_kwargs,
            )

            # Run MCTS
            minscore, mindepth, minparameter, evals = mcts.run(
                niterations=self.niterations,
                nexpand=self.nexpand,
                nsimulate=self.nsimulate,
                patience=self.patience,
                score_threshold_iterations=self.score_threshold_iterations,
                head_data=head_data,
                verbose=self.verbose,
            )

            best_stats[i] = {
                "a": a,
                "b": b,
                "score": float(minscore),
                "depth": mindepth,
                "parameter": minparameter,
                "lb": lb,
                "ub": ub,
            }

            if best_result is None or minscore < best_result[0]:
                best_result = (float(minscore), minparameter)

            total_calculations += int(evals)
            self.logger.info(f"[Tree {i}] best score: {minscore}")
            self.logger.info(f"[Tree {i}] best parameter: {minparameter}")

        self.best_result = best_result
        return best_stats, total_calculations


def run_global_tree_batch(global_batch: MCTSBatch, top_k: int = 10):
    """
    Run the global batch and keep only the top_k trees with lowest score.
    """
    stats_global, total_evals = global_batch.run()

    if not stats_global:
        return {}, total_evals

    sorted_stats = sorted(stats_global.items(), key=lambda x: x[1]["score"])[:top_k]
    top_stats = {idx: result for idx, result in sorted_stats}

    return top_stats, total_evals


def run_local_tree_batch(
    local_batch: MCTSBatch,
    top_stats: Dict[int, Dict[str, Any]],
    objfunc: Any,
    logger: logging.Logger,
    n_iterations: int = 1000,
    stagnation_threshold: int = 2,
    alpha: float = 0.5,
    target: float = 0.0,
    aggressive_drop_factor: float = 0.1,
    epsilon: float = 1e-12,
    total_evals_sofar: int = 0,
):
    """
    Local refinement loop with adaptive b:

    - `b` is a depth/window scaling parameter. Larger b => broader effective
      search window via MCTS depth scaling (both hypersphere and logistic modes).

    - If a config improves (lower score than last), update:

          b_new = b_old * ((f_curr - target + eps) / (f_last - target + eps))**alpha

      which makes b smaller as you get closer to the target (if alpha > 0).

    - If a config does not improve, shrink b aggressively:

          b_new = b_old * aggressive_drop_factor

    - Always preserve the best-performing tree (preserved_idx) to avoid
      pruning everything accidentally.

    Returns
    -------
    top_stats : dict
        Refined per-tree stats after local optimization.
    total_evals_sofar : int
        Total number of evaluations used (global + local).
    minscore : float
        Best score found.
    best_parameter : list or None
        Parameters corresponding to minscore.
    """
    if not top_stats:
        logger.warning("run_local_tree_batch called with empty top_stats.")
        return top_stats, total_evals_sofar, float("inf"), None

    # Deep-ish copies to track best per-tree states
    prev_best: Dict[int, Dict[str, Any]] = {
        idx: dict(result) for idx, result in top_stats.items()
    }
    stagnation_counter: Dict[int, int] = {idx: 0 for idx in top_stats}

    previous_b: Dict[int, float] = {idx: result["b"] for idx, result in top_stats.items()}
    previous_score: Dict[int, float] = {idx: result["score"] for idx, result in top_stats.items()}

    minscore, best_parameter = min(
        ((v["score"], v["parameter"]) for v in top_stats.values()),
        key=lambda x: x[0],
    )
    loop_stagnation_counter = 0  # not currently used in stopping, but kept if needed

    # Always keep the currently best config from pruning
    preserved_idx = min(previous_score.items(), key=lambda x: x[1])[0]

    i = 0
    while True:
        logger.info(f"=== Local Optimization Round {i + 1} ===")

        new_config_dict: Dict[int, Dict[str, Any]] = {}
        improved = False

        for idx, result in top_stats.items():
            if stagnation_counter[idx] > stagnation_threshold and idx != preserved_idx:
                # Pruned due to stagnation (except best)
                continue

            curr_score = float(result["score"])
            curr_param = np.asarray(result["parameter"], dtype=float)
            curr_a = float(result["a"])
            curr_depth = result["depth"]
            curr_lb = np.asarray(result["lb"], dtype=float)
            curr_ub = np.asarray(result["ub"], dtype=float)

            if idx not in previous_b:
                previous_b[idx] = float(result["b"])
                previous_score[idx] = curr_score

            last_b = previous_b[idx]
            last_score = previous_score[idx]

     
            
            if curr_score < last_score:
                # Improvement: use distance to target to scale b
                prev_dist = abs(last_score - target) + epsilon
                curr_dist = abs(curr_score - target) + epsilon
            
                ratio = curr_dist / prev_dist          # strictly positive
                ratio = max(ratio, epsilon)           # numeric safety
            
                updated_b = last_b * (ratio ** alpha)
            
                stagnation_counter[idx] = 0
                prev_best[idx] = dict(result)
                ref_param = curr_param
                ref_a = curr_a
                ref_b = updated_b
            
            else:
                # No improvement: shrink b aggressively and revert to previous best
                updated_b = last_b * aggressive_drop_factor
            
                stagnation_counter[idx] += 1
                ref = prev_best[idx]
                ref_param = np.asarray(ref["parameter"], dtype=float)
                ref_a = ref["a"]
                ref_b = updated_b
                curr_lb = np.asarray(ref["lb"], dtype=float)
                curr_ub = np.asarray(ref["ub"], dtype=float)
                curr_depth = ref["depth"]

                

            previous_score[idx] = curr_score
            previous_b[idx] = updated_b

            new_config_dict[idx] = {
                "a": ref_a,
                "b": ref_b,
                "root_data": ref_param,
                "head_data": None,
                "lower_bounds": curr_lb,
                "upper_bounds": curr_ub,
            }

        if not new_config_dict:
            print(f"All configs pruned at iteration {i + 1}. Exiting.")
            break

        # Update the local batch configs and rerun MCTS on the new windows/parameters
        local_batch.config_dict = new_config_dict
        stats_local, evals = local_batch.run()
        total_evals_sofar += evals

        # Refresh top_stats with locally refined results
        for idx in new_config_dict:
            if idx in stats_local:
                top_stats[idx] = stats_local[idx]

        # Update global best
        for key in top_stats:
            score = float(top_stats[key]["score"])
            if score < minscore:
                preserved_idx = key
                minscore = score
                best_parameter = top_stats[key]["parameter"]
                improved = True

        if not improved:
            loop_stagnation_counter += 1
        else:
            loop_stagnation_counter = 0

        print(
            f"Local Round: {i + 1}\n"
            f"  Best score: {minscore}\n"
            f"  Parameter: {best_parameter}\n"
            f"  Total evaluation: {total_evals_sofar}"
        )

        # Stopping criteria
        if total_evals_sofar > n_iterations:
            print("Stopping due to total evaluation threshold reached.")
            break

        i += 1

    return top_stats, total_evals_sofar, minscore, best_parameter


def run_optimisation_mcts(
    objfunc,
    logger,
    lb: np.ndarray,
    ub: np.ndarray,
    ntrees: int,
    top_k: int,
    a_min: float = 0.008,
    a_max: float = 0.08,
    b0: float = 0.5,
    explore_constant_min: float = 1e1,
    niterations_local: int = 5,
    niterations_global: int = 5,
    nexpand: int = 2,
    nsimulate: int = 1,
    nplayouts: int = 5,
    max_depth_global: int = 12,
    n_total_evals: int = 10000,
    score_threshold_iterations_global: int = 2,
    patience_mcts_global: int = 4,
    patience_mcts_local: int = 1,
    stagnation_threshold_pruning: int = 2,
    scale_step_factor: float = 0.005,  # reserved for future use
    alpha: float = 0.5,
    target: float = 0.0,
    aggressive_drop_factor: float = 0.1,
    epsilon: float = 1e-30,
    verbose: bool = False,
    sampling_mode: str = "hypersphere",
    logistic_kwargs: Optional[Dict[str, Any]] = None,
):
    """
    Global + local optimisation driver using an ensemble of MCTS trees.

    Notes
    -----
    - (a, b0) are per-tree depth/window scaling parameters; b is adapted in the
      local routine and passed back into MCTS.
    - sampling_mode: "hypersphere" or "logistic" (forwarded to MCTS via MCTSBatch).
    - logistic_kwargs: kwargs for logistic mode; passed to MCTS, then to the
      unified Perturbate (which internally manages directional surrogates).
    """
    a_values = np.linspace(a_min, a_max, ntrees)

    # -------------------- Generate Initial Root Points (Latin Hypercube) --------------------
    head_points = _latin(ntrees, lb, ub)

    config_dict: Dict[int, Dict[str, Any]] = {
        i: {
            "a": float(a),
            "b": float(b0),
            "root_data": head_points[i],
            "head_data": None,
            "lower_bounds": lb,
            "upper_bounds": ub,
        }
        for i, a in enumerate(a_values)
    }

    # -------------------- Construct Global Batch --------------------
    global_batch = MCTSBatch(
        objfunc=objfunc,
        config_dict=config_dict,
        logger=logger,
        explore_constant_min=explore_constant_min,
        explore_constant_max=1e300,
        max_depth=max_depth_global,
        niterations=niterations_global,
        nexpand=nexpand,
        nsimulate=nsimulate,
        nplayouts=nplayouts,
        patience=patience_mcts_global,
        score_threshold_iterations=score_threshold_iterations_global,
        verbose=verbose,
        sampling_mode=sampling_mode,
        logistic_kwargs=logistic_kwargs,
    )

    print("========== Global Trees run commenced =============")
    logger.info("Starting global MCTS batch...")

    # -------------------- Run Global Search --------------------
    top_stats, total_evals_global = run_global_tree_batch(
        global_batch, top_k=top_k
    )

    if not top_stats:
        logger.warning("No valid configs after global search.")
        print("No valid configs after global search.")
        return float("inf"), None

    best = next(iter(top_stats.values()))
    print(
        f"Global Round:\n"
        f"  Best score: {best['score']}\n"
        f"  Parameter: {best['parameter']}\n"
        f"  Total evaluation: {total_evals_global}"
    )

    # -------------------- Prepare Local Config --------------------
    local_config_dict: Dict[int, Dict[str, Any]] = {
        idx: {
            "a": result["a"],
            "b": result["b"],
            "root_data": result["parameter"],
            "head_data": None,
            "lower_bounds": result["lb"],
            "upper_bounds": result["ub"],
        }
        for idx, result in top_stats.items()
    }

    # -------------------- Construct Local Batch --------------------
    local_batch = MCTSBatch(
        objfunc=objfunc,
        config_dict=local_config_dict,
        logger=logger,
        max_depth=int(1e6),
        niterations=niterations_local,
        nexpand=nexpand,
        nsimulate=nsimulate,
        patience=patience_mcts_local,
        nplayouts=nplayouts,
        score_threshold_iterations=int(1e6),
        explore_constant_min=0.0,
        explore_constant_max=0.0,
        verbose=verbose,
        sampling_mode=sampling_mode,
        logistic_kwargs=logistic_kwargs,
    )

    print("\n\n========== Local Trees run commenced =============")
    logger.info("Starting local MCTS refinement...")

    # -------------------- Run Local Refinement --------------------
    top_stats_refined, total_evals, minscore, best_parameter = run_local_tree_batch(
        local_batch=local_batch,
        top_stats=top_stats,
        objfunc=objfunc,
        logger=logger,
        n_iterations=n_total_evals,
        stagnation_threshold=stagnation_threshold_pruning,
        alpha=alpha,
        target=target,
        aggressive_drop_factor=aggressive_drop_factor,
        epsilon=epsilon,
        total_evals_sofar=total_evals_global,
    )

    # -------------------- Final Output --------------------
    logger.info(f"Total evaluations performed: {total_evals}")
    print("========== Optimization Complete ==========")
    print(f"Total evaluations: {total_evals}")

    return minscore, best_parameter
