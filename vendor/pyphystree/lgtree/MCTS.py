import logging
from math import log
from typing import List, Optional, Tuple, Deque, Dict, Any
from collections import deque

import numpy as np

from .utils import Perturbate


class Node:
    """
    A class representing a single node in the MCTS tree.
    """

    def __init__(self, node_id: int, parent: Optional["Node"] = None, depth: int = 0):
        self.node_id = node_id
        self.parent = parent
        self.depth = depth
        self.visits: int = 1
        self.children: List[int] = []
        # indices into MCTS.score_list / parameter_list
        self.playout_data: List[int] = []
        self.score: Optional[float] = None
        self.parameter: Optional[List[float]] = None

    def add_child(self, child_id: int) -> None:
        self.children.append(child_id)

    def add_playout(self, idx: int) -> None:
        self.playout_data.append(idx)

    def update_score(self, score: float, parameter: List[float]) -> None:
        self.score = score
        self.parameter = parameter


class MCTS:
    """
    Continuous MCTS with pluggable sampling_mode: "hypersphere" or "logistic",
    adapted to the API expected by lgtree.py.

    All perturbation logic (pure hypersphere vs logistic surrogate) is handled
    by the unified Perturbate class from utils.py.
    """

    def __init__(
        self,
        root_data: List[float],
        lower_bound: np.ndarray,
        upper_bound: np.ndarray,
        evaluate,
        nplayouts: int = 10,
        explore_constant_max: float = 1.0,
        explore_constant_min: float = 0.1,
        max_depth: int = 12,
        a: float = 4e-2,
        b: float = 1.0 / np.sqrt(2.0),
        logger: Optional[logging.Logger] = None,
        sampling_mode: str = "hypersphere",
        logistic_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args (matching lgtree.py expectations):
            root_data        : initial parameter vector for the root.
            lower_bound      : array-like lower bounds.
            upper_bound      : array-like upper bounds.
            evaluate         : function(params) -> (relaxed_params, score).
            nplayouts        : playouts per node.
            explore_constant_max/min : UCB exploration constants.
            max_depth        : max tree depth.
            a, b             : depth / window scaling parameters.
                                - `a` controls exponential decay with depth.
                                - `b` sets overall window size (applies to both
                                  hypersphere and logistic modes).
            logger           : logger instance.
            sampling_mode    : "hypersphere" or "logistic".
            logistic_kwargs  : kwargs for logistic surrogate (DirectionalLogisticSurrogate),
                               passed down through unified Perturbate.
        """
        # Core settings
        self.evaluate = evaluate
        self.nplayouts = nplayouts
        self.max_depth = max_depth
        self.explore_constant_max = explore_constant_max
        self.explore_constant_min = explore_constant_min
        self.explore_constant = explore_constant_max
        self.a = float(a)
        self.b = float(b)

        self.logger = logger or logging.getLogger(__name__)

        # Bounds
        self.lb = np.asarray(lower_bound, dtype=float)
        self.ub = np.asarray(upper_bound, dtype=float)
        if self.lb.shape != self.ub.shape:
            raise ValueError("lower_bound and upper_bound must have same shape.")
        if np.any(self.ub <= self.lb):
            self.logger.warning("Some upper_bound <= lower_bound; check your bounds.")

        # Sampling mode
        self.sampling_mode = sampling_mode.lower().strip()
        if self.sampling_mode not in ("hypersphere", "logistic"):
            raise ValueError(f"Unknown sampling_mode: {self.sampling_mode}")

        # Logistic / perturbator config
        self.logistic_kwargs: Dict[str, Any] = dict(logistic_kwargs or {})

        # IMPORTANT: avoid passing a,b twice (explicit and via kwargs)
        kw = dict(self.logistic_kwargs)
        kw.pop("a", None)
        kw.pop("b", None)

        # Unified perturbator (handles both hypersphere and logistic internally)
        self.perturbator = Perturbate(
            lb=self.lb,
            ub=self.ub,
            sampling_mode=self.sampling_mode,
            a=self.a,
            b=self.b,
            **kw,
        )

        # Global playout index (indexes into score_list / parameter_list)
        self.idx: int = 0

        # Initialize root node
        root_data = np.asarray(root_data, dtype=float)
        data_relaxed, score = self.evaluate(root_data)
        self.root_node = Node(node_id=0, parent=None, depth=0)
        self.root_node.update_score(float(score), list(data_relaxed))

        self.score_list: List[float] = [float(score)]             # index 0 = root eval
        self.parameter_list: List[List[float]] = [list(data_relaxed)]

        # Node registry
        self.nodes: Dict[int, Node] = {0: self.root_node}

        # Initial playouts from root
        self.perform_playouts(self.root_node, self.nplayouts)

    # ---------- playouts ----------

    def perform_playouts(self, node: Node, nplayouts: int) -> None:
        """
        Perform playouts for a given node.

        Uses the unified Perturbate:
          - in "hypersphere" mode: depth-scaled random hypersphere moves;
          - in "logistic"  mode: DirectionalLogisticSurrogate-guided moves,
            with record_outcome called after each evaluation.
        """
        if node.parameter is None:
            self.logger.warning(
                f"Node {node.node_id} has no parameter set; skipping playouts."
            )
            return

        x_center = np.asarray(node.parameter, dtype=float)

        # Reference score for success / failure labelling
        f_ref = (
            node.score
            if node.score is not None
            else min(self.score_list)
        )

        for _ in range(nplayouts):
            self.idx += 1

            # Propose a new point from unified perturbator
            playdata = self.perturbator.generate_new_parameter(
                parameter=x_center,
                depth=node.depth,
                node_id=node.node_id,
                parent_id=node.parent.node_id if node.parent is not None else None,
            )

            playdata_relaxed, playscore = self.evaluate(playdata)

            node.add_playout(self.idx)
            self.score_list.append(float(playscore))
            self.parameter_list.append(list(playdata_relaxed))

            # Update logistic surrogate (no-op in pure hypersphere mode)
            self.perturbator.record_outcome(
                node_id=node.node_id,
                x_center=x_center,
                x_trial=np.asarray(playdata_relaxed, dtype=float),
                f_trial=float(playscore),
                f_ref=float(f_ref),
            )

            # Update node's best and reference if improvement
            if playscore < f_ref or node.score is None:
                f_ref = float(playscore)
                node.update_score(f_ref, list(playdata_relaxed))
                x_center = np.asarray(playdata_relaxed, dtype=float)

    # ---------- tree utilities ----------

    def get_descendants(self, node: Node) -> List[int]:
        """
        Get all descendant node IDs of a node (including itself).
        """
        descendants_ids: List[int] = []
        nodes_to_process: Deque[Node] = deque([node])

        while nodes_to_process:
            current_node = nodes_to_process.popleft()
            descendants_ids.append(current_node.node_id)
            nodes_to_process.extend(
                self.nodes[child_id] for child_id in current_node.children
            )

        return descendants_ids

    def backpropagate_and_select(self) -> Optional[Node]:
        """
        Select a node according to UCB over descendants' best scores.
        """

        # Special case: if only the root exists, start expanding from root
        if len(self.nodes) == 1:
            return self.root_node

        best_node: Optional[Node] = None
        best_ucb: float = -float("inf")

        for node in self.nodes.values():
            # Skip root as a target once there are children; we always expand below it
            if node.parent is None:
                continue

            if node.depth > self.max_depth:
                ucb_score = -1e10
            else:
                descendants_ids = self.get_descendants(node)
                all_indexes: List[int] = []
                for descendant_id in descendants_ids:
                    descendant_node = self.nodes[descendant_id]
                    all_indexes.extend(descendant_node.playout_data)

                if not all_indexes:
                    ucb_score = -1e9
                else:
                    # We are minimizing: best_reward = min score
                    best_reward = min(self.score_list[i] for i in all_indexes)
                    parent_visits = self.nodes[node.parent.node_id].visits
                    ucb_score = -best_reward + self.explore_constant * np.sqrt(
                        log(parent_visits) / node.visits
                    )

            if ucb_score > best_ucb:
                best_ucb = ucb_score
                best_node = node

        return best_node

    # ---------- public API used by lgtree.py ----------

    def run(
        self,
        niterations: int = 200,
        nexpand: int = 3,
        nsimulate: int = 1,
        patience: int = 10,
        score_threshold_iterations: int = 5,
        head_data: Optional[Any] = None,  # unused in this implementation
        verbose: bool = True,
        tolerance: float = 1e-6,
    ) -> Tuple[float, Optional[int], List[float], int]:
        """
        Run the MCTS loop.

        Returns
        -------
        minscore     : best (lowest) score found.
        mindepth     : depth of the node that generated the best playout
                       (0 if root eval was best, None if not found).
        minparameter : parameter vector corresponding to minscore.
        evals        : total number of evaluations performed in this MCTS.
        """
        # Initial best
        best_score = min(self.score_list)
        patience_counter = 0
        no_improvement_counter = 0

        if tolerance <= 0:
            decimals = 6
        else:
            decimals = max(0, int(-np.log10(tolerance)))

        for iteration in range(niterations):
            # Expansion
            for _ in range(nexpand):
                selected_node = self.backpropagate_and_select()
                if selected_node is None:
                    self.logger.warning("No node available for expansion.")
                    break
                self.expand_node(selected_node)

            # Simulation
            for _ in range(nsimulate):
                selected_node = self.backpropagate_and_select()
                if selected_node is None:
                    self.logger.warning("No node available for simulation.")
                    break
                self.simulate_node(selected_node)

            current_best_score = min(self.score_list)

            if round(current_best_score, decimals) == round(best_score, decimals):
                patience_counter += 1
                no_improvement_counter += 1
            else:
                patience_counter = 0
                no_improvement_counter = 0

            best_score = current_best_score

            # Adjust exploration constant based on stagnation
            if no_improvement_counter >= score_threshold_iterations:
                self.explore_constant = self.explore_constant_max
                self.logger.info(
                    f"Score stagnant for {score_threshold_iterations} iters; "
                    f"explore_constant = {self.explore_constant_max}."
                )
            else:
                self.explore_constant = self.explore_constant_min
                self.logger.info(
                    f"Using minimum exploration constant {self.explore_constant_min}."
                )

            self.logger.info(
                f"Iteration {iteration + 1}: "
                f"Total evals: {len(self.score_list)}, "
                f"Best score: {best_score}, "
                f"Explore C: {self.explore_constant}"
            )

            if verbose:
                print(
                    f"[MCTS] Iter {iteration + 1}: "
                    f"evals={len(self.score_list)}, best_score={best_score:.6g}, "
                    f"C={self.explore_constant:.3g}"
                )

            if patience_counter >= patience:
                self.logger.info(
                    f"Early stopping after {patience_counter} iterations "
                    f"with no improvement."
                )
                break

        # ---- Extract global best result ----
        minscore = min(self.score_list)
        minidx = int(self.score_list.index(minscore))
        minparameter = self.parameter_list[minidx]

        # Determine depth of node that owns this playout index
        mindepth: Optional[int] = None
        if minidx == 0:
            # root evaluation
            mindepth = 0
        else:
            for node in self.nodes.values():
                if minidx in node.playout_data:
                    mindepth = node.depth
                    break

        evals = len(self.score_list)
        return float(minscore), mindepth, list(minparameter), int(evals)

    # ---------- expand / simulate ----------

    def expand_node(self, parent_node: Node) -> None:
        """
        Expand the given parent node: pick its best playout and create a child.
        """
        if not parent_node.playout_data:
            self.logger.warning(
                f"Parent node {parent_node.node_id} has no playouts; "
                f"performing playouts before expansion."
            )
            self.perform_playouts(parent_node, self.nplayouts)

        parent_node.visits += 1

        best_play_index = min(
            parent_node.playout_data,
            key=lambda idx: self.score_list[idx],
        )
        best_play_score = self.score_list[best_play_index]
        best_play_param = self.parameter_list[best_play_index]

        child_id = self.idx + 1
        child_node = Node(
            node_id=child_id,
            parent=parent_node,
            depth=parent_node.depth + 1,
        )
        child_node.update_score(float(best_play_score), list(best_play_param))

        self.nodes[child_node.node_id] = child_node
        parent_node.add_child(child_node.node_id)

        self.perform_playouts(child_node, self.nplayouts)

    def simulate_node(self, node: Node) -> None:
        """
        Just perform additional playouts from this node (no new child).
        """
        node.visits += 1
        self.perform_playouts(node, self.nplayouts)
