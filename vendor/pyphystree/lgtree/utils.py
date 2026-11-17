import numpy as np
from pyDOE import lhs
from typing import List, Tuple, Dict, Any, Optional
from random import random
from sklearn.linear_model import LogisticRegression
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

# ============================================================
# 1. Latin Hypercube Sampling
# ============================================================

def _latin(npoints: int, lb: List[float], ub: List[float]) -> List[np.ndarray]:
    """
    Use Latin Hypercube Sampling to generate initial points for MCTS.
    """
    lb_arr = np.asarray(lb, dtype=float)
    ub_arr = np.asarray(ub, dtype=float)
    dim = len(ub_arr)

    samples_LH = lhs(dim, samples=npoints)
    seedpoints = [lb_arr + samp * (ub_arr - lb_arr) for samp in samples_LH]
    return seedpoints


# ============================================================
# 2. Adaptive window scaling for 'b' (search window)
# ============================================================

def adaptive_window_scaling(
    w_prev: float,
    f_prev: float,
    f_curr: float,
    f_target: float = 0.0,
    eps: float = 1e-8,
    alpha: float = 1.0,
    aggressive_drop: float = 0.5,
) -> float:
    """
    Adaptive window update with fallback aggressive reduction.
    """
    prev_dist = abs(f_prev - f_target) + eps
    curr_dist = abs(f_curr - f_target) + eps

    if f_curr < f_prev:
        ratio = curr_dist / prev_dist
        ratio = max(ratio, eps)
        decay_factor = ratio ** alpha
    else:
        decay_factor = aggressive_drop

    w_new = w_prev * decay_factor
    return max(w_new, eps)


# ============================================================
# Depth scaling and basic hypersphere step
# ============================================================

def Depthscale(depth: int, a: float = 0.01, b: float = 0.5) -> float:
    """
    Depth-based radius scaling: s(depth) = b * exp(-a * depth^2)
    """
    scale = b * np.exp(-a * depth ** 2)
    return float(scale)


def _hypersphere_step_reduced(
    x_r: np.ndarray,
    depth: int,
    a: float,
    b: float,
) -> np.ndarray:
    """
    Standard isotropic hypersphere step in reduced coordinates [0,1]^d.
    """
    x_r = np.asarray(x_r, dtype=float)
    dim = x_r.shape[0]

    r_max = Depthscale(depth, a=a, b=b)
    if r_max <= 0.0:
        return np.clip(x_r, 0.0, 1.0)

    # Isotropic direction
    u = np.random.normal(0.0, 1.0, dim)
    norm = np.linalg.norm(u)
    if norm < 1e-12:
        u = np.ones(dim, dtype=float)
        norm = np.sqrt(float(dim))

    # Volume-uniform radius
    xi = random()
    r = r_max * (xi ** (1.0 / float(dim)))

    x_step = (r / norm) * u
    x_new_r = x_r + x_step
    x_new_r = np.clip(x_new_r, 0.0, 1.0)
    return x_new_r


# ============================================================
# 3. Logistic Surrogate: A Directional Sampler
# ============================================================

class DirectionalLogisticSurrogate:
    """
    Implements the coupled Logistic Regression surrogate for directional sampling.
    """

    def __init__(self, dim: int, rbf_count: int = 10, history_window: int = 100):
        self.dim = dim
        self.rbf_count = rbf_count
        self.history_window = history_window
        
        # History storage: deltas (trial - center) and labels (1 for success, 0 for failure)
        self.history_deltas: List[np.ndarray] = []
        self.history_labels: List[int] = []
        
        # Models
        self.model_dir = LogisticRegression(solver='liblinear', warm_start=True)
        self.model_dist = LogisticRegression(solver='liblinear', warm_start=True)
        self.is_fitted = False
        
        # RBF Centers (initialized lazily or static)
        # We assume reduced space distance usually < 0.5, but we can scale dynamically
        self.rbf_centers = np.linspace(0.0, 0.5, self.rbf_count)

    def _get_rbf_features(self, r: float) -> np.ndarray:
        """Projects scalar distance r into RBF feature space."""
        # phi_i(r) = exp(-(r - c_i)^2)
        # We also create a vector of distances for batch processing
        return np.exp(-((r - self.rbf_centers) ** 2) * 100.0) # *100 scales width

    def update(self, delta: np.ndarray, success: bool):
        """Adds a new observation to history."""
        self.history_deltas.append(delta)
        self.history_labels.append(1 if success else 0)
        
        # Maintain window size
        if len(self.history_deltas) > self.history_window:
            self.history_deltas.pop(0)
            self.history_labels.pop(0)

    def train(self):
        """Trains the two logistic regression models on current history."""
        if len(self.history_labels) < 10:  # Minimum data requirement
            return

        deltas = np.array(self.history_deltas)
        labels = np.array(self.history_labels)
        
        # 1. Directional Features: sign(delta)
        # Use simple epsilon to avoid sign(0) issues
        X_dir = np.sign(deltas + 1e-9)
        
        # 2. Distance Features: RBFs of norm(delta)
        norms = np.linalg.norm(deltas, axis=1)
        X_dist = np.array([self._get_rbf_features(r) for r in norms])

        # Fit models (catch errors if only one class exists in history)
        try:
            if len(np.unique(labels)) > 1:
                self.model_dir.fit(X_dir, labels)
                self.model_dist.fit(X_dist, labels)
                self.is_fitted = True
        except Exception:
            # Fallback if fitting fails (e.g. convergence issues)
            self.is_fitted = False

    def _optimize_direction_hill_climbing(self, current_weights: np.ndarray) -> np.ndarray:
        """
        Stochastic hill-climbing to find optimal u = sign(delta).
        """
        # Start from the direction suggested by the weights themselves
        best_u = np.sign(current_weights)
        # Handle zeros in weights
        best_u[best_u == 0] = 1.0
        
        # Calculate score: w^T * u
        current_score = np.dot(current_weights, best_u)
        
        # Iteratively try flipping dimensions
        # (Since this is a linear model, the optimal is actually just sign(weights),
        # but we implement the hill climber as described to allow for stochasticity
        # or future non-linear extensions).
        max_iters = self.dim * 2
        for _ in range(max_iters):
            # Pick a random dimension to flip
            idx = np.random.randint(0, self.dim)
            candidate_u = best_u.copy()
            candidate_u[idx] *= -1
            
            candidate_score = np.dot(current_weights, candidate_u)
            
            if candidate_score > current_score:
                best_u = candidate_u
                current_score = candidate_score
        
        return best_u

    def _sample_distance_inverse_transform(self, r_max: float) -> float:
        """
        Inverse Transform Sampling for step size r using the learned distance model.
        """
        if not self.is_fitted:
            return random() * r_max

        # Construct CDF via numerical integration
        # Create a grid of r values
        grid_size = 50
        r_grid = np.linspace(0, r_max, grid_size)
        
        # Predict P(success|r) for each r
        # Features for batch
        features = np.array([self._get_rbf_features(r) for r in r_grid])
        probs = self.model_dist.predict_proba(features)[:, 1] # Probability of class 1
        
        # Numerical integration (Cumulative Sum)
        cdf = np.cumsum(probs)
        cdf = cdf / cdf[-1] # Normalize
        
        # Inverse sampling
        target_p = random()
        
        # Use interpolation to invert CDF
        # Find index where cdf >= target_p
        idx = np.searchsorted(cdf, target_p)
        
        if idx == 0:
            return r_grid[0]
        if idx >= grid_size:
            return r_grid[-1]
            
        # Linear interpolation between idx-1 and idx
        r0, r1 = r_grid[idx-1], r_grid[idx]
        c0, c1 = cdf[idx-1], cdf[idx]
        
        # r = r0 + (target - c0) * (r1 - r0) / (c1 - c0)
        return r0 + (target_p - c0) * (r1 - r0) / (c1 - c0)

    def generate_proposal(self, x_current: np.ndarray, r_max: float) -> np.ndarray:
        """
        Generate a new trial point using the trained surrogate.
        """
        if not self.is_fitted:
            # Fallback to heuristics if model not ready
            return self._heuristic_proposal(x_current, r_max)

        # 1. Optimize Direction
        # Get weights w_dir from model (coef_ is shape (1, n_features))
        weights = self.model_dir.coef_[0]
        u_opt = self._optimize_direction_hill_climbing(weights)
        
        # Add slight noise to direction to ensure exploration (probabilistic nature)
        if random() < 0.1:
            flip_idx = np.random.randint(0, self.dim)
            u_opt[flip_idx] *= -1

        # 2. Sample Step Size
        r_opt = self._sample_distance_inverse_transform(r_max)
        
        # Construct Step
        # u is a sign vector, normalize it to unit vector for direction
        u_norm = u_opt / np.linalg.norm(u_opt)
        
        # New point
        x_new = x_current + r_opt * u_norm
        return np.clip(x_new, 0.0, 1.0)

    def _heuristic_proposal(self, x_current: np.ndarray, r_max: float) -> np.ndarray:
        """
        Bootstrapping phase: Physical heuristics.
        """
        strategy = random()
        
        if strategy < 0.33:
            # 1. Momentum/Center bias (simplified as random step towards 0.5)
            direction = 0.5 - x_current
            dist = np.linalg.norm(direction)
            if dist > 1e-6:
                direction /= dist
            r = random() * r_max
            return np.clip(x_current + r * direction, 0.0, 1.0)
            
        elif strategy < 0.66:
            # 2. Diagonal Extrema exploration
            # Pick a random corner
            corner = np.random.randint(0, 2, self.dim).astype(float)
            direction = corner - x_current
            dist = np.linalg.norm(direction)
            if dist > 1e-6:
                direction /= dist
            r = random() * r_max
            return np.clip(x_current + r * direction, 0.0, 1.0)
            
        else:
            # 3. Pure random (Hypersphere fallback)
            return _hypersphere_step_reduced(x_current, 0, 0.01, r_max)  # Dummy depth/a params


# ============================================================
# Unified Perturbate wrapper 
# ============================================================

class Perturbate:
    """
    Unified perturbation helper handling both hypersphere and logistic modes.
    """

    def __init__(
        self,
        lb,
        ub,
        sampling_mode: str = "hypersphere",
        a: float = 4e-2,
        b: float = 0.5,
        rng: Optional[np.random.Generator] = None,
        **logistic_kwargs: Any,
    ) -> None:
        self.lb = np.asarray(lb, dtype=float)
        self.ub = np.asarray(ub, dtype=float)
        if self.lb.shape != self.ub.shape:
            raise ValueError("lb and ub must have the same shape in Perturbate.")
        self.dim = int(len(self.ub))

        self.mode = sampling_mode.lower().strip()
        if self.mode not in ("hypersphere", "logistic"):
            raise ValueError(f"Unknown sampling_mode: {sampling_mode}")

        self.a = float(a)
        self.b = float(b)
        self.rng = rng or np.random.default_rng()

        # Initialize Logistic Surrogate if mode is active
        self.surrogate = None
        if self.mode == "logistic":
            self.surrogate = DirectionalLogisticSurrogate(
                dim=self.dim,
                rbf_count=logistic_kwargs.get("rbf_count", 10),
                history_window=logistic_kwargs.get("history_window", 100)
            )

    # ------------- coordinate helpers -------------

    def _to_reduced(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        denom = (self.ub - self.lb)
        denom = np.where(denom == 0.0, 1e-12, denom)
        return (x - self.lb) / denom

    def _from_reduced(self, x_r: np.ndarray) -> np.ndarray:
        x_r = np.asarray(x_r, dtype=float)
        return x_r * (self.ub - self.lb) + self.lb

    # ------------- public API -------------

    def generate_new_parameter(
        self,
        parameter: np.ndarray,
        depth: int,
        node_id: Optional[int] = None,
        parent_id: Optional[int] = None,
    ) -> np.ndarray:
        """
        Propose a new parameter vector.
        """
        x_center = np.asarray(parameter, dtype=float)
        x_center_r = self._to_reduced(x_center)
        
        # Calculate max radius based on depth scaling
        r_max = Depthscale(depth, a=self.a, b=self.b)

        if self.mode == "logistic" and self.surrogate is not None:
            # Periodically retrain (e.g., every call, or optimize to train less often)
            self.surrogate.train()
            
            x_new_r = self.surrogate.generate_proposal(x_center_r, r_max)
        else:
            # Fallback to Hypersphere
            x_new_r = _hypersphere_step_reduced(x_center_r, depth, a=self.a, b=self.b)

        return self._from_reduced(x_new_r)

    def record_outcome(
        self,
        node_id: int,
        x_center: np.ndarray,
        x_trial: np.ndarray,
        f_trial: float,
        f_ref: float,
    ) -> None:
        """
        Record the outcome of a trial to train the surrogate.
        """
        if self.mode != "logistic" or self.surrogate is None:
            return

        # Calculate delta in reduced space to be consistent with training features
        x_center_r = self._to_reduced(x_center)
        x_trial_r = self._to_reduced(x_trial)
        delta = x_trial_r - x_center_r
        
        # Success definition: did we improve over reference?
        is_success = f_trial < f_ref
        
        self.surrogate.update(delta, is_success)