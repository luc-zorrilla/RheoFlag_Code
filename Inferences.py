from typing import Any, Callable, Dict, List, Tuple, Optional, Type
from functools import partial
from dataclasses import dataclass, field
import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import minimize, OptimizeResult
import scipy.differentiate as sd
from itertools import product
import dill as pickle
from pathlib import Path

# ============================================================================
# FUNCTIONS
# ============================================================================

# TODO: turn bounded optimisation algorithm into unbounded algorithms using such transform

def transform_to_unbounded(t, bounds):
    """
    Transform unbounded R^n to bounded/semi-bounded domain.
    
    For each dimension:
    - Fully bounded [a, b]: sigmoid transform x = a + (b-a) / (1 + exp(-t))
    - Lower bounded [a, ∞): exponential transform x = a + exp(t)
    - Upper bounded (-∞, b]: negative exponential x = b - exp(t)
    - Unbounded (-∞, ∞): identity transform x = t
    
    Parameters:
        t: ndarray of shape (n,) - unbounded variables
        bounds: object with .lb and .ub attributes (arrays or None elements)
    
    Returns:
        x: ndarray of shape (n,) - transformed variables
    """
    bounds_low = np.asarray(bounds.lb, dtype=object)
    bounds_high = np.asarray(bounds.ub, dtype=object)
    
    x = np.zeros_like(t, dtype=float)
    
    for i in range(len(t)):
        lower = bounds_low[i]
        upper = bounds_high[i]
        
        lower_finite = lower is not None and np.isfinite(float(lower))
        upper_finite = upper is not None and np.isfinite(float(upper))
        
        if lower_finite and upper_finite:
            # Fully bounded [a, b]: sigmoid
            lower_f = float(lower)
            upper_f = float(upper)
            sigmoid = 1.0 / (1.0 + np.exp(-t[i]))
            x[i] = lower_f + (upper_f - lower_f) * sigmoid
            
        elif lower_finite and not upper_finite:
            # Lower bounded [a, ∞): exponential
            lower_f = float(lower)
            x[i] = lower_f + np.exp(t[i])
            
        elif not lower_finite and upper_finite:
            # Upper bounded (-∞, b]: negative exponential
            upper_f = float(upper)
            x[i] = upper_f - np.exp(t[i])
            
        else:
            # Unbounded (-∞, ∞): identity
            x[i] = t[i]
    
    return x

def jacobian_diagonal(t, bounds):
    """
    Compute diagonal of Jacobian matrix for transformation.
    
    For each dimension:
    - Fully bounded [a, b]: J_ii = (b-a) * sigmoid(t_i) * (1 - sigmoid(t_i))
    - Lower bounded [a, ∞): J_ii = exp(t_i)
    - Upper bounded (-∞, b]: J_ii = -exp(t_i)
    - Unbounded (-∞, ∞): J_ii = 1.0
    
    Parameters:
        t: ndarray of shape (n,) - unbounded variables
        bounds: object with .lb and .ub attributes
    
    Returns:
        diag_J: ndarray of shape (n,) - diagonal elements of Jacobian
    """
    bounds_low = np.asarray(bounds.lb, dtype=object)
    bounds_high = np.asarray(bounds.ub, dtype=object)
    
    diag_J = np.zeros_like(t, dtype=float)
    
    for i in range(len(t)):
        lower = bounds_low[i]
        upper = bounds_high[i]
        
        lower_finite = lower is not None and np.isfinite(float(lower))
        upper_finite = upper is not None and np.isfinite(float(upper))
        
        if lower_finite and upper_finite:
            # Fully bounded [a, b]
            lower_f = float(lower)
            upper_f = float(upper)
            sigmoid = 1.0 / (1.0 + np.exp(-t[i]))
            diag_J[i] = (upper_f - lower_f) * sigmoid * (1.0 - sigmoid)
            
        elif lower_finite and not upper_finite:
            # Lower bounded [a, ∞)
            diag_J[i] = np.exp(t[i])
            
        elif not lower_finite and upper_finite:
            # Upper bounded (-∞, b]
            diag_J[i] = -np.exp(t[i])
            
        else:
            # Unbounded (-∞, ∞)
            diag_J[i] = 1.0
    
    return diag_J

def inverse_transform(x, bounds):
    """
    Inverse of transform_to_unbounded: map from bounded space to parameter space.
    
    Used to find t0 that corresponds to a given x0.
    
    Parameters:
        x: ndarray of shape (n,) - point in bounded domain
        bounds: object with .lb and .ub attributes
    
    Returns:
        t: ndarray of shape (n,) - corresponding unbounded parameter
    """
    bounds_low = np.asarray(bounds.lb, dtype=object)
    bounds_high = np.asarray(bounds.ub, dtype=object)
    
    t = np.zeros_like(x, dtype=float)
    
    for i in range(len(x)):
        lower = bounds_low[i]
        upper = bounds_high[i]
        
        lower_finite = lower is not None and np.isfinite(float(lower))
        upper_finite = upper is not None and np.isfinite(float(upper))
        
        if lower_finite and upper_finite:
            # Fully bounded [a, b]: invert sigmoid
            lower_f = float(lower)
            upper_f = float(upper)
            x_normalized = (x[i] - lower_f) / (upper_f - lower_f)
            # Clip to avoid log(0) or log(inf)
            x_normalized = np.clip(x_normalized, 1e-15, 1 - 1e-15)
            t[i] = np.log(x_normalized / (1.0 - x_normalized))
            
        elif lower_finite and not upper_finite:
            # Lower bounded [a, ∞): invert exponential
            lower_f = float(lower)
            t[i] = np.log(np.maximum(x[i] - lower_f, 1e-15))
            
        elif not lower_finite and upper_finite:
            # Upper bounded (-∞, b]: invert negative exponential
            upper_f = float(upper)
            t[i] = np.log(np.maximum(upper_f - x[i], 1e-15))
            
        else:
            # Unbounded (-∞, ∞): identity
            t[i] = x[i]
    
    return t

def compute_hessian_bounded(f, x0, bounds, **hessian_kwargs):
    """
    Compute Hessian of f with respect to bounded/semi-bounded variables.
    
    The function f operates on variables x in a potentially bounded domain.
    We transform to unbounded space, compute the Hessian there, and transform back.
    
    Transformation: x = transform_to_unbounded(t, bounds)
    Chain rule: C_x = J_x^T @ C_t @ J_x --> H_x = J_x⁻¹ @ H_t @ (J_x^T)⁻¹  (where J_x is diagonal Jacobian matrix)
    
    Parameters:
        f: callable(x) - function on bounded domain
        x0: ndarray of shape (n,) - point to evaluate Hessian (in bounded space)
        bounds: object with .lb and .ub attributes
            Each element can be a float or None (for unbounded sides)
        **hessian_kwargs: passed to scipy.differentiate.hessian
            (tolerances, maxiter, order, initial_step, step_factor)
    
    Returns:
        H: ndarray of shape (n, n) - Hessian in bounded space at x0
    """
    # Find t0 that maps to x0
    t0 = inverse_transform(x0, bounds)
    
    # Define wrapped function in unbounded space
    def unbounded_function(t):
        x = transform_to_unbounded(t, bounds)
        return f(x)
    
    # Compute Hessian in unbounded space
    H_unbounded = sd.hessian(unbounded_function, t0, **hessian_kwargs).ddf

    # Compute Jacobian diagonal at t0
    Jm1_diag = 1./jacobian_diagonal(t0, bounds)
    Jm1 = np.diag(Jm1_diag)
    
    # Transform back to bounded space: H_bounded = J⁻¹ @ H_unbounded @ J⁻¹
    H_bounded = Jm1 @ H_unbounded @ Jm1
    
    return H_bounded

def Vectorize_Functional(func, m):
    """ 
    This function vectorizes a functional with m input parameters, 
    by wrapping it inside another function.
    """

    def f_vec(x):

        x = np.array(x, copy=False)
        if x.ndim < 1 or x.shape[0] != m:
            raise ValueError(f"Expected first dim {m}, got {x.shape}")

        # Flatten extra dims
        extra_shape = x.shape[1:]
        p = int(np.prod(extra_shape, dtype=int)) if extra_shape else 1
        x_flat = x.reshape(m, p)

        # apply func to each column
        out = np.empty(p, dtype=float)
        for j in range(p):
            out[j] = func(x_flat[:, j])
        # reshape back to extra_shape
        return out.reshape(extra_shape)

    return f_vec

# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class InferenceResult:
    """
    Container for a single inference result.
    
    Attributes:
        params: Inferred internal parameters dict
        loss: Final loss value
        covariance: Parameter covariance matrix (from Hessian inversion)
        hessian: Hessian matrix
        std_errors: Standard errors (sqrt of diagonal covariance)
        iterations: Number of optimizer iterations
        success: Whether optimization converged
        message: Optimizer message
    """
    
    params: Dict[str, float]
    loss: float
    covariance: Optional[np.ndarray] = None
    hessian: Optional[np.ndarray] = None
    std_errors: Optional[np.ndarray] = None
    iterations: int = 0
    success: bool = False
    message: str = ""
    optimizer_result: Optional[OptimizeResult] = None

    def save(self, filepath: str) -> None:
        """Save to disk using dill."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filepath: str) -> 'InferenceResult':
        """Load from disk using dill."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

@dataclass
class PipelinePass:
    """
    Definition of a single inference pass in the pipeline.
    
    Attributes:
        name: Human-readable pass identifier
        model_class: Model subclass to use in this pass
        ground_truths: List of ground truth arrays (or single array)
        ext_params_list: List of external params per ground truth (or single dict)
        sim_params_list: List of simulation params per ground truth (or single dict)
        product_or_zip: whether or not to do a cartesian product of ext_params_list and sim_params_list
        param_keys_to_infer: Which internal parameters to infer in this pass
        fixed_params: Dict of {param_name: value} for parameters inferred in prior passes
        compose_int_params: Composition function for int_params (via compose)
        optimizer: optimizer class instance to run the inference optimisation
        optimizer_kwargs: arguments for optimizer
    """

    name: str
    model_class: Type
    ground_truths: List[np.ndarray]
    ext_params_list: List[Dict[str, Any]]
    sim_params_list: List[Dict[str, Any]]
    param_keys_to_infer: List[str]
    fixed_params: Dict[str, float] = field(default_factory=dict)
    product_or_zip: str = "product"
    compose_int_params: Optional[Callable] = None
    compose_ext_params: Optional[Callable] = None
    compose_sim_params: Optional[Callable] = None
    optimizer: Callable = None
    optimizer_kwargs: Dict[str, Any] = None

# ============================================================================
# EXTENDED INFERENCE CLASS (Multi-Ground-Truth Support)
# ============================================================================

@dataclass
class Inference:
    """
    Optimization-based parameter inference with multi-ground-truth support.
    
    Given:
    - A Model subclass
    - Single or multiple ground truths (same internal parameters, varying conditions)
    - Custom loss function, with aggregated loss across all ground truths
    
    Do:
    - Infers internal parameters via optimization

    Methods
    - __init__ : constructor
    - _normalize_list: handles single and multiple ground truths similarly
    - objective: 
    - infer inference of internal parameters (int_params)
    - infer_batch: parallel inference over multiple initial guesses        
    - _compute_hessian
    """
    
    model_class: Type
    ground_truths: List[np.ndarray] | np.ndarray
    loss_fn: Callable
    ext_params_list: List[Any] | Any = None
    sim_params_list: List[Any] | Any = None
    optimizer: Callable = field(default_factory=lambda: minimize)
    optimizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    n_jobs: int = -1
    product_or_zip: str = "product"

    # Derived / computed fields (not in __init__)
    result: Optional[OptimizeResult] = field(default=None, init=False, repr=False)
    hessian: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    covariance: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _objective_n_jobs: int = field(default=-1, init=False, repr=False)
    _normalized_ground_truths: List[np.ndarray] = field(default_factory=list, init=False, repr=False)
    _normalized_ext_params: List[Any] = field(default_factory=list, init=False, repr=False)
    _normalized_sim_params: List[Any] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self):
        """Normalize and validate parameters after dataclass initialization."""
        if not self.optimizer_kwargs:
            self.optimizer_kwargs = {}
        
        # Normalize all three parameter groups together
        (
            self._normalized_ground_truths,
            self._normalized_ext_params,
            self._normalized_sim_params,
        ) = self._normalize_lists(
            self.ground_truths,
            self.ext_params_list,
            self.sim_params_list,
            self.product_or_zip,
        )
        
        # Verify consistency
        assert (
            len(self._normalized_ground_truths)
            == len(self._normalized_ext_params)
            == len(self._normalized_sim_params)
        ), (
            f"Mismatched lengths: {len(self._normalized_ground_truths)} GTs vs "
            f"{len(self._normalized_ext_params)} ext vs {len(self._normalized_sim_params)} sim"
        )

    # ========== Parameter Normalization ==========
    """Handle single and multiple ground truths, external and simulation parameters."""

    @staticmethod
    def _normalize_lists(
        ground_truths: List[np.ndarray] | np.ndarray,
        ext_params_list: List[Any] | Any,
        sim_params_list: List[Any] | Any,
        product_or_zip: str = "product",
    ) -> tuple:
        """
        Normalize ground_truths, ext_params_list, and sim_params_list together.
        
        - ext_params_list and sim_params_list are independent sets
        - ground_truths must have length:
            -  len(ext_params_list) * len(sim_params_list) if product, 
            - len(ext_params_list) == len(sim_params_list) if zip.
        - ext_params_list and sim_params_list are converted to Cartesian product order, or a zip
        - product_or_zip: "zip" or "product". Makes a product or a zip from ext_params_list, sim_params_list

        Args:
            ground_truths: Single array or list of arrays
            ext_params_list: Single dict or list of dicts
            sim_params_list: Single dict or list of dicts
        
        Returns:
            Tuple of (ground_truths_list, ext_params_list, sim_params_list)
        
        Raises:
            ValueError: If lengths are incompatible
        """
        # Step 1: Convert to lists (not replicated yet)
        gt_list = ground_truths if isinstance(ground_truths, list) else [ground_truths]
        ext_list = ext_params_list if isinstance(ext_params_list, list) else [ext_params_list]
        sim_list = sim_params_list if isinstance(sim_params_list, list) else [sim_params_list]
        
        # Step 2: Determine expected number of conditions
        if "product" in product_or_zip:
            n_conditions = len(ext_list) * len(sim_list)
        elif "zip" in product_or_zip:
            if abs(len(sim_list) - len(ext_list)) > 0:
                raise ValueError("sim_list and ext_list do not have the same length.")
            n_conditions = len(ext_list)
        else:
            raise ValueError("product_or_zip should either be 'zip' or 'product'")
        
        # Step 3: Verify ground_truths count
        if len(gt_list) != n_conditions:
            raise ValueError(
                f"ground_truths length ({len(gt_list)}) "
                f"must equal ext_params_list * sim_params_list (product) or ext_params_list (zip)"
                f"({len(gt_list)} = {n_conditions})"
            )
        
        # Step 4: Create paired lists in order
        if "product" in product_or_zip:
            paired_params = list(product(ext_list, sim_list))
        elif "zip" in product_or_zip:
            paired_params = list(zip(ext_list, sim_list))
        ext_list = [ep for ep, _ in paired_params]
        sim_list = [sp for _, sp in paired_params]
        
        return gt_list, ext_list, sim_list

    # ========== Loss Computation ==========
    """Compute losses for individual ground truths and aggregated objectives."""

    @staticmethod
    def _compute_single_loss(
        model_class: Type,
        loss_fn: Callable,
        int_params: Dict[str, float],
        ext_params: Any,
        sim_params: Any,
        ground_truth: np.ndarray,
    ) -> float:
        """
        Compute loss for a single ground truth in isolation.
        Static method enables pickling for parallel execution.
        
        Args:
            model_class: Model to instantiate
            loss_fn: Loss function
            int_params: Internal parameters dict
            ext_params: External parameters for this condition
            sim_params: Simulation parameters for this condition
            ground_truth: Target data for this condition
        
        Returns:
            Scalar loss value
        """
        instance = model_class(int_params, ext_params, sim_params)
        predicted = instance.simulate_single()['value']
        loss_i = loss_fn(predicted, ground_truth)
        return loss_i

    def objective(
        self,
        param_vector: np.ndarray,
        param_keys: Tuple[str, ...],
    ) -> float:
        """
        Aggregated objective function with parallelized loss computation.
        Computes losses for all ground truths in parallel, then sums.
        """
        int_params = {key: param_vector[i] for i, key in enumerate(param_keys)}
        
        # Parallel computation of individual losses
        losses = joblib.Parallel(n_jobs=self._objective_n_jobs, backend='loky')(
            joblib.delayed(self._compute_single_loss)(
                self.model_class,
                self.loss_fn,
                int_params,
                ext_params,
                sim_params,
                gt,
            )
            for gt, ext_params, sim_params in zip(
                self._normalized_ground_truths,
                self._normalized_ext_params,
                self._normalized_sim_params,
            )
        )
        
        # return sum(losses)

        valid_losses = [loss for loss in losses if np.isfinite(loss)]
        if not valid_losses:
            return np.inf  # Penalize if all losses are NaN
        return sum(valid_losses)

    # ========== Hessian & Uncertainty ==========
    """Compute Hessian and covariance for parameter uncertainty estimation."""


    def _compute_hessian(self, param_keys: Tuple[str, ...]):
        """
        Compute Hessian numerically via finite differences.
        Invert to estimate parameter covariance (inverse of Fisher information).
        
        Args:
            param_keys: Parameter names corresponding to optimization variables
        """
        n_params = len(param_keys)
        vec_func = Vectorize_Functional( # TODO: can't I vectorize it always?? This could speed up computation.
                lambda x: self.objective(x, param_keys), 
                m = n_params,
            )

        if "bounds" in self.optimizer_kwargs.keys():
            self.hessian = compute_hessian_bounded(
                f = vec_func, 
                x0 = self.result.x, 
                bounds = self.optimizer_kwargs['bounds'],
            )

        else:
            self.hessian = sd.hessian(
                f = vec_func,
                x = self.result.x,
            ).ddf
        

        # Invert Hessian to get covariance
        try:
            self.covariance = np.linalg.inv(self.hessian)
        except np.linalg.LinAlgError:
            print(f"Warning: Hessian singular at optimum for {param_keys}, covariance unavailable.")
            self.covariance = np.ones_like(self.hessian) * np.inf

    # ========== Parameter Inference ==========
    """Run optimization to infer parameters from ground truths."""

    def infer(
        self,
        initial_guess: Dict[str, float],
        objective_n_jobs: int = None,
    ) -> InferenceResult:
        """
        Run optimization to infer parameters from multiple ground truths.
        
        Args:
            initial_guess: Dict like {'x': 2.5, 'y': 1.0}
            objective_n_jobs: Override n_jobs for loss computation within objective.
                            If None, uses self.n_jobs. Set to 1 if combining with
                            parallel infer_batch to avoid nested parallelism.        
        
        Returns:
            InferenceResult with inferred parameters, uncertainties, convergence info
        """
        # Handle nested parallelism: if using infer_batch, set objective_n_jobs=1
        if objective_n_jobs is not None:
            self._objective_n_jobs = objective_n_jobs
        else:
            self._objective_n_jobs = self.n_jobs

        param_keys = tuple(initial_guess.keys())
        x0 = np.array([initial_guess[key] for key in param_keys])

        # Run optimization
        self.result = self.optimizer(
            partial(self.objective, param_keys=param_keys),
            x0,
            **self.optimizer_kwargs
        )
        
        # Compute Hessian and Covariance if optimisation succeeded
        if self.result.success:
            self._compute_hessian(param_keys)
        else: 
            self.hessian = np.zeros(len(param_keys))
            self.covariance = np.ones(len(param_keys)) * np.inf

        # Reconstruct optimal parameters
        optimal_params = {key: self.result.x[i] for i, key in enumerate(param_keys)}
        
        return InferenceResult(
            params=optimal_params,
            loss=self.result.fun,
            covariance=self.covariance,
            hessian=self.hessian,
            std_errors=np.sqrt(np.diag(self.covariance)) if self.covariance is not None else None,
            iterations=self.result.nit,
            success=self.result.success,
            message=self.result.message,
            optimizer_result = self.result,
        )
    
    def infer_batch(
        self,
        initial_guesses: List[Dict[str, float]],
        parallelize_objectives: bool = False,
    ) -> List[InferenceResult]:
        """
        Run inference on multiple initial guesses in parallel.
        
        Each initial guess is optimized independently against the same ground truth(s).
        Useful for robustness checks or global optimization strategies.
        
        Args:
            initial_guesses: List of dicts, e.g., [{'x': 1.0}, {'x': 2.0}]
            parallelize_objectives: If False (default), disables parallelism within
                                each objective call to avoid nested parallelism.
                                Set True if each ground truth is expensive and
                                initial_guesses is small.
        
        Returns:
            List of InferenceResult objects (one per initial guess)
        """

        # Disable objective parallelism when doing batch parallelism...
        objective_n_jobs = self.n_jobs if parallelize_objectives else 1
        # ... unless there is only one guess.
        if len(initial_guesses) == 1:
            objective_n_jobs = self.n_jobs
                
        results = joblib.Parallel(n_jobs=self.n_jobs, backend='loky')(
            joblib.delayed(self.infer)(ig, objective_n_jobs=objective_n_jobs)
            for ig in initial_guesses
        )
        return results
    
# ============================================================================
# MULTI-PASS INFERENCE PIPELINE
# ============================================================================

@dataclass
class InferencePipeline:
    """
    Sequential inference pipeline for parameter estimation.
    
    Implements a multi-pass strategy where each pass infers a subset of parameters
    while fixing results from prior passes via model composition.
    """

    passes: List["PipelinePass"]
    loss_fn: Callable
    n_jobs_per_pass: int = -1
    
    # Derived / computed fields (not in __init__)
    results: List["InferenceResult"] = field(default_factory=list, init=False, repr=False)
    parameter_trajectory: List[Dict[str, float]] = field(default_factory=list, init=False, repr=False)    
    
    # ========== Factory Methods ==========
    """Create InferencePipeline instances with special initialization logic."""

    @classmethod
    def from_factory(
        cls,
        make_pipeline_fn: Callable[..., "InferencePipeline"],
        *args,
        **kwargs,
    ) -> "InferencePipeline":
        """
        Create an InferencePipeline using a model-specific factory function.
        
        The factory function is responsible
        for defining passes, loss function, and other configuration. This method
        simply delegates to it and returns the result.
        
        Args:
            make_pipeline_fn: Callable that returns an InferencePipeline instance.
            *args: Positional arguments passed to make_pipeline_fn.
            **kwargs: Keyword arguments passed to make_pipeline_fn.
        
        Returns:
            Configured InferencePipeline instance.
        
        Example:
            pipeline = InferencePipeline.from_factory(
                custom_make_inference_pipeline,
                ground_truths=[gt1, gt2],
                n_jobs_per_pass=-1,
            )
        """
        return make_pipeline_fn(*args, **kwargs)

    # ========== Pipeline Execution ==========
    """Run sequential inference passes with parameter accumulation."""

    def run(
        self,
        initial_guesses_per_pass: List[List[Dict[str, float]]],
        verbose = True,
    ) -> List[InferenceResult]:
        """
        Execute the pipeline sequentially: Pass 1 → Pass 2 → ...
        
        Results from each pass inform the next via fixed parameters.
        
        Args:
            initial_guesses_per_pass: List of initial guess lists, one per pass.
                        E.g., [[{'x': 1.0, 'y': 2.0}], [{'z': 0.5}]]
        
        Returns:
            List of InferenceResult objects (one per pass)

        Raises:
            AssertionError: If initial_guesses_per_pass length doesn't match passes.
        """

        assert len(initial_guesses_per_pass) == len(self.passes), (
            f"Must provide initial guesses for each pass: "
            f"got {len(initial_guesses_per_pass)}, expected {len(self.passes)}"
        )
        accumulated_params = {}  # Parameters inferred so far
        
        for pass_idx, (pass_def, initial_guesses) in enumerate( # This has to be computed sequentially.
            zip(self.passes, initial_guesses_per_pass)
        ):

            if verbose:
                self._print_pass_header(pass_idx, pass_def, accumulated_params)

            # Build model with fixed parameters from prior passes
            model_for_pass = self._build_pass_model(
                pass_def,
                fixed_params=accumulated_params, # If there are fixed parameters, compose the model to enforce them
            )
            
            # Create and run Inference for this pass
            inference = Inference(
                model_class=model_for_pass,
                ground_truths=pass_def.ground_truths,
                loss_fn=self.loss_fn,
                ext_params_list=pass_def.ext_params_list,
                sim_params_list=pass_def.sim_params_list,
                optimizer=pass_def.optimizer,
                optimizer_kwargs=pass_def.optimizer_kwargs,
                n_jobs=self.n_jobs_per_pass,
                product_or_zip=pass_def.product_or_zip,
            )

            # Run inference on all initial guesses for this pass
            if verbose:
                print(f"Running {len(initial_guesses)} inference(s) in parallel...")
            
            pass_results = inference.infer_batch(initial_guesses)

            # Select best result by loss TODO See if this is what I want to do or not.
            best_result = self._select_best_result(pass_results)
            self.results.append(best_result)    
            
            if verbose:
                self._print_pass_results(pass_idx, pass_def, best_result, pass_results)
            
            # Accumulate parameters for next pass
            accumulated_params.update(best_result.params)
            self.parameter_trajectory.append(accumulated_params.copy())
            
            if verbose:
                print(f"  Accumulated parameters for next pass: {accumulated_params}\n")
        
        return self.results

    # ========== Model Composition ==========
    """Build pass-specific models with fixed parameters from prior passes."""

    def _build_pass_model(
        self,
        pass_def: PipelinePass,
        fixed_params: Dict[str, float],
    ) -> Type:
        """
        Build the model for a single pass, enforcing fixed parameters from prior passes.
        
        Pass 1 returns the base model unchanged. Pass 2+ wraps it via compose
        to merge fixed parameters with newly inferred ones.

        Args:
            pass_def: PipelinePass definition for this pass
            fixed_params: Parameters inferred in prior passes (to hold constant)
        
        Returns:
            Model class (potentially wrapped via compose)
        """

        if not fixed_params:
            # Pass 1: No composition needed
            return pass_def.model_class
        
        # Pass 2+: Compose to enforce fixed parameters
        def compose_int_params_with_fixed(int_params, ext_params, sim_params):
            """Merge fixed parameters from prior passes with newly inferred ones."""
            merged = {**fixed_params, **int_params}
            return merged
        
        composed = pass_def.model_class.compose(
            compose_int_params=compose_int_params_with_fixed,
            compose_ext_params=pass_def.compose_ext_params,
            compose_sim_params=pass_def.compose_sim_params,
        )
        
        return composed
    
    # ========== Result Selection & Logging ==========
    """Select best results and print execution summaries."""

    @staticmethod
    def _select_best_result(pass_results: List["InferenceResult"]) -> "InferenceResult":
        """
        Select result with lowest finite loss, penalizing NaN or Inf losses.
        """
        return min(
            pass_results,
            key=lambda r: r.loss if np.isfinite(r.loss) else np.inf,
        )

    @staticmethod
    def _print_pass_header(
        pass_idx: int,
        pass_def: "PipelinePass",
        accumulated_params: Dict[str, float],
    ):
        """Print header for a pipeline pass."""
        print(f"\n{'='*60}")
        print(f"Pipeline Pass {pass_idx + 1}: {pass_def.name}")
        print(f"Inferring: {pass_def.param_keys_to_infer}")
        print(f"Fixed from prior passes: {list(accumulated_params.keys())}")
        print(f"{'='*60}")
    
    @staticmethod
    def _print_pass_results(
        pass_idx: int,
        pass_def: "PipelinePass",
        best_result: "InferenceResult",
        all_results: List["InferenceResult"],
    ):
        """Print results summary for a pipeline pass."""
        print(f"\nPass {pass_idx + 1} Results:")
        print(f"  Best loss: {best_result.loss:.8e}")
        print(f"  Best parameters: {best_result.params}")
        print(f"  Converged: {best_result.success}")
        
        if best_result.std_errors is not None:
            std_dict = dict(zip(pass_def.param_keys_to_infer, best_result.std_errors))
            print(f"  Standard errors: {std_dict}")
        
        losses = [r.loss for r in all_results]
        print(f"  Loss range across {len(all_results)} runs: [{min(losses):.8e}, {max(losses):.8e}]")
        if len(all_results) > 1:
            print(f"  Loss std dev: {np.std(losses):.8e}")

    # ========== Analysis & Reporting ==========
    """Extract and format results across passes."""

    def get_parameter_trajectory(self) -> Dict[str, List[Optional[float]]]:
        """
        Extract parameter trajectory across all passes.
        
        Parameters inferred in Pass 1 appear in all subsequent passes.
        Parameters inferred in Pass N appear as None in earlier passes.
        
        Returns:
            Dict like {'x': [1.0, 1.05], 'y': [2.0, 2.01], 'z': [None, 0.5]}.
        """
        if not self.parameter_trajectory:
            return {}
        
        # Collect all unique parameter names
        all_param_names = set()
        for params in self.parameter_trajectory:
            all_param_names.update(params.keys())
        
        # Build trajectory with None for missing parameters
        trajectory = {name: [] for name in all_param_names}
        for params_dict in self.parameter_trajectory:
            for name in all_param_names:
                trajectory[name].append(params_dict.get(name, None))
        
        return trajectory
    
    def summary(self) -> str:
        """
        Generate a formatted summary of pipeline execution.
        
        Includes losses, parameters, standard errors, and convergence status per pass.
        
        Returns:
            Human-readable summary string.
        """
        if not self.results:
            return "Pipeline not yet executed."
        
        lines = [
            "\n" + "="*80,
            "INFERENCE PIPELINE SUMMARY",
            "="*80,
        ]
        
        for pass_idx, (pass_def, result) in enumerate(zip(self.passes, self.results)):
            lines.append(f"\nPass {pass_idx + 1}: {pass_def.name}")
            lines.append(f"  Model: {pass_def.model_class.__name__}")
            lines.append(f"  Final loss: {result.loss:.8e}")
            lines.append(f"  Success: {result.success}")
            lines.append(f"  Iterations: {result.iterations}")
            lines.append(f"  Parameters:")
            
            for param_name, param_value in result.params.items():
                std_err_str = ""
                if result.std_errors is not None:
                    param_keys = tuple(pass_def.param_keys_to_infer)
                    if param_name in param_keys:
                        idx = param_keys.index(param_name)
                        std_err = result.std_errors[idx]
                        std_err_str = f" ± {std_err:.6e}"
                
                lines.append(f"    {param_name}: {param_value:.6e}{std_err_str}")
        
        lines.append("\n" + "="*80)
        return "\n".join(lines)
    
