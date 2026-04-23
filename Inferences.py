from typing import Any, Callable, Dict, List, Tuple, Optional, Type
from functools import partial
from dataclasses import dataclass, field
import numpy as np
import joblib
from scipy.optimize import minimize, OptimizeResult
import scipy.differentiate as sd
from itertools import product
from Models import compose_model

# ============================================================================
# FUNCTIONS
# ============================================================================


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
        param_keys_to_infer: Which internal parameters to infer in this pass
        fixed_params: Dict of {param_name: value} for parameters inferred in prior passes
        compose_int_params: Composition function for int_params (via compose_model)
    """
    name: str
    model_class: Type
    ground_truths: List[np.ndarray]
    ext_params_list: List[Dict[str, Any]]
    sim_params_list: List[Dict[str, Any]]
    param_keys_to_infer: List[str]
    fixed_params: Dict[str, float] = field(default_factory=dict)
    compose_int_params: Optional[Callable] = None
    compose_ext_params: Optional[Callable] = None
    compose_sim_params: Optional[Callable] = None


# ============================================================================
# EXTENDED INFERENCE CLASS (Multi-Ground-Truth Support)
# ============================================================================

class Inference:
    """
    Optimization-based parameter inference with multi-ground-truth support.
    
    Given:
    - A Model subclass (e.g., Square)
    - Single or multiple ground truths (same internal parameters, varying conditions)
    - Custom loss function, with aggregated loss across all ground truths
    
    Methods
    - __init__ : constructor
    - _normalize_list: handles single and multiple ground truths similarly
    - objective: 
    - infer inference of internal parameters (int_params)
    - infer_batch: parallel inference over multiple initial guesses        
    - _compute_hessian: 
    """
    
    def __init__(
        self,
        model_class: Type,
        ground_truths: List[np.ndarray] | np.ndarray,
        loss_fn: Callable,
        ext_params_list: List[Any] | Any = None,
        sim_params_list: List[Any] | Any = None,
        optimizer: Callable = None,
        optimizer_kwargs: Dict[str, Any] = None,
        n_jobs: int = -1,  # Parallelization across initial guesses AND objective calls
    ):
        """
        Args:
            model_class: Model subclass (e.g., Square, ReducedModel)
            ground_truths: Single array or list of arrays (multiple conditions)
            loss_fn: Callable(predicted, ground_truth) -> scalar loss
                    Will be called once per ground truth; results are summed
            ext_params_list: Single dict or list of dicts (one per ground truth)
            sim_params_list: Single dict or list of dicts (one per ground truth)
            optimizer: Scipy optimizer (default: minimize with L-BFGS-B)
            optimizer_kwargs: Dict of optimizer settings
            n_jobs: Number of parallel jobs for initial guess batches (-1 = all cores)
        """
        self.model_class = model_class
        self.loss_fn = loss_fn
        self.optimizer = optimizer or minimize
        self.optimizer_kwargs = optimizer_kwargs
        self.n_jobs = n_jobs  # Separate from batch parallelism
        self._objective_n_jobs = n_jobs  # Separate from batch parallelism
        self.result: Optional[OptimizeResult] = None
        self.hessian: Optional[np.ndarray] = None
        self.covariance: Optional[np.ndarray] = None

        # Normalize all three together
        self.ground_truths, self.ext_params_list, self.sim_params_list = self._normalize_lists(
            ground_truths,
            ext_params_list,
            sim_params_list,
        )

        # Verify consistency
        assert len(self.ground_truths) == len(self.ext_params_list) == len(self.sim_params_list), (
            f"Mismatched lengths: {len(self.ground_truths)} GTs vs "
            f"{len(self.ext_params_list)} ext vs {len(self.sim_params_list)} sim"
        )

    @staticmethod
    def _normalize_lists(
        ground_truths: List[np.ndarray] | np.ndarray,
        ext_params_list: List[Any] | Any,
        sim_params_list: List[Any] | Any,
    ) -> tuple:
        """
        Normalize ground_truths, ext_params_list, and sim_params_list together.
        
        - ext_params_list and sim_params_list are independent sets
        - ground_truths must have length = len(ext_params_list) × len(sim_params_list)
        - ext_params_list and sim_params_list are converted to Cartesian product order
        
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
        
        # ext_list and sim_list are independent; generate all combinations
        n_conditions = len(ext_list) * len(sim_list)
        
        # Verify ground_truths count matches
        if len(gt_list) != n_conditions:
            raise ValueError(
                f"use_product=True: ground_truths length ({len(gt_list)}) "
                f"must equal ext_params_list × sim_params_list "
                f"({len(ext_list)} × {len(sim_list)} = {n_conditions})"
            )
        
        # Create paired lists from product (in order)
        paired_params = list(product(ext_list, sim_list))
        ext_list = [ep for ep, _ in paired_params]
        sim_list = [sp for _, sp in paired_params]
        
        return gt_list, ext_list, sim_list

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
                self.ground_truths,
                self.ext_params_list,
                self.sim_params_list,
            )
        )
        
        return sum(losses)    
    
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
        
        # Compute Hessian and covariance
        self._compute_hessian(param_keys)
        
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
            # TODO: add optimizer result directly?
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
        # Disable objective parallelism when doing batch parallelism
        objective_n_jobs = self.n_jobs if parallelize_objectives else 1
                
        results = joblib.Parallel(n_jobs=self.n_jobs, backend='loky')(
            joblib.delayed(self.infer)(ig, objective_n_jobs=objective_n_jobs)
            for ig in initial_guesses
        )
        return results
    
    def _compute_hessian(self, param_keys: Tuple[str, ...]):
        """
        Compute Hessian numerically via finite differences.
        Invert to estimate parameter covariance (inverse of Fisher information).
        
        Args:
            param_keys: Parameter names corresponding to optimization variables
        """
        n_params = len(param_keys)

        self.hessian = sd.hessian(
                f = Vectorize_Functional(
                    lambda x: self.objective(x, param_keys), 
                    m = n_params,
                ), 
                x = self.result.x).ddf
        

        # Invert Hessian to get covariance (approximation of parameter uncertainty)
        try:
            self.covariance = np.linalg.inv(self.hessian)
        except np.linalg.LinAlgError:
            print(f"Warning: Hessian singular at optimum for {param_keys}, covariance unavailable.")
            self.covariance = None

# ============================================================================
# MULTI-PASS INFERENCE PIPELINE
# ============================================================================

class InferencePipeline:
    """
    Sequential inference pipeline for parameter estimation.
    
    Implements a multi-pass strategy:
    - Pass 1: Infer low-dim subset with reduced model
    - Pass 2: Fix Pass 1 results, infer additional parameters
    - etc.
    
    Each pass uses `compose_model` to enforce fixed parameters from prior passes.
    """
    
    def __init__(
        self,
        passes: List[PipelinePass],
        loss_fn: Callable,
        n_jobs_per_pass: int = -1,
        optimizer: Callable = None, # TODO: should this be here or in a pass?
        optimizer_kwargs: Dict[str, Any] = None, # TODO: should this be here or in a pass?
    ):
        """
        Args:
            passes: List of PipelinePass instances (in sequential order)
            loss_fn: Shared loss function across all passes
            n_jobs_per_pass: Parallelization for initial guesses within each pass
        """
        self.passes = passes
        self.loss_fn = loss_fn
        self.n_jobs_per_pass = n_jobs_per_pass
        self.results: List[InferenceResult] = []
        self.parameter_trajectory: List[Dict[str, float]] = []
        self.optimizer = optimizer # TODO: should this be in the "passes" data?
        self.optimizer_kwargs = optimizer_kwargs # TODO: should this be in the "passes" data?
    
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
        """
        assert len(initial_guesses_per_pass) == len(self.passes), \
            f"Must provide initial guesses for each pass"
        
        accumulated_params = {}  # Parameters inferred so far
        
        for pass_idx, (pass_def, initial_guesses) in enumerate(
            zip(self.passes, initial_guesses_per_pass)
        ):
            print(f"\n{'='*60}")
            print(f"Pipeline Pass {pass_idx + 1}: {pass_def.name}")
            print(f"Inferring: {pass_def.param_keys_to_infer}")
            print(f"Fixed from prior passes: {list(accumulated_params.keys())}")
            print(f"{'='*60}")
            
            # Build the model for this pass
            model_for_pass = self._build_pass_model(
                pass_def,
                fixed_params=accumulated_params # If there are fixed parameters, compose the model to enforce them
            )
            
            # Create Inference instance for this pass
            inference = Inference(
                model_class=model_for_pass,
                ground_truths=pass_def.ground_truths,
                loss_fn=self.loss_fn,
                ext_params_list=pass_def.ext_params_list,
                sim_params_list=pass_def.sim_params_list,
                optimizer=self.optimizer,
                optimizer_kwargs=self.optimizer_kwargs,
                n_jobs=self.n_jobs_per_pass,
            )

            # Run inference on all initial guesses for this pass
            if verbose:
                print(f"Running {len(initial_guesses)} inference(s) in parallel...")
            pass_results = inference.infer_batch(initial_guesses)
            
            # ===== SELECT BEST RESULT =====
            # Choose the result with the lowest loss TODO See if this is what I want to do or not.
            best_result = min(pass_results, key=lambda r: r.loss)
            
            if verbose:
                print(f"\nPass {pass_idx + 1} Results:")
                print(f"  Best loss: {best_result.loss:.8e}")
                print(f"  Best parameters: {best_result.params}")
                print(f"  Converged: {best_result.success}")
                if best_result.std_errors is not None:
                    print(f"  Standard errors: {dict(zip(pass_def.param_keys_to_infer, best_result.std_errors))}")
                
                # Compare all results for robustness
                losses = [r.loss for r in pass_results]
                print(f"  Loss range across {len(pass_results)} runs: [{min(losses):.8e}, {max(losses):.8e}]")
                if len(pass_results) > 1:
                    print(f"  Loss std dev: {np.std(losses):.8e}")
            
            # Store this pass's best result
            self.results.append(best_result)
            
            # Update accumulated parameters: add newly inferred params
            accumulated_params.update(best_result.params) # TODO: Warning! In pass 1 I need to constrain viscosity to 0, but in pass 2 no.
            self.parameter_trajectory.append(accumulated_params.copy())
            
            if verbose:
                print(f"  Accumulated parameters for next pass: {accumulated_params}")
        
        return self.results

    def _build_pass_model(
        self,
        pass_def: PipelinePass,
        fixed_params: Dict[str, float],
    ) -> Type:
        """
        Build the model for a single pass, enforcing fixed parameters from prior passes.
        
        If this is Pass 1 (no fixed params), return the model as-is.
        If this is Pass 2+ (fixed params exist), compose the model to enforce them.
        
        Args:
            pass_def: PipelinePass definition for this pass
            fixed_params: Parameters inferred in prior passes (to be held constant)
        
        Returns:
            Model class (potentially wrapped via compose_model)
        """
        if not fixed_params:
            # Pass 1: No composition needed
            return pass_def.model_class
        
        # Pass 2+: Compose model to fix parameters from prior passes
        # The composition function will merge fixed_params with newly inferred ones
        def compose_int_params_with_fixed(int_params, ext_params, sim_params):
            """
            Merge fixed parameters (from prior passes) with newly inferred ones.
            
            int_params contains only the parameters being inferred in this pass.
            We extend it with the fixed parameters from prior passes.
            """
            merged = {**fixed_params, **int_params}
            return merged
        
        composed = compose_model(
            pass_def.model_class,
            compose_int_params=compose_int_params_with_fixed,
            compose_ext_params=pass_def.compose_ext_params,
            compose_sim_params=pass_def.compose_sim_params,
        )
        
        return composed
    
    def get_parameter_trajectory(self) -> Dict[str, List[float]]:
        """
        Extract parameter trajectory across all passes.
        
        Returns:
            Dict like {'x': [1.0, 1.05], 'y': [2.0, 2.01], 'z': [None, 0.5]}
            where None indicates a parameter not yet inferred in that pass.
        """
        if not self.parameter_trajectory:
            return {}
        
        # Get all unique parameter names across all passes
        all_param_names = set()
        for params in self.parameter_trajectory:
            all_param_names.update(params.keys())
        
        # Build trajectory dict
        trajectory = {name: [] for name in all_param_names}
        
        for step, params_dict in enumerate(self.parameter_trajectory):
            for name in all_param_names:
                trajectory[name].append(params_dict.get(name, None))
        
        return trajectory
    
    def summary(self) -> str:
        """
        Generate a human-readable summary of the pipeline execution.
        
        Returns:
            Formatted string with losses, parameters, and uncertainties per pass
        """
        if not self.results:
            return "Pipeline not yet executed."
        
        summary_lines = [
            "\n" + "="*80,
            "INFERENCE PIPELINE SUMMARY",
            "="*80,
        ]
        
        for pass_idx, (pass_def, result) in enumerate(zip(self.passes, self.results)):
            summary_lines.append(f"\nPass {pass_idx + 1}: {pass_def.name}")
            summary_lines.append(f"  Model: {pass_def.model_class.__name__}")
            summary_lines.append(f"  Final loss: {result.loss:.8e}")
            summary_lines.append(f"  Success: {result.success}")
            summary_lines.append(f"  Iterations: {result.iterations}")
            summary_lines.append(f"  Parameters:")
            
            for param_name, param_value in result.params.items():
                if result.std_errors is not None:
                    param_keys = tuple(pass_def.param_keys_to_infer)
                    idx = param_keys.index(param_name) if param_name in param_keys else None
                    if idx is not None:
                        std_err = result.std_errors[idx]
                        summary_lines.append(
                            f"    {param_name}: {param_value:.6e} ± {std_err:.6e}"
                        )
                    else:
                        summary_lines.append(f"    {param_name}: {param_value:.6e}")
                else:
                    summary_lines.append(f"    {param_name}: {param_value:.6e}")
        
        summary_lines.append("\n" + "="*80 + "\n")
        return "\n".join(summary_lines)    
            
# ============================================================================
# USAGE EXAMPLE WITH YOUR BASINHOPPING OPTIMIZER ON VISCOELASTIC FILAMENT
# ============================================================================

if __name__ == "__main__":
    """
    Example: Two-pass inference on ViscoElasticFilament model.
    
    Pass 1: Infer Sp4 only (elastic filament)
    Pass 2: Infer tau_b (viscoelastic filament with Sp4 fixed from Pass 1)
    """
    
    # ===== Define ground truth data ===== # TODO: make_ground_truth(...)
    ground_truth_data = np.array([...])  # Your measured data
    ground_truth_ext_params = {'flow_rate': 0.5}
    ground_truth_sim_params = {'dt': 0.01, 'duration': 10.0}
    
    # ===== Define MSE loss =====
    def mse_loss(predicted, ground_truth):
        return np.mean((predicted - ground_truth) ** 2)
    
    # ===== Define pipeline passes =====

    # Pass 1: Reduced model, infer Sp4 only
    pass_1 = PipelinePass(
        name="Sp4 Inference (Elastic Model)",
        model_class=ViscoElasticFilament_Models.ReducedModel_Sp4Only,  # Simple model # TODO: Modify this
        ground_truths=[ground_truth_data],
        ext_params_list=[ground_truth_ext_params],
        sim_params_list=[ground_truth_sim_params],
        param_keys_to_infer=['Sp4'],
        fixed_params={},
    )
    
    # Pass 2: Full model, infer tau_b (with Sp4 fixed from Pass 1)
    pass_2 = PipelinePass(
        name="Ka Inference (ViscoElastic Model, Elasticity fixed)",
        model_class=ViscoElasticFilament_Models.FullModel, # TODO: modify this as it does not exist
        ground_truths=[ground_truth_data],
        ext_params_list=[ground_truth_ext_params],
        sim_params_list=[ground_truth_sim_params],
        param_keys_to_infer=['tau_b'],
        fixed_params={},  # Will be auto-filled from Pass 1 result
    )
    
    # ===== Create pipeline =====
    pipeline = InferencePipeline(
        passes=[pass_1, pass_2],
        loss_fn=mse_loss,
        optimizer=basinhopping_optimizer,
        optimizer_kwargs={
            'bounds': Bounds(lb=[1e-6], ub=[np.inf]),
            'minimum_gradient': False,
            'minimum_hessian': False,
            'local_minimizer_kwargs': {
                'method': 'L-BFGS-B',
                'jac': '3-point',
                'options': {
                    'disp': True,
                    'ftol': 1e-8,
                    'gtol': 1e-8,
                    'eps': 1e-8,
                    'finite_diff_rel_step': 1e-6,
                },
            },
            'global_minimizer_kwargs': {
                'niter': 9,
                'T': 0,
                'stepsize': 5,
                'tol': 1e-10,
            }
        },
        n_jobs_per_pass=-1,  # Use all cores within each pass
    )
    
    # ===== Define initial guesses per pass =====
    # Each initial guess is a list of dicts, one dict per initial condition to try
    initial_guesses = [
        # Pass 1: Try 3 different initial guesses for Sp4
        [
            {'Sp4': 0.5},
            {'Sp4': 2.5},
            {'Sp4': 5.0},
        ],
        # Pass 2: Try 2 initial guesses for tau_b
        [
            {'tau_b': 0},
            {'tau_b': 0.1},
            {'tau_b': 1.0},
        ],
    ]
    
    # ===== Run pipeline =====
    results = pipeline.run(
        initial_guesses_per_pass=initial_guesses,
        verbose=True,
    )
    
    # ===== Print summary =====
    print(pipeline.summary())
    
    # ===== Extract final parameters =====
    final_params = pipeline.parameter_trajectory[-1]
    print(f"\nFinal inferred parameters: {final_params}")
    
    # ===== Get parameter trajectory =====
    trajectory = pipeline.get_parameter_trajectory()
    print(f"Parameter trajectory: {trajectory}")
    
    # ===== Access individual results =====
    pass_1_result = pipeline.results[0]
    pass_2_result = pipeline.results[1]
    
    print(f"\nPass 1 - Sp4: {pass_1_result.params['Sp4']:.6e}")
    print(f"Pass 2 - Ka: {pass_2_result.params['tau_b']:.6e}")