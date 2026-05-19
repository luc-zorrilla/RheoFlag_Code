from Models import Model, compose_model
from ViscoElasticFilament_Models import (
    StraightLine, 
    ViscoElasticFilament, 
    ViscoElasticFilament_FlowParams, 
    ViscoElasticFilament_FlowParams_ScalarBending,
)
from Inferences import Inference, InferencePipeline, PipelinePass, InferenceResult

from scipy.optimize import minimize, Bounds
from _basinhopping_mod import *
import numpy as np
from itertools import product
from pathlib import Path

import json
from dataclasses import asdict, dataclass
from typing import Optional, List, Dict, Any, Callable
import pytest
import copy

### Optimization schemes

class RandomDisplacementBounds:
    """random displacement with bounds:  see: https://stackoverflow.com/a/21967888/2320035
        Modified: dropped acceptance-rejection sampling
    """
    def __init__(self, bounds, stepsize=0.5):
        self.bounds = bounds
        self.stepsize = stepsize

    def __call__(self, x):
        """take a random step but ensure the new position is within the bounds """

        sl, sb = self.bounds.residual(x) # Lower and upper residual between x and the bounds
        min_step = np.maximum(-sl, -self.stepsize)
        max_step = np.minimum(sb, self.stepsize)

        random_step = np.random.uniform(low=min_step, high=max_step, size=x.shape)
        xnew = x + random_step

        return xnew

def basinhopping_optimizer(
    objective,
    x0,
    bounds=None,
    local_minimizer_kwargs: Dict[str, Any] = None,
    global_minimizer_kwargs: Dict[str, Any] = None,
):
    """
    Basin-hopping with local minimization (L-BFGS-B by default).
    
    Combines global optimization (basin-hopping) with local optimization (L-BFGS-B),
    using bounded random displacement for exploration and optional Jacobian and Hessian computation.
    
    Args (Individual):
        objective: Callable(flat_array) -> scalar loss
        x0: Initial guess (flat array)
        bounds: Bounds object (scipy.optimize.Bounds or custom with .residual() method), shared between global and local optimizers
    
    Args (Local Minimizer):
        local_minimizer_kwargs: Dict with L-BFGS-B configuration by default:
            {
                'method': 'L-BFGS-B', # Local optimization method
                'jac': '3-point',  # Jacobian specification
                'options':{
                    'ftol':1e-8, 
                    'gtol':1e-8, 
                    'eps': 1e-8, 
                    'finite_diff_rel_step':1e-6,
                    }
                ... (other local optimizer options) # 'maxiter': 1000,
            }
    
    Args (Global Minimizer):
        global_minimizer_kwargs: Dict with basin-hopping configuration:
            {
                'niter': 9,  # Basin-hopping iterations
                'stepsize': 5,  # Maximum step size for perturbations
                'T': 0,  # Temperature for Metropolis acceptance
                'tol': 1e-10,  # Early stopping tolerance --> Not in basinhopping function
            }
    
    Returns:
        OptimizeResult with:
        - x: Optimal parameters
        - fun: Final loss value
        - success: Convergence success flag
        - nit: Number of basin-hopping iterations
        - X_local, F_local: Local optimization trajectories
        - X_global, F_global, accept_global: Global search trajectory
    """
    
    # --- Set defaults ---
    local_minimizer_kwargs = local_minimizer_kwargs or {
        'method': 'L-BFGS-B', # Local optimization method
        'jac': '3-point',  # Jacobian specification
        'options':{
            'ftol':1e-8,  # Functional tolerance for local minimizer
            'gtol':1e-8,  # Gradient tolerance for local minimizer
            'eps': 1e-8,  # ?
            'finite_diff_rel_step':1e-6, # ?
        }
    }

    global_minimizer_kwargs = global_minimizer_kwargs or {
        'niter': 9,  # Basin-hopping iterations
        'stepsize': 5,  # Maximum step size for perturbations
        'T': 0,  # Temperature for Metropolis acceptance
        'tol': 1e-10,  # Early stopping tolerance
    }
    
    # --- Extract global minimizer parameters ---
    niter = global_minimizer_kwargs.pop('niter', 9)
    T = global_minimizer_kwargs.pop('T', 0)
    stepsize = global_minimizer_kwargs.pop('stepsize', 5)
    tol = global_minimizer_kwargs.pop('tol', 1e-10)
    
    # --- Extract local minimizer parameters ---
    method = local_minimizer_kwargs.pop('method', "L-BFGS-B")
    jac = local_minimizer_kwargs.pop('jac', '3-point')
    options = local_minimizer_kwargs.pop(
        'options', {
            'ftol':1e-8, 
            'gtol':1e-8, 
            'eps': 1e-8, 
            'finite_diff_rel_step':1e-6,
        })
    
    X_local = []
    F_local = []
    X_global = []
    F_global = []
    accept_global = []
    
    # --- Local minimization wrapper ---
    def wrapped_minimize(fun, x0, args=(), method=None, jac=None, hess=None, 
                        hessp=None, bounds=None, constraints=(), tol=None, 
                        callback=None, options=None):
        """Wrapper to capture starting point of each local minimization."""
        
        x = copy.deepcopy(x0)
        f = fun(x)
        X_local.append([x])
        F_local.append([f])
        
        result_minimize = minimize(
            fun, x0, args=args, method=method, jac=jac, hess=hess, hessp=hessp,
            bounds=bounds, constraints=constraints, tol=tol, callback=callback,
            options=options,
        )

        print(result_minimize)
        return result_minimize
    
    # --- Callback for local minimizer (L-BFGS-B) ---
    def local_callback_function(*, intermediate_result):
        """Capture each iteration of L-BFGS-B."""
        x_loc = copy.deepcopy(intermediate_result.x)
        f_loc = copy.deepcopy(intermediate_result.fun)
        X_local[-1].append(x_loc)
        F_local[-1].append(f_loc)
    
    # --- Callback for global minimizer (basin-hopping) ---
    def global_callback_function(x, f, accept):
        """Capture each basin-hopping step."""
        X_global.append(copy.deepcopy(x))
        F_global.append(copy.deepcopy(f))
        accept_global.append(accept)
        
        # Early stopping if tolerance reached
        if f < tol:
            return True
        return False
    
    # --- Local minimizer full configuration ---

    local_minimizer_kwargs.update({
        'method':method,
        'jac':jac,
        'options': options,        
        "bounds": bounds,
        "callback": local_callback_function,
    })
    
    # --- Global minimizer full configuration ---
    bounded_step = RandomDisplacementBounds(bounds = bounds, stepsize=stepsize)

    global_minimizer_kwargs.update({
        'niter': niter,
        'stepsize':stepsize,
        'T':T,
        'callback': global_callback_function,
        'minimize_wrapper': wrapped_minimize,
        'take_step': bounded_step,
    })
    
    ret = basinhopping(
        func=objective,
        x0=x0,
        minimizer_kwargs=local_minimizer_kwargs,
        **global_minimizer_kwargs,
    )
    
    x_final = ret.x
    
    # --- Attach optimization history ---
    ret.X_local = X_local
    ret.F_local = F_local
    ret.X_global = X_global
    ret.F_global = F_global
    ret.accept_global = accept_global
    
    return ret

def rel_mse_loss_fn() -> Callable:
    """
    Define Relative Mean Square Error loss function.
    Returns np.inf if prediction is None (failed simulation).
    """
    def loss_function(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
        if predicted is None:
            return np.inf
        # Flatten arrays
        pred_flat = np.asarray(predicted).flatten()
        truth_flat = np.asarray(ground_truth).flatten()
        
        # Truncate to match lengths
        min_len = min(len(pred_flat), len(truth_flat))
        return np.linalg.norm(pred_flat[:min_len] - truth_flat[:min_len])**2 / np.linalg.norm(truth_flat[:min_len])**2
    
    return loss_function

def make_ground_truth_int_params(
    Sp4 = 1e0,
    N = 10,
    k0 = 1e13,
    bool_EI = True,
    Beta = 0,
    tau_b = 0,
    tau_s = 0,
    gamma = 2,
    n_L = [0,0],
    m_L = 0,
    X_0 = StraightLine(10),
):

    assert X_0.shape[0] == N+2, f"{X_0.shape[0]} is not of shape {N+2}"
    return {
        'Sp4': Sp4,           # Ground truth to recover
        'N': N,            
        'k0': k0,            
        'bool_EI': bool_EI,      
        'Beta': Beta,           
        'tau_b': tau_b,           
        'tau_s': tau_s,       
        'gamma': gamma,         
        'n_L': n_L,            
        'm_L': m_L,             
        'X_0': X_0,  # Initial state
    }

def make_ground_truth_ext_params(
    Lambdas = [[0,0]]*10,
    Zetas = [0]*10,
    A = 1e-6,
    w0 = 0, # Static flow
    psi = np.pi/2,    
):
    assert abs(len(Lambdas) - len(Zetas)) == 0, f"{abs(len(Lambdas) - len(Zetas))} != 0"
    return {
        "Lambdas": Lambdas,
        "Zetas": Zetas,
        "A":A,
        "w0":w0,
        "psi":psi,            
    }

def make_ground_truth_sim_params(
    T_span = (1e6, 2e6),
    T_eval = np.linspace(1e6, 2e6, int(1e0)),
    method = "hybr",
    T_sim_max = 300,
):
    return {
        "T_span": T_span,
        "T_eval": T_eval,
        "method": method,
        "T_sim_max": T_sim_max,            
    }

def make_ground_truth_data_list(
    ground_truth_int_params,
    ground_truth_ext_params_list,
    ground_truth_sim_params_list,
    product_or_zip, # "product" or "zip"
):
    """
    Generate ground truth data using the 
    ViscoElasticFilament_FlowParams_ScalarBending model with known parameters
    across multiple external and simulation parameter sets.
    
    Returns a list of ground truth arrays (one per condition).
    """    
    ground_truths = []
    
    for ext_params, sim_params in eval(product_or_zip)(
        ground_truth_ext_params_list,
        ground_truth_sim_params_list
    ):
        # Instantiate model with ground truth internal parameters
        instance = ViscoElasticFilament_FlowParams_ScalarBending( 
            ground_truth_int_params,
            ext_params,
            sim_params
        )
        
        # Simulate to generate ground truth
        sim_result = instance.simulate_single()
        gt_data = sim_result['value']
        
        ground_truths.append(gt_data)
    
    return ground_truths

def model_params_only_flow(
    ground_truth_int_params,
    param_keys_to_infer,
):
    """
    Create a composed model for ViscoElasticFilament_FlowParams that
    only varies params in param_keys_to_infer
    
    The embedding function accepts a reduced parameter dict {key: value, ...}
    and embeds it into the full internal parameters, keeping all others fixed.
    """
    fixed_params = ground_truth_int_params.copy()
    
    def embed_params_flow(
        reduced_int_params: Dict[str, float],
        ext_params: Any,
        sim_params: Any,
    ) -> Dict[str, Any]:
        """
        Transform reduced internal parameters into full int_params dict.
        
        Args:
            reduced_int_params: Dict containing {key: inferred_value, ...}
            ext_params: Passed through unchanged (not modified here)
            sim_params: Passed through unchanged (not modified here)
        
        Returns:
            Full int_params dict with keys updated, all other values fixed.
        """
        full_params = fixed_params.copy()
        
        # Update only params from keys; all other parameters remain fixed
        for key in param_keys_to_infer:
            if key in reduced_int_params:
                full_params[key] = reduced_int_params[key]           
            
        return full_params
    
    # Create composed model with the embedding function
    ComposedModel = compose_model(
        ViscoElasticFilament_FlowParams_ScalarBending,
        compose_int_params=embed_params_flow,
    )
    return ComposedModel

def make_optimizer_kwargs(
    bounds = Bounds(lb=1e-6, ub=np.inf),
    local_minimizer_kwargs = {
        'method': 'L-BFGS-B',
        'jac': '3-point',
        'options': {
            'ftol': 1e-8,
            'gtol': 1e-8,
            'eps': 1e-8,
            'finite_diff_rel_step': 1e-6,
        },
    },
    global_minimizer_kwargs = {
        'niter': 9,
        'T': 0,
        'stepsize': 5,
        'tol': 1e-10,
    },
):
    return {
        'bounds': bounds,
        'local_minimizer_kwargs': local_minimizer_kwargs,
        'global_minimizer_kwargs': global_minimizer_kwargs,
    }

def _convert_to_serializable(obj):
    """Recursively convert NumPy and other non-serializable types to JSON-compatible types."""
    if isinstance(obj, dict):
        return {k: _convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

@dataclass
class ResultManifestEntry:
    """Single entry in the results manifest."""
    int_params: Dict[str, float]
    ext_params: Optional[Dict[str, float]]
    result_file: str
    mode: str
    pass_idx: Optional[int] = None
    cumul_indices: Optional[Dict[str, tuple]] = None

class ResultsManifest:
    """Centralized registry of all inference results."""
    
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.entries: List[ResultManifestEntry] = []
    
    def add_single_result(self, int_params, ext_params, result_file):
        """Register a single-mode inference result."""
        # Convert NumPy types to Python native types
        int_params = _convert_to_serializable(int_params)
        ext_params = _convert_to_serializable(ext_params)
        
        entry = ResultManifestEntry(
            int_params=int_params,
            ext_params=ext_params,
            result_file=str(result_file.relative_to(self.base_path)),
            mode='single',
        )
        self.entries.append(entry)
    
    def add_cumulative_result(self, int_params, result_file, pass_idx, cumul_indices):
        """Register a cumulative-mode pass result."""
        # Convert NumPy types to Python native types
        int_params = _convert_to_serializable(int_params)
        cumul_indices = _convert_to_serializable(cumul_indices)
        
        entry = ResultManifestEntry(
            int_params=int_params,
            ext_params=None,
            result_file=str(result_file.relative_to(self.base_path)),
            mode='cumulative_pass',
            pass_idx=pass_idx,
            cumul_indices=cumul_indices,
        )
        self.entries.append(entry)
    
    def save(self):
        """Write manifest to JSON with proper type conversion."""
        manifest_file = self.base_path / 'results_manifest.json'
        
        # Convert all entries to dicts and ensure JSON serializability
        manifest_data = {
            'entries': [_convert_to_serializable(asdict(entry)) for entry in self.entries]
        }
        
        with open(manifest_file, 'w') as f:
            json.dump(manifest_data, f, indent=2)
        
        print(f"Manifest saved: {manifest_file}")
    
    def query_by_params(self, **kwargs) -> List[ResultManifestEntry]:
        """Find results matching parameter values."""
        results = []
        for entry in self.entries:
            match = True
            for key, value in kwargs.items():
                if key not in entry.int_params or entry.int_params[key] != value:
                    match = False
                    break
            if match:
                results.append(entry)
        return results

def load_results_manifest(base_path):
    """Load the results manifest from a completed inference run."""
    manifest_file = base_path / 'results_manifest.json'
    with open(manifest_file, 'r') as f:
        return json.load(f)

def load_parameter_reference(base_path):
    """Load the parameter reference file."""
    ref_file = base_path / 'parameter_reference.json'
    with open(ref_file, 'r') as f:
        return json.load(f)

def load_result_file(base_path, result_file_path):
    """Load a pickled result file."""
    full_path = base_path / result_file_path
    with open(full_path, 'rb') as f:
        return pickle.load(f)

def _make_optimizer_bounds(param_keys_to_infer):
    """Create optimizer bounds for given parameter keys."""
    lb = [
        0 if ('Beta' in param_key or 'tau_b' in param_key or 'tau_s' in param_key) 
        else (1e-6 if 'Sp4' in param_key else 0) 
        for param_key in param_keys_to_infer
    ]
    ub = [np.inf] * len(param_keys_to_infer)
    return Bounds(lb=lb, ub=ub)

def _build_cumulative_ext_params(
    ext_param_range,
    ground_truth_fixed_ext_params,
    ext_param_dict,
    ext_param_indices,
    cumul_param_indices,
):
    """Build cumulative external parameter list.
    
    Args:
        ext_param_range: Dict of external parameter ranges
        ground_truth_fixed_ext_params: Fixed external parameters
        ext_param_dict: Current external parameter values
        ext_param_indices: Current iteration indices
        cumul_param_indices: Dict mapping params to (start_idx, end_idx)
    
    Returns:
        List of cumulative external parameter dicts
    """
    ext_param_names = list(ext_param_range.keys())
    ground_truth_ext_params_list_cumul = []
    
    # Build cumulation ranges
    cumul_ranges = {}
    for param_name in ext_param_range.keys():
        if param_name in cumul_param_indices:
            start_idx, end_idx = cumul_param_indices[param_name]
            cumul_ranges[param_name] = (start_idx, end_idx)
        else:
            cumul_ranges[param_name] = (0, len(ext_param_range[param_name]) - 1)
    
    # Generate all combinations within cumulation ranges
    index_ranges = [
        range(cumul_ranges[name][0], cumul_ranges[name][1] + 1) 
        for name in ext_param_names
    ]
    
    for index_combo in product(*index_ranges):
        cumul_ext_params = {
            **ground_truth_fixed_ext_params,
        }
        
        for param_name, idx in zip(ext_param_names, index_combo):
            cumul_ext_params[param_name] = ext_param_range[param_name][idx]
        
        ground_truth_ext_params_list_cumul.append(
            make_ground_truth_ext_params(**cumul_ext_params)
        )
    
    return ground_truth_ext_params_list_cumul

def _determine_inference_passes(param_keys_to_infer, elastic_params_list, viscous_params_list, min_w0):
    """Determine number of passes and split parameters based on min_w0.
    
    Returns:
        (n_passes, list of pass_configs)
        where each pass_config = {
            'name': str,
            'param_keys': list,
            'w0_filter': callable or None,
        }
    """
    
    elastic_keys = [k for k in param_keys_to_infer if k in elastic_params_list]
    viscous_keys = [k for k in param_keys_to_infer if k in viscous_params_list]
    unknown_keys = [k for k in param_keys_to_infer if k not in elastic_params_list and k not in viscous_params_list]
    
    if unknown_keys:
        raise ValueError(f"Unknown parameters for inference: {unknown_keys}. # TODO: full inference")
    
    if elastic_keys and viscous_keys:
        # Two passes
        return 2, [
            {
                'name': 'Elastic Inference',
                'param_keys': elastic_keys,
                'w0_filter': lambda w0: w0 == min_w0,
            },
            {
                'name': 'Viscous Inference',
                'param_keys': viscous_keys,
                'w0_filter': lambda w0: w0 > 0,
            },
        ]
    elif elastic_keys:
        # One pass: elastic only
        return 1, [
            {
                'name': 'Elastic Inference',
                'param_keys': elastic_keys,
                'w0_filter': lambda w0: w0 == min_w0,
            },
        ]
    else:
        # One pass: viscous only
        return 1, [
            {
                'name': 'Viscous Inference',
                'param_keys': viscous_keys,
                'w0_filter': lambda w0: w0 > 0,
            },
        ]

def _filter_ext_params_by_w0(ext_params_list, w0_filter):
    """Filter external parameters by w0 value.
    
    Args:
        ext_params_list: List of external parameter dicts
        w0_filter: Function that takes w0 value and returns bool
    
    Returns:
        Filtered list of external parameter dicts
    """
    return [
        ext_params for ext_params in ext_params_list
        if w0_filter(ext_params.get('w0', 0))
    ]

def run_single_ext_param_inference(
    int_params,
    ext_params,
    sim_params,
    param_keys_to_infer,
    initial_guesses,
    loss_fn=None,
    n_jobs=-1,
    optimizer=None,
    optimizer_kwargs=None,
):
    """Run single inference pass with fixed internal/external/sim parameters.
    
    No phase-splitting: all param_keys_to_infer are inferred in one pass.
    
    Args:
        int_params: Dict of internal parameters (fully specified)
        ext_params: Dict of external parameters (fully specified)
        sim_params: Dict of simulation parameters (fully specified)
        param_keys_to_infer: List of parameter keys to infer
        initial_guesses: List of initial guess dicts
        loss_fn: Loss function (default: rel_mse_loss_fn())
        n_jobs: Number of parallel jobs
        optimizer: Optimizer function (default: basinhopping_optimizer)
        optimizer_kwargs: Dict of optimizer kwargs
    
    Returns:
        InferenceResult object
    """
    
    if loss_fn is None:
        loss_fn = rel_mse_loss_fn()
    
    if optimizer is None:
        optimizer = basinhopping_optimizer
    
    if optimizer_kwargs is None:
        bounds = _make_optimizer_bounds(param_keys_to_infer)
        optimizer_kwargs = make_optimizer_kwargs(bounds=bounds)
    
    # Create ground truth objects
    ground_truth_int_params = make_ground_truth_int_params(**int_params)
    ground_truth_ext_params = make_ground_truth_ext_params(**ext_params)
    ground_truth_sim_params = make_ground_truth_sim_params(**sim_params)
    
    # Create model with only parameters to infer
    model_class = model_params_only_flow(
        ground_truth_int_params,
        param_keys_to_infer,
    )
    
    # Create ground truth data list
    ground_truth_data_list = make_ground_truth_data_list(
        ground_truth_int_params,
        [ground_truth_ext_params],
        [ground_truth_sim_params],
        "product",
    )
    
    # Create single pass
    pass_obj = PipelinePass(
        name='Single Inference',
        model_class=model_class,
        ground_truths=ground_truth_data_list,
        ext_params_list=[ground_truth_ext_params],
        sim_params_list=[ground_truth_sim_params],
        param_keys_to_infer=param_keys_to_infer,
        fixed_params={},
        optimizer=optimizer,
        optimizer_kwargs=optimizer_kwargs,
    )
    
    # Create and run pipeline
    pipeline = InferencePipeline(
        passes=[pass_obj],
        loss_fn=loss_fn,
        n_jobs_per_pass=n_jobs,
    )
    
    pass_initial_guesses = [[g for g in initial_guesses]]
    result = pipeline.run(pass_initial_guesses, verbose=True)[0]
    
    return result

def run_cumulative_inference(
    int_params,
    ext_param_range,
    ground_truth_fixed_ext_params,
    sim_params,
    param_keys_to_infer,
    initial_guesses,
    cumul_param_indices,
    loss_fn=None,
    n_jobs=-1,
    elastic_params_list=None,
    viscous_params_list=None,
):
    """Run cumulative inference with phase-splitting over external parameter range.
    
    Implements two-pass inference (elastic and viscous) over a range of external
    parameters specified by cumul_param_indices.
    
    Args:
        int_params: Dict of internal parameters (fully specified)
        ext_param_range: Dict of external parameter ranges (full ranges)
        ground_truth_fixed_ext_params: Dict of fixed external parameters
        sim_params: Dict of simulation parameters (fully specified)
        param_keys_to_infer: List of parameter keys to infer
        initial_guesses: List of initial guess dicts
        cumul_param_indices: Dict mapping ext param names to (start_idx, end_idx)
        loss_fn: Loss function
        n_jobs: Number of parallel jobs
        elastic_params_list: Parameters in elastic pass
        viscous_params_list: Parameters in viscous pass
    
    Returns:
        List of (pass_idx, result) tuples
    """
    
    if elastic_params_list is None:
        elastic_params_list = ['Sp4', 'Beta']
    if viscous_params_list is None:
        viscous_params_list = ['tau_b', 'tau_s']
    if loss_fn is None:
        loss_fn = rel_mse_loss_fn()
    
    # Compute min_w0
    w0_values = []
    if 'w0' in ground_truth_fixed_ext_params:
        w0_values.append(ground_truth_fixed_ext_params['w0'])
    if 'w0' in ext_param_range:
        w0_values.extend(ext_param_range['w0'])
    min_w0 = min(w0_values) if w0_values else 0
    
    # Build cumulative external parameters list
    ext_param_names = list(ext_param_range.keys())
    cumul_ranges = {}
    for param_name in ext_param_range.keys():
        if param_name in cumul_param_indices:
            start_idx, end_idx = cumul_param_indices[param_name]
            cumul_ranges[param_name] = (start_idx, end_idx)
        else:
            cumul_ranges[param_name] = (0, len(ext_param_range[param_name]) - 1)
    
    # Generate all combinations within cumulation ranges
    index_ranges = [
        range(cumul_ranges[name][0], cumul_ranges[name][1] + 1)
        for name in ext_param_names
    ]
    
    ext_params_list_cumul = []
    for index_combo in product(*index_ranges):
        cumul_ext_params = {
            **ground_truth_fixed_ext_params,
        }
        for param_name, idx in zip(ext_param_names, index_combo):
            cumul_ext_params[param_name] = ext_param_range[param_name][idx]
        
        ext_params_list_cumul.append(cumul_ext_params)
    
    # Create ground truth objects
    ground_truth_int_params = make_ground_truth_int_params(**int_params)
    ground_truth_ext_params_list_cumul = [
        make_ground_truth_ext_params(**ext_params) for ext_params in ext_params_list_cumul
    ]
    ground_truth_sim_params = make_ground_truth_sim_params(**sim_params)
    
    # Determine passes
    n_passes, pass_configs = _determine_inference_passes(
        param_keys_to_infer,
        elastic_params_list,
        viscous_params_list,
        min_w0,
    )
    
    # Create ground truth data list for all cumulative external params
    ground_truth_data_list_cumul = make_ground_truth_data_list(
        ground_truth_int_params,
        ground_truth_ext_params_list_cumul,
        [ground_truth_sim_params],
        "product",
    )
    
    # Run passes
    passes_results = []
    for pass_idx, pass_config in enumerate(pass_configs):
        param_keys_to_infer_per_pass = pass_config['param_keys']
        
        # Create model with only parameters for this pass
        model_class = model_params_only_flow(
            ground_truth_int_params,
            param_keys_to_infer_per_pass,
        )
        
        # Create bounds for this pass
        bounds = _make_optimizer_bounds(param_keys_to_infer_per_pass)
        optimizer_kwargs = make_optimizer_kwargs(bounds=bounds)
        
        # Filter external parameters by w0
        filtered_ext_params_list = _filter_ext_params_by_w0(
            ground_truth_ext_params_list_cumul,
            pass_config['w0_filter'],
        )
        
        if not filtered_ext_params_list:
            print(f"  Cumulative Pass {pass_idx}: No external parameters suited for {pass_config['name']}")
            passes_results.append(None)
            continue
        
        # Create corresponding data list
        filtered_ground_truth_data_list = make_ground_truth_data_list(
            ground_truth_int_params,
            filtered_ext_params_list,
            [ground_truth_sim_params],
            "product",
        )
        
        # Create pass
        pass_obj = PipelinePass(
            name=f"{pass_config['name']} - Cumulative",
            model_class=model_class,
            ground_truths=filtered_ground_truth_data_list,
            ext_params_list=filtered_ext_params_list,
            sim_params_list=[ground_truth_sim_params],
            param_keys_to_infer=param_keys_to_infer_per_pass,
            fixed_params={},
            optimizer=basinhopping_optimizer,
            optimizer_kwargs=optimizer_kwargs,
        )
        
        # Run pipeline
        pipeline = InferencePipeline(
            passes=[pass_obj],
            loss_fn=loss_fn,
            n_jobs_per_pass=n_jobs,
        )
        
        # Filter initial guesses for this pass
        pass_initial_guesses = [
            [
                {k: v for k, v in guess.items() if k in param_keys_to_infer_per_pass}
                for guess in initial_guesses
            ]
        ]
        
        result = pipeline.run(pass_initial_guesses, verbose=True)[0]
        passes_results.append((pass_idx, result))
    
    return passes_results

def _run_single_mode(base_path, manifest, ground_truth_int_params, int_param_dict, 
                    int_param_names, int_param_indices, ext_param_range, ext_param_names,
                    ground_truth_fixed_ext_params, sim_param_dict, param_keys_to_infer,
                    initial_guesses, loss_fn, n_jobs):
    """Run single-mode inference and register results."""
    
    ext_param_indices_iter = product(
        *[range(len(ext_param_range[name])) for name in ext_param_names]
    )
    
    for ext_param_indices in ext_param_indices_iter:
        ext_param_dict = {}
        for param_name, idx in zip(ext_param_names, ext_param_indices):
            ext_param_dict[param_name] = ext_param_range[param_name][idx]
        
        ground_truth_ext_params = {
            **ground_truth_fixed_ext_params,
            **ext_param_dict,
        }
        
        # Build result folder with cleaner naming
        result_folder = _build_result_folder_path(
            base_path, 'single', int_param_dict, ext_param_dict
        )
        result_folder.mkdir(parents=True, exist_ok=True)
        
        print(f"Processing: {result_folder.name}")
        
        result = run_single_ext_param_inference(
            int_params=ground_truth_int_params,
            ext_params=ground_truth_ext_params,
            sim_params=sim_param_dict,
            param_keys_to_infer=param_keys_to_infer,
            initial_guesses=initial_guesses,
            loss_fn=loss_fn,
            n_jobs=n_jobs,
        )
        
        result_file = result_folder / 'result.pkl'
        result.save(str(result_file.resolve()))
        
        # Register in manifest
        manifest.add_single_result(
            int_params=ground_truth_int_params,
            ext_params=ground_truth_ext_params,
            result_file=result_file,
        )

        manifest.save()
        print(f"\nResults manifest saved to {base_path / 'results_manifest.json'}")  

def _run_cumulative_mode(base_path, manifest, ground_truth_int_params, int_param_dict,
                         int_param_names, int_param_indices, ext_param_range,
                         ground_truth_fixed_ext_params, sim_param_dict, param_keys_to_infer,
                         initial_guesses, cumul_param_indices_list, loss_fn, n_jobs,
                         elastic_params_list, viscous_params_list):
    """Run cumulative-mode inference and register results."""

    # Loop through each cumulative parameter index configuration
    for cumul_config_idx, cumul_param_indices in enumerate(cumul_param_indices_list):
        ext_params_summary = {}
        ext_params_summary.update(ground_truth_fixed_ext_params)

        if cumul_param_indices:
            for param_name, (start_idx, end_idx) in cumul_param_indices.items():
                if param_name in ext_param_range:
                    start_val = ext_param_range[param_name][start_idx]
                    end_val = ext_param_range[param_name][end_idx]
                    ext_params_summary[param_name] = f"{start_val:.4g}_to_{end_val:.4g}"

        result_folder_base = _build_result_folder_path_cumulative(
            base_path, int_param_dict, ext_params_summary
        )
        result_folder_base.parent.mkdir(parents=True, exist_ok=True)
        result_folder_base.mkdir(parents=True, exist_ok=True)

        print(f"Processing cumulative (config {cumul_config_idx}): {result_folder_base.name}")

        passes_results = run_cumulative_inference(
            int_params=ground_truth_int_params,
            ext_param_range=ext_param_range,
            ground_truth_fixed_ext_params=ground_truth_fixed_ext_params,
            sim_params=sim_param_dict,
            param_keys_to_infer=param_keys_to_infer,
            initial_guesses=initial_guesses,
            cumul_param_indices=cumul_param_indices,
            loss_fn=loss_fn,
            n_jobs=n_jobs,
            elastic_params_list=elastic_params_list,
            viscous_params_list=viscous_params_list,
        )

        for pass_idx, result in passes_results:
            if result is not None:
                result_file = result_folder_base / f'pass_{pass_idx:02d}.pkl'
                result.save(str(result_file.resolve()))

                manifest.add_cumulative_result(
                    int_params=ground_truth_int_params,
                    result_file=result_file,
                    pass_idx=pass_idx,
                    cumul_indices=cumul_param_indices,
                )

        # Save manifest after each cumulative configuration
        manifest.save()
        print(f"Manifest saved after config {cumul_config_idx}")                

def _build_result_folder_path(base_path, mode, int_params, ext_params):
    """Build result folder path with readable naming."""
    parts = []
    
    # Internal parameters
    for name, value in sorted(int_params.items()):
        parts.append(f"{name}={value:.4g}".replace(' ', ''))
    
    # External parameters (include for both single and cumulative)
    if ext_params:
        for name, value in sorted(ext_params.items()):
            parts.append(f"{name}={value:.4g}".replace(' ', ''))
    
    folder_name = '__'.join(parts) if parts else 'results'
    return base_path / folder_name

def _build_result_folder_path_cumulative(base_path, int_params, ext_params_summary):
    """Build cumulative result folder path with readable naming including ext param ranges."""
    parts = []
    
    # Internal parameters
    for name, value in sorted(int_params.items()):
        parts.append(f"{name}={value:.4g}".replace(' ', ''))
    
    # External parameters (including ranges for cumulative)
    for name, value in sorted(ext_params_summary.items()):
        if isinstance(value, str):
            # For range values like "1e-4_to_1e-2"
            parts.append(f"{name}={value}")
        else:
            parts.append(f"{name}={value:.4g}".replace(' ', ''))
    
    folder_name = '__'.join(parts) if parts else 'results'
    return base_path / folder_name

def _save_parameter_reference(base_path, int_param_range, ext_param_range, cumul_param_indices):
    """Save parameter ranges as reference (static lookup)."""
    reference = {
        'internal_parameters': {
            name: {i: float(val) for i, val in enumerate(values)}
            for name, values in int_param_range.items()
        },
        'external_parameters': {
            name: {i: float(val) for i, val in enumerate(values)}
            for name, values in ext_param_range.items()
        },
        'cumulative_indices': _convert_to_serializable(cumul_param_indices or {}),
    }
    
    with open(base_path / 'parameter_reference.json', 'w') as f:
        json.dump(reference, f, indent=2)
    
    print(f"Parameter reference saved: {base_path / 'parameter_reference.json'}") 
def run_inference_pipeline(
    base_path,
    int_param_range,
    ground_truth_fixed_int_params,
    ext_param_range,
    ground_truth_fixed_ext_params,
    sim_param_dict,
    param_keys_to_infer,
    initial_guesses,
    mode='single',
    cumul_param_indices=None,
    loss_fn=None,
    n_jobs=-1,
    elastic_params_list=None,
    viscous_params_list=None,
):
    """Orchestrator: loop through internal parameters and run inference.

    Args:
        base_path: Root directory for results
        int_param_range: Dict of internal parameter ranges
        ground_truth_fixed_int_params: Fixed internal parameters
        ext_param_range: Dict of external parameter ranges
        ground_truth_fixed_ext_params: Fixed external parameters
        sim_param_dict: Dict of simulation parameters
        param_keys_to_infer: List of parameters to infer
        initial_guesses: List of initial guess dicts
        mode: 'single' (loop ext params) or 'cumulative' (bulk ext params)
        cumul_param_indices: For cumulative mode, list of dicts, each dict containing 
                            (start, end) indices per ext param
        loss_fn: Loss function
        n_jobs: Number of parallel jobs
        elastic_params_list: For cumulative mode
        viscous_params_list: For cumulative mode
    """

    if elastic_params_list is None:
        elastic_params_list = ['Sp4', 'Beta']
    if viscous_params_list is None:
        viscous_params_list = ['tau_b', 'tau_s']
    if loss_fn is None:
        loss_fn = rel_mse_loss_fn()

    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)

    manifest = ResultsManifest(base_path)

    _save_parameter_reference(base_path, int_param_range, ext_param_range, cumul_param_indices)

    int_param_names = list(int_param_range.keys())
    ext_param_names = list(ext_param_range.keys())

    int_param_indices_iter = product(
        *[range(len(int_param_range[name])) for name in int_param_names]
    )

    for int_param_indices in int_param_indices_iter:
        int_param_dict = {}
        folder_name_parts_int = []

        for param_name, idx in zip(int_param_names, int_param_indices):
            value = int_param_range[param_name][idx]
            int_param_dict[param_name] = value
            folder_name_parts_int.append(f'{param_name}_{idx:03d}')

        ground_truth_int_params = {
            **ground_truth_fixed_int_params,
            **int_param_dict,
        }

        if mode == 'single':

            _run_single_mode(
                base_path,
                manifest,
                ground_truth_int_params,
                int_param_dict,
                int_param_names,
                int_param_indices,
                ext_param_range,
                ext_param_names,
                ground_truth_fixed_ext_params,
                sim_param_dict,
                param_keys_to_infer,
                initial_guesses,
                loss_fn,
                n_jobs,
            )          

        elif mode == 'cumulative':
            # Ensure cumul_param_indices is a list; wrap single dict in list if needed
            cumul_indices_list = cumul_param_indices if isinstance(cumul_param_indices, list) else [cumul_param_indices]
            
            _run_cumulative_mode(
                base_path,
                manifest,
                ground_truth_int_params,
                int_param_dict,
                int_param_names,
                int_param_indices,
                ext_param_range,
                ground_truth_fixed_ext_params,
                sim_param_dict,
                param_keys_to_infer,
                initial_guesses,
                cumul_indices_list,
                loss_fn,
                n_jobs,
                elastic_params_list,
                viscous_params_list,
            )


if __name__ == "__main__":
    
    """ Tests """

    # pytest.main([
    #     __file__,
    #     "-vv",
    #     "--tb=short",
    #     "-s",
    # ])

    """ Main """

    # Infer elasticities

    ## Bending elasticity (Sp4 = 1e-3->1e3, Beta = 0)

    if False:
        base_path = Path(__file__).resolve().parent.parent / 'Inference' / 'FromSimulationData' / 'ElasticInference_BendingElasticity'
    
        int_param_range = {
            'Sp4': np.pow(10, np.linspace(start=-3, stop=3, num=3)),
        }
        ground_truth_fixed_int_params = {}

        A_vec = np.pow(10, np.linspace(start=-6, stop=-2, num=50))
        ext_param_range = {
            'A': A_vec,
        }
        ground_truth_fixed_ext_params = {}

        initial_guesses = [
            {'Sp4': 1e-1},
        ]
        param_keys_to_infer = list(initial_guesses[0].keys())

        sim_param_dict = {}

        cumul_param_indices = [{'A':(0, k)} for k in range(A_vec.shape[0])]


        # ========== RUN INFERENCES ==========
        
        # Test 1: Single external parameter inference
        print("=" * 80)
        print("TEST 1: Single External Parameter Inference Mode")
        print("=" * 80)
        run_inference_pipeline(
            base_path=base_path / 'SingleExtParams',
            int_param_range=int_param_range,
            ground_truth_fixed_int_params=ground_truth_fixed_int_params,
            ext_param_range=ext_param_range,
            ground_truth_fixed_ext_params=ground_truth_fixed_ext_params,
            sim_param_dict=sim_param_dict,
            param_keys_to_infer=param_keys_to_infer,
            initial_guesses=initial_guesses,
            mode='single',
        )

        # Test 2: Cumulative inference
        print("\n" + "=" * 80)
        print("TEST 2: Cumulative Inference Mode")
        print("=" * 80)
        run_inference_pipeline(
            base_path=base_path / 'CumulativeExtParams',
            int_param_range=int_param_range,
            ground_truth_fixed_int_params=ground_truth_fixed_int_params,
            ext_param_range=ext_param_range,
            ground_truth_fixed_ext_params=ground_truth_fixed_ext_params,
            sim_param_dict=sim_param_dict,
            param_keys_to_infer=param_keys_to_infer,
            initial_guesses=initial_guesses,
            mode='cumulative',
            cumul_param_indices=cumul_param_indices,
        )


    ## Shear elasticity (Sp4 = 1, Beta = 1e-3->1e3) 
    
    if False:
        base_path = Path(__file__).resolve().parent.parent / 'Inference' / 'FromSimulationData' / 'ElasticInference_ShearElasticity'

        int_param_range = {
            'Beta': np.pow(10, np.linspace(start=-3, stop=3, num=3)),
        }
        ground_truth_fixed_int_params = {}

        A_vec = np.pow(10, np.linspace(start=-6, stop=-2, num=50))
        ext_param_range = {
            'A': A_vec,
        }
        ground_truth_fixed_ext_params = {}

        initial_guesses = [
            {'Beta': 0},
        ]
        param_keys_to_infer = (initial_guesses[0].keys())

        sim_param_dict = {}

        cumul_param_indices = [{'A':(0, k)} for k in range(A_vec.shape[0])]

        # ========== RUN INFERENCES ==========
        
        # Test 1: Single external parameter inference
        print("=" * 80)
        print("TEST 1: Single External Parameter Inference Mode")
        print("=" * 80)
        run_inference_pipeline(
            base_path=base_path / 'SingleExtParams',
            int_param_range=int_param_range,
            ground_truth_fixed_int_params=ground_truth_fixed_int_params,
            ext_param_range=ext_param_range,
            ground_truth_fixed_ext_params=ground_truth_fixed_ext_params,
            sim_param_dict=sim_param_dict,
            param_keys_to_infer=param_keys_to_infer,
            initial_guesses=initial_guesses,
            mode='single',
        )

        # Test 2: Cumulative inference
        print("\n" + "=" * 80)
        print("TEST 2: Cumulative Inference Mode")
        print("=" * 80)
        run_inference_pipeline(
            base_path=base_path / 'CumulativeExtParams',
            int_param_range=int_param_range,
            ground_truth_fixed_int_params=ground_truth_fixed_int_params,
            ext_param_range=ext_param_range,
            ground_truth_fixed_ext_params=ground_truth_fixed_ext_params,
            sim_param_dict=sim_param_dict,
            param_keys_to_infer=param_keys_to_infer,
            initial_guesses=initial_guesses,
            mode='cumulative',
            cumul_param_indices=cumul_param_indices,
        )

    ## Bending + Shear elasticities (Sp4 = 1e-3->1e3, Beta = 1e-3->1e3)

    if False:
        base_path = Path(__file__).resolve().parent.parent / 'Inference' / 'FromSimulationData' / 'ElasticInference_BendingShearElasticity'
    
        int_param_range = {
            'Sp4': np.pow(10, np.linspace(start=-3, stop=3, num=3)),
            'Beta': np.pow(10, np.linspace(start=-3, stop=3, num=3)),
        }
        ground_truth_fixed_int_params = {}

        A_vec = np.pow(10, np.linspace(start=-6, stop=-2, num=50))
        ext_param_range = {
            'A': A_vec,
        }
        ground_truth_fixed_ext_params = {}

        initial_guesses = [
            {'Sp4': 1e-1, 'Beta': 0},
        ]
        param_keys_to_infer = list(initial_guesses[0].keys())

        sim_param_dict = {}

        cumul_param_indices = [{'A':(0, k)} for k in range(A_vec.shape[0])]

        # ========== RUN INFERENCES ==========
        
        # Test 1: Single external parameter inference
        print("=" * 80)
        print("TEST 1: Single External Parameter Inference Mode")
        print("=" * 80)
        run_inference_pipeline(
            base_path=base_path / 'SingleExtParams',
            int_param_range=int_param_range,
            ground_truth_fixed_int_params=ground_truth_fixed_int_params,
            ext_param_range=ext_param_range,
            ground_truth_fixed_ext_params=ground_truth_fixed_ext_params,
            sim_param_dict=sim_param_dict,
            param_keys_to_infer=param_keys_to_infer,
            initial_guesses=initial_guesses,
            mode='single',
        )

        # Test 2: Cumulative inference
        print("\n" + "=" * 80)
        print("TEST 2: Cumulative Inference Mode")
        print("=" * 80)
        run_inference_pipeline(
            base_path=base_path / 'CumulativeExtParams',
            int_param_range=int_param_range,
            ground_truth_fixed_int_params=ground_truth_fixed_int_params,
            ext_param_range=ext_param_range,
            ground_truth_fixed_ext_params=ground_truth_fixed_ext_params,
            sim_param_dict=sim_param_dict,
            param_keys_to_infer=param_keys_to_infer,
            initial_guesses=initial_guesses,
            mode='cumulative',
            cumul_param_indices=cumul_param_indices,
        )

    # Infer viscosities

    ## Bending viscosity (Sp4 = 1e-3->1e3, Beta = 0, tau_b = 1e-3->1e3)

    if True:
        base_path = Path(__file__).resolve().parent.parent / 'Inference' / 'FromSimulationData' / 'ElasticInference_BendingShearElasticity'
    
        int_param_range = {
            'Sp4': np.pow(10, np.linspace(start=-3, stop=3, num=3)),
            'Beta': np.pow(10, np.linspace(start=-3, stop=3, num=3)),
        }
        ground_truth_fixed_int_params = {}

        A_vec = np.pow(10, np.linspace(start=-6, stop=-2, num=50))
        ext_param_range = {
            'A': A_vec,
        }
        ground_truth_fixed_ext_params = {}

        initial_guesses = [
            {'Sp4': 1e-1, 'Beta': 0},
        ]
        param_keys_to_infer = list(initial_guesses[0].keys())

        sim_param_dict = {}

        cumul_param_indices = [{'A':(0, k)} for k in range(A_vec.shape[0])]

        # ========== RUN INFERENCES ==========
        
        # Test 1: Single external parameter inference
        print("=" * 80)
        print("TEST 1: Single External Parameter Inference Mode")
        print("=" * 80)
        run_inference_pipeline(
            base_path=base_path / 'SingleExtParams',
            int_param_range=int_param_range,
            ground_truth_fixed_int_params=ground_truth_fixed_int_params,
            ext_param_range=ext_param_range,
            ground_truth_fixed_ext_params=ground_truth_fixed_ext_params,
            sim_param_dict=sim_param_dict,
            param_keys_to_infer=param_keys_to_infer,
            initial_guesses=initial_guesses,
            mode='single',
        )

        # Test 2: Cumulative inference
        print("\n" + "=" * 80)
        print("TEST 2: Cumulative Inference Mode")
        print("=" * 80)
        run_inference_pipeline(
            base_path=base_path / 'CumulativeExtParams',
            int_param_range=int_param_range,
            ground_truth_fixed_int_params=ground_truth_fixed_int_params,
            ext_param_range=ext_param_range,
            ground_truth_fixed_ext_params=ground_truth_fixed_ext_params,
            sim_param_dict=sim_param_dict,
            param_keys_to_infer=param_keys_to_infer,
            initial_guesses=initial_guesses,
            mode='cumulative',
            cumul_param_indices=cumul_param_indices,
        )

    ## Shear viscosity (Sp4 = 1e-3->1e3, Beta = 1e-3->1e3, tau_s = 1e-3->1e3)

    ## Bending + Shear viscosity (Sp4 = 1e-3->1e3, Beta = 1e-3->1e3, tau_b = 1e-3 -> 1e3, tau_s = 1e-3->1e3)
        
    # Elasticity + Viscosity