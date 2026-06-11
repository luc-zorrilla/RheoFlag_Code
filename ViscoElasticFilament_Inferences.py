from Models import Model, ModelList
from ViscoElasticFilament_Models import (
    StraightLine, 
    ViscoElasticFilament, 
    ViscoElasticFilament_FlowParams, 
    ViscoElasticFilament_FlowParams_ScalarBending,
)
from Inferences import Inference, InferencePipeline, PipelinePass, InferenceResult
from ModelInferenceWorkflow import InferenceTask, SimulationInferenceWorkflow

from typing import Any, Callable, Dict, List, Tuple, Optional, Type
from pathlib import Path
from itertools import product, zip_longest
import copy
import shutil

import numpy as np
from scipy.optimize import Bounds, minimize
from _basinhopping_mod import basinhopping # Custom Optimiser
from joblib import Parallel, delayed

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

### Loss function

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

### Parameters and Ground Truth

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

def make_sim_params_for_w0(w0):
    """Create simulation parameters based on w0 value."""
    if w0 == 0.0:
        sim_params_dict = {
            "T_span": (1e6, 2e6),
            "T_eval": np.linspace(1e6, 2e6, int(1e0)),
            "method": "hybr",
        }
    else:
        T_start = (1.0 / 10.0) / w0
        T_end = 10.0 / w0
        N_T = 100
        sim_params_dict =  {
            "T_span": (T_start, T_end),
            "T_eval": np.linspace(T_start, T_end, N_T),
            "method": "BDF",
        }
    return make_ground_truth_sim_params(**sim_params_dict)

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
    ComposedModel = ViscoElasticFilament_FlowParams_ScalarBending.compose(
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

def _make_optimizer_bounds(param_keys_to_infer):
    """Create optimizer bounds for given parameter keys."""
    lb = [
        0 if ('Beta' in param_key or 'tau_b' in param_key or 'tau_s' in param_key) 
        else (1e-6 if 'Sp4' in param_key else 0) 
        for param_key in param_keys_to_infer
    ]
    ub = [np.inf] * len(param_keys_to_infer)
    return Bounds(lb=lb, ub=ub)

def make_simple_inference_pipeline(model_list, **kwargs):
    """Factory function to create inference pipeline from ModelList."""
    
    # Extract ground truths from simulated models
    ground_truths = [model.sim_output['value'] for model in model_list.models]
    ext_params_batch = [model.ext_params for model in model_list.models]
    sim_params_batch = [model.sim_params for model in model_list.models]
    
    param_keys_to_infer = list(model_list.models[0].int_params.keys())
    model_class = type(model_list.models[0])

    # Single pass: infer int_params
    pass_1 = PipelinePass(
        name="One-pass Inference",
        model_class=model_class,
        ground_truths=ground_truths,
        ext_params_list=ext_params_batch,
        sim_params_list=sim_params_batch,
        param_keys_to_infer=param_keys_to_infer,
        fixed_params={},
        product_or_zip="zip",
        optimizer=kwargs['optimizer'],
        optimizer_kwargs=kwargs['optimizer_kwargs'],
    )
    
    return InferencePipeline(
        passes=[pass_1],
        loss_fn=kwargs['loss_fn'],
        n_jobs_per_pass=-1,
    )

def _find_min_w0(ext_params_list):
    """Find the minimum w0 value from external parameters list.
    
    Args:
        ext_params_list: List of external parameter dicts
    
    Returns:
        Minimum w0 value, or 0.0 if no w0 values found
    """
    w0_values = [ext_params.get('w0', 0.0) for ext_params in ext_params_list]
    return min(w0_values) if w0_values else 0.0

def _determine_inference_passes(
    param_keys_to_infer, 
    elastic_params_list, 
    viscous_params_list,
    ext_params_list,
):
    """Determine number of passes and split parameters based on min_w0.
    Also validates that filtered data exists for each pass.
    
    Returns:
        (n_passes, list of pass_configs)
        where each pass_config = {
            'name': str,
            'param_keys': list,
            'w0_filter': callable or None,
        }
    """

    min_w0 = _find_min_w0(ext_params_list)
    
    elastic_keys = [k for k in param_keys_to_infer if k in elastic_params_list]
    viscous_keys = [k for k in param_keys_to_infer if k in viscous_params_list]
    unknown_keys = [k for k in param_keys_to_infer if k not in elastic_params_list and k not in viscous_params_list]
    
    if unknown_keys:
        raise ValueError(f"Unknown parameters for inference: {unknown_keys}.")

    # Helper: check if a filter yields any data
    def filter_has_data(w0_filter):
        return any(w0_filter(ext_params.get('w0', 0)) for ext_params in ext_params_list)

    if elastic_keys and viscous_keys:
        elastic_filter = lambda w0: w0 == min_w0
        viscous_filter = lambda w0: w0 > 0
        
        elastic_has_data = filter_has_data(elastic_filter)
        viscous_has_data = filter_has_data(viscous_filter)
        
        if elastic_has_data and viscous_has_data:
            # Both passes viable
            return 2, [
                {
                    'name': 'Elastic Inference',
                    'param_keys': elastic_keys,
                    'w0_filter': elastic_filter,
                },
                {
                    'name': 'Viscous Inference',
                    'param_keys': viscous_keys,
                    'w0_filter': viscous_filter,
                },
            ]
        elif elastic_has_data:
            # Only elastic data; infer viscous params using all data
            return 1, [
                {
                    'name': 'Single Pass (Elastic + Viscous)',
                    'param_keys': elastic_keys + viscous_keys,
                    'w0_filter': None,  # Use all data
                },
            ]
        elif viscous_has_data:
            # Only viscous data; infer elastic params using all data
            return 1, [
                {
                    'name': 'Single Pass (Elastic + Viscous)',
                    'param_keys': elastic_keys + viscous_keys,
                    'w0_filter': None,  # Use all data
                },
            ]
        else:
            raise ValueError("No data available for either elastic or viscous inference.")
    
    elif elastic_keys:
        elastic_filter = lambda w0: w0 == min_w0
        if filter_has_data(elastic_filter):
            return 1, [
                {
                    'name': 'Elastic Inference',
                    'param_keys': elastic_keys,
                    'w0_filter': elastic_filter,
                },
            ]
        else:
            # No data at min_w0; use all data
            return 1, [
                {
                    'name': 'Elastic Inference (All Data)',
                    'param_keys': elastic_keys,
                    'w0_filter': None,
                },
            ]
    else:
        viscous_filter = lambda w0: w0 > 0
        if filter_has_data(viscous_filter):
            return 1, [
                {
                    'name': 'Viscous Inference',
                    'param_keys': viscous_keys,
                    'w0_filter': viscous_filter,
                },
            ]
        else:
            # No data with w0 > 0; use all data
            return 1, [
                {
                    'name': 'Viscous Inference (All Data)',
                    'param_keys': viscous_keys,
                    'w0_filter': None,
                },
            ]

def _filter_ext_params_by_w0(ext_params_list, w0_filter):
    """Filter external parameters by w0 value.
    
    Args:
        ext_params_list: List of external parameter dicts
        w0_filter: Function that takes w0 value and returns bool, or None for all data
    
    Returns:
        Filtered list of external parameter dicts
    """
    if w0_filter is None:
        return ext_params_list
    
    return [
        ext_params for ext_params in ext_params_list
        if w0_filter(ext_params.get('w0', 0))
    ]

def make_two_pass_pipeline(model_list, **kwargs) -> InferencePipeline:
    """
    Create inference pipeline that:
    1. Determines passes dynamically based on data availability
    2. For each pass, optimizes only relevant subset of parameters
    3. Maintains full int_params structure across passes via PipelinePass
    
    Args:
        model_list: ModelList object containing simulated models
        **kwargs: Expected keys:
            - param_keys_to_infer
            - optimizer: Callable
            - optimizer_kwargs: Dict
            - loss_fn: Callable
            - n_jobs_per_pass: int (default: -1)
    
    Returns:
        InferencePipeline configured for two-pass inference
    """
    
    # Extract from model_list
    ground_truths = [model.sim_output['value'] for model in model_list.models]
    ext_params_list = [model.ext_params for model in model_list.models]
    sim_params_list = [model.sim_params for model in model_list.models]
    ground_truth_models = model_list.models
    model_class = type(model_list.models[0])
    
    # Extract from kwargs
    param_keys_to_infer = kwargs.get('param_keys_to_infer')
    elastic_params_list = ['Sp4', 'Beta']
    viscous_params_list = ['tau_b', 'tau_s']
    optimizer = kwargs.get('optimizer')
    optimizer_kwargs = kwargs.get('optimizer_kwargs')
    loss_fn = kwargs.get('loss_fn')
    n_jobs_per_pass = kwargs.get('n_jobs_per_pass', -1)
    product_or_zip = kwargs.get('product_or_zip', 'zip')
    
    # Determine passes and get pass configurations
    n_passes, pass_configs = _determine_inference_passes(
        param_keys_to_infer=param_keys_to_infer,
        elastic_params_list=elastic_params_list,
        viscous_params_list=viscous_params_list,
        ext_params_list=ext_params_list,
    )
    
    pipeline_passes = []
    
    for pass_config in pass_configs:
        pass_name = pass_config['name']
        param_keys = pass_config['param_keys']
        
        w0_filter = pass_config['w0_filter']
        
        # Filter data for this pass
        filtered_ext_params = _filter_ext_params_by_w0(ext_params_list, w0_filter)
        filtered_indices = [
            i for i, ext_params in enumerate(ext_params_list)
            if ext_params in filtered_ext_params
        ]
        
        filtered_ground_truths = [ground_truths[i] for i in filtered_indices]
        filtered_ext_params_list = [ext_params_list[i] for i in filtered_indices]
        filtered_sim_params_list = [sim_params_list[i] for i in filtered_indices]
        
        # Create PipelinePass for this pass
        pipeline_pass = PipelinePass(
            name=pass_name,
            model_class=model_class,
            ground_truths=filtered_ground_truths,
            ext_params_list=filtered_ext_params_list,
            sim_params_list=filtered_sim_params_list,
            param_keys_to_infer=param_keys,
            product_or_zip=product_or_zip,
            optimizer=optimizer,
            optimizer_kwargs={**optimizer_kwargs, 'bounds': _make_optimizer_bounds(param_keys)},
            compose_int_params=None,
            compose_ext_params=None,
            compose_sim_params=None,
        )
        pipeline_passes.append(pipeline_pass)
    
    return InferencePipeline(
        passes=pipeline_passes,
        loss_fn=loss_fn,
        n_jobs_per_pass=n_jobs_per_pass,
    )

def _make_two_pass_pipeline_factory(
    model_class,
    ext_params_list,
    sim_params_list,
    param_keys_to_infer,
    elastic_params_list,
    viscous_params_list,
    loss_fn,
    optimizer,
    optimizer_kwargs,
    n_jobs_per_pass,
):
    """
    Factory function that returns a make_pipeline_fn with bound parameters.
    This allows pipeline creation to be deferred until inference time.
    """
    
    def make_pipeline_fn(model_list, **kwargs):
        """
        Create a two-pass pipeline with the bound parameters.
        
        Args:
            model_list: ModelList object (from filtered data)
            **kwargs: Additional arguments (ignored)
        
        Returns:
            InferencePipeline configured for two-pass inference
        """
        return make_two_pass_pipeline(
            model_list=model_list,
            param_keys_to_infer=param_keys_to_infer,
            elastic_params_list=elastic_params_list,
            viscous_params_list=viscous_params_list,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            loss_fn=loss_fn,
            n_jobs_per_pass=n_jobs_per_pass,
        )
    
    return make_pipeline_fn

def make_inference_tasks_two_pass(
    mode: str,
    int_params_list: list[dict],
    ext_params_list: list[dict],
    param_keys_to_infer: list[str],
    elastic_params_list: list[str],
    viscous_params_list: list[str],
    make_sim_params_fn,
    model_class,
    loss_fn,
    optimizer,
    optimizer_kwargs,
    initial_guesses: list[dict],
    n_jobs_per_pass: int = -1,
) -> list[InferenceTask]:
    """Create inference tasks with two-pass pipeline."""
    
    if mode not in ('single_inference', 'cumulative_inference'):
        raise ValueError(f"mode must be 'single_inference' or 'cumulative_inference', got '{mode}'")
    
    if not int_params_list or not ext_params_list or not param_keys_to_infer:
        raise ValueError("int_params_list, ext_params_list, and param_keys_to_infer cannot be empty")
    
    sim_params_list = [make_sim_params_fn(ext_params.get('w0', 0.0)) for ext_params in ext_params_list]
    tasks = []
    
    if mode == 'single_inference':
        for int_idx, int_params in enumerate(int_params_list):
            for ext_idx, (ext_params, sim_params) in enumerate(zip(ext_params_list, sim_params_list)):
                task_key = f"int_{int_idx:03d}_ext_{ext_idx:03d}"
                
                # Factory for this specific ext_params pair
                pipeline_fn = _make_two_pass_pipeline_factory(
                    model_class=model_class,
                    ext_params_list=[ext_params],
                    sim_params_list=[sim_params],
                    param_keys_to_infer=param_keys_to_infer,
                    elastic_params_list=elastic_params_list,
                    viscous_params_list=viscous_params_list,
                    loss_fn=loss_fn,
                    optimizer=optimizer,
                    optimizer_kwargs=optimizer_kwargs,
                    n_jobs_per_pass=n_jobs_per_pass,
                )
                
                task = InferenceTask(
                    task_key=task_key,
                    int_idx=int_idx,
                    pair_indices=[(ext_idx,)],
                    make_pipeline_fn=pipeline_fn,
                    pipeline_kwargs={},
                    initial_guesses=initial_guesses,
                )
                tasks.append(task)
    
    elif mode == 'cumulative_inference':
        for int_idx, int_params in enumerate(int_params_list):
            task_key = f"int_{int_idx:03d}_cumulative"
            
            # Factory with all ext_params
            pipeline_fn = _make_two_pass_pipeline_factory(
                model_class=model_class,
                ext_params_list=ext_params_list,
                sim_params_list=sim_params_list,
                param_keys_to_infer=param_keys_to_infer,
                elastic_params_list=elastic_params_list,
                viscous_params_list=viscous_params_list,
                loss_fn=loss_fn,
                optimizer=optimizer,
                optimizer_kwargs=optimizer_kwargs,
                n_jobs_per_pass=n_jobs_per_pass,
            )
            
            task = InferenceTask(
                task_key=task_key,
                int_idx=int_idx,
                pair_indices=None,
                make_pipeline_fn=pipeline_fn,
                pipeline_kwargs={},
                initial_guesses=initial_guesses,
            )
            tasks.append(task)
    
    return tasks

# =============== TESTS ===============================

def test_workflow_with_inference():
    """
    Test ViscoElasticFilament inference with:
    - One internal parameter: Sp4
    - One external parameter set: A = 1e-6, w0 = 0
    - One inference pass
    """

    checkpoint_dir = Path("./test_checkpoints_vef")
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    
    workflow = SimulationInferenceWorkflow(checkpoint_dir=checkpoint_dir)
    
    print("\n" + "=" * 80)
    print("TEST: ViscoElasticFilament Single-Pass Inference")
    print("=" * 80)
    
    # =========================================================================
    # PHASE 1: SIMULATION
    # =========================================================================
    print("\nPHASE 1: Running Simulations")
    print("-" * 80)
    
    # Single internal parameter: Sp4 = 1e0
    ground_truth_Sp4 = 1e0
    int_params_list = [
        make_ground_truth_int_params(Sp4=ground_truth_Sp4)
    ]
    
    print(f"Internal parameters:")
    print(f"  Sp4 (ground truth): {ground_truth_Sp4:.4e}")
    print(f"  Keys in int_params: {list(int_params_list[0].keys())}")
    
    # Reduce model to parameters to infer
    param_keys_to_infer = ['Sp4']
    ReducedModel = model_params_only_flow( # TODO: this should eventually loop through int_params_list
        int_params_list[0],
        param_keys_to_infer,
    )    

    # Single external parameter set: A = 1e-6, w0 = 0
    A_value = 1e-6
    w0 = 0
    ext_params = make_ground_truth_ext_params(A=A_value, w0 = 0)
    sim_params = make_sim_params_for_w0(w0=w0)
    
    ext_params_list = [ext_params]
    sim_params_list = [sim_params]
    
    print(f"\nExternal parameters:")
    print(f"  A: {A_value:.4e}")
    print(f"  w0 (via sim_params): 0")
    print(f"  T_span: {sim_params.get('T_span')}")
    
    # Run simulations
    model_lists = workflow.run_simulations(
        int_params_list=int_params_list,
        ext_params_list=ext_params_list,
        sim_params_list=sim_params_list,
        model_class=ReducedModel, # TODO: this should become model_class_list eventually
        n_jobs=1,
    )
    
    print(f"\n✓ Simulations complete")
    print(f"  ModelLists created: {len(model_lists)}")
    
    # Inspect simulation results
    for int_idx, model_list in model_lists.items():
        print(f"\n  int_idx={int_idx}:")
        print(f"    Number of models: {len(model_list.models)}")
        
        for model_idx, model in enumerate(model_list.models):
            print(f"\n    model[{model_idx}]:")
            print(f"      ext_params['A']: {model.ext_params.get('A'):.4e}")
            print(f"      sim_output shape: {model.sim_output.get('value', np.array([])).shape}")
            print(f"      sim_output (first 5 values): {model.sim_output.get('value', np.array([]))[:5]}")

    # =========================================================================
    # PHASE 2: ONE-PASS INFERENCE
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: One-Pass Inference")
    print("-" * 70)
    
    # Define initial guess for Sp4 (intentionally wrong for testing)
    initial_guesses = [{'Sp4': 1e-1}]
    param_keys_to_infer = ['Sp4']
    
    print(f"\nInference setup:")
    print(f"  Parameter to infer: Sp4")
    print(f"  Initial guess: {initial_guesses[0]['Sp4']:.4e}")
    print(f"  Ground truth: {ground_truth_Sp4:.4e}")
    
    # Optimizer setup
    optimizer = basinhopping_optimizer
    bounds = _make_optimizer_bounds(param_keys_to_infer)
    optimizer_kwargs = make_optimizer_kwargs(bounds=bounds)

    # Loss function
    loss_fn = rel_mse_loss_fn()

    # Create inference tasks: one for each int_idx
    
    initial_guesses = [{'Sp4': 1e-1}]
    pipeline_kwargs = dict(
        optimizer = optimizer,
        optimizer_kwargs = optimizer_kwargs,
        loss_fn = loss_fn,
    )
    
    inference_tasks = []
    for int_idx in range(len(int_params_list)):
        task = InferenceTask(
            task_key=f"infer_int_{int_idx}",
            int_idx=int_idx,
            pair_indices=None,  # Use all models in the ModelList
            make_pipeline_fn=make_simple_inference_pipeline,
            pipeline_kwargs=pipeline_kwargs,
            initial_guesses=initial_guesses,
        )
        inference_tasks.append(task)
    
    print(f"\nCreated {len(inference_tasks)} inference task(s)")
    for task in inference_tasks:
        print(f"  - {task.task_key}: int_idx={task.int_idx}")
    
    # Run inferences
    inference_results = workflow.run_inferences(
        inference_tasks=inference_tasks,
        model_lists=model_lists,
        n_jobs=1,
    )
    
    print(f"\n✓ Inferences complete: {len(inference_results)} result(s)")
    
    # =========================================================================
    # PHASE 3: VERIFY INFERENCE RESULTS
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 3: Verification")
    print("-" * 70)
    print(f"\nInference Results:")
    
    all_success = True
    for int_idx in range(len(int_params_list)):
        task_key = f"infer_int_{int_idx}"
        result = inference_results[task_key]
        true_int_params = int_params_list[int_idx]['Sp4']
        inferred_int_params = result[0].params['Sp4']
        error = abs(inferred_int_params - true_int_params)
        success = error < 0.01
        
        print(f"\n  {task_key}:")
        print(f"    True int_params: {true_int_params}")
        print(f"    Inferred int_params: {inferred_int_params:.4f}")
        print(f"    Error: {error:.6f}")
        print(f"    Converged: {result[0].success}")
        print(f"    Final loss: {result[0].loss:.6e}")
        print(f"    Status: {'✓ PASS' if success else '✗ FAIL'}")
        
        all_success = all_success and success
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    checkpoint_status = workflow.get_checkpoint_status()
    print(f"Checkpoint Status:")
    print(f"  Stage: {checkpoint_status.stage}")
    print(f"  Simulation entries: {len(checkpoint_status.simulation_entries)}")
    print(f"  Inference entries: {len(checkpoint_status.inference_entries)}")
    
    if all_success:
        print("\n✓ WORKFLOW TEST PASSED!")
    else:
        print("\n✗ WORKFLOW TEST FAILED!")
    print("=" * 70)
    
    assert all_success, "All inference results should match true parameters"

def test_workflow_with_inference_multi_sp4(sp4_values: list[float] = None):
    """
    Test ViscoElasticFilament inference with:
    - Multiple internal parameters: Sp4 values from a provided list
    - One external parameter set: A = 1e-6, w0 = 0 (shared across all int_params)
    - One inference pass per internal parameter
    
    Args:
        sp4_values: List of Sp4 values to test (default: [0.1, 1.0, 10.0])
    """
    
    if sp4_values is None:
        sp4_values = [0.1, 1.0, 10.0]
    
    checkpoint_dir = Path("./test_checkpoints_vef_multi")
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    
    workflow = SimulationInferenceWorkflow(checkpoint_dir=checkpoint_dir)
    
    print("\n" + "=" * 80)
    print(f"TEST: ViscoElasticFilament Multi-Sp4 Inference ({len(sp4_values)} cases)")
    print("=" * 80)
    
    # =========================================================================
    # PHASE 1: SIMULATION
    # =========================================================================
    print("\nPHASE 1: Running Simulations")
    print("-" * 80)
    
    # Multiple internal parameters: Sp4 values from input list
    int_params_list = [
        make_ground_truth_int_params(Sp4=sp4_val)
        for sp4_val in sp4_values
    ]
    
    print(f"Internal parameters:")
    print(f"  Number of Sp4 values: {len(sp4_values)}")
    for idx, sp4 in enumerate(sp4_values):
        print(f"    [{idx}] Sp4 = {sp4:.4e}")
    print(f"  Keys in int_params: {list(int_params_list[0].keys())}")
    
    # Single external parameter set: A = 1e-6, w0 = 0
    A_value = 1e-6
    w0 = 0
    ext_params = make_ground_truth_ext_params(A=A_value, w0=0)
    sim_params = make_sim_params_for_w0(w0=w0)
    
    print(f"\nExternal parameters (shared across all int_params):")
    print(f"  A: {A_value:.4e}")
    print(f"  w0 (via sim_params): {w0}")
    print(f"  T_span: {sim_params.get('T_span')}")
    
    # For nested loop: ext_params_list and sim_params_list are single-element lists
    # The workflow will pair each int_params with each (ext_params, sim_params)
    ext_params_list = [ext_params]
    sim_params_list = [sim_params]
    
    # Get the reduced model class for inference
    param_keys_to_infer = ['Sp4']
    ReducedModel = model_params_only_flow(
        int_params_list[0],
        param_keys_to_infer,
    )
    
    # Run simulations: outer loop over int_params, inner loop over ext/sim pairs
    model_lists = workflow.run_simulations(
        int_params_list=int_params_list,
        ext_params_list=ext_params_list,
        sim_params_list=sim_params_list,
        model_class=ReducedModel,
        n_jobs=1,
    )
    
    print(f"\n✓ Simulations complete")
    print(f"  ModelLists created: {len(model_lists)}")
    
    # Inspect simulation results
    for int_idx, model_list in model_lists.items():
        sp4_val = sp4_values[int_idx]
        print(f"\n  int_idx={int_idx} (Sp4={sp4_val:.4e}):")
        print(f"    Number of models: {len(model_list.models)}")
        
        for model_idx, model in enumerate(model_list.models):
            print(f"\n    model[{model_idx}]:")
            print(f"      ext_params['A']: {model.ext_params.get('A'):.4e}")
            sim_output = model.sim_output.get('value', np.array([]))
            print(f"      sim_output shape: {sim_output.shape}")
            if len(sim_output) > 0:
                print(f"      sim_output (first 5 values): {sim_output[:5]}")

    # =========================================================================
    # PHASE 2: ONE-PASS INFERENCE FOR EACH Sp4
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: One-Pass Inference for Each Sp4 Value")
    print("-" * 70)
    
    # Initial guess strategy: intentionally offset from ground truth
    initial_guesses = [{'Sp4': 1e-1}]
    
    print(f"\nInference setup:")
    print(f"  Parameter to infer: Sp4")
    print(f"  Initial guess: {initial_guesses[0]['Sp4']:.4e}")
    print(f"  Ground truth values:")
    for idx, sp4 in enumerate(sp4_values):
        print(f"    [{idx}] {sp4:.4e}")
    
    # Optimizer setup
    optimizer = basinhopping_optimizer
    bounds = _make_optimizer_bounds(param_keys_to_infer)
    optimizer_kwargs = make_optimizer_kwargs(bounds=bounds)

    # Loss function
    loss_fn = rel_mse_loss_fn()

    # Create inference tasks: one for each int_idx
    pipeline_kwargs = dict(
        optimizer=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        loss_fn=loss_fn,
    )
    
    inference_tasks = []
    for int_idx in range(len(int_params_list)):
        task = InferenceTask(
            task_key=f"infer_sp4_{int_idx}",
            int_idx=int_idx,
            pair_indices=None,  # Use all models in the ModelList for this int_idx
            make_pipeline_fn=make_simple_inference_pipeline,
            pipeline_kwargs=pipeline_kwargs,
            initial_guesses=initial_guesses,
        )
        inference_tasks.append(task)
    
    print(f"\nCreated {len(inference_tasks)} inference task(s):")
    for idx, task in enumerate(inference_tasks):
        print(f"  - {task.task_key}: int_idx={task.int_idx}, Sp4 (true)={sp4_values[idx]:.4e}")
    
    # Run inferences
    inference_results = workflow.run_inferences(
        inference_tasks=inference_tasks,
        model_lists=model_lists,
        n_jobs=1,
    )
    
    print(f"\n✓ Inferences complete: {len(inference_results)} result(s)")
    
    # =========================================================================
    # PHASE 3: VERIFY INFERENCE RESULTS
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 3: Verification")
    print("-" * 70)
    
    # Create summary table
    results_summary = []
    all_success = True
    
    for int_idx in range(len(int_params_list)):
        task_key = f"infer_sp4_{int_idx}"
        result = inference_results[task_key]
        
        true_sp4 = sp4_values[int_idx]
        inferred_sp4 = result[0].params['Sp4']
        error = abs(inferred_sp4 - true_sp4)
        rel_error = error / true_sp4 if true_sp4 != 0 else 0
        success = error < 0.01  # Absolute tolerance
        
        results_summary.append({
            'int_idx': int_idx,
            'true_sp4': true_sp4,
            'inferred_sp4': inferred_sp4,
            'error': error,
            'rel_error': rel_error,
            'converged': result[0].success,
            'final_loss': result[0].loss,
            'status': '✓ PASS' if success else '✗ FAIL',
        })
        
        all_success = all_success and success
    
    # Print formatted results
    print(f"\nInference Results Summary:")
    print(f"\n{'idx':<4} {'True Sp4':<12} {'Inferred Sp4':<14} {'Abs Error':<12} {'Rel Error':<12} {'Loss':<12} {'Status':<8}")
    print("-" * 90)
    for res in results_summary:
        print(
            f"{res['int_idx']:<4} "
            f"{res['true_sp4']:<12.4e} "
            f"{res['inferred_sp4']:<14.4e} "
            f"{res['error']:<12.6e} "
            f"{res['rel_error']:<12.4%} "
            f"{res['final_loss']:<12.4e} "
            f"{res['status']:<8}"
        )
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    checkpoint_status = workflow.get_checkpoint_status()
    print(f"Checkpoint Status:")
    print(f"  Stage: {checkpoint_status.stage}")
    print(f"  Simulation entries: {len(checkpoint_status.simulation_entries)}")
    print(f"  Inference entries: {len(checkpoint_status.inference_entries)}")
    
    success_count = sum(1 for res in results_summary if res['status'] == '✓ PASS')
    print(f"\nResults: {success_count}/{len(results_summary)} cases passed")
    
    if all_success:
        print("\n✓ WORKFLOW TEST PASSED!")
    else:
        print("\n✗ WORKFLOW TEST FAILED!")
    print("=" * 70)
    
    assert all_success, f"All inference results should match true parameters (within 0.01 tolerance)"
    
    return results_summary

def test_workflow_with_inference_multi_ext_params(a_values: list[float] = None):
    """
    Test ViscoElasticFilament inference with:
    - Single internal parameter: Sp4 = 1.0 (fixed)
    - Multiple external parameters: A values from a provided list
    - One inference pass per external parameter set
    
    Args:
        a_values: List of A (amplitude) values to test (default: [1e-6, 1e-5, 1e-4, 1e-3])
    """
    
    if a_values is None:
        a_values = [1e-6, 1e-5, 1e-4, 1e-3]
    
    checkpoint_dir = Path("./test_checkpoints_vef_multi_ext")
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    
    workflow = SimulationInferenceWorkflow(checkpoint_dir=checkpoint_dir)
    
    print("\n" + "=" * 80)
    print(f"TEST: ViscoElasticFilament Multi-A Inference ({len(a_values)} cases)")
    print("=" * 80)
    
    # =========================================================================
    # PHASE 1: SIMULATION
    # =========================================================================
    print("\nPHASE 1: Running Simulations")
    print("-" * 80)
    
    # Single internal parameter: Sp4 = 1.0 (fixed)
    ground_truth_Sp4 = 1.0
    int_params = make_ground_truth_int_params(Sp4=ground_truth_Sp4)
    int_params_list = [int_params]
    
    print(f"Internal parameters (fixed):")
    print(f"  Sp4: {ground_truth_Sp4:.4e}")
    print(f"  Keys in int_params: {list(int_params.keys())}")
    
    # Multiple external parameter sets: varying A values, w0 = 0
    w0 = 0
    sim_params = make_sim_params_for_w0(w0=w0)
    
    ext_params_list = [
        make_ground_truth_ext_params(A=a_val, w0=0)
        for a_val in a_values
    ]
    sim_params_list = [sim_params] * len(a_values)
    
    print(f"\nExternal parameters (varying A, w0 = 0):")
    print(f"  Number of A values: {len(a_values)}")
    for idx, a_val in enumerate(a_values):
        print(f"    [{idx}] A = {a_val:.4e}")
    print(f"  T_span: {sim_params.get('T_span')}")
    
    # Get the reduced model class for inference
    param_keys_to_infer = ['Sp4']
    ReducedModel = model_params_only_flow(
        int_params,
        param_keys_to_infer,
    )
    
    # Run simulations: outer loop over int_params (single), inner loop over ext/sim pairs
    model_lists = workflow.run_simulations(
        int_params_list=int_params_list,
        ext_params_list=ext_params_list,
        sim_params_list=sim_params_list,
        model_class=ReducedModel,
        n_jobs=1,
    )
    
    print(f"\n✓ Simulations complete")
    print(f"  ModelLists created: {len(model_lists)}")
    
    # Inspect simulation results
    for int_idx, model_list in model_lists.items():
        print(f"\n  int_idx={int_idx} (Sp4={ground_truth_Sp4:.4e}):")
        print(f"    Number of models: {len(model_list.models)}")
        
        for model_idx, model in enumerate(model_list.models):
            a_val = a_values[model_idx]
            print(f"\n    model[{model_idx}] (A={a_val:.4e}):")
            print(f"      ext_params['A']: {model.ext_params.get('A'):.4e}")
            sim_output = model.sim_output.get('value', np.array([]))
            print(f"      sim_output shape: {sim_output.shape}")
            if len(sim_output) > 0:
                print(f"      sim_output (first 5 values): {sim_output[:5]}")

    # =========================================================================
    # PHASE 2: ONE-PASS INFERENCE FOR EACH A VALUE
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: One-Pass Inference for Each A Value")
    print("-" * 70)
    
    # Initial guess for Sp4: intentionally offset from ground truth
    initial_guesses = [{'Sp4': 1e-1}]
    
    print(f"\nInference setup:")
    print(f"  Parameter to infer: Sp4")
    print(f"  Initial guess: {initial_guesses[0]['Sp4']:.4e}")
    print(f"  Ground truth Sp4: {ground_truth_Sp4:.4e}")
    print(f"  A values to test:")
    for idx, a_val in enumerate(a_values):
        print(f"    [{idx}] A = {a_val:.4e}")
    
    # Optimizer setup
    optimizer = basinhopping_optimizer
    bounds = _make_optimizer_bounds(param_keys_to_infer)
    optimizer_kwargs = make_optimizer_kwargs(bounds=bounds)

    # Loss function
    loss_fn = rel_mse_loss_fn()

    # Create inference task: single int_idx with all pair indices (one per ext_params)
    pipeline_kwargs = dict(
        optimizer=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        loss_fn=loss_fn,
    )
    
    # Create one task per A value, using pair_indices to select specific models
    inference_tasks = []
    for ext_idx in range(len(a_values)):
        task = InferenceTask(
            task_key=f"infer_a_{ext_idx}",
            int_idx=0,  # Single internal parameter set
            pair_indices=[ext_idx],  # Select only the model for this A value
            make_pipeline_fn=make_simple_inference_pipeline,
            pipeline_kwargs=pipeline_kwargs,
            initial_guesses=initial_guesses,
        )
        inference_tasks.append(task)
    
    print(f"\nCreated {len(inference_tasks)} inference task(s):")
    for idx, task in enumerate(inference_tasks):
        print(f"  - {task.task_key}: int_idx={task.int_idx}, A={a_values[idx]:.4e}, pair_indices={task.pair_indices}")
    
    # Run inferences
    inference_results = workflow.run_inferences(
        inference_tasks=inference_tasks,
        model_lists=model_lists,
        n_jobs=1,
    )
    
    print(f"\n✓ Inferences complete: {len(inference_results)} result(s)")
    
    # =========================================================================
    # PHASE 3: VERIFY INFERENCE RESULTS
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 3: Verification")
    print("-" * 70)
    
    # Create summary table
    results_summary = []
    all_success = True
    
    for ext_idx in range(len(a_values)):
        task_key = f"infer_a_{ext_idx}"
        result = inference_results[task_key]
        
        a_val = a_values[ext_idx]
        true_sp4 = ground_truth_Sp4
        inferred_sp4 = result[0].params['Sp4']
        error = abs(inferred_sp4 - true_sp4)
        rel_error = error / true_sp4 if true_sp4 != 0 else 0
        success = error < 0.01  # Absolute tolerance
        
        results_summary.append({
            'ext_idx': ext_idx,
            'a_value': a_val,
            'true_sp4': true_sp4,
            'inferred_sp4': inferred_sp4,
            'error': error,
            'rel_error': rel_error,
            'converged': result[0].success,
            'final_loss': result[0].loss,
            'status': '✓ PASS' if success else '✗ FAIL',
        })
        
        all_success = all_success and success
    
    # Print formatted results
    print(f"\nInference Results Summary (Sp4 inference across varying A values):")
    print(f"\n{'idx':<4} {'A':<12} {'True Sp4':<12} {'Inferred Sp4':<14} {'Abs Error':<12} {'Rel Error':<12} {'Loss':<12} {'Status':<8}")
    print("-" * 106)
    for res in results_summary:
        print(
            f"{res['ext_idx']:<4} "
            f"{res['a_value']:<12.4e} "
            f"{res['true_sp4']:<12.4e} "
            f"{res['inferred_sp4']:<14.4e} "
            f"{res['error']:<12.6e} "
            f"{res['rel_error']:<12.4%} "
            f"{res['final_loss']:<12.4e} "
            f"{res['status']:<8}"
        )
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    checkpoint_status = workflow.get_checkpoint_status()
    print(f"Checkpoint Status:")
    print(f"  Stage: {checkpoint_status.stage}")
    print(f"  Simulation entries: {len(checkpoint_status.simulation_entries)}")
    print(f"  Inference entries: {len(checkpoint_status.inference_entries)}")
    
    success_count = sum(1 for res in results_summary if res['status'] == '✓ PASS')
    print(f"\nResults: {success_count}/{len(results_summary)} cases passed")
    
    if all_success:
        print("\n✓ WORKFLOW TEST PASSED!")
    else:
        print("\n✗ WORKFLOW TEST FAILED!")
    print("=" * 70)
    
    assert all_success, f"All inference results should match true Sp4 (within 0.01 tolerance)"
    
    return results_summary

def test_workflow_with_inference_single_task_multi_ext(a_values: list[float] = None):
    """
    Test ViscoElasticFilament inference with:
    - Single internal parameter: Sp4 = 1.0 (fixed)
    - Multiple external parameters: A values from a provided list
    - Single inference task optimizing across ALL external parameters simultaneously
    
    Args:
        a_values: List of A (amplitude) values to test (default: [1e-6, 1e-5, 1e-4, 1e-3])
    """
    
    if a_values is None:
        a_values = [1e-6, 1e-5, 1e-4, 1e-3]
    
    checkpoint_dir = Path("./test_checkpoints_vef_single_task_multi_ext")
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    
    workflow = SimulationInferenceWorkflow(checkpoint_dir=checkpoint_dir)
    
    print("\n" + "=" * 80)
    print(f"TEST: ViscoElasticFilament Single-Task Multi-A Inference ({len(a_values)} models)")
    print("=" * 80)
    
    # =========================================================================
    # PHASE 1: SIMULATION
    # =========================================================================
    print("\nPHASE 1: Running Simulations")
    print("-" * 80)
    
    # Single internal parameter: Sp4 = 1.0 (fixed)
    ground_truth_Sp4 = 1.0
    int_params = make_ground_truth_int_params(Sp4=ground_truth_Sp4)
    int_params_list = [int_params]
    
    print(f"Internal parameters (fixed):")
    print(f"  Sp4: {ground_truth_Sp4:.4e}")
    print(f"  Keys in int_params: {list(int_params.keys())}")
    
    # Multiple external parameter sets: varying A values, w0 = 0
    w0 = 0
    sim_params = make_sim_params_for_w0(w0=w0)
    
    ext_params_list = [
        make_ground_truth_ext_params(A=a_val, w0=0)
        for a_val in a_values
    ]
    sim_params_list = [sim_params] * len(a_values)
    
    print(f"\nExternal parameters (varying A, w0 = 0):")
    print(f"  Number of A values: {len(a_values)}")
    for idx, a_val in enumerate(a_values):
        print(f"    [{idx}] A = {a_val:.4e}")
    print(f"  T_span: {sim_params.get('T_span')}")
    
    # Get the reduced model class for inference
    param_keys_to_infer = ['Sp4']
    ReducedModel = model_params_only_flow(
        int_params,
        param_keys_to_infer,
    )
    
    # Run simulations: outer loop over int_params (single), inner loop over ext/sim pairs
    model_lists = workflow.run_simulations(
        int_params_list=int_params_list,
        ext_params_list=ext_params_list,
        sim_params_list=sim_params_list,
        model_class=ReducedModel,
        n_jobs=1,
    )
    
    print(f"\n✓ Simulations complete")
    print(f"  ModelLists created: {len(model_lists)}")
    
    # Inspect simulation results
    for int_idx, model_list in model_lists.items():
        print(f"\n  int_idx={int_idx} (Sp4={ground_truth_Sp4:.4e}):")
        print(f"    Number of models in ModelList: {len(model_list.models)}")
        
        for model_idx, model in enumerate(model_list.models):
            a_val = a_values[model_idx]
            print(f"\n    model[{model_idx}] (A={a_val:.4e}):")
            print(f"      ext_params['A']: {model.ext_params.get('A'):.4e}")
            sim_output = model.sim_output.get('value', np.array([]))
            print(f"      sim_output shape: {sim_output.shape}")
            if len(sim_output) > 0:
                print(f"      sim_output (first 5 values): {sim_output[:5]}")

    # =========================================================================
    # PHASE 2: SINGLE INFERENCE TASK ACROSS ALL A VALUES
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: Single Inference Task (All Models)")
    print("-" * 70)
    
    # Initial guess for Sp4: intentionally offset from ground truth
    initial_guesses = [{'Sp4': 1e-1}]
    
    print(f"\nInference setup:")
    print(f"  Parameter to infer: Sp4")
    print(f"  Initial guess: {initial_guesses[0]['Sp4']:.4e}")
    print(f"  Ground truth Sp4: {ground_truth_Sp4:.4e}")
    print(f"  Number of models to optimize across: {len(a_values)}")
    print(f"  A values:")
    for idx, a_val in enumerate(a_values):
        print(f"    [{idx}] A = {a_val:.4e}")
    
    # Optimizer setup
    optimizer = basinhopping_optimizer
    bounds = _make_optimizer_bounds(param_keys_to_infer)
    optimizer_kwargs = make_optimizer_kwargs(bounds=bounds)

    # Loss function
    loss_fn = rel_mse_loss_fn()

    # Create single inference task with NO pair_indices (uses all models)
    pipeline_kwargs = dict(
        optimizer=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        loss_fn=loss_fn,
    )
    
    task = InferenceTask(
        task_key="infer_sp4_all_a_values",
        int_idx=0,  # Single internal parameter set
        pair_indices=None,  # Use ALL models in the ModelList
        make_pipeline_fn=make_simple_inference_pipeline,
        pipeline_kwargs=pipeline_kwargs,
        initial_guesses=initial_guesses,
    )
    
    print(f"\nCreated single inference task:")
    print(f"  task_key: {task.task_key}")
    print(f"  int_idx: {task.int_idx}")
    print(f"  pair_indices: {task.pair_indices} (None = use all {len(a_values)} models)")
    
    # Run inference
    inference_results = workflow.run_inferences(
        inference_tasks=[task],
        model_lists=model_lists,
        n_jobs=1,
    )
    
    print(f"\n✓ Inference complete")
    
    # =========================================================================
    # PHASE 3: VERIFY INFERENCE RESULTS
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 3: Verification")
    print("-" * 70)
    
    result = inference_results["infer_sp4_all_a_values"]
    
    true_sp4 = ground_truth_Sp4
    inferred_sp4 = result[0].params['Sp4']
    error = abs(inferred_sp4 - true_sp4)
    rel_error = error / true_sp4 if true_sp4 != 0 else 0
    success = error < 0.01  # Absolute tolerance
    
    print(f"\nInference Results:")
    print(f"  True Sp4: {true_sp4:.4e}")
    print(f"  Inferred Sp4: {inferred_sp4:.4e}")
    print(f"  Absolute Error: {error:.6e}")
    print(f"  Relative Error: {rel_error:.4%}")
    print(f"  Final Loss: {result[0].loss:.6e}")
    print(f"  Converged: {result[0].success}")
    print(f"  Status: {'✓ PASS' if success else '✗ FAIL'}")
    
    # Detailed loss breakdown per A value (if available)
    print(f"\nDetailed Results by A Value:")
    print(f"{'idx':<4} {'A':<12} {'Loss (this A)':<14}")
    print("-" * 30)
    
    # Access per-model losses if available in result structure
    if hasattr(result[0], 'per_model_losses') and result[0].per_model_losses is not None:
        for idx, (a_val, model_loss) in enumerate(zip(a_values, result[0].per_model_losses)):
            print(f"{idx:<4} {a_val:<12.4e} {model_loss:<14.6e}")
    else:
        print("(Per-model loss breakdown not available in result)")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    checkpoint_status = workflow.get_checkpoint_status()
    print(f"Checkpoint Status:")
    print(f"  Stage: {checkpoint_status.stage}")
    print(f"  Simulation entries: {len(checkpoint_status.simulation_entries)}")
    print(f"  Inference entries: {len(checkpoint_status.inference_entries)}")
    
    print(f"\nInference Details:")
    print(f"  Number of external parameter sets (A values): {len(a_values)}")
    print(f"  Number of inference tasks: 1")
    print(f"  Models optimized together: {len(a_values)}")
    
    if success:
        print("\n✓ WORKFLOW TEST PASSED!")
    else:
        print("\n✗ WORKFLOW TEST FAILED!")
    print("=" * 70)
    
    assert success, f"Inferred Sp4 should match true value (within 0.01 tolerance)"
    
    return {
        'task_key': task.task_key,
        'true_sp4': true_sp4,
        'inferred_sp4': inferred_sp4,
        'error': error,
        'rel_error': rel_error,
        'converged': result[0].success,
        'final_loss': result[0].loss,
        'num_models': len(a_values),
        'a_values': a_values,
    }

def test_workflow_with_inference_multi_sp4_tau_b(
    sp4_values: list[float] = None,
    tau_b_values: list[float] = None,
) -> list[dict]:
    """
    Test ViscoElasticFilament one-pass inference with:
    - Multiple internal parameters: All combinations of Sp4 and tau_b values
    - Single external parameter set: A = 1e-6, w0 = 0 (shared across all int_params)
    - One inference pass per internal parameter combination
    - Verify that inference truly loops through all int_params combinations
    
    Args:
        sp4_values: List of Sp4 values to test (default: [0.5, 1.0, 2.0])
        tau_b_values: List of tau_b values to test (default: [0.5, 1.0, 2.0])
    
    Returns:
        List of result dictionaries containing inference outcomes for each int_params
    """
    
    if sp4_values is None:
        sp4_values = [0.5, 1.0, 2.0]
    if tau_b_values is None:
        tau_b_values = [0, 1.0, 2.0]
    
    checkpoint_dir = Path("./test_checkpoints_vef_multi_sp4_tau_b")
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    
    workflow = SimulationInferenceWorkflow(checkpoint_dir=checkpoint_dir)
    
    print("\n" + "=" * 80)
    print(f"TEST: ViscoElasticFilament Multi-Sp4-tau_b One-Pass Inference")
    print(f"      ({len(sp4_values)} Sp4 × {len(tau_b_values)} tau_b = {len(sp4_values) * len(tau_b_values)} cases)")
    print("=" * 80)
    
    # =========================================================================
    # PHASE 1: SIMULATION
    # =========================================================================
    print("\nPHASE 1: Running Simulations")
    print("-" * 80)
    
    # Create all combinations of (Sp4, tau_b)
    int_params_list = [
        make_ground_truth_int_params(Sp4=sp4_val, tau_b=tau_b_val)
        for sp4_val in sp4_values
        for tau_b_val in tau_b_values
    ]
    
    # Build mapping for easy lookup during verification
    int_params_map = {}  # (Sp4, tau_b) -> int_idx
    for int_idx, int_params in enumerate(int_params_list):
        sp4 = int_params.get('Sp4')
        tau_b = int_params.get('tau_b')
        int_params_map[(sp4, tau_b)] = int_idx
    
    print(f"Internal parameters (all combinations):")
    print(f"  Sp4 values: {sp4_values}")
    print(f"  tau_b values: {tau_b_values}")
    print(f"  Total combinations: {len(int_params_list)}")
    print(f"  Keys in int_params: {list(int_params_list[0].keys())}")
    
    print(f"\n  Detailed int_params combinations:")
    for int_idx, int_params in enumerate(int_params_list):
        sp4 = int_params.get('Sp4')
        tau_b = int_params.get('tau_b')
        print(f"    [{int_idx:2d}] Sp4={sp4:.4e}, tau_b={tau_b:.4e}")
    
    # Single external parameter set: A = 1e-6, w0 = 0
    A_value = 1e-6
    w0 = 0.0
    ext_params = make_ground_truth_ext_params(A=A_value, w0=w0)
    sim_params = make_sim_params_for_w0(w0=w0)
    
    print(f"\nExternal parameters (shared across all int_params):")
    print(f"  A: {A_value:.4e}")
    print(f"  w0 (via sim_params): {w0}")
    print(f"  T_span: {sim_params.get('T_span')}")
    print(f"  Number of ext_params sets: 1")
    
    # Single-element lists for ext/sim params
    ext_params_list = [ext_params]
    sim_params_list = [sim_params]
    
    # Get reduced model class (using first int_params as template)
    param_keys_to_infer = ['Sp4', 'tau_b']
    ReducedModel = model_params_only_flow(
        int_params_list[0],
        param_keys_to_infer,
    )
    
    # Run simulations: outer loop over all int_params combinations, 
    # inner loop over single ext/sim pair
    model_lists = workflow.run_simulations(
        int_params_list=int_params_list,
        ext_params_list=ext_params_list,
        sim_params_list=sim_params_list,
        model_class=ReducedModel,
        n_jobs=1,
    )
    
    print(f"\n✓ Simulations complete")
    print(f"  ModelLists created: {len(model_lists)}")
    print(f"  Expected: {len(int_params_list)}")
    assert len(model_lists) == len(int_params_list), \
        f"Mismatch: {len(model_lists)} ModelLists vs {len(int_params_list)} int_params"
    
    # Inspect simulation results to verify loop coverage
    print(f"\n  Simulation results for each int_idx:")
    for int_idx, model_list in model_lists.items():
        sp4_val = int_params_list[int_idx].get('Sp4')
        tau_b_val = int_params_list[int_idx].get('tau_b')
        
        assert len(model_list.models) == 1, \
            f"int_idx={int_idx}: Expected 1 model (1 ext_params), got {len(model_list.models)}"
        
        model = model_list.models[0]
        
        # Verify ext_params matches
        assert abs(model.ext_params.get('A') - A_value) < 1e-15, \
            f"int_idx={int_idx}: ext_params['A'] mismatch"
        
        sim_output = model.sim_output.get('value', np.array([]))
        
        print(f"    int_idx={int_idx:2d} (Sp4={sp4_val:.4e}, tau_b={tau_b_val:.4e}): "
            f"1 model, sim_output shape={sim_output.shape}")
    
    # =========================================================================
    # PHASE 2: ONE-PASS INFERENCE FOR EACH INT_PARAMS
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: One-Pass Inference for Each int_params Combination")
    print("-" * 70)
    
    # Intentionally offset initial guesses from ground truth
    initial_guesses = [
        {'Sp4': 1e-1, 'tau_b': 1e-1}
    ]
    
    print(f"\nInference setup:")
    print(f"  Parameters to infer: {param_keys_to_infer}")
    print(f"  Initial guess: {initial_guesses[0]}")
    print(f"  Number of int_params combinations: {len(int_params_list)}")
    
    # Optimizer setup
    optimizer = basinhopping_optimizer
    bounds = _make_optimizer_bounds(param_keys_to_infer)
    optimizer_kwargs = make_optimizer_kwargs(bounds=bounds)
    loss_fn = rel_mse_loss_fn()
    
    pipeline_kwargs = dict(
        optimizer=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        loss_fn=loss_fn,
    )
    
    # Create inference tasks: one for each int_idx
    inference_tasks = []
    for int_idx in range(len(int_params_list)):
        sp4_true = int_params_list[int_idx].get('Sp4')
        tau_b_true = int_params_list[int_idx].get('tau_b')
        
        task = InferenceTask(
            task_key=f"infer_sp4_tau_b_{int_idx}",
            int_idx=int_idx,
            pair_indices=None,  # Use all models in the ModelList for this int_idx
            make_pipeline_fn=make_simple_inference_pipeline,
            pipeline_kwargs=pipeline_kwargs,
            initial_guesses=initial_guesses,
        )
        inference_tasks.append(task)
    
    print(f"\nCreated {len(inference_tasks)} inference task(s):")
    print(f"{'Task Key':<30} {'int_idx':<8} {'Sp4 (true)':<14} {'tau_b (true)':<14}")
    print("-" * 66)
    for idx, task in enumerate(inference_tasks):
        sp4_true = int_params_list[idx].get('Sp4')
        tau_b_true = int_params_list[idx].get('tau_b')
        print(f"{task.task_key:<30} {task.int_idx:<8} {sp4_true:<14.4e} {tau_b_true:<14.4e}")
    
    # Run inferences
    print(f"\nRunning {len(inference_tasks)} inference task(s)...")
    inference_results = workflow.run_inferences(
        inference_tasks=inference_tasks,
        model_lists=model_lists,
        n_jobs=1,
    )
    
    print(f"\n✓ Inferences complete: {len(inference_results)} result(s)")
    assert len(inference_results) == len(inference_tasks), \
        f"Mismatch: {len(inference_results)} results vs {len(inference_tasks)} tasks"
    
    # =========================================================================
    # PHASE 3: VERIFICATION
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 3: Verification")
    print("-" * 70)
    
    results_summary = []
    all_success = True
    convergence_failures = []
    accuracy_failures = []
    
    abs_tolerance = 0.01
    rel_tolerance = 0.05  # 5%
    
    print(f"\nAccuracy criteria:")
    print(f"  Absolute tolerance: {abs_tolerance}")
    print(f"  Relative tolerance: {rel_tolerance * 100:.1f}%")
    print(f"  (Both must be satisfied for a PASS)")
    
    for int_idx in range(len(int_params_list)):
        task_key = f"infer_sp4_tau_b_{int_idx}"
        
        assert task_key in inference_results, \
            f"Missing result for task_key={task_key}"
        
        result = inference_results[task_key]
        
        # Extract ground truth
        true_sp4 = int_params_list[int_idx].get('Sp4')
        true_tau_b = int_params_list[int_idx].get('tau_b')
        
        # Extract inferred values
        inferred_sp4 = result[0].params['Sp4']
        inferred_tau_b = result[0].params['tau_b']
        
        # Compute errors
        sp4_abs_error = abs(inferred_sp4 - true_sp4)
        sp4_rel_error = sp4_abs_error / true_sp4 if true_sp4 != 0 else 0
        
        tau_b_abs_error = abs(inferred_tau_b - true_tau_b)
        tau_b_rel_error = tau_b_abs_error / true_tau_b if true_tau_b != 0 else 0
        
        # Determine success (both abs and rel within tolerance)
        sp4_success = (sp4_abs_error < abs_tolerance) and (sp4_rel_error < rel_tolerance)
        tau_b_success = (tau_b_abs_error < abs_tolerance) and (tau_b_rel_error < rel_tolerance)
        overall_success = sp4_success and tau_b_success
        
        if not result[0].success:
            convergence_failures.append(int_idx)
        
        if not overall_success:
            accuracy_failures.append(int_idx)
        
        results_summary.append({
            'int_idx': int_idx,
            'true_sp4': true_sp4,
            'true_tau_b': true_tau_b,
            'inferred_sp4': inferred_sp4,
            'inferred_tau_b': inferred_tau_b,
            'sp4_abs_error': sp4_abs_error,
            'sp4_rel_error': sp4_rel_error,
            'tau_b_abs_error': tau_b_abs_error,
            'tau_b_rel_error': tau_b_rel_error,
            'converged': result[0].success,
            'final_loss': result[0].loss,
            'sp4_success': sp4_success,
            'tau_b_success': tau_b_success,
            'overall_success': overall_success,
            'status': '✓ PASS' if overall_success else '✗ FAIL',
        })
        
        all_success = all_success and overall_success
    
    # Print formatted results table
    print(f"\nInference Results Summary:")
    print(f"\n{'idx':<4} {'Sp4 (T/I)':<20} {'tau_b (T/I)':<20} {'Sp4 Error':<15} {'tau_b Error':<15} {'Loss':<12} {'Status':<8}")
    print("-" * 110)
    
    for res in results_summary:
        true_inferred_sp4 = f"{res['true_sp4']:.2e}/{res['inferred_sp4']:.2e}"
        true_inferred_tau_b = f"{res['true_tau_b']:.2e}/{res['inferred_tau_b']:.2e}"
        sp4_error_str = f"{res['sp4_abs_error']:.2e} ({res['sp4_rel_error']:.1%})"
        tau_b_error_str = f"{res['tau_b_abs_error']:.2e} ({res['tau_b_rel_error']:.1%})"
        
        print(
            f"{res['int_idx']:<4} "
            f"{true_inferred_sp4:<20} "
            f"{true_inferred_tau_b:<20} "
            f"{sp4_error_str:<15} "
            f"{tau_b_error_str:<15} "
            f"{res['final_loss']:<12.4e} "
            f"{res['status']:<8}"
        )
    
    # Print detailed failure analysis
    if convergence_failures:
        print(f"\n⚠ Convergence Failures ({len(convergence_failures)}):")
        for int_idx in convergence_failures:
            res = results_summary[int_idx]
            print(f"  int_idx={int_idx}: "
                f"Sp4={res['true_sp4']:.4e}, tau_b={res['true_tau_b']:.4e}, "
                f"final_loss={res['final_loss']:.4e}")
    
    if accuracy_failures:
        print(f"\n⚠ Accuracy Failures ({len(accuracy_failures)}):")
        for int_idx in accuracy_failures:
            res = results_summary[int_idx]
            if not res['sp4_success']:
                print(f"  int_idx={int_idx} (Sp4 fail): "
                    f"true={res['true_sp4']:.4e}, inferred={res['inferred_sp4']:.4e}, "
                    f"abs_error={res['sp4_abs_error']:.4e}, rel_error={res['sp4_rel_error']:.2%}")
            if not res['tau_b_success']:
                print(f"  int_idx={int_idx} (tau_b fail): "
                    f"true={res['true_tau_b']:.4e}, inferred={res['inferred_tau_b']:.4e}, "
                    f"abs_error={res['tau_b_abs_error']:.4e}, rel_error={res['tau_b_rel_error']:.2%}")


def test_workflow_with_inference_two_pass_elastic_viscous(a_values: list[float] = None):
    """
    Test ViscoElasticFilament inference with:
    - Single internal parameter set: Multiple parameters to infer (elastic + viscous)
    - Multiple external parameters: A values from a provided list
    - Two-pass inference: Split parameters by w0 frequency
      * Pass 1 (Elastic): Optimizes elastic params at w0 = min_w0
      * Pass 2 (Viscous): Optimizes viscous params at w0 > 0
    
    Args:
        a_values: List of A (amplitude) values to test (default: [1e-6, 1e-5, 1e-4, 1e-3])
    """
    
    if a_values is None:
        a_values = [1e-6, 1e-5, 1e-4, 1e-3]
    
    checkpoint_dir = Path("./test_checkpoints_vef_two_pass_elastic_viscous")
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    
    workflow = SimulationInferenceWorkflow(checkpoint_dir=checkpoint_dir)
    
    print("\n" + "=" * 80)
    print(f"TEST: ViscoElasticFilament Two-Pass Elastic/Viscous Inference ({len(a_values)} models)")
    print("=" * 80)
    
    print("\nPHASE 1: Running Simulations")
    print("-" * 80)
    
    ground_truth_Sp4 = 1.0
    ground_truth_tau_b = 1.0
    int_params = make_ground_truth_int_params(Sp4=ground_truth_Sp4, tau_b=ground_truth_tau_b)
    int_params_list = [int_params]
    
    print(f"Internal parameters (fixed, to be split across passes):")
    print(f"  Sp4 (elastic): {ground_truth_Sp4:.4e}")
    print(f"  tau_b (viscous): {ground_truth_tau_b:.4e}")
    print(f"  Keys in int_params: {list(int_params.keys())}")
    
    elastic_params_list = ['Sp4']
    viscous_params_list = ['tau_b']
    param_keys_to_infer = ['Sp4', 'tau_b']
    
    # Create two sets of external parameters: one at w0=0 (elastic), one at w0>0 (viscous)
    w0_values = [0.0, 1.0]
    ext_params_list = []
    sim_params_list = []
    
    print(f"\nExternal parameters (varying A and w0):")
    print(f"  Number of A values: {len(a_values)}")
    print(f"  w0 values: {w0_values}")
    
    for w0 in w0_values:
        sim_params = make_sim_params_for_w0(w0=w0)
        for idx, a_val in enumerate(a_values):
            ext_params = make_ground_truth_ext_params(A=a_val, w0=w0)
            ext_params_list.append(ext_params)
            sim_params_list.append(sim_params)
            if w0 == w0_values[0] and idx == 0:
                print(f"  [{len(ext_params_list) - 1}] A={a_val:.4e}, w0={w0} (elastic)")
            elif w0 == w0_values[-1] and idx == 0:
                print(f"  [{len(ext_params_list) - 1}] A={a_val:.4e}, w0={w0} (viscous)")
            if idx == len(a_values) - 1:
                print(f"  ...through [{len(ext_params_list) - 1}] A={a_val:.4e}, w0={w0}")
    
    print(f"  Total external parameter sets: {len(ext_params_list)}")
    
    ReducedModel = model_params_only_flow(
        int_params,
        param_keys_to_infer,
    )
    
    model_lists = workflow.run_simulations(
        int_params_list=int_params_list,
        ext_params_list=ext_params_list,
        sim_params_list=sim_params_list,
        model_class=ReducedModel,
        n_jobs=1,
    )
    
    print(f"\n✓ Simulations complete")
    print(f"  ModelLists created: {len(model_lists)}")
    
    for int_idx, model_list in enumerate(model_lists.items()):
        print(f"\n  int_idx={int_idx} (Sp4={ground_truth_Sp4:.4e}, tau_b={ground_truth_tau_b:.4e}):")
        print(f"    Total models in ModelList: {len(model_list[1].models)}")
        print(f"    Models at w0=0 (elastic): {len(a_values)}")
        print(f"    Models at w0>0 (viscous): {len(a_values)}")

    print("\n" + "=" * 70)
    print("PHASE 2: Two-Pass Inference (Elastic then Viscous)")
    print("-" * 70)
    
    initial_guesses = [{'Sp4': 1e-1, 'tau_b': 0}]
    
    print(f"\nInference setup:")
    print(f"  Parameters to infer: Sp4 (elastic), tau_b (viscous)")
    print(f"  Initial guesses: Sp4={initial_guesses[0]['Sp4']:.4e}, tau_b={initial_guesses[0]['tau_b']:.4e}")
    print(f"  Ground truth: Sp4={ground_truth_Sp4:.4e}, tau_b={ground_truth_tau_b:.4e}")
    print(f"  Number of external parameter sets: {len(ext_params_list)}")
    print(f"    - At w0=0 (elastic pass): {len(a_values)}")
    print(f"    - At w0>0 (viscous pass): {len(a_values)}")
    
    optimizer = basinhopping_optimizer
    bounds = _make_optimizer_bounds(param_keys_to_infer)
    optimizer_kwargs = make_optimizer_kwargs(bounds=bounds)
    loss_fn = rel_mse_loss_fn()

    # Create single inference task with NO pair_indices (uses all models)
    pipeline_kwargs = dict(
        optimizer=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        loss_fn=loss_fn,
        param_keys_to_infer=param_keys_to_infer,
    )
    
    task = InferenceTask(
        task_key="infer_sp4_tau_b",
        int_idx=0,  # Single internal parameter set
        pair_indices=None,  # Use ALL models in the ModelList
        make_pipeline_fn=make_two_pass_pipeline,
        pipeline_kwargs=pipeline_kwargs,
        initial_guesses=initial_guesses,
    )
    
    print(f"\nCreated single inference task:")
    print(f"  task_key: {task.task_key}")
    print(f"  int_idx: {task.int_idx}")
    print(f"  pair_indices: {task.pair_indices} (None = use all {len(a_values)} models)")
    
    # Run inference
    inference_results = workflow.run_inferences(
        inference_tasks=[task],
        model_lists=model_lists,
        n_jobs=1,
    )
    
    print(f"\n✓ Two-pass inference complete")
    
    print("\n" + "=" * 70)
    print("PHASE 3: Verification")
    print("-" * 70)
    
    # For demonstration, we'll check if parameters converged
    # (In practice, you'd extract results from the pipeline execution)
    print(f"\nInference Results Summary:")
    print(f"  True Sp4 (elastic): {ground_truth_Sp4:.4e}")
    print(f"  True tau_b (viscous): {ground_truth_tau_b:.4e}")
    print(f"  Pass 1 (Elastic): Optimized Sp4 on {len(a_values)} models at w0=min_w0")
    print(f"  Pass 2 (Viscous): Optimized tau_b on {len(a_values)} models at w0>0")
    
    print(f"\nTwo-Pass Strategy Benefits:")
    print(f"  ✓ Elastic params (Sp4) optimized only on elastic data (w0=min_w0)")
    print(f"  ✓ Viscous params (tau_b) optimized only on viscous data (w0>0)")
    print(f"  ✓ Reduces parameter space per pass → faster convergence")
    print(f"  ✓ Better physical separation of elastic vs. viscous response")
    
    print("\n" + "=" * 70)
    checkpoint_status = workflow.get_checkpoint_status()
    print(f"Checkpoint Status:")
    print(f"  Stage: {checkpoint_status.stage}")
    print(f"  Simulation entries: {len(checkpoint_status.simulation_entries)}")
    print(f"  Inference entries: {len(checkpoint_status.inference_entries)}")
    
    print(f"\nPipeline Configuration:")
    print(f"  Number of internal parameter sets: 1")
    print(f"  Number of external parameter sets: {len(ext_params_list)}")
    print(f"  Parameters split across passes: {param_keys_to_infer}")
    
    print("\n✓ TWO-PASS WORKFLOW TEST COMPLETED!")
    print("=" * 70)
    
    return {
        'task_key': 'two_pass_elastic_viscous',
        'elastic_params': elastic_params_list,
        'viscous_params': viscous_params_list,
        'true_sp4': ground_truth_Sp4,
        'true_tau_b': ground_truth_tau_b,
        'num_models': len(ext_params_list),
        'models_per_pass': len(a_values),
        'a_values': a_values,
        'w0_values': w0_values,
    }

def test_workflow_with_inference_two_pass_elastic_viscous_multi_int_params(
    sp4_values: list[float] = None,
    tau_b_values: list[float] = None,
    a_values: list[float] = None,
):
    """
    Test ViscoElasticFilament two-pass inference with:
    - Multiple internal parameters: Cartesian product of Sp4 and tau_b values
    - Multiple external parameters: A values from a provided list
    - Two-pass inference per int_params: Split elastic (Sp4) and viscous (tau_b) params
      * Pass 1 (Elastic): Optimizes Sp4 at w0 = min_w0
      * Pass 2 (Viscous): Optimizes tau_b at w0 > 0
    - Verification that all int_params combinations are simulated and inferred
    
    Args:
        sp4_values: List of Sp4 values to test (default: [0.5, 1.0, 2.0])
        tau_b_values: List of tau_b values to test (default: [0.5, 1.0, 2.0])
        a_values: List of A (amplitude) values to test (default: [1e-6, 1e-5, 1e-4])
    """
    
    if sp4_values is None:
        sp4_values = [1.0]
    if tau_b_values is None:
        tau_b_values = [0, 1.0]
    if a_values is None:
        a_values = [1e-6]
    
    checkpoint_dir = Path("./test_checkpoints_vef_two_pass_multi_int")
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    
    workflow = SimulationInferenceWorkflow(checkpoint_dir=checkpoint_dir)
    
    print("\n" + "=" * 80)
    print(f"TEST: ViscoElasticFilament Two-Pass Multi-Int-Params Inference")
    print(f"      (Sp4: {len(sp4_values)} × tau_b: {len(tau_b_values)} = {len(sp4_values) * len(tau_b_values)} cases)")
    print("=" * 80)
    
    print("\nPHASE 0: Parameter Space Setup")
    print("-" * 80)
    
    print(f"\nInternal Parameters (Cartesian product):")
    print(f"  Sp4 values: {sp4_values}")
    print(f"  tau_b values: {tau_b_values}")
    print(f"  Total combinations: {len(sp4_values) * len(tau_b_values)}")
    
    # Create all combinations of (Sp4, tau_b)
    int_params_list = []
    int_params_metadata = []  # Track (sp4, tau_b) for each entry
    
    for sp4 in sp4_values:
        for tau_b in tau_b_values:
            int_params = make_ground_truth_int_params(Sp4=sp4, tau_b=tau_b)
            int_params_list.append(int_params)
            int_params_metadata.append({'Sp4': sp4, 'tau_b': tau_b})
    
    print(f"\n  Generated {len(int_params_list)} int_params:")
    for idx, metadata in enumerate(int_params_metadata):
        print(f"    [{idx}] Sp4={metadata['Sp4']:.4e}, tau_b={metadata['tau_b']:.4e}")
    
    # External parameters: vary A across two w0 values
    w0_values = [0.0, 1.0]
    ext_params_list = []
    sim_params_list = []
    
    print(f"\nExternal Parameters:")
    print(f"  A values: {a_values} ({len(a_values)} values)")
    print(f"  w0 values: {w0_values}")
    print(f"  Total ext_params per int_params: {len(a_values) * len(w0_values)}")
    
    for w0 in w0_values:
        sim_params = make_sim_params_for_w0(w0=w0)
        for idx, a_val in enumerate(a_values):
            ext_params = make_ground_truth_ext_params(A=a_val, w0=w0)
            ext_params_list.append(ext_params)
            sim_params_list.append(sim_params)
    
    print(f"  Total external parameter sets: {len(ext_params_list)}")
    
    elastic_params_list = ['Sp4']
    viscous_params_list = ['tau_b']
    param_keys_to_infer = ['Sp4', 'tau_b']
    
    print(f"\nParameters to infer:")
    print(f"  Elastic (Pass 1): {elastic_params_list}")
    print(f"  Viscous (Pass 2): {viscous_params_list}")
    
    print("\n" + "=" * 80)
    print("PHASE 1: Running Simulations for All Int-Params Combinations")
    print("-" * 80)
    
    ReducedModel = model_params_only_flow(
        int_params_list[0],
        param_keys_to_infer,
    )
    
    model_lists = workflow.run_simulations(
        int_params_list=int_params_list,
        ext_params_list=ext_params_list,
        sim_params_list=sim_params_list,
        model_class=ReducedModel,
        n_jobs=1,
    )
    
    print(f"\n✓ Simulations complete")
    print(f"  Total ModelLists created: {len(model_lists)}")
    print(f"  Expected: {len(int_params_list)}")
    
    # Verify all int_params were simulated
    assert len(model_lists) == len(int_params_list), \
        f"Expected {len(int_params_list)} ModelLists, got {len(model_lists)}"
    
    print(f"\nDetailed ModelList breakdown:")
    for int_idx, (key, model_list) in enumerate(model_lists.items()):
        metadata = int_params_metadata[int_idx]
        num_models = len(model_list.models)
        expected_models = len(ext_params_list)
        
        print(f"\n  int_idx={int_idx} (Sp4={metadata['Sp4']:.4e}, tau_b={metadata['tau_b']:.4e}):")
        print(f"    Models in this ModelList: {num_models}")
        print(f"    Expected: {expected_models}")
        print(f"    Models per w0 group: {len(a_values)}")
        
        assert num_models == expected_models, \
            f"int_idx {int_idx}: expected {expected_models} models, got {num_models}"
        
        # Verify models are ordered: first all w0=0, then all w0>0
        for model_idx, model in enumerate(model_list.models):
            a_val = model.ext_params.get('A')
            w0_idx = model_idx // len(a_values)  # Which w0 group
            expected_w0 = w0_values[w0_idx]
            actual_w0 = model.ext_params.get('w0')
            
            if model_idx < len(a_values):
                assert actual_w0 == 0.0, \
                    f"int_idx {int_idx}, model_idx {model_idx}: expected w0=0, got {actual_w0}"
            else:
                assert actual_w0 > 0, \
                    f"int_idx {int_idx}, model_idx {model_idx}: expected w0>0, got {actual_w0}"
    
    print(f"\n✓ All int_params successfully simulated with correct model ordering")
    
    print("\n" + "=" * 80)
    print("PHASE 2: Two-Pass Inference for Each Int-Params Combination")
    print("-" * 80)
    
    initial_guesses = [{'Sp4': 1e-1, 'tau_b': 0}]
    
    optimizer = basinhopping_optimizer
    bounds = _make_optimizer_bounds(param_keys_to_infer)
    optimizer_kwargs = make_optimizer_kwargs(bounds=bounds)
    loss_fn = rel_mse_loss_fn()
    
    pipeline_kwargs = dict(
        optimizer=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        loss_fn=loss_fn,
        param_keys_to_infer=param_keys_to_infer,
    )
    
    # Create one inference task per int_params combination
    inference_tasks = []
    for int_idx in range(len(int_params_list)):
        metadata = int_params_metadata[int_idx]
        task_key = f"infer_sp4_{metadata['Sp4']:.1e}_tau_b_{metadata['tau_b']:.1e}"
        
        task = InferenceTask(
            task_key=task_key,
            int_idx=int_idx,
            pair_indices=None,  # Use all models for this int_params
            make_pipeline_fn=make_two_pass_pipeline,
            pipeline_kwargs=pipeline_kwargs,
            initial_guesses=initial_guesses,
        )
        inference_tasks.append(task)
    
    print(f"\nCreated {len(inference_tasks)} inference task(s):")
    for idx, task in enumerate(inference_tasks):
        metadata = int_params_metadata[idx]
        print(f"  [{idx}] {task.task_key}")
        print(f"       True: Sp4={metadata['Sp4']:.4e}, tau_b={metadata['tau_b']:.4e}")
    
    # Run all inference tasks
    inference_results = workflow.run_inferences(
        inference_tasks=inference_tasks,
        model_lists=model_lists,
        n_jobs=1,
    )
    
    print(f"\n✓ All two-pass inferences complete: {len(inference_results)} result(s)")
    
    print("\n" + "=" * 80)
    print("PHASE 3: Verification & Results Summary")
    print("-" * 80)
    
    results_summary = []
    all_success = True
    
    print(f"\nDetailed Results:")
    print(f"\n{'idx':<4} {'True Sp4':<12} {'True tau_b':<12} {'Inferred Sp4':<14} {'Inferred tau_b':<14} {'Sp4 Rel Error':<14} {'tau_b Rel Error':<14} {'Status':<8}")
    print("-" * 110)
    
    for int_idx in range(len(int_params_list)):
        metadata = int_params_metadata[int_idx]
        true_sp4 = metadata['Sp4']
        true_tau_b = metadata['tau_b']
        
        task_key = f"infer_sp4_{true_sp4:.1e}_tau_b_{true_tau_b:.1e}"
        result = inference_results[task_key]
        
        inferred_sp4 = result[0].params['Sp4']
        inferred_tau_b = result[0].params['tau_b']
        
        sp4_error = abs(inferred_sp4 - true_sp4)
        sp4_rel_error = sp4_error / true_sp4 if true_sp4 != 0 else 0
        
        tau_b_error = abs(inferred_tau_b - true_tau_b)
        tau_b_rel_error = tau_b_error / true_tau_b if true_tau_b != 0 else 0
        
        # Success if both parameters are within 10% relative error
        success = (sp4_rel_error < 0.1) and (tau_b_rel_error < 0.1)
        
        results_summary.append({
            'int_idx': int_idx,
            'true_sp4': true_sp4,
            'true_tau_b': true_tau_b,
            'inferred_sp4': inferred_sp4,
            'inferred_tau_b': inferred_tau_b,
            'sp4_rel_error': sp4_rel_error,
            'tau_b_rel_error': tau_b_rel_error,
            'converged': result[0].success,
            'final_loss': result[0].loss,
            'status': '✓ PASS' if success else '✗ FAIL',
        })
        
        print(
            f"{int_idx:<4} "
            f"{true_sp4:<12.4e} "
            f"{true_tau_b:<12.4e} "
            f"{inferred_sp4:<14.4e} "
            f"{inferred_tau_b:<14.4e} "
            f"{sp4_rel_error:<14.4%} "
            f"{tau_b_rel_error:<14.4%} "
            f"{results_summary[-1]['status']:<8}"
        )
        
        all_success = all_success and success
    
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("-" * 80)
    
    passed = sum(1 for res in results_summary if res['status'] == '✓ PASS')
    failed = len(results_summary) - passed
    
    print(f"\nResults: {passed}/{len(results_summary)} cases passed")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    
    avg_sp4_rel_error = sum(res['sp4_rel_error'] for res in results_summary) / len(results_summary)
    avg_tau_b_rel_error = sum(res['tau_b_rel_error'] for res in results_summary) / len(results_summary)
    
    print(f"\nAverage Relative Errors:")
    print(f"  Sp4:   {avg_sp4_rel_error:.4%}")
    print(f"  tau_b: {avg_tau_b_rel_error:.4%}")
    
    print(f"\nParameter Space Coverage:")
    print(f"  Int_params combinations: {len(int_params_list)}")
    print(f"  External param sets: {len(ext_params_list)}")
    print(f"  Total models across all int_params: {len(int_params_list) * len(ext_params_list)}")
    
    print(f"\nTwo-Pass Strategy per Int-Params:")
    print(f"  ✓ Pass 1 (Elastic): Optimizes Sp4 on {len(a_values)} models at w0=0")
    print(f"  ✓ Pass 2 (Viscous): Optimizes tau_b on {len(a_values)} models at w0>0")
    
    print("\n" + "=" * 80)
    checkpoint_status = workflow.get_checkpoint_status()
    print(f"Checkpoint Status:")
    print(f"  Stage: {checkpoint_status.stage}")
    print(f"  Simulation entries: {len(checkpoint_status.simulation_entries)}")
    print(f"    Expected: {len(int_params_list)}")
    print(f"  Inference entries: {len(checkpoint_status.inference_entries)}")
    print(f"    Expected: {len(inference_tasks)}")
    
    assert len(checkpoint_status.simulation_entries) == len(int_params_list), \
        f"Expected {len(int_params_list)} simulation entries, got {len(checkpoint_status.simulation_entries)}"
    assert len(checkpoint_status.inference_entries) == len(inference_tasks), \
        f"Expected {len(inference_tasks)} inference entries, got {len(checkpoint_status.inference_entries)}"
    
    print("\n" + "=" * 80)
    if all_success:
        print("✓ TWO-PASS MULTI-INT-PARAMS WORKFLOW TEST PASSED!")
    else:
        print("✗ TWO-PASS MULTI-INT-PARAMS WORKFLOW TEST FAILED!")
    print("=" * 80)
    
    assert all_success, f"All inference results should match true parameters (within 10% rel error)"
    
    return {
        'results_summary': results_summary,
        'sp4_values': sp4_values,
        'tau_b_values': tau_b_values,
        'a_values': a_values,
        'w0_values': w0_values,
        'num_int_params': len(int_params_list),
        'num_ext_params': len(ext_params_list),
        'total_models': len(int_params_list) * len(ext_params_list),
        'passed': passed,
        'failed': failed,
    }

def test_workflow_elastic_viscous_condensed(
    sp4_values: list[float] = None,
    tau_b_values: list[float] = None,
    a_values: list[float] = None,
):
    """
    Test ViscoElasticFilament two-pass inference with:
    - Multiple internal parameters: Cartesian product of Sp4 and tau_b values
    - Multiple external parameters: A values from a provided list
    - Two-pass inference per int_params: Split elastic (Sp4) and viscous (tau_b) params
      * Pass 1 (Elastic): Optimizes Sp4 at w0 = min_w0
      * Pass 2 (Viscous): Optimizes tau_b at w0 > 0
    - Verification that all int_params combinations are simulated and inferred
    
    Args:
        sp4_values: List of Sp4 values to test (default: [1.0])
        tau_b_values: List of tau_b values to test (default: [0, 1.0])
        a_values: List of A (amplitude) values to test (default: [1e-6])
    """
    
    if sp4_values is None:
        sp4_values = [1.0]
    if tau_b_values is None:
        tau_b_values = [0, 1.0]
    if a_values is None:
        a_values = [1e-6]
    
    checkpoint_dir = Path("./test_checkpoints_vef_two_pass_multi_int")
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    
    workflow = SimulationInferenceWorkflow(checkpoint_dir=checkpoint_dir)
    
    print("\n" + "=" * 80)
    print(f"TEST: ViscoElasticFilament Two-Pass Multi-Int-Params Inference")
    print(f"      (Sp4: {len(sp4_values)} × tau_b: {len(tau_b_values)} = {len(sp4_values) * len(tau_b_values)} cases)")
    print("=" * 80)
    
    # PHASE 0: Parameter Space Setup
    print("\nPHASE 0: Parameter Space Setup")
    print("-" * 80)
    
    int_params_list, int_params_metadata = _setup_internal_params(sp4_values, tau_b_values)
    w0_values = [0.0, 1.0]
    ext_params_list, sim_params_list = _setup_external_params(w0_values, a_values)
    
    elastic_params_list = ['Sp4']
    viscous_params_list = ['tau_b']
    param_keys_to_infer = ['Sp4', 'tau_b']
    
    print(f"\nInternal Parameters (Cartesian product):")
    print(f"  Sp4 values: {sp4_values}")
    print(f"  tau_b values: {tau_b_values}")
    print(f"  Total combinations: {len(int_params_list)}")
    _print_int_params_summary(int_params_metadata)
    
    print(f"\nExternal Parameters:")
    print(f"  A values: {a_values} ({len(a_values)} values)")
    print(f"  w0 values: {w0_values}")
    print(f"  Total external parameter sets: {len(ext_params_list)}")
    
    print(f"\nParameters to infer:")
    print(f"  Elastic (Pass 1): {elastic_params_list}")
    print(f"  Viscous (Pass 2): {viscous_params_list}")
    
    # PHASE 1: Run Simulations
    print("\n" + "=" * 80)
    print("PHASE 1: Running Simulations for All Int-Params Combinations")
    print("-" * 80)
    
    ReducedModel = model_params_only_flow(
        int_params_list[0],
        param_keys_to_infer,
    )
    
    model_lists = workflow.run_simulations(
        int_params_list=int_params_list,
        ext_params_list=ext_params_list,
        sim_params_list=sim_params_list,
        model_class=ReducedModel,
        n_jobs=1,
    )
    
    print(f"\n✓ Simulations complete")
    print(f"  Total ModelLists created: {len(model_lists)}")
    print(f"  Expected: {len(int_params_list)}")
    
    assert len(model_lists) == len(int_params_list)
    
    _verify_model_lists(model_lists, int_params_metadata, w0_values, len(a_values))
    print(f"\n✓ All int_params successfully simulated with correct model ordering")
    
    # PHASE 2: Two-Pass Inference
    print("\n" + "=" * 80)
    print("PHASE 2: Two-Pass Inference for Each Int-Params Combination")
    print("-" * 80)
    
    initial_guesses = [{'Sp4': 1e-1, 'tau_b': 0}]
    optimizer = basinhopping_optimizer
    bounds = _make_optimizer_bounds(param_keys_to_infer)
    optimizer_kwargs = make_optimizer_kwargs(bounds=bounds)
    loss_fn = rel_mse_loss_fn()
    
    inference_tasks = make_inference_tasks_two_pass(
        mode='cumulative_inference',
        int_params_list=int_params_list,
        ext_params_list=ext_params_list,
        param_keys_to_infer=param_keys_to_infer,
        elastic_params_list=elastic_params_list,
        viscous_params_list=viscous_params_list,
        make_sim_params_fn=make_sim_params_for_w0,
        model_class=ReducedModel,
        loss_fn=loss_fn,
        optimizer=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        initial_guesses=initial_guesses,
        n_jobs_per_pass=-1,
    )
    
    print(f"\nCreated {len(inference_tasks)} inference task(s)")
    _print_inference_tasks(inference_tasks, int_params_metadata)
    
    inference_results = workflow.run_inferences(
        inference_tasks=inference_tasks,
        model_lists=model_lists,
        n_jobs=1,
    )
    
    print(f"\n✓ All two-pass inferences complete: {len(inference_results)} result(s)")
    
    # PHASE 3: Verification & Results Summary
    print("\n" + "=" * 80)
    print("PHASE 3: Verification & Results Summary")
    print("-" * 80)
    
    results_summary = _compute_inference_results(
        inference_results, 
        int_params_metadata, 
        inference_tasks
    )
    
    _print_results_table(results_summary)
    _print_summary_statistics(results_summary, int_params_list, ext_params_list, len(a_values))
    
    # Final checkpoint verification
    print("\n" + "=" * 80)
    checkpoint_status = workflow.get_checkpoint_status()
    print(f"Checkpoint Status:")
    print(f"  Stage: {checkpoint_status.stage}")
    print(f"  Simulation entries: {len(checkpoint_status.simulation_entries)}")
    print(f"    Expected: {len(int_params_list)}")
    print(f"  Inference entries: {len(checkpoint_status.inference_entries)}")
    print(f"    Expected: {len(inference_tasks)}")
    
    assert len(checkpoint_status.simulation_entries) == len(int_params_list)
    assert len(checkpoint_status.inference_entries) == len(inference_tasks)
    
    all_success = all(res['status'] == '✓ PASS' for res in results_summary)
    
    print("\n" + "=" * 80)
    if all_success:
        print("✓ TWO-PASS MULTI-INT-PARAMS WORKFLOW TEST PASSED!")
    else:
        print("✗ TWO-PASS MULTI-INT-PARAMS WORKFLOW TEST FAILED!")
    print("=" * 80)
    
    assert all_success, "All inference results should match true parameters (within 10% rel error)"
    
    return {
        'results_summary': results_summary,
        'sp4_values': sp4_values,
        'tau_b_values': tau_b_values,
        'a_values': a_values,
        'w0_values': w0_values,
        'num_int_params': len(int_params_list),
        'num_ext_params': len(ext_params_list),
        'total_models': len(int_params_list) * len(ext_params_list),
        'passed': sum(1 for res in results_summary if res['status'] == '✓ PASS'),
        'failed': sum(1 for res in results_summary if res['status'] == '✗ FAIL'),
    }

def test_workflow_elastic_viscous_general(
    int_param_ranges: Dict[str, list] = None,
    ext_param_ranges: Dict[str, list] = None,
    inference_mode: str = 'cumulative_inference',
    elastic_params_list: list = None,
    viscous_params_list: list = None,
    optimizer = None,
    optimizer_kwargs: Dict = None,
    n_jobs_per_pass: int = -1,
    n_jobs_simulation: int = 1,
    n_jobs_inference: int = 1,
):
    """
    Generalized two-pass inference workflow with flexible parameter configuration.
    
    Supports:
    - Multiple internal parameters: Any combination of Sp4, Beta, tau_b, tau_s, etc.
    - Multiple external parameters: Any combination of A, w0, psi, etc.
    - Flexible inference mode: 'single_inference' or 'cumulative_inference'
    - Customizable optimizer and loss function
    
    Args:
        int_param_ranges: Dict mapping param names to lists of values.
            E.g., {'Sp4': [1.0], 'tau_b': [0, 1.0], 'Beta': [0.5, 1.0]}
            Default: {'Sp4': [1.0], 'tau_b': [0, 1.0]}
        
        ext_param_ranges: Dict mapping param names to lists of values.
            E.g., {'A': [1e-6], 'w0': [0.0, 1.0], 'psi': [0, 45]}
            Default: {'A': [1e-6], 'w0': [0.0, 1.0]}
        
        inference_mode: 'single_inference' or 'cumulative_inference'
            Default: 'cumulative_inference'
        
        elastic_params_list: List of parameters to optimize in pass 1 (at min w0).
            E.g., ['Sp4', 'Beta']
            Default: ['Sp4']
        
        viscous_params_list: List of parameters to optimize in pass 2 (at w0 > 0).
            E.g., ['tau_b', 'tau_s']
            Default: ['tau_b']
        
        optimizer: Optimizer function (e.g., basinhopping_optimizer)
            Default: basinhopping_optimizer
        
        optimizer_kwargs: Dict of kwargs for optimizer.
            If None, auto-generated from param bounds.
        
        n_jobs_per_pass: Parallelization for each inference pass. -1 = use all cores.
            Default: -1
        
        n_jobs_simulation: Parallelization for simulation phase. -1 = use all cores.
            Default: 1
        
        n_jobs_inference: Parallelization for inference tasks. -1 = use all cores.
            Default: 1
    
    Returns:
        Dict containing:
            - results_summary: Detailed results for each int_params combination
            - int_param_ranges: Input internal parameter ranges
            - ext_param_ranges: Input external parameter ranges
            - inference_mode: Mode used
            - passed: Number of passing tests
            - failed: Number of failing tests
            - total_models: Total models generated
    """
    
    # ========== Defaults ==========
    if int_param_ranges is None:
        int_param_ranges = {'Sp4': [1.0], 'tau_b': [0, 1.0]}
    
    if ext_param_ranges is None:
        ext_param_ranges = {'A': [1e-6], 'w0': [0.0, 1.0]}
    
    if elastic_params_list is None:
        elastic_params_list = ['Sp4']
    
    if viscous_params_list is None:
        viscous_params_list = ['tau_b']
    
    if optimizer is None:
        optimizer = basinhopping_optimizer
    
    # ========== Setup & Validation ==========
    assert inference_mode in ['single_inference', 'cumulative_inference'], \
        f"inference_mode must be 'single_inference' or 'cumulative_inference', got {inference_mode}"
    
    param_keys_to_infer = list(set(elastic_params_list) | set(viscous_params_list))
    
    checkpoint_dir = Path(f"./test_checkpoints_vef_two_pass_general_{inference_mode}")
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    
    workflow = SimulationInferenceWorkflow(checkpoint_dir=checkpoint_dir)
    
    print("\n" + "=" * 80)
    print(f"TEST: ViscoElasticFilament Two-Pass General Workflow")
    print(f"      Mode: {inference_mode}")
    print(f"      Internal parameters: {list(int_param_ranges.keys())}")
    print(f"      External parameters: {list(ext_param_ranges.keys())}")
    print("=" * 80)
    
    # PHASE 0: Parameter Space Setup
    print("\nPHASE 0: Parameter Space Setup")
    print("-" * 80)
    
    # Create Cartesian product of internal parameters
    int_param_names = list(int_param_ranges.keys())
    int_param_values = [int_param_ranges[name] for name in int_param_names]
    int_params_list = [
        dict(zip(int_param_names, combo))
        for combo in product(*int_param_values)
    ]

    int_params_list = [make_ground_truth_int_params(**int_params) for int_params in int_params_list]

    # Create metadata for ground truth lookup
    int_params_metadata = [
        {name: params[name] for name in int_param_names}
        for params in int_params_list
    ]
    
    print(f"\nInternal Parameters (Cartesian product):")
    for name, values in int_param_ranges.items():
        print(f"  {name}: {values} ({len(values)} values)")
    print(f"  Total combinations: {len(int_params_list)}")
    _print_int_params_summary(int_params_metadata)

    # Create external parameters based on which ext_params are "special"
    # (i.e., need special handling like w0 for determining passes)
    ext_param_names = list(ext_param_ranges.keys())
    ext_param_values = [ext_param_ranges[name] for name in ext_param_names]
    
    ext_params_list = [
        dict(zip(ext_param_names, combo))
        for combo in product(*ext_param_values)
    ]
    ext_params_list = [make_ground_truth_ext_params(**ext_params) for ext_params in ext_params_list]
    
    # Create sim_params_list: should mirror ext_params but with w0 split logic
    
    # Check if 'w0' is in external parameters
    if 'w0' in ext_param_ranges:
        # Build sim_params_list using make_sim_params_for_w0 logic
        sim_params_list = []
        for ext_params in ext_params_list:
            w0_value = ext_params.get('w0')
            # make_sim_params_for_w0 returns sim_params dict based on w0
            sim_params = make_sim_params_for_w0(w0_value)
            sim_params_list.append(sim_params)
    else: # Fill with w0 = 0
        w0_value = 0
        sim_params_list = [make_sim_params_for_w0(w0) for ep in ext_params_list]

    print(f"\nExternal Parameters:")
    for name, values in ext_param_ranges.items():
        print(f"  {name}: {values} ({len(values)} values)")
    print(f"  Total external parameter sets: {len(ext_params_list)}")
    
    print(f"\nParameters to infer:")
    print(f"  Elastic (Pass 1, at min w0): {elastic_params_list}")
    print(f"  Viscous (Pass 2, at w0 > 0): {viscous_params_list}")
    print(f"  All: {param_keys_to_infer}")
    
    # PHASE 1: Run Simulations
    print("\n" + "=" * 80)
    print("PHASE 1: Running Simulations for All Int-Params Combinations")
    print("-" * 80)
    
    ReducedModel = model_params_only_flow(
        int_params_list[0],
        param_keys_to_infer,
    )
    
    model_lists = workflow.run_simulations(
        int_params_list=int_params_list,
        ext_params_list=ext_params_list,
        sim_params_list=sim_params_list,
        model_class=ReducedModel,
        n_jobs=n_jobs_simulation,
    )
    
    print(f"\n✓ Simulations complete")
    print(f"  Total ModelLists created: {len(model_lists)}")
    print(f"  Expected: {len(int_params_list)}")
    
    assert len(model_lists) == len(int_params_list)
    
    # PHASE 2: Two-Pass Inference
    print("\n" + "=" * 80)
    print("PHASE 2: Two-Pass Inference for Each Int-Params Combination")
    print("-" * 80)
    
    # Create initial guesses for all params to infer
    initial_guesses = {
        param: 1e-1 if param.startswith('Sp') or param.startswith('Beta') else 0
        for param in param_keys_to_infer
    }
    initial_guesses_list = [initial_guesses]
    
    # Build optimizer_kwargs if not provided
    if optimizer_kwargs is None:
        optimizer_kwargs = make_optimizer_kwargs(
            bounds=_make_optimizer_bounds(param_keys_to_infer)
        )
    
    loss_fn = rel_mse_loss_fn()
    
    inference_tasks = make_inference_tasks_two_pass(
        mode=inference_mode,
        int_params_list=int_params_list,
        ext_params_list=ext_params_list,
        param_keys_to_infer=param_keys_to_infer,
        elastic_params_list=elastic_params_list,
        viscous_params_list=viscous_params_list,
        make_sim_params_fn=make_sim_params_for_w0,
        model_class=ReducedModel,
        loss_fn=loss_fn,
        optimizer=optimizer,
        optimizer_kwargs=optimizer_kwargs,
        initial_guesses=initial_guesses_list,
        n_jobs_per_pass=n_jobs_per_pass,
    )
    
    print(f"\nCreated {len(inference_tasks)} inference task(s)")
    _print_inference_tasks(inference_tasks, int_params_metadata, param_keys_to_infer)
    
    inference_results = workflow.run_inferences(
        inference_tasks=inference_tasks,
        model_lists=model_lists,
        n_jobs=n_jobs_inference,
    )
    
    print(f"\n✓ All two-pass inferences complete: {len(inference_results)} result(s)")
    
    # PHASE 3: Verification & Results Summary
    print("\n" + "=" * 80)
    print("PHASE 3: Verification & Results Summary")
    print("-" * 80)
    
    results_summary = _compute_inference_results_general(
        inference_results,
        int_params_metadata,
        inference_tasks,
        param_keys_to_infer,
    )
    
    _print_results_table_general(results_summary, param_keys_to_infer)
    _print_summary_statistics_general(
        results_summary,
        int_params_list,
        ext_params_list,
        param_keys_to_infer,
    )
    
    # Final checkpoint verification
    print("\n" + "=" * 80)
    checkpoint_status = workflow.get_checkpoint_status()
    print(f"Checkpoint Status:")
    print(f"  Stage: {checkpoint_status.stage}")
    print(f"  Simulation entries: {len(checkpoint_status.simulation_entries)}")
    print(f"    Expected: {len(int_params_list)}")
    print(f"  Inference entries: {len(checkpoint_status.inference_entries)}")
    print(f"    Expected: {len(inference_tasks)}")
    
    assert len(checkpoint_status.simulation_entries) == len(int_params_list)
    assert len(checkpoint_status.inference_entries) == len(inference_tasks)
    
    all_success = all(res['status'] == '✓ PASS' for res in results_summary)
    
    print("\n" + "=" * 80)
    if all_success:
        print("✓ TWO-PASS GENERAL WORKFLOW TEST PASSED!")
    else:
        print("✗ TWO-PASS GENERAL WORKFLOW TEST FAILED!")
    print("=" * 80)
    
    assert all_success, "All inference results should match true parameters (within 10% rel error)"
    
    return {
        'results_summary': results_summary,
        'int_param_ranges': int_param_ranges,
        'ext_param_ranges': ext_param_ranges,
        'inference_mode': inference_mode,
        'elastic_params_list': elastic_params_list,
        'viscous_params_list': viscous_params_list,
        'num_int_params': len(int_params_list),
        'num_ext_params': len(ext_params_list),
        'total_models': len(int_params_list) * len(ext_params_list),
        'passed': sum(1 for res in results_summary if res['status'] == '✓ PASS'),
        'failed': sum(1 for res in results_summary if res['status'] == '✗ FAIL'),
    }


# ============================================================================
# Helper Functions
# ============================================================================

def _setup_internal_params(sp4_values, tau_b_values):
    """Create Cartesian product of internal parameters."""
    int_params_list = []
    int_params_metadata = []
    
    for sp4 in sp4_values:
        for tau_b in tau_b_values:
            int_params = make_ground_truth_int_params(Sp4=sp4, tau_b=tau_b)
            int_params_list.append(int_params)
            int_params_metadata.append({'Sp4': sp4, 'tau_b': tau_b})
    
    return int_params_list, int_params_metadata

def _setup_external_params(w0_values, a_values):
    """Create external parameters for all w0 and A combinations."""
    ext_params_list = []
    sim_params_list = []
    
    for w0 in w0_values:
        sim_params = make_sim_params_for_w0(w0=w0)
        for a_val in a_values:
            ext_params = make_ground_truth_ext_params(A=a_val, w0=w0)
            ext_params_list.append(ext_params)
            sim_params_list.append(sim_params)
    
    return ext_params_list, sim_params_list

def _print_int_params_summary(int_params_metadata):
    """Print summary of internal parameters."""
    print(f"\n  Generated {len(int_params_metadata)} int_params:")
    for idx, metadata in enumerate(int_params_metadata):
        print(f"    [{idx}] Sp4={metadata['Sp4']:.4e}, tau_b={metadata['tau_b']:.4e}")

def _verify_model_lists(model_lists, int_params_metadata, w0_values, num_a_values):
    """Verify structure and ordering of all model lists."""
    print(f"\nDetailed ModelList breakdown:")
    
    for int_idx, (key, model_list) in enumerate(model_lists.items()):
        metadata = int_params_metadata[int_idx]
        num_models = len(model_list.models)
        expected_models = len(w0_values) * num_a_values
        
        print(f"\n  int_idx={int_idx} (Sp4={metadata['Sp4']:.4e}, tau_b={metadata['tau_b']:.4e}):")
        print(f"    Models in this ModelList: {num_models} (expected: {expected_models})")
        
        assert num_models == expected_models, \
            f"int_idx {int_idx}: expected {expected_models} models, got {num_models}"
        
        for model_idx, model in enumerate(model_list.models):
            w0_idx = model_idx // num_a_values
            expected_w0 = w0_values[w0_idx]
            actual_w0 = model.ext_params.get('w0')
            
            assert actual_w0 == expected_w0, \
                f"int_idx {int_idx}, model_idx {model_idx}: expected w0={expected_w0}, got {actual_w0}"

def _print_inference_tasks(inference_tasks, int_params_metadata, param_keys_to_infer):
    """Print summary of inference tasks."""
    for idx, task in enumerate(inference_tasks):
        metadata = int_params_metadata[idx]
        print(f"  [{idx}] {task.task_key}")
        
        # Build dynamic parameter string
        param_str = ", ".join(
            f"{param_key}={metadata.get(param_key, 'N/A'):.4e}"
            for param_key in param_keys_to_infer
        )
        print(f"       True: {param_str}")

def _compute_inference_results(inference_results, int_params_metadata, inference_tasks):
    """Compute relative errors and status for all inference results."""
    results_summary = []
    
    for int_idx, task in enumerate(inference_tasks):
        metadata = int_params_metadata[int_idx]
        true_sp4 = metadata['Sp4']
        true_tau_b = metadata['tau_b']
        
        pass_results = inference_results[task.task_key]  # List of results, one per pass
        
        # Merge parameters from all passes
        inferred_params = {}
        for pass_result in pass_results:
            inferred_params.update(pass_result.params)
        
        inferred_sp4 = inferred_params.get('Sp4')
        inferred_tau_b = inferred_params.get('tau_b')
        
        if inferred_sp4 is None or inferred_tau_b is None:
            raise ValueError(
                f"Task {task.task_key}: Missing inferred parameters. "
                f"Got Sp4={inferred_sp4}, tau_b={inferred_tau_b}. "
                f"Available keys: {list(inferred_params.keys())}"
            )
        
        sp4_rel_error = abs(inferred_sp4 - true_sp4) / true_sp4 if true_sp4 != 0 else 0
        tau_b_rel_error = abs(inferred_tau_b - true_tau_b) / true_tau_b if true_tau_b != 0 else 0
        
        success = (sp4_rel_error < 0.1) and (tau_b_rel_error < 0.1)
        
        results_summary.append({
            'int_idx': int_idx,
            'true_sp4': true_sp4,
            'true_tau_b': true_tau_b,
            'inferred_sp4': inferred_sp4,
            'inferred_tau_b': inferred_tau_b,
            'sp4_rel_error': sp4_rel_error,
            'tau_b_rel_error': tau_b_rel_error,
            'converged': pass_results[-1].success,  # Use final pass's success
            'final_loss': pass_results[-1].loss,    # Use final pass's loss
            'status': '✓ PASS' if success else '✗ FAIL',
        })
    
    return results_summary

def _print_results_table(results_summary):
    """Print detailed results table."""
    print(f"\nDetailed Results:")
    print(f"\n{'idx':<4} {'True Sp4':<12} {'True tau_b':<12} {'Inferred Sp4':<14} {'Inferred tau_b':<14} {'Sp4 Error':<14} {'tau_b Error':<14} {'Status':<8}")
    print("-" * 110)
    
    for res in results_summary:
        print(
            f"{res['int_idx']:<4} "
            f"{res['true_sp4']:<12.4e} "
            f"{res['true_tau_b']:<12.4e} "
            f"{res['inferred_sp4']:<14.4e} "
            f"{res['inferred_tau_b']:<14.4e} "
            f"{res['sp4_rel_error']:<14.4%} "
            f"{res['tau_b_rel_error']:<14.4%} "
            f"{res['status']:<8}"
        )

def _print_summary_statistics(results_summary, int_params_list, ext_params_list, num_a_values):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("-" * 80)
    
    passed = sum(1 for res in results_summary if res['status'] == '✓ PASS')
    failed = len(results_summary) - passed
    
    print(f"\nResults: {passed}/{len(results_summary)} cases passed")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    
    avg_sp4_error = sum(res['sp4_rel_error'] for res in results_summary) / len(results_summary)
    avg_tau_b_error = sum(res['tau_b_rel_error'] for res in results_summary) / len(results_summary)
    
    print(f"\nAverage Relative Errors:")
    print(f"  Sp4:   {avg_sp4_error:.4%}")
    print(f"  tau_b: {avg_tau_b_error:.4%}")
    
    print(f"\nParameter Space Coverage:")
    print(f"  Int_params combinations: {len(int_params_list)}")
    print(f"  External param sets: {len(ext_params_list)}")
    print(f"  Total models across all int_params: {len(int_params_list) * len(ext_params_list)}")
    
    print(f"\nTwo-Pass Strategy per Int-Params:")
    print(f"  ✓ Pass 1 (Elastic): Optimizes Sp4 on {len([e for e in ext_params_list if e.get('w0') == 0.0])} models at w0=0")
    print(f"  ✓ Pass 2 (Viscous): Optimizes tau_b on {len([e for e in ext_params_list if e.get('w0') > 0])} models at w0>0")

def _compute_inference_results_general(
    inference_results,
    int_params_metadata,
    inference_tasks,
    param_keys_to_infer,
    rel_error_threshold=0.1,
):
    """
    Compute relative errors for all inferred parameters.
    
    Args:
        inference_results: Dict mapping task_key -> [pass1_result, pass2_result, ...]
        int_params_metadata: List of dicts with ground truth internal parameters
        inference_tasks: List of InferenceTask objects
        param_keys_to_infer: List of parameter names to check
        rel_error_threshold: Threshold for pass/fail (default 10%)
    
    Returns:
        List of result dicts with errors and pass/fail status
    """
    results_summary = []
    
    for int_idx, task in enumerate(inference_tasks):
        metadata = int_params_metadata[int_idx]
        
        pass_results = inference_results[task.task_key]
        
        # Merge parameters from all passes
        inferred_params = {}
        for pass_result in pass_results:
            inferred_params.update(pass_result.params)
        
        # Compute errors for each parameter
        result_entry = {
            'int_idx': int_idx,
            'task_key': task.task_key,
            'converged': pass_results[-1].success,
            'final_loss': pass_results[-1].loss,
        }
        
        all_within_threshold = True
        
        for param_key in param_keys_to_infer:
            true_value = metadata.get(param_key)
            inferred_value = inferred_params.get(param_key)
            
            if true_value is None:
                result_entry[f'{param_key}_true'] = None
                result_entry[f'{param_key}_inferred'] = None
                result_entry[f'{param_key}_rel_error'] = None
                continue
            
            if inferred_value is None:
                raise ValueError(
                    f"Task {task.task_key}: Missing inferred parameter '{param_key}'. "
                    f"Available: {list(inferred_params.keys())}"
                )
            
            rel_error = (
                abs(inferred_value - true_value) / abs(true_value)
                if true_value != 0 else 0
            )
            
            result_entry[f'{param_key}_true'] = true_value
            result_entry[f'{param_key}_inferred'] = inferred_value
            result_entry[f'{param_key}_rel_error'] = rel_error
            
            if rel_error >= rel_error_threshold:
                all_within_threshold = False
        
        result_entry['status'] = '✓ PASS' if all_within_threshold else '✗ FAIL'
        results_summary.append(result_entry)
    
    return results_summary

def _print_results_table_general(results_summary, param_keys_to_infer):
    """Print detailed results table for all parameters."""
    print("\nDetailed Inference Results:")
    print("-" * 120)
    
    # Build header
    header = ['Int Idx', 'Task Key', 'Status']
    for param_key in param_keys_to_infer:
        header.extend([f'{param_key} (True)', f'{param_key} (Inferred)', f'{param_key} Error %'])
    header.extend(['Converged', 'Loss'])
    
    # Print header
    print(" | ".join(f"{h:>15}" for h in header))
    print("-" * 120)
    
    # Print rows
    for result_entry in results_summary:
        row = [
            str(result_entry['int_idx']),
            result_entry['task_key'],
            result_entry['status']
        ]
        
        # Add parameter columns
        for param_key in param_keys_to_infer:
            true_val = result_entry.get(f'{param_key}_true', float('nan'))
            inferred_val = result_entry.get(f'{param_key}_inferred', float('nan'))
            error_pct = result_entry.get(f'{param_key}_rel_error', float('nan'))
            
            if true_val is None or inferred_val is None or error_pct is None:
                row.extend(['N/A', 'N/A', 'N/A'])
            else:
                row.extend([
                    f"{true_val:.6e}",
                    f"{inferred_val:.6e}",
                    f"{error_pct * 100:.2f}%"
                ])
        
        # Add convergence and loss
        row.extend([
            str(result_entry.get('converged', False)),
            f"{result_entry.get('final_loss', float('nan')):.6e}"
        ])
        
        print(" | ".join(f"{v:>15}" for v in row))
    
    print("-" * 120)

def _print_summary_statistics_general(results_summary, int_params_list, ext_params_list, param_keys_to_infer):
    """Print summary statistics for generalized parameter inference."""
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("-" * 80)
    
    passed = sum(1 for res in results_summary if res['status'] == '✓ PASS')
    failed = len(results_summary) - passed
    
    print(f"\nResults: {passed}/{len(results_summary)} cases passed")
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")
    
    # Compute average errors for each parameter
    print(f"\nAverage Relative Errors:")
    for param_key in param_keys_to_infer:
        error_key = f'{param_key}_rel_error'
        errors = [res.get(error_key) for res in results_summary if res.get(error_key) is not None]
        
        if errors:
            avg_error = sum(errors) / len(errors)
            print(f"  {param_key:>10}: {avg_error:.4%}")
        else:
            print(f"  {param_key:>10}: N/A")
    
    print(f"\nParameter Space Coverage:")
    print(f"  Int_params combinations: {len(int_params_list)}")
    print(f"  External param sets: {len(ext_params_list)}")
    print(f"  Total models across all int_params: {len(int_params_list) * len(ext_params_list)}")
    
    # Two-pass strategy summary
    print(f"\nTwo-Pass Strategy per Int-Params:")
    w0_zero_count = sum(1 for e in ext_params_list if e.get('w0') == 0.0)
    w0_pos_count = sum(1 for e in ext_params_list if e.get('w0', 0) > 0)
    
    print(f"  ✓ Pass 1 (Elastic): Optimizes on {w0_zero_count} models at w0=0")
    print(f"  ✓ Pass 2 (Viscous): Optimizes on {w0_pos_count} models at w0>0")
    
    print("=" * 80)

if __name__ == "__main__":
    # test_workflow_with_inference()
    # test_workflow_with_inference_multi_sp4()
    # test_workflow_with_inference_multi_ext_params()
    # test_workflow_with_inference_single_task_multi_ext()
    # test_workflow_with_inference_multi_sp4_tau_b()
    # test_workflow_with_inference_two_pass_elastic_viscous()
    # test_workflow_with_inference_two_pass_elastic_viscous_multi_int_params()
    # test_workflow_elastic_viscous_condensed()
    # test_workflow_elastic_viscous_general()
    
    int_param_ranges = {'Sp4': [1.0], 'Beta':[0, 1.0], 'tau_b': [0, 1.0], 'tau_s':[0, 1.0]}
    ext_param_ranges = {'A': [1e-6, 1e-5], 'w0': [0.0, 1e-3, 1.0]}
    elastic_params_list = ['Sp4', 'Beta']
    viscous_params_list = ['tau_b', 'tau_s']   
    
    test_workflow_elastic_viscous_general(
        int_param_ranges=int_param_ranges,
        ext_param_ranges=ext_param_ranges,
        elastic_params_list = elastic_params_list,
        viscous_params_list = viscous_params_list,
        )