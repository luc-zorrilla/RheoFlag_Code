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
    def __init__(self, bounds, stepsize):
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
                    'gtol':1e-6, 
                    'eps': 1e-8, 
                    'finite_diff_rel_step':None,
                    }
                ... (other local optimizer options) # 'maxiter': 1000,
            }
    
    Args (Global Minimizer):
        global_minimizer_kwargs: Dict with basin-hopping configuration:
            {
                'niter': 49,  # Basin-hopping iterations
                'stepsize': 5,  # Maximum step size for perturbations
                'T': 1,  # Temperature for Metropolis acceptance
                'tol': 1e-10,  # Early stopping tolerance --> Not in basinhopping function
                'stepwise_factor':1-1e-16, # Stepsize multiplication/division factor
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
            'gtol':1e-6,  # Gradient tolerance for local minimizer
            'eps': 1e-8,  # ?
            'finite_diff_rel_step':None, # ?
        }
    }

    global_minimizer_kwargs = global_minimizer_kwargs or {
        'niter': 49,  # Basin-hopping iterations
        'stepsize': 10,  # Maximum step size for perturbations
        'T': 1,  # Temperature for Metropolis acceptance
        'tol': 1e-10,  # Early stopping tolerance
        'stepwise_factor':1 - 1e-16, # Stepsize multiplication factor per update
    }
    
    # --- Extract global minimizer parameters ---
    niter = global_minimizer_kwargs.pop('niter', 49)
    T = global_minimizer_kwargs.pop('T', 1)
    stepsize = global_minimizer_kwargs.pop('stepsize', 10)
    stepwise_factor = global_minimizer_kwargs.pop('stepwise_factor', 1-1e-16)
    tol = global_minimizer_kwargs.pop('tol', 1e-10)
    
    # --- Extract local minimizer parameters ---
    method = local_minimizer_kwargs.pop('method', "L-BFGS-B")
    jac = local_minimizer_kwargs.pop('jac', '3-point')
    options = local_minimizer_kwargs.pop(
        'options', {
            'ftol':1e-8, 
            'gtol':1e-6, 
            'eps': 1e-8, 
            'finite_diff_rel_step':None,
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
        'stepwise_factor':stepwise_factor,
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
        T_start = 10.0 / w0
        T_end = 20.0 / w0
        dT = (1 / 10.0) / w0
        N_T = int((T_end-T_start)/dT) + 1
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
            'gtol': 1e-6,
            'eps': 1e-8,
            'finite_diff_rel_step': None,
        },
    },
    global_minimizer_kwargs = {
        'niter': 49,
        'T': 1,
        'stepsize': 10,
        'stepwise_factor':1-1e-16,
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
            optimizer_kwargs={**optimizer_kwargs, 'bounds': _make_optimizer_bounds(param_keys)}, # TODO: change eps here?
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
                    pair_indices=[ext_idx],
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
                pair_indices=list(range(len(ext_params_list))), # None
                make_pipeline_fn=pipeline_fn,
                pipeline_kwargs={},
                initial_guesses=initial_guesses,
            )
            tasks.append(task)
    
    return tasks

# =====================
# === Main Function ===
# ===================== 

def workflow_elastic_viscous_general(
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
    checkpoint_str = "./test_checkpoints_vef_elastic_viscous_general",
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
    
    checkpoint_dir = Path(checkpoint_str + f"_{inference_mode}")
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
    _print_int_params_summary(int_params_metadata, param_keys_to_infer)

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
        sim_params_list = [make_sim_params_for_w0(w0_value) for ep in ext_params_list]

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
        param: 1e-1 if param.startswith('Sp') else 0
        for param in param_keys_to_infer
    }
    initial_guesses_list = [initial_guesses]
    
    # Build optimizer_kwargs if not provided
    if optimizer_kwargs is None:
        optimizer_kwargs = make_optimizer_kwargs(
            bounds=_make_optimizer_bounds(param_keys_to_infer),
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
    # _print_inference_tasks(inference_tasks, int_params_metadata, param_keys_to_infer)
    
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

def _print_int_params_summary(int_params_metadata, param_keys_to_infer):
    """Print summary of internal parameters."""
    print(f"\n  Generated {len(int_params_metadata)} int_params:")
    for idx, metadata in enumerate(int_params_metadata):
        # Build dynamic parameter string
        param_str = ", ".join(
            f"{param_key}={metadata.get(param_key, 'N/A'):.4e}"
            for param_key in param_keys_to_infer
        )
        print(f"    [{idx}] {param_str}")

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
    Compute relative errors for all inferred parameters and extract sigma from Hessian.
    
    Args:
        inference_results: Dict mapping task_key -> [pass1_result, pass2_result, ...]
        int_params_metadata: List of dicts with ground truth internal parameters
        inference_tasks: List of InferenceTask objects
        param_keys_to_infer: List of parameter names to check
        rel_error_threshold: Threshold for pass/fail (default 10%)
    
    Returns:
        List of result dicts with errors, pass/fail status, and sigma values
    """
    results_summary = []
    
    for result, task in zip(inference_results, inference_tasks):
        int_idx = task.int_idx  # ← Use task's stored index, not loop counter

        metadata = int_params_metadata[int_idx]
        
        pass_results = inference_results[task.task_key]
        
        # Merge parameters from all passes
        inferred_params = {}
        std_errors_dict = {}
        
        for pass_result in pass_results:
            inferred_params.update(pass_result.params)
            
            # Extract standard errors (sigma) from the last pass result
            if pass_result.std_errors is not None:
                # std_errors is an array; map to parameter names
                param_names = list(pass_result.params.keys())
                for i, param_name in enumerate(param_names):
                    if i < len(pass_result.std_errors):
                        std_errors_dict[param_name] = pass_result.std_errors[i]
        
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
            sigma_value = std_errors_dict.get(param_key)
            
            if true_value is None:
                result_entry[f'{param_key}_true'] = None
                result_entry[f'{param_key}_inferred'] = None
                result_entry[f'{param_key}_rel_error'] = None
                result_entry[f'{param_key}_sigma'] = None
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
            result_entry[f'{param_key}_sigma'] = sigma_value 
            
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

    # Bending Elasticity - Sp4

    int_param_ranges = {'Sp4': [1.0]}
    A_vec = np.pow(10, np.linspace(start = -6, stop = -1, num = 6))
    ext_param_ranges = {'A': A_vec}
    elastic_params_list = ['Sp4']
    viscous_params_list = []

    inference_mode = "single_inference"
    checkpoint_str = "./Results/BendingElasticity/BendingElasticity"

    workflow_elastic_viscous_general(
        int_param_ranges=int_param_ranges,
        ext_param_ranges=ext_param_ranges,
        elastic_params_list = elastic_params_list,
        viscous_params_list = viscous_params_list,
        inference_mode = inference_mode,
        checkpoint_str=checkpoint_str,
        )

    inference_mode = "cumulative_inference"   
    for k in range(6):
        A_vec_k = np.pow(10, np.linspace(start = -6, stop = -6+k, num = k+1))
        ext_param_ranges = {'A': A_vec_k}
        checkpoint_str = f"./Results/BendingElasticity/BendingElasticity_{k}"

        workflow_elastic_viscous_general(
            int_param_ranges=int_param_ranges,
            ext_param_ranges=ext_param_ranges,
            elastic_params_list = elastic_params_list,
            viscous_params_list = viscous_params_list,
            inference_mode = inference_mode,
            checkpoint_str=checkpoint_str,
            )

    # Shear Elasticity - Beta

    int_param_ranges = {'Beta': [1.0]}
    A_vec = np.pow(10, np.linspace(start = -6, stop = -1, num = 6))
    ext_param_ranges = {'A': A_vec}
    elastic_params_list = ['Beta']
    viscous_params_list = []

    inference_mode = "single_inference"
    checkpoint_str = "./Results/ShearElasticity/ShearElasticity"

    workflow_elastic_viscous_general(
        int_param_ranges=int_param_ranges,
        ext_param_ranges=ext_param_ranges,
        elastic_params_list = elastic_params_list,
        viscous_params_list = viscous_params_list,
        inference_mode = inference_mode,
        checkpoint_str=checkpoint_str,
        )

    inference_mode = "cumulative_inference"   
    for k in range(6):
        A_vec_k = np.pow(10, np.linspace(start = -6, stop = -6+k, num = k+1))
        ext_param_ranges = {'A': A_vec_k}
        checkpoint_str = f"./Results/ShearElasticity/ShearElasticity_{k}"

        workflow_elastic_viscous_general(
            int_param_ranges=int_param_ranges,
            ext_param_ranges=ext_param_ranges,
            elastic_params_list = elastic_params_list,
            viscous_params_list = viscous_params_list,
            inference_mode = inference_mode,
            checkpoint_str=checkpoint_str,
            )

    # Bending & Shear Elasticities - Sp4, Beta

    int_param_ranges = {'Sp4': [1.0], 'Beta': [1.0]}
    A_vec = np.pow(10, np.linspace(start = -6, stop = -1, num = 6))
    ext_param_ranges = {'A': A_vec}
    elastic_params_list = ['Sp4', 'Beta']
    viscous_params_list = []

    inference_mode = "single_inference"
    checkpoint_str = "./Results/BendingShearElasticity/BendingShearElasticity"

    workflow_elastic_viscous_general(
        int_param_ranges=int_param_ranges,
        ext_param_ranges=ext_param_ranges,
        elastic_params_list = elastic_params_list,
        viscous_params_list = viscous_params_list,
        inference_mode = inference_mode,
        checkpoint_str=checkpoint_str,
        )

    inference_mode = "cumulative_inference"   
    for k in range(6):
        A_vec_k = np.pow(10, np.linspace(start = -6, stop = -6+k, num = k+1))
        ext_param_ranges = {'A': A_vec_k}
        checkpoint_str = f"./Results/BendingShearElasticity/BendingShearElasticity_{k}"

        workflow_elastic_viscous_general(
            int_param_ranges=int_param_ranges,
            ext_param_ranges=ext_param_ranges,
            elastic_params_list = elastic_params_list,
            viscous_params_list = viscous_params_list,
            inference_mode = inference_mode,
            checkpoint_str=checkpoint_str,
            )

    # Bending Viscosity (Fixed Bending Elasticity)

    int_param_ranges = {'tau_b': [1.0]}
    A_vec = [1e-6]
    w0_vec = np.pow(10, -np.linspace(start = -3, stop = 3, num = 7))
    ext_param_ranges = {'A': A_vec, 'w0':w0_vec}
    elastic_params_list = []
    viscous_params_list = ['tau_b']

    inference_mode = "single_inference"
    checkpoint_str = "./Results/BendingViscosity/BendingViscosity"

    workflow_elastic_viscous_general(
        int_param_ranges=int_param_ranges,
        ext_param_ranges=ext_param_ranges,
        elastic_params_list = elastic_params_list,
        viscous_params_list = viscous_params_list,
        inference_mode = inference_mode,
        checkpoint_str=checkpoint_str,
        )

    inference_mode = "cumulative_inference"   
    for l in range(7):
        w0_vec_l = np.pow(10, -np.linspace(start = -3, stop = -3+l, num = l+1))
        ext_param_ranges = {'A': A_vec, 'w0':w0_vec_l}
        checkpoint_str = f"./Results/BendingViscosity/BendingViscosity_{l}"

        workflow_elastic_viscous_general(
            int_param_ranges=int_param_ranges,
            ext_param_ranges=ext_param_ranges,
            elastic_params_list = elastic_params_list,
            viscous_params_list = viscous_params_list,
            inference_mode = inference_mode,
            checkpoint_str=checkpoint_str,
            )

    # Shear Viscosity (Fixed Bending Elasticity & Shear Elasticity)

    int_param_ranges = {'tau_s': [1.0], 'Beta':[1.0]}
    A_vec = [1e-6]
    w0_vec = np.pow(10, -np.linspace(start = -3, stop = 3, num = 7))
    ext_param_ranges = {'A': A_vec, 'w0':w0_vec}
    elastic_params_list = []
    viscous_params_list = ['tau_s']

    inference_mode = "single_inference"
    checkpoint_str = "./Results/ShearViscosity/ShearViscosity"

    workflow_elastic_viscous_general(
        int_param_ranges=int_param_ranges,
        ext_param_ranges=ext_param_ranges,
        elastic_params_list = elastic_params_list,
        viscous_params_list = viscous_params_list,
        inference_mode = inference_mode,
        checkpoint_str=checkpoint_str,
        )

    inference_mode = "cumulative_inference"   
    for l in range(7):
        w0_vec_l = np.pow(10, -np.linspace(start = -3, stop = -3+l, num = l+1))
        ext_param_ranges = {'A': A_vec, 'w0':w0_vec_l}
        checkpoint_str = f"./Results/ShearViscosity/ShearViscosity_{l}"

        workflow_elastic_viscous_general(
            int_param_ranges=int_param_ranges,
            ext_param_ranges=ext_param_ranges,
            elastic_params_list = elastic_params_list,
            viscous_params_list = viscous_params_list,
            inference_mode = inference_mode,
            checkpoint_str=checkpoint_str,
            )
    

    # Bending & Shear Viscosities (Fixed Bending & Shear Elasticities)

    int_param_ranges = {'tau_b': [1.0], 'tau_s':[1.0], 'Beta':[1.0]}
    A_vec = [1e-6]
    w0_vec = np.pow(10, -np.linspace(start = -3, stop = 3, num = 7))
    ext_param_ranges = {'A': A_vec, 'w0':w0_vec}
    elastic_params_list = []
    viscous_params_list = ['tau_b', 'tau_s']

    inference_mode = "single_inference"
    checkpoint_str = "./Results/BendingShearViscosity/BendingShearViscosity"

    workflow_elastic_viscous_general(
        int_param_ranges=int_param_ranges,
        ext_param_ranges=ext_param_ranges,
        elastic_params_list = elastic_params_list,
        viscous_params_list = viscous_params_list,
        inference_mode = inference_mode,
        checkpoint_str=checkpoint_str,
        )

    inference_mode = "cumulative_inference"   
    for l in range(7):
        w0_vec_l = np.pow(10, -np.linspace(start = -3, stop = -3+l, num = l+1))
        ext_param_ranges = {'A': A_vec, 'w0':w0_vec_l}
        checkpoint_str = f"./Results/BendingShearViscosity/BendingShearViscosity_{l}"

        workflow_elastic_viscous_general(
            int_param_ranges=int_param_ranges,
            ext_param_ranges=ext_param_ranges,
            elastic_params_list = elastic_params_list,
            viscous_params_list = viscous_params_list,
            inference_mode = inference_mode,
            checkpoint_str=checkpoint_str,
            )