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

# def _determine_inference_passes(
#     param_keys_to_infer, 
#     elastic_params_list, 
#     viscous_params_list,
#     ext_params_list,
#     min_w0,
# ):
#     """Determine number of passes and split parameters based on min_w0.
#     Also validates that filtered data exists for each pass.
    
#     Returns:
#         (n_passes, list of pass_configs)
#         where each pass_config = {
#             'name': str,
#             'param_keys': list,
#             'w0_filter': callable or None,
#         }
#     """
    
#     elastic_keys = [k for k in param_keys_to_infer if k in elastic_params_list]
#     viscous_keys = [k for k in param_keys_to_infer if k in viscous_params_list]
#     unknown_keys = [k for k in param_keys_to_infer if k not in elastic_params_list and k not in viscous_params_list]
    
#     if unknown_keys:
#         raise ValueError(f"Unknown parameters for inference: {unknown_keys}.")

#     # Helper: check if a filter yields any data
#     def filter_has_data(w0_filter):
#         return any(w0_filter(ext_params.get('w0', 0)) for ext_params in ext_params_list)

#     if elastic_keys and viscous_keys:
#         elastic_filter = lambda w0: w0 == min_w0
#         viscous_filter = lambda w0: w0 > 0
        
#         elastic_has_data = filter_has_data(elastic_filter)
#         viscous_has_data = filter_has_data(viscous_filter)
        
#         if elastic_has_data and viscous_has_data:
#             # Both passes viable
#             return 2, [
#                 {
#                     'name': 'Elastic Inference',
#                     'param_keys': elastic_keys,
#                     'w0_filter': elastic_filter,
#                 },
#                 {
#                     'name': 'Viscous Inference',
#                     'param_keys': viscous_keys,
#                     'w0_filter': viscous_filter,
#                 },
#             ]
#         elif elastic_has_data:
#             # Only elastic data; infer viscous params using all data
#             return 1, [
#                 {
#                     'name': 'Single Pass (Elastic + Viscous)',
#                     'param_keys': elastic_keys + viscous_keys,
#                     'w0_filter': None,  # Use all data
#                 },
#             ]
#         elif viscous_has_data:
#             # Only viscous data; infer elastic params using all data
#             return 1, [
#                 {
#                     'name': 'Single Pass (Elastic + Viscous)',
#                     'param_keys': elastic_keys + viscous_keys,
#                     'w0_filter': None,  # Use all data
#                 },
#             ]
#         else:
#             raise ValueError("No data available for either elastic or viscous inference.")
    
#     elif elastic_keys:
#         elastic_filter = lambda w0: w0 == min_w0
#         if filter_has_data(elastic_filter):
#             return 1, [
#                 {
#                     'name': 'Elastic Inference',
#                     'param_keys': elastic_keys,
#                     'w0_filter': elastic_filter,
#                 },
#             ]
#         else:
#             # No data at min_w0; use all data
#             return 1, [
#                 {
#                     'name': 'Elastic Inference (All Data)',
#                     'param_keys': elastic_keys,
#                     'w0_filter': None,
#                 },
#             ]
#     else:
#         viscous_filter = lambda w0: w0 > 0
#         if filter_has_data(viscous_filter):
#             return 1, [
#                 {
#                     'name': 'Viscous Inference',
#                     'param_keys': viscous_keys,
#                     'w0_filter': viscous_filter,
#                 },
#             ]
#         else:
#             # No data with w0 > 0; use all data
#             return 1, [
#                 {
#                     'name': 'Viscous Inference (All Data)',
#                     'param_keys': viscous_keys,
#                     'w0_filter': None,
#                 },
#             ]

# def _filter_ext_params_by_w0(ext_params_list, w0_filter):
#     """Filter external parameters by w0 value.
    
#     Args:
#         ext_params_list: List of external parameter dicts
#         w0_filter: Function that takes w0 value and returns bool, or None for all data
    
#     Returns:
#         Filtered list of external parameter dicts
#     """
#     if w0_filter is None:
#         return ext_params_list
    
#     return [
#         ext_params for ext_params in ext_params_list
#         if w0_filter(ext_params.get('w0', 0))
#     ]

# def make_two_pass_pipeline(
#     model_class,
#     ground_truths,
#     ext_params_list,
#     sim_params_list,
#     param_keys_to_infer,
#     elastic_params_list,
#     viscous_params_list,
#     loss_fn,
#     optimizer,
#     min_w0,
#     product_or_zip,
#     optimizer_kwargs,
#     n_jobs_per_pass,
#     ground_truth_models,
# ) -> InferencePipeline:
#     """
#     Create inference pipeline that:
#     1. Determines passes dynamically based on data availability
#     2. For each pass, optimizes only relevant subset of parameters
#     3. Maintains full int_params structure across passes via PipelinePass
#     """
#     import copy
    
#     # Determine passes and get pass configurations
#     n_passes, pass_configs = _determine_inference_passes(
#         param_keys_to_infer=param_keys_to_infer,
#         elastic_params_list=elastic_params_list,
#         viscous_params_list=viscous_params_list,
#         ext_params_list=ext_params_list,
#         min_w0=min_w0,
#     )
    
#     pipeline_passes = []
    
#     for pass_config in pass_configs:
#         pass_name = pass_config['name']
#         param_keys = pass_config['param_keys']
#         w0_filter = pass_config['w0_filter']
        
#         # Filter data for this pass
#         filtered_ext_params = _filter_ext_params_by_w0(ext_params_list, w0_filter)
#         filtered_indices = [
#             i for i, ext_params in enumerate(ext_params_list)
#             if ext_params in filtered_ext_params
#         ]
        
#         filtered_ground_truths = [ground_truths[i] for i in filtered_indices]
#         filtered_ext_params_list = [ext_params_list[i] for i in filtered_indices]
#         filtered_sim_params_list = [sim_params_list[i] for i in filtered_indices]
        
#         # Create PipelinePass for this pass
#         pipeline_pass = PipelinePass(
#             name=pass_name,
#             model_class=model_class,
#             ground_truths=filtered_ground_truths,
#             ext_params_list=filtered_ext_params_list,
#             sim_params_list=filtered_sim_params_list,
#             param_keys_to_infer=param_keys,
#             product_or_zip=product_or_zip,
#             optimizer=optimizer,
#             optimizer_kwargs={**optimizer_kwargs, 'bounds': _make_optimizer_bounds(param_keys)},
#             compose_int_params=None,  # Will be set by InferencePipeline._build_pass_model
#             compose_ext_params=None,
#             compose_sim_params=None,
#         )
#         pipeline_passes.append(pipeline_pass)
    
#     return InferencePipeline(
#         passes=pipeline_passes,
#         loss_fn=loss_fn,
#         n_jobs_per_pass=n_jobs_per_pass,
#     )

# def make_inference_pipeline_single(
#     model_list: ModelList,
#     initial_guesses: List[Dict[str, float]],
#     loss_fn: Callable,
#     optimizer: Callable,
#     optimizer_kwargs: Optional[Dict[str, Any]] = None,
#     n_jobs_per_pass: int = 1,
# ) -> InferencePipeline:
#     """
#     Factory function to create inference pipeline from ModelList.
    
#     Args:
#         model_list: ModelList with simulated models
#         initial_guesses: List of initial parameter guesses
#         loss_fn: Loss function
#         optimizer: Optimizer callable
#         optimizer_kwargs: Optimizer kwargs
#         n_jobs_per_pass: Parallel jobs per pass
    
#     Returns:
#         InferencePipeline
#     """
#     import copy
    
#     if optimizer_kwargs is None:
#         optimizer_kwargs = make_optimizer_kwargs()
    
#     ground_truths = []
#     ext_params_list = []
#     sim_params_list = []

#     for model in model_list.models:
#         ground_truths.append(model.sim_output['value'])
#         ext_params_list.append(copy.deepcopy(model.ext_params))
#         sim_params_list.append(copy.deepcopy(model.sim_params))
    
#     print("make_inference_pipeline_single")
#     print(f"ext_params_list = {ext_params_list}")

#     param_keys_to_infer = list(initial_guesses[0].keys())
    
#     pipeline_pass = PipelinePass(
#         name="Parameter Inference",
#         model_class=type(model_list.models[0]),
#         ground_truths=ground_truths,
#         ext_params_list=ext_params_list,
#         sim_params_list=sim_params_list,
#         param_keys_to_infer=param_keys_to_infer,
#         product_or_zip="zip",
#         optimizer=optimizer,
#         optimizer_kwargs=optimizer_kwargs,
#     )
    
#     return InferencePipeline(
#         passes=[pipeline_pass],
#         loss_fn=loss_fn,
#         n_jobs_per_pass=n_jobs_per_pass,
#     )


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
    A_value = 1e-4
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

    # Loss function setup
    def loss_fn(predicted_list, ground_truth_list):
        """MSE loss aggregated across all models."""
        total_loss = 0.0
        for predicted, gt in zip(predicted_list, ground_truth_list):
            total_loss += np.mean((predicted - gt) ** 2)
        return total_loss / len(ground_truth_list)
        
    def make_simple_inference_pipeline(model_list, **kwargs):
        """Factory function to create inference pipeline from ModelList."""
        
        # Extract ground truths from simulated models
        ground_truths = [model.sim_output['value'] for model in model_list.models]
        ext_params_batch = [model.ext_params for model in model_list.models]
        sim_params_batch = [model.sim_params for model in model_list.models]
        
        # Single pass: infer int_params
        pass_1 = PipelinePass(
            name="InferSp4",
            model_class=ReducedModel,
            ground_truths=ground_truths,
            ext_params_list=ext_params_batch,
            sim_params_list=sim_params_batch,
            param_keys_to_infer=['Sp4'],
            fixed_params={},
            product_or_zip="zip",
            optimizer=basinhopping_optimizer,
            optimizer_kwargs=optimizer_kwargs,
        )
        
        return InferencePipeline(
            passes=[pass_1],
            loss_fn=loss_fn,
            n_jobs_per_pass=-1,
        )
    
    # Create inference tasks: one for each int_idx
    
    initial_guesses = [{'Sp4': 1e-1}]
    
    inference_tasks = []
    for int_idx in range(len(int_params_list)):
        task = InferenceTask(
            task_key=f"infer_int_{int_idx}",
            int_idx=int_idx,
            pair_indices=None,  # Use all models in the ModelList
            make_pipeline_fn=make_simple_inference_pipeline,
            pipeline_kwargs={},
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


if __name__ == "__main__":
    test_workflow_with_inference()
