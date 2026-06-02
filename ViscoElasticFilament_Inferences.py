from Models import Model, ModelList
from ViscoElasticFilament_Models import (
    StraightLine, 
    ViscoElasticFilament, 
    ViscoElasticFilament_FlowParams, 
    ViscoElasticFilament_FlowParams_ScalarBending,
)
from Inferences import Inference, InferencePipeline, PipelinePass, InferenceResult
from ModelInferenceWorkflow import InferenceTask, SimulationInferenceWorkflow

import numpy as np
from itertools import product, zip_longest
from pathlib import Path
from scipy.optimize import Bounds

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

def model_params_only_flow( # TODO: this might not be necessary as a full function
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

def make_one_pass_pipeline(
    model_class: Type[Model],
    ground_truths: List[np.ndarray],
    ext_params_list: List[Dict[str, Any]],
    sim_params_list: List[Dict[str, Any]],
    param_keys_to_infer: List[str],
    loss_fn: Callable,
    optimizer: Callable,
    name: str = "pass_0",
    product_or_zip: str = "product",
    fixed_params: Optional[Dict[str, float]] = None,
    compose_int_params: Optional[Callable] = None,
    compose_ext_params: Optional[Callable] = None,
    compose_sim_params: Optional[Callable] = None,
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
    n_jobs_per_pass: int = -1,
) -> InferencePipeline:
    """
    Factory function to create a single-pass InferencePipeline.
    
    Args:
        model_class: Model subclass to use in the pass
        ground_truths: List of ground truth arrays
        ext_params_list: List of external params per ground truth
        sim_params_list: List of simulation params per ground truth
        param_keys_to_infer: Which internal parameters to infer
        loss_fn: Loss function for the pipeline
        optimizer: Optimizer class instance
        name: Human-readable pass identifier
        product_or_zip: "product" or "zip" for params combination strategy
        fixed_params: Dict of parameters from prior passes (defaults to empty)
        compose_int_params: Composition function for int_params
        compose_ext_params: Composition function for ext_params
        compose_sim_params: Composition function for sim_params
        optimizer_kwargs: Arguments for optimizer
        n_jobs_per_pass: Number of parallel jobs (-1 for all cores)
    
    Returns:
        InferencePipeline with a single PipelinePass configured
    """
    if fixed_params is None:
        fixed_params = {}
    if optimizer_kwargs is None:
        optimizer_kwargs = {}
    
    # Create the single pass
    pass_0 = PipelinePass(
        name=name,
        model_class=model_class,
        ground_truths=ground_truths,
        ext_params_list=ext_params_list,
        sim_params_list=sim_params_list,
        param_keys_to_infer=param_keys_to_infer,
        fixed_params=fixed_params,
        product_or_zip=product_or_zip,
        compose_int_params=compose_int_params,
        compose_ext_params=compose_ext_params,
        compose_sim_params=compose_sim_params,
        optimizer=optimizer,
        optimizer_kwargs=optimizer_kwargs,
    )
    
    # Return configured pipeline
    return InferencePipeline(
        passes=[pass_0],
        loss_fn=loss_fn,
        n_jobs_per_pass=n_jobs_per_pass,
    )


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
        raise ValueError(f"Unknown parameters for inference: {unknown_keys}.")
    
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

def make_two_pass_pipeline(
    model_class: Type,
    ground_truths: List[np.ndarray],
    ext_params_list: List[Dict[str, Any]],
    sim_params_list: List[Dict[str, Any]],
    param_keys_to_infer: List[str],
    elastic_params_list: List[str],
    viscous_params_list: List[str],
    loss_fn: Callable,
    optimizer: Callable,
    min_w0: float = 0.0,
    product_or_zip: str = "product",
    compose_int_params: Optional[Callable] = None,
    compose_ext_params: Optional[Callable] = None,
    compose_sim_params: Optional[Callable] = None,
    optimizer_kwargs: Optional[Dict[str, Any]] = None,
    n_jobs_per_pass: int = -1,
) -> InferencePipeline:
    """
    Factory function to create a two-pass InferencePipeline.
    
    Splits parameters into elastic (pass 1) and viscous (pass 2) inference passes.
    Pass 1 (elastic) filters ext_params_list to w0 == min_w0.
    Pass 2 (viscous) uses fixed params from pass 1 and filters to w0 > 0.
    
    Args:
        model_class: Model subclass to use in both passes
        ground_truths: List of ground truth arrays
        ext_params_list: List of external params per ground truth
        sim_params_list: List of simulation params per ground truth
        param_keys_to_infer: Which internal parameters to infer
        elastic_params_list: List of parameter names that are elastic
        viscous_params_list: List of parameter names that are viscous
        loss_fn: Loss function for the pipeline
        optimizer: Optimizer class instance
        min_w0: Minimum w0 value for elastic pass filtering
        product_or_zip: "product" or "zip" for params combination strategy
        compose_int_params: Composition function for int_params
        compose_ext_params: Composition function for ext_params
        compose_sim_params: Composition function for sim_params
        optimizer_kwargs: Arguments for optimizer
        n_jobs_per_pass: Number of parallel jobs (-1 for all cores)
    
    Returns:
        InferencePipeline with two PipelinePass instances (elastic then viscous)
    
    Raises:
        ValueError: If elastic_keys and viscous_keys are both empty
    """
    if optimizer_kwargs is None:
        optimizer_kwargs = {}
    
    # Determine passes and their configurations
    n_passes, pass_configs = _determine_inference_passes(
        param_keys_to_infer,
        elastic_params_list,
        viscous_params_list,
        min_w0,
    )

    passes = []
    fixed_params = {}  # Accumulates inferred params from prior passes
    
    for i, pass_config in enumerate(pass_configs):
        # Filter ext_params_list by w0
        filtered_ext_params = _filter_ext_params_by_w0(
            ext_params_list,
            pass_config['w0_filter'],
        )
        
        # Create ground_truths corresponding to filtered ext_params
        filtered_ground_truths = [
            gt for gt, ext_params in zip(ground_truths, ext_params_list)
            if pass_config['w0_filter'](ext_params.get('w0', 0))
        ]
        
        # Ensure we have matching lengths
        if len(filtered_ground_truths) != len(filtered_ext_params):
            raise ValueError(
                f"Mismatch in filtered ground_truths and ext_params for pass {i}: "
                f"{len(filtered_ground_truths)} vs {len(filtered_ext_params)}"
            )
        
        # Create pass
        pass_obj = PipelinePass(
            name=pass_config['name'],
            model_class=model_class,
            ground_truths=filtered_ground_truths,
            ext_params_list=filtered_ext_params,
            sim_params_list=sim_params_list,
            param_keys_to_infer=pass_config['param_keys'],
            fixed_params=fixed_params.copy(),  # Use accumulated params
            product_or_zip=product_or_zip,
            compose_int_params=compose_int_params,
            compose_ext_params=compose_ext_params,
            compose_sim_params=compose_sim_params,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
        )
        passes.append(pass_obj)
    
    # Return configured pipeline
    return InferencePipeline(
        passes=passes,
        loss_fn=loss_fn,
        n_jobs_per_pass=n_jobs_per_pass,
    )

if __name__ == "__main__":
    
    # ========================================================================
    # Generate ground truth data and parameter ranges
    # ========================================================================

    # Internal parameters: Sp4 and tau_b (will be used in product with ext/sim)
    Sp4_values = [1e0, 1e1]
    tau_b_values = [0, 1]

    int_params_list = [
        make_ground_truth_int_params(Sp4=Sp4, tau_b=tau_b)
        for Sp4 in Sp4_values
        for tau_b in tau_b_values
    ]
    
    # External and simulation parameters: A and w0 (zipped together)
    w0_values = [0.0, 1e-3]
    A_values = [1e-6, 1e-5]
    
    def make_sim_params_for_w0(w0):
        """Create simulation parameters based on w0 value."""
        if w0 == 0.0:
            return {
                "T_span": (1e6, 2e6),
                "T_eval": np.linspace(1e6, 2e6, int(1e0)),
                "method": "hybr",
            }
        else:
            T_start = (1.0 / 10.0) / w0
            T_end = 10.0 / w0
            N_T = 100
            return {
                "T_span": (T_start, T_end),
                "T_eval": np.linspace(T_start, T_end, N_T),
                "method": "BDF",
            }
    
    # Zip ext_params and sim_params together (they depend on w0)
    ext_and_sim_pairs = [
        (
            make_ground_truth_ext_params(A=A, w0=w0),
            make_sim_params_for_w0(w0),
        )
        for A in A_values
        for w0 in w0_values
    ]
    
    # Now create the same-length lists via product of int_params with ext/sim pairs
    n_int_params = len(int_params_list)
    n_ext_sim_pairs = len(ext_and_sim_pairs)
    
    # Repeat int_params for each ext/sim pair
    ground_truth_int_params_list = [
        int_params_list[i % n_int_params]
        for _ in range(n_ext_sim_pairs)
        for i in range(n_int_params)
    ]
    
    # Tile ext_params and sim_params to match
    ground_truth_ext_params_list = [
        pair[0]
        for pair in ext_and_sim_pairs
        for _ in range(n_int_params)
    ]
    
    ground_truth_sim_params_list = [
        pair[1]
        for pair in ext_and_sim_pairs
        for _ in range(n_int_params)
    ]
    
    # Verify all lists have the same length
    assert len(ground_truth_int_params_list) == len(ground_truth_ext_params_list) == len(ground_truth_sim_params_list)
    print(f"Parameter list length: {len(ground_truth_int_params_list)}")
    
    ground_truths = make_ground_truth_data_list(
        ground_truth_int_params_list,
        ground_truth_ext_params_list,
        ground_truth_sim_params_list,
        product_or_zip="zip"  # Use zip since lists are already aligned
    )
    
    print(f"Generated {len(ground_truths)} ground truth datasets")

    # ========================================================================
    # Create workflow
    # ========================================================================
    
    workflow = SimulationInferenceWorkflow(checkpoint_dir=Path("./vef_checkpoints_two_pass"))
    
    # ========================================================================
    # Run simulations # TODO: Can this be done later on instead?
    # ========================================================================
    
    model_lists, _ = workflow.run(
        int_params_list=ground_truth_int_params_list,
        ext_params_list=ground_truth_ext_params_list,
        sim_params_list=ground_truth_sim_params_list,
        model_class=ViscoElasticFilament_FlowParams_ScalarBending,
        inference_tasks=[],  # Will define inference tasks below
        n_jobs_simulation=-1,
        n_jobs_inference=-1,
    )
    
    print(f"Generated {len(model_lists)} model lists")
    print(f"Model list keys: {list(model_lists.keys())[:5]}...")  # Show first 5
    
    # ========================================================================
    # Define inference task with two-pass pipeline
    # ========================================================================
    
    initial_guesses=[ # TODO: check where to put this and if it is correct
        {"Sp4": 1e-1, "tau_b": 0},  # Initial guess for inference
    ]
    param_keys_to_infer = list(initial_guesses[0].keys())

    # Elastic and viscous parameter lists
    elastic_params = ["Sp4", "Beta"] # Infer in pass 1 (Elastic Pass)
    viscous_params = ["tau_b", "tau_s"]  # Infer in pass 2 (Viscous Pass)
    
    # Create inference task
    inference_task = InferenceTask(
        task_id=0,
        model_indices=tuple((i, j, k) for i in range(len(int_params_list_expanded))
                                    for j in range(len(ext_params_list))
                                    for k in range(len(sim_params_list))),
        make_pipeline_fn=custom_make_inference_pipeline_two_pass,
        pipeline_kwargs={
            "model_class": ViscoElasticFilament_FlowParams_ScalarBending,
            "ground_truths": ground_truths,
            "ext_params_list": ground_truth_ext_params_list,
            "sim_params_list": ground_truth_sim_params_list,
            "param_keys_to_infer": param_keys_to_infer,
            "elastic_params_list": elastic_params,
            "viscous_params_list": viscous_params,
            "loss_fn": rel_mse_loss_fn,
            "optimizer": basinhopping_optimizer,
            "min_w0": 0.0,
            "product_or_zip": "zip",  # Use zip for aligned lists
            "optimizer_kwargs": optimizer_kwargs,
            "n_jobs_per_pass": -1,
        },
        initial_guesses=initial_guesses,
    )
    
    # ========================================================================
    # Run inference
    # ========================================================================
    
    print("\nStarting two-pass inference...")
    model_lists, inference_results = workflow.run(
        int_params_list=ground_truth_int_params_list,
        ext_params_list=ground_truth_ext_params_list,
        sim_params_list=ground_truth_sim_params_list,
        model_class=ViscoElasticFilament_FlowParams_ScalarBending,
        inference_tasks=[inference_task],
        n_jobs_simulation=-1,
        n_jobs_inference=-1,
    )
    
    print(f"\nInference complete!")
    print(f"Results keys: {list(inference_results.keys())}")
    
    # Retrieve and inspect results
    if 0 in inference_results:
        result = inference_results[0]
        print(f"\nPass 1 (Elastic) inferred: {result.get('pass_0_params', {})}")
        print(f"Pass 2 (Viscous) inferred: {result.get('pass_1_params', {})}")
