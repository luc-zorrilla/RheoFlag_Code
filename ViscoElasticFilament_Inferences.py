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

from typing import Dict, Any, Callable
import pytest
import copy

### Optimization schemes

class RandomDisplacementBounds(object):
    """random displacement with bounds:  see: https://stackoverflow.com/a/21967888/2320035
        Modified! (dropped acceptance-rejection sampling for a more specialized approach)
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
    minimum_gradient=False,
    minimum_hessian=True,
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
        minimum_gradient: Compute Jacobian at optimum (default False)
        minimum_hessian: Compute Hessian at optimum (default True)
    
    Args (Local Minimizer):
        local_minimizer_kwargs: Dict with L-BFGS-B configuration by default:
            {
                'method': 'L-BFGS-B', # Local optimization method
                'jac': '3-point',  # Jacobian specification
                'options':{
                    'disp': True ,
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
                'tol': 1e-10,  # Early stopping tolerance
                ... (other basin-hopping options)
            }
    
    Returns:
        OptimizeResult with:
        - x: Optimal parameters
        - fun: Final loss value
        - success: Convergence success flag
        - nit: Number of basin-hopping iterations
        - jacobian: Gradient at optimum (if minimum_gradient=True)
        - hessian: Hessian at optimum (if minimum_hessian=True)
        - X_local, F_local: Local optimization trajectories
        - X_global, F_global, accept_global: Global search trajectory
    """
    
    # --- Set defaults ---
    local_minimizer_kwargs = local_minimizer_kwargs or {
        'method': 'L-BFGS-B', # Local optimization method
        'jac': '3-point',  # Jacobian specification
        'options':{
            'disp': True , # ?
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
            'disp': True ,
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
        
        return minimize(
            fun, x0, args=args, method=method, jac=jac, hess=hess, hessp=hessp,
            bounds=bounds, constraints=constraints, tol=tol, callback=callback,
            options=options
        )
    
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
        "bounds": bounds,
        "callback": local_callback_function,
    })
    
    # --- Global minimizer full configuration ---
    bounded_step = RandomDisplacementBounds(bounds, stepsize=stepsize)
    
    global_minimizer_kwargs.update({
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
    
    # --- Compute gradient/Hessian at convergence ---
    if minimum_gradient or minimum_hessian:
        m = x0.shape[0]
        
        if minimum_gradient:
            if ret.success:
                vec_func = Vectorize_Functional(objective, m)
                # Determine step direction based on boundary proximity
                sl, sb = bounds.residual(x_final)
                if np.all(sl == 0):
                    step_direction = 1
                elif np.all(sb == 0):
                    step_direction = -1
                elif np.all(sl >= 0) & np.all(sb >= 0):
                    step_direction = 0
                else:
                    step_direction = np.inf
                
                g = sd.jacobian(f=vec_func, x=x_final, step_direction=step_direction).df
                ret.setdefault('jacobian', g)
            else:
                ret.setdefault('jacobian', np.ones(m) * np.inf)
        
        if minimum_hessian:
            if ret.success:
                vec_func = Vectorize_Functional(objective, m)
                hess_result = sd.hessian(f=vec_func, x=x_final)
                if hess_result.success:
                    h = hess_result.ddf
                else:
                    h = np.zeros((m, m))
                ret.setdefault('hessian', h)
            else:
                ret.setdefault('hessian', np.zeros((m, m)))
    
    # --- Attach optimization history ---
    ret.X_local = X_local
    ret.F_local = F_local
    ret.X_global = X_global
    ret.F_global = F_global
    ret.accept_global = accept_global
    
    return ret

# ====================
# ======= Loss =======
# ====================

@pytest.fixture
def mse_loss_fn() -> Callable:
    """
    Define Mean Square Error loss function.
    Returns np.inf if prediction is None (failed simulation).
    """
    def loss_fn(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
        if predicted is None:
            return np.inf
        # Flatten arrays
        pred_flat = np.asarray(predicted).flatten()
        truth_flat = np.asarray(ground_truth).flatten()
        
        # Truncate to match lengths
        min_len = min(len(pred_flat), len(truth_flat))
        return np.linalg.norm(pred_flat[:min_len] - truth_flat[:min_len])**2 / np.linalg.norm(truth_flat[:min_len])**2
    
    return loss_fn    

# class TestViscoElasticFilament_OnePassInference_BendingShearElasticity:
#     """ Infer Sp4 and Beta in a one-pass inference, with 
#         - static experimental data in the first pass, with varying flow amplitude

#     Parameters to be inferred:
#         - Sp4
#         - Beta
#     """

#     # ======================
#     # ==== Ground Truth ====
#     # ======================

#     @pytest.fixture
#     def ground_truth_int_params(self):
#         """
#         Define internal parameters with a known Sp4 ground truth value.
#         All other parameters are fixed for inference.
#         """
#         N = 10
#         X0 = StraightLine(N)
#         return {
#             'Sp4': 1e0,           # Ground truth to recover
#             'N': 10,            
#             'k0': 1e13,            
#             'bool_EI': True,      
#             'Beta': 1e0,           # Ground truth to recover
#             'tau_b': 0,           
#             'tau_s': 0,       
#             'gamma': 2,         
#             'n_L': [0,0],            
#             'm_L': 0,             
#             'X_0': X0,  # Initial state
#         }

#     @pytest.fixture
#     def ground_truth_ext_flow_params_static_list(self):
#         """Define external parameters for static response (fixed during inference)."""
#         N = 10
#         return [{
#                 "Lambdas": [[0,0]]*N,
#                 "Zetas": [0]*N,
#                 "A":1e-6,
#                 "w0":0, # Static flow
#                 "psi":np.pi/2,
#             },
#                 {
#                 "Lambdas": [[0,0]]*N,
#                 "Zetas": [0]*N,
#                 "A":2e-6,
#                 "w0":0, # Static flow
#                 "psi":np.pi/2,
#             },      
#             #     {
#             #     "Lambdas": [[0,0]]*N,
#             #     "Zetas": [0]*N,
#             #     "A":3e-6,
#             #     "w0":0, # Static flow
#             #     "psi":np.pi/2,
#             # },              
#             #     {
#             #     "Lambdas": [[0,0]]*N,
#             #     "Zetas": [0]*N,
#             #     "A":4e-6,
#             #     "w0":0, # Static flow
#             #     "psi":np.pi/2,
#             # },
#             #     {
#             #     "Lambdas": [[0,0]]*N,
#             #     "Zetas": [0]*N,
#             #     "A":5e-6,
#             #     "w0":0, # Static flow
#             #     "psi":np.pi/2,
#             # },                            
#         ]

#     @pytest.fixture
#     def ground_truth_sim_params_static_list(self):
#         """
#         Define simulation parameters.
#         Use "broyden1" or "hybr" method for root finding algorithm (fixed point). 
#         Remark: Due to the metastable status of X (clamped), "hybr" works better.
#         """
#         return [{
#             "T_span": (1e6, 2e6),
#             "T_eval": np.linspace(1e6, 2e6, int(1e0)), # minimum two elements here.
#             "method": "hybr",
#             "T_sim_max": 300,
#         },
#         ]

#     @pytest.fixture
#     def ground_truth_flow_data_static_list(
#         self,
#         ground_truth_int_params,
#         ground_truth_ext_flow_params_static_list,
#         ground_truth_sim_params_static_list,
#     ):
#         """
#         Generate ground truth data using the 
#         ViscoElasticFilament_FlowParams_ScalarBending model with known parameters
#         across multiple external and simulation parameter sets.
        
#         Returns a list of ground truth arrays (one per condition).
#         """
#         ground_truths = []
        
#         for ext_params, sim_params in product(
#             ground_truth_ext_flow_params_static_list,
#             ground_truth_sim_params_static_list
#         ):
#             # Instantiate model with ground truth internal parameters
#             instance = ViscoElasticFilament_FlowParams_ScalarBending( 
#                 ground_truth_int_params,
#                 ext_params,
#                 sim_params
#             )
            
#             # Simulate to generate ground truth
#             sim_result = instance.simulate_single()
#             gt_data = sim_result['value']
            
#             ground_truths.append(gt_data)
        
#         return ground_truths
    
#     # ======================
#     # ======= Models =======
#     # ======================

#     @pytest.fixture
#     def elastic_model_flow_sp4_beta_only(
#         self,
#         ground_truth_int_params
#     ):
#         """
#         Create a composed model for ViscoElasticFilament_FlowParams that
#             - only varies Sp4 and Beta
        
#         The embedding function accepts a reduced parameter dict {'Sp4': value, 'Beta': value}
#         and embeds it into the full internal parameters, keeping all others fixed.
#         """
#         fixed_params = ground_truth_int_params.copy()
        
#         def embed_sp4_beta_flow(
#             reduced_int_params: Dict[str, float],
#             ext_params: Any,
#             sim_params: Any,
#         ) -> Dict[str, Any]:
#             """
#             Transform reduced internal parameters into full int_params dict.
            
#             Args:
#                 reduced_int_params: Dict containing {'Sp4': inferred_value, 'Beta': inferred_value}
#                 ext_params: Passed through unchanged (not modified here)
#                 sim_params: Passed through unchanged (not modified here)
            
#             Returns:
#                 Full int_params dict with Sp4 and Beta updated, all other values fixed.
#             """
#             full_params = fixed_params.copy()
            
#             # Update only Sp4 and Beta; all other parameters remain fixed
#             if 'Sp4' in reduced_int_params:
#                 full_params['Sp4'] = reduced_int_params['Sp4']
#             if 'Beta' in reduced_int_params:
#                 full_params['Beta'] = reduced_int_params['Beta']                
            
#             return full_params
        
#         # Create composed model with the embedding function
#         MultiElasticModel = compose_model(
#             ViscoElasticFilament_FlowParams_ScalarBending,
#             compose_int_params=embed_sp4_beta_flow,
#         )
#         return MultiElasticModel

#     # ==========================
#     # ======= Optimizers =======
#     # ==========================

#     @pytest.fixture
#     def basinhopping_optimizer_instance(self):
#         """
#         Return the basinhopping optimizer function with standard configuration.
#         """
#         return basinhopping_optimizer        

#     @pytest.fixture
#     def optimizer_kwargs_sp4_beta(self):
#         return {
#             'bounds': Bounds(lb=[1e-6, 0], ub=[np.inf, np.inf]),
#             'minimum_gradient': False,
#             'minimum_hessian': False,
#             'local_minimizer_kwargs': {
#                 'method': 'L-BFGS-B',
#                 'jac': '3-point',
#                 'options': {
#                     'disp': True,
#                     'ftol': 1e-8,
#                     'gtol': 1e-8,
#                     'eps': 1e-8,
#                     'finite_diff_rel_step': 1e-6,
#                 },
#             },
#             'global_minimizer_kwargs': {
#                 'niter': 9,
#                 'T': 0,
#                 'stepsize': 5,
#                 'tol': 1e-10,
#             }
#         }

#     # ==========================
#     # ========= Passes =========
#     # ==========================

#     # Pass 1: Reduced model, infer Sp4 and Beta only
#     @pytest.fixture
#     def pass_1(
#         self,
#         elastic_model_flow_sp4_beta_only,
#         ground_truth_ext_flow_params_static_list,
#         ground_truth_sim_params_static_list,
#         ground_truth_flow_data_static_list,
#         basinhopping_optimizer_instance,
#         optimizer_kwargs_sp4_beta,
#     ):
#         """ First pass for the viscoelastic inference. """
        
#         first_pass = PipelinePass(
#             name="Sp4-Beta Inference (MultiElastic Model)",
#             model_class=elastic_model_flow_sp4_beta_only,
#             ground_truths=ground_truth_flow_data_static_list,
#             ext_params_list=ground_truth_ext_flow_params_static_list,
#             sim_params_list=ground_truth_sim_params_static_list,
#             param_keys_to_infer=['Sp4', 'Beta'],
#             fixed_params={},
#             optimizer=basinhopping_optimizer_instance,
#             optimizer_kwargs=optimizer_kwargs_sp4_beta,
#         )

#         return first_pass

#     # ============================
#     # ========= Pipeline =========
#     # ============================
#     @pytest.fixture
#     def multielastic_pipeline(self, pass_1, mse_loss_fn):

#         pipeline = InferencePipeline(
#             passes=[pass_1],
#             loss_fn=mse_loss_fn,
#             n_jobs_per_pass=-1,  # Use all cores within each pass
#         )
#         return pipeline

#     # ============================
#     # ========= Tests =========
#     # ============================
#     def test_multielastic_inference(
#         self,
#         multielastic_pipeline,
#     ):
        
#         # Multiple initial guesses: Pass 1 for Sp4, Pass 2 for tau_b
#         initial_guesses_per_pass = [
#             [
#                 {'Sp4': 10.0, 'Beta':0.0},
#                 # {'Sp4': 0.1, 'Beta':1.0},
#                 # {'Sp4': 1.0, 'Beta':0.0},
#             ],
#         ]
        
#         # ===== ACT =====
#         results = multielastic_pipeline.run(initial_guesses_per_pass, verbose=True)
        
#         # ===== ASSERT =====
        
#         # Check that we got exactly one result (one pass)
#         assert len(results) == 1, "Expected one InferenceResult for one-pass pipeline"
        
#         result_pass1 = results[0]
        
#         # ===== PASS 1: Sp4-Beta INFERENCE =====
#         print("\n" + "="*60)
#         print("PASS 1 ASSERTIONS (Sp4-Beta inference)")
#         print("="*60)
        
#         # 1. Check Pass 1 convergence
#         assert result_pass1.success, (
#             f"Pass 1 optimization did not converge. Message: {result_pass1.message}"
#         )
#         assert result_pass1.iterations > 0, "Pass 1: No iterations were performed"
        
#         # 2. Check Sp4 exists and is physically reasonable
#         assert 'Sp4' in result_pass1.params, "Sp4 not in Pass 1 inferred parameters"
#         sp4_inferred = result_pass1.params['Sp4']
#         assert sp4_inferred > 0, f"Sp4 must be positive, got {sp4_inferred}"
#         assert sp4_inferred < 1e6, f"Sp4 unreasonably large: {sp4_inferred}"

#         # 3. Check Beta exists and is physically reasonable
#         assert 'Beta' in result_pass1.params, "Beta not in Pass 1 inferred parameters"
#         Beta_inferred = result_pass1.params['Beta']
#         assert Beta_inferred >= 0, f"Beta must be positive, got {Beta_inferred}"
#         assert Beta_inferred < 1e6, f"Beta unreasonably large: {Beta_inferred}"
        
#         # 4. Check Pass 1 loss
#         assert result_pass1.loss > 0, "Pass 1 loss should be positive"
#         assert result_pass1.loss < 1e2, (
#             f"Pass 1 loss suspiciously high: {result_pass1.loss}. Check model/data scale."
#         )
        
#         # 5. Check Pass 1 uncertainty quantification
#         assert result_pass1.covariance is not None, "Pass 1: Covariance not computed"
#         assert result_pass1.hessian is not None, "Pass 1: Hessian not computed"
#         assert result_pass1.std_errors is not None, "Pass 1: Standard errors not computed"
#         assert len(result_pass1.std_errors) == 2, "Pass 1: Expected 2 std_error for 2 parameter (Sp4, Beta)"
        
#         std_err_sp4 = result_pass1.std_errors[0]
#         assert std_err_sp4 > 0, f"Pass 1: Sp4 std error must be positive, got {std_err_sp4}"
#         assert std_err_sp4 < sp4_inferred * 10, (
#             f"Pass 1: Std error unreasonably large relative to Sp4: "
#             f"Sp4={sp4_inferred:.3e} ± {std_err_sp4:.3e}"
#         )
        
#         std_err_Beta = result_pass1.std_errors[1]
#         assert std_err_Beta > 0, f"Pass 1: Beta std error must be positive, got {std_err_Beta}"
#         assert std_err_Beta < Beta_inferred * 10, (
#             f"Pass 1: Std error unreasonably large relative to Beta: "
#             f"Sp4={Beta_inferred:.3e} ± {std_err_Beta:.3e}"
#         )

#         print(f"✓ Pass 1 - Sp4 inferred: {sp4_inferred:.6e} ± {std_err_sp4:.6e}")
#         print(f"✓ Pass 1 - Beta inferred: {Beta_inferred:.6e} ± {std_err_Beta:.6e}")
#         print(f"✓ Pass 1 - Final loss: {result_pass1.loss:.8e}")
#         print(f"✓ Pass 1 - Iterations: {result_pass1.iterations}")
        
#         # ===== PIPELINE-LEVEL ASSERTIONS =====
#         print("\n" + "="*60)
#         print("PIPELINE-LEVEL ASSERTIONS")
#         print("="*60)
        
#         # Check parameter trajectory
#         trajectory = multielastic_pipeline.get_parameter_trajectory()
#         assert 'Sp4' in trajectory, "Sp4 not in trajectory"
#         assert 'Beta' in trajectory, "Beta not in trajectory"
#         assert len(trajectory['Sp4']) == 1, "Sp4 trajectory should have 1 steps"
#         assert len(trajectory['Beta']) == 1, "Beta trajectory should have 1 steps"
        
#         # Generate and verify summary
#         summary = multielastic_pipeline.summary()
#         print(summary)
#         assert summary, "Summary generation failed"
        
#         print("\n✓ ALl one-pass inference tests passed!")

class TestViscoElasticFilament_OnePassInference_BendingShearViscosity:
    """ Infer tau_b and tau_s in a one-pass inference, with 
        - fixed Sp4 = 1, Beta = 1
        - dynamic experimental data in the first pass, with varying ?? # TODO: think about this

    Parameters to be inferred:
        - tau_b
        - tau_s
    """

    # ======================
    # ==== Ground Truth ====
    # ======================

    @pytest.fixture
    def ground_truth_int_params(self):
        """
        Define internal parameters with known tau_b and tau_s ground truth value.
        All other parameters are fixed for inference.
        """
        N = 10
        X0 = StraightLine(N)
        return {
            'Sp4': 1e0,          
            'N': 10,            
            'k0': 1e13,            
            'bool_EI': True,      
            'Beta': 1e0,           
            'tau_b': 1e0,       # Ground truth to recover    
            'tau_s': 1e0,       # Ground truth to recover
            'gamma': 2,         
            'n_L': [0,0],            
            'm_L': 0,             
            'X_0': X0,  # Initial state
        }

    @pytest.fixture
    def ground_truth_ext_flow_params_dynamic_list(self):
        """Define external parameters for dynamic response (fixed during inference)."""
        N = 10
        return [{
                "Lambdas": [[0,0]]*N,
                "Zetas": [0]*N,
                "A":1e-6,
                "w0":1e0, # Dynamic flow
                "psi":np.pi/2,
            },
            {
                "Lambdas": [[0,0]]*N,
                "Zetas": [0]*N,
                "A":1e-6,
                "w0":1e-1, # Dynamic flow
                "psi":np.pi/2,
            },            
            #     {
            #     "Lambdas": [[0,0]]*N,
            #     "Zetas": [0]*N,
            #     "A":2e-6,
            #     "w0":1e1, # Static flow
            #     "psi":np.pi/2,
            # },               
        ]


    @pytest.fixture
    def ground_truth_sim_params_dynamic_list(self):
        """
        Define simulation parameters.
        Use BDF method for faster convergence compared to RK45.
        """
        return [{
            "T_span": (1e1, 2e1),
            "T_eval": np.linspace(1e1, 2e1, int(1e1)), # minimum two elements here.
            "method": "BDF",
            "T_sim_max": 300,
        },
        ]

    @pytest.fixture
    def ground_truth_flow_data_dynamic_list(
        self,
        ground_truth_int_params,
        ground_truth_ext_flow_params_dynamic_list,
        ground_truth_sim_params_dynamic_list,
    ):
        """
        Generate ground truth data using the 
        ViscoElasticFilament_FlowParams_ScalarBending model with known parameters
        across multiple external and simulation parameter sets.
        
        Returns a list of ground truth arrays (one per condition).
        """
        ground_truths = []
        
        for ext_params, sim_params in product(
            ground_truth_ext_flow_params_dynamic_list,
            ground_truth_sim_params_dynamic_list
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
    
    # ======================
    # ======= Models =======
    # ======================

    @pytest.fixture
    def viscous_model_flow_tau_b_tau_s_only(
        self,
        ground_truth_int_params
    ):
        """
        Create a composed model for ViscoElasticFilament_FlowParams that
            - only varies tau_b and tau_s
        
        The embedding function accepts a reduced parameter dict {'tau_b': value, 'tau_s': value}
        and embeds it into the full internal parameters, keeping all others fixed.
        """
        fixed_params = ground_truth_int_params.copy()
        
        def embed_tau_b_tau_s_flow(
            reduced_int_params: Dict[str, float],
            ext_params: Any,
            sim_params: Any,
        ) -> Dict[str, Any]:
            """
            Transform reduced internal parameters into full int_params dict.
            
            Args:
                reduced_int_params: Dict containing {'tau_b': inferred_value, 'tau_s': inferred_value}
                ext_params: Passed through unchanged (not modified here)
                sim_params: Passed through unchanged (not modified here)
            
            Returns:
                Full int_params dict with tau_b and tau_s updated, all other values fixed.
            """
            full_params = fixed_params.copy()
            
            # Update only Sp4 and Beta; all other parameters remain fixed
            if 'tau_b' in reduced_int_params:
                full_params['tau_b'] = reduced_int_params['tau_b']
            if 'tau_s' in reduced_int_params:
                full_params['tau_s'] = reduced_int_params['tau_s']                
            
            return full_params
        
        # Create composed model with the embedding function
        MultiViscousModel = compose_model(
            ViscoElasticFilament_FlowParams_ScalarBending,
            compose_int_params=embed_tau_b_tau_s_flow,
        )
        return MultiViscousModel

    # ==========================
    # ======= Optimizers =======
    # ==========================

    @pytest.fixture
    def basinhopping_optimizer_instance(self):
        """
        Return the basinhopping optimizer function with standard configuration.
        """
        return basinhopping_optimizer        

    @pytest.fixture
    def optimizer_kwargs_tau_b_tau_s(self):
        return {
            'bounds': Bounds(lb=[0, 0], ub=[np.inf, np.inf]),
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
        }

    # ==========================
    # ========= Passes =========
    # ==========================

    # Pass 1: Reduced model, infer tau_b and tau_s only
    @pytest.fixture
    def pass_1(
        self,
        viscous_model_flow_tau_b_tau_s_only,
        ground_truth_ext_flow_params_dynamic_list,
        ground_truth_sim_params_dynamic_list,
        ground_truth_flow_data_dynamic_list,
        basinhopping_optimizer_instance,
        optimizer_kwargs_tau_b_tau_s,
    ):
        """ First pass for the viscoelastic inference. """
        
        first_pass = PipelinePass(
            name="tau_b-tau_s Inference (MultiViscous Model)",
            model_class=viscous_model_flow_tau_b_tau_s_only,
            ground_truths=ground_truth_flow_data_dynamic_list,
            ext_params_list=ground_truth_ext_flow_params_dynamic_list,
            sim_params_list=ground_truth_sim_params_dynamic_list,
            param_keys_to_infer=['tau_b', 'tau_s'],
            fixed_params={},
            optimizer=basinhopping_optimizer_instance,
            optimizer_kwargs=optimizer_kwargs_tau_b_tau_s,
        )

        return first_pass

    # ============================
    # ========= Pipeline =========
    # ============================
    @pytest.fixture
    def multiviscous_pipeline(self, pass_1, mse_loss_fn):

        pipeline = InferencePipeline(
            passes=[pass_1],
            loss_fn=mse_loss_fn,
            n_jobs_per_pass=-1,  # Use all cores within each pass
        )
        return pipeline

    # ============================
    # ========= Tests =========
    # ============================
    def test_multiviscous_inference(
        self,
        multiviscous_pipeline,
    ):
        
        # Multiple initial guesses: Pass 1 for tau_b
        initial_guesses_per_pass = [
            [
                {'tau_b': 0.0, 'tau_s':0.0},
                # {'tau_b': 0.0, 'tau_s':1.0},
                # {'tau_b': 1.0, 'tau_s':0.0},
            ],
        ]
        
        # ===== ACT =====
        results = multiviscous_pipeline.run(initial_guesses_per_pass, verbose=True)
        
        # ===== ASSERT =====
        
        # Check that we got exactly one result (one pass)
        assert len(results) == 1, "Expected one InferenceResult for one-pass pipeline"
        
        result_pass1 = results[0]
        
        # ===== PASS 1: Sp4-Beta INFERENCE =====
        print("\n" + "="*60)
        print("PASS 1 ASSERTIONS (tau_b-tau_s inference)")
        print("="*60)
        
        # 1. Check Pass 1 convergence
        assert result_pass1.success, (
            f"Pass 1 optimization did not converge. Message: {result_pass1.message}"
        )
        assert result_pass1.iterations > 0, "Pass 1: No iterations were performed"
        
        # 2. Check tau_b exists and is physically reasonable
        assert 'tau_b' in result_pass1.params, "tau_b not in Pass 1 inferred parameters"
        tau_b_inferred = result_pass1.params['tau_b']
        assert tau_b_inferred >= 0, f"tau_b must be positive, got {tau_b_inferred}"
        assert tau_b_inferred < 1e6, f"tau_b unreasonably large: {tau_b_inferred}"

        # 3. Check tau_s exists and is physically reasonable
        assert 'tau_s' in result_pass1.params, "tau_s not in Pass 1 inferred parameters"
        tau_s_inferred = result_pass1.params['tau_s']
        assert tau_s_inferred >= 0, f"tau_s must be positive, got {tau_s_inferred}"
        assert tau_s_inferred < 1e6, f"tau_s unreasonably large: {tau_s_inferred}"
        
        # 4. Check Pass 1 loss
        assert result_pass1.loss > 0, "Pass 1 loss should be positive"
        assert result_pass1.loss < 1e2, (
            f"Pass 1 loss suspiciously high: {result_pass1.loss}. Check model/data scale."
        )
        
        # 5. Check Pass 1 uncertainty quantification
        assert result_pass1.covariance is not None, "Pass 1: Covariance not computed"
        assert result_pass1.hessian is not None, "Pass 1: Hessian not computed"
        assert result_pass1.std_errors is not None, "Pass 1: Standard errors not computed"
        assert len(result_pass1.std_errors) == 2, "Pass 1: Expected 2 std_error for 2 parameter (tau_b, tau_s)"
        
        std_err_tau_b = result_pass1.std_errors[0]
        assert std_err_tau_b > 0, f"Pass 1: tau_b std error must be positive, got {std_err_tau_b}"
        assert std_err_tau_b < tau_b_inferred * 10, (
            f"Pass 1: Std error unreasonably large relative to tau_b: "
            f"tau_b={tau_b_inferred:.3e} ± {std_err_tau_b:.3e}"
        )
        
        std_err_tau_s = result_pass1.std_errors[1]
        assert std_err_tau_s > 0, f"Pass 1: tau_s std error must be positive, got {std_err_tau_s}"
        assert std_err_tau_s < tau_s_inferred * 10, (
            f"Pass 1: Std error unreasonably large relative to tau_s: "
            f"tau_b={tau_s_inferred:.3e} ± {std_err_tau_s:.3e}"
        )

        print(f"✓ Pass 1 - tau_b inferred: {tau_b_inferred:.6e} ± {std_err_tau_b:.6e}")
        print(f"✓ Pass 1 - tau_s inferred: {tau_s_inferred:.6e} ± {std_err_tau_s:.6e}")
        print(f"✓ Pass 1 - Final loss: {result_pass1.loss:.8e}")
        print(f"✓ Pass 1 - Iterations: {result_pass1.iterations}")
        
        # ===== PIPELINE-LEVEL ASSERTIONS =====
        print("\n" + "="*60)
        print("PIPELINE-LEVEL ASSERTIONS")
        print("="*60)
        
        # Check parameter trajectory
        trajectory = multielastic_pipeline.get_parameter_trajectory()
        assert 'tau_b' in trajectory, "tau_b not in trajectory"
        assert 'tau_s' in trajectory, "tau_s not in trajectory"
        assert len(trajectory['tau_b']) == 1, "tau_b trajectory should have 1 steps"
        assert len(trajectory['tau_s']) == 1, "tau_s trajectory should have 1 steps"
        
        # Generate and verify summary
        summary = multielastic_pipeline.summary()
        print(summary)
        assert summary, "Summary generation failed"
        
        print("\n✓ ALl one-pass inference tests passed!")

# class TestViscoElasticFilament_TwoPassInference_BendingElasticityShearViscosity:
#     """ Infer Sp4 and tau_s in a two-pass inference, with 
#         - static experimental data in the first pass,
#         - dynamic experimental data in the second pass.

#     Parameters to be inferred:
#         - Sp4: 1.0
#         - tau_s: 1e6
#     """

#     # ======================
#     # ==== Ground Truth ====
#     # ======================

#     @pytest.fixture
#     def ground_truth_int_params(self):
#         """
#         Define internal parameters with a known Sp4 ground truth value.
#         All other parameters are fixed for inference.
#         """
#         N = 10
#         X0 = StraightLine(N)
#         return {
#             'Sp4': 1.0,           # Ground truth to recover
#             'N': 10,            
#             'k0': 1e13,            
#             'bool_EI': True,      
#             'Beta':0,        
#             'taus_b': [0]*(N-1),  
#             'tau_s': 1e6,        # Ground truth to recover
#             'gamma': 2,         
#             'n_L': [0,0],            
#             'm_L': 0,             
#             'X_0': X0,  # Initial state
#         }

#     @pytest.fixture
#     def ground_truth_ext_flow_params_static(self):
#         """Define external parameters for static response (fixed during inference)."""
#         N = 10
#         return {
#             "Lambdas": [[0,0]]*N,
#             "Zetas": [0]*N,
#             "A":1e-6,
#             "w0":0, # Static
#             "psi":np.pi/2,
#         }

#     @pytest.fixture
#     def ground_truth_ext_flow_params_dynamic(self):
#         """Define external parameters for static response (fixed during inference)."""
#         N = 10
#         return {
#             "Lambdas": [[0,0]]*N,
#             "Zetas": [0]*N,
#             "A":1e-6,
#             "w0":1e-6, # Dynamic
#             "psi":np.pi/2,
#         }        

#     @pytest.fixture
#     def ground_truth_sim_params_static(self):
#         """
#         Define simulation parameters.
#         Use BDF method for faster convergence compared to RK45.
#         """
#         return {
#             "T_span": (1e9, 2e9),
#             "T_eval": np.linspace(1e9, 2e9, int(2e0)), # minimum two elements here.
#             "method": "BDF",
#             "T_sim_max": 300,
#         }

#     @pytest.fixture
#     def ground_truth_sim_params_dynamic(self):
#         """
#         Define simulation parameters.
#         Use BDF method for faster convergence compared to RK45.
#         """
#         return {
#             "T_span": (1e7, 2e7),
#             "T_eval": np.linspace(1e7, 2e7, int(1e1)),
#             "method": "BDF",
#             "T_sim_max": 300,
#         }

#     @pytest.fixture
#     def ground_truth_flow_data_static(
#         self, 
#         ground_truth_int_params, 
#         ground_truth_ext_flow_params_static, 
#         ground_truth_sim_params_static,
#     ):
#         """
#         Generate ground truth synthetic data by simulating ViscoElasticFilament
#         with known parameters.
        
#         Returns:
#             np.ndarray: Simulated trajectory (shape depends on model)
#         """
#         model = ViscoElasticFilament_FlowParams(
#             int_params=ground_truth_int_params,
#             ext_params=ground_truth_ext_flow_params_static,
#             sim_params=ground_truth_sim_params_static
#         )
#         output = model.simulate_single()
        
#         assert output['value'] is not None, (
#             f"Ground truth simulation failed. Full output: {output}"
#         )

#         return output['value']


#     @pytest.fixture
#     def ground_truth_flow_data_dynamic(
#         self,
#         ground_truth_int_params, 
#         ground_truth_ext_flow_params_dynamic, 
#         ground_truth_sim_params_dynamic,
#     ):
#         """
#         Generate ground truth synthetic data by simulating ViscoElasticFilament
#         with known parameters.
        
#         Returns:
#             np.ndarray: Simulated trajectory (shape depends on model)
#         """
#         model = ViscoElasticFilament_FlowParams(
#             int_params=ground_truth_int_params,
#             ext_params=ground_truth_ext_flow_params_dynamic,
#             sim_params=ground_truth_sim_params_dynamic,
#         )
#         output = model.simulate_single()
        
#         assert output['value'] is not None, (
#             f"Ground truth simulation failed. Full output: {output}"
#         )

#         return output['value']
    
#     # ======================
#     # ======= Models =======
#     # ======================

#     @pytest.fixture
#     def elastic_model_flow_sp4_only(
#         self,
#         ground_truth_int_params
#     ):
#         """
#         Create a composed model for ViscoElasticFilament_FlowParams that
#             - only varies Sp4
#             - sets tau_s to zero (elastic model)
        
#         The embedding function accepts a reduced parameter dict {'Sp4': value}
#         and embeds it into the full internal parameters, keeping all others fixed.
#         """
#         fixed_params = ground_truth_int_params.copy()
#         fixed_params['tau_s'] = 0 # Set dynamic parameters to 0.
        
#         def embed_sp4_flow(
#             reduced_int_params: Dict[str, float],
#             ext_params: Any,
#             sim_params: Any,
#         ) -> Dict[str, Any]:
#             """
#             Transform reduced internal parameters into full int_params dict.
            
#             Args:
#                 reduced_int_params: Dict containing {'Sp4': inferred_value}
#                 ext_params: Passed through unchanged (not modified here)
#                 sim_params: Passed through unchanged (not modified here)
            
#             Returns:
#                 Full int_params dict with Sp4 updated, all other values fixed.
#             """
#             full_params = fixed_params.copy()
            
#             # Update only Sp4; all other parameters remain fixed
#             if 'Sp4' in reduced_int_params:
#                 full_params['Sp4'] = reduced_int_params['Sp4']
            
#             return full_params
        
#         # Create composed model with the embedding function
#         ElasticModel = compose_model(
#             ViscoElasticFilament_FlowParams,
#             compose_int_params=embed_sp4_flow,
#         )
#         return ElasticModel

#     @pytest.fixture
#     def viscoelastic_model_flow_sp4_tau_s_only(
#         self,
#         ground_truth_int_params,
#     ):
#         """
#         Create a composed model for ViscoElasticFilament_FlowParams that only varies Sp4 and tau_s.
        
#         The embedding function accepts a reduced parameter dict {'Sp4': value, 'tau_s': value}
#         and embeds it into the full internal parameters, keeping all others fixed.
#         """
#         fixed_params = ground_truth_int_params.copy()
        
#         def embed_sp4_tau_s_flow(
#             reduced_int_params: Dict[str, float],
#             ext_params: Any,
#             sim_params: Any,
#         ) -> Dict[str, Any]:
#             """
#             Transform reduced internal parameters into full int_params dict.
            
#             Args:
#                 reduced_int_params: Dict containing {'Sp4': inferred_value, 'tau_s': inferred_value}
#                 ext_params: Passed through unchanged (not modified here)
#                 sim_params: Passed through unchanged (not modified here)
            
#             Returns:
#                 Full int_params dict with Sp4 and tau_s updated, all other values fixed.
#             """
#             full_params = fixed_params.copy()
            
#             # Update only Sp4; all other parameters remain fixed
#             if 'Sp4' in reduced_int_params:
#                 full_params['Sp4'] = reduced_int_params['Sp4']

#             if 'tau_s' in reduced_int_params:
#                 full_params['tau_s'] = reduced_int_params['tau_s']
            
#             return full_params
        
#         # Create composed model with the embedding function
#         ViscoElasticModel = compose_model(
#             ViscoElasticFilament_FlowParams,
#             compose_int_params=embed_sp4_tau_s_flow,
#         )
#         return ViscoElasticModel

#     # ==========================
#     # ======= Optimizers =======
#     # ==========================

#     @pytest.fixture
#     def basinhopping_optimizer_instance(self):
#         """
#         Return the basinhopping optimizer function with standard configuration.
#         """
#         return basinhopping_optimizer        

#     @pytest.fixture
#     def optimizer_kwargs_sp4(self):
#         return {
#             'bounds': Bounds(lb=[1e-6], ub=[np.inf]),
#             'minimum_gradient': False,
#             'minimum_hessian': False,
#             'local_minimizer_kwargs': {
#                 'method': 'L-BFGS-B',
#                 'jac': '3-point',
#                 'options': {
#                     'disp': True,
#                     'ftol': 1e-8,
#                     'gtol': 1e-8,
#                     'eps': 1e-8,
#                     'finite_diff_rel_step': 1e-6,
#                 },
#             },
#             'global_minimizer_kwargs': {
#                 'niter': 9,
#                 'T': 0,
#                 'stepsize': 5,
#                 'tol': 1e-10,
#             }
#         }

#     @pytest.fixture
#     def optimizer_kwargs_tau_s(self, optimizer_kwargs_sp4):
#         optimizer_kwargs = optimizer_kwargs_sp4
#         optimizer_kwargs['bounds'] = Bounds(lb=[0], ub=[np.inf])
#         return optimizer_kwargs

#     # ==========================
#     # ========= Passes =========
#     # ==========================

#     # Pass 1: Reduced model, infer Sp4 only
#     @pytest.fixture
#     def pass_1(
#         self,
#         elastic_model_flow_sp4_only,
#         ground_truth_ext_flow_params_static,
#         ground_truth_sim_params_static,
#         ground_truth_flow_data_static,
#         basinhopping_optimizer_instance,
#         optimizer_kwargs_sp4,
#     ):
#         """ First pass for the viscoelastic inference. """
        
#         first_pass = PipelinePass(
#             name="Sp4 Inference (Elastic Model)",
#             model_class=elastic_model_flow_sp4_only,
#             ground_truths=[ground_truth_flow_data_static],
#             ext_params_list=[ground_truth_ext_flow_params_static],
#             sim_params_list=[ground_truth_sim_params_static],
#             param_keys_to_infer=['Sp4'],
#             fixed_params={},
#             optimizer=basinhopping_optimizer_instance,
#             optimizer_kwargs=optimizer_kwargs_sp4,
#         )

#         return first_pass

#     # Pass 2:
#     @pytest.fixture
#     def pass_2(
#         self,
#         viscoelastic_model_flow_sp4_tau_s_only,
#         ground_truth_ext_flow_params_dynamic,
#         ground_truth_sim_params_dynamic,
#         ground_truth_flow_data_dynamic,
#         basinhopping_optimizer_instance,
#         optimizer_kwargs_tau_s,
#     ):
#         """ Second pass for the viscoelastic inference. """
        
#         second_pass = PipelinePass(
#             name="tau_s Inference (ViscoElastic Model)",
#             model_class=viscoelastic_model_flow_sp4_tau_s_only,
#             ground_truths=[ground_truth_flow_data_dynamic],
#             ext_params_list=[ground_truth_ext_flow_params_dynamic],
#             sim_params_list=[ground_truth_sim_params_dynamic],
#             param_keys_to_infer=['tau_s'],
#             fixed_params={},
#             optimizer=basinhopping_optimizer_instance,
#             optimizer_kwargs=optimizer_kwargs_tau_s,
#         )

#         return second_pass

#     # ============================
#     # ========= Pipeline =========
#     # ============================
#     @pytest.fixture
#     def viscoelastic_pipeline(self, pass_1, pass_2, mse_loss_fn):

#         pipeline = InferencePipeline(
#             passes=[pass_1, pass_2],
#             loss_fn=mse_loss_fn,
#             n_jobs_per_pass=-1,  # Use all cores within each pass
#         )
#         return pipeline

#     # ============================
#     # ========= Tests =========
#     # ============================
#     def test_viscoelastic_inference(
#         self,
#         viscoelastic_pipeline,
#     ):

#         # Multiple initial guesses: Pass 1 for Sp4, Pass 2 for tau_s
#         initial_guesses_per_pass = [
#             [
#                 # {'Sp4': 0.1},
#                 {'Sp4': 10.0},
#             ],
#             [
#                 {'tau_s': 0},
#                 # {'tau_s': 10.0},
#             ],
#         ]
        
#         # ===== ACT =====
#         results = viscoelastic_pipeline.run(initial_guesses_per_pass, verbose=True)
        
#         # ===== ASSERT =====
        
#         # Check that we got exactly two results (two passes)
#         assert len(results) == 2, "Expected two InferenceResult for two-pass pipeline"
        
#         result_pass1, result_pass2 = results
        
#         # ===== PASS 1: Sp4 INFERENCE =====
#         print("\n" + "="*60)
#         print("PASS 1 ASSERTIONS (Sp4 inference)")
#         print("="*60)
        
#         # 1. Check Pass 1 convergence
#         assert result_pass1.success, (
#             f"Pass 1 optimization did not converge. Message: {result_pass1.message}"
#         )
#         assert result_pass1.iterations > 0, "Pass 1: No iterations were performed"
        
#         # 2. Check Sp4 exists and is physically reasonable
#         assert 'Sp4' in result_pass1.params, "Sp4 not in Pass 1 inferred parameters"
#         sp4_inferred = result_pass1.params['Sp4']
#         assert sp4_inferred > 0, f"Sp4 must be positive, got {sp4_inferred}"
#         assert sp4_inferred < 1e6, f"Sp4 unreasonably large: {sp4_inferred}"
        
#         # 3. Check Pass 1 loss
#         assert result_pass1.loss > 0, "Pass 1 loss should be positive"
#         assert result_pass1.loss < 1e2, (
#             f"Pass 1 loss suspiciously high: {result_pass1.loss}. Check model/data scale."
#         )
        
#         # 4. Check Pass 1 uncertainty quantification
#         assert result_pass1.covariance is not None, "Pass 1: Covariance not computed"
#         assert result_pass1.hessian is not None, "Pass 1: Hessian not computed"
#         assert result_pass1.std_errors is not None, "Pass 1: Standard errors not computed"
#         assert len(result_pass1.std_errors) == 1, "Pass 1: Expected 1 std_error for 1 parameter (Sp4)"
        
#         std_err_sp4 = result_pass1.std_errors[0]
#         assert std_err_sp4 > 0, f"Pass 1: Sp4 std error must be positive, got {std_err_sp4}"
#         assert std_err_sp4 < sp4_inferred * 10, (
#             f"Pass 1: Std error unreasonably large relative to Sp4: "
#             f"Sp4={sp4_inferred:.3e} ± {std_err_sp4:.3e}"
#         )
        
#         print(f"✓ Pass 1 - Sp4 inferred: {sp4_inferred:.6e} ± {std_err_sp4:.6e}")
#         print(f"✓ Pass 1 - Final loss: {result_pass1.loss:.8e}")
#         print(f"✓ Pass 1 - Iterations: {result_pass1.iterations}")
        
#         # ===== PASS 2: tau_s INFERENCE (Sp4 FIXED) =====
#         print("\n" + "="*60)
#         print("PASS 2 ASSERTIONS (tau_s inference with Sp4 fixed)")
#         print("="*60)
        
#         # 1. Check Pass 2 convergence
#         assert result_pass2.success, (
#             f"Pass 2 optimization did not converge. Message: {result_pass2.message}"
#         )
#         assert result_pass2.iterations > 0, "Pass 2: No iterations were performed"
        
#         # 2. Check tau_s exists and is physically reasonable
#         assert 'tau_s' in result_pass2.params, "tau_s not in Pass 2 inferred parameters"
#         tau_s_inferred = result_pass2.params['tau_s']
#         assert tau_s_inferred >= 0, f"tau_s must be non-negative, got {tau_s_inferred}"
#         assert tau_s_inferred < 1e4, f"tau_s unreasonably large: {tau_s_inferred}"
        
#         # 3. Check Pass 2 loss
#         assert result_pass2.loss > 0, "Pass 2 loss should be positive"
#         assert result_pass2.loss < 1e2, (
#             f"Pass 2 loss suspiciously high: {result_pass2.loss}. Check model/data scale."
#         )
        
#         # 4. Check Pass 2 uncertainty quantification (only for tau_s, not Sp4)
#         assert result_pass2.covariance is not None, "Pass 2: Covariance not computed"
#         assert result_pass2.hessian is not None, "Pass 2: Hessian not computed"
#         assert result_pass2.std_errors is not None, "Pass 2: Standard errors not computed"
#         assert len(result_pass2.std_errors) == 1, "Pass 2: Expected 1 std_error for 1 inferred parameter (tau_s)"
        
#         std_err_tau_s = result_pass2.std_errors[0]
#         assert std_err_tau_s > 0, f"Pass 2: tau_s std error must be positive, got {std_err_tau_s}"
#         assert std_err_tau_s < tau_s_inferred * 10, (
#             f"Pass 2: Std error unreasonably large relative to tau_s: "
#             f"tau_s={tau_s_inferred:.3e} ± {std_err_tau_s:.3e}"
#         )
        
#         print(f"✓ Pass 2 - tau_s inferred: {tau_s_inferred:.6e} ± {std_err_tau_s:.6e}")
#         print(f"✓ Pass 2 - Sp4 fixed (from Pass 1): {result_pass1.params['Sp4']:.6e}")
#         print(f"✓ Pass 2 - Final loss: {result_pass2.loss:.8e}")
#         print(f"✓ Pass 2 - Iterations: {result_pass2.iterations}")
        
#         # ===== PIPELINE-LEVEL ASSERTIONS =====
#         print("\n" + "="*60)
#         print("PIPELINE-LEVEL ASSERTIONS")
#         print("="*60)
        
#         # Check parameter trajectory
#         trajectory = viscoelastic_pipeline.get_parameter_trajectory()
#         assert 'Sp4' in trajectory, "Sp4 not in trajectory"
#         assert 'tau_s' in trajectory, "tau_s not in trajectory"
#         assert len(trajectory['Sp4']) == 2, "Sp4 trajectory should have 2 steps"
#         assert len(trajectory['tau_s']) == 2, "tau_s trajectory should have 2 steps"
        
#         # Sp4 should stay constant (inferred in Pass 1, fixed in Pass 2)
#         assert trajectory['Sp4'][0] == trajectory['Sp4'][1], (
#             f"Sp4 should be constant across passes. "
#             f"Pass1={trajectory['Sp4'][0]}, Pass2={trajectory['Sp4'][1]}"
#         )
        
#         # tau_s should appear in Pass 2 (None or absent in Pass 1)
#         assert trajectory['tau_s'][1] == tau_s_inferred, (
#             f"tau_s trajectory doesn't match Pass 2 result"
#         )
        
#         # Generate and verify summary
#         summary = pipeline.summary()
#         print(summary)
#         assert summary, "Summary generation failed"
        
#         print("\n✓ All two-pass inference tests passed!")

# class TestViscoElasticFilament_TwoPassInference_BendingElasticityBendingViscosity:
#     """ Infer Sp4 and tau_b in a two-pass inference, with 
#         - static experimental data in the first pass,
#         - dynamic experimental data in the second pass.

#     Parameters to be inferred:
#         - Sp4: 1.0
#         - tau_b: 1e6 
#     """

#     # ======================
#     # ==== Ground Truth ====
#     # ======================

#     @pytest.fixture
#     def ground_truth_int_params(self):
#         """
#         Define internal parameters with a known Sp4 ground truth value.
#         All other parameters are fixed for inference.
#         """
#         N = 10
#         X0 = StraightLine(N)
#         return {
#             'Sp4': 1.0,           # Ground truth to recover
#             'N': 10,            
#             'k0': 1e13,            
#             'bool_EI': True,      
#             'Beta':0,        
#             'tau_b': 1e0,           # Ground truth to recover
#             'tau_s': 0,       
#             'gamma': 2,         
#             'n_L': [0,0],            
#             'm_L': 0,             
#             'X_0': X0,  # Initial state
#         }

#     @pytest.fixture
#     def ground_truth_ext_flow_params_static(self):
#         """Define external parameters for static response (fixed during inference)."""
#         N = 10
#         return {
#             "Lambdas": [[0,0]]*N,
#             "Zetas": [0]*N,
#             "A":1e-6,
#             "w0":0, # Static flow
#             "psi":np.pi/2,
#         }


#     @pytest.fixture
#     def ground_truth_sim_params_static(self):
#         """
#         Define simulation parameters.
#         Use BDF method for faster convergence compared to RK45.
#         """
#         return {
#             "T_span": (1e6, 2e6),
#             "T_eval": np.linspace(1e6, 2e6, int(2e0)), # minimum two elements here.
#             "method": "BDF",
#             "T_sim_max": 300,
#         }


#     @pytest.fixture
#     def ground_truth_flow_data_static(
#         self, 
#         ground_truth_int_params, 
#         ground_truth_ext_flow_params_static, 
#         ground_truth_sim_params_static,
#     ):
#         """
#         Generate ground truth synthetic data by simulating ViscoElasticFilament
#         with known parameters.
        
#         Returns:
#             np.ndarray: Simulated trajectory (shape depends on model)
#         """
#         model = ViscoElasticFilament_FlowParams_ScalarBending(
#             int_params=ground_truth_int_params,
#             ext_params=ground_truth_ext_flow_params_static,
#             sim_params=ground_truth_sim_params_static
#         )
#         output = model.simulate_single()
        
#         assert output['value'] is not None, (
#             f"Ground truth simulation failed. Full output: {output}"
#         )

#         return output['value']

#     @pytest.fixture
#     def ground_truth_ext_flow_params_dynamic(self):
#         """Define external parameters for static response (fixed during inference)."""
#         N = 10
#         return {
#             "Lambdas": [[0,0]]*N,
#             "Zetas": [0]*N,
#             "A":1e-6,
#             "w0":1e0, # Dynamic flow
#             "psi":np.pi/2,
#         }        

#     @pytest.fixture
#     def ground_truth_sim_params_dynamic(self):
#         """
#         Define simulation parameters.
#         Use BDF method for faster convergence compared to RK45.
#         """
#         return {
#             "T_span": (1e1, 2e1),
#             "T_eval": np.linspace(1e1, 2e1, int(1e1)),
#             "method": "BDF",
#             "T_sim_max": 300,
#         }

#     @pytest.fixture
#     def ground_truth_flow_data_dynamic(
#         self,
#         ground_truth_int_params, 
#         ground_truth_ext_flow_params_dynamic, 
#         ground_truth_sim_params_dynamic,
#     ):
#         """
#         Generate ground truth synthetic data by simulating ViscoElasticFilament
#         with known parameters.
        
#         Returns:
#             np.ndarray: Simulated trajectory (shape depends on model)
#         """
#         model = ViscoElasticFilament_FlowParams_ScalarBending(
#             int_params=ground_truth_int_params,
#             ext_params=ground_truth_ext_flow_params_dynamic,
#             sim_params=ground_truth_sim_params_dynamic,
#         )
#         output = model.simulate_single()
        
#         assert output['value'] is not None, (
#             f"Ground truth simulation failed. Full output: {output}"
#         )

#         return output['value']
    
#     # ======================
#     # ======= Models =======
#     # ======================

#     @pytest.fixture
#     def elastic_model_flow_sp4_only(
#         self,
#         ground_truth_int_params
#     ):
#         """
#         Create a composed model for ViscoElasticFilament_FlowParams that
#             - only varies Sp4
#             - sets tau_s to zero (elastic model)
        
#         The embedding function accepts a reduced parameter dict {'Sp4': value}
#         and embeds it into the full internal parameters, keeping all others fixed.
#         """
#         fixed_params = ground_truth_int_params.copy()
#         fixed_params['tau_b'] = 0 # Set dynamic parameters to 0.
        
#         def embed_sp4_flow(
#             reduced_int_params: Dict[str, float],
#             ext_params: Any,
#             sim_params: Any,
#         ) -> Dict[str, Any]:
#             """
#             Transform reduced internal parameters into full int_params dict.
            
#             Args:
#                 reduced_int_params: Dict containing {'Sp4': inferred_value}
#                 ext_params: Passed through unchanged (not modified here)
#                 sim_params: Passed through unchanged (not modified here)
            
#             Returns:
#                 Full int_params dict with Sp4 updated, all other values fixed.
#             """
#             full_params = fixed_params.copy()
            
#             # Update only Sp4; all other parameters remain fixed
#             if 'Sp4' in reduced_int_params:
#                 full_params['Sp4'] = reduced_int_params['Sp4']
            
#             return full_params
        
#         # Create composed model with the embedding function
#         ElasticModel = compose_model(
#             ViscoElasticFilament_FlowParams_ScalarBending,
#             compose_int_params=embed_sp4_flow,
#         )
#         return ElasticModel

#     @pytest.fixture
#     def viscoelastic_model_flow_sp4_tau_b_only(
#         self,
#         ground_truth_int_params,
#     ):
#         """
#         Create a composed model for ViscoElasticFilament_FlowParams that only varies Sp4 and tau_b.
        
#         The embedding function accepts a reduced parameter dict {'Sp4': value, 'tau_b': value}
#         and embeds it into the full internal parameters, keeping all others fixed.
#         """
#         fixed_params = ground_truth_int_params.copy()
        
#         def embed_sp4_tau_b_flow(
#             reduced_int_params: Dict[str, float],
#             ext_params: Any,
#             sim_params: Any,
#         ) -> Dict[str, Any]:
#             """
#             Transform reduced internal parameters into full int_params dict.
            
#             Args:
#                 reduced_int_params: Dict containing {'Sp4': inferred_value, 'tau_b': inferred_value}
#                 ext_params: Passed through unchanged (not modified here)
#                 sim_params: Passed through unchanged (not modified here)
            
#             Returns:
#                 Full int_params dict with Sp4 and tau_b updated, all other values fixed.
#             """
#             full_params = fixed_params.copy()
            
#             # Update only Sp4; all other parameters remain fixed
#             if 'Sp4' in reduced_int_params:
#                 full_params['Sp4'] = reduced_int_params['Sp4']

#             if 'tau_b' in reduced_int_params:
#                 full_params['tau_b'] = reduced_int_params['tau_b']
            
#             return full_params
        
#         # Create composed model with the embedding function
#         ViscoElasticModel = compose_model(
#             ViscoElasticFilament_FlowParams_ScalarBending,
#             compose_int_params=embed_sp4_tau_b_flow,
#         )
#         return ViscoElasticModel

#     # ==========================
#     # ======= Optimizers =======
#     # ==========================

#     @pytest.fixture
#     def basinhopping_optimizer_instance(self):
#         """
#         Return the basinhopping optimizer function with standard configuration.
#         """
#         return basinhopping_optimizer        

#     @pytest.fixture
#     def optimizer_kwargs_sp4(self):
#         return {
#             'bounds': Bounds(lb=[1e-6], ub=[np.inf]),
#             'minimum_gradient': False,
#             'minimum_hessian': False,
#             'local_minimizer_kwargs': {
#                 'method': 'L-BFGS-B',
#                 'jac': '3-point',
#                 'options': {
#                     'disp': True,
#                     'ftol': 1e-8,
#                     'gtol': 1e-8,
#                     'eps': 1e-8,
#                     'finite_diff_rel_step': 1e-6,
#                 },
#             },
#             'global_minimizer_kwargs': {
#                 'niter': 9,
#                 'T': 0,
#                 'stepsize': 5,
#                 'tol': 1e-10,
#             }
#         }

#     @pytest.fixture
#     def optimizer_kwargs_tau_b(self, optimizer_kwargs_sp4):
#         optimizer_kwargs = optimizer_kwargs_sp4
#         optimizer_kwargs['bounds'] = Bounds(lb=[0], ub=[np.inf])
#         return optimizer_kwargs

#     # ==========================
#     # ========= Passes =========
#     # ==========================

#     # Pass 1: Reduced model, infer Sp4 only
#     @pytest.fixture
#     def pass_1(
#         self,
#         elastic_model_flow_sp4_only,
#         ground_truth_ext_flow_params_static,
#         ground_truth_sim_params_static,
#         ground_truth_flow_data_static,
#         basinhopping_optimizer_instance,
#         optimizer_kwargs_sp4,
#     ):
#         """ First pass for the viscoelastic inference. """
        
#         first_pass = PipelinePass(
#             name="Sp4 Inference (Elastic Model)",
#             model_class=elastic_model_flow_sp4_only,
#             ground_truths=[ground_truth_flow_data_static],
#             ext_params_list=[ground_truth_ext_flow_params_static],
#             sim_params_list=[ground_truth_sim_params_static],
#             param_keys_to_infer=['Sp4'],
#             fixed_params={},
#             optimizer=basinhopping_optimizer_instance,
#             optimizer_kwargs=optimizer_kwargs_sp4,
#         )

#         return first_pass

#     # Pass 2:
#     @pytest.fixture
#     def pass_2(
#         self,
#         viscoelastic_model_flow_sp4_tau_b_only,
#         ground_truth_ext_flow_params_dynamic,
#         ground_truth_sim_params_dynamic,
#         ground_truth_flow_data_dynamic,
#         basinhopping_optimizer_instance,
#         optimizer_kwargs_tau_b,
#     ):
#         """ Second pass for the viscoelastic inference. """
        
#         second_pass = PipelinePass(
#             name="tau_b Inference (ViscoElastic Model)",
#             model_class=viscoelastic_model_flow_sp4_tau_b_only,
#             ground_truths=[ground_truth_flow_data_dynamic],
#             ext_params_list=[ground_truth_ext_flow_params_dynamic],
#             sim_params_list=[ground_truth_sim_params_dynamic],
#             param_keys_to_infer=['tau_b'],
#             fixed_params={},
#             optimizer=basinhopping_optimizer_instance,
#             optimizer_kwargs=optimizer_kwargs_tau_b,
#         )

#         return second_pass

#     # ============================
#     # ========= Pipeline =========
#     # ============================
#     @pytest.fixture
#     def viscoelastic_pipeline(self, pass_1, pass_2, mse_loss_fn):

#         pipeline = InferencePipeline(
#             passes=[pass_1, pass_2],
#             loss_fn=mse_loss_fn,
#             n_jobs_per_pass=-1,  # Use all cores within each pass
#         )
#         return pipeline

#     # ============================
#     # ========= Tests =========
#     # ============================
#     def test_viscoelastic_inference(
#         self,
#         viscoelastic_pipeline,
#     ):

#         # Multiple initial guesses: Pass 1 for Sp4, Pass 2 for tau_b
#         initial_guesses_per_pass = [
#             [
#                 # {'Sp4': 0.1},
#                 {'Sp4': 10.0},
#             ],
#             [
#                 {'tau_b': 0},
#                 # {'tau_b': 10.0},
#             ],
#         ]
        
#         # ===== ACT =====
#         results = viscoelastic_pipeline.run(initial_guesses_per_pass, verbose=True)
        
#         # ===== ASSERT =====
        
#         # Check that we got exactly two results (two passes)
#         assert len(results) == 2, "Expected two InferenceResult for two-pass pipeline"
        
#         result_pass1, result_pass2 = results
        
#         # ===== PASS 1: Sp4 INFERENCE =====
#         print("\n" + "="*60)
#         print("PASS 1 ASSERTIONS (Sp4 inference)")
#         print("="*60)
        
#         # 1. Check Pass 1 convergence
#         assert result_pass1.success, (
#             f"Pass 1 optimization did not converge. Message: {result_pass1.message}"
#         )
#         assert result_pass1.iterations > 0, "Pass 1: No iterations were performed"
        
#         # 2. Check Sp4 exists and is physically reasonable
#         assert 'Sp4' in result_pass1.params, "Sp4 not in Pass 1 inferred parameters"
#         sp4_inferred = result_pass1.params['Sp4']
#         assert sp4_inferred > 0, f"Sp4 must be positive, got {sp4_inferred}"
#         assert sp4_inferred < 1e6, f"Sp4 unreasonably large: {sp4_inferred}"
        
#         # 3. Check Pass 1 loss
#         assert result_pass1.loss > 0, "Pass 1 loss should be positive"
#         assert result_pass1.loss < 1e2, (
#             f"Pass 1 loss suspiciously high: {result_pass1.loss}. Check model/data scale."
#         )
        
#         # 4. Check Pass 1 uncertainty quantification
#         assert result_pass1.covariance is not None, "Pass 1: Covariance not computed"
#         assert result_pass1.hessian is not None, "Pass 1: Hessian not computed"
#         assert result_pass1.std_errors is not None, "Pass 1: Standard errors not computed"
#         assert len(result_pass1.std_errors) == 1, "Pass 1: Expected 1 std_error for 1 parameter (Sp4)"
        
#         std_err_sp4 = result_pass1.std_errors[0]
#         assert std_err_sp4 > 0, f"Pass 1: Sp4 std error must be positive, got {std_err_sp4}"
#         assert std_err_sp4 < sp4_inferred * 10, (
#             f"Pass 1: Std error unreasonably large relative to Sp4: "
#             f"Sp4={sp4_inferred:.3e} ± {std_err_sp4:.3e}"
#         )
        
#         print(f"✓ Pass 1 - Sp4 inferred: {sp4_inferred:.6e} ± {std_err_sp4:.6e}")
#         print(f"✓ Pass 1 - Final loss: {result_pass1.loss:.8e}")
#         print(f"✓ Pass 1 - Iterations: {result_pass1.iterations}")
        
#         # ===== PASS 2: tau_b INFERENCE (Sp4 FIXED) =====
#         print("\n" + "="*60)
#         print("PASS 2 ASSERTIONS (tau_b inference with Sp4 fixed)")
#         print("="*60)
        
#         # 1. Check Pass 2 convergence
#         assert result_pass2.success, (
#             f"Pass 2 optimization did not converge. Message: {result_pass2.message}"
#         )
#         assert result_pass2.iterations > 0, "Pass 2: No iterations were performed"
        
#         # 2. Check tau_s exists and is physically reasonable
#         assert 'tau_b' in result_pass2.params, "tau_b not in Pass 2 inferred parameters"
#         tau_b_inferred = result_pass2.params['tau_b']
#         assert tau_b_inferred >= 0, f"tau_b must be non-negative, got {tau_b_inferred}"
#         assert tau_b_inferred < 1e4, f"tau_s unreasonably large: {tau_b_inferred}"
        
#         # 3. Check Pass 2 loss
#         assert result_pass2.loss > 0, "Pass 2 loss should be positive"
#         assert result_pass2.loss < 1e2, (
#             f"Pass 2 loss suspiciously high: {result_pass2.loss}. Check model/data scale."
#         )

#         # 4. Check Pass 2 uncertainty quantification (only for tau_s, not Sp4)
#         assert result_pass2.covariance is not None, "Pass 2: Covariance not computed"
#         assert result_pass2.hessian is not None, "Pass 2: Hessian not computed"
#         assert result_pass2.std_errors is not None, "Pass 2: Standard errors not computed"
#         assert len(result_pass2.std_errors) == 1, "Pass 2: Expected 1 std_error for 1 inferred parameter (tau_s)"
        
#         std_err_tau_b = result_pass2.std_errors[0]
#         assert std_err_tau_b > 0, f"Pass 2: tau_b std error must be positive, got {std_err_tau_b}"
#         assert std_err_tau_b < tau_b_inferred * 10, (
#             f"Pass 2: Std error unreasonably large relative to tau_s: "
#             f"tau_s={tau_b_inferred:.3e} ± {std_err_tau_b:.3e}"
#         )
        
#         print(f"✓ Pass 2 - tau_b inferred: {tau_b_inferred:.6e} ± {std_err_tau_b:.6e}")
#         print(f"✓ Pass 2 - Sp4 fixed (from Pass 1): {result_pass1.params['Sp4']:.6e}")
#         print(f"✓ Pass 2 - Final loss: {result_pass2.loss:.8e}")
#         print(f"✓ Pass 2 - Iterations: {result_pass2.iterations}")
        
#         # ===== PIPELINE-LEVEL ASSERTIONS =====
#         print("\n" + "="*60)
#         print("PIPELINE-LEVEL ASSERTIONS")
#         print("="*60)
        
#         # Check parameter trajectory
#         trajectory = viscoelastic_pipeline.get_parameter_trajectory()
#         assert 'Sp4' in trajectory, "Sp4 not in trajectory"
#         assert 'tau_b' in trajectory, "tau_b not in trajectory"
#         assert len(trajectory['Sp4']) == 2, "Sp4 trajectory should have 2 steps"
#         assert len(trajectory['tau_b']) == 2, "tau_b trajectory should have 2 steps"
        
#         # Sp4 should stay constant (inferred in Pass 1, fixed in Pass 2)
#         assert trajectory['Sp4'][0] == trajectory['Sp4'][1], (
#             f"Sp4 should be constant across passes. "
#             f"Pass1={trajectory['Sp4'][0]}, Pass2={trajectory['Sp4'][1]}"
#         )
        
#         # tau_s should appear in Pass 2 (None or absent in Pass 1)
#         assert trajectory['tau_b'][1] == tau_b_inferred, (
#             f"tau_b trajectory doesn't match Pass 2 result"
#         )
        
#         # Generate and verify summary
#         summary = viscoelastic_pipeline.summary()
#         print(summary)
#         assert summary, "Summary generation failed"
        
#         print("\n✓ All two-pass inference tests passed!")

# class TestViscoElasticFilament_TwoPassInference_BendingShearElasticityBendingViscosity:
#     """ Infer Sp4 and tau_b in a two-pass inference, with 
#         - static experimental data in the first pass,
#         - dynamic experimental data in the second pass.

#     Parameters to be inferred:
#         - Sp4
#         - Beta
#         - tau_b
#     """

#     # ======================
#     # ==== Ground Truth ====
#     # ======================

#     @pytest.fixture
#     def ground_truth_int_params(self):
#         """
#         Define internal parameters with known Sp4, Beta, tau_b ground truth values.
#         All other parameters are fixed for inference.
#         """
#         N = 10
#         X0 = StraightLine(N)
#         return {
#             'Sp4': 1e0,           # Ground truth to recover
#             'N': 10,            
#             'k0': 1e13,            
#             'bool_EI': True,      
#             'Beta': 1e0,              # Ground truth to recover
#             'tau_b': 1e0,           # Ground truth to recover
#             'tau_s': 0,       
#             'gamma': 2,         
#             'n_L': [0,0],            
#             'm_L': 0,             
#             'X_0': X0,  # Initial state
#         }

#     @pytest.fixture
#     def ground_truth_ext_flow_params_static(self):
#         """Define external parameters for static response (fixed during inference)."""
#         N = 10
#         return {
#             "Lambdas": [[0,0]]*N,
#             "Zetas": [0]*N,
#             "A":1e-6,
#             "w0":0, # Static flow
#             "psi":np.pi/2,
#         }  

#     @pytest.fixture
#     def ground_truth_sim_params_static(self):
#         """
#         Define simulation parameters.
#         Use BDF method for faster convergence compared to RK45.
#         """
#         return {
#             "T_span": (1e6, 2e6),
#             "T_eval": np.linspace(1e6, 2e6, int(2e0)), # minimum two elements here.
#             "method": "BDF",
#             "T_sim_max": 300,
#         }

#     @pytest.fixture
#     def ground_truth_flow_data_static(
#         self, 
#         ground_truth_int_params, 
#         ground_truth_ext_flow_params_static, 
#         ground_truth_sim_params_static,
#     ):
#         """
#         Generate ground truth synthetic data by simulating ViscoElasticFilament
#         with known parameters.
        
#         Returns:
#             np.ndarray: Simulated trajectory (shape depends on model)
#         """
#         model = ViscoElasticFilament_FlowParams_ScalarBending(
#             int_params=ground_truth_int_params,
#             ext_params=ground_truth_ext_flow_params_static,
#             sim_params=ground_truth_sim_params_static
#         )
#         output = model.simulate_single()
        
#         assert output['value'] is not None, (
#             f"Ground truth simulation failed. Full output: {output}"
#         )

#         return output['value']

#     @pytest.fixture
#     def ground_truth_ext_flow_params_dynamic(self):
#         """Define external parameters for static response (fixed during inference)."""
#         N = 10
#         return {
#             "Lambdas": [[0,0]]*N,
#             "Zetas": [0]*N,
#             "A":1e-6,
#             "w0":1e0, # Dynamic flow
#             "psi":np.pi/2,
#         }        

#     @pytest.fixture
#     def ground_truth_sim_params_dynamic(self):
#         """
#         Define simulation parameters.
#         Use BDF method for faster convergence compared to RK45.
#         """
#         return {
#             "T_span": (1e1, 2e1),
#             "T_eval": np.linspace(1e1, 2e1, int(1e1)),
#             "method": "BDF",
#             "T_sim_max": 300,
#         }

#     @pytest.fixture
#     def ground_truth_flow_data_dynamic(
#         self,
#         ground_truth_int_params, 
#         ground_truth_ext_flow_params_dynamic, 
#         ground_truth_sim_params_dynamic,
#     ):
#         """
#         Generate ground truth synthetic data by simulating ViscoElasticFilament
#         with known parameters.
        
#         Returns:
#             np.ndarray: Simulated trajectory (shape depends on model)
#         """
#         model = ViscoElasticFilament_FlowParams_ScalarBending(
#             int_params=ground_truth_int_params,
#             ext_params=ground_truth_ext_flow_params_dynamic,
#             sim_params=ground_truth_sim_params_dynamic,
#         )
#         output = model.simulate_single()
        
#         assert output['value'] is not None, (
#             f"Ground truth simulation failed. Full output: {output}"
#         )

#         return output['value']
    
#     # ======================
#     # ======= Models =======
#     # ======================

#     @pytest.fixture
#     def multielastic_model_flow_sp4_beta_only(
#         self,
#         ground_truth_int_params
#     ):
#         """
#         Create a composed model for ViscoElasticFilament_FlowParams that
#             - only varies Sp4 and Beta
#             - sets tau_s to zero (elastic model)
        
#         The embedding function accepts a reduced parameter dict {'Sp4': value, 'Beta':value}
#         and embeds it into the full internal parameters, keeping all others fixed.
#         """
#         fixed_params = ground_truth_int_params.copy()
#         fixed_params['tau_b'] = 0 # Set dynamic parameters to 0.
        
#         def embed_sp4_beta_flow(
#             reduced_int_params: Dict[str, float],
#             ext_params: Any,
#             sim_params: Any,
#         ) -> Dict[str, Any]:
#             """
#             Transform reduced internal parameters into full int_params dict.
            
#             Args:
#                 reduced_int_params: Dict containing {'Sp4': inferred_value, 'Beta':inferred_value}
#                 ext_params: Passed through unchanged (not modified here)
#                 sim_params: Passed through unchanged (not modified here)
            
#             Returns:
#                 Full int_params dict with Sp4 and Beta updated, all other values fixed.
#             """
#             full_params = fixed_params.copy()
            
#             # Update only Sp4 and Beta; all other parameters remain fixed
#             if 'Sp4' in reduced_int_params:
#                 full_params['Sp4'] = reduced_int_params['Sp4']
#             if 'Beta' in reduced_int_params:
#                 full_params['Beta'] = reduced_int_params['Beta']
            
#             return full_params
        
#         # Create composed model with the embedding function
#         MultiElasticModel = compose_model(
#             ViscoElasticFilament_FlowParams_ScalarBending,
#             compose_int_params=embed_sp4_beta_flow,
#         )
#         return MultiElasticModel

#     @pytest.fixture
#     def viscoelastic_model_flow_sp4_beta_tau_b_only(
#         self,
#         ground_truth_int_params,
#     ):
#         """
#         Create a composed model for ViscoElasticFilament_FlowParams that only varies Sp4, Beta and tau_b.
        
#         The embedding function accepts a reduced parameter dict {'Sp4': value, 'Beta': value, 'tau_b': value}
#         and embeds it into the full internal parameters, keeping all others fixed.
#         """
#         fixed_params = ground_truth_int_params.copy()
        
#         def embed_sp4_beta_tau_b_flow(
#             reduced_int_params: Dict[str, float],
#             ext_params: Any,
#             sim_params: Any,
#         ) -> Dict[str, Any]:
#             """
#             Transform reduced internal parameters into full int_params dict.
            
#             Args:
#                 reduced_int_params: Dict containing {'Sp4': inferred_value, 'Beta': inferred_value, 'tau_b': inferred_value}
#                 ext_params: Passed through unchanged (not modified here)
#                 sim_params: Passed through unchanged (not modified here)
            
#             Returns:
#                 Full int_params dict with Sp4 and tau_b updated, all other values fixed.
#             """
#             full_params = fixed_params.copy()
            
#             # Update only Sp4, Beta and tau_b; all other parameters remain fixed
#             if 'Sp4' in reduced_int_params:
#                 full_params['Sp4'] = reduced_int_params['Sp4']

#             if 'Beta' in reduced_int_params:
#                 full_params['Beta'] = reduced_int_params['Beta']

#             if 'tau_b' in reduced_int_params:
#                 full_params['tau_b'] = reduced_int_params['tau_b']
            
#             return full_params
        
#         # Create composed model with the embedding function
#         ViscoElasticModel = compose_model(
#             ViscoElasticFilament_FlowParams_ScalarBending,
#             compose_int_params=embed_sp4_beta_tau_b_flow,
#         )
#         return ViscoElasticModel

#     # ==========================
#     # ======= Optimizers =======
#     # ==========================

#     @pytest.fixture
#     def basinhopping_optimizer_instance(self):
#         """
#         Return the basinhopping optimizer function with standard configuration.
#         """
#         return basinhopping_optimizer        

#     @pytest.fixture
#     def optimizer_kwargs_sp4_beta(self):
#         return {
#             'bounds': Bounds(lb=[1e-6, 0], ub=[np.inf, np.inf]),
#             'minimum_gradient': False,
#             'minimum_hessian': False,
#             'local_minimizer_kwargs': {
#                 'method': 'L-BFGS-B',
#                 'jac': '3-point',
#                 'options': {
#                     'disp': True,
#                     'ftol': 1e-8,
#                     'gtol': 1e-8,
#                     'eps': 1e-8,
#                     'finite_diff_rel_step': 1e-6,
#                 },
#             },
#             'global_minimizer_kwargs': {
#                 'niter': 9,
#                 'T': 0,
#                 'stepsize': 5,
#                 'tol': 1e-10,
#             }
#         }

#     @pytest.fixture
#     def optimizer_kwargs_tau_b(self, optimizer_kwargs_sp4_beta):
#         optimizer_kwargs = optimizer_kwargs_sp4_beta
#         optimizer_kwargs['bounds'] = Bounds(lb=[0], ub=[np.inf])
#         return optimizer_kwargs

#     # ==========================
#     # ========= Passes =========
#     # ==========================

#     # Pass 1: Reduced model, infer Sp4 and Beta only
#     @pytest.fixture
#     def pass_1(
#         self,
#         multielastic_model_flow_sp4_beta_only,
#         ground_truth_ext_flow_params_static,
#         ground_truth_sim_params_static,
#         ground_truth_flow_data_static,
#         basinhopping_optimizer_instance,
#         optimizer_kwargs_sp4_beta,
#     ):
#         """ First pass for the viscoelastic inference. """
        
#         first_pass = PipelinePass(
#             name="Sp4-Beta Inference (MultiElastic Model)",
#             model_class=multielastic_model_flow_sp4_beta_only,
#             ground_truths=[ground_truth_flow_data_static],
#             ext_params_list=[ground_truth_ext_flow_params_static],
#             sim_params_list=[ground_truth_sim_params_static],
#             param_keys_to_infer=['Sp4', 'Beta'],
#             fixed_params={},
#             optimizer=basinhopping_optimizer_instance,
#             optimizer_kwargs=optimizer_kwargs_sp4_beta,
#         )

#         return first_pass

#     # Pass 2:
#     @pytest.fixture
#     def pass_2(
#         self,
#         viscoelastic_model_flow_sp4_beta_tau_b_only,
#         ground_truth_ext_flow_params_dynamic,
#         ground_truth_sim_params_dynamic,
#         ground_truth_flow_data_dynamic,
#         basinhopping_optimizer_instance,
#         optimizer_kwargs_tau_b,
#     ):
#         """ Second pass for the viscoelastic inference. """
        
#         second_pass = PipelinePass(
#             name="tau_b Inference (ViscoElastic Model)",
#             model_class=viscoelastic_model_flow_sp4_beta_tau_b_only,
#             ground_truths=[ground_truth_flow_data_dynamic],
#             ext_params_list=[ground_truth_ext_flow_params_dynamic],
#             sim_params_list=[ground_truth_sim_params_dynamic],
#             param_keys_to_infer=['tau_b'],
#             fixed_params={},
#             optimizer=basinhopping_optimizer_instance,
#             optimizer_kwargs=optimizer_kwargs_tau_b,
#         )

#         return second_pass

#     # ============================
#     # ========= Pipeline =========
#     # ============================
#     @pytest.fixture
#     def viscoelastic_pipeline(self, pass_1, pass_2, mse_loss_fn):

#         pipeline = InferencePipeline(
#             passes=[pass_1, pass_2],
#             loss_fn=mse_loss_fn,
#             n_jobs_per_pass=-1,  # Use all cores within each pass
#         )
#         return pipeline

#     # ============================
#     # ========= Tests =========
#     # ============================
#     def test_viscoelastic_inference(
#         self,
#         viscoelastic_pipeline,
#     ):

#         # Multiple initial guesses: Pass 1 for Sp4-Beta, Pass 2 for tau_b
#         initial_guesses_per_pass = [
#             [
#                 # {'Sp4': 1e0, 'Beta':0},
#                 # {'Sp4': 1e-1, 'Beta':1e0},
#                 {'Sp4': 10, 'Beta':0},
#             ],
#             [
#                 {'tau_b': 0},
#                 # {'tau_b': 10.0},
#             ],
#         ]
        
#         # ===== ACT =====
#         results = viscoelastic_pipeline.run(initial_guesses_per_pass, verbose=True)
        
#         # ===== ASSERT =====
        
#         # Check that we got exactly two results (two passes)
#         assert len(results) == 2, "Expected two InferenceResult for two-pass pipeline"
        
#         result_pass1, result_pass2 = results
        
#         # ===== PASS 1: Sp4 INFERENCE =====
#         print("\n" + "="*60)
#         print("PASS 1 ASSERTIONS (Sp4 inference)")
#         print("="*60)
        
#         # 1. Check Pass 1 convergence
#         assert result_pass1.success, (
#             f"Pass 1 optimization did not converge. Message: {result_pass1.message}"
#         )
#         assert result_pass1.iterations > 0, "Pass 1: No iterations were performed"
        
#         # 2. Check Sp4 and Beta exist and are physically reasonable

#         assert 'Sp4' in result_pass1.params, "Sp4 not in Pass 1 inferred parameters"
#         sp4_inferred = result_pass1.params['Sp4']
#         assert sp4_inferred > 0, f"Sp4 must be positive, got {sp4_inferred}"
#         assert sp4_inferred < 1e6, f"Sp4 unreasonably large: {sp4_inferred}"

#         assert 'Beta' in result_pass1.params, "Beta not in Pass 1 inferred parameters"
#         beta_inferred = result_pass1.params['Beta']
#         assert beta_inferred >= 0, f"Beta must be non-negative, got {beta_inferred}"
#         assert beta_inferred < 1e6, f"Beta unreasonably large: {beta_inferred}"      

#         # 3. Check Pass 1 loss
#         assert result_pass1.loss > 0, "Pass 1 loss should be positive"
#         assert result_pass1.loss < 1e2, (
#             f"Pass 1 loss suspiciously high: {result_pass1.loss}. Check model/data scale."
#         )
        
#         # 4. Check Pass 1 uncertainty quantification
#         assert result_pass1.covariance is not None, "Pass 1: Covariance not computed"
#         assert result_pass1.hessian is not None, "Pass 1: Hessian not computed"
#         assert result_pass1.std_errors is not None, "Pass 1: Standard errors not computed"
#         assert len(result_pass1.std_errors) == 2, "Pass 1: Expected 1 std_error for 2 parameter (Sp4, Beta)"
        
#         std_err_sp4 = result_pass1.std_errors[0]
#         assert std_err_sp4 > 0, f"Pass 1: Sp4 std error must be positive, got {std_err_sp4}"
#         assert std_err_sp4 < sp4_inferred * 10, (
#             f"Pass 1: Std error unreasonably large relative to Sp4: "
#             f"Sp4={sp4_inferred:.3e} ± {std_err_sp4:.3e}"
#         )
#         std_err_beta = result_pass1.std_errors[1]
#         assert std_err_beta > 0, f"Pass 1: Beta std error must be positive, got {std_err_beta}"
#         assert std_err_beta < beta_inferred * 10, (
#             f"Pass 1: Std error unreasonably large relative to Beta: "
#             f"Beta={beta_inferred:.3e} ± {std_err_beta:.3e}"
#         )
        
#         print(f"✓ Pass 1 - Sp4 inferred: {sp4_inferred:.6e} ± {std_err_sp4:.6e}")
#         print(f"✓ Pass 1 - Beta inferred: {beta_inferred:.6e} ± {std_err_beta:.6e}")
#         print(f"✓ Pass 1 - Final loss: {result_pass1.loss:.8e}")
#         print(f"✓ Pass 1 - Iterations: {result_pass1.iterations}")
        
#         # ===== PASS 2: tau_b INFERENCE (Sp4 and Beta FIXED) =====
#         print("\n" + "="*60)
#         print("PASS 2 ASSERTIONS (tau_b inference with Sp4 and Beta fixed)")
#         print("="*60)
        
#         # 1. Check Pass 2 convergence
#         assert result_pass2.success, (
#             f"Pass 2 optimization did not converge. Message: {result_pass2.message}"
#         )
#         assert result_pass2.iterations > 0, "Pass 2: No iterations were performed"
        
#         # 2. Check tau_s exists and is physically reasonable
#         assert 'tau_b' in result_pass2.params, "tau_b not in Pass 2 inferred parameters"
#         tau_b_inferred = result_pass2.params['tau_b']
#         assert tau_b_inferred >= 0, f"tau_b must be non-negative, got {tau_b_inferred}"
#         assert tau_b_inferred < 1e4, f"tau_s unreasonably large: {tau_b_inferred}"
        
#         # 3. Check Pass 2 loss
#         assert result_pass2.loss > 0, "Pass 2 loss should be positive"
#         assert result_pass2.loss < 1e2, (
#             f"Pass 2 loss suspiciously high: {result_pass2.loss}. Check model/data scale."
#         )

#         # 4. Check Pass 2 uncertainty quantification (only for tau_s, not Sp4)
#         assert result_pass2.covariance is not None, "Pass 2: Covariance not computed"
#         assert result_pass2.hessian is not None, "Pass 2: Hessian not computed"
#         assert result_pass2.std_errors is not None, "Pass 2: Standard errors not computed"
#         assert len(result_pass2.std_errors) == 1, "Pass 2: Expected 1 std_error for 1 inferred parameter (tau_s)"
        
#         std_err_tau_b = result_pass2.std_errors[0]
#         assert std_err_tau_b > 0, f"Pass 2: tau_b std error must be positive, got {std_err_tau_b}"
#         assert std_err_tau_b < tau_b_inferred * 10, (
#             f"Pass 2: Std error unreasonably large relative to tau_s: "
#             f"tau_s={tau_b_inferred:.3e} ± {std_err_tau_b:.3e}"
#         )
        
#         print(f"✓ Pass 2 - tau_b inferred: {tau_b_inferred:.6e} ± {std_err_tau_b:.6e}")
#         print(f"✓ Pass 2 - Sp4 fixed (from Pass 1): {result_pass1.params['Sp4']:.6e}")
#         print(f"✓ Pass 2 - Beta fixed (from Pass 1): {result_pass1.params['Beta']:.6e}")
#         print(f"✓ Pass 2 - Final loss: {result_pass2.loss:.8e}")
#         print(f"✓ Pass 2 - Iterations: {result_pass2.iterations}")
        
#         # ===== PIPELINE-LEVEL ASSERTIONS =====
#         print("\n" + "="*60)
#         print("PIPELINE-LEVEL ASSERTIONS")
#         print("="*60)
        
#         # Check parameter trajectory
#         trajectory = viscoelastic_pipeline.get_parameter_trajectory()
#         assert 'Sp4' in trajectory, "Sp4 not in trajectory"
#         assert 'Beta' in trajectory, "Beta not in trajectory"
#         assert 'tau_b' in trajectory, "tau_b not in trajectory"
#         assert len(trajectory['Sp4']) == 2, "Sp4 trajectory should have 2 steps"
#         assert len(trajectory['Beta']) == 2, "Beta trajectory should have 2 steps"
#         assert len(trajectory['tau_b']) == 2, "tau_b trajectory should have 2 steps"
        
#         # Sp4 should stay constant (inferred in Pass 1, fixed in Pass 2)
#         assert trajectory['Sp4'][0] == trajectory['Sp4'][1], (
#             f"Sp4 should be constant across passes. "
#             f"Pass1={trajectory['Sp4'][0]}, Pass2={trajectory['Sp4'][1]}"
#         )

#         # Beta should stay constant (inferred in Pass 1, fixed in Pass 2)
#         assert trajectory['Beta'][0] == trajectory['Beta'][1], (
#             f"Beta should be constant across passes. "
#             f"Pass1={trajectory['Beta'][0]}, Pass2={trajectory['Beta'][1]}"
#         )
        
#         # tau_s should appear in Pass 2 (None or absent in Pass 1)
#         assert trajectory['tau_b'][1] == tau_b_inferred, (
#             f"tau_b trajectory doesn't match Pass 2 result"
#         )
        
#         # Generate and verify summary
#         summary = viscoelastic_pipeline.summary()
#         print(summary)
#         assert summary, "Summary generation failed"
        
#         print("\n✓ All two-pass inference tests passed!")

# class TestViscoElasticFilament_TwoPassInference_BendingShearElasticityShearViscosity:
#     """ Infer Sp4, Beta and tau_s in a two-pass inference, with 
#         - static experimental data in the first pass,
#         - dynamic experimental data in the second pass.

#     Parameters to be inferred:
#         - Sp4
#         - Beta
#         - tau_s
#     """

#     # ======================
#     # ==== Ground Truth ====
#     # ======================

#     @pytest.fixture
#     def ground_truth_int_params(self):
#         """
#         Define internal parameters with known Sp4, Beta, tau_b ground truth values.
#         All other parameters are fixed for inference.
#         """
#         N = 10
#         X0 = StraightLine(N)
#         return {
#             'Sp4': 1e0,           # Ground truth to recover
#             'N': 10,            
#             'k0': 1e13,            
#             'bool_EI': True,      
#             'Beta': 1e0,              # Ground truth to recover
#             'tau_b': 0,           
#             'tau_s': 1e0,       # Ground truth to recover
#             'gamma': 2,         
#             'n_L': [0,0],            
#             'm_L': 0,             
#             'X_0': X0,  # Initial state
#         }

#     @pytest.fixture
#     def ground_truth_ext_flow_params_static(self):
#         """Define external parameters for static response (fixed during inference)."""
#         N = 10
#         return {
#             "Lambdas": [[0,0]]*N,
#             "Zetas": [0]*N,
#             "A":1e-6,
#             "w0":0, # Static flow
#             "psi":np.pi/2,
#         }  

#     @pytest.fixture
#     def ground_truth_sim_params_static(self):
#         """
#         Define simulation parameters.
#         Use BDF method for faster convergence compared to RK45.
#         """
#         return {
#             "T_span": (1e6, 2e6),
#             "T_eval": np.linspace(1e6, 2e6, int(2e0)), # minimum two elements here.
#             "method": "BDF",
#             "T_sim_max": 300,
#         }

#     @pytest.fixture
#     def ground_truth_flow_data_static(
#         self, 
#         ground_truth_int_params, 
#         ground_truth_ext_flow_params_static, 
#         ground_truth_sim_params_static,
#     ):
#         """
#         Generate ground truth synthetic data by simulating ViscoElasticFilament
#         with known parameters.
        
#         Returns:
#             np.ndarray: Simulated trajectory (shape depends on model)
#         """
#         model = ViscoElasticFilament_FlowParams_ScalarBending(
#             int_params=ground_truth_int_params,
#             ext_params=ground_truth_ext_flow_params_static,
#             sim_params=ground_truth_sim_params_static
#         )
#         output = model.simulate_single()
        
#         assert output['value'] is not None, (
#             f"Ground truth simulation failed. Full output: {output}"
#         )

#         return output['value']

#     @pytest.fixture
#     def ground_truth_ext_flow_params_dynamic(self):
#         """Define external parameters for static response (fixed during inference)."""
#         N = 10
#         return {
#             "Lambdas": [[0,0]]*N,
#             "Zetas": [0]*N,
#             "A":1e-6,
#             "w0":1e0, # Dynamic flow
#             "psi":np.pi/2,
#         }        

#     @pytest.fixture
#     def ground_truth_sim_params_dynamic(self):
#         """
#         Define simulation parameters.
#         Use BDF method for faster convergence compared to RK45.
#         """
#         return {
#             "T_span": (1e1, 2e1),
#             "T_eval": np.linspace(1e1, 2e1, int(1e1)),
#             "method": "BDF",
#             "T_sim_max": 300,
#         }

#     @pytest.fixture
#     def ground_truth_flow_data_dynamic(
#         self,
#         ground_truth_int_params, 
#         ground_truth_ext_flow_params_dynamic, 
#         ground_truth_sim_params_dynamic,
#     ):
#         """
#         Generate ground truth synthetic data by simulating ViscoElasticFilament
#         with known parameters.
        
#         Returns:
#             np.ndarray: Simulated trajectory (shape depends on model)
#         """
#         model = ViscoElasticFilament_FlowParams_ScalarBending(
#             int_params=ground_truth_int_params,
#             ext_params=ground_truth_ext_flow_params_dynamic,
#             sim_params=ground_truth_sim_params_dynamic,
#         )
#         output = model.simulate_single()
        
#         assert output['value'] is not None, (
#             f"Ground truth simulation failed. Full output: {output}"
#         )

#         return output['value']
    
#     # ======================
#     # ======= Models =======
#     # ======================

#     @pytest.fixture
#     def multielastic_model_flow_sp4_beta_only(
#         self,
#         ground_truth_int_params
#     ):
#         """
#         Create a composed model for ViscoElasticFilament_FlowParams that
#             - only varies Sp4 and Beta
#             - sets tau_s to zero (elastic model)
        
#         The embedding function accepts a reduced parameter dict {'Sp4': value, 'Beta':value}
#         and embeds it into the full internal parameters, keeping all others fixed.
#         """
#         fixed_params = ground_truth_int_params.copy()
#         fixed_params['tau_s'] = 0 # Set dynamic parameters to 0.
        
#         def embed_sp4_beta_flow(
#             reduced_int_params: Dict[str, float],
#             ext_params: Any,
#             sim_params: Any,
#         ) -> Dict[str, Any]:
#             """
#             Transform reduced internal parameters into full int_params dict.
            
#             Args:
#                 reduced_int_params: Dict containing {'Sp4': inferred_value, 'Beta':inferred_value}
#                 ext_params: Passed through unchanged (not modified here)
#                 sim_params: Passed through unchanged (not modified here)
            
#             Returns:
#                 Full int_params dict with Sp4 and Beta updated, all other values fixed.
#             """
#             full_params = fixed_params.copy()
            
#             # Update only Sp4 and Beta; all other parameters remain fixed
#             if 'Sp4' in reduced_int_params:
#                 full_params['Sp4'] = reduced_int_params['Sp4']
#             if 'Beta' in reduced_int_params:
#                 full_params['Beta'] = reduced_int_params['Beta']
            
#             return full_params
        
#         # Create composed model with the embedding function
#         MultiElasticModel = compose_model(
#             ViscoElasticFilament_FlowParams_ScalarBending,
#             compose_int_params=embed_sp4_beta_flow,
#         )
#         return MultiElasticModel

#     @pytest.fixture
#     def viscoelastic_model_flow_sp4_beta_tau_s_only(
#         self,
#         ground_truth_int_params,
#     ):
#         """
#         Create a composed model for ViscoElasticFilament_FlowParams that only varies Sp4, Beta and tau_s.
        
#         The embedding function accepts a reduced parameter dict {'Sp4': value, 'Beta': value, 'tau_s': value}
#         and embeds it into the full internal parameters, keeping all others fixed.
#         """
#         fixed_params = ground_truth_int_params.copy()
        
#         def embed_sp4_beta_tau_s_flow(
#             reduced_int_params: Dict[str, float],
#             ext_params: Any,
#             sim_params: Any,
#         ) -> Dict[str, Any]:
#             """
#             Transform reduced internal parameters into full int_params dict.
            
#             Args:
#                 reduced_int_params: Dict containing {'Sp4': inferred_value, 'Beta': inferred_value, 'tau_s': inferred_value}
#                 ext_params: Passed through unchanged (not modified here)
#                 sim_params: Passed through unchanged (not modified here)
            
#             Returns:
#                 Full int_params dict with Sp4 and tau_s updated, all other values fixed.
#             """
#             full_params = fixed_params.copy()
            
#             # Update only Sp4, Beta and tau_b; all other parameters remain fixed
#             if 'Sp4' in reduced_int_params:
#                 full_params['Sp4'] = reduced_int_params['Sp4']

#             if 'Beta' in reduced_int_params:
#                 full_params['Beta'] = reduced_int_params['Beta']

#             if 'tau_s' in reduced_int_params:
#                 full_params['tau_s'] = reduced_int_params['tau_s']
            
#             return full_params
        
#         # Create composed model with the embedding function
#         ViscoElasticModel = compose_model(
#             ViscoElasticFilament_FlowParams_ScalarBending,
#             compose_int_params=embed_sp4_beta_tau_s_flow,
#         )
#         return ViscoElasticModel

#     # ==========================
#     # ======= Optimizers =======
#     # ==========================

#     @pytest.fixture
#     def basinhopping_optimizer_instance(self):
#         """
#         Return the basinhopping optimizer function with standard configuration.
#         """
#         return basinhopping_optimizer        

#     @pytest.fixture
#     def optimizer_kwargs_sp4_beta(self):
#         return {
#             'bounds': Bounds(lb=[1e-6, 0], ub=[np.inf, np.inf]),
#             'minimum_gradient': False,
#             'minimum_hessian': False,
#             'local_minimizer_kwargs': {
#                 'method': 'L-BFGS-B',
#                 'jac': '3-point',
#                 'options': {
#                     'disp': True,
#                     'ftol': 1e-8,
#                     'gtol': 1e-8,
#                     'eps': 1e-8,
#                     'finite_diff_rel_step': 1e-6,
#                 },
#             },
#             'global_minimizer_kwargs': {
#                 'niter': 9,
#                 'T': 0,
#                 'stepsize': 5,
#                 'tol': 1e-10,
#             }
#         }

#     @pytest.fixture
#     def optimizer_kwargs_tau_s(self, optimizer_kwargs_sp4_beta):
#         optimizer_kwargs = optimizer_kwargs_sp4_beta
#         optimizer_kwargs['bounds'] = Bounds(lb=[0], ub=[np.inf])
#         return optimizer_kwargs

#     # ==========================
#     # ========= Passes =========
#     # ==========================

#     # Pass 1: Reduced model, infer Sp4 and Beta only
#     @pytest.fixture
#     def pass_1(
#         self,
#         multielastic_model_flow_sp4_beta_only,
#         ground_truth_ext_flow_params_static,
#         ground_truth_sim_params_static,
#         ground_truth_flow_data_static,
#         basinhopping_optimizer_instance,
#         optimizer_kwargs_sp4_beta,
#     ):
#         """ First pass for the viscoelastic inference. """
        
#         first_pass = PipelinePass(
#             name="Sp4-Beta Inference (MultiElastic Model)",
#             model_class=multielastic_model_flow_sp4_beta_only,
#             ground_truths=[ground_truth_flow_data_static],
#             ext_params_list=[ground_truth_ext_flow_params_static],
#             sim_params_list=[ground_truth_sim_params_static],
#             param_keys_to_infer=['Sp4', 'Beta'],
#             fixed_params={},
#             optimizer=basinhopping_optimizer_instance,
#             optimizer_kwargs=optimizer_kwargs_sp4_beta,
#         )

#         return first_pass

#     # Pass 2:
#     @pytest.fixture
#     def pass_2(
#         self,
#         viscoelastic_model_flow_sp4_beta_tau_s_only,
#         ground_truth_ext_flow_params_dynamic,
#         ground_truth_sim_params_dynamic,
#         ground_truth_flow_data_dynamic,
#         basinhopping_optimizer_instance,
#         optimizer_kwargs_tau_s,
#     ):
#         """ Second pass for the viscoelastic inference. """
        
#         second_pass = PipelinePass(
#             name="tau_s Inference (ViscoElastic Model)",
#             model_class=viscoelastic_model_flow_sp4_beta_tau_s_only,
#             ground_truths=[ground_truth_flow_data_dynamic],
#             ext_params_list=[ground_truth_ext_flow_params_dynamic],
#             sim_params_list=[ground_truth_sim_params_dynamic],
#             param_keys_to_infer=['tau_s'],
#             fixed_params={},
#             optimizer=basinhopping_optimizer_instance,
#             optimizer_kwargs=optimizer_kwargs_tau_s,
#         )

#         return second_pass

#     # ============================
#     # ========= Pipeline =========
#     # ============================
#     @pytest.fixture
#     def viscoelastic_pipeline(self, pass_1, pass_2, mse_loss_fn):

#         pipeline = InferencePipeline(
#             passes=[pass_1, pass_2],
#             loss_fn=mse_loss_fn,
#             n_jobs_per_pass=-1,  # Use all cores within each pass
#         )
#         return pipeline

#     # ============================
#     # ========= Tests =========
#     # ============================
#     def test_viscoelastic_inference(
#         self,
#         viscoelastic_pipeline,
#     ):

#         # Multiple initial guesses: Pass 1 for Sp4-Beta, Pass 2 for tau_s
#         initial_guesses_per_pass = [
#             [
#                 # {'Sp4': 1e0, 'Beta':0},
#                 # {'Sp4': 1e-1, 'Beta':1e0},
#                 {'Sp4': 10, 'Beta':0},
#             ],
#             [
#                 {'tau_s': 0},
#                 # {'tau_b': 10.0},
#             ],
#         ]
        
#         # ===== ACT =====
#         results = viscoelastic_pipeline.run(initial_guesses_per_pass, verbose=True)
        
#         # ===== ASSERT =====
        
#         # Check that we got exactly two results (two passes)
#         assert len(results) == 2, "Expected two InferenceResult for two-pass pipeline"
        
#         result_pass1, result_pass2 = results
        
#         # ===== PASS 1: Sp4 INFERENCE =====
#         print("\n" + "="*60)
#         print("PASS 1 ASSERTIONS (Sp4 inference)")
#         print("="*60)
        
#         # 1. Check Pass 1 convergence
#         assert result_pass1.success, (
#             f"Pass 1 optimization did not converge. Message: {result_pass1.message}"
#         )
#         assert result_pass1.iterations > 0, "Pass 1: No iterations were performed"
        
#         # 2. Check Sp4 and Beta exist and are physically reasonable

#         assert 'Sp4' in result_pass1.params, "Sp4 not in Pass 1 inferred parameters"
#         sp4_inferred = result_pass1.params['Sp4']
#         assert sp4_inferred > 0, f"Sp4 must be positive, got {sp4_inferred}"
#         assert sp4_inferred < 1e6, f"Sp4 unreasonably large: {sp4_inferred}"

#         assert 'Beta' in result_pass1.params, "Beta not in Pass 1 inferred parameters"
#         beta_inferred = result_pass1.params['Beta']
#         assert beta_inferred >= 0, f"Beta must be non-negative, got {beta_inferred}"
#         assert beta_inferred < 1e6, f"Beta unreasonably large: {beta_inferred}"      

#         # 3. Check Pass 1 loss
#         assert result_pass1.loss > 0, "Pass 1 loss should be positive"
#         assert result_pass1.loss < 1e2, (
#             f"Pass 1 loss suspiciously high: {result_pass1.loss}. Check model/data scale."
#         )
        
#         # 4. Check Pass 1 uncertainty quantification
#         assert result_pass1.covariance is not None, "Pass 1: Covariance not computed"
#         assert result_pass1.hessian is not None, "Pass 1: Hessian not computed"
#         assert result_pass1.std_errors is not None, "Pass 1: Standard errors not computed"
#         assert len(result_pass1.std_errors) == 2, "Pass 1: Expected 1 std_error for 2 parameter (Sp4, Beta)"
        
#         std_err_sp4 = result_pass1.std_errors[0]
#         assert std_err_sp4 > 0, f"Pass 1: Sp4 std error must be positive, got {std_err_sp4}"
#         assert std_err_sp4 < sp4_inferred * 10, (
#             f"Pass 1: Std error unreasonably large relative to Sp4: "
#             f"Sp4={sp4_inferred:.3e} ± {std_err_sp4:.3e}"
#         )
#         std_err_beta = result_pass1.std_errors[1]
#         assert std_err_beta > 0, f"Pass 1: Beta std error must be positive, got {std_err_beta}"
#         assert std_err_beta < beta_inferred * 10, (
#             f"Pass 1: Std error unreasonably large relative to Beta: "
#             f"Beta={beta_inferred:.3e} ± {std_err_beta:.3e}"
#         )
        
#         print(f"✓ Pass 1 - Sp4 inferred: {sp4_inferred:.6e} ± {std_err_sp4:.6e}")
#         print(f"✓ Pass 1 - Beta inferred: {beta_inferred:.6e} ± {std_err_beta:.6e}")
#         print(f"✓ Pass 1 - Final loss: {result_pass1.loss:.8e}")
#         print(f"✓ Pass 1 - Iterations: {result_pass1.iterations}")
        
#         # ===== PASS 2: tau_s INFERENCE (Sp4 and Beta FIXED) =====
#         print("\n" + "="*60)
#         print("PASS 2 ASSERTIONS (tau_s inference with Sp4 and Beta fixed)")
#         print("="*60)
        
#         # 1. Check Pass 2 convergence
#         assert result_pass2.success, (
#             f"Pass 2 optimization did not converge. Message: {result_pass2.message}"
#         )
#         assert result_pass2.iterations > 0, "Pass 2: No iterations were performed"
        
#         # 2. Check tau_s exists and is physically reasonable
#         assert 'tau_s' in result_pass2.params, "tau_s not in Pass 2 inferred parameters"
#         tau_s_inferred = result_pass2.params['tau_s']
#         assert tau_s_inferred >= 0, f"tau_s must be non-negative, got {tau_s_inferred}"
#         assert tau_s_inferred < 1e4, f"tau_s unreasonably large: {tau_s_inferred}"
        
#         # 3. Check Pass 2 loss
#         assert result_pass2.loss > 0, "Pass 2 loss should be positive"
#         assert result_pass2.loss < 1e2, (
#             f"Pass 2 loss suspiciously high: {result_pass2.loss}. Check model/data scale."
#         )

#         # 4. Check Pass 2 uncertainty quantification (only for tau_s, not Sp4)
#         assert result_pass2.covariance is not None, "Pass 2: Covariance not computed"
#         assert result_pass2.hessian is not None, "Pass 2: Hessian not computed"
#         assert result_pass2.std_errors is not None, "Pass 2: Standard errors not computed"
#         assert len(result_pass2.std_errors) == 1, "Pass 2: Expected 1 std_error for 1 inferred parameter (tau_s)"
        
#         std_err_tau_s = result_pass2.std_errors[0]
#         assert std_err_tau_s > 0, f"Pass 2: tau_s std error must be positive, got {std_err_tau_s}"
#         assert std_err_tau_s < tau_s_inferred * 10, (
#             f"Pass 2: Std error unreasonably large relative to tau_s: "
#             f"tau_s={tau_s_inferred:.3e} ± {std_err_tau_s:.3e}"
#         )
        
#         print(f"✓ Pass 2 - tau_s inferred: {tau_s_inferred:.6e} ± {std_err_tau_s:.6e}")
#         print(f"✓ Pass 2 - Sp4 fixed (from Pass 1): {result_pass1.params['Sp4']:.6e}")
#         print(f"✓ Pass 2 - Beta fixed (from Pass 1): {result_pass1.params['Beta']:.6e}")
#         print(f"✓ Pass 2 - Final loss: {result_pass2.loss:.8e}")
#         print(f"✓ Pass 2 - Iterations: {result_pass2.iterations}")
        
#         # ===== PIPELINE-LEVEL ASSERTIONS =====
#         print("\n" + "="*60)
#         print("PIPELINE-LEVEL ASSERTIONS")
#         print("="*60)
        
#         # Check parameter trajectory
#         trajectory = viscoelastic_pipeline.get_parameter_trajectory()
#         assert 'Sp4' in trajectory, "Sp4 not in trajectory"
#         assert 'Beta' in trajectory, "Beta not in trajectory"
#         assert 'tau_s' in trajectory, "tau_s not in trajectory"
#         assert len(trajectory['Sp4']) == 2, "Sp4 trajectory should have 2 steps"
#         assert len(trajectory['Beta']) == 2, "Beta trajectory should have 2 steps"
#         assert len(trajectory['tau_s']) == 2, "tau_s trajectory should have 2 steps"
        
#         # Sp4 should stay constant (inferred in Pass 1, fixed in Pass 2)
#         assert trajectory['Sp4'][0] == trajectory['Sp4'][1], (
#             f"Sp4 should be constant across passes. "
#             f"Pass1={trajectory['Sp4'][0]}, Pass2={trajectory['Sp4'][1]}"
#         )

#         # Beta should stay constant (inferred in Pass 1, fixed in Pass 2)
#         assert trajectory['Beta'][0] == trajectory['Beta'][1], (
#             f"Beta should be constant across passes. "
#             f"Pass1={trajectory['Beta'][0]}, Pass2={trajectory['Beta'][1]}"
#         )
        
#         # tau_s should appear in Pass 2 (None or absent in Pass 1)
#         assert trajectory['tau_s'][1] == tau_s_inferred, (
#             f"tau_s trajectory doesn't match Pass 2 result"
#         )
        
#         # Generate and verify summary
#         summary = viscoelastic_pipeline.summary()
#         print(summary)
#         assert summary, "Summary generation failed"
        
#         print("\n✓ All two-pass inference tests passed!")


if __name__ == "__main__":
    
    pytest.main([
        __file__,
        "-vv",
        "--tb=short",
        "-s",
    ])