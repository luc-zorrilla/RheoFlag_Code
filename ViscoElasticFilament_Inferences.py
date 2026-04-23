from Models import Model, compose_model
from ViscoElasticFilament_Models import (
    StraightLine, 
    ViscoElasticFilament, 
    ViscoElasticFilament_create_params_list, # TODO: check if necessary
    ViscoElasticFilament_FlowParams, 
    ViscoElasticFilament_FlowParams_create_params_list, # TODO: check if necessary
)
from Inferences import Inference, InferencePipeline, PipelinePass, InferenceResult

from scipy.optimize import minimize, Bounds
from _basinhopping_mod import *
import numpy as np

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

class TestViscoElasticFilament_TwoPassInference:
    """ Infer Sp4 and tau_s in a two-pass inference, with 
        - static experimental data in the first pass,
        - dynamic experimental data in the second pass.

    Parameters to be inferred:
        - Sp4: 1.0
        - tau_b: 1.0 
    """

    # ====================
    # ======= Loss =======
    # ====================

    @pytest.fixture
    def mse_loss_fn(self) -> Callable:
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

    # ======================
    # ==== Ground Truth ====
    # ======================

    @pytest.fixture
    def ground_truth_int_params(self):
        """
        Define internal parameters with a known Sp4 ground truth value.
        All other parameters are fixed for inference.
        """
        N = 10
        X0 = StraightLine(N)
        return {
            'Sp4': 1.0,           # Ground truth to recover
            'N': 10,            
            'k0': 1e13,            
            'bool_EI': True,      
            'Beta':0,        
            'taus_b': [0]*(N-1),  
            'tau_s': 1e6,        # Ground truth to recover
            'gamma': 2,         
            'n_L': [0,0],            
            'm_L': 0,             
            'X_0': X0,  # Initial state
        }

    @pytest.fixture
    def ground_truth_ext_flow_params_static(self):
        """Define external parameters for static response (fixed during inference)."""
        N = 10
        return {
            "Lambdas": [[0,0]]*N,
            "Zetas": [0]*N,
            "A":1e-6,
            "w0":0, # Static
            "psi":np.pi/2,
        }

    @pytest.fixture
    def ground_truth_ext_flow_params_dynamic(self):
        """Define external parameters for static response (fixed during inference)."""
        N = 10
        return {
            "Lambdas": [[0,0]]*N,
            "Zetas": [0]*N,
            "A":1e-6,
            "w0":1e-6, # Dynamic
            "psi":np.pi/2,
        }        

    @pytest.fixture
    def ground_truth_sim_params_static(self):
        """
        Define simulation parameters.
        Use BDF method for faster convergence compared to RK45.
        """
        return {
            "T_span": (1e9, 2e9),
            "T_eval": np.linspace(1e9, 2e9, int(2e0)), # minimum two elements here.
            "method": "BDF",
            "T_sim_max": 300,
        }

    @pytest.fixture
    def ground_truth_sim_params_dynamic(self):
        """
        Define simulation parameters.
        Use BDF method for faster convergence compared to RK45.
        """
        return {
            "T_span": (1e7, 2e7),
            "T_eval": np.linspace(1e7, 2e7, int(1e2)),
            "method": "BDF",
            "T_sim_max": 300,
        }

    @pytest.fixture
    def ground_truth_flow_data_static(
        self, 
        ground_truth_int_params, 
        ground_truth_ext_flow_params_static, 
        ground_truth_sim_params_static,
    ):
        """
        Generate ground truth synthetic data by simulating ViscoElasticFilament
        with known parameters.
        
        Returns:
            np.ndarray: Simulated trajectory (shape depends on model)
        """
        model = ViscoElasticFilament_FlowParams(
            int_params=ground_truth_int_params,
            ext_params=ground_truth_ext_flow_params_static,
            sim_params=ground_truth_sim_params_static
        )
        output = model.simulate_single()
        
        assert output['value'] is not None, (
            f"Ground truth simulation failed. Full output: {output}"
        )

        return output['value']


    @pytest.fixture
    def ground_truth_flow_data_dynamic(
        self,
        ground_truth_int_params, 
        ground_truth_ext_flow_params_dynamic, 
        ground_truth_sim_params_dynamic,
    ):
        """
        Generate ground truth synthetic data by simulating ViscoElasticFilament
        with known parameters.
        
        Returns:
            np.ndarray: Simulated trajectory (shape depends on model)
        """
        model = ViscoElasticFilament_FlowParams(
            int_params=ground_truth_int_params,
            ext_params=ground_truth_ext_flow_params_dynamic,
            sim_params=ground_truth_sim_params_dynamic,
        )
        output = model.simulate_single()
        
        assert output['value'] is not None, (
            f"Ground truth simulation failed. Full output: {output}"
        )

        return output['value']
    
    # ======================
    # ======= Models =======
    # ======================

    @pytest.fixture
    def elastic_model_flow_sp4_only(
        self,
        ground_truth_int_params
    ):
        """
        Create a composed model for ViscoElasticFilament_FlowParams that
            - only varies Sp4
            - sets tau_s to zero (elastic model)
        
        The embedding function accepts a reduced parameter dict {'Sp4': value}
        and embeds it into the full internal parameters, keeping all others fixed.
        """
        fixed_params = ground_truth_int_params.copy()
        fixed_params['tau_s'] = 0 # Set dynamic parameters to 0.
        
        def embed_sp4_flow(
            reduced_int_params: Dict[str, float],
            ext_params: Any,
            sim_params: Any,
        ) -> Dict[str, Any]:
            """
            Transform reduced internal parameters into full int_params dict.
            
            Args:
                reduced_int_params: Dict containing {'Sp4': inferred_value}
                ext_params: Passed through unchanged (not modified here)
                sim_params: Passed through unchanged (not modified here)
            
            Returns:
                Full int_params dict with Sp4 updated, all other values fixed.
            """
            full_params = fixed_params.copy()
            
            # Update only Sp4; all other parameters remain fixed
            if 'Sp4' in reduced_int_params:
                full_params['Sp4'] = reduced_int_params['Sp4']
            
            return full_params
        
        # Create composed model with the embedding function
        ElasticModel = compose_model(
            ViscoElasticFilament_FlowParams,
            compose_int_params=embed_sp4_flow,
        )
        return ElasticModel

    @pytest.fixture
    def viscoelastic_model_flow_sp4_tau_s_only(
        self,
        ground_truth_int_params,
    ):
        """
        Create a composed model for ViscoElasticFilament_FlowParams that only varies Sp4 and tau_s.
        
        The embedding function accepts a reduced parameter dict {'Sp4': value, 'tau_s': value}
        and embeds it into the full internal parameters, keeping all others fixed.
        """
        fixed_params = ground_truth_int_params.copy()
        
        def embed_sp4_tau_s_flow(
            reduced_int_params: Dict[str, float],
            ext_params: Any,
            sim_params: Any,
        ) -> Dict[str, Any]:
            """
            Transform reduced internal parameters into full int_params dict.
            
            Args:
                reduced_int_params: Dict containing {'Sp4': inferred_value, 'tau_s': inferred_value}
                ext_params: Passed through unchanged (not modified here)
                sim_params: Passed through unchanged (not modified here)
            
            Returns:
                Full int_params dict with Sp4 and tau_s updated, all other values fixed.
            """
            full_params = fixed_params.copy()
            
            # Update only Sp4; all other parameters remain fixed
            if 'Sp4' in reduced_int_params:
                full_params['Sp4'] = reduced_int_params['Sp4']

            if 'tau_s' in reduced_int_params:
                full_params['tau_s'] = reduced_int_params['tau_s']
            
            return full_params
        
        # Create composed model with the embedding function
        ViscoElasticModel = compose_model(
            ViscoElasticFilament_FlowParams,
            compose_int_params=embed_sp4_tau_s_flow,
        )
        return ViscoElasticModel

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
    def optimizer_kwargs_sp4(self):
        return {
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
        }

    @pytest.fixture
    def optimizer_kwargs_tau_s(self, optimizer_kwargs_sp4):
        optimizer_kwargs = optimizer_kwargs_sp4
        optimizer_kwargs['bounds'] = Bounds(lb=[0], ub=[np.inf])
        return optimizer_kwargs

    # ==========================
    # ========= Passes =========
    # ==========================

    # Pass 1: Reduced model, infer Sp4 only
    @pytest.fixture
    def pass_1(
        self,
        elastic_model_flow_sp4_only,
        ground_truth_ext_flow_params_static,
        ground_truth_sim_params_static,
        ground_truth_flow_data_static,
        basinhopping_optimizer_instance,
        optimizer_kwargs_sp4,
    ):
        """ First pass for the viscoelastic inference. """
        
        first_pass = PipelinePass(
            name="Sp4 Inference (Elastic Model)",
            model_class=elastic_model_flow_sp4_only,
            ground_truths=[ground_truth_flow_data_static],
            ext_params_list=[ground_truth_ext_flow_params_static],
            sim_params_list=[ground_truth_sim_params_static],
            param_keys_to_infer=['Sp4'],
            fixed_params={},
            optimizer=basinhopping_optimizer_instance,
            optimizer_kwargs=optimizer_kwargs_sp4,
        )

        return first_pass

    # Pass 2:
    @pytest.fixture
    def pass_2(
        self,
        viscoelastic_model_flow_sp4_tau_s_only,
        ground_truth_ext_flow_params_dynamic,
        ground_truth_sim_params_dynamic,
        ground_truth_flow_data_dynamic,
        basinhopping_optimizer_instance,
        optimizer_kwargs_tau_s,
    ):
        """ Second pass for the viscoelastic inference. """
        
        second_pass = PipelinePass(
            name="tau_s Inference (ViscoElastic Model)",
            model_class=viscoelastic_model_flow_sp4_tau_s_only,
            ground_truths=[ground_truth_flow_data_dynamic],
            ext_params_list=[ground_truth_ext_flow_params_dynamic],
            sim_params_list=[ground_truth_sim_params_dynamic],
            param_keys_to_infer=['tau_s'],
            fixed_params={},
            optimizer=basinhopping_optimizer_instance,
            optimizer_kwargs=optimizer_kwargs_tau_s,
        )

        return second_pass

    # ============================
    # ========= Pipeline =========
    # ============================
    @pytest.fixture
    def viscoelastic_pipeline(self, pass_1, pass_2, mse_loss_fn):

        pipeline = InferencePipeline(
            passes=[pass_1, pass_2],
            loss_fn=mse_loss_fn,
            n_jobs_per_pass=-1,  # Use all cores within each pass
        )
        return pipeline

    # ============================
    # ========= Tests =========
    # ============================
    def test_viscoelastic_inference(
        self,
        viscoelastic_pipeline,
    ):

        # Multiple initial guesses: Pass 1 for Sp4, Pass 2 for tau_s
        initial_guesses_per_pass = [
            [
                # {'Sp4': 0.1},
                {'Sp4': 10.0},
            ],
            [
                {'tau_s': 0},
                # {'tau_s': 10.0},
            ],
        ]
        
        # ===== ACT =====
        results = viscoelastic_pipeline.run(initial_guesses_per_pass, verbose=True)
        
        # ===== ASSERT =====
        
        # Check that we got exactly two results (two passes)
        assert len(results) == 2, "Expected two InferenceResult for two-pass pipeline"
        
        result_pass1, result_pass2 = results
        
        # ===== PASS 1: Sp4 INFERENCE =====
        print("\n" + "="*60)
        print("PASS 1 ASSERTIONS (Sp4 inference)")
        print("="*60)
        
        # 1. Check Pass 1 convergence
        assert result_pass1.success, (
            f"Pass 1 optimization did not converge. Message: {result_pass1.message}"
        )
        assert result_pass1.iterations > 0, "Pass 1: No iterations were performed"
        
        # 2. Check Sp4 exists and is physically reasonable
        assert 'Sp4' in result_pass1.params, "Sp4 not in Pass 1 inferred parameters"
        sp4_inferred = result_pass1.params['Sp4']
        assert sp4_inferred > 0, f"Sp4 must be positive, got {sp4_inferred}"
        assert sp4_inferred < 1e6, f"Sp4 unreasonably large: {sp4_inferred}"
        
        # 3. Check Pass 1 loss
        assert result_pass1.loss > 0, "Pass 1 loss should be positive"
        assert result_pass1.loss < 1e2, (
            f"Pass 1 loss suspiciously high: {result_pass1.loss}. Check model/data scale."
        )
        
        # 4. Check Pass 1 uncertainty quantification
        assert result_pass1.covariance is not None, "Pass 1: Covariance not computed"
        assert result_pass1.hessian is not None, "Pass 1: Hessian not computed"
        assert result_pass1.std_errors is not None, "Pass 1: Standard errors not computed"
        assert len(result_pass1.std_errors) == 1, "Pass 1: Expected 1 std_error for 1 parameter (Sp4)"
        
        std_err_sp4 = result_pass1.std_errors[0]
        assert std_err_sp4 > 0, f"Pass 1: Sp4 std error must be positive, got {std_err_sp4}"
        assert std_err_sp4 < sp4_inferred * 10, (
            f"Pass 1: Std error unreasonably large relative to Sp4: "
            f"Sp4={sp4_inferred:.3e} ± {std_err_sp4:.3e}"
        )
        
        print(f"✓ Pass 1 - Sp4 inferred: {sp4_inferred:.6e} ± {std_err_sp4:.6e}")
        print(f"✓ Pass 1 - Final loss: {result_pass1.loss:.8e}")
        print(f"✓ Pass 1 - Iterations: {result_pass1.iterations}")
        
        # ===== PASS 2: tau_s INFERENCE (Sp4 FIXED) =====
        print("\n" + "="*60)
        print("PASS 2 ASSERTIONS (tau_s inference with Sp4 fixed)")
        print("="*60)
        
        # 1. Check Pass 2 convergence
        assert result_pass2.success, (
            f"Pass 2 optimization did not converge. Message: {result_pass2.message}"
        )
        assert result_pass2.iterations > 0, "Pass 2: No iterations were performed"
        
        # 2. Check tau_s exists and is physically reasonable
        assert 'tau_s' in result_pass2.params, "tau_s not in Pass 2 inferred parameters"
        tau_s_inferred = result_pass2.params['tau_s']
        assert tau_s_inferred >= 0, f"tau_s must be non-negative, got {tau_s_inferred}"
        assert tau_s_inferred < 1e4, f"tau_s unreasonably large: {tau_s_inferred}"
        
        # 3. Check Pass 2 loss
        assert result_pass2.loss > 0, "Pass 2 loss should be positive"
        assert result_pass2.loss < 1e2, (
            f"Pass 2 loss suspiciously high: {result_pass2.loss}. Check model/data scale."
        )
        
        # 4. Expect Pass 2 loss <= Pass 1 loss (we're adding a parameter, so fit improves)
        assert result_pass2.loss <= result_pass1.loss * 1.1, (  # Allow 10% tolerance for noise
            f"Pass 2 loss should improve (or stay similar) over Pass 1. "
            f"Pass1={result_pass1.loss:.3e}, Pass2={result_pass2.loss:.3e}"
        )
        
        # 5. Check Pass 2 uncertainty quantification (only for tau_s, not Sp4)
        assert result_pass2.covariance is not None, "Pass 2: Covariance not computed"
        assert result_pass2.hessian is not None, "Pass 2: Hessian not computed"
        assert result_pass2.std_errors is not None, "Pass 2: Standard errors not computed"
        assert len(result_pass2.std_errors) == 1, "Pass 2: Expected 1 std_error for 1 inferred parameter (tau_s)"
        
        std_err_tau_s = result_pass2.std_errors[0]
        assert std_err_tau_s > 0, f"Pass 2: tau_s std error must be positive, got {std_err_tau_s}"
        assert std_err_tau_s < tau_s_inferred * 10, (
            f"Pass 2: Std error unreasonably large relative to tau_s: "
            f"tau_s={tau_s_inferred:.3e} ± {std_err_tau_s:.3e}"
        )
        
        print(f"✓ Pass 2 - tau_s inferred: {tau_s_inferred:.6e} ± {std_err_tau_s:.6e}")
        print(f"✓ Pass 2 - Sp4 fixed (from Pass 1): {result_pass1.params['Sp4']:.6e}")
        print(f"✓ Pass 2 - Final loss: {result_pass2.loss:.8e}")
        print(f"✓ Pass 2 - Iterations: {result_pass2.iterations}")
        
        # ===== PIPELINE-LEVEL ASSERTIONS =====
        print("\n" + "="*60)
        print("PIPELINE-LEVEL ASSERTIONS")
        print("="*60)
        
        # Check parameter trajectory
        trajectory = viscoelastic_pipeline.get_parameter_trajectory()
        assert 'Sp4' in trajectory, "Sp4 not in trajectory"
        assert 'tau_s' in trajectory, "tau_s not in trajectory"
        assert len(trajectory['Sp4']) == 2, "Sp4 trajectory should have 2 steps"
        assert len(trajectory['tau_s']) == 2, "tau_s trajectory should have 2 steps"
        
        # Sp4 should stay constant (inferred in Pass 1, fixed in Pass 2)
        assert trajectory['Sp4'][0] == trajectory['Sp4'][1], (
            f"Sp4 should be constant across passes. "
            f"Pass1={trajectory['Sp4'][0]}, Pass2={trajectory['Sp4'][1]}"
        )
        
        # tau_s should appear in Pass 2 (None or absent in Pass 1)
        assert trajectory['tau_s'][1] == tau_s_inferred, (
            f"tau_s trajectory doesn't match Pass 2 result"
        )
        
        # Generate and verify summary
        summary = pipeline.summary()
        print(summary)
        assert summary, "Summary generation failed"
        
        print("\n✓ All two-pass inference tests passed!")

if __name__ == "__main__":
    
    pytest.main([
        __file__,
        "-vv",
        "--tb=short",
        "-s",
    ])