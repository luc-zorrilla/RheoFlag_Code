import Models
import ViscoElasticFilament_Models
import Inferences
from scipy.optimize import minimize, Bounds
from _basinhopping_mod import *
import copy
import numpy as np


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


if __name__ == "__main__":
    
    inference = Inference(
        model_class=composed_model_sp4_only,
        ground_truth=ground_truth_data,
        loss_fn=mse_loss_fn,
        optimizer=basinhopping_optimizer,
        optimizer_kwargs={
            
            # Individual arguments
            'bounds': Bounds(lb=[1e-6], ub=[np.inf]), # Those default bounds are for Sp4, which is strictly positive.
            'minimum_gradient': False,
            'minimum_hessian': False,
            
            # Local minimizer arguments
            'local_minimizer_kwargs': {
                'method':'L-BFGS-B',
                'jac': '3-point',
                "options":{
                    'disp': True ,
                    'ftol':1e-8, 
                    'gtol':1e-8, 
                    'eps': 1e-8, 
                    'finite_diff_rel_step':1e-6,
                    # 'maxiter': 1000,
                    },                
            },

            # Global minimizer arguments
            'global_minimizer_kwargs': {
                'niter': 9,
                'T': 0,
                'stepsize': 5,
                'tol': 1e-10,
            }

        }
    )

    result = inference.infer(
        initial_guess={'Sp4': 2.5},
        ext_params=ground_truth_ext_params,
        sim_params=ground_truth_sim_params
    )

    print(f"Inferred Sp4: {result['params']['Sp4']}")
    print(f"Basin-hopping acceptance rate: {np.mean(result.result.accept_global)}")