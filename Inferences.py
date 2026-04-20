import numpy as np
from scipy.optimize import minimize, approx_fprime
import scipy.differentiate as sd
from typing import Callable, Dict, Any, Tuple
import joblib
from functools import partial
import logging
from Models import Model, Square

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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

class Inference:
    """
    Optimization-based parameter inference.
    
    Given:
    - A Model subclass (e.g., Square)
    - Ground truth data
    - Custom loss function
    
    Infers:
    - Internal parameters (int_params)
    - Parameter covariance (from Hessian)
    """

    def __init__(
        self,
        model_class,
        ground_truth: np.ndarray,
        loss_fn: Callable,
        optimizer_kwargs: Dict[str, Any] = None,
    ):
        """
        Args:
            model_class: Model subclass (e.g., Square)
            ground_truth: Target data (numpy array)
            loss_fn: Callable(predicted, ground_truth) -> scalar loss
            optimizer_kwargs: Dict of kwargs for scipy.optimize.minimize()
                             (e.g., {'method': 'L-BFGS-B', 'options': {'ftol': 1e-6}})
        """
        self.model_class = model_class
        self.ground_truth = ground_truth
        self.loss_fn = loss_fn
        self.optimizer_kwargs = optimizer_kwargs or {
            'method': 'L-BFGS-B',
            'options': {'ftol': 1e-8, 'maxiter': 1000}
        }
        self.result = None # Result should contain all information from the inference steps.
        self.hessian = None # Not needed
        self.covariance = None # Not needed

    def objective(
        self,
        param_vector: np.ndarray,
        param_keys: Tuple[str, ...],
        ext_params: Any,
        sim_params: Any,
    ) -> float:
        """
        Objective function for optimizer.
        
        Args:
            param_vector: Flat array of parameter values to optimize
            param_keys: Keys mapping vector indices to parameter names
            ext_params, sim_params: External and simulation parameters (fixed)
        
        Returns:
            Scalar loss value
        """
        # Reconstruct int_params dict from flat vector
        int_params = {key: param_vector[i] for i, key in enumerate(param_keys)}
        
        # Run simulation
        instance = self.model_class(int_params, ext_params, sim_params)
        predicted = instance.simulate_single()['value']
        
        # Compute loss
        loss = self.loss_fn(predicted, self.ground_truth)
        return loss

    def infer(
        self,
        initial_guess: Dict[str, float],
        ext_params: Any = None,
        sim_params: Any = None,
    ) -> Dict[str, Any]:
        """
        Run optimization to infer parameters.
        
        Args:
            initial_guess: Dict like {'x': 2.5} for Square model
            ext_params, sim_params: External/simulation parameters (can be None)
        
        Returns:
            Dict with keys:
            - 'params': Optimal parameter dict
            - 'loss': Final loss value
            - 'covariance': Parameter covariance (from Hessian inverse)
            - 'std_errors': Standard errors for each parameter
        """
        param_keys = tuple(initial_guess.keys())
        x0 = np.array([initial_guess[key] for key in param_keys])
        
        # Run optimization
        self.result = minimize(
            partial(
                self.objective,
                param_keys=param_keys,
                ext_params=ext_params,
                sim_params=sim_params,
            ),
            x0,
            **self.optimizer_kwargs
        )
        
        if not self.result.success:
            print(f"Optimization warning: {self.result.message}")
        
        # Compute Hessian numerically
        self._compute_hessian(param_keys, ext_params, sim_params)
        
        # Reconstruct optimal parameters
        optimal_params = {key: self.result.x[i] for i, key in enumerate(param_keys)}
        
        return {
            'params': optimal_params,
            'loss': self.result.fun,
            'covariance': self.covariance,
            'hessian': self.hessian,
            'std_errors': np.sqrt(np.diag(self.covariance)) if self.covariance is not None else None,
            'iterations': self.result.nit,
        }

    def _compute_hessian( # This could be replaced by a simple ddf from scipy...
        self,
        param_keys: Tuple[str, ...],
        ext_params: Any,
        sim_params: Any,
        eps: float = 1e-5,
    ):
        """
        Compute Hessian numerically using finite differences.
        Invert to get parameter covariance.
        
        Args:
            param_keys: Parameter keys
            ext_params, sim_params: Model parameters
            eps: Step size for finite differences
        """

        m = len(param_keys) # number of variables

        self.hessian = sd.hessian(
                f = Vectorize_Functional(partial(
                    self.objective,
                    param_keys=param_keys,
                    ext_params=ext_params,
                    sim_params=sim_params,
                    ),
                m = m,
                ), 
                x = self.result.x).ddf

        # # Gradient function for Hessian computation
        # def grad_fn(x):
        #     return approx_fprime(
        #         x,
        #         partial(
        #             self.objective,
        #             param_keys=param_keys,
        #             ext_params=ext_params,
        #             sim_params=sim_params,
        #         ),
        #         epsilon=eps
        #     )
        
        # # Numerical Hessian (finite differences of gradients)
        # n = len(self.result.x)
        # hessian = np.zeros((n, n))
        
        # for i in range(n):
        #     x_plus = self.result.x.copy()
        #     x_plus[i] += eps
        #     x_minus = self.result.x.copy()
        #     x_minus[i] -= eps
            
        #     grad_plus = grad_fn(x_plus)
        #     grad_minus = grad_fn(x_minus)
            
        #     hessian[i, :] = (grad_plus - grad_minus) / (2 * eps)
        
        # self.hessian = hessian
        
        try:
            self.covariance = np.linalg.inv(self.hessian)
        except np.linalg.LinAlgError:
            print("Warning: Hessian singular, covariance unavailable.")
            self.covariance = None

class BatchInference:
    """
    Parallel inference over multiple ground-truth samples or initial guesses.
    """

    def __init__(self, inference: Inference, n_jobs: int = -1):
        """
        Args:
            inference: Inference instance (will be reused)
            n_jobs: Number of parallel jobs (-1 = all CPUs)
        """
        self.inference = inference
        self.n_jobs = n_jobs

    def infer_batch(
        self,
        initial_guesses: list,
        ext_params_batch: list = None,
        sim_params_batch: list = None,
    ) -> list:
        """
        Run inference on multiple initial guesses in parallel.
        
        Args:
            initial_guesses: List of dicts, each like {'x': 2.5}
            ext_params_batch, sim_params_batch: Corresponding parameters (can be None)
        
        Returns:
            List of inference results
        """
        if ext_params_batch is None:
            ext_params_batch = [None] * len(initial_guesses)
        if sim_params_batch is None:
            sim_params_batch = [None] * len(initial_guesses)
        
        results = joblib.Parallel(n_jobs=self.n_jobs)(
            joblib.delayed(self.inference.infer)(ig, ep, sp)
            for ig, ep, sp in zip(initial_guesses, ext_params_batch, sim_params_batch)
        )
        return results