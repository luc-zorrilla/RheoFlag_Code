import numpy as np
from scipy.optimize import minimize
import scipy.differentiate as sd
from typing import Callable, Dict, Any, Tuple
import joblib
from functools import partial
import logging
from Models import Model, Square
from itertools import product
import pandas as pd

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
        optimizer: Callable = None,
        optimizer_kwargs: Dict[str, Any] = None,
        ):
        """
        Args:
            model_class: Model subclass (e.g., Square)
            ground_truth: Target data (numpy array)
            loss_fn: Callable(predicted, ground_truth) -> scalar loss
            optimizer: Callable that takes (objective, x0, **kwargs) and returns OptimizeResult
                Defaults to scipy.optimize.minimize
            optimizer_kwargs: Dict of kwargs for the optimizer
        """
        self.model_class = model_class
        self.ground_truth = ground_truth
        self.loss_fn = loss_fn
        self.optimizer = optimizer or minimize
        self.optimizer_kwargs = optimizer_kwargs or {
            'method': 'L-BFGS-B',
            'options': {'ftol': 1e-8, 'maxiter': 1000}
        }
        self.result = None
        self.hessian = None
        self.covariance = None

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
        """
        param_keys = tuple(initial_guess.keys())
        x0 = np.array([initial_guess[key] for key in param_keys])
        
        # Run optimization with pluggable optimizer
        self.result = self.optimizer(
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


    def infer_batch(
        self,
        initial_guesses: list,
        ext_params: Any = None,
        sim_params: Any = None,
        ) -> list:
        """
        Run inference on multiple initial guesses in parallel (for a fixed ground truth).
        
        Args:
            initial_guesses: List of dicts, each like {'x': 2.5}
            ext_params, sim_params: corresponding parameters (can be None)
        
        Returns:
            List of inference results
        """
        ext_params_batch = [ext_params] * len(initial_guesses)
        sim_params_batch = [sim_params] * len(initial_guesses)
        
        results = joblib.Parallel(n_jobs=self.n_jobs)(
            joblib.delayed(self.inference.infer)(ig, ep, sp)
            for ig, ep, sp in zip(initial_guesses, ext_params_batch, sim_params_batch)
        )
        return results

    def _compute_hessian(
        self,
        param_keys: Tuple[str, ...],
        ext_params: Any,
        sim_params: Any,
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
        
        try:
            self.covariance = np.linalg.inv(self.hessian)
        except np.linalg.LinAlgError:
            print("Warning: Hessian singular, covariance unavailable.")
            self.covariance = None
    
    
    