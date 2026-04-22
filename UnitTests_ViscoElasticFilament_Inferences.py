"""
UnitTests_ViscoElasticFilament_Models_Inference

Test suite for inferring the Sp4 internal parameter of the ViscoElasticFilament model.
Uses compose_model to create a reduced inference problem (only Sp4 varies).
"""

# ========================================================================
# MODULES
# ========================================================================

import pytest
import numpy as np
from functools import partial
from typing import Dict, Any, Callable
from scipy import stats
from itertools import product
import pandas as pd
from pathlib import Path

# Import your modules (adjust paths as needed)
from Models import Model, compose_model
from ViscoElasticFilament_Models import StraightLine, ViscoElasticFilament, ViscoElasticFilament_create_params_list, ViscoElasticFilament_FlowParams, ViscoElasticFilament_FlowParams_create_params_list
from Inferences import Inference, InferencePipeline, PipelinePass, InferenceResult
from ViscoElasticFilament_Inferences import basinhopping_optimizer
from scipy.optimize import Bounds

# ========================================================================
# FUNCTIONS
# ========================================================================

def print_inference_summary(
    result: Dict[str, Any],
    ground_truth_params: Dict[str, float] = None,
    confidence_level: float = 0.95,
) -> None:
    """
    Print a formatted summary of inferred parameters with error bars and statistics.
    
    Args:
        result: Dict returned by Inference.infer()
        ground_truth_params: Dict of true parameter values for comparison
        confidence_level: Confidence level for CI (default 95%)
    """
    
    inferred_params = result.params
    std_errors = result.std_errors
    loss = result.loss
    iterations = result.iterations
    
    # Compute z-score for confidence interval
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    
    print(f"\n{'='*80}")
    print(f"INFERENCE SUMMARY")
    print(f"{'='*80}")
    
    print(f"\nOptimization Results:")
    print(f"  Final loss:        {loss:.8e}")
    print(f"  Iterations:        {iterations}")
    
    print(f"\n{'Parameter':<20} {'Inferred':<18} {'Std Error':<18} {'95% CI':<30}")
    print(f"{'-'*80}")
    
    for param_name in inferred_params.keys():
        inferred_val = inferred_params[param_name]
        
        # Get standard error (or mark as unavailable)
        if std_errors is not None:
            param_idx = list(inferred_params.keys()).index(param_name)
            std_err = std_errors[param_idx]
            margin = z_score * std_err
            ci_lower = inferred_val - margin
            ci_upper = inferred_val + margin
            ci_str = f"[{ci_lower:.6e}, {ci_upper:.6e}]"
            std_err_str = f"{std_err:.6e}"
        else:
            std_err_str = "N/A"
            ci_str = "N/A (singular Hessian)"
        
        print(f"{param_name:<20} {inferred_val:.8e}   {std_err_str:<18} {ci_str:<30}")
    
    # Comparison with ground truth (if provided)
    if ground_truth_params is not None:
        print(f"\n{'Ground Truth Comparison':<20} {'Relative Error':<20} {'Error (std)':<20}")
        print(f"{'-'*80}")
        
        for param_name in inferred_params.keys():
            if param_name not in ground_truth_params:
                print(f"{param_name:<20} {'(not provided)':<20}")
                continue
            
            inferred_val = inferred_params[param_name]
            gt_val = ground_truth_params[param_name]
            
            # Relative error
            rel_error = abs(inferred_val - gt_val) / abs(gt_val) if gt_val != 0 else np.inf
            
            # Error in standard deviations (z-score)
            if std_errors is not None:
                param_idx = list(inferred_params.keys()).index(param_name)
                std_err = std_errors[param_idx]
                z_error = abs(inferred_val - gt_val) / std_err if std_err > 0 else np.inf
                z_str = f"{z_error:.2f}σ"
            else:
                z_str = "N/A"
            
            status = "✓ PASS" if rel_error < 0.1 else "✗ FAIL"
            print(f"{param_name:<20} {rel_error*100:>6.2f}%            {z_str:<20} {status}")
    
    print(f"\n{'='*80}\n")

def print_optimization_history(inference_instance, plot_flag: bool = False) -> None:
    """
    Print optimization trajectory statistics and optionally plot convergence.
    
    Args:
        inference_instance: Inference object after running infer()
        plot_flag: If True, attempt to plot loss landscape (requires matplotlib)
    """
    hist = inference_instance.result
    
    if not hasattr(hist, 'F_global') or len(hist.F_global) == 0:
        print("Warning: No optimization history available")
        return
    
    losses = np.array(hist.F_global)
    acceptances = np.array(hist.accept_global)
    
    print(f"\n{'='*80}")
    print(f"OPTIMIZATION HISTORY")
    print(f"{'='*80}")
    
    print(f"\nBasin-hopping Statistics:")
    print(f"  Initial loss:           {losses[0]:.8e}")
    print(f"  Final loss:             {losses[-1]:.8e}")
    print(f"  Best loss:              {np.min(losses):.8e}")
    print(f"  Worst loss:             {np.max(losses):.8e}")
    print(f"  Mean loss:              {np.mean(losses):.8e}")
    print(f"  Std dev (loss):         {np.std(losses):.8e}")
    
    improvement = (losses[0] - losses[-1]) / losses[0] * 100
    print(f"  Total improvement:      {improvement:.2f}%")
    
    # Acceptance statistics
    n_accepted = np.sum(acceptances)
    total_steps = len(acceptances)
    acceptance_rate = n_accepted / total_steps if total_steps > 0 else 0
    
    print(f"\nAcceptance Statistics:")
    print(f"  Total steps:            {total_steps}")
    print(f"  Accepted steps:         {n_accepted}")
    print(f"  Rejection rate:         {(1-acceptance_rate)*100:.2f}%")
    print(f"  Acceptance rate:        {acceptance_rate*100:.2f}%")
    
    # Convergence analysis
    recent_window = min(5, len(losses))
    recent_loss_change = (losses[-recent_window] - losses[-1]) / losses[-recent_window] * 100
    print(f"\nConvergence Analysis (last {recent_window} steps):")
    print(f"  Loss reduction:         {recent_loss_change:.4f}%")
    print(f"  Converged:              {'Yes ✓' if recent_loss_change < 1.0 else 'No ✗'}")
    
    print(f"\n{'='*80}\n")
    
    # Optional: Plot convergence
    if plot_flag:
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 1, figsize=(10, 8))
            
            # Loss convergence
            axes[0].semilogy(range(len(losses)), losses, 'b-o', markersize=4)
            axes[0].set_xlabel('Basin-hopping iteration')
            axes[0].set_ylabel('Loss (log scale)')
            axes[0].set_title('Loss Convergence')
            axes[0].grid(True, alpha=0.3)
            
            # Acceptance rate (cumulative)
            cumulative_accept_rate = np.cumsum(acceptances) / np.arange(1, len(acceptances) + 1)
            axes[1].plot(range(len(acceptances)), cumulative_accept_rate, 'g-o', markersize=4)
            axes[1].set_xlabel('Basin-hopping iteration')
            axes[1].set_ylabel('Cumulative acceptance rate')
            axes[1].set_title('Acceptance Rate Over Time')
            axes[1].set_ylim([0, 1])
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('inference_convergence.png', dpi=150, bbox_inches='tight')
            print("Plot saved to 'inference_convergence.png'\n")
            
        except ImportError:
            print("(matplotlib not available for plotting)")
    
    def to_dataframe(self):
        """Convert results to pandas DataFrame."""
        return pd.DataFrame(self.results)
    
    def save_csv(self, filepath='test_results.csv'):
        """Save results to CSV."""
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)
        print(f"\n✓ Results saved to {filepath}")
    
    def print_summary(self):
        df = self.to_dataframe()
        
        # Debug: Print collector state
        print(f"\n[DEBUG] Collector has {len(self.results)} results")
        if len(self.results) > 0:
            print(f"[DEBUG] First result: {self.results[0]}")
        
        # Handle empty results
        if len(df) == 0:
            print("\n" + "="*120)
            print("TEST RESULTS SUMMARY")
            print("="*120)
            print("No test results collected.")
            print("="*120 + "\n")
            return
        
        passed = len(df[df['Status'] == '✓ PASS'])
        failed = len(df[df['Status'] == '✗ FAIL'])
        total = len(df)
        
        print(f"\n{'='*120}")
        print(f"TEST RESULTS SUMMARY")
        print(f"{'='*120}")
        print(f"Total tests: {total}")
        print(f"Passed: {passed} ({passed/total*100:.1f}%)")
        print(f"Failed: {failed} ({failed/total*100:.1f}%)")
        print(f"{'='*120}\n")
        print(df.to_string(index=False))
        print(f"\n{'='*120}\n")

# ========================================================================
# FIXTURES
# ========================================================================

@pytest.fixture
def ground_truth_int_params():
    """
    Define internal parameters with a known Sp4 ground truth value.
    All other parameters are fixed for inference.
    """
    N = 10
    X0 = StraightLine(N)
    return {
        'Sp4': 1.0,           # Ground truth to recover
        'N': 10,            # Fixed
        'k0': 1e13,            # Fixed
        'bool_EI': True,      # Fixed
        'Beta':0,        # Fixed
        'taus_b': [0]*(N-1),  # Fixed
        'tau_s': 0,        # Fixed
        'gamma': 2,         # Fixed
        'n_L': [0,0],            # Fixed
        'm_L': 0,             # Fixed
        'X_0': X0,  # Initial state (fixed for reproducibility)
    }

@pytest.fixture
def ground_truth_ext_params():
    """Define external parameters (fixed during inference)."""
    N = 10
    return {
        "Lambdas": [[0,1e-6]]*N,
        "Zetas": [0]*N,
        "InterpFlow": 0
    }

@pytest.fixture
def ground_truth_ext_flow_params():
    """Define external parameters (fixed during inference)."""
    N = 10
    return {
        "Lambdas": [[0,0]]*N,
        "Zetas": [0]*N,
        "A":1e-6,
        "w0":1e-6,
        "psi":np.pi/2,
    }

@pytest.fixture
def ground_truth_sim_params():
    """
    Define simulation parameters.
    Use BDF method for faster convergence compared to RK45.
    """
    return {
        "T_span": (0.0, 1e3),
        "T_eval": np.linspace(0, 1e3, int(1e2)),
        "method": "BDF",
        "T_sim_max": 300                  # Fast implicit solver
    }

@pytest.fixture
def ground_truth_data(ground_truth_int_params, ground_truth_ext_params, ground_truth_sim_params):
    """
    Generate ground truth synthetic data by simulating ViscoElasticFilament
    with known parameters.
    
    Returns:
        np.ndarray: Simulated trajectory (shape depends on model)
    """
    model = ViscoElasticFilament(
        int_params=ground_truth_int_params,
        ext_params=ground_truth_ext_params,
        sim_params=ground_truth_sim_params
    )
    output = model.simulate_single()

    assert output['value'] is not None, (
        f"Ground truth simulation failed. Full output: {output}"
    )

    return output['value']


@pytest.fixture
def ground_truth_flow_data(ground_truth_int_params, ground_truth_ext_flow_params, ground_truth_sim_params):
    """
    Generate ground truth synthetic data by simulating ViscoElasticFilament
    with known parameters.
    
    Returns:
        np.ndarray: Simulated trajectory (shape depends on model)
    """
    model = ViscoElasticFilament_FlowParams(
        int_params=ground_truth_int_params,
        ext_params=ground_truth_ext_flow_params,
        sim_params=ground_truth_sim_params
    )
    output = model.simulate_single()
    
    assert output['value'] is not None, (
        f"Ground truth simulation failed. Full output: {output}"
    )

    return output['value']

@pytest.fixture
def ground_truth_ext_params_multi():
    """
    Multiple sets of external parameters.
    Each represents a different experimental condition.
    """
    N = 10
    return [
        {
            "Lambdas": [[0, 1e-6]] * N,
            "Zetas": [0] * N,
            "InterpFlow": 0
        },
        {
            "Lambdas": [[0, 2e-6]] * N,
            "Zetas": [1e-3] * N,
            "InterpFlow": 0
        },
        {
            "Lambdas": [[0, 0.5e-6]] * N,
            "Zetas": [0.5e-3] * N,
            "InterpFlow": 1
        },
    ]

@pytest.fixture
def ground_truth_ext_flow_params_multi():
    """
    Multiple sets of external parameters.
    Each represents a different experimental condition.
    """
    N = 10
    return [
        {
            "Lambdas": [[0, 0]] * N,
            "Zetas": [0] * N,
            "A": 1e-6,
            "w0":1e-6,
            "psi":np.pi/2,
        },
        {
            "Lambdas": [[0, 0]] * N,
            "Zetas": [0] * N,
            "A": 1e-6,
            "w0":1e-6,
            "psi":np.pi/2,
        },
        {
            "Lambdas": [[0, 0]] * N,
            "Zetas": [0] * N,            
            "A": 1e-6,
            "w0":1e-6,
            "psi":np.pi/2,
        },
        {
            "Lambdas": [[0, 0]] * N,
            "Zetas": [0] * N,            
            "A": 1e-6,
            "w0":1e-6,
            "psi":np.pi/2,
        },
        {
            "Lambdas": [[0, 0]] * N,
            "Zetas": [0] * N,            
            "A": 1e-6,
            "w0":1e-6,
            "psi":np.pi/2,
        },
        {
            "Lambdas": [[0, 0]] * N,
            "Zetas": [0] * N,            
            "A": 1e-6,
            "w0":1e-6,
            "psi":np.pi/2,
        },
        {
            "Lambdas": [[0, 0]] * N,
            "Zetas": [0] * N,            
            "A": 1e-6,
            "w0":1e-6,
            "psi":np.pi/2,
        },
        {
            "Lambdas": [[0, 0]] * N,
            "Zetas": [0] * N,            
            "A": 1e-6,
            "w0":1e-6,
            "psi":np.pi/2,
        },                                        
        # {
        #     "Lambdas": [[0, 0]] * N,
        #     "Zetas": [0] * N,            
        #     "A": 1e-3,
        #     "w0":1e-6,
        #     "psi":np.pi/2,
        # },        
        # {
        #     "Lambdas": [[0, 0]] * N,
        #     "Zetas": [0] * N,            
        #     "A": 1e-3,
        #     "w0":1e-3,
        #     "psi":np.pi/2,
        # },
        # {
        #     "Lambdas": [[0, 0]] * N,
        #     "Zetas": [0] * N,            
        #     "A": 1e-3,
        #     "w0":1e-3,
        #     "psi":np.pi/2,
        # },                      
    ]    

@pytest.fixture
def ground_truth_sim_params_multi():
    """
    Multiple sets of simulation parameters.
    Each represents different integration settings.
    """
    return [
        {
            "T_span": (0.0, 1e3),
            "T_eval": np.linspace(0, 1e3, int(1e2)),
            "method": "BDF",
            "T_sim_max": 300
        },
    ]


@pytest.fixture
def ground_truth_data_multi(
    ground_truth_int_params,
    ground_truth_ext_params_multi,
    ground_truth_sim_params_multi,
    composed_model_sp4_only,
):
    """
    Generate ground truth data using the model with known parameters
    across multiple external and simulation parameter sets.
    
    Returns a list of ground truth arrays (one per condition).
    """
    ground_truths = []
    
    for ext_params, sim_params in product(
        ground_truth_ext_params_multi,
        ground_truth_sim_params_multi
    ):
        # Instantiate model with ground truth internal parameters
        instance = composed_model_flow_sp4_only( 
            ground_truth_int_params,
            ext_params,
            sim_params
        )
        
        # Simulate to generate ground truth
        sim_result = instance.simulate_single()
        gt_data = sim_result['value']
        
        ground_truths.append(gt_data)
    
    return ground_truths


@pytest.fixture
def ground_truth_flow_data_multi(
    ground_truth_int_params,
    ground_truth_ext_flow_params_multi,
    ground_truth_sim_params_multi,
    composed_model_flow_sp4_only,
):
    """
    Generate ground truth data using the model with known parameters
    across multiple external and simulation parameter sets.
    
    Returns a list of ground truth arrays (one per condition).
    """
    ground_truths = []
    
    for ext_params, sim_params in product(
        ground_truth_ext_flow_params_multi,
        ground_truth_sim_params_multi
    ):
        # Instantiate model with ground truth internal parameters
        instance = composed_model_flow_sp4_only(
            ground_truth_int_params,
            ext_params,
            sim_params
        )
        
        # Simulate to generate ground truth
        sim_result = instance.simulate_single()
        gt_data = sim_result['value']
        
        ground_truths.append(gt_data)
    
    return ground_truths

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

@pytest.fixture
def composed_model_sp4_only(ground_truth_int_params):
    """
    Create a composed model that only varies Sp4.
    
    The compose function takes the reduced int_params {'Sp4': value}
    and embeds it into the full parameter dict with fixed other values.
    """
    # Create a copy of fixed parameters
    fixed_params = ground_truth_int_params.copy()
    
    def embed_sp4(reduced_int_params: Dict[str, float], ext_params: Any, sim_params: Any) -> Dict[str, Any]:
        """
        Transform reduced parameters {'Sp4': x} into full int_params.
        
        Args:
            reduced_int_params: Dict with key 'Sp4' and inferred value
            ext_params, sim_params: Passed through (not used here)
        
        Returns:
            Full int_params dict with Sp4 updated, others fixed
        """
        full_params = fixed_params.copy()
        full_params['Sp4'] = reduced_int_params['Sp4']
        return full_params
    
    # Create composed model with the embedding function
    ComposedModel = compose_model(
        ViscoElasticFilament,
        compose_int_params=embed_sp4
    )
    return ComposedModel

@pytest.fixture
def composed_model_flow_sp4_only(ground_truth_int_params):
    """
    Create a composed model for ViscoElasticFilament_FlowParams that only varies Sp4.
    
    The embedding function accepts a reduced parameter dict {'Sp4': value}
    and embeds it into the full internal parameters, keeping all others fixed.
    """
    fixed_params = ground_truth_int_params.copy()
    
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
    ComposedModel = compose_model(
        ViscoElasticFilament_FlowParams,
        compose_int_params=embed_sp4_flow,
    )
    return ComposedModel


@pytest.fixture
def composed_model_flow_sp4_tau_s_only(ground_truth_int_params):
    """
    Create a composed model for ViscoElasticFilament_FlowParams that only varies Sp4.
    
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
            full_params['tau_s'] = reduced_int_params['tau_']
        
        return full_params
    
    # Create composed model with the embedding function
    ComposedModel = compose_model(
        ViscoElasticFilament_FlowParams,
        compose_int_params=embed_sp4_tau_s_flow,
    )
    return ComposedModel

@pytest.fixture
def basinhopping_optimizer_instance():
    """
    Return the basinhopping optimizer function with standard configuration.
    """
    return basinhopping_optimizer

@pytest.fixture
def optimizer_kwargs():
    return 
    {
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
def inference_instance(
    basinhopping_optimizer_instance,
    optimizer_kwargs,
    composed_model_sp4_only,
    ground_truth_data,
    ground_truth_ext_params,
    ground_truth_sim_params,
    mse_loss_fn,
):
    """
    Create an Inference instance configured for Sp4 parameter recovery.
    
    Uses basin-hopping as the global optimizer with L-BFGS-B for local minimization.
    """
    return Inference(
        model_class=composed_model_sp4_only,
        ground_truths=ground_truth_data,
        loss_fn=mse_loss_fn,
        ext_params_list = ground_truth_ext_params,
        sim_params_list = ground_truth_sim_params,
        optimizer=basinhopping_optimizer_instance,
        optimizer_kwargs=optimizer_kwargs,
        n_jobs = -1,
    )

@pytest.fixture
def inference_instance_flow(
    basinhopping_optimizer_instance,
    optimizer_kwargs,
    composed_model_flow_sp4_only,
    ground_truth_flow_data,
    ground_truth_ext_flow_params,
    ground_truth_sim_params,
    mse_loss_fn,
):
    """
    Create an Inference instance for Sp4 recovery using flow parameters.
    
    Configuration:
    - Global optimizer: basin-hopping (stochastic global search)
    - Local minimizer: L-BFGS-B (bounded quasi-Newton)
    - Bounds: Sp4 in [1e-6, inf)
    - Tolerance: tight convergence criteria for precise parameter recovery
    """
    return Inference(
        model_class=composed_model_flow_sp4_only,
        ground_truths=ground_truth_flow_data,
        loss_fn=mse_loss_fn,
        ext_params_list = ground_truth_ext_flow_params,
        sim_params_list = ground_truth_sim_params,        
        optimizer=basinhopping_optimizer_instance,
        optimizer_kwargs=optimizer_kwargs,
        n_jobs = -1,
    )

@pytest.fixture
def initial_guesses_sp4():
    """
    Fixed set of initial Sp4 guesses.
    Ground truth Sp4 = 1.0
    """
    return [
        {'Sp4': 0.3},
        {'Sp4': 0.8},
        {'Sp4': 1.2},
        {'Sp4': 2.0},
        {'Sp4': 5.0},
    ]

# ========================================================================
# TESTS
# ========================================================================

class TestViscoElasticFilamentSp4Inference:
    """
    Integration tests for inferring the Sp4 parameter of ViscoElasticFilament.
    
    Strategy:
    1. Define ground truth parameters with a known Sp4 value (e.g., 1.0)
    2. Simulate forward to generate synthetic ground truth data
    3. Create a composed model that only varies Sp4 (others fixed)
    4. Run inference to recover the ground truth Sp4
    5. Verify convergence and uncertainty estimates
    """

    def test_inference_sp4_recovery(
        self,
        inference_instance,
        ground_truth_ext_params,
        ground_truth_sim_params,
    ):
        """
        Test that the inference pipeline can recover Sp4 parameter
        from synthetic ground truth data using basin-hopping optimization.
        
        Expected behavior:
        - Inferred Sp4 should be close to ground truth value (1.0)
        - Optimization should converge with reasonable acceptance rate
        """
        result = inference_instance.infer(
            initial_guess={'Sp4': 2.5},
        )
        
        # Print detailed summaries
        print_optimization_history(inference_instance, plot_flag=False)
        print_inference_summary(
            result,
            ground_truth_params={'Sp4': 1.0},
            confidence_level=0.95
        )

        # Extract results
        inferred_sp4 = result.params['Sp4']
        acceptance_rate = np.mean(inference_instance.result.accept_global)
        ground_truth_sp4 = 1.0
        
        # Assertions
        assert inferred_sp4 is not None, "Inference failed to recover Sp4"
        assert inferred_sp4 > 0, "Inferred Sp4 must be positive"
        assert np.isfinite(inferred_sp4), "Inferred Sp4 must be finite"
        
        relative_error = abs(inferred_sp4 - ground_truth_sp4) / ground_truth_sp4
        assert relative_error < 0.1, (
            f"Inferred Sp4={inferred_sp4:.6f} deviates >10% from ground truth {ground_truth_sp4}"
        )
        
        assert acceptance_rate > 0, "No basin-hopping steps accepted"
        
        # Optional: assert on covariance/Hessian if computed
        if result.covariance is not None:
            assert result.covariance.shape == (1, 1), "Covariance shape mismatch"
            assert result.std_errors[0] > 0, "Standard error must be positive"
        
        print(f"✓ Inferred Sp4: {inferred_sp4:.6f} (ground truth: {ground_truth_sp4})")
        print(f"✓ Relative error: {relative_error*100:.2f}%")
        print(f"✓ Basin-hopping acceptance rate: {acceptance_rate:.2%}")

    # def test_inference_sp4_recovery_flow(
    #     self,
    #     inference_instance_flow,
    #     ground_truth_ext_flow_params,
    #     ground_truth_sim_params,
    # ):
    #     """
    #     Test that the inference pipeline recovers Sp4 from synthetic flow data.
        
    #     Validates:
    #     1. Sp4 recovery accuracy (relative error < 10%)
    #     2. Optimization convergence (acceptance rate > 0)
    #     3. Parameter bounds satisfaction (Sp4 > 0, finite)
    #     4. Covariance matrix validity (if computed)
    #     5. External flow parameters remain unchanged
        
    #     Expected behavior:
    #     - Inferred Sp4 ≈ 1.0 (ground truth)
    #     - Relative error < 10%
    #     - Acceptance rate > 0
    #     """
    #     # Run inference with initial guess far from ground truth
    #     result = inference_instance_flow.infer(
    #         initial_guess={'Sp4': 250},
    #         # ext_params=ground_truth_ext_flow_params,
    #         # sim_params=ground_truth_sim_params,
    #     )
        
    #     # Print detailed optimization history and summary
    #     print_optimization_history(inference_instance_flow, plot_flag=False)
    #     print_inference_summary(
    #         result,
    #         ground_truth_params={'Sp4': 1.0},
    #         confidence_level=0.95,
    #     )
        
    #     # Extract inferred results
    #     inferred_sp4 = result.params['Sp4']
    #     acceptance_rate = np.mean(inference_instance_flow.result.accept_global)
    #     ground_truth_sp4 = 1.0
        
    #     # ===== ASSERTION 1: Sp4 successfully inferred =====
    #     assert inferred_sp4 is not None, (
    #         "Inference failed: Sp4 not returned"
    #     )
        
    #     # ===== ASSERTION 2: Sp4 is physically valid =====
    #     assert inferred_sp4 > 0, (
    #         f"Inferred Sp4={inferred_sp4} must be positive"
    #     )
    #     assert np.isfinite(inferred_sp4), (
    #         f"Inferred Sp4={inferred_sp4} must be finite"
    #     )
        
    #     # ===== ASSERTION 3: Relative error within tolerance =====
    #     relative_error = abs(inferred_sp4 - ground_truth_sp4) / ground_truth_sp4
    #     assert relative_error < 0.1, (
    #         f"Inferred Sp4={inferred_sp4:.6f} deviates >{relative_error*100:.2f}% "
    #         f"from ground truth {ground_truth_sp4}"
    #     )
        
    #     # ===== ASSERTION 4: Optimization converged =====
    #     assert acceptance_rate > 0, (
    #         "Basin-hopping failed: no steps accepted"
    #     )
        
    #     # ===== ASSERTION 5: Covariance/uncertainty valid =====
    #     if result.covariance is not None:
    #         assert result.covariance.shape == (1, 1), (
    #             f"Covariance shape {result.covariance.shape} != (1, 1)"
    #         )
    #         assert result.std_errors[0] > 0, (
    #             f"Standard error must be positive, got {result.std_errors[0]}"
    #         )
        
    #     # ===== ASSERTION 6: External flow parameters unchanged =====
    #     # Verify that flow parameters are passed but not modified by optimizer
    #     assert 'A' in ground_truth_ext_flow_params, "Missing A in ext_params"
    #     assert 'w0' in ground_truth_ext_flow_params, "Missing w0 in ext_params"
    #     assert 'psi' in ground_truth_ext_flow_params, "Missing psi in ext_params"
        
    #     print(f"\n{'='*70}")
    #     print(f"FLOW PARAMETER INFERENCE RESULTS (Sp4 recovery)")
    #     print(f"{'='*70}")
    #     print(f"✓ Inferred Sp4:              {inferred_sp4:.6f}")
    #     print(f"✓ Ground truth Sp4:          {ground_truth_sp4}")
    #     print(f"✓ Relative error:            {relative_error*100:.2f}%")
    #     print(f"✓ Basin-hopping acceptance:  {acceptance_rate:.2%}")
    #     print(f"✓ Fixed flow amplitude (A):  {ground_truth_ext_flow_params['A']:.2e}")
    #     print(f"✓ Fixed angular freq (w0):   {ground_truth_ext_flow_params['w0']:.2e} rad/s")
    #     print(f"✓ Fixed flow angle (psi):    {ground_truth_ext_flow_params['psi']:.4f} rad")
    #     print(f"{'='*70}\n")

    # @pytest.mark.parametrize("initial_sp4", [0.1, 0.5, 2.5, 5.0, 10.0])
    # def test_inference_sp4_recovery_flow_robust(
    #     self,
    #     initial_sp4,
    #     composed_model_flow_sp4_only,
    #     ground_truth_flow_data,
    #     mse_loss_fn,
    #     basinhopping_optimizer_instance,
    #     optimizer_kwargs,
    #     ground_truth_ext_flow_params,
    #     ground_truth_sim_params,
    # ):
    #     """
    #     Robustness test: verify Sp4 recovery works across multiple initial guesses.
        
    #     Tests convergence behavior when starting far from and near the ground truth.
    #     """
    #     # Create a fresh inference instance for this initial guess
    #     inference = Inference(
    #         model_class=composed_model_flow_sp4_only,
    #         ground_truths=ground_truth_flow_data,
    #         loss_fn=mse_loss_fn,
    #         ext_params_list=ground_truth_ext_flow_params,
    #         sim_params_list=ground_truth_sim_params,
    #         optimizer=basinhopping_optimizer_instance,
    #         optimizer_kwargs=optimizer_kwargs,
    #         n_jobs=-1,
    #     )
        
    #     # Run inference
    #     result = inference.infer(
    #         initial_guess={'Sp4': initial_sp4},
    #     )
        
    #     # Validate recovery
    #     inferred_sp4 = result.params['Sp4']
    #     ground_truth_sp4 = 1.0
    #     relative_error = abs(inferred_sp4 - ground_truth_sp4) / ground_truth_sp4
        
    #     assert inferred_sp4 > 0, f"Inferred Sp4={inferred_sp4} must be positive"
    #     assert np.isfinite(inferred_sp4), f"Inferred Sp4={inferred_sp4} must be finite"
    #     assert relative_error < 0.2, (  # Slightly relaxed tolerance for robustness test
    #         f"Initial guess {initial_sp4}: "
    #         f"Inferred Sp4={inferred_sp4:.6f} deviates {relative_error*100:.2f}% "
    #         f"from ground truth {ground_truth_sp4}"
    #     )
        
    #     print(f"✓ Initial Sp4: {initial_sp4:>5.1f} → Inferred: {inferred_sp4:.6f} "
    #         f"(error: {relative_error*100:>6.2f}%)")

    # def test_inference_loss_landscape_flow(
    #     self,
    #     composed_model_flow_sp4_only,
    #     ground_truth_flow_data,
    #     mse_loss_fn,
    #     ground_truth_ext_flow_params,
    #     ground_truth_sim_params,
    # ):
    #     """
    #     Diagnostic test: visualize the loss landscape around ground truth Sp4.
        
    #     Validates that:
    #     1. Loss minimum is near ground truth Sp4 = 1.0
    #     2. Loss function is smooth and well-conditioned
    #     3. Gradient direction points toward ground truth from distant starting points
    #     4. No pathological spikes or discontinuities in the landscape
        
    #     This test ensures the optimization problem is well-posed before running
    #     expensive inference procedures.
    #     """
    #     ground_truth_sp4 = 1.0
        
    #     # Sweep over a logarithmic range around ground truth
    #     sp4_values = np.logspace(-1, 1.5, 40)  # 0.1 to 31.623
    #     losses = []
        
    #     print(f"\n{'='*70}")
    #     print(f"Loss Landscape Diagnostic (Ground Truth Sp4={ground_truth_sp4})")
    #     print(f"{'='*70}")
    #     print(f"{'Sp4':>10} | {'Loss':>15} | {'ΔLoss (vs GT)':>15}")
    #     print(f"{'-'*50}")
        
    #     for sp4 in sp4_values:
    #         # Construct reduced parameters
    #         reduced_int_params = {'Sp4': sp4}
            
    #         # Instantiate the composed model
    #         model = composed_model_flow_sp4_only(
    #             int_params=reduced_int_params,
    #             ext_params=ground_truth_ext_flow_params,
    #             sim_params=ground_truth_sim_params,
    #         )
            
    #         # Simulate with current Sp4 value
    #         try:
    #             output = model.simulate_single()
    #             simulated_trajectory = output['value']
                
    #             # Compute loss
    #             loss = mse_loss_fn(simulated_trajectory, ground_truth_flow_data)
    #             losses.append(loss)
                
    #             delta_loss = loss - mse_loss_fn(
    #                 ground_truth_flow_data, ground_truth_flow_data
    #             )
                
    #             if True:# sp4 in [0.1, 0.326, 1.061, 3.455, 11.253]:  # Print key points
    #                 print(f"{sp4:>10.3f} | {loss:>15.6e} | {delta_loss:>15.6e}")
            
    #         except Exception as e:
    #             print(f"⚠ Simulation failed at Sp4={sp4:.3f}: {str(e)[:40]}")
    #             losses.append(np.nan)
        
    #     losses = np.array(losses)
        
    #     # ===== ASSERTION 1: Loss finite and real =====
    #     valid_losses = losses[~np.isnan(losses)]
    #     assert len(valid_losses) > 0, "No valid loss values computed"
    #     assert np.all(valid_losses >= 0), "Loss values must be non-negative"
        
    #     # ===== ASSERTION 2: Minimum near ground truth =====
    #     min_loss_idx = np.nanargmin(losses)
    #     sp4_at_min = sp4_values[min_loss_idx]
    #     min_loss = losses[min_loss_idx]
        
    #     # Allow 50% deviation in Sp4 at minimum (diagnostic test)
    #     relative_sp4_error_at_min = abs(sp4_at_min - ground_truth_sp4) / ground_truth_sp4
    #     assert relative_sp4_error_at_min < 0.5, (
    #         f"Loss minimum at Sp4={sp4_at_min:.3f}, "
    #         f"expected near {ground_truth_sp4} "
    #         f"(relative error: {relative_sp4_error_at_min*100:.1f}%)"
    #     )
        
    #     # ===== ASSERTION 3: Loss landscape is smooth =====
    #     # Check for large discontinuous jumps (indicates numerical issues)
    #     valid_mask = ~np.isnan(losses)
    #     if np.sum(valid_mask) > 2:
    #         valid_losses_subset = losses[valid_mask]
    #         loss_gradients = np.abs(np.diff(valid_losses_subset))
            
    #         # Flag if max gradient is suspiciously large
    #         max_gradient = np.max(loss_gradients)
    #         median_gradient = np.median(loss_gradients)
            
    #         if median_gradient > 0:
    #             gradient_ratio = max_gradient / median_gradient
    #             assert gradient_ratio < 100, (
    #                 f"Loss landscape has discontinuities "
    #                 f"(max gradient {gradient_ratio:.1f}x median)"
    #             )
        
    #     # ===== ASSERTION 4: Loss decreases toward ground truth =====
    #     # Compare loss at 0.5*GT vs 2*GT (both equidistant log-scale from GT)
    #     idx_05 = np.argmin(np.abs(sp4_values - 0.5 * ground_truth_sp4))
    #     idx_20 = np.argmin(np.abs(sp4_values - 2.0 * ground_truth_sp4))
    #     idx_10 = np.argmin(np.abs(sp4_values - ground_truth_sp4))
        
    #     loss_at_05 = losses[idx_05]
    #     loss_at_20 = losses[idx_20]
    #     loss_at_10 = losses[idx_10]
        
    #     if not np.isnan(loss_at_05) and not np.isnan(loss_at_20) and not np.isnan(loss_at_10):
    #         # Both flanks should have higher loss than center
    #         assert loss_at_10 <= loss_at_05 * 1.5, (
    #             f"Loss at Sp4=0.5 ({loss_at_05:.3e}) should be higher "
    #             f"than at Sp4=1.0 ({loss_at_10:.3e})"
    #         )
    #         assert loss_at_10 <= loss_at_20 * 1.5, (
    #             f"Loss at Sp4=2.0 ({loss_at_20:.3e}) should be higher "
    #             f"than at Sp4=1.0 ({loss_at_10:.3e})"
    #         )
        
    #     print(f"{'-'*50}")
    #     print(f"✓ Loss minimum at Sp4={sp4_at_min:.6f}")
    #     print(f"✓ Ground truth Sp4={ground_truth_sp4}")
    #     print(f"✓ Relative error: {relative_sp4_error_at_min*100:.2f}%")
    #     print(f"✓ Minimum loss: {min_loss:.6e}")
    #     print(f"✓ Loss landscape smooth: ✓")
    #     print(f"{'='*70}\n")

    # def test_inference_sp4_recovery_flow_batch(
    #     self,
    #     composed_model_flow_sp4_only,
    #     ground_truth_flow_data,
    #     mse_loss_fn,
    #     basinhopping_optimizer_instance,
    #     optimizer_kwargs,
    #     ground_truth_ext_flow_params,
    #     ground_truth_sim_params,
    # ):
    #     """
    #     Batch test: verify Sp4 recovery works across multiple initial guesses.
        
    #     Tests convergence behavior when starting far from and near the ground truth
    #     using batch inference for efficiency.
    #     """
    #     # Create a single inference instance for batch processing
    #     inference = Inference(
    #         model_class=composed_model_flow_sp4_only,
    #         ground_truths=ground_truth_flow_data,
    #         loss_fn=mse_loss_fn,
    #         ext_params_list=ground_truth_ext_flow_params,
    #         sim_params_list=ground_truth_sim_params,
    #         optimizer=basinhopping_optimizer_instance,
    #         optimizer_kwargs=optimizer_kwargs,
    #         n_jobs=-1,
    #     )
        
    #     # Create initial guesses for all parametrized values
    #     initial_guesses = [{'Sp4': sp4} for sp4 in [0.1, 0.5, 2.5, 5.0, 10.0]]
    #     # Run batch inference
    #     results = inference.infer_batch(initial_guesses)
        
    #     # Validate recovery
    #     for l in range(len(results)):
    #         result = results[l]
    #         initial_sp4 = initial_guesses[l]['Sp4']
    #         inferred_sp4 = result.params['Sp4']
    #         ground_truth_sp4 = 1.0
    #         relative_error = abs(inferred_sp4 - ground_truth_sp4) / ground_truth_sp4
            
    #         assert inferred_sp4 > 0, f"Inferred Sp4={inferred_sp4} must be positive"
    #         assert np.isfinite(inferred_sp4), f"Inferred Sp4={inferred_sp4} must be finite"
    #         assert relative_error < 0.2, (  # Slightly relaxed tolerance for robustness test
    #             f"Initial guess {initial_sp4}: "
    #             f"Inferred Sp4={inferred_sp4:.6f} deviates {relative_error*100:.2f}% "
    #             f"from ground truth {ground_truth_sp4}"
    #         )
            
    #         print(f"✓ Initial Sp4: {initial_sp4:>5.1f} → Inferred: {inferred_sp4:.6f} "
    #             f"(error: {relative_error*100:>6.2f}%)")

    # def test_inference_multi_conditions_robust(
    #     self,
    #     composed_model_flow_sp4_only,
    #     ground_truth_flow_data_multi,
    #     ground_truth_ext_flow_params_multi,
    #     ground_truth_sim_params_multi,
    #     mse_loss_fn,
    #     basinhopping_optimizer_instance,
    #     optimizer_kwargs,
    # ):
    #     """
    #     Robustness test: verify Sp4 recovery across multiple external and 
    #     simulation parameter sets.
        
    #     This test:
    #     1. Generates ground truth data under different conditions
    #     2. Attempts to recover Sp4 using inference with a single initial guess
    #     3. Validates convergence across all conditions simultaneously
    #     """
    #     # Initial guesses to test robustness
    #     initial_guess = {'Sp4': 2.5}
        
    #     # Create inference instance with multiple ground truths and parameters
    #     inference = Inference(
    #         model_class=composed_model_flow_sp4_only,
    #         ground_truths=ground_truth_flow_data_multi,  # List of 3 arrays
    #         loss_fn=mse_loss_fn,
    #         ext_params_list=ground_truth_ext_flow_params_multi,  # List of 3 dicts
    #         sim_params_list=ground_truth_sim_params_multi,  # List of 3 dicts
    #         optimizer=basinhopping_optimizer_instance,
    #         optimizer_kwargs=optimizer_kwargs,
    #         n_jobs=-1,
    #     )
        
    #     # Run batch inference across all initial guesses
    #     result = inference.infer(initial_guess)
        
    #     # Ground truth value to recover
    #     ground_truth_sp4 = 1.0
        
    #     # Validate convergence for each initial guess
    #     print("\n" + "=" * 70)
    #     print("Robustness Test: Sp4 Recovery Across Multiple Conditions")
    #     print("=" * 70)
        
        
    #     initial_sp4 = initial_guess['Sp4']
    #     inferred_sp4 = result.params['Sp4']
    #     relative_error = abs(inferred_sp4 - ground_truth_sp4) / ground_truth_sp4
        
    #     # Assertions
    #     assert inferred_sp4 > 0, (
    #         f"Inferred Sp4={inferred_sp4} must be positive"
    #     )
    #     assert np.isfinite(inferred_sp4), (
    #         f"Inferred Sp4={inferred_sp4} must be finite"
    #     )
    #     assert relative_error < 0.2, (
    #         f"Initial guess {initial_sp4}: "
    #         f"Inferred Sp4={inferred_sp4:.6f} deviates {relative_error*100:.2f}% "
    #         f"from ground truth {ground_truth_sp4}"
    #     )
        
    #     print(f"✓ Initial Sp4: {initial_sp4:>5.1f} → Inferred: {inferred_sp4:.6f} "
    #         f"(error: {relative_error*100:>6.2f}%, loss: {result.loss:.2e})")
    
    #     print("=" * 70)

    # def test_inference_multi_conditions_robust_batch(
    #     self,
    #     composed_model_flow_sp4_only,
    #     ground_truth_flow_data_multi,
    #     ground_truth_ext_flow_params_multi,
    #     ground_truth_sim_params_multi,
    #     mse_loss_fn,
    #     basinhopping_optimizer_instance,
    #     optimizer_kwargs,
    # ):
    #     """
    #     Robustness test: verify Sp4 recovery across multiple external and 
    #     simulation parameter sets.
        
    #     This test:
    #     1. Generates ground truth data under different conditions
    #     2. Attempts to recover Sp4 using batch inference with multiple initial guesses
    #     3. Validates convergence across all conditions simultaneously
    #     """
    #     # Initial guesses to test robustness
    #     initial_guesses = [{'Sp4': sp4} for sp4 in [0.1, 0.5, 2.5, 5.0, 10.0]]
        
    #     # Create inference instance with multiple ground truths and parameters
    #     inference = Inference(
    #         model_class=composed_model_flow_sp4_only,
    #         ground_truths=ground_truth_flow_data_multi,  # List of 3 arrays
    #         loss_fn=mse_loss_fn,
    #         ext_params_list=ground_truth_ext_flow_params_multi,  # List of 3 dicts
    #         sim_params_list=ground_truth_sim_params_multi,  # List of 3 dicts
    #         optimizer=basinhopping_optimizer_instance,
    #         optimizer_kwargs=optimizer_kwargs,
    #         n_jobs=-1,
    #     )
        
    #     # Run batch inference across all initial guesses
    #     results = inference.infer_batch(initial_guesses)
        
    #     # Ground truth value to recover
    #     ground_truth_sp4 = 1.0
        
    #     # Validate convergence for each initial guess
    #     print("\n" + "=" * 70)
    #     print("Robustness Test: Sp4 Recovery Across Multiple Conditions")
    #     print("=" * 70)
        
    #     for initial_guess_dict, result in zip(initial_guesses, results):
    #         initial_sp4 = initial_guess_dict['Sp4']
    #         inferred_sp4 = result.params['Sp4']
    #         relative_error = abs(inferred_sp4 - ground_truth_sp4) / ground_truth_sp4
            
    #         # Assertions
    #         assert inferred_sp4 > 0, (
    #             f"Inferred Sp4={inferred_sp4} must be positive"
    #         )
    #         assert np.isfinite(inferred_sp4), (
    #             f"Inferred Sp4={inferred_sp4} must be finite"
    #         )
    #         assert relative_error < 0.2, (
    #             f"Initial guess {initial_sp4}: "
    #             f"Inferred Sp4={inferred_sp4:.6f} deviates {relative_error*100:.2f}% "
    #             f"from ground truth {ground_truth_sp4}"
    #         )
            
    #         print(f"✓ Initial Sp4: {initial_sp4:>5.1f} → Inferred: {inferred_sp4:.6f} "
    #             f"(error: {relative_error*100:>6.2f}%, loss: {result.loss:.2e})")
        
    #     print("=" * 70)

# ===================================
# ======= MultiPass Fixtures ========
# ===================================

# ===== Define pipeline passes =====

# Pass 2:
# @pytest.fixture
# def pass_2(composed_model_flow_sp4_tau_s_only, ground_truth_flow_data, ground_truth_ext_flow_params, ground_truth_sim_params):
#     """ First pass for the viscoelastic inference. """
#     # 
#     second_pass = PipelinePass(
#         name="tau_s Inference (ViscoElastic Model)",
#         model_class=composed_model_flow_sp4_tau_s_only,
#         ground_truths=[ground_truth_flow_data],
#         ext_params_list=[ground_truth_ext_flow_params],
#         sim_params_list=[ground_truth_sim_params],
#         param_keys_to_infer=['tau_s'],
#         fixed_params={}, # Will be auto-filled from Pass 1 result
#     )

#     return second_pass

class TestViscoElasticFilamentMultiPassInference:

    def test_emtpy_pass(composed_model_flow_sp4_only, ground_truth_flow_data, ground_truth_ext_flow_params, ground_truth_sim_params
    ):

        empty_pass = PipelinePass(
            name="Sp4 Inference (Elastic Model)",
            model_class=composed_model_flow_sp4_only,
            ground_truths=[ground_truth_flow_data],
            ext_params_list=[ground_truth_ext_flow_params],
            sim_params_list=[ground_truth_sim_params],
            param_keys_to_infer=['Sp4'],
            fixed_params={},
        )

        assert type(empty_pass.name) == str

    # Pass 1: Reduced model, infer Sp4 only
    @pytest.fixture
    def pass_1(composed_model_flow_sp4_only, ground_truth_flow_data, ground_truth_ext_flow_params, ground_truth_sim_params):
        """ First pass for the viscoelastic inference. """
        
        first_pass = PipelinePass(
            name="Sp4 Inference (Elastic Model)",
            model_class=composed_model_flow_sp4_only,
            ground_truths=[ground_truth_flow_data],
            ext_params_list=[ground_truth_ext_flow_params],
            sim_params_list=[ground_truth_sim_params],
            param_keys_to_infer=['Sp4'],
            fixed_params={},
        )

        return first_pass

    def test_pass_1(pass_1):
        assert type(pass_1.name) == str

    # def test_onepass_inference_sp4model(
    #     pass_1,
    #     mse_loss_fn,
    #     basinhopping_optimizer_instance,
    #     optimizer_kwargs,
    # ):
    #     """
    #     Test single-pass inference: infer Sp4 using reduced (elastic) model.
        
    #     Verifies:
    #     - Pipeline executes without error
    #     - Optimization converges
    #     - Inferred Sp4 is within reasonable bounds
    #     - Uncertainties are computed
    #     - Loss is acceptable
    #     """

    #     assert type(pass_1) == str
        
    #     # ===== ARRANGE =====
    #     pipeline = InferencePipeline(
    #         passes=[pass_1],
    #         loss_fn=mse_loss_fn,
    #         n_jobs_per_pass=-1,  # Use all cores within each pass
    #         optimizer=basinhopping_optimizer_instance,
    #         optimizer_kwargs=optimizer_kwargs,
    #     )
        
    #     # Multiple initial guesses for robustness (Sp4 is typically 0.01 to 10 Pa·s)
    #     initial_guesses_per_pass = [
    #         [
    #             {'Sp4': 0.1},
    #             {'Sp4': 1.0},
    #             {'Sp4': 5.0},
    #         ]
    #     ]
        
    #     # ===== ACT =====
    #     results = pipeline.run(initial_guesses_per_pass)
        
    #     # ===== ASSERT =====
        
    #     # Check that we got exactly one result (one pass)
    #     assert len(results) == 1, "Expected one InferenceResult for one-pass pipeline"
        
    #     result = results[0]
        
    #     # 1. Check convergence
    #     assert result.success, (
    #         f"Optimization did not converge. Message: {result.message}"
    #     )
    #     assert result.iterations > 0, "No iterations were performed"
        
    #     # 2. Check inferred parameter exists and is physically reasonable
    #     assert 'Sp4' in result.params, "Sp4 not in inferred parameters"
    #     sp4_inferred = result.params['Sp4']
    #     assert sp4_inferred > 0, f"Sp4 must be positive, got {sp4_inferred}"
    #     assert sp4_inferred < 1e6, f"Sp4 unreasonably large: {sp4_inferred}"
        
    #     # 3. Check loss is acceptable (adjust threshold based on your problem)
    #     assert result.loss > 0, "Loss should be positive"
    #     assert result.loss < 1e2, (  # Adjust threshold based on ground truth scale
    #         f"Loss suspiciously high: {result.loss}. Check model/data scale."
    #     )
        
    #     # 4. Check uncertainty quantification
    #     assert result.covariance is not None, "Covariance not computed"
    #     assert result.hessian is not None, "Hessian not computed"
    #     assert result.std_errors is not None, "Standard errors not computed"
    #     assert len(result.std_errors) == 1, "Expected 1 std_error for 1 parameter"
        
    #     std_err_sp4 = result.std_errors[0]
    #     assert std_err_sp4 > 0, f"Std error must be positive, got {std_err_sp4}"
    #     assert std_err_sp4 < sp4_inferred * 10, (
    #         f"Std error unreasonably large relative to parameter: "
    #         f"Sp4={sp4_inferred:.3e} ± {std_err_sp4:.3e}"
    #     )
        
    #     # 5. Check parameter trajectory
    #     trajectory = pipeline.get_parameter_trajectory()
    #     assert 'Sp4' in trajectory, "Sp4 not in trajectory"
    #     assert len(trajectory['Sp4']) == 1, "Trajectory should have 1 step"
    #     assert trajectory['Sp4'][0] == sp4_inferred, "Trajectory doesn't match final result"
        
    #     # 6. Print summary for inspection
    #     summary = pipeline.summary()
    #     print(summary)
    #     assert summary, "Summary generation failed"
        
    #     # ===== DIAGNOSTIC OUTPUT =====
    #     print(f"\n✓ Sp4 inferred: {sp4_inferred:.6e} ± {std_err_sp4:.6e}")
    #     print(f"✓ Final loss: {result.loss:.8e}")
    #     print(f"✓ Iterations: {result.iterations}")
    #     print(f"✓ Covariance rank: {np.linalg.matrix_rank(result.covariance)}")

if __name__ == "__main__":
    
    pytest.main([
        __file__,
        "-vv",
        "--tb=short",
        "-s",
    ])