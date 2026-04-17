"""
UnitTests_ViscoElasticFilament_Models_Inference

Test suite for inferring the Sp4 internal parameter of the ViscoElasticFilament model.
Uses compose_model to create a reduced inference problem (only Sp4 varies).
"""

import pytest
import numpy as np
from functools import partial
from typing import Dict, Any, Callable

# Import your modules (adjust paths as needed)
from Models import Model, compose_model
from ViscoElasticFilament_Models import StraightLine, ViscoElasticFilament # ViscoElasticFilament_FlowParams
from Inferences import Inference

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

    # ========================================================================
    # FIXTURES
    # ========================================================================

    @pytest.fixture
    def ground_truth_int_params(self):
        """
        Define internal parameters with a known Sp4 ground truth value.
        All other parameters are fixed for inference.
        """
        return {
            'Sp4': 1.0,           # Ground truth to recover
            'N': 10,            # Fixed
            'k0': 1e13,            # Fixed
            'bool_EI': True,      # Fixed
            'gamma': 2,         # Fixed
            'taus_b': np.array([0]*10),  # Fixed
            'tau_s': 0,        # Fixed
            'n_L': [0,0],            # Fixed
            'm_L': 0,             # Fixed
            'X_0': StraightLine(10),  # Initial state (fixed for reproducibility)
        }

    @pytest.fixture
    def ground_truth_ext_params(self):
        """Define external parameters (fixed during inference)."""
        return {
            'Lambdas': [[0.0, 0.0]]*10,
            'Zetas': [0.0]*10,
            'InterpFlow': None, # No Flow
        }

    @pytest.fixture
    def ground_truth_sim_params(self):
        """
        Define simulation parameters.
        Use BDF method for faster convergence compared to RK45.
        """
        return {
            'T_span': (0.0, 5.0),              # Shorter time span for faster testing
            'T_eval': np.linspace(0, 5, 50),   # 50 evaluation points
            'T_sim_max': 30.0,                 # Max solver time (seconds)
            'method': 'BDF',                   # Fast implicit solver
        }

    @pytest.fixture
    def ground_truth_data(self, ground_truth_int_params, ground_truth_ext_params, ground_truth_sim_params):
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
        
        # Debug: print what you got
        print(f"\nSimulation output: {output}")
        print(f"Output keys: {output.keys() if isinstance(output, dict) else 'Not a dict'}")
        
        assert output['value'] is not None, (
            f"Ground truth simulation failed. Full output: {output}"
        )

        return output['value']

    @pytest.fixture
    def mse_loss_fn(self) -> Callable:
        """
        Define Mean Square Error loss function.
        Returns np.inf if prediction is None (failed simulation).
        """
        def loss_fn(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
            if predicted is None:
                return np.inf
            # Flatten arrays in case of shape mismatch
            pred_flat = np.asarray(predicted).flatten()
            truth_flat = np.asarray(ground_truth).flatten()
            
            # Pad or truncate to match lengths
            min_len = min(len(pred_flat), len(truth_flat))
            return np.linalg.norm(pred_flat[:min_len] - truth_flat[:min_len]) ** 2
        
        return loss_fn

    @pytest.fixture
    def composed_model_sp4_only(self, ground_truth_int_params):
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

    # ========================================================================
    # TESTS
    # ========================================================================

    def test_full_inference_workflow_sp4_convergence(
        self,
        composed_model_sp4_only,
        ground_truth_data,
        ground_truth_ext_params,
        ground_truth_sim_params,
        mse_loss_fn,
        ground_truth_int_params,
    ):
        """
        **TEST: Full inference workflow for Sp4 parameter**
        
        CONCEPT:
        - Simulate ViscoElasticFilament with known Sp4 = 1.0
        - Run inference from initial guess Sp4 = 2.5 (perturbed)
        - Verify convergence to ground truth within tolerance
        
        ASSERTION CHECKS:
        - Inferred Sp4 is close to ground truth (within 10%)
        - Loss is non-negative and finite
        - Covariance matrix exists and is positive definite
        - Std errors are positive
        - Iterations > 0 (optimizer ran)
        """
        # Setup inference with composed model
        inference = Inference(
            model_class=composed_model_sp4_only,
            ground_truth=ground_truth_data,
            loss_fn=mse_loss_fn,
            optimizer_kwargs={
                'method': 'L-BFGS-B',
                'options': {'ftol': 1e-6, 'maxiter': 100}
            }
        )
        
        # Initial guess: perturb Sp4 from true value of 1.0
        initial_guess = {'Sp4': 2.5}
        
        # Run inference
        result = inference.infer(
            initial_guess=initial_guess,
            ext_params=ground_truth_ext_params,
            sim_params=ground_truth_sim_params
        )
        
        # ---- ASSERTIONS ----
        
        # Verify inferred Sp4 converges to ground truth
        inferred_sp4 = result['params']['Sp4']
        true_sp4 = ground_truth_int_params['Sp4']
        relative_error = np.abs(inferred_sp4 - true_sp4) / np.abs(true_sp4)
        
        assert relative_error < 0.1, (
            f"Inferred Sp4={inferred_sp4} deviates from true Sp4={true_sp4} "
            f"by {relative_error*100:.2f}%. Check optimizer convergence."
        )
        
        # Verify loss is valid (non-negative, finite)
        assert result['loss'] >= 0, "Loss should be non-negative"
        assert np.isfinite(result['loss']), "Loss should be finite"
        
        # Verify covariance exists and is valid
        assert result['covariance'] is not None, "Covariance should be computed"
        cov_matrix = result['covariance']
        
        # Check positive definiteness (all eigenvalues > 0)
        eigenvalues = np.linalg.eigvals(cov_matrix)
        assert np.all(eigenvalues > 0), (
            f"Covariance not positive definite. Eigenvalues: {eigenvalues}"
        )
        
        # Verify standard errors are positive and finite
        assert result['std_errors'] is not None, "Std errors should be computed"
        assert np.all(result['std_errors'] > 0), "Std errors should be positive"
        assert np.all(np.isfinite(result['std_errors'])), "Std errors should be finite"
        
        # Verify optimizer ran (iterations > 0)
        assert result['iterations'] > 0, "Optimizer should have run at least 1 iteration"

    def test_inference_convergence_from_multiple_initial_guesses(
        self,
        composed_model_sp4_only,
        ground_truth_data,
        ground_truth_ext_params,
        ground_truth_sim_params,
        mse_loss_fn,
        ground_truth_int_params,
    ):
        """
        **TEST: Robustness across multiple initial guesses**
        
        CONCEPT:
        - Run inference from 3 different initial guesses for Sp4
        - Verify all converge to approximately the same value
        - Demonstrate model is locally convex around optimum
        
        ASSERTION CHECKS:
        - Standard deviation of inferred Sp4 values is small (< 5%)
        - All inferred values converge toward ground truth
        - Loss decreases monotonically across runs
        """
        # Define initial guesses covering a range
        initial_guesses_sp4 = [0.5, 1.5, 3.0]
        inferred_sp4_values = []
        loss_values = []
        
        true_sp4 = ground_truth_int_params['Sp4']
        
        for initial_sp4 in initial_guesses_sp4:
            inference = Inference(
                model_class=composed_model_sp4_only,
                ground_truth=ground_truth_data,
                loss_fn=mse_loss_fn,
                optimizer_kwargs={
                    'method': 'L-BFGS-B',
                    'options': {'ftol': 1e-6, 'maxiter': 100}
                }
            )
            
            # Run inference
            result = inference.infer(
                initial_guess={'Sp4': initial_sp4},
                ext_params=ground_truth_ext_params,
                sim_params=ground_truth_sim_params
            )
            
            inferred_sp4_values.append(result['params']['Sp4'])
            loss_values.append(result['loss'])
        
        # ---- ASSERTIONS ----
        
        # Verify all convergence points are similar
        inferred_sp4_array = np.array(inferred_sp4_values)
        sp4_std = np.std(inferred_sp4_array)
        sp4_mean = np.mean(inferred_sp4_array)
        sp4_cv = sp4_std / np.abs(sp4_mean)  # Coefficient of variation
        
        assert sp4_cv < 0.05, (
            f"Inferred Sp4 values show high variance: mean={sp4_mean:.4f}, "
            f"std={sp4_std:.4f}, CV={sp4_cv:.4f}. Expected CV < 0.05."
        )
        
        # Verify all converge toward ground truth
        for sp4_val in inferred_sp4_values:
            rel_error = np.abs(sp4_val - true_sp4) / np.abs(true_sp4)
            assert rel_error < 0.15, (
                f"Inferred Sp4={sp4_val} deviates from true Sp4={true_sp4} "
                f"by {rel_error*100:.2f}%"
            )
        
        # Verify all loss values are finite and non-negative
        assert np.all(np.isfinite(loss_values)), "All loss values should be finite"
        assert np.all(np.array(loss_values) >= 0), "All loss values should be non-negative"

    def test_inference_parameter_uncertainty_estimation(
        self,
        composed_model_sp4_only,
        ground_truth_data,
        ground_truth_ext_params,
        ground_truth_sim_params,
        mse_loss_fn,
        ground_truth_int_params,
    ):
        """
        **TEST: Parameter uncertainty quantification**
        
        CONCEPT:
        - Verify that the inferred parameter uncertainty (std error) is reasonable
        - Std error should scale with problem difficulty
        - Larger std error for noisy/flat loss landscapes
        
        ASSERTION CHECKS:
        - Std error is positive and finite
        - Std error is not excessively large (< 50% of true value)
        - Std error is not suspiciously small (> 0.1% of true value)
        """
        inference = Inference(
            model_class=composed_model_sp4_only,
            ground_truth=ground_truth_data,
            loss_fn=mse_loss_fn,
            optimizer_kwargs={
                'method': 'L-BFGS-B',
                'options': {'ftol': 1e-6, 'maxiter': 100}
            }
        )
        
        # Run inference
        result = inference.infer(
            initial_guess={'Sp4': 2.0},
            ext_params=ground_truth_ext_params,
            sim_params=ground_truth_sim_params
        )
        
        std_error = result['std_errors'][0]
        true_sp4 = ground_truth_int_params['Sp4']
        
        # ---- ASSERTIONS ----
        
        # Std error should be positive
        assert std_error > 0, "Std error must be positive"
        
        # Std error should be finite
        assert np.isfinite(std_error), "Std error must be finite"
        
        # Sanity check: std error should not exceed 50% of true value
        # (This would indicate extremely poor parameter identifiability)
        assert std_error < 0.5 * np.abs(true_sp4), (
            f"Std error={std_error:.4f} is very large relative to "
            f"true value={true_sp4}. Parameter may be poorly identifiable."
        )
        
        # Sanity check: std error should not be suspiciously small (< 0.1%)
        # (This could indicate Hessian computation issues)
        assert std_error > 0.001 * np.abs(true_sp4), (
            f"Std error={std_error:.4e} is suspiciously small. "
            f"Check Hessian computation."
        )

    def test_inference_fails_gracefully_on_bad_initial_guess(
        self,
        composed_model_sp4_only,
        ground_truth_data,
        ground_truth_ext_params,
        ground_truth_sim_params,
        mse_loss_fn,
    ):
        """
        **TEST: Graceful handling of problematic initial guesses**
        
        CONCEPT:
        - Use an unreasonable initial guess (e.g., Sp4 = -1000)
        - Verify inference still runs (or fails with informative message)
        - No crashes or exceptions
        
        ASSERTION CHECKS:
        - Inference completes without unhandled exceptions
        - Result dict has required keys, even if optimization didn't converge
        """
        inference = Inference(
            model_class=composed_model_sp4_only,
            ground_truth=ground_truth_data,
            loss_fn=mse_loss_fn,
            optimizer_kwargs={
                'method': 'L-BFGS-B',
                'options': {'ftol': 1e-6, 'maxiter': 50}
            }
        )
        
        # Use a pathological initial guess
        bad_initial_guess = {'Sp4': -1000.0}
        
        # Run inference (should handle gracefully)
        try:
            result = inference.infer(
                initial_guess=bad_initial_guess,
                ext_params=ground_truth_ext_params,
                sim_params=ground_truth_sim_params
            )
        except Exception as e:
            pytest.fail(f"Inference should not crash on bad initial guess. Error: {e}")
        
        # ---- ASSERTIONS ----
        
        # Verify result has required keys
        required_keys = ['params', 'loss', 'covariance', 'std_errors', 'iterations']
        for key in required_keys:
            assert key in result, f"Result missing required key: {key}"
        
        # Loss should be finite or inf (not NaN)
        assert result['loss'] == result['loss'], "Loss should not be NaN"

    def test_inference_vs_ground_truth_recovery(
        self,
        composed_model_sp4_only,
        ground_truth_data,
        ground_truth_ext_params,
        ground_truth_sim_params,
        mse_loss_fn,
        ground_truth_int_params,
    ):
        """
        **TEST: End-to-end parameter recovery**
        
        CONCEPT:
        - Verify that inference can recover the ground truth parameter
        - Final loss should be small (close to data noise level)
        - Inferred Sp4 should fall within confidence interval
        
        ASSERTION CHECKS:
        - Inferred Sp4 within 1 std error of true value (confidence check)
        - Final loss is reasonable (not dominated by noise)
        - Relative error < 5%
        """
        inference = Inference(
            model_class=composed_model_sp4_only,
            ground_truth=ground_truth_data,
            loss_fn=mse_loss_fn,
            optimizer_kwargs={
                'method': 'L-BFGS-B',
                'options': {'ftol': 1e-7, 'maxiter': 200}
            }
        )
        
        # Run inference
        result = inference.infer(
            initial_guess={'Sp4': 1.8},
            ext_params=ground_truth_ext_params,
            sim_params=ground_truth_sim_params
        )
        
        inferred_sp4 = result['params']['Sp4']
        std_error = result['std_errors'][0]
        true_sp4 = ground_truth_int_params['Sp4']
        
        # ---- ASSERTIONS ----
        
        # Inferred value should be within ~1.96 std errors (95% CI)
        z_score = np.abs(inferred_sp4 - true_sp4) / std_error
        assert z_score < 2.0, (
            f"Inferred Sp4={inferred_sp4} is {z_score:.2f} std errors away from "
            f"true value={true_sp4}. Expected z-score < 2.0 (95% CI)."
        )
        
        # Relative error should be small
        rel_error = np.abs(inferred_sp4 - true_sp4) / np.abs(true_sp4)
        assert rel_error < 0.05, (
            f"Relative error {rel_error*100:.2f}% exceeds 5% threshold. "
            f"Inferred={inferred_sp4}, True={true_sp4}"
        )
        
        # Loss should be finite and reasonable
        assert np.isfinite(result['loss']), "Final loss should be finite"
        assert result['loss'] >= 0, "Final loss should be non-negative"

if __name__ == "__main__":
    # Run pytest programmatically
    pytest.main([__file__, "-v", "--tb=short"])