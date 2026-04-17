# UnitTests_Inference.py
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from functools import partial
import logging

from Inferences import Inference, BatchInference
from Models import Model, Square

# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def mse_loss():
    """Mean Squared Error loss function."""
    return lambda pred, truth: np.mean((pred - truth) ** 2)


@pytest.fixture
def true_x():
    """Ground truth parameter value."""
    return 3.5


@pytest.fixture
def ground_truth(true_x):
    """Generate ground truth from known parameter (Square: x^2)."""
    model = Square(int_params={'x': true_x}, ext_params=None, sim_params=None)
    return model.simulate_single()['value']


@pytest.fixture
def inference_instance(ground_truth, mse_loss):
    """Inference object ready for testing."""
    return Inference(
        model_class=Square,
        ground_truth=ground_truth,
        loss_fn=mse_loss,
        optimizer_kwargs={'method': 'L-BFGS-B', 'options': {'ftol': 1e-9, 'maxiter': 1000}}
    )


@pytest.fixture
def initial_guess_dict():
    """Initial parameter guess for optimization."""
    return {'x': 2.0}


@pytest.fixture
def batch_initial_guesses():
    """Multiple initial guesses for batch inference."""
    return [{'x': 1.0}, {'x': 2.0}, {'x': 3.0}, {'x': 4.5}]


# ============================================================================
# TESTS: Initialization
# ============================================================================

class TestInferenceInitialization:
    """Test suite for Inference.__init__()"""

    def test_init_default_optimizer_kwargs(self, mse_loss, ground_truth):
        """Test Inference initializes with default optimizer kwargs."""
        inference = Inference(
            model_class=Square,
            ground_truth=ground_truth,
            loss_fn=mse_loss
        )
        
        assert inference.optimizer_kwargs['method'] == 'L-BFGS-B'
        assert 'ftol' in inference.optimizer_kwargs['options']
        assert 'maxiter' in inference.optimizer_kwargs['options']

    def test_init_custom_optimizer_kwargs(self, mse_loss, ground_truth):
        """Test Inference accepts and stores custom optimizer kwargs."""
        custom_kwargs = {
            'method': 'Nelder-Mead',
            'options': {'xatol': 1e-5, 'fatol': 1e-5}
        }
        inference = Inference(
            model_class=Square,
            ground_truth=ground_truth,
            loss_fn=mse_loss,
            optimizer_kwargs=custom_kwargs
        )
        
        assert inference.optimizer_kwargs['method'] == 'Nelder-Mead'
        assert inference.optimizer_kwargs['options']['xatol'] == 1e-5

    def test_init_attributes_stored(self, mse_loss, ground_truth):
        """Test all initialization attributes are correctly stored."""
        inference = Inference(
            model_class=Square,
            ground_truth=ground_truth,
            loss_fn=mse_loss
        )
        
        assert inference.model_class == Square
        np.testing.assert_array_equal(inference.ground_truth, ground_truth)
        assert inference.loss_fn == mse_loss
        assert inference.result is None
        assert inference.hessian is None
        assert inference.covariance is None


# ============================================================================
# TESTS: Objective Function
# ============================================================================

class TestObjectiveFunction:
    """Test suite for Inference.objective()"""

    def test_objective_returns_scalar(self, inference_instance, initial_guess_dict):
        """Test objective() returns a scalar loss value."""
        param_vector = np.array([initial_guess_dict['x']])
        param_keys = ('x',)
        
        loss = inference_instance.objective(
            param_vector=param_vector,
            param_keys=param_keys,
            ext_params=None,
            sim_params=None
        )
        
        assert isinstance(loss, (float, np.floating))
        assert np.isscalar(loss) or loss.shape == ()

    def test_objective_loss_computation_exact_match(self, mse_loss, true_x):
        """Test objective() computes loss correctly with known parameters."""
        ground_truth = true_x ** 2  # For Square model
        
        inference = Inference(
            model_class=Square,
            ground_truth=ground_truth,
            loss_fn=mse_loss
        )
        
        # When param_vector matches true_x, loss should be near zero
        param_vector = np.array([true_x])
        loss = inference.objective(
            param_vector=param_vector,
            param_keys=('x',),
            ext_params=None,
            sim_params=None
        )
        
        assert loss < 1e-6  # Should be ~0 (or very small due to numeric precision)

    def test_objective_loss_increases_with_distance(self, inference_instance):
        """Test loss increases as parameters move away from optimum."""
        param_keys = ('x',)
        
        loss_near = inference_instance.objective(
            param_vector=np.array([3.4]),
            param_keys=param_keys,
            ext_params=None,
            sim_params=None
        )
        
        loss_far = inference_instance.objective(
            param_vector=np.array([1.0]),
            param_keys=param_keys,
            ext_params=None,
            sim_params=None
        )
        
        assert loss_far > loss_near

    @pytest.mark.parametrize("x_val", [0.5, 1.5, 2.5, 5.0])
    def test_objective_multiple_parameter_values(self, inference_instance, x_val):
        """Test objective() with multiple parameter values (parametrized)."""
        param_vector = np.array([x_val])
        
        loss = inference_instance.objective(
            param_vector=param_vector,
            param_keys=('x',),
            ext_params=None,
            sim_params=None
        )
        
        # Loss should be non-negative
        assert loss >= 0.0
        # Loss should be finite
        assert np.isfinite(loss)


# ============================================================================
# TESTS: Parameter Inference (infer method)
# ============================================================================

class TestParameterInference:
    """Test suite for Inference.infer()"""

    def test_infer_basic_execution(self, inference_instance, initial_guess_dict):
        """Test basic infer() call completes without error."""
        result = inference_instance.infer(
            initial_guess=initial_guess_dict,
            ext_params=None,
            sim_params=None
        )
        
        assert result is not None
        assert isinstance(result, dict)

    def test_infer_result_structure(self, inference_instance, initial_guess_dict):
        """Test infer() returns all expected keys."""
        result = inference_instance.infer(
            initial_guess=initial_guess_dict,
            ext_params=None,
            sim_params=None
        )
        
        required_keys = {'params', 'loss', 'covariance', 'std_errors', 'iterations'}
        assert required_keys.issubset(result.keys())

    def test_infer_params_dict_structure(self, inference_instance, initial_guess_dict):
        """Test result['params'] is a dict with correct keys."""
        result = inference_instance.infer(
            initial_guess=initial_guess_dict,
            ext_params=None,
            sim_params=None
        )
        
        assert isinstance(result['params'], dict)
        assert 'x' in result['params']
        assert isinstance(result['params']['x'], (float, np.floating))

    def test_infer_loss_is_non_negative(self, inference_instance, initial_guess_dict):
        """Test final loss is non-negative."""
        result = inference_instance.infer(
            initial_guess=initial_guess_dict,
            ext_params=None,
            sim_params=None
        )
        
        assert result['loss'] >= 0.0
        assert np.isfinite(result['loss'])

    def test_infer_iterations_positive(self, inference_instance, initial_guess_dict):
        """Test iterations count is positive."""
        result = inference_instance.infer(
            initial_guess=initial_guess_dict,
            ext_params=None,
            sim_params=None
        )
        
        assert result['iterations'] > 0

    @pytest.mark.parametrize("initial_x", [0.5, 1.5, 2.5, 4.0, 5.5])
    def test_infer_multiple_initial_guesses(self, inference_instance, initial_x, true_x):
        """Test inference converges with different starting points (parametrized)."""
        result = inference_instance.infer(
            initial_guess={'x': initial_x},
            ext_params=None,
            sim_params=None
        )
        
        # Inferred x should be close to true_x
        inferred_x = result['params']['x']
        assert np.abs(inferred_x - true_x) < 0.1  # Within 0.1 of true value

    def test_infer_with_none_params(self, inference_instance, initial_guess_dict):
        """Test infer() handles None for ext_params and sim_params."""
        result = inference_instance.infer(
            initial_guess=initial_guess_dict,
            ext_params=None,
            sim_params=None
        )
        
        assert result is not None
        assert 'params' in result

    def test_infer_std_errors_non_negative(self, inference_instance, initial_guess_dict):
        """Test standard errors are non-negative."""
        result = inference_instance.infer(
            initial_guess=initial_guess_dict,
            ext_params=None,
            sim_params=None
        )
        
        if result['std_errors'] is not None:
            assert np.all(result['std_errors'] >= 0.0)

    def test_infer_parameter_recovery(self, true_x, mse_loss):
        """Test inference can recover ground truth parameter accurately."""
        ground_truth = true_x ** 2
        
        inference = Inference(
            model_class=Square,
            ground_truth=ground_truth,
            loss_fn=mse_loss,
            optimizer_kwargs={'method': 'L-BFGS-B', 'options': {'ftol': 1e-10}}
        )
        
        result = inference.infer(
            initial_guess={'x': 2.0},
            ext_params=None,
            sim_params=None
        )
        
        recovered_x = result['params']['x']
        
        # Should recover within 5% of true value
        relative_error = np.abs(recovered_x - true_x) / np.abs(true_x)
        assert relative_error < 0.05


# ============================================================================
# TESTS: Hessian & Covariance
# ============================================================================

class TestHessianAndCovariance:
    """Test suite for Hessian computation and covariance estimation"""

    def test_hessian_computed_after_infer(self, inference_instance, initial_guess_dict):
        """Test Hessian is computed after infer()."""
        result = inference_instance.infer(
            initial_guess=initial_guess_dict,
            ext_params=None,
            sim_params=None
        )
        
        assert inference_instance.hessian is not None

    def test_hessian_is_square_matrix(self, inference_instance, initial_guess_dict):
        """Test Hessian is a square matrix."""
        inference_instance.infer(
            initial_guess=initial_guess_dict,
            ext_params=None,
            sim_params=None
        )
        
        n_params = len(initial_guess_dict)
        assert inference_instance.hessian.shape == (n_params, n_params)

    def test_hessian_is_symmetric(self, inference_instance, initial_guess_dict):
        """Test Hessian is approximately symmetric."""
        inference_instance.infer(
            initial_guess=initial_guess_dict,
            ext_params=None,
            sim_params=None
        )
        
        # Check symmetry: H ≈ H^T
        np.testing.assert_array_almost_equal(
            inference_instance.hessian,
            inference_instance.hessian.T,
            decimal=3
        )

    def test_covariance_computed_after_infer(self, inference_instance, initial_guess_dict):
        """Test covariance is computed after infer()."""
        result = inference_instance.infer(
            initial_guess=initial_guess_dict,
            ext_params=None,
            sim_params=None
        )
        
        assert inference_instance.covariance is not None

    def test_covariance_is_inverse_of_hessian(self, inference_instance, initial_guess_dict):
        """Test covariance ≈ inv(Hessian)."""
        inference_instance.infer(
            initial_guess=initial_guess_dict,
            ext_params=None,
            sim_params=None
        )
        
        # Verify: Hessian @ Covariance ≈ I
        product = inference_instance.hessian @ inference_instance.covariance
        np.testing.assert_array_almost_equal(
            product,
            np.eye(product.shape[0]),
            decimal=3
        )

    def test_covariance_positive_definite(self, inference_instance, initial_guess_dict):
        """Test covariance matrix is positive definite."""
        inference_instance.infer(
            initial_guess=initial_guess_dict,
            ext_params=None,
            sim_params=None
        )
        
        # All eigenvalues should be positive
        eigenvalues = np.linalg.eigvalsh(inference_instance.covariance)
        assert np.all(eigenvalues > 0.0)

    def test_std_errors_from_covariance_diagonal(self, inference_instance, initial_guess_dict):
        """Test std_errors are sqrt of covariance diagonal."""
        result = inference_instance.infer(
            initial_guess=initial_guess_dict,
            ext_params=None,
            sim_params=None
        )
        
        expected_std_errors = np.sqrt(np.diag(inference_instance.covariance))
        np.testing.assert_array_almost_equal(
            result['std_errors'],
            expected_std_errors,
            decimal=10
        )


# ============================================================================
# TESTS: Edge Cases & Error Handling
# ============================================================================

class TestEdgeCasesAndErrorHandling:
    """Test suite for edge cases and error handling"""

    def test_singular_hessian_graceful_handling(self):
        """Test graceful handling when Hessian is singular."""
        # Create a scenario where Hessian might be singular
        # by using a poorly scaled loss function
        def constant_loss(pred, truth):
            return 1.0  # Constant loss -> singular Hessian
        
        inference = Inference(
            model_class=Square,
            ground_truth=12.25,
            loss_fn=constant_loss,
            optimizer_kwargs={'method': 'L-BFGS-B', 'options': {'ftol': 1e-8}}
        )
        
        result = inference.infer(
            initial_guess={'x': 2.0},
            ext_params=None,
            sim_params=None
        )
        
        # Should not crash; covariance may be None
        assert 'params' in result

    def test_infer_with_very_tight_tolerance(self, inference_instance, initial_guess_dict, true_x):
        """Test inference with very tight optimization tolerance."""
        tight_inference = Inference(
            model_class=Square,
            ground_truth=inference_instance.ground_truth,
            loss_fn=inference_instance.loss_fn,
            optimizer_kwargs={'method': 'L-BFGS-B', 'options': {'ftol': 1e-12}}
        )
        
        result = tight_inference.infer(
            initial_guess=initial_guess_dict,
            ext_params=None,
            sim_params=None
        )
        
        # Tighter tolerance should give better parameter recovery
        inferred_x = result['params']['x']
        assert np.abs(inferred_x - true_x) < 0.01  # Very close to true value

    def test_infer_with_max_iterations_limit(self, mse_loss, ground_truth, initial_guess_dict):
        """Test inference respects maxiter constraint."""
        limited_inference = Inference(
            model_class=Square,
            ground_truth=ground_truth,
            loss_fn=mse_loss,
            optimizer_kwargs={'method': 'L-BFGS-B', 'options': {'maxiter': 5}}
        )
        
        result = limited_inference.infer(
            initial_guess=initial_guess_dict,
            ext_params=None,
            sim_params=None
        )
        
        # Should stop within iteration limit
        assert result['iterations'] <= 5

    def test_infer_result_stored_in_instance(self, inference_instance, initial_guess_dict):
        """Test that infer() stores result in self.result."""
        assert inference_instance.result is None  # Initially None
        
        inference_instance.infer(
            initial_guess=initial_guess_dict,
            ext_params=None,
            sim_params=None
        )
        
        assert inference_instance.result is not None
        assert inference_instance.result.success or not inference_instance.result.success  # Has success attr

    def test_objective_with_various_param_dimensions(self, mse_loss, ground_truth):
        """Test objective function handles parameter dict correctly."""
        inference = Inference(
            model_class=Square,
            ground_truth=ground_truth,
            loss_fn=mse_loss
        )
        
        # Test with single parameter
        param_vector = np.array([2.5])
        loss = inference.objective(
            param_vector=param_vector,
            param_keys=('x',),
            ext_params=None,
            sim_params=None
        )
        
        assert np.isfinite(loss)
        assert loss >= 0.0

    def test_infer_loss_decreases_from_initial_guess(self, inference_instance, initial_guess_dict, true_x):
        """Test that optimization decreases loss from initial guess."""
        # Compute initial loss
        initial_loss = inference_instance.objective(
            param_vector=np.array([initial_guess_dict['x']]),
            param_keys=('x',),
            ext_params=None,
            sim_params=None
        )
        
        # Run optimization
        result = inference_instance.infer(
            initial_guess=initial_guess_dict,
            ext_params=None,
            sim_params=None
        )
        
        final_loss = result['loss']
        
        # Final loss should be less than initial loss
        assert final_loss <= initial_loss

    def test_inference_with_zero_ground_truth(self, mse_loss):
        """Test inference when ground truth is zero."""
        ground_truth = 0.0
        
        inference = Inference(
            model_class=Square,
            ground_truth=ground_truth,
            loss_fn=mse_loss
        )
        
        result = inference.infer(
            initial_guess={'x': 2.0},
            ext_params=None,
            sim_params=None
        )
        
        # Should converge to x ≈ 0
        assert np.abs(result['params']['x']) < 0.1


# ============================================================================
# TESTS: Batch Inference
# ============================================================================

class TestBatchInference:
    """Test suite for BatchInference class"""

    def test_batch_inference_init(self, inference_instance):
        """Test BatchInference initializes correctly."""
        batch_inf = BatchInference(inference_instance, n_jobs=-1)
        
        assert batch_inf.inference == inference_instance
        assert batch_inf.n_jobs == -1

    def test_batch_inference_init_n_jobs_single(self, inference_instance):
        """Test BatchInference initializes with single job."""
        batch_inf = BatchInference(inference_instance, n_jobs=1)
        
        assert batch_inf.n_jobs == 1

    def test_batch_inference_parallel_execution(self, inference_instance, batch_initial_guesses):
        """Test batch_inference runs multiple inferences in parallel."""
        batch_inf = BatchInference(inference_instance, n_jobs=-1)
        
        results = batch_inf.infer_batch(
            initial_guesses=batch_initial_guesses,
            ext_params_batch=None,
            sim_params_batch=None
        )
        
        # Should return list of results matching number of initial guesses
        assert len(results) == len(batch_initial_guesses)
        assert all(isinstance(r, dict) for r in results)

    def test_batch_inference_result_structure(self, inference_instance, batch_initial_guesses):
        """Test each batch result has correct structure."""
        batch_inf = BatchInference(inference_instance, n_jobs=-1)
        
        results = batch_inf.infer_batch(
            initial_guesses=batch_initial_guesses,
            ext_params_batch=None,
            sim_params_batch=None
        )
        
        required_keys = {'params', 'loss', 'covariance', 'std_errors', 'iterations'}
        for result in results:
            assert required_keys.issubset(result.keys())

    def test_batch_inference_all_converge(self, inference_instance, batch_initial_guesses, true_x):
        """Test that batch inference converges for all initial guesses."""
        batch_inf = BatchInference(inference_instance, n_jobs=-1)
        
        results = batch_inf.infer_batch(
            initial_guesses=batch_initial_guesses,
            ext_params_batch=None,
            sim_params_batch=None
        )
        
        # All should converge to similar x value
        inferred_xs = [r['params']['x'] for r in results]
        
        # All within reasonable range of true value
        for x in inferred_xs:
            assert np.abs(x - true_x) < 0.15

    def test_batch_inference_with_external_params(self, inference_instance):
        """Test batch inference with external parameters."""
        initial_guesses = [{'x': 1.0}, {'x': 2.0}, {'x': 3.0}]
        ext_params_batch = [None, None, None]  # Could be other values if model uses them
        sim_params_batch = [None, None, None]
        
        batch_inf = BatchInference(inference_instance, n_jobs=-1)
        
        results = batch_inf.infer_batch(
            initial_guesses=initial_guesses,
            ext_params_batch=ext_params_batch,
            sim_params_batch=sim_params_batch
        )
        
        assert len(results) == 3

    @pytest.mark.parametrize("n_jobs", [1, 2, -1])
    def test_batch_inference_various_n_jobs(self, inference_instance, batch_initial_guesses, n_jobs):
        """Test batch inference with different n_jobs settings (parametrized)."""
        batch_inf = BatchInference(inference_instance, n_jobs=n_jobs)
        
        results = batch_inf.infer_batch(
            initial_guesses=batch_initial_guesses,
            ext_params_batch=None,
            sim_params_batch=None
        )
        
        assert len(results) == len(batch_initial_guesses)
        assert all('params' in r for r in results)

    def test_batch_inference_empty_list(self, inference_instance):
        """Test batch inference with empty initial guesses."""
        batch_inf = BatchInference(inference_instance, n_jobs=-1)
        
        results = batch_inf.infer_batch(
            initial_guesses=[],
            ext_params_batch=None,
            sim_params_batch=None
        )
        
        assert results == []

    def test_batch_inference_single_guess(self, inference_instance):
        """Test batch inference with single initial guess."""
        batch_inf = BatchInference(inference_instance, n_jobs=-1)
        
        results = batch_inf.infer_batch(
            initial_guesses=[{'x': 2.5}],
            ext_params_batch=None,
            sim_params_batch=None
        )
        
        assert len(results) == 1
        assert 'params' in results[0]

    def test_batch_inference_consistency_with_sequential(self, inference_instance, batch_initial_guesses):
        """Test batch results match sequential inference."""
        batch_inf = BatchInference(inference_instance, n_jobs=1)  # Sequential
        batch_results = batch_inf.infer_batch(
            initial_guesses=batch_initial_guesses,
            ext_params_batch=None,
            sim_params_batch=None
        )
        
        # Run sequential inference manually
        sequential_results = []
        for ig in batch_initial_guesses:
            result = inference_instance.infer(ig, None, None)
            sequential_results.append(result)
        
        # Compare results (should be very similar)
        for batch_res, seq_res in zip(batch_results, sequential_results):
            np.testing.assert_almost_equal(
                batch_res['params']['x'],
                seq_res['params']['x'],
                decimal=5
            )


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Integration tests combining multiple components"""

    def test_full_inference_workflow(self, mse_loss, true_x):
        """Test complete inference workflow from setup to result."""
        # Setup
        ground_truth = true_x ** 2
        inference = Inference(
            model_class=Square,
            ground_truth=ground_truth,
            loss_fn=mse_loss
        )
        
        # Run inference
        result = inference.infer(
            initial_guess={'x': 1.5},
            ext_params=None,
            sim_params=None
        )
        
        # Verify complete result structure
        assert result['params']['x'] > 0
        assert result['loss'] >= 0
        assert result['covariance'] is not None
        assert result['std_errors'] is not None
        assert result['iterations'] > 0

    def test_batch_and_single_consistency(self, inference_instance, initial_guess_dict):
        """Test that batch inference is consistent with single inference."""
        # Single inference
        single_result = inference_instance.infer(
            initial_guess=initial_guess_dict,
            ext_params=None,
            sim_params=None
        )
        
        # Batch inference (with one guess)
        batch_inf = BatchInference(inference_instance, n_jobs=1)
        batch_results = batch_inf.infer_batch(
            initial_guesses=[initial_guess_dict],
            ext_params_batch=None,
            sim_params_batch=None
        )
        
        # Should be nearly identical
        np.testing.assert_almost_equal(
            single_result['params']['x'],
            batch_results[0]['params']['x'],
            decimal=6
        )

    def test_multiple_runs_convergence_robustness(self, mse_loss, true_x):
        """Test that inference is robust across multiple runs."""
        ground_truth = true_x ** 2
        
        inferred_xs = []
        for i in range(3):
            inference = Inference(
                model_class=Square,
                ground_truth=ground_truth,
                loss_fn=mse_loss
            )
            
            result = inference.infer(
                initial_guess={'x': 1.0 + i * 0.5},
                ext_params=None,
                sim_params=None
            )
            
            inferred_xs.append(result['params']['x'])
        
        # All should converge to similar value
        assert np.std(inferred_xs) < 0.05


# ============================================================================
# FIXTURES FOR PARAMETRIZATION
# ============================================================================

@pytest.fixture(params=[0.5, 1.5, 3.0, 5.0])
def various_true_x(request):
    """Parametrized fixture for different true x values."""
    return request.param


# ============================================================================
# PYTEST CONFIGURATION & MARKERS
# ============================================================================

def test_dense_batch_inference():
    """Test batch inference with larger number of initial guesses."""
    ground_truth = 9.0  # true_x = 3.0
    inference = Inference(
        model_class=Square,
        ground_truth=ground_truth,
        loss_fn=lambda p, t: np.mean((p - t) ** 2)
    )
    
    # 20 initial guesses
    batch_size = 20
    initial_guesses = [{'x': 0.5 + i * 0.3} for i in range(batch_size)]
    
    batch_inf = BatchInference(inference, n_jobs=-1)
    results = batch_inf.infer_batch(initial_guesses)
    
    assert len(results) == batch_size
    assert all('loss' in r for r in results)


if __name__ == "__main__":
    # Run pytest programmatically
    pytest.main([__file__, "-v", "--tb=short"])
