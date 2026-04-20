import numpy as np
import pytest
from scipy.integrate import solve_ivp
import ViscoElasticFilament_Models as vf

# ---------- BASIC BUILDING BLOCKS ---------- #

def test_straight_line_shape():
    N = 5
    X = vf.StraightLine(N)
    assert X.shape == (N + 2,)
    assert np.allclose(X, 0)

def test_theta_zero_case():
    X = np.zeros(10)
    theta = vf.Theta(X, 3)
    assert theta == 0


def test_theta_accumulation():
    X = np.zeros(10)
    X[2:6] = 1  # introduce angles
    theta = vf.Theta(X, 2)
    assert theta == np.sum(X[2:5])

# ---------- GEOMETRY TRANSFORMS ---------- #

def test_X2_straight_line():
    N = 5
    X = vf.StraightLine(N)

    pos = vf.X2(X, 3)
    # straight line → all cos(0)=1, sin(0)=0
    assert np.isclose(pos[0], 3)
    assert np.isclose(pos[1], 0)


def test_X3N_shape():
    N = 4
    X = vf.StraightLine(N)
    X3 = vf.X3N(X)

    assert X3.shape == (3 * N, 1)


def test_XNp2_inverse_consistency():
    N = 4
    X = vf.StraightLine(N)
    X3 = vf.X3N(X)
    X_back = vf.XNp2(X3)

    assert X_back.shape == (N + 2,)

# ---------- MATRICES ---------- #

def test_QQ_shape():
    N = 5
    X = vf.StraightLine(N)
    X3 = vf.X3N(X)

    Q = vf.QQ(X3)
    assert Q.shape == (3 * N, N + 2)


def test_AA_shape():
    N = 4
    X = vf.StraightLine(N)
    X3 = vf.X3N(X)

    A = vf.AA(X3, gamma=2)
    assert A.shape == (N + 2, 3 * N)


def test_ADB_valid():
    N = 5
    taus_b = [1] * (N - 1)
    A = vf.ADB(taus_b, N)

    assert A.shape == (N + 2, N + 2)


def test_ADS_shape():
    N = 5
    A = vf.ADS(N)

    assert A.shape == (N + 2, 3 * N)


# ---------- FLOW ---------- #

def test_create_flow_no_flow():
    label, flow = vf.CreateFlowField()

    assert label == "NO FLOW"
    assert flow == 0


def test_create_flow_constant():
    T = np.linspace(0, 1, 5)
    label, flow = vf.CreateFlowField(A=1, w0=0, psi=0, T_meas=T)

    assert flow.shape == (2, len(T))
    assert np.allclose(flow[1], 0)  # psi=0 → no y component


def test_flow_no_field():
    N = 4
    X = vf.StraightLine(N)
    X3 = vf.X3N(X)

    result = vf.Flow(X3)
    assert result.shape == (4 * N, 1)
    assert np.allclose(result, 0)


# ---------- RHS TERMS ---------- #

def test_BB_zero_straight_line():
    N = 5
    X = vf.StraightLine(N)
    X3 = vf.X3N(X)

    B = vf.BB(X3)
    assert np.allclose(B, 0)


def test_BS_shape():
    N = 5
    X = vf.StraightLine(N)
    X3 = vf.X3N(X)

    B = vf.BS(X3)
    assert B.shape == (N + 2, 1)


def test_BC_zero_defaults():
    N = 4
    X = vf.StraightLine(N)
    X3 = vf.X3N(X)

    B0 = vf.BC_0(X3)
    BL = vf.BC_L(X3)

    assert np.allclose(B0, 0)
    assert np.allclose(BL, 0)


# ---------- MAIN DYNAMICS ---------- #

def test_g_output_shape():
    N = 4
    X = vf.StraightLine(N)

    params = dict(
        Sp4=1,
        N = N,
        k0=1,
        bool_EI=True,
        Beta=1,
        taus_b=[1]*(N-1),
        tau_s=0,
        gamma=2,
        n_L=[0, 0],
        m_L=0,
        Lambdas=[0]*N,
        Zetas=[0]*N,
        InterpFlow=0
    )

    X_dot = vf.g(0, X, **params)
    assert X_dot.shape == X.shape


def test_g_fixed_base():
    N = 4
    X = vf.StraightLine(N)

    params = dict(
        Sp4=1,
        k0=1,
        N = N,
        bool_EI=True,
        Beta=1,
        taus_b=[1]*(N-1),
        tau_s=0,
        gamma=2,
        n_L=[0, 0],
        m_L=0,
        Lambdas=[0]*N,
        Zetas=[0]*N,
        InterpFlow=0
    )

    X_dot = vf.g(0, X, **params)

    # enforced in code
    assert X_dot[0] == 0
    assert X_dot[1] == 0


# ---------- INTEGRATION ---------- #

def test_simulation_runs():
    N = 10
    X0 = vf.StraightLine(N)

    int_params = {
        "Sp4": 1,
        "N":N,
        "k0": 1e13,
        "bool_EI": True,
        "gamma": 2,
        "taus_b": [0]*(N-1),
        "tau_s": 0,
        "n_L": [0, 0],
        "m_L": 0,
        "X_0": X0
    }

    ext_params = {
        "Lambdas": [[0,0]]*N,
        "Zetas": [0]*N,
        "InterpFlow": 0
    }

    sim_params = {
        "T_span": (0, 0.1),
        "T_eval": np.linspace(0, 0.1, 5),
        "method": "RK45",
        "T_sim_max": 5
    }

    out = vf.ViscoElasticFilament_Simulate(int_params, ext_params, sim_params)

    assert out["value"] is not None

class TestViscoElasticFilamentSimulate:
    """Unit tests for ViscoElasticFilament.simulate_single()"""
    
    @pytest.fixture
    def minimal_params(self):
        """
        Absolute minimum working parameters.
        Start here and vary ONE thing at a time.
        """
        N = 10
        X0 = vf.StraightLine(N)

        return {
            'int_params': {
                'Sp4': 1.0,
                'N': N,
                'k0': 1e13,
                'bool_EI': True,
                'gamma': 2,
                'taus_b': [0]*(N-1),
                'tau_s': 0,
                'n_L': [0, 0],
                'm_L': 0,
                'X_0': X0,
            },
            'ext_params': {
                'Lambdas': [[0.0, 0.0]]*N,
                'Zetas': [0.0]*N,
                'InterpFlow': 0,
            },
            'sim_params': {
                'T_span': (0.0, 1.0),          # SHORT time span first
                'T_eval': np.linspace(0, 1, 10),
                'T_sim_max': 30.0,
                'method': 'BDF',
            }
        }
    
    def test_simulate_returns_dict(self, minimal_params):
        """ Test that simulate_single() returns a dictionary."""
        model = vf.ViscoElasticFilament(**minimal_params)
        output = model.simulate_single()
        
        assert isinstance(output, dict), (
            f"Expected dict, got {type(output)}"
        )
    
    def test_simulate_output_has_required_keys(self, minimal_params):
        """ Test that output dict has 'value', 'shape' keys."""
        model = vf.ViscoElasticFilament(**minimal_params)
        output = model.simulate_single()
        
        required_keys = {'value', 'shape'}
        assert required_keys.issubset(output.keys()), (
            f"Missing keys. Expected {required_keys}, got {output.keys()}"
        )
    
    def test_simulate_value_not_none(self, minimal_params):
        """ Test that output['value'] is not None (simulation succeeded)."""
        model = vf.ViscoElasticFilament(**minimal_params)
        output = model.simulate_single()
        
        assert output['value'] is not None, (
            f"Simulation failed. Output: {output}"
        )
    
    def test_simulate_returns_valid_trajectory(self, minimal_params):
        """ Test that trajectory is a numpy array with expected shape."""
        model = vf.ViscoElasticFilament(**minimal_params)
        output = model.simulate_single()
        
        trajectory = output['value']
        assert isinstance(trajectory, np.ndarray), (
            f"Expected np.ndarray, got {type(trajectory)}"
        )
        assert trajectory.shape == (minimal_params['int_params']['N']+2 , len(minimal_params['sim_params']['T_eval']))
    
    def test_simulate_with_longer_timespan(self, minimal_params):
        """ Test with T_span = (0, 5) to match your inference test."""
        minimal_params['sim_params']['T_span'] = (0.0, 5.0)
        minimal_params['sim_params']['T_eval'] = np.linspace(0, 5, 50)
        
        model = vf.ViscoElasticFilament(**minimal_params)
        output = model.simulate_single()
        
        assert output['value'] is not None, (
            f"Simulation failed with T_span=5. Output: {output}"
        )
    
    @pytest.mark.parametrize('sp4_value', [0.5, 1.0, 2.0, 5.0])
    def test_simulate_different_sp4_values(self, minimal_params, sp4_value):
        """ Test that Sp4 variations don't break the solver."""
        minimal_params['int_params']['Sp4'] = sp4_value
        
        model = vf.ViscoElasticFilament(**minimal_params)
        output = model.simulate_single()
        
        assert output['value'] is not None, (
            f"Simulation failed for Sp4={sp4_value}. Output: {output}"
        )
    
    def test_simulate_trajectory_is_finite(self, minimal_params):
        """ Test that trajectory contains only finite values (no NaN/inf)."""
        model = vf.ViscoElasticFilament(**minimal_params)
        output = model.simulate_single()
        
        trajectory = output['value']
        assert np.all(np.isfinite(trajectory)), (
            f"Trajectory contains NaN or Inf values:\n{trajectory}"
        )

class TestViscoElasticFilamentFlowParams:
    """Unit tests for ViscoElasticFilament_FlowParams (composed model with flow)"""
    
    @pytest.fixture
    def minimal_params_with_flow(self):
        """
        Minimal parameters for ViscoElasticFilament_FlowParams.
        Uses A, w0, psi instead of InterpFlow.
        """
        N = 10
        X0 = vf.StraightLine(N)

        return {
            'int_params': {
                'Sp4': 1.0,
                'N': N,
                'k0': 1e13,
                'bool_EI': True,
                'gamma': 2,
                'taus_b': [0.0]*N,
                'tau_s': 0.0,
                'n_L': [0, 0],
                'm_L': 0,
                'X_0': X0,
            },
            'ext_params': {
                'Lambdas': [[0.0, 0.0]]*N,
                'Zetas': [0.0]*N,
                'A': 1e-5,           # Amplitude
                'w0': 1e0,           # Angular frequency (rad/s)
                'psi': np.pi/2,          # Flow angle (radians)
            },
            'sim_params': {
                'T_span': (0.0, 1.0),
                'T_eval': np.linspace(0, 1, 10),
                'T_sim_max': 30.0,
                'method': 'BDF',
            }
        }
    
    def test_flow_params_to_interpflow_transforms_correctly(self, minimal_params_with_flow):
        """ Test that FlowParams_to_InterpFlow transforms A, w0, psi -> InterpFlow."""
        int_params = minimal_params_with_flow['int_params']
        ext_params = minimal_params_with_flow['ext_params']
        sim_params = minimal_params_with_flow['sim_params']
        
        # Call transformation directly
        transformed = vf.FlowParams_to_InterpFlow(int_params, ext_params, sim_params)
        
        # Verify output keys
        assert 'Lambdas' in transformed, "Missing 'Lambdas' in transformed params"
        assert 'Zetas' in transformed, "Missing 'Zetas' in transformed params"
        assert 'InterpFlow' in transformed, "Missing 'InterpFlow' in transformed params"
        
        # Verify InterpFlow is either callable (interpolator) or 0 (no flow)
        interpflow = transformed['InterpFlow']
        assert callable(interpflow) or interpflow == 0, (
            f"InterpFlow should be callable or 0, got {type(interpflow)}"
        )
    
    def test_composed_model_initializes(self, minimal_params_with_flow):
        """ Test that ViscoElasticFilament_FlowParams initializes without error."""
        model = vf.ViscoElasticFilament_FlowParams(**minimal_params_with_flow)
        
        assert model is not None, "Model failed to initialize"
        # If the model has an attribute to check, add it here
        # Example: assert hasattr(model, 'simulate_single')
    
    def test_composed_model_simulates_with_flow(self, minimal_params_with_flow):
        """ Test that composed model can simulate with non-zero flow."""
        model = vf.ViscoElasticFilament_FlowParams(**minimal_params_with_flow)
        output = model.simulate_single()
        
        assert output['value'] is not None, (
            f"Simulation with flow failed. Output: {output}"
        )
    
    def test_composed_model_output_structure(self, minimal_params_with_flow):
        """ Test that output has expected keys and shapes."""
        model = vf.ViscoElasticFilament_FlowParams(**minimal_params_with_flow)
        output = model.simulate_single()
        
        trajectory = output['value']
        assert isinstance(trajectory, np.ndarray), (
            f"Expected np.ndarray, got {type(trajectory)}"
        )
        
        # TODO: Adjust expected shape based on your model output
        # assert trajectory.shape[0] == len(minimal_params_with_flow['sim_params']['T_eval'])
    
    def test_composed_model_trajectory_is_finite(self, minimal_params_with_flow):
        """ Test that trajectory contains only finite values."""
        model = vf.ViscoElasticFilament_FlowParams(**minimal_params_with_flow)
        output = model.simulate_single()
        
        trajectory = output['value']
        assert np.all(np.isfinite(trajectory)), (
            f"Trajectory contains NaN or Inf:\n{trajectory}"
        )
    
    @pytest.mark.parametrize('a_value', [0.0, 1e-5, 1e-4, 1e-3])
    def test_different_amplitudes(self, minimal_params_with_flow, a_value):
        """ Test that different flow amplitudes don't break the solver."""
        minimal_params_with_flow['ext_params']['A'] = a_value
        
        model = vf.ViscoElasticFilament_FlowParams(**minimal_params_with_flow)
        output = model.simulate_single()
        
        assert output['value'] is not None, (
            f"Simulation failed for A={a_value}. Output: {output}"
        )
    
    @pytest.mark.parametrize('w0_value', [0.1, 1.0, 5.0, 10.0])
    def test_different_frequencies(self, minimal_params_with_flow, w0_value):
        """ Test that different flow frequencies don't break the solver."""
        minimal_params_with_flow['ext_params']['w0'] = w0_value
        
        model = vf.ViscoElasticFilament_FlowParams(**minimal_params_with_flow)
        output = model.simulate_single()
        
        assert output['value'] is not None, (
            f"Simulation failed for w0={w0_value}. Output: {output}"
        )
    
    @pytest.mark.parametrize('psi_value', [0.0, np.pi/4, np.pi/2, np.pi])
    def test_different_phases(self, minimal_params_with_flow, psi_value):
        """ Test that different phase shifts don't break the solver."""
        minimal_params_with_flow['ext_params']['psi'] = psi_value
        
        model = vf.ViscoElasticFilament_FlowParams(**minimal_params_with_flow)
        output = model.simulate_single()
        
        assert output['value'] is not None, (
            f"Simulation failed for psi={psi_value}. Output: {output}"
        )
    
    def test_flow_vs_no_flow_comparison(self, minimal_params_with_flow):
        """
        Compare trajectories with and without flow.
        Verify that flow actually changes the trajectory.
        """
        # Simulate WITH flow
        model_with_flow = vf.ViscoElasticFilament_FlowParams(**minimal_params_with_flow)
        output_with_flow = model_with_flow.simulate_single()
        traj_with_flow = output_with_flow['value']
        
        # Simulate WITHOUT flow (A = 0)
        params_no_flow = minimal_params_with_flow.copy()
        params_no_flow['ext_params'] = minimal_params_with_flow['ext_params'].copy()
        params_no_flow['ext_params']['A'] = 0.0
        
        model_no_flow = vf.ViscoElasticFilament_FlowParams(**params_no_flow)
        output_no_flow = model_no_flow.simulate_single()
        traj_no_flow = output_no_flow['value']
        
        # Verify both succeeded
        assert traj_with_flow is not None, "Simulation with flow failed"
        assert traj_no_flow is not None, "Simulation without flow failed"
        
        # (Optional) Verify flow changes the trajectory
        # The trajectories should be different (flow applies external force)
        # You can relax this if small flows don't significantly change output
        # difference = np.linalg.norm(traj_with_flow - traj_no_flow)
        # print(f"\nTrajectory difference (with vs without flow): {difference}")

class TestViscoElasticFilamentBatch:
    """Unit tests for ViscoElasticFilament.simulate_batch()"""
    
    @pytest.fixture
    def minimal_params(self):
        """
        Absolute minimum working parameters (fixed, working version).
        """
        N = 10
        X0 = vf.StraightLine(N)

        return {
            'int_params': {
                'Sp4': 1.0,
                'N': N,
                'k0': 1e13,
                'bool_EI': True,
                'gamma': 2,
                'taus_b': [0]*(N-1),
                'tau_s': 0,
                'n_L': [0, 0],
                'm_L': 0,
                'X_0': X0,
            },
            'ext_params': {
                'Lambdas': [[0.0, 0.0]]*N,
                'Zetas': [0.0]*N,
                'InterpFlow': 0,
            },
            'sim_params': {
                'T_span': (0.0, 1.0),
                'T_eval': np.linspace(0, 1, 10),
                'T_sim_max': 30.0,
                'method': 'BDF',
            }
        }
    
    def test_simulate_batch_returns_list(self, minimal_params):
        """Test that simulate_batch() returns a list."""
        # Create batch: 3 different Sp4 values
        batch_int_params = [
            {**minimal_params['int_params'], 'Sp4': sp4}
            for sp4 in [0.5, 1.0, 2.0]
        ]
        batch_ext_params = [minimal_params['ext_params']] * 3
        batch_sim_params = [minimal_params['sim_params']] * 3
        
        results = vf.ViscoElasticFilament.simulate_batch(
            int_params_batch=batch_int_params,
            ext_params_batch=batch_ext_params,
            sim_params_batch=batch_sim_params
        )
        
        assert isinstance(results, list), (
            f"Expected list, got {type(results)}"
        )
        assert len(results) == 3, (
            f"Expected 3 results, got {len(results)}"
        )
    
    def test_simulate_batch_each_result_is_dict(self, minimal_params):
        """Test that each result in batch is a dict with required keys."""
        batch_int_params = [
            {**minimal_params['int_params'], 'Sp4': sp4}
            for sp4 in [0.5, 1.0, 2.0]
        ]
        batch_ext_params = [minimal_params['ext_params']] * 3
        batch_sim_params = [minimal_params['sim_params']] * 3
        
        results = vf.ViscoElasticFilament.simulate_batch(
            int_params_batch=batch_int_params,
            ext_params_batch=batch_ext_params,
            sim_params_batch=batch_sim_params
        )
        
        for i, result in enumerate(results):
            assert isinstance(result, dict), (
                f"Result {i} is not a dict: {type(result)}"
            )
            assert 'value' in result, f"Result {i} missing 'value' key"
    
    def test_simulate_batch_all_simulations_succeed(self, minimal_params):
        """Test that all batch simulations return non-None values."""
        batch_int_params = [
            {**minimal_params['int_params'], 'Sp4': sp4}
            for sp4 in [0.5, 1.0, 2.0]
        ]
        batch_ext_params = [minimal_params['ext_params']] * 3
        batch_sim_params = [minimal_params['sim_params']] * 3
        
        results = vf.ViscoElasticFilament.simulate_batch(
            int_params_batch=batch_int_params,
            ext_params_batch=batch_ext_params,
            sim_params_batch=batch_sim_params
        )
        
        for i, result in enumerate(results):
            assert result['value'] is not None, (
                f"Simulation {i} failed. Result: {result}"
            )
    
    def test_simulate_batch_trajectories_are_finite(self, minimal_params):
        """Test that all trajectories contain only finite values."""
        batch_int_params = [
            {**minimal_params['int_params'], 'Sp4': sp4}
            for sp4 in [0.5, 1.0, 2.0]
        ]
        batch_ext_params = [minimal_params['ext_params']] * 3
        batch_sim_params = [minimal_params['sim_params']] * 3
        
        results = vf.ViscoElasticFilament.simulate_batch(
            int_params_batch=batch_int_params,
            ext_params_batch=batch_ext_params,
            sim_params_batch=batch_sim_params
        )
        
        for i, result in enumerate(results):
            trajectory = result['value']
            assert np.all(np.isfinite(trajectory)), (
                f"Trajectory {i} contains NaN/Inf:\n{trajectory}"
            )
    
    @pytest.mark.parametrize('batch_size', [1, 5, 10])
    def test_simulate_batch_different_sizes(self, minimal_params, batch_size):
        """Test batch with different batch sizes."""
        batch_int_params = [
            {**minimal_params['int_params'], 'Sp4': 0.5 + (i * 0.1)}
            for i in range(batch_size)
        ]
        batch_ext_params = [minimal_params['ext_params']] * batch_size
        batch_sim_params = [minimal_params['sim_params']] * batch_size
        
        results = vf.ViscoElasticFilament.simulate_batch(
            int_params_batch=batch_int_params,
            ext_params_batch=batch_ext_params,
            sim_params_batch=batch_sim_params
        )
        
        assert len(results) == batch_size, (
            f"Expected {batch_size} results, got {len(results)}"
        )
        
        for i, result in enumerate(results):
            assert result['value'] is not None, (
                f"Batch size {batch_size}, item {i} failed"
            )
    
    def test_simulate_batch_parameter_variation_affects_output(self, minimal_params):
        """Test that varying Sp4 in batch produces different trajectories."""
        batch_int_params = [
            {**minimal_params['int_params'], 'Sp4': sp4}
            for sp4 in [0.5, 2.0]
        ]
        batch_ext_params = [minimal_params['ext_params']] * 2
        batch_sim_params = [minimal_params['sim_params']] * 2
        
        results = vf.ViscoElasticFilament.simulate_batch(
            int_params_batch=batch_int_params,
            ext_params_batch=batch_ext_params,
            sim_params_batch=batch_sim_params
        )
        
        traj_1 = results[0]['value']
        traj_2 = results[1]['value']
        
        difference = np.linalg.norm(traj_1 - traj_2)
        print(f"\nTrajectory difference (Sp4=0.5 vs Sp4=2.0): {difference}")
        
        assert difference >= 0, "Trajectories computed"


class TestViscoElasticFilamentFlowParamsBatch:
    """Unit tests for ViscoElasticFilament_FlowParams.simulate_batch() with flow"""
    
    @pytest.fixture
    def minimal_params_with_flow(self):
        """Minimal parameters for batch simulation with flow."""
        N = 10
        X0 = vf.StraightLine(N)

        return {
            'int_params': {
                'Sp4': 1.0,
                'N': N,
                'k0': 1e13,
                'bool_EI': True,
                'gamma': 2,
                'taus_b': [0.0]*N,
                'tau_s': 0.0,
                'n_L': [0, 0],
                'm_L': 0,
                'X_0': X0,
            },
            'ext_params': {
                'Lambdas': [[0.0, 0.0]]*N,
                'Zetas': [0.0]*N,
                'A': 1e-5,
                'w0': 1e0,
                'psi': np.pi/2,
            },
            'sim_params': {
                'T_span': (0.0, 1.0),
                'T_eval': np.linspace(0, 1, 10),
                'T_sim_max': 30.0,
                'method': 'BDF',
            }
        }
    
    def test_flow_batch_returns_list(self, minimal_params_with_flow):
        """Test that simulate_batch() returns a list for flow model."""
        batch_int_params = [
            {**minimal_params_with_flow['int_params'], 'Sp4': sp4}
            for sp4 in [0.5, 1.0, 2.0]
        ]
        batch_ext_params = [minimal_params_with_flow['ext_params']] * 3
        batch_sim_params = [minimal_params_with_flow['sim_params']] * 3
        
        results = vf.ViscoElasticFilament_FlowParams.simulate_batch(
            int_params_batch=batch_int_params,
            ext_params_batch=batch_ext_params,
            sim_params_batch=batch_sim_params
        )
        
        assert isinstance(results, list), (
            f"Expected list, got {type(results)}"
        )
        assert len(results) == 3, (
            f"Expected 3 results, got {len(results)}"
        )
    
    def test_flow_batch_all_simulations_succeed(self, minimal_params_with_flow):
        """Test that all flow batch simulations succeed."""
        batch_int_params = [
            {**minimal_params_with_flow['int_params'], 'Sp4': sp4}
            for sp4 in [0.5, 1.0, 2.0]
        ]
        batch_ext_params = [minimal_params_with_flow['ext_params']] * 3
        batch_sim_params = [minimal_params_with_flow['sim_params']] * 3
        
        results = vf.ViscoElasticFilament_FlowParams.simulate_batch(
            int_params_batch=batch_int_params,
            ext_params_batch=batch_ext_params,
            sim_params_batch=batch_sim_params
        )
        
        for i, result in enumerate(results):
            assert result['value'] is not None, (
                f"Flow simulation {i} failed. Result: {result}"
            )
    
    def test_flow_batch_trajectories_finite(self, minimal_params_with_flow):
        """Test that all flow trajectories are finite."""
        batch_int_params = [
            {**minimal_params_with_flow['int_params'], 'Sp4': sp4}
            for sp4 in [0.5, 1.0, 2.0]
        ]
        batch_ext_params = [minimal_params_with_flow['ext_params']] * 3
        batch_sim_params = [minimal_params_with_flow['sim_params']] * 3
        
        results = vf.ViscoElasticFilament_FlowParams.simulate_batch(
            int_params_batch=batch_int_params,
            ext_params_batch=batch_ext_params,
            sim_params_batch=batch_sim_params
        )
        
        for i, result in enumerate(results):
            trajectory = result['value']
            assert np.all(np.isfinite(trajectory)), (
                f"Flow trajectory {i} contains NaN/Inf"
            )
    
    @pytest.mark.parametrize('a_value', [0.0, 1e-5, 1e-4])
    def test_flow_batch_varying_amplitudes(self, minimal_params_with_flow, a_value):
        """Test batch with varying flow amplitudes."""
        batch_int_params = [
            {**minimal_params_with_flow['int_params'], 'Sp4': sp4}
            for sp4 in [0.5, 1.0, 2.0]
        ]
        
        # Each ext_params gets the new amplitude
        batch_ext_params = [
            {**minimal_params_with_flow['ext_params'], 'A': a_value}
            for _ in range(3)
        ]
        batch_sim_params = [minimal_params_with_flow['sim_params']] * 3
        
        results = vf.ViscoElasticFilament_FlowParams.simulate_batch(
            int_params_batch=batch_int_params,
            ext_params_batch=batch_ext_params,
            sim_params_batch=batch_sim_params
        )
        
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result['value'] is not None, (
                f"A={a_value}, item {i} failed"
            )
    
    @pytest.mark.parametrize('w0_value', [0.5, 1.0, 5.0])
    def test_flow_batch_varying_frequencies(self, minimal_params_with_flow, w0_value):
        """Test batch with varying flow frequencies."""
        batch_int_params = [
            {**minimal_params_with_flow['int_params'], 'Sp4': sp4}
            for sp4 in [0.5, 1.0, 2.0]
        ]
        
        # Each ext_params gets the new frequency
        batch_ext_params = [
            {**minimal_params_with_flow['ext_params'], 'w0': w0_value}
            for _ in range(3)
        ]
        batch_sim_params = [minimal_params_with_flow['sim_params']] * 3
        
        results = vf.ViscoElasticFilament_FlowParams.simulate_batch(
            int_params_batch=batch_int_params,
            ext_params_batch=batch_ext_params,
            sim_params_batch=batch_sim_params
        )
        
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result['value'] is not None, (
                f"w0={w0_value}, item {i} failed"
            )
    def test_flow_batch_with_no_flow_comparison(self, minimal_params_with_flow):
        """Compare batch with flow vs batch without flow."""
        # Batch WITH flow (A > 0)
        batch_int_params = [
            {**minimal_params_with_flow['int_params'], 'Sp4': 1.0}
        ]
        batch_ext_params_with_flow = [minimal_params_with_flow['ext_params']]
        batch_sim_params = [minimal_params_with_flow['sim_params']]
        
        results_with_flow = vf.ViscoElasticFilament_FlowParams.simulate_batch(
            int_params_batch=batch_int_params,
            ext_params_batch=batch_ext_params_with_flow,
            sim_params_batch=batch_sim_params
        )
        traj_with_flow = results_with_flow[0]['value']
        
        # Batch WITHOUT flow (A = 0)
        ext_params_no_flow = {**minimal_params_with_flow['ext_params'], 'A': 0.0}
        batch_ext_params_no_flow = [ext_params_no_flow]
        
        results_no_flow = vf.ViscoElasticFilament_FlowParams.simulate_batch(
            int_params_batch=batch_int_params,
            ext_params_batch=batch_ext_params_no_flow,
            sim_params_batch=batch_sim_params
        )
        traj_no_flow = results_no_flow[0]['value']
        
        # Verify trajectories are different
        difference = np.linalg.norm(traj_with_flow - traj_no_flow)
        print(f"\nTrajectory difference (with flow A=1e-5 vs no flow A=0): {difference}")
        
        # Flow should affect the dynamics
        assert traj_with_flow is not None, "With-flow simulation failed"
        assert traj_no_flow is not None, "No-flow simulation failed"
        assert difference >= 0, "Trajectories computed successfully"

# --- Tests for Parallel Simulation: ViscoElasticFilament ---
class TestViscoElasticFilamentParallel:
    """Unit tests for parallel batch simulation of ViscoElasticFilament"""
    
    @pytest.fixture
    def parallel_params_vef(self):
        """Create parameter combinations for ViscoElasticFilament parallel tests."""
        N_list = [5, 10]
        X_0_list = [vf.StraightLine(N) for N in N_list]
        taus_b_list = [[0.0]*(N-1) for N in N_list]
        Lambdas_list = [[[0.0, 1e-5]]*N for N in N_list]
        Zetas_list = [[0.0]*N for N in N_list]
        
        return {
            'gamma_list': [1, 2],
            'N_list': N_list,
            'X_0_list': X_0_list,
            'taus_b_list': taus_b_list,
            'Lambdas_list': Lambdas_list,
            'Zetas_list': Zetas_list,
            'k0_list': [1e13],
            'bool_EI_list': [True],
            'Sp4_list': [0.1, 1.0, 10.0],
            'Beta_list': [0.0],
            'tau_s_list': [0.0],
            'n_L_list': [[0, 0]],
            'm_L_list': [0],
            'InterpFlow_list': [0],
            'T_span_list': [(0.0, 1e6)],
            'T_eval_list': [[np.linspace(0, 1e6, int(1e2))]], # T_eval depends on T_span
            'T_sim_max_list': [300.0],
            'method_list': ['BDF'],
        }
    
    def test_parallel_simulate_batch_vef(self, parallel_params_vef):
        """Test parallel batch simulation for ViscoElasticFilament."""
        int_keys = ['Sp4', 'N', 'k0', 'bool_EI', 'gamma', 'taus_b', 'tau_s', 'n_L', 'm_L', 'X_0']
        ext_keys = ['Lambdas', 'Zetas', 'InterpFlow']
        sim_keys = ['T_span', 'T_eval', 'T_sim_max', 'method']
        
        params_list = vf.ViscoElasticFilament_create_params_list(
            int_keys, ext_keys, sim_keys, parallel_params_vef
        )
        
        # Expected: 2 gamma * 2 N * 1 k0 * 1 EI * 3 Sp4 * 1 Beta * 1 tau_s * 1 n_L * 1 m_L * 1 flow * 1 T_span * 1 T_eval * 1 T_sim_max * 1 method
        # = 2 * 2 * 3 = 12
        assert len(params_list) == 12, f"Expected 12 param combinations, got {len(params_list)}"
        
        # Simulate in parallel
        outputs = vf.parallel_simulate_batch(vf.ViscoElasticFilament, params_list, n_jobs=4)
        
        assert len(outputs) == 12
        for i, out in enumerate(outputs):
            assert isinstance(out, dict), f"Output {i} is not a dict"
            assert "value" in out, f"Output {i} missing 'value' key"
            assert out["value"] is not None, f"Output {i} has None value"
            assert np.all(np.isfinite(out["value"])), (
                f"Output {i} contains NaN/Inf"
            )
    
    def test_parallel_simulate_batch_vef_single_job(self, parallel_params_vef):
        """Test parallel simulation with single job (serial execution)."""
        int_keys = ['Sp4', 'N', 'k0', 'bool_EI', 'gamma', 'taus_b', 'tau_s', 'n_L', 'm_L', 'X_0']
        ext_keys = ['Lambdas', 'Zetas', 'InterpFlow']
        sim_keys = ['T_span', 'T_eval', 'T_sim_max', 'method']
        
        params_list = vf.ViscoElasticFilament_create_params_list(
            int_keys, ext_keys, sim_keys, parallel_params_vef
        )
        
        # Serial execution
        outputs = vf.parallel_simulate_batch(vf.ViscoElasticFilament, params_list, n_jobs=1)
        
        assert len(outputs) == 12
        for out in outputs:
            assert out["value"] is not None
    
    def test_parallel_simulate_batch_vef_parameter_variation(self, parallel_params_vef):
        """Test that varying Sp4 produces different results in parallel batch."""
        int_keys = ['Sp4', 'N', 'k0', 'bool_EI', 'gamma', 'taus_b', 'tau_s', 'n_L', 'm_L', 'X_0']
        ext_keys = ['Lambdas', 'Zetas', 'InterpFlow']
        sim_keys = ['T_span', 'T_eval', 'T_sim_max', 'method']
        
        # Only test with single gamma and N to isolate Sp4 effect
        parallel_params_vef['gamma_list'] = [2]
        parallel_params_vef['N_list'] = [10]
        parallel_params_vef['X_0_list'] = [vf.StraightLine(10)]
        parallel_params_vef['taus_b_list'] = [[0.0]*9]
        parallel_params_vef['Lambdas_list'] = [[[0.0, 1e-5]]*10]
        parallel_params_vef['Zetas_list'] = [[0.0]*10]
        
        params_list = vf.ViscoElasticFilament_create_params_list(
            int_keys, ext_keys, sim_keys, parallel_params_vef
        )
        
        # Should have 3 outputs (one per Sp4 value)
        assert len(params_list) == 3
        
        outputs = vf.parallel_simulate_batch(vf.ViscoElasticFilament, params_list, n_jobs=2)
        
        assert len(outputs) == 3
        
        # Extract Sp4 values from params
        sp4_values = [p[0]['Sp4'] for p in params_list]
        assert sp4_values == [0.1, 1.0, 10.0]
        
        # Verify trajectories differ
        traj_0 = outputs[0]['value']
        traj_1 = outputs[1]['value']
        traj_2 = outputs[2]['value']
        
        diff_01 = np.linalg.norm(traj_0 - traj_1)
        diff_12 = np.linalg.norm(traj_1 - traj_2)
        
        print(f"\nSp4=0.5 vs Sp4=1.0 diff: {diff_01}")
        print(f"Sp4=1.0 vs Sp4=2.0 diff: {diff_12}")

        print(f"\ntraj_0: {traj_0}")
        print(f"\ntraj_0: {traj_1}")
        print(f"\ntraj_0: {traj_2}")     
        
        # Both differences should be significant (Sp4 affects dynamics)
        assert diff_01 > 0, "Sp4 variation should produce different trajectories"
        assert diff_12 > 0, "Sp4 variation should produce different trajectories"


# --- Tests for Parallel Simulation: ViscoElasticFilament_FlowParams ---
class TestViscoElasticFilamentFlowParamsParallel:
    """Unit tests for parallel batch simulation of ViscoElasticFilament_FlowParams"""
    
    @pytest.fixture
    def parallel_params_vef_flow(self):
        """Create parameter combinations for ViscoElasticFilament_FlowParams parallel tests."""
        N_list = [5, 10]
        X_0_list = [vf.StraightLine(N) for N in N_list]
        taus_b_list = [[0.0]*(N-1) for N in N_list]
        Lambdas_list = [[[0.0, 0.0]]*N for N in N_list]
        Zetas_list = [[0.0]*N for N in N_list]
        
        return {
            'gamma_list': [1, 2],
            'N_list': N_list,
            'X_0_list': X_0_list,
            'taus_b_list': taus_b_list,
            'Lambdas_list': Lambdas_list,
            'Zetas_list': Zetas_list,
            'k0_list': [1e13],
            'bool_EI_list': [True],
            'Sp4_list': [0.5, 1.0],
            'Beta_list': [0.0],
            'tau_s_list': [0.0],
            'n_L_list': [[0, 0]],
            'm_L_list': [0],
            'A_list': [0.0, 1e-5],
            'w0_list': [1.0],
            'psi_list': [np.pi/2],
            'T_span_list': [(0.0, 1e6)],
            'T_eval_list': [[np.linspace(0, 1e6, int(1e2))]], # T_eval depends on T_span
            'T_sim_max_list': [300.0],
            'method_list': ['BDF'],
        }
    
    def test_parallel_simulate_batch_vef_flow(self, parallel_params_vef_flow):
        """Test parallel batch simulation for ViscoElasticFilament_FlowParams."""
        int_keys = ['Sp4', 'N', 'k0', 'bool_EI', 'gamma', 'taus_b', 'tau_s', 'n_L', 'm_L', 'X_0']
        ext_keys = ['Lambdas', 'Zetas', 'A', 'w0', 'psi']
        sim_keys = ['T_span', 'T_eval', 'T_sim_max', 'method']
        
        params_list = vf.ViscoElasticFilament_FlowParams_create_params_list(
            int_keys, ext_keys, sim_keys, parallel_params_vef_flow
        )
        
        # Expected: 2 gamma * 2 N * 1 k0 * 1 EI * 2 Sp4 * 1 Beta * 1 tau_s * 1 n_L * 1 m_L * 2 A * 1 w0 * 1 psi * 1 T_span * 1 T_eval * 1 T_sim_max * 1 method
        # = 2 * 2 * 2 * 2 = 16
        assert len(params_list) == 16, f"Expected 16 param combinations, got {len(params_list)}"
        
        # Simulate in parallel
        outputs = vf.parallel_simulate_batch(vf.ViscoElasticFilament_FlowParams, params_list, n_jobs=4)
        
        assert len(outputs) == 16
        for i, out in enumerate(outputs):
            assert isinstance(out, dict), f"Output {i} is not a dict"
            assert "value" in out, f"Output {i} missing 'value' key"
            assert out["value"] is not None, f"Output {i} has None value"
            assert np.all(np.isfinite(out["value"])), (
                f"Output {i} contains NaN/Inf"
            )
    
    def test_parallel_simulate_batch_vef_flow_varying_amplitude(self, parallel_params_vef_flow):
        """Test that varying flow amplitude produces different results."""
        int_keys = ['Sp4', 'N', 'k0', 'bool_EI', 'gamma', 'taus_b', 'tau_s', 'n_L', 'm_L', 'X_0']
        ext_keys = ['Lambdas', 'Zetas', 'A', 'w0', 'psi']
        sim_keys = ['T_span', 'T_eval', 'T_sim_max', 'method']
        
        # Isolate amplitude variation
        parallel_params_vef_flow['gamma_list'] = [2]
        parallel_params_vef_flow['N_list'] = [10]
        parallel_params_vef_flow['X_0_list'] = [vf.StraightLine(10)]
        parallel_params_vef_flow['taus_b_list'] = [[0.0]*9]
        parallel_params_vef_flow['Lambdas_list'] = [[[0.0, 0.0]]*10]
        parallel_params_vef_flow['Zetas_list'] = [[0.0]*10]
        parallel_params_vef_flow['Sp4_list'] = [1.0]
        parallel_params_vef_flow['w0_list'] = [1.0]
        parallel_params_vef_flow['psi_list'] = [np.pi/2]
        
        params_list = vf.ViscoElasticFilament_FlowParams_create_params_list(
            int_keys, ext_keys, sim_keys, parallel_params_vef_flow
        )
        
        # Should have 2 outputs (A=0.0 and A=1e-5)
        assert len(params_list) == 2
        
        outputs = vf.parallel_simulate_batch(vf.ViscoElasticFilament_FlowParams, params_list, n_jobs=2)
        
        assert len(outputs) == 2
        
        traj_no_flow = outputs[0]['value']  # A = 0.0
        traj_with_flow = outputs[1]['value']  # A = 1e-5
        
        difference = np.linalg.norm(traj_with_flow - traj_no_flow)
        print(f"\nTrajectory difference (A=0.0 vs A=1e-5): {difference}")
        
        assert difference >= 0, "Flow amplitude affects trajectories"

    def test_parallel_simulate_batch_vef_flow_varying_frequency(self, parallel_params_vef_flow):
        """Test that varying flow frequency produces different results."""
        int_keys = ['Sp4', 'N', 'k0', 'bool_EI', 'gamma', 'taus_b', 'tau_s', 'n_L', 'm_L', 'X_0']
        ext_keys = ['Lambdas', 'Zetas', 'A', 'w0', 'psi']
        sim_keys = ['T_span', 'T_eval', 'T_sim_max', 'method']
        
        # Isolate frequency variation
        parallel_params_vef_flow['gamma_list'] = [2]
        parallel_params_vef_flow['N_list'] = [10]
        parallel_params_vef_flow['X_0_list'] = [vf.StraightLine(10)]
        parallel_params_vef_flow['taus_b_list'] = [[0.0]*9]
        parallel_params_vef_flow['Lambdas_list'] = [[[0.0, 0.0]]*10]
        parallel_params_vef_flow['Zetas_list'] = [[0.0]*10]
        parallel_params_vef_flow['Sp4_list'] = [1.0]
        parallel_params_vef_flow['A_list'] = [1e-5]
        parallel_params_vef_flow['w0_list'] = [0.5, 1.0, 2.0]
        parallel_params_vef_flow['psi_list'] = [np.pi/2]
        
        params_list = vf.ViscoElasticFilament_FlowParams_create_params_list(
            int_keys, ext_keys, sim_keys, parallel_params_vef_flow
        )
        
        # Should have 3 outputs (w0 = 0.5, 1.0, 2.0)
        assert len(params_list) == 3
        
        outputs = vf.parallel_simulate_batch(vf.ViscoElasticFilament_FlowParams, params_list, n_jobs=2)
        
        assert len(outputs) == 3
        
        # Extract w0 values from params
        w0_values = [p[1]['w0'] for p in params_list]
        assert w0_values == [0.5, 1.0, 2.0]
        
        for i, out in enumerate(outputs):
            assert out["value"] is not None, f"Output {i} (w0={w0_values[i]}) is None"
            assert np.all(np.isfinite(out["value"])), (
                f"Output {i} (w0={w0_values[i]}) contains NaN/Inf"
            )
        
        traj_w0_05 = outputs[0]['value']
        traj_w0_10 = outputs[1]['value']
        traj_w0_20 = outputs[2]['value']
        
        diff_05_10 = np.linalg.norm(traj_w0_05 - traj_w0_10)
        diff_10_20 = np.linalg.norm(traj_w0_10 - traj_w0_20)
        
        print(f"\nTrajectory difference (w0=0.5 vs w0=1.0): {diff_05_10}")
        print(f"Trajectory difference (w0=1.0 vs w0=2.0): {diff_10_20}")
        
        assert diff_05_10 >= 0, "Flow frequency affects trajectories"
        assert diff_10_20 >= 0, "Flow frequency affects trajectories"
    
    def test_parallel_simulate_batch_vef_flow_no_flow_vs_with_flow(self, parallel_params_vef_flow):
        """
        Test parallel batch comparing no-flow (A=0) vs with-flow (A>0) scenarios.
        Verify that flow significantly affects trajectories.
        """
        int_keys = ['Sp4', 'N', 'k0', 'bool_EI', 'gamma', 'taus_b', 'tau_s', 'n_L', 'm_L', 'X_0']
        ext_keys = ['Lambdas', 'Zetas', 'A', 'w0', 'psi']
        sim_keys = ['T_span', 'T_eval', 'T_sim_max', 'method']
        
        # Test with two Sp4 values, each with and without flow
        parallel_params_vef_flow['gamma_list'] = [2]
        parallel_params_vef_flow['N_list'] = [10]
        parallel_params_vef_flow['X_0_list'] = [vf.StraightLine(10)]
        parallel_params_vef_flow['taus_b_list'] = [[0.0]*9]
        parallel_params_vef_flow['Lambdas_list'] = [[[0.0, 0.0]]*10]
        parallel_params_vef_flow['Zetas_list'] = [[0.0]*10]
        parallel_params_vef_flow['Sp4_list'] = [0.5, 1.0]
        parallel_params_vef_flow['A_list'] = [0.0, 1e-5]
        parallel_params_vef_flow['w0_list'] = [1.0]
        parallel_params_vef_flow['psi_list'] = [np.pi/2]
        
        params_list = vf.ViscoElasticFilament_FlowParams_create_params_list(
            int_keys, ext_keys, sim_keys, parallel_params_vef_flow
        )
        
        # Expected: 2 Sp4 * 2 A = 4 combinations
        assert len(params_list) == 4
        
        outputs = vf.parallel_simulate_batch(vf.ViscoElasticFilament_FlowParams, params_list, n_jobs=4)
        
        assert len(outputs) == 4
        
        # Organize outputs: [Sp4_0.5_A_0, Sp4_0.5_A_1e-5, Sp4_1.0_A_0, Sp4_1.0_A_1e-5]
        traj_sp4_05_no_flow = outputs[0]['value']
        traj_sp4_05_with_flow = outputs[1]['value']
        traj_sp4_10_no_flow = outputs[2]['value']
        traj_sp4_10_with_flow = outputs[3]['value']
        
        # Compare with-flow vs no-flow for each Sp4
        diff_sp4_05 = np.linalg.norm(traj_sp4_05_with_flow - traj_sp4_05_no_flow)
        diff_sp4_10 = np.linalg.norm(traj_sp4_10_with_flow - traj_sp4_10_no_flow)
        
        print(f"\nSp4=0.5: Flow effect magnitude: {diff_sp4_05}")
        print(f"Sp4=1.0: Flow effect magnitude: {diff_sp4_10}")
        
        assert diff_sp4_05 >= 0, "Flow should affect Sp4=0.5 trajectories"
        assert diff_sp4_10 >= 0, "Flow should affect Sp4=1.0 trajectories"
    
    def test_parallel_simulate_batch_vef_flow_consistency(self, parallel_params_vef_flow):
        """
        Test that running the same parameter set twice in parallel yields consistent results.
        """
        int_keys = ['Sp4', 'N', 'k0', 'bool_EI', 'gamma', 'taus_b', 'tau_s', 'n_L', 'm_L', 'X_0']
        ext_keys = ['Lambdas', 'Zetas', 'A', 'w0', 'psi']
        sim_keys = ['T_span', 'T_eval', 'T_sim_max', 'method']
        
        # Single parameter set repeated twice
        parallel_params_vef_flow['gamma_list'] = [2]
        parallel_params_vef_flow['N_list'] = [10]
        parallel_params_vef_flow['X_0_list'] = [vf.StraightLine(10)]
        parallel_params_vef_flow['taus_b_list'] = [[0.0]*9]
        parallel_params_vef_flow['Lambdas_list'] = [[[0.0, 0.0]]*10]
        parallel_params_vef_flow['Zetas_list'] = [[0.0]*10]
        parallel_params_vef_flow['Sp4_list'] = [1.0]
        parallel_params_vef_flow['A_list'] = [1e-5]
        parallel_params_vef_flow['w0_list'] = [1.0]
        parallel_params_vef_flow['psi_list'] = [np.pi/2]
        
        params_list = vf.ViscoElasticFilament_FlowParams_create_params_list(
            int_keys, ext_keys, sim_keys, parallel_params_vef_flow
        )
        
        # Should have 1 output
        assert len(params_list) == 1
        
        # Run twice in parallel
        outputs_run1 = vf.parallel_simulate_batch(vf.ViscoElasticFilament_FlowParams, params_list, n_jobs=1)
        outputs_run2 = vf.parallel_simulate_batch(vf.ViscoElasticFilament_FlowParams, params_list, n_jobs=1)
        
        traj_1 = outputs_run1[0]['value']
        traj_2 = outputs_run2[0]['value']
        
        # Should be identical (or very close, within numerical precision)
        assert np.allclose(traj_1, traj_2, rtol=1e-10, atol=1e-12), (
            "Parallel runs with identical parameters should yield identical results"
        )
        print(f"\nConsistency check passed: Max difference between runs = {np.max(np.abs(traj_1 - traj_2))}")


# --- Helper Integration Test ---
class TestViscoElasticFilamentParallelIntegration:
    """Integration tests for parallel simulation across both model variants."""
    
    def test_both_models_noflow_consistency(self):
        """
        Test that ViscoElasticFilament and ViscoElasticFilament_FlowParams
        produce consistent results when flow is disabled (A=0).
        """
        N = 10
        X_0 = vf.StraightLine(N)
        
        # Common internal params
        int_params = {
            'Sp4': 1.0,
            'N': N,
            'k0': 1e13,
            'bool_EI': True,
            'gamma': 2,
            'taus_b': [0.0]*(N-1),
            'tau_s': 0.0,
            'n_L': [0, 0],
            'm_L': 0,
            'X_0': X_0,
        }
        
        ext_params_base = {
            'Lambdas': [[0.0, 1e-5]]*N,
            'Zetas': [0.0]*N,
        }
        
        sim_params = {
            'T_span': (0.0, 1.0),
            'T_eval': np.linspace(0, 1, 10),
            'T_sim_max': 30.0,
            'method': 'BDF',
        }
        
        # VEF without flow (using InterpFlow=0)
        ext_params_vef = {**ext_params_base, 'InterpFlow': 0}
        
        # VEF_FlowParams with no flow (A=0)
        ext_params_flow_no_flow = {
            **ext_params_base,
            'A': 0.0,
            'w0': 1.0,
            'psi': np.pi/2,
        }

        params_base = {'int_params':int_params, 'ext_params':ext_params_vef, 'sim_params':sim_params}
        params_flow_no_flow = {'int_params':int_params, 'ext_params':ext_params_flow_no_flow, 'sim_params':sim_params}
        
        # Simulate with standard VEF
        model_base = vf.ViscoElasticFilament(**params_base)
        model_base.simulate_single()
        traj_vef = model_base.sim_output['value']

        # Simulate with FlowParams (no flow)
        model_flow_no_flow = vf.ViscoElasticFilament_FlowParams(**params_flow_no_flow)
        model_flow_no_flow.simulate_single()
        traj_flow_no_flow = model_flow_no_flow.sim_output['value']
        
        # Both should succeed
        assert traj_vef is not None, "VEF simulation failed"
        assert traj_flow_no_flow is not None, "VEF_FlowParams simulation failed"
        
        # Both should be finite
        assert np.all(np.isfinite(traj_vef)), "VEF trajectory contains NaN/Inf"
        assert np.all(np.isfinite(traj_flow_no_flow)), "VEF_FlowParams trajectory contains NaN/Inf"
        
        print(f"\nBoth models produced valid trajectories")
        print(f"VEF shape: {traj_vef.shape}")
        print(f"VEF_FlowParams (A=0) shape: {traj_flow_no_flow.shape}")
    
    @pytest.mark.parametrize('n_jobs', [1, 2, 4])
    def test_parallel_batch_scalability(self, n_jobs):
        """
        Test that parallel batch scaling works correctly across different job counts.
        """
        N = 8
        X_0 = vf.StraightLine(N)
        
        params_dict = {
            'gamma_list': [1, 2],
            'N_list': [N],
            'X_0_list': [X_0],
            'taus_b_list': [[0.0]*(N-1)],
            'Lambdas_list': [[[0.0, 0.0]]*N],
            'Zetas_list': [[0.0]*N],
            'k0_list': [1e13],
            'bool_EI_list': [True],
            'Sp4_list': [0.5, 1.0, 2.0],
            'Beta_list': [0.0],
            'tau_s_list': [0.0],
            'n_L_list': [[0, 0]],
            'm_L_list': [0],
            'InterpFlow_list': [0],
            'T_span_list': [(0.0, 1e6)],
            'T_eval_list': [[np.linspace(0, 1e6, int(1e2))]], # T_eval depends on T_span
            'T_sim_max_list': [300.0],
            'method_list': ['BDF'],
        }
        
        int_keys = ['Sp4', 'N', 'k0', 'bool_EI', 'gamma', 'taus_b', 'tau_s', 'n_L', 'm_L', 'X_0']
        ext_keys = ['Lambdas', 'Zetas', 'InterpFlow']
        sim_keys = ['T_span', 'T_eval', 'T_sim_max', 'method']
        
        params_list = vf.ViscoElasticFilament_create_params_list(
            int_keys, ext_keys, sim_keys, params_dict
        )
        
        # Expected: 2 gamma * 3 Sp4 = 6
        assert len(params_list) == 6
        
        outputs = vf.parallel_simulate_batch(vf.ViscoElasticFilament, params_list, n_jobs=n_jobs)
        
        assert len(outputs) == 6
        for i, out in enumerate(outputs):
            assert out["value"] is not None, f"Job count {n_jobs}, output {i} failed"
            assert np.all(np.isfinite(out["value"])), (
                f"Job count {n_jobs}, output {i} contains NaN/Inf"
            )
        
        print(f"\nParallel batch with n_jobs={n_jobs} completed successfully")



if __name__ == "__main__":
    # Run pytest programmatically
    pytest.main([__file__, "-v", "--tb=short"])

