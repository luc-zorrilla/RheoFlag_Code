import pytest
import numpy as np
from Models import Model, merge_models, merge_multiple_models, compose_model, parallel_simulate_batch
from ViscoElasticFilament_Models import ViscoElasticFilament, FlowParams_to_InterpFlow

# --- Fixtures ---
@pytest.fixture
def square_model():
    return Square(int_params=np.array([2.0]), ext_params=None, sim_params=None)

@pytest.fixture
def square_models():
    return [
        Square(int_params=np.array([1.0]), ext_params=None, sim_params=None),
        Square(int_params=np.array([2.0]), ext_params=None, sim_params=None),
        Square(int_params=np.array([3.0]), ext_params=None, sim_params=None),
    ]

# --- Tests for ViscoElasticFilament ---
def test_visco_elastic_filament():

    # Create an instance of the composed model
    int_params = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    ext_params = {'InterpFlow':0, 'Lambdas': 1.0, 'Zetas': 1.0}
    sim_params = {'T_span':[0,10], 'T_eval':np.linspace(0, 10, 100), 'T_sim_max':3600, 'method':'BDF'}

    composed_model = ViscoElasticFilament(int_params, ext_params, sim_params)

    # Simulate single
    output = composed_model.simulate_single()
    assert "value" in output
    assert "shape" in output
    assert output["shape"] == (10, 100)  # Example shape, adjust as needed

# --- Tests for ViscoElasticFilament_FlowParams ---
def test_visco_elastic_filament_flow_params():
    # Define the ViscoElasticFilament_FlowParams class by composing the ViscoElasticFilament class
    ViscoElasticFilament_FlowParams = compose_model(
        ViscoElasticFilament,
        compose_ext_params=FlowParams_to_InterpFlow
    )

    # Create an instance of the composed model
    int_params = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    ext_params = {'A': 1.0, 'w0': np.pi/2, 'Lambdas': 1.0, 'Zetas': 1.0}
    sim_params = {'T_span':[0,10], 'T_eval':np.linspace(0, 10, 100), 'T_sim_max':3600, 'method':'BDF'}

    composed_model = ViscoElasticFilament_FlowParams(int_params, ext_params, sim_params)

    # Simulate single
    output = composed_model.simulate_single()
    assert "value" in output
    assert "shape" in output
    assert output["shape"] == (10, 100)  # Example shape, adjust as needed