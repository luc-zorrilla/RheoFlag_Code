import pytest
import numpy as np
from Models import Model, Square, merge_models, merge_multiple_models, parallel_simulate_batch

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

# --- Tests for Square Model ---
def test_square_single(square_model):
    output = square_model.simulate_single()
    assert "value" in output
    assert "shape" in output
    assert np.allclose(output["value"], np.array([4.0]))
    assert output["shape"] == (1,)

def test_square_batch():
    int_params = [np.array([1.0]), np.array([2.0]), np.array([3.0])]
    ext_params = [None, None, None]
    sim_params = [None, None, None]
    outputs = Square.simulate_batch(int_params, ext_params, sim_params)
    assert len(outputs) == 3
    for i, out in enumerate(outputs):
        assert "value" in out
        assert "shape" in out
        assert np.allclose(out["value"], np.array([(i+1)**2]))
        assert out["shape"] == (1,)

# --- Tests for Model Merging ---
def test_merge_models(square_models):
    model1, model2 = square_models[0], square_models[1]
    MergedModel = merge_models(model1, model2, int_strategy="concat")
    merged = MergedModel()
    output = merged.simulate_single()
    assert "value" in output
    assert "shape" in output
    assert output["value"].shape == (2,1)
    assert np.allclose(output["value"], np.array([[1.], [4.]]))

def test_merge_multiple_models(square_models):
    MergedModel = merge_multiple_models(*square_models, int_strategy="concat")
    merged = MergedModel()
    output = merged.simulate_single()
    assert "value" in output
    assert "shape" in output
    assert output["value"].shape == (3,1)
    assert np.allclose(output["value"], np.array([[1.], [4.], [9.]]))

def test_merge_models_strategies(square_models):
    model1, model2 = square_models[0], square_models[1]
    MergedModel = merge_models(model1, model2, int_strategy="model1")
    merged = MergedModel()
    output = merged.simulate_single()
    assert "value" in output
    assert "shape" in output
    assert np.allclose(output["value"], np.array([1.0]))

# --- Tests for Parallel Simulation ---
def test_parallel_simulate_batch():
    int_params = [np.array([i]) for i in range(1, 6)]
    ext_params = [None] * 5
    sim_params = [None] * 5
    outputs = parallel_simulate_batch(Square, int_params, ext_params, sim_params, batch_size=2)
    assert len(outputs) == 5
    for i, out in enumerate(outputs):
        assert "value" in out
        assert "shape" in out
        assert np.allclose(out["value"], np.array([(i+1)**2]))
        assert out["shape"] == (1,)

# --- Tests for Output Shape Compatibility ---
def test_output_shape_compatibility(square_model):
    output = square_model.simulate_single()
    ground_truth = {"value": np.array([4.0]), "shape": (1,)}
    assert output["shape"] == ground_truth["shape"]
    assert np.allclose(output["value"], ground_truth["value"])

def test_output_shape_mismatch(square_model):
    output = square_model.simulate_single()
    ground_truth = {"value": np.array([4.0]), "shape": (2,)}
    with pytest.raises(AssertionError):
        assert output["shape"] == ground_truth["shape"]