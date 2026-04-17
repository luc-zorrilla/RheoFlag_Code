import pytest
import numpy as np
from Models import Model, Square, Square_create_params_list, unpickle_model, compose_model, parallel_simulate_batch # reduce_model, reconstruct_model
import tempfile
import shutil
import dill as pickle # enhanced pickle library that handles function pickling as well
from pathlib import Path
import json
from deepdiff import DeepDiff

# --- Fixtures ---
@pytest.fixture
def square_model():
    return Square(int_params={'x':np.array([2.0])}, ext_params=None, sim_params=None)

@pytest.fixture
def square_models():
    return [
        Square(int_params={'x':np.array([1.0])}, ext_params=None, sim_params=None),
        Square(int_params={'x':np.array([2.0])}, ext_params=None, sim_params=None),
        Square(int_params={'x':np.array([3.0])}, ext_params=None, sim_params=None),
    ]

# --- Tests for Square Model ---
def test_square_single(square_model):
    output = square_model.simulate_single()
    assert "value" in output
    assert "shape" in output
    assert np.allclose(output["value"], np.array([4.0]))
    assert output["shape"] == (1,)

def test_square_batch():
    int_params = [{'x':np.array([1.0])}, {'x':np.array([2.0])}, {'x':np.array([3.0])}]
    ext_params = [None, None, None]
    sim_params = [None, None, None]
    outputs = Square.simulate_batch(int_params, ext_params, sim_params)
    assert len(outputs) == 3
    for i, out in enumerate(outputs):
        assert "value" in out
        assert "shape" in out
        assert np.allclose(out["value"], np.array([(i+1)**2]))
        assert out["shape"] == (1,)

# --- Tests for Parallel Simulation ---
def test_parallel_simulate_batch():
    x_list = [np.array([i]) for i in range(1, 6)]

    # int_params, ext_params, sim_params --> params_list instead, through function Square_create_params_list
    int_params_keys = ['x']
    ext_params_keys = []
    sim_params_keys = []
    params_list_dict = {'x_list':x_list}
    params_list = Square_create_params_list(int_params_keys, ext_params_keys, sim_params_keys, params_list_dict)

    # Simulate in parallel
    outputs = parallel_simulate_batch(Square, params_list, n_jobs=4)
        
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

# --- Tests for compose_model ---
def test_compose_model_square():
    # Define a transformation function for int_params
    def transform_int_params(int_params, ext_params, sim_params):
        return {'x':int_params['x'] ** 0.5}

    # Compose the Square model with the transformation function
    ComposedSquare = compose_model(Square, compose_int_params=transform_int_params)

    # Create an instance of the composed model
    composed_model = ComposedSquare(int_params={'x':np.array([4.0])}, ext_params=None, sim_params=None)

    # Simulate single
    output = composed_model.simulate_single()
    assert "value" in output
    assert "shape" in output
    assert np.allclose(output["value"], np.array([4.0]))  # 2.0 ** 2
    assert output["shape"] == (1,)

def test_compose_model_with_ext_params():
    # Define a transformation function for ext_params
    def transform_ext_params(int_params, ext_params, sim_params):
        return {"transformed": True} if ext_params is None else ext_params

    # Compose the Square model with the transformation function
    ComposedSquare = compose_model(Square, compose_ext_params=transform_ext_params)

    # Create an instance of the composed model
    composed_model = ComposedSquare(int_params={'x':np.array([2.0])}, ext_params=None, sim_params=None)

    # Simulate single
    output = composed_model.simulate_single()
    assert "value" in output
    assert "shape" in output
    assert np.allclose(output["value"], np.array([4.0]))
    assert output["shape"] == (1,)

def test_compose_model_with_sim_params():
    # Define a transformation function for sim_params
    def transform_sim_params(int_params, ext_params, sim_params):
        return {"transformed": True} if sim_params is None else sim_params

    # Compose the Square model with the transformation function
    ComposedSquare = compose_model(Square, compose_sim_params=transform_sim_params)

    # Create an instance of the composed model
    composed_model = ComposedSquare(int_params={'x':np.array([2.0])}, ext_params=None, sim_params=None)

    # Simulate single
    output = composed_model.simulate_single()
    assert "value" in output
    assert "shape" in output
    assert np.allclose(output["value"], np.array([4.0]))
    assert output["shape"] == (1,)

class TestSquareModelSerialization:
    """Test suite for Square model serialization (write, read, pickle, unpickle)."""

    @pytest.fixture
    def temp_dir(self):
        """Create and clean up a temporary directory for test files."""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        shutil.rmtree(tmpdir)

    @pytest.fixture
    def square_instance(self):
        """Create a basic Square model instance with simulated output."""
        model = Square(int_params={'x': np.array([1.0, 2.0, 3.0])}, ext_params=None, sim_params=None)
        model.simulate_single()
        return model

    # ========== write_sim_output Tests ==========
    def test_write_sim_output_creates_file(self, square_instance, temp_dir):
        """Test that write_sim_output creates a file at the specified path."""
        filepath = Path(temp_dir) / "output.json"
        square_instance.write_sim_output(filepath)
        assert filepath.exists(), "Output file was not created."

    def test_write_sim_output_creates_parent_dirs(self, square_instance, temp_dir):
        """Test that write_sim_output creates parent directories if they don't exist."""
        filepath = Path(temp_dir) / "subdir1" / "subdir2" / "output.json"
        square_instance.write_sim_output(filepath)
        assert filepath.exists(), "Parent directories were not created."

    def test_write_sim_output_valid_json(self, square_instance, temp_dir):
        """Test that the written file contains valid JSON."""
        filepath = Path(temp_dir) / "output.json"
        square_instance.write_sim_output(filepath)
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        assert isinstance(data, dict), "JSON is not a dictionary."
        assert 'int_params' in data, "Missing 'int_params' key."
        assert 'ext_params' in data, "Missing 'ext_params' key."
        assert 'sim_params' in data, "Missing 'sim_params' key."
        assert 'sim_output' in data, "Missing 'sim_output' key."

    def test_write_sim_output_content_structure(self, square_instance, temp_dir):
        """Test that sim_output has correct structure: 'value' and 'shape'."""
        filepath = Path(temp_dir) / "output.json"
        square_instance.write_sim_output(filepath)
        
        square_instance.read_sim_output(filepath)
        
        assert 'value' in square_instance.sim_output, "Missing 'value' in sim_output."
        assert 'shape' in square_instance.sim_output, "Missing 'shape' in sim_output."
        assert isinstance(square_instance.sim_output['value'], np.ndarray), "'value' should be a list after JSON serialization."
        assert isinstance(square_instance.sim_output['shape'], tuple), "'shape' should be a list."

    def test_write_sim_output_raises_without_simulation(self, temp_dir):
        """Test that write_sim_output raises ValueError if simulate_single() hasn't been run."""
        model = Square(int_params={'x': 5.0}, ext_params=None, sim_params=None)
        filepath = Path(temp_dir) / "output.json"
        
        with pytest.raises(ValueError, match="No simulation output"):
            model.write_sim_output(filepath)

    # ========== read_sim_output Tests ==========
    def test_read_sim_output_restores_data(self, square_instance, temp_dir):
        """Test that read_sim_output correctly restores all data."""
        filepath = Path(temp_dir) / "output.json"
        square_instance.write_sim_output(filepath)
        
        new_model = Square(int_params=None, ext_params=None, sim_params=None)
        new_model.read_sim_output(filepath)

        assert DeepDiff(new_model.int_params, square_instance.int_params) == {}, "int_params not restored correctly."
        assert DeepDiff(new_model.ext_params, square_instance.ext_params) == {}, "ext_params not restored correctly."
        assert DeepDiff(new_model.sim_params, square_instance.sim_params) == {}, "sim_params not restored correctly."

    def test_read_sim_output_restores_sim_output(self, square_instance, temp_dir):
        """Test that sim_output is correctly restored as numpy arrays."""
        filepath = Path(temp_dir) / "output.json"
        square_instance.write_sim_output(filepath)
        
        new_model = Square(int_params=None, ext_params=None, sim_params=None)
        new_model.read_sim_output(filepath)
        
        assert isinstance(new_model.sim_output['value'], np.ndarray), "'value' should be a numpy array."
        np.testing.assert_array_equal(new_model.sim_output['value'], square_instance.sim_output['value'])
        assert new_model.sim_output['shape'] == square_instance.sim_output['shape'], "Shape not restored correctly."

    def test_read_sim_output_file_not_found(self, temp_dir):
        """Test that read_sim_output raises FileNotFoundError if file doesn't exist."""
        model = Square(int_params=None, ext_params=None, sim_params=None)
        filepath = Path(temp_dir) / "nonexistent.json"
        
        with pytest.raises(FileNotFoundError):
            model.read_sim_output(filepath)

    # ========== Round-trip Tests (write → read) ==========
    def test_write_read_round_trip(self, temp_dir):
        """Test complete round-trip: simulate → write → read → verify."""
        original_model = Square(int_params={'x': np.array([2.0, 4.0, 6.0])}, ext_params=None, sim_params=None)
        original_model.simulate_single()
        
        filepath = Path(temp_dir) / "roundtrip.json"
        original_model.write_sim_output(filepath)
        
        restored_model = Square(int_params=None, ext_params=None, sim_params=None)
        restored_model.read_sim_output(filepath)
        
        np.testing.assert_array_equal(restored_model.sim_output['value'], original_model.sim_output['value'])
        assert restored_model.sim_output['shape'] == original_model.sim_output['shape']

    # ========== pickle_model Tests ==========
    def test_pickle_model_creates_file(self, square_instance, temp_dir):
        """Test that pickle_model creates a pickle file."""
        filepath = Path(temp_dir) / "model.pkl"
        square_instance.pickle_model(filepath)
        assert filepath.exists(), "Pickle file was not created."

    def test_pickle_model_creates_parent_dirs(self, square_instance, temp_dir):
        """Test that pickle_model creates parent directories."""
        filepath = Path(temp_dir) / "subdir1" / "subdir2" / "model.pkl"
        square_instance.pickle_model(filepath)
        assert filepath.exists(), "Parent directories were not created."

    def test_pickle_model_valid_pickle(self, square_instance, temp_dir):
        """Test that the pickled file contains a valid pickle object."""
        filepath = Path(temp_dir) / "model.pkl"
        square_instance.pickle_model(filepath)
        
        with open(filepath, 'rb') as f:
            loaded = pickle.load(f)
        
        assert isinstance(loaded, Square), "Loaded object is not a Square instance."

    # ========== unpickle_model Tests ==========
    def test_unpickle_model_restores_instance(self, square_instance, temp_dir):
        """Test that unpickle_model correctly restores the model instance."""
        filepath = Path(temp_dir) / "model.pkl"
        square_instance.pickle_model(filepath)
        
        restored_model = unpickle_model(filepath)
        
        assert isinstance(restored_model, Square), "Restored object is not a Square instance."
        assert DeepDiff(restored_model.int_params, square_instance.int_params) == {}
        assert DeepDiff(restored_model.ext_params, square_instance.ext_params) == {}
        assert DeepDiff(restored_model.sim_params, square_instance.sim_params) == {}

    def test_unpickle_model_restores_sim_output(self, square_instance, temp_dir):
        """Test that unpickle_model preserves sim_output with numpy arrays."""
        filepath = Path(temp_dir) / "model.pkl"
        square_instance.pickle_model(filepath)
        
        restored_model = unpickle_model(filepath)
        
        assert restored_model.sim_output is not None, "sim_output was not restored."
        np.testing.assert_array_equal(restored_model.sim_output['value'], square_instance.sim_output['value'])
        assert restored_model.sim_output['shape'] == square_instance.sim_output['shape']

    def test_unpickle_model_file_not_found(self, temp_dir):
        """Test that unpickle_model raises FileNotFoundError if file doesn't exist."""
        filepath = Path(temp_dir) / "nonexistent.pkl"
        
        with pytest.raises(FileNotFoundError):
            unpickle_model(filepath)

    # ========== Round-trip Tests (pickle → unpickle) ==========
    def test_pickle_unpickle_round_trip(self, temp_dir):
        """Test complete round-trip: simulate → pickle → unpickle → verify."""
        original_model = Square(int_params={'x': np.array([5.0, 10.0])}, ext_params={'ext': 'value'}, sim_params={'option': True})
        original_model.simulate_single()
        
        filepath = Path(temp_dir) / "roundtrip.pkl"
        original_model.pickle_model(filepath)
        
        restored_model = unpickle_model(filepath)
        
        np.testing.assert_array_equal(restored_model.sim_output['value'], original_model.sim_output['value'])
        assert DeepDiff(restored_model.int_params, original_model.int_params) == {}
        assert DeepDiff(restored_model.ext_params, original_model.ext_params) == {}
        assert DeepDiff(restored_model.sim_params, original_model.sim_params) == {}

    # ========== Cross-format Tests ==========
    def test_json_vs_pickle_equivalence(self, temp_dir):
        """Test that JSON and pickle serialization produce equivalent results."""
        original_model = Square(int_params={'x': np.array([1.0, 2.0, 3.0])}, ext_params=None, sim_params=None)
        original_model.simulate_single()
        
        json_path = Path(temp_dir) / "model.json"
        pkl_path = Path(temp_dir) / "model.pkl"
        
        original_model.write_sim_output(json_path)
        original_model.pickle_model(pkl_path)
        
        json_model = Square(int_params=None, ext_params=None, sim_params=None)
        json_model.read_sim_output(json_path)
        
        pkl_model = unpickle_model(pkl_path)
        
        np.testing.assert_array_equal(json_model.sim_output['value'], pkl_model.sim_output['value'])
        assert json_model.sim_output['shape'] == pkl_model.sim_output['shape']