from joblib import Parallel, delayed
import json
import dill as pickle # enhanced pickle library that handles function pickling as well
import copy
import numpy as np

from typing import Any, Callable, Iterable, List, Sequence, Type, Optional, Tuple, Dict, Literal
from dataclasses import dataclass, field
from itertools import zip_longest, product
from pathlib import Path

class NumpyTupleEncoder(json.JSONEncoder):
    """Custom JSON encoder that recursively marks numpy arrays and tuples with type metadata."""
    
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {
                "__numpy_array__": True,
                "dtype": str(obj.dtype),
                "value": obj.tolist()
            }
        elif isinstance(obj, tuple):
            return {
                "__tuple__": True,
                "value": list(obj)
            }
        return super().default(obj)
    
    def encode(self, o):
        """Pre-process the entire object tree to mark tuples and arrays."""
        o = self._mark_types(o)
        return super().encode(o)
    
    @staticmethod
    def _mark_types(obj):
        """Recursively walk the object tree and mark numpy arrays and tuples."""
        if isinstance(obj, np.ndarray):
            return {
                "__numpy_array__": True,
                "dtype": str(obj.dtype),
                "value": obj.tolist()
            }
        elif isinstance(obj, tuple):
            return {
                "__tuple__": True,
                "value": [NumpyTupleEncoder._mark_types(item) for item in obj]
            }
        elif isinstance(obj, dict):
            return {key: NumpyTupleEncoder._mark_types(val) for key, val in obj.items()}
        elif isinstance(obj, list):
            return [NumpyTupleEncoder._mark_types(item) for item in obj]
        return obj

class NumpyTupleDecoder(json.JSONDecoder):
    """Custom JSON decoder that recursively reconstructs numpy arrays and tuples."""
    def __init__(self, *args, **kwargs):
        super().__init__(object_hook=self.object_hook, *args, **kwargs)

    @staticmethod
    def object_hook(obj):
        """Recursively process all dictionary objects."""
        if isinstance(obj, dict):
            # Check for numpy array marker
            if obj.get("__numpy_array__"):
                return np.array(obj["value"], dtype=obj["dtype"])
            # Check for tuple marker
            elif obj.get("__tuple__"):
                # Recursively decode tuple elements in case they contain arrays/tuples
                decoded_value = [
                    NumpyTupleDecoder.object_hook(item) if isinstance(item, dict) else item
                    for item in obj["value"]
                ]
                return tuple(decoded_value)
            else:
                # Recursively process regular dictionary values
                return {key: NumpyTupleDecoder.object_hook(val) if isinstance(val, dict) else val 
                        for key, val in obj.items()}
        elif isinstance(obj, list):
            # Recursively process list elements
            return [NumpyTupleDecoder.object_hook(item) if isinstance(item, dict) else item 
                    for item in obj]
        return obj

# --- Core model base ---
@dataclass
class Model:
    """
    Base model class. Handles forward simulation only

    Subclasses SHOULD:
    - implement simulate_single(self) -> Dict[str, Any] (with keys "value" and "shape")
    - optionally implement simulate_batch(cls, ...) for performance.

    Constructor semantics:
    - int_params: numpy array or scalar representing internal parameters
    - ext_params: external parameters for this instance (may be None)
    - sim_params: simulation parameters / options (may be None)
    - sim_output: initialized as None; populated after simulate_single() or simulate_batch() is called
    """

    int_params: Any
    ext_params: Any
    sim_params: Any
    sim_output: Optional[Dict[str, Any]] = field(default=None, init=False, repr=True)

    def __post_init__(self):
        """Hook for subclasses to customize parameter initialization."""
        pass

    # # ========== Simulation ==========
    # """Core forward simulation interface."""

    # @abstractmethod
    # def simulate_single(self) -> Dict[str, Any]:
    #     """
    #     Run forward simulation for a single instance. Must be overridden.
    #     Subclasses should populate self.sim_output with the result before returning.
    #     Returns: {"value": np.ndarray, "shape": tuple}
    #     """
    #     raise NotImplementedError

    # ========== Factory Methods ==========
    """Create Model instances with special initialization logic."""

    @classmethod
    def compose(
        cls,
        compose_int_params: Optional[Callable[[Any, Any, Any], Any]] = None,
        compose_ext_params: Optional[Callable[[Any, Any, Any], Any]] = None,
        compose_sim_params: Optional[Callable[[Any, Any, Any], Any]] = None,
    ) -> Type[Model]:
        """
        Create a new Model subclass with composed parameters.
        """

        class ComposedModel(cls):
            def __post_init__(self):

                # Call parent's __post_init__ if it exists
                if hasattr(super(), '__post_init__'):
                    super().__post_init__()

                """Called after dataclass __init__. Transform parameters here."""
                if compose_int_params:
                    self.int_params = compose_int_params(self.int_params, self.ext_params, self.sim_params)
                if compose_ext_params:
                    self.ext_params = compose_ext_params(self.int_params, self.ext_params, self.sim_params)
                if compose_sim_params:
                    self.sim_params = compose_sim_params(self.int_params, self.ext_params, self.sim_params)
                
        ComposedModel.__name__ = f"Composed{cls.__name__}"
        return ComposedModel


    # ========== Persistence ==========
    """Save and load model state (pickling, JSON, etc)."""

    def write_sim_output(self, filepath):
        """Write simulation output with numpy array and tuple preservation."""
        if self.sim_output is None:
            raise ValueError("No simulation output. Run simulate_single() first.")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'int_params': self.int_params,
            'ext_params': self.ext_params,
            'sim_params': self.sim_params,
            'sim_output': {
                'value': self.sim_output['value'],
                'shape': self.sim_output['shape']
            }
        }

        # Pre-process to mark all tuples and arrays
        data = NumpyTupleEncoder._mark_types(data)        
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, cls=NumpyTupleEncoder)

    @classmethod
    def read_sim_output(cls, filepath):
        """Read simulation output and restore numpy arrays and tuples."""
        with open(filepath, 'r') as f:
            data = json.load(f, cls=NumpyTupleDecoder)

        # Create instance with loaded parameters
        instance = cls(
            int_params=data['int_params'],
            ext_params=data['ext_params'],
            sim_params=data['sim_params']
        )
        
        # Populate the derived sim_output
        instance.sim_output = {
            'value': data['sim_output']['value'],
            'shape': data['sim_output']['shape']
        }
        
        return instance     

    def pickle_model(self, filepath):
        """Pickle entire Model instance (preserves all state)."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f, protocol=-1)

    @classmethod
    def unpickle(cls, filepath) -> "Model":
        """Unpickle and restore a Model instance."""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        return model

@dataclass
class ModelList:
    """Container for batch simulation of multiple Model instances."""
    
    models: List[Model] = field(default_factory=list)
    results: Optional[List[Dict[str, Any]]] = field(default=None, init=False, repr=False)
    
    # ========== Simulation ==========
    """Core forward simulation interface."""

    def simulate(self, n_jobs: int = -1, parallel: bool = True) -> List[Dict[str, Any]]:
        """
        Run simulation on all models in parallel.
        
        Args:
            n_jobs: Number of jobs (-1 = all cores, 1 = serial). Defaults to -1.
        """
            
        if not parallel:
            self.results = [model.simulate_single() for model in self.models]
        else:
            self.results = Parallel(n_jobs=n_jobs)(
                delayed(model.simulate_single)() for model in self.models
            )
        return self.results

    # ========== Factory Methods ==========
    """Create Model instances with special initialization logic."""

    @classmethod
    def from_params(
        cls,
        model_class: type,
        int_params_batch: Sequence[Any],
        ext_params_batch: Sequence[Any],
        sim_params_batch: Sequence[Any],
    ) -> "ModelList":
        """Create a ModelList from batched parameters."""
        models = [
            model_class(ip, ep, sp)
            for ip, ep, sp in zip(int_params_batch, ext_params_batch, sim_params_batch)
        ]
        return cls(models=models)

    # ========== Persistence ==========
    """Save and load model state (pickling, JSON, etc)."""

    def write_all(self, dirpath):
        """Write all models to separate files."""
        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)
        
        for idx, model in enumerate(self.models):
            model.write_sim_output(dirpath / f"model_{idx:04d}.json")
    
    @classmethod
    def read_all(cls, filepath_list) -> "ModelList":
        """Read all models from a list of paths.
        Note: this method will usually not be used. """
        models = []
        
        for filepath in filepath_list:
            instance = Model.read_sim_output(filepath)
            models.append(instance)
        
        return cls(models=models)

# ========== Simple Model Implementation ==========

@dataclass
class SimpleModel(Model):
    """A simple concrete model for testing."""
    
    def simulate_single(self) -> Dict[str, Any]:
        """Simple simulation: multiply int_params by ext_params."""
        result = self.int_params * self.ext_params
        self.sim_output = {
            'value': np.array([result]),
            'shape': (1,)
        }
        return self.sim_output


# ========== Test: Double Composition ==========

def test_double_composition():
    """Test composing a model twice with different parameter transformations."""
    
    print("=" * 60)
    print("TEST: Double Composition")
    print("=" * 60)
    
    # --- Composition 1: Scale internal parameters by 2 ---
    def compose_int_params_v1(int_p, ext_p, sim_p):
        """First composition: scale internal params by 2."""
        print(f"  [Compose 1] int_params: {int_p} -> {int_p * 2}")
        return int_p * 2
    
    ComposedModel_v1 = SimpleModel.compose(
        compose_int_params=compose_int_params_v1
    )
    
    # --- Composition 2: Add offset to external parameters ---
    def compose_ext_params_v2(int_p, ext_p, sim_p):
        """Second composition: add 10 to external params."""
        print(f"  [Compose 2] ext_params: {ext_p} -> {ext_p + 10}")
        return ext_p + 10
    
    DoubleComposedModel = ComposedModel_v1.compose(
        compose_ext_params=compose_ext_params_v2
    )
    
    # --- Test Execution ---
    print("\n1. Creating instance with int_params=5, ext_params=3")
    instance = DoubleComposedModel(
        int_params=5,
        ext_params=3,
        sim_params=None
    )
    
    print(f"   After composition:")
    print(f"   - int_params: {instance.int_params} (expected: 10)")
    print(f"   - ext_params: {instance.ext_params} (expected: 13)")
    
    print("\n2. Running simulation")
    result = instance.simulate_single()
    expected_value = 10 * 13  # (5*2) * (3+10) = 10 * 13 = 130
    actual_value = result['value'][0]
    
    print(f"   Simulation output: {actual_value}")
    print(f"   Expected: {expected_value}")
    
    # --- Assertions (manual validation) ---
    assert instance.int_params == 10, f"int_params should be 10, got {instance.int_params}"
    assert instance.ext_params == 13, f"ext_params should be 13, got {instance.ext_params}"
    assert actual_value == expected_value, f"Result should be {expected_value}, got {actual_value}"
    
    print("\n✓ All assertions passed!")
    print("=" * 60)

def test_double_composition_with_all_params():
    """Test double composition affecting all three parameter types."""
    
    print("\n" + "=" * 60)
    print("TEST: Double Composition (All Parameters)")
    print("=" * 60)
    
    # --- Composition 1: Modify all three parameter types ---
    def compose_all_v1(int_p, ext_p, sim_p):
        print(f"  [Compose 1] int_params: {int_p} -> {int_p + 1}")
        return int_p + 1
    
    def compose_ext_v1(int_p, ext_p, sim_p):
        print(f"  [Compose 1] ext_params: {ext_p} -> {ext_p * 2}")
        return ext_p * 2
    
    def compose_sim_v1(int_p, ext_p, sim_p):
        print(f"  [Compose 1] sim_params: {sim_p} -> 'modified_v1'")
        return 'modified_v1'
    
    ComposedModel_v1 = SimpleModel.compose(
        compose_int_params=compose_all_v1,
        compose_ext_params=compose_ext_v1,
        compose_sim_params=compose_sim_v1
    )
    
    # --- Composition 2: Further modification ---
    def compose_int_v2(int_p, ext_p, sim_p):
        print(f"  [Compose 2] int_params: {int_p} -> {int_p * 10}")
        return int_p * 10
    
    # IMPORTANT: Compose v2 FIRST, then v1 as parent
    # This way v1 runs first (adds 1), then v2 runs (multiplies by 10)
    DoubleComposedModel = ComposedModel_v1.compose(
        compose_int_params=compose_int_v2
    )
    
    print("\n1. Creating instance with int_params=2, ext_params=5, sim_params='initial'")
    instance = DoubleComposedModel(
        int_params=2,
        ext_params=5,
        sim_params='initial'
    )
    
    print(f"   After composition:")
    print(f"   - int_params: {instance.int_params} (expected: 30 = (2+1)*10)")
    print(f"   - ext_params: {instance.ext_params} (expected: 10 = 5*2)")
    print(f"   - sim_params: {instance.sim_params} (expected: 'modified_v1')")
    
    # --- Assertions ---
    assert instance.int_params == 30, f"int_params should be 30, got {instance.int_params}"
    assert instance.ext_params == 10, f"ext_params should be 10, got {instance.ext_params}"
    assert instance.sim_params == 'modified_v1', f"sim_params should be 'modified_v1', got {instance.sim_params}"
    
    print("\n✓ All assertions passed!")
    print("=" * 60)

def test_modellist_basic_creation():
    """Test basic ModelList creation and model storage."""
    
    print("\n" + "=" * 60)
    print("TEST: ModelList Basic Creation")
    print("=" * 60)
    
    # Create individual models
    models = [
        SimpleModel(int_params=1, ext_params=2, sim_params=None),
        SimpleModel(int_params=3, ext_params=4, sim_params=None),
        SimpleModel(int_params=5, ext_params=6, sim_params=None),
    ]
    
    # Create ModelList
    model_list = ModelList(models=models)
    
    print(f"Created ModelList with {len(model_list.models)} models")
    assert len(model_list.models) == 3, "Should have 3 models"
    assert model_list.results is None, "Results should be None before simulation"
    
    print("✓ ModelList created successfully")
    print("=" * 60)


def test_modellist_from_params():
    """Test ModelList.from_params factory method."""
    
    print("\n" + "=" * 60)
    print("TEST: ModelList.from_params Factory")
    print("=" * 60)
    
    int_params_batch = [1, 3, 5]
    ext_params_batch = [2, 4, 6]
    sim_params_batch = [None, None, None]
    
    model_list = ModelList.from_params(
        model_class=SimpleModel,
        int_params_batch=int_params_batch,
        ext_params_batch=ext_params_batch,
        sim_params_batch=sim_params_batch,
    )
    
    print(f"Created ModelList with {len(model_list.models)} models")
    assert len(model_list.models) == 3, "Should have 3 models"
    
    # Verify parameters are correctly assigned
    for idx, model in enumerate(model_list.models):
        assert model.int_params == int_params_batch[idx], \
            f"Model {idx} int_params mismatch"
        assert model.ext_params == ext_params_batch[idx], \
            f"Model {idx} ext_params mismatch"
    
    print("✓ All models created with correct parameters")
    print("=" * 60)


def test_modellist_simulate_serial():
    """Test ModelList.simulate() in serial mode."""
    
    print("\n" + "=" * 60)
    print("TEST: ModelList Serial Simulation")
    print("=" * 60)
    
    int_params_batch = [2, 3, 4]
    ext_params_batch = [5, 6, 7]
    
    model_list = ModelList.from_params(
        model_class=SimpleModel,
        int_params_batch=int_params_batch,
        ext_params_batch=ext_params_batch,
        sim_params_batch=[None] * 3,
    )
    
    print("Running serial simulation (n_jobs=1)...")
    results = model_list.simulate(n_jobs=1, parallel=False)
    
    assert len(results) == 3, "Should have 3 results"
    assert model_list.results is not None, "ModelList.results should be set"
    
    # Verify results
    expected_values = [2*5, 3*6, 4*7]  # [10, 18, 28]
    for idx, (result, expected) in enumerate(zip(results, expected_values)):
        actual = result['value'][0]
        print(f"  Model {idx}: {actual} (expected: {expected})")
        assert actual == expected, f"Model {idx} result mismatch"
    
    print("✓ Serial simulation completed correctly")
    print("=" * 60)


def test_modellist_simulate_parallel():
    """Test ModelList.simulate() in parallel mode."""
    
    print("\n" + "=" * 60)
    print("TEST: ModelList Parallel Simulation")
    print("=" * 60)
    
    int_params_batch = [2, 3, 4, 5]
    ext_params_batch = [5, 6, 7, 8]
    
    model_list = ModelList.from_params(
        model_class=SimpleModel,
        int_params_batch=int_params_batch,
        ext_params_batch=ext_params_batch,
        sim_params_batch=[None] * 4,
    )
    
    print("Running parallel simulation (n_jobs=-1)...")
    results = model_list.simulate(n_jobs=-1, parallel=True)
    
    assert len(results) == 4, "Should have 4 results"
    assert model_list.results is not None, "ModelList.results should be set"
    
    # Verify results match serial computation
    expected_values = [2*5, 3*6, 4*7, 5*8]  # [10, 18, 28, 40]
    for idx, (result, expected) in enumerate(zip(results, expected_values)):
        actual = result['value'][0]
        print(f"  Model {idx}: {actual} (expected: {expected})")
        assert actual == expected, f"Model {idx} result mismatch"
    
    print("✓ Parallel simulation completed correctly")
    print("=" * 60)


def test_modellist_with_composed_models():
    """Test ModelList with composed models."""
    
    print("\n" + "=" * 60)
    print("TEST: ModelList with Composed Models")
    print("=" * 60)
    
    # Create a composed model that scales int_params by 2
    def compose_int_params(int_p, ext_p, sim_p):
        return int_p * 2
    
    ComposedModel = SimpleModel.compose(
        compose_int_params=compose_int_params
    )
    
    # Create batch with composed model
    int_params_batch = [1, 2, 3]
    ext_params_batch = [5, 6, 7]
    
    model_list = ModelList.from_params(
        model_class=ComposedModel,
        int_params_batch=int_params_batch,
        ext_params_batch=ext_params_batch,
        sim_params_batch=[None] * 3,
    )
    
    print("Created ModelList with composed models")
    print("Running simulation...")
    results = model_list.simulate(n_jobs=1, parallel=False)
    
    # Expected: (int_params * 2) * ext_params
    expected_values = [1*2*5, 2*2*6, 3*2*7]  # [10, 24, 42]
    for idx, (result, expected) in enumerate(zip(results, expected_values)):
        actual = result['value'][0]
        print(f"  Model {idx}: {actual} (expected: {expected})")
        assert actual == expected, f"Model {idx} result mismatch"
    
    print("✓ Composed models in ModelList work correctly")
    print("=" * 60)


def test_modellist_with_double_composition():
    """Test ModelList with doubly-composed models."""
    
    print("\n" + "=" * 60)
    print("TEST: ModelList with Double-Composed Models")
    print("=" * 60)
    
    # First composition: scale int_params by 2
    def compose_int_v1(int_p, ext_p, sim_p):
        return int_p * 2
    
    # Second composition: scale ext_params by 3
    def compose_ext_v2(int_p, ext_p, sim_p):
        return ext_p * 3
    
    ComposedModel_v1 = SimpleModel.compose(
        compose_int_params=compose_int_v1
    )
    
    DoubleComposedModel = ComposedModel_v1.compose(
        compose_ext_params=compose_ext_v2
    )
    
    # Create batch
    int_params_batch = [1, 2, 3]
    ext_params_batch = [5, 6, 7]
    
    model_list = ModelList.from_params(
        model_class=DoubleComposedModel,
        int_params_batch=int_params_batch,
        ext_params_batch=ext_params_batch,
        sim_params_batch=[None] * 3,
    )
    
    print("Created ModelList with double-composed models")
    print("Running simulation...")
    results = model_list.simulate(n_jobs=1, parallel=False)
    
    # Expected: (int_params * 2) * (ext_params * 3)
    expected_values = [1*2*5*3, 2*2*6*3, 3*2*7*3]  # [30, 72, 126]
    for idx, (result, expected) in enumerate(zip(results, expected_values)):
        actual = result['value'][0]
        print(f"  Model {idx}: {actual} (expected: {expected})")
        assert actual == expected, f"Model {idx} result mismatch"
    
    print("✓ Double-composed models in ModelList work correctly")
    print("=" * 60)

# ========== Run Tests ==========

if __name__ == "__main__":
    test_double_composition()
    test_double_composition_with_all_params()
    test_modellist_basic_creation()
    test_modellist_from_params()
    test_modellist_simulate_serial()
    test_modellist_simulate_parallel()
    test_modellist_with_composed_models()
    test_modellist_with_double_composition()
    print("\n✅ All tests completed successfully!\n")
