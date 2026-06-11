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
    int_params: Any
    ext_params: Any
    sim_params: Any
    sim_output: Optional[Dict[str, Any]] = field(default=None, init=False, repr=True)

    def simulate_single(self) -> Dict[str, Any]:
        """Override in subclasses."""
        raise NotImplementedError

    @classmethod
    def compose(
        cls,
        compose_int_params: Optional[Callable[[Any, Any, Any], Any]] = None,
        compose_ext_params: Optional[Callable[[Any, Any, Any], Any]] = None,
        compose_sim_params: Optional[Callable[[Any, Any, Any], Any]] = None,
    ) -> Type["Model"]:
        """Create a subclass that transforms parameters before simulation."""

        class ComposedModel(cls):


            
            def simulate_single(self) -> Dict[str, Any]:
                int_params = self.int_params
                ext_params = self.ext_params
                sim_params = self.sim_params
                
                if compose_int_params:
                    int_params = compose_int_params(int_params, ext_params, sim_params)
                if compose_ext_params:
                    ext_params = compose_ext_params(int_params, ext_params, sim_params)
                if compose_sim_params:
                    sim_params = compose_sim_params(int_params, ext_params, sim_params)
                
                # Temporarily swap and call parent's simulate_single
                orig_int, orig_ext, orig_sim = self.int_params, self.ext_params, self.sim_params
                self.int_params, self.ext_params, self.sim_params = int_params, ext_params, sim_params
                try:
                    # Call parent class's simulate_single, skipping ComposedModel's version
                    return super(ComposedModel, self).simulate_single()
                finally:
                    self.int_params, self.ext_params, self.sim_params = orig_int, orig_ext, orig_sim

        ComposedModel.__name__ = f"Composed{cls.__name__}"
        return ComposedModel

    @classmethod
    def reduce(cls, param_keys_to_keep: List[str]) -> Type:
        """
        Create a reduced model class that only tracks specified internal parameters.
        
        The resulting model will have a smaller int_params dict containing only
        the specified keys. Other parameters are assumed to be handled externally
        (e.g., fixed from prior inference passes).
        
        Args:
            param_keys_to_keep: List of internal parameter keys to retain
            
        Returns:
            A new Model subclass with reduced int_params
        """
        base_model = cls
        
        def compose_int_params_reduced(int_params, ext_params, sim_params):
            """Keep only specified parameter keys."""
            if isinstance(int_params, dict):
                return {
                    key: int_params[key]
                    for key in param_keys_to_keep
                    if key in int_params
                }
            return int_params
        
        return base_model.compose(
            compose_int_params=compose_int_params_reduced,
        )

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
        # Extract scalar from dict if int_params is a dict
        int_val = self.int_params['int_params'] if isinstance(self.int_params, dict) else self.int_params
        ext_val = self.ext_params if not isinstance(self.ext_params, dict) else self.ext_params.get('value', self.ext_params)
        
        result = int_val * ext_val
        self.sim_output = {
            'value': np.array([result]),
            'shape': (1,)
        }
        return self.sim_output

# ========== Tests ================
# --- Simple Model for Testing ---
@dataclass
class SquareModel(Model):
    """Simple model that squares the internal parameter 'x'."""
    
    def simulate_single(self) -> Dict[str, Any]:
        """Square the internal parameter x."""
        x = self.int_params['x']
        result = {
            'value': x ** 2,
            'shape': (1,),
        }
        self.sim_output = result
        return result


# --- Composition Functions ---
def double_x(int_params, ext_params, sim_params):
    """Transform: x -> 2*x"""
    new_params = int_params.copy()
    new_params['x'] *= 2
    return new_params

def apply_offset(int_params, ext_params, sim_params):
    """Transform int_params using ext_params['offset']"""
    new_params = int_params.copy()
    offset = ext_params.get('offset', 0)
    new_params['x'] += offset
    return new_params

def add_sim_scale(int_params, ext_params, sim_params):
    """Transform: x -> x * scale (from sim_params)"""
    new_params = int_params.copy()
    scale = sim_params.get('scale', 1)
    new_params['x'] *= scale
    return new_params


# --- Tests ---
def test_no_composition():
    """Test basic model without composition."""
    print("Test 1: No composition")
    model = SquareModel(
        int_params={'x': 3},
        ext_params=None,
        sim_params=None,
    )
    result = model.simulate_single()
    assert result['value'] == 9, f"Expected 9, got {result['value']}"
    assert result['x_original'] == 3
    assert model.int_params['x'] == 3, "Original params should be unchanged"
    print(f"  ✓ x=3 -> x²=9 (original x={model.int_params['x']})")


def test_single_composition_int_params():
    """Test single composition on int_params."""
    print("\nTest 2: Single composition (double_x)")
    SquareModel_Double = SquareModel.compose(compose_int_params=double_x)
    
    model = SquareModel_Double(
        int_params={'x': 3},
        ext_params=None,
        sim_params=None,
    )
    result = model.simulate_single()
    # x=3 -> double_x -> x=6 -> square -> 36
    assert result['value'] == 36, f"Expected 36, got {result['value']}"
    assert model.int_params['x'] == 3, "Original params should be unchanged"
    print(f"  ✓ x=3 -[double_x]-> x=6 -> x²=36 (original x={model.int_params['x']})")

def test_single_composition_ext_params():
    """Test single composition on ext_params."""
    print("\nTest 3: Single composition (apply_offset via int_params)")
    SquareModel_Offset = SquareModel.compose(compose_int_params=apply_offset)
    
    model = SquareModel_Offset(
        int_params={'x': 3},
        ext_params={'offset': 2},
        sim_params=None,
    )
    result = model.simulate_single()
    # x=3 -> apply_offset(offset=2) -> x=5 -> square -> 25
    assert result['value'] == 25, f"Expected 25, got {result['value']}"
    assert model.int_params['x'] == 3, "Original int_params should be unchanged"
    assert model.ext_params['offset'] == 2, "Original ext_params should be unchanged"
    print(f"  ✓ x=3 -[add_offset(2)]-> x=5 -> x²=25 (original x={model.int_params['x']})")

def test_nested_compositions():
    """Test multiple compositions in single compose call."""
    print("\nTest 5: Multiple compositions (int + ext)")
    SquareModel_Composed = SquareModel.compose(
        compose_int_params=double_x,
    )
    SquareModel_Composed_Composed = SquareModel_Composed.compose(
        compose_int_params=double_x,
    )
    
    model = SquareModel_Composed_Composed(
        int_params={'x': 1},
        ext_params=None,
        sim_params=None,
    )
    result = model.simulate_single()
    # x=1 -> double_x -> x=2 -> double_x -> x=4 --> Square --> 16
    assert result['value'] == 16, f"Expected 16, got {result['value']}"
    assert model.int_params['x'] == 1
    print(f"  ✓ x=1 -[double_x]-> x=2 -[double_x]-> x=4 -> x²=16")

# Run all tests
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Composition Method")
    print("=" * 60)
    
    # test_no_composition() # OK
    # test_single_composition_int_params() # OK
    # test_single_composition_ext_params() # OK
    test_nested_compositions()
    
    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)