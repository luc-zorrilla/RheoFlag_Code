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
class Model():
    """
    Base model class. Handles forward simulation only.
    
    Stores original parameters for inspection and composition chaining.
    """
    int_params: Any
    ext_params: Any
    sim_params: Any
    sim_output: Optional[Dict[str, Any]] = field(default=None, init=False, repr=True)
    
    # Track original parameters (set in __post_init__)
    _orig_int_params: Any = field(default=None, init=False, repr=False)
    _orig_ext_params: Any = field(default=None, init=False, repr=False)
    _orig_sim_params: Any = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Initialize original parameter tracking if not already set."""
        # Store originals BEFORE any composition modifies them
        if self._orig_int_params is None:
            self._orig_int_params = copy.deepcopy(self.int_params)
        if self._orig_ext_params is None:
            self._orig_ext_params = copy.deepcopy(self.ext_params)
        if self._orig_sim_params is None:
            self._orig_sim_params = copy.deepcopy(self.sim_params)

    @classmethod
    def compose(
        cls,
        compose_int_params: Optional[Callable[[Any, Any, Any]]] = None,
        compose_ext_params: Optional[Callable[[Any, Any, Any]]] = None,
        compose_sim_params: Optional[Callable[[Any, Any, Any]]] = None,
    ) -> Type[Model]:
        """
        Create a new Model subclass with composed parameters.
        """
        class ComposedModel(cls):

            def __post_init__(self):
                # Call parent's __post_init__ first (stores originals NOW)
                super().__post_init__()
                
                # NOW apply compositions to the current parameters
                if compose_int_params:
                    self.int_params = compose_int_params(
                        self._orig_int_params, 
                        self._orig_ext_params, 
                        self._orig_sim_params,
                    )
                if compose_ext_params:
                    self.ext_params = compose_ext_params(
                        self._orig_int_params, 
                        self._orig_ext_params, 
                        self._orig_sim_params,
                    )
                if compose_sim_params:
                    self.sim_params = compose_sim_params(
                        self._orig_int_params, 
                        self._orig_ext_params, 
                        self._orig_sim_params,
                    )

        ComposedModel = dataclass(ComposedModel) # Enforce post_init
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
            # Store originals to reconstruct composed models correctly
            '_orig_int_params': self._orig_int_params,
            '_orig_ext_params': self._orig_ext_params,
            '_orig_sim_params': self._orig_sim_params,
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
        
        # Restore original parameters (important for composed models)
        instance._orig_int_params = data['_orig_int_params']
        instance._orig_ext_params = data['_orig_ext_params']
        instance._orig_sim_params = data['_orig_sim_params']
        
        # Populate the derived sim_output
        instance.sim_output = {
            'value': data['sim_output']['value'],
            'shape': data['sim_output']['shape']
        }
        
        return instance     

    def pickle_model(self, filepath):
        """Pickle entire Model instance (preserves all state including originals)."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f, protocol=-1)

    @classmethod
    def unpickle(cls, filepath) -> "Model":
        """Unpickle and restore a Model instance."""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        # Ensure originals are set (defensive against old pickles)
        if model._orig_int_params is None:
            model._orig_int_params = copy.deepcopy(model.int_params)
        if model._orig_ext_params is None:
            model._orig_ext_params = copy.deepcopy(model.ext_params)
        if model._orig_sim_params is None:
            model._orig_sim_params = copy.deepcopy(model.sim_params)
        
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