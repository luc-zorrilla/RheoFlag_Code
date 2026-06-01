from joblib import Parallel, delayed
from typing import Any, Callable, Iterable, List, Sequence, Type, Optional, Tuple, Dict, Literal
from abc import abstractmethod
from dataclasses import dataclass, field
import numpy as np
import logging
from itertools import zip_longest, product
import json
import dill as pickle # enhanced pickle library that handles function pickling as well
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

# TODO: remove when .compose() is debugged
# def compose_model(
#     model_class: Type[Model],
#     compose_int_params: Optional[Callable[[Any], Any]] = None,
#     compose_ext_params: Optional[Callable[[Any], Any]] = None,
#     compose_sim_params: Optional[Callable[[Any], Any]] = None
# ) -> Type[Model]:
#     """
#     Create a new Model class by composing input parameters with functions.

#     The composed model applies composition functions to the internal, external, and
#     simulation parameters before passing them to the base model's simulate_single() method.

#     Args:
#         model_class: A Model subclass to compose.
#         compose_int_params: Optional callable that transforms internal parameters.
#                         Signature: Any -> Any
#         compose_ext_params: Optional callable that transforms external parameters.
#                         Signature: Any -> Any
#         compose_sim_params: Optional callable that transforms simulation parameters.
#                         Signature: Any -> Any

#     Returns:
#         A new Model class with composed parameter behavior.

#     Example:
#         # Create Identity from Square + sqrt on int_params
#         class Square(Model):
#             def simulate_single(self) -> Dict[str, Any]:
#                 return {"value": self.int_params['x'] ** 2, "shape": self.int_params.shape}

#         Identity = compose_model(
#             Square,
#             compose_int_params=np.sqrt,
#             compose_ext_params=lambda d: {**d, "scale": d["scale"] * 0.5}
#         )
#         identity_instance = Identity(int_params={'x':25.0}, ext_params={"scale": 2.0}, sim_params=None)
#         output = identity_instance.simulate_single()  # Square(sqrt(25.0)) with modified ext_params
#     """

#     class ComposedModel(model_class):
#         def __init__(self, int_params: np.ndarray, ext_params: Any, sim_params: Any):
#             # Apply composition functions
#             transformed_int_params = int_params
#             transformed_ext_params = ext_params
#             transformed_sim_params = sim_params

#             if compose_int_params:
#                 transformed_int_params = compose_int_params(int_params, ext_params, sim_params)
#             if compose_ext_params:
#                 transformed_ext_params = compose_ext_params(int_params, ext_params, sim_params)
#             if compose_sim_params:
#                 transformed_sim_params = compose_sim_params(int_params, ext_params, sim_params)

#             # Initialize the base model with transformed parameters
#             super().__init__(transformed_int_params, transformed_ext_params, transformed_sim_params)

# --- Core model base ---
@dataclass
class Model():
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

    # ========== Simulation ==========
    """Core forward simulation interface."""

    @abstractmethod
    def simulate_single(self) -> Dict[str, Any]:
        """
        Run forward simulation for a single instance. Must be overridden.
        Subclasses should populate self.sim_output with the result before returning.
        Returns: {"value": np.ndarray, "shape": tuple}
        """
        raise NotImplementedError

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
        Inherits @dataclass behavior from parent.

        The composed model applies composition functions to the internal, external, and
        simulation parameters before passing them to the base model's simulate_single() method.

        Args:
            model_class: A Model subclass to compose.
            compose_int_params: Optional callable that transforms internal parameters.
                            Signature: Any -> Any
            compose_ext_params: Optional callable that transforms external parameters.
                            Signature: Any -> Any
            compose_sim_params: Optional callable that transforms simulation parameters.
                            Signature: Any -> Any

        Returns:
            A new Model class with composed parameter behavior.
        """

        class ComposedModel(cls):

            def __post_init__(self):
                # Call parent's __post_init__ if it exists (for nested compositions)
                if hasattr(super(), '__post_init__'):
                    super().__post_init__()
                
                # Store originals AFTER parent transformation
                orig_int = self.int_params
                orig_ext = self.ext_params
                orig_sim = self.sim_params

                if compose_int_params:
                    self.int_params = compose_int_params(orig_int, orig_ext, orig_sim)
                if compose_ext_params:
                    self.ext_params = compose_ext_params(orig_int, orig_ext, orig_sim)
                if compose_sim_params:
                    self.sim_params = compose_sim_params(orig_int, orig_ext, orig_sim)

        ComposedModel = dataclass(ComposedModel) # Critical to regenerate __init__ with __post_init__
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
            'int_params': self.int_params,  # Handles numpy arrays and tuples via encoder
            'ext_params': self.ext_params,
            'sim_params': self.sim_params,
            'sim_output': {
                'value': self.sim_output['value'],
                'shape': self.sim_output['shape']  # Will be encoded as tuple
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
        """Pickle entire Model instance."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f, protocol = -1)

    @classmethod
    def unpickle(cls, filepath) -> "Model":
        """Unpickle and restore a Model instance."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)       
            
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