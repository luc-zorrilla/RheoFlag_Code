from joblib import Parallel, delayed
from typing import Any, Callable, Iterable, List, Sequence, Type, Optional, Tuple, Dict, Literal
import numpy as np
import logging
from itertools import zip_longest, product
import json
import dill as pickle # enhanced pickle library that handles function pickling as well
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
class Model:
    """
    Base model class.

    Subclasses SHOULD:
    - implement simulate_single(self) -> Dict[str, Any] (with keys "value" and "shape")
    - optionally implement simulate_batch(cls, ...) for performance.

    Constructor semantics:
    - int_params: numpy array or scalar representing internal parameters
    - ext_params: external parameters for this instance (may be None)
    - sim_params: simulation parameters / options (may be None)
    - sim_output: initialized as None; populated after simulate_single() or simulate_batch() is called
    """

    def __init__(self, int_params: Any, ext_params: Any, sim_params: Any):
        self.int_params = int_params
        self.ext_params = ext_params
        self.sim_params = sim_params
        self.sim_output: Optional[Dict[str, Any]] = None  # {"value": np.ndarray, "shape": tuple}

    def simulate_single(self) -> Dict[str, Any]:
        """
        Run forward simulation for a single instance. Must be overridden.
        Subclasses should populate self.sim_output with the result before returning.
        Returns: {"value": np.ndarray, "shape": tuple}
        """
        raise NotImplementedError

    @classmethod
    def simulate_batch(
        cls,
        int_params_batch: Sequence[Any],
        ext_params_batch: Sequence[Any],
        sim_params_batch: Sequence[Any],
    ) -> List[Dict[str, Any]]:
        """
        Default batch implementation: map simulate_single over inputs.
        Subclasses are encouraged to override with a vectorized/batched implementation for speed.
        Returns: List[{"value": np.ndarray, "shape": tuple}]
        """
        results = []
        for ip, ep, sp in zip(int_params_batch, ext_params_batch, sim_params_batch):
            instance = cls(ip, ep, sp)
            output = instance.simulate_single()
            results.append(output)
        return results
    
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

    def read_sim_output(self, filepath):
        """Read simulation output and restore numpy arrays and tuples."""
        with open(filepath, 'r') as f:
            data = json.load(f, cls=NumpyTupleDecoder)
        
        self.int_params = data['int_params']
        self.ext_params = data['ext_params']
        self.sim_params = data['sim_params']
        self.sim_output = {
            'value': data['sim_output']['value'],  # Already a numpy array from decoder
            'shape': data['sim_output']['shape']   # Already a tuple from decoder
        }

    def pickle_model(self, filepath):
        """Pickle entire Model instance."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self, f, protocol = -1)

def unpickle_model(filepath):
    """Unpickle and restore a Model instance."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)    

# ----------------------
# Example model: Square (x -> x**2)
# ----------------------
class Square(Model):
    """
    Simple model where internal parameter x maps to x**2.
    - int_params: dict
    - ext_params, sim_params: unused here but accepted for generality
    """

    def simulate_single(self) -> Dict[str, Any]:
        x = np.asarray(self.int_params['x'], dtype=float)
        output = x ** 2
        self.sim_output = {"value": output, "shape": output.shape}
        return self.sim_output

    @classmethod
    def simulate_batch(
        cls,
        int_params_batch: Sequence[Any],
        ext_params_batch: Sequence[Any],
        sim_params_batch: Sequence[Any],
    ) -> List[Dict[str, Any]]:
        """
        Vectorized batch implementation.
        Works if int_params_batch are broadcastable to an array shape (n, k).
        Returns: List[{"value": np.ndarray, "shape": tuple}]
        """
        flat = np.asarray([int_params['x'] for int_params in int_params_batch], dtype=float)
        results = []
        for arr in flat:
            output = arr ** 2
            results.append({"value": output, "shape": output.shape})
        return results

def compose_model(
    model_class: Type[Model],
    compose_int_params: Optional[Callable[[Any], Any]] = None,
    compose_ext_params: Optional[Callable[[Any], Any]] = None,
    compose_sim_params: Optional[Callable[[Any], Any]] = None
) -> Type[Model]:
    """
    Create a new Model class by composing input parameters with functions.

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

    Example:
        # Create Identity from Square + sqrt on int_params
        class Square(Model):
            def simulate_single(self) -> Dict[str, Any]:
                return {"value": self.int_params['x'] ** 2, "shape": self.int_params.shape}

        Identity = compose_model(
            Square,
            compose_int_params=np.sqrt,
            compose_ext_params=lambda d: {**d, "scale": d["scale"] * 0.5}
        )
        identity_instance = Identity(int_params={'x':25.0}, ext_params={"scale": 2.0}, sim_params=None)
        output = identity_instance.simulate_single()  # Square(sqrt(25.0)) with modified ext_params
    """

    class ComposedModel(model_class):
        def __init__(self, int_params: np.ndarray, ext_params: Any, sim_params: Any):
            # Apply composition functions
            transformed_int_params = int_params
            transformed_ext_params = ext_params
            transformed_sim_params = sim_params

            if compose_int_params:
                transformed_int_params = compose_int_params(int_params, ext_params, sim_params)
            if compose_ext_params:
                transformed_ext_params = compose_ext_params(int_params, ext_params, sim_params)
            if compose_sim_params:
                transformed_sim_params = compose_sim_params(int_params, ext_params, sim_params)

            # Initialize the base model with transformed parameters
            super().__init__(transformed_int_params, transformed_ext_params, transformed_sim_params)

    @classmethod
    def simulate_batch( # TODO: parallelize this method
        cls,
        int_params_batch: Sequence[Any],
        ext_params_batch: Sequence[Any],
        sim_params_batch: Sequence[Any],
    ) -> List[Dict[str, Any]]:
        """
        Batch simulation: apply composition functions to all parameters,
        then run base model batch.
        """
        transformed_int_params_batch = []
        transformed_ext_params_batch = []
        transformed_sim_params_batch = []

        # Use zip_longest to handle batches of different sizes
        for int_params, ext_params, sim_params in zip_longest(int_params_batch, ext_params_batch, sim_params_batch, fillvalue=None):
            if int_params is None or ext_params is None or sim_params is None:
                continue  # Skip this iteration if any of the parameters are None

            if compose_int_params:
                transformed_int_params = compose_int_params(int_params, ext_params, sim_params)
            else:
                transformed_int_params = int_params

            if compose_ext_params:
                transformed_ext_params = compose_ext_params(int_params, ext_params, sim_params)
            else:
                transformed_ext_params = ext_params

            if compose_sim_params:
                transformed_sim_params = compose_sim_params(int_params, ext_params, sim_params)
            else:
                transformed_sim_params = sim_params

            transformed_int_params_batch.append(transformed_int_params)
            transformed_ext_params_batch.append(transformed_ext_params)
            transformed_sim_params_batch.append(transformed_sim_params)

        return super().simulate_batch(
            transformed_int_params_batch,
            transformed_ext_params_batch,
            transformed_sim_params_batch
        )


    return ComposedModel

# ----------------------
# Parallel runner
# ----------------------
def _batch_worker( # TODO: Is it necessary? Also, this could be parallelized too.
    model_cls: Type[Model],
    int_params_batch: Sequence[Any],
    ext_params_batch: Sequence[Any],
    sim_params_batch: Sequence[Any],
) -> List[Dict[str, Any]]:
    """
    Top-level worker executed in a separate process.
    - Constructs model instances (or calls model_cls.simulate_batch if implemented)
    - Returns list of output dictionaries in the same order as the inputs.
    """
    try:
        outputs = model_cls.simulate_batch(int_params_batch, ext_params_batch, sim_params_batch)
    except Exception as e:
        logger.error(f"Batch simulation failed: {e}")
        # Fallback to item-wise instantiation
        outputs = []
        for ip, ep, sp in zip(int_params_batch, ext_params_batch, sim_params_batch):
            m = model_cls(ip, ep, sp)
            outputs.append(m.simulate_single())
    return outputs

def parallel_simulate_batch( # TODO: Is it necessary?
    model_cls: Type[Model],
    params_list: Iterable[Tuple[Any, Any, Any]],
    batch_size: int = 32,
    n_jobs: int = 4,
) -> List[Dict[str, Any]]:
    """joblib handles pickling of nested classes better."""
    int_list, ext_list, sim_list = zip(*params_list)
    n = len(int_list)

    results = []
    for i in range(0, n, batch_size):
        chunk_int = list(int_list[i : i + batch_size])
        chunk_ext = list(ext_list[i : i + batch_size])
        chunk_sim = list(sim_list[i : i + batch_size])
        
        batch_results = Parallel(n_jobs=n_jobs)(
            delayed(_batch_worker)(model_cls, [ip], [ep], [sp]) 
            for ip, ep, sp in zip(chunk_int, chunk_ext, chunk_sim)
        )
        for batch in batch_results:
            results.extend(batch)
    
    return results