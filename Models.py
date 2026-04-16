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

def numpy_serializer(obj):
    """Function to serialize NumPy types for json.dumps default."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # Add checks for other numpy types if needed (np.integer, np.floating)
    # raise TypeError # Optionally raise error for unhandled types
    # For simplicity here, we only handle ndarray
    return obj # Or let default handle/error out

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
        """Write simulation output and metadata as human-readable JSON."""
        if self.sim_output is None:
            raise ValueError("No simulation output. Run simulate_single() first.")
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'int_params': self.int_params,
            'ext_params': self.ext_params,
            'sim_params': self.sim_params,
            'sim_output': {
                'value': self.sim_output['value'].tolist(),
                'shape': self.sim_output['shape']
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=numpy_serializer)

    def read_sim_output(self, filepath):
        """Read simulation output and metadata from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.int_params = data['int_params']
        self.ext_params = data['ext_params']
        self.sim_params = data['sim_params']
        self.sim_output = {
            'value': np.array(data['sim_output']['value']),
            'shape': tuple(data['sim_output']['shape'])
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


def Square_create_params_list(int_keys: List[str], ext_keys: List[str], sim_keys: List[str], params_list_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """ Generate all combinations of the parameter lists. 
    Useful for parallel computations of varying model parameters. 
    
    Note: Could be generalised -- in this case -- to loop over all keys."""
    
    params_list = []
    for x in params_list_dict["x_list"]: 

        int_params = {key:eval(key) for key in int_keys}
        ext_params = {key:eval(key) for key in ext_keys}
        sim_params = {key:eval(key) for key in sim_keys}
        params_list.append((int_params, ext_params, sim_params))

    return params_list


# --- Functions on models ---
def reduce_model(
    original_model: Type[Model],
    int_param_reduction: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ext_param_reduction: Optional[Callable[[Any], Any]] = None,
    sim_param_reduction: Optional[Callable[[Any], Any]] = None
) -> Tuple[Type[Model], Dict[str, Any]]:
    """
    Reduce the dimensionality of a model by applying reduction transformations to its parameters.
    """
    metadata = {
        "original_model": original_model,
        "int_param_reduction": int_param_reduction,
        "ext_param_reduction": ext_param_reduction,
        "sim_param_reduction": sim_param_reduction,
    }

    class ReducedModel(Model):
        def __init__(self, int_params: np.ndarray, ext_params: Any, sim_params: Any):
            if int_param_reduction:
                int_params = int_param_reduction(int_params)
            if ext_param_reduction:
                ext_params = ext_param_reduction(ext_params)
            if sim_param_reduction:
                sim_params = sim_param_reduction(sim_params)
            super().__init__(int_params, ext_params, sim_params)

        def simulate_single(self) -> Dict[str, Any]:
            output = original_model(self.int_params, self.ext_params, self.sim_params).simulate_single()
            self.sim_output = output
            return output

        @classmethod
        def simulate_batch(
            cls,
            int_params_batch: Sequence[Any],
            ext_params_batch: Sequence[Any],
            sim_params_batch: Sequence[Any],
        ) -> List[Dict[str, Any]]:
            if int_param_reduction:
                int_params_batch = [int_param_reduction(ip) for ip in int_params_batch]
            if ext_param_reduction:
                ext_params_batch = [ext_param_reduction(ep) for ep in ext_params_batch]
            if sim_param_reduction:
                sim_params_batch = [sim_param_reduction(sp) for sp in sim_params_batch]
            return original_model.simulate_batch(int_params_batch, ext_params_batch, sim_params_batch)

    return ReducedModel, metadata

def reconstruct_model(
    reduced_model: Type[Model],
    metadata: Dict[str, Any],
    int_param_reconstruction: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ext_param_reconstruction: Optional[Callable[[Any], Any]] = None,
    sim_param_reconstruction: Optional[Callable[[Any], Any]] = None
) -> Type[Model]:
    """
    Reconstruct the original model from a reduced model by applying inverse transformations.
    """
    original_model = metadata["original_model"]

    class ReconstructedModel(Model):
        def __init__(self, int_params: np.ndarray, ext_params: Any, sim_params: Any):
            if int_param_reconstruction:
                int_params = int_param_reconstruction(int_params)
            if ext_param_reconstruction:
                ext_params = ext_param_reconstruction(ext_params)
            if sim_param_reconstruction:
                sim_params = sim_param_reconstruction(sim_params)
            super().__init__(int_params, ext_params, sim_params)

        def simulate_single(self) -> Dict[str, Any]:
            output = original_model(self.int_params, self.ext_params, self.sim_params).simulate_single()
            self.sim_output = output
            return output

        @classmethod
        def simulate_batch(
            cls,
            int_params_batch: Sequence[Any],
            ext_params_batch: Sequence[Any],
            sim_params_batch: Sequence[Any],
        ) -> List[Dict[str, Any]]:
            if int_param_reconstruction:
                int_params_batch = [int_param_reconstruction(ip) for ip in int_params_batch]
            if ext_param_reconstruction:
                ext_params_batch = [ext_param_reconstruction(ep) for ep in ext_params_batch]
            if sim_param_reconstruction:
                sim_params_batch = [sim_param_reconstruction(sp) for sp in sim_params_batch]
            return original_model.simulate_batch(int_params_batch, ext_params_batch, sim_params_batch)

    return ReconstructedModel

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
    def simulate_batch(
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
def _batch_worker(
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

def parallel_simulate_batch(
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