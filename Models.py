from concurrent.futures import ProcessPoolExecutor
from typing import Any, Callable, Iterable, List, Sequence, Type, Optional, Tuple, Dict, Literal
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

    def __init__(self, int_params: np.ndarray, ext_params: Any, sim_params: Any):
        self.int_params = np.asarray(int_params, dtype=float)
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
        """Write simulation output and Model metadata."""
        # TO BE COMPLETED
        return

    def pickle_model(self, filepath):
        """Pickle Model instance."""
        # TO BE COMPLETED
        return

# ----------------------
# Example model: Square (x -> x**2)
# ----------------------
class Square(Model):
    """
    Simple model where internal parameter x maps to x**2.
    - int_params: scalar or 1D array (we treat it elementwise)
    - ext_params, sim_params: unused here but accepted for generality
    """

    def simulate_single(self) -> Dict[str, Any]:
        x = np.asarray(self.int_params, dtype=float)
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
        flat = np.asarray(int_params_batch, dtype=float)
        results = []
        for arr in flat:
            output = arr ** 2
            results.append({"value": output, "shape": output.shape})
        return results

# ---------------------------------
# Include ViscoElasticFilament here (update similarly)
# ---------------------------------

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

# --- Model Merging ---
def merge_models(
    model1: Model,
    model2: Model,
    *,
    int_strategy: Literal["concat", "model1", "model2"] = "concat",
    ext_strategy: Literal["model1", "model2", "merge"] = "model1",
    sim_strategy: Literal["model1", "model2", "merge"] = "model1",
) -> Type[Model]:
    """
    Merge two instances of the same Model subclass into a single new Model subclass.
    """
    if type(model1) != type(model2):
        raise ValueError(
            f"Both models must be instances of the same subclass. "
            f"Got {type(model1).__name__} and {type(model2).__name__}."
        )

    original_model_class = type(model1)

    # Compute merged int_params based on strategy
    if int_strategy == "concat":
        merged_int_params = np.vstack([model1.int_params, model2.int_params])
    elif int_strategy == "model1":
        merged_int_params = model1.int_params.copy()
    elif int_strategy == "model2":
        merged_int_params = model2.int_params.copy()
    else:
        raise ValueError(
            f"Unknown int_strategy: {int_strategy}. "
            f"Choose from: 'concat', 'model1', 'model2'."
        )

    # Compute merged ext_params based on strategy
    if ext_strategy == "model1":
        merged_ext_params = model1.ext_params
    elif ext_strategy == "model2":
        merged_ext_params = model2.ext_params
    elif ext_strategy == "merge":
        merged_ext_params = model1.ext_params if model1.ext_params is not None else model2.ext_params
    else:
        raise ValueError(
            f"Unknown ext_strategy: {ext_strategy}. "
            f"Choose from: 'model1', 'model2', 'merge'."
        )

    # Compute merged sim_params based on strategy
    if sim_strategy == "model1":
        merged_sim_params = model1.sim_params
    elif sim_strategy == "model2":
        merged_sim_params = model2.sim_params
    elif sim_strategy == "merge":
        merged_sim_params = model1.sim_params if model1.sim_params is not None else model2.sim_params
    else:
        raise ValueError(
            f"Unknown sim_strategy: {sim_strategy}. "
            f"Choose from: 'model1', 'model2', 'merge'."
        )

    class MergedModel(Model):
        """
        A merged model combining two parent Model instances.
        """

        def __init__(
            self,
            int_params: Optional[np.ndarray] = None,
            ext_params: Any = None,
            sim_params: Any = None,
        ):
            if int_params is None:
                int_params = merged_int_params
            if ext_params is None:
                ext_params = merged_ext_params
            if sim_params is None:
                sim_params = merged_sim_params
            super().__init__(int_params, ext_params, sim_params)
            self.parent_model1 = model1
            self.parent_model2 = model2
            self.int_strategy = int_strategy
            self.ext_strategy = ext_strategy
            self.sim_strategy = sim_strategy

        def simulate_single(self) -> Dict[str, Any]:
            instance = original_model_class(self.int_params, self.ext_params, self.sim_params)
            output = instance.simulate_single()
            self.sim_output = output
            return output

        @classmethod
        def simulate_batch(
            cls,
            int_params_batch: Sequence[Any],
            ext_params_batch: Sequence[Any],
            sim_params_batch: Sequence[Any],
        ) -> List[Dict[str, Any]]:
            return original_model_class.simulate_batch(
                int_params_batch, ext_params_batch, sim_params_batch
            )

        def get_parent_outputs(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
            output1 = self.parent_model1.simulate_single()
            output2 = self.parent_model2.simulate_single()
            return output1, output2

        def get_merged_info(self) -> Dict[str, Any]:
            return {
                "int_strategy": self.int_strategy,
                "ext_strategy": self.ext_strategy,
                "sim_strategy": self.sim_strategy,
                "parent_model1": self.parent_model1,
                "parent_model2": self.parent_model2,
                "parent1_int_params": self.parent_model1.int_params,
                "parent2_int_params": self.parent_model2.int_params,
                "parent1_ext_params": self.parent_model1.ext_params,
                "parent2_ext_params": self.parent_model2.ext_params,
                "parent1_sim_params": self.parent_model1.sim_params,
                "parent2_sim_params": self.parent_model2.sim_params,
                "merged_int_params": self.int_params,
                "merged_ext_params": self.ext_params,
                "merged_sim_params": self.sim_params,
            }

    return MergedModel

# --- Multiple Model Merging ---
def merge_multiple_models(
    *models: Model,
    int_strategy: Literal["average", "concat", "first", "last"] = "average",
    ext_strategy: Literal["first", "last", "merge"] = "first",
    sim_strategy: Literal["first", "last", "merge"] = "first",
) -> Type[Model]:
    """
    Merge an arbitrary number of Model instances into a single new Model subclass.
    """
    if len(models) < 2:
        raise ValueError("At least two models must be provided for merging.")
    if not all(type(m) == type(models[0]) for m in models):
        raise ValueError("All models must be instances of the same subclass.")

    original_model_class = type(models[0])

    # Compute merged int_params based on strategy
    if int_strategy == "concat":
        merged_int_params = np.vstack([m.int_params for m in models])
    elif int_strategy == "first":
        merged_int_params = models[0].int_params.copy()
    elif int_strategy == "last":
        merged_int_params = models[-1].int_params.copy()
    elif int_strategy == "average":
        merged_int_params = np.mean([m.int_params for m in models], axis=0)
    else:
        raise ValueError(
            f"Unknown int_strategy: {int_strategy}. "
            f"Choose from: 'concat', 'first', 'last', 'average'."
        )

    # Compute merged ext_params based on strategy
    if ext_strategy == "first":
        merged_ext_params = models[0].ext_params
    elif ext_strategy == "last":
        merged_ext_params = models[-1].ext_params
    elif ext_strategy == "merge":
        merged_ext_params = next((m.ext_params for m in models if m.ext_params is not None), None)
    else:
        raise ValueError(
            f"Unknown ext_strategy: {ext_strategy}. "
            f"Choose from: 'first', 'last', 'merge'."
        )

    # Compute merged sim_params based on strategy
    if sim_strategy == "first":
        merged_sim_params = models[0].sim_params
    elif sim_strategy == "last":
        merged_sim_params = models[-1].sim_params
    elif sim_strategy == "merge":
        merged_sim_params = next((m.sim_params for m in models if m.sim_params is not None), None)
    else:
        raise ValueError(
            f"Unknown sim_strategy: {sim_strategy}. "
            f"Choose from: 'first', 'last', 'merge'."
        )

    class MergedModel(Model):
        """
        A merged model combining multiple parent Model instances.
        """

        def __init__(
            self,
            int_params: Optional[np.ndarray] = None,
            ext_params: Any = None,
            sim_params: Any = None,
        ):
            if int_params is None:
                int_params = merged_int_params
            if ext_params is None:
                ext_params = merged_ext_params
            if sim_params is None:
                sim_params = merged_sim_params
            super().__init__(int_params, ext_params, sim_params)
            self.parent_models = models
            self.int_strategy = int_strategy
            self.ext_strategy = ext_strategy
            self.sim_strategy = sim_strategy

        def simulate_single(self) -> Dict[str, Any]:
            instance = original_model_class(self.int_params, self.ext_params, self.sim_params)
            output = instance.simulate_single()
            self.sim_output = output
            return output

        @classmethod
        def simulate_batch(
            cls,
            int_params_batch: Sequence[Any],
            ext_params_batch: Sequence[Any],
            sim_params_batch: Sequence[Any],
        ) -> List[Dict[str, Any]]:
            return original_model_class.simulate_batch(
                int_params_batch, ext_params_batch, sim_params_batch
            )

        def get_parent_outputs(self) -> List[Dict[str, Any]]:
            return [m.simulate_single() for m in self.parent_models]

        def get_merged_info(self) -> Dict[str, Any]:
            return {
                "int_strategy": self.int_strategy,
                "ext_strategy": self.ext_strategy,
                "sim_strategy": self.sim_strategy,
                "parent_models": self.parent_models,
                "parent_int_params": [m.int_params for m in self.parent_models],
                "parent_ext_params": [m.ext_params for m in self.parent_models],
                "parent_sim_params": [m.sim_params for m in self.parent_models],
                "merged_int_params": self.int_params,
                "merged_ext_params": self.ext_params,
                "merged_sim_params": self.sim_params,
            }

    return MergedModel

def compose_model(
    model_class: Type[Model],
    compose_int_params: Optional[Callable[[np.ndarray], np.ndarray]] = None,
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
                           Signature: np.ndarray -> np.ndarray
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
                return {"value": self.int_params ** 2, "shape": self.int_params.shape}

        Identity = compose_model(
            Square,
            compose_int_params=np.sqrt,
            compose_ext_params=lambda d: {**d, "scale": d["scale"] * 0.5}
        )
        identity_instance = Identity(int_params=25.0, ext_params={"scale": 2.0}, sim_params=None)
        output = identity_instance.simulate_single()  # Square(sqrt(25.0)) with modified ext_params
    """

    class ComposedModel(model_class):
        def __init__(self, int_params: np.ndarray, ext_params: Any, sim_params: Any):
            # Apply composition functions
            transformed_int_params = int_params
            transformed_ext_params = ext_params
            transformed_sim_params = sim_params

            if compose_int_params:
                transformed_int_params = compose_int_params(int_params)
            if compose_ext_params:
                transformed_ext_params = compose_ext_params(ext_params)
            if compose_sim_params:
                transformed_sim_params = compose_sim_params(sim_params)

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
            transformed_int_params_batch = int_params_batch
            transformed_ext_params_batch = ext_params_batch
            transformed_sim_params_batch = sim_params_batch

            if compose_int_params:
                transformed_int_params_batch = [
                    compose_int_params(ip) for ip in int_params_batch
                ]
            if compose_ext_params:
                transformed_ext_params_batch = [
                    compose_ext_params(ep) for ep in ext_params_batch
                ]
            if compose_sim_params:
                transformed_sim_params_batch = [
                    compose_sim_params(sp) for sp in sim_params_batch
                ]

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
    int_params_list: Iterable[Any],
    ext_params_list: Iterable[Any],
    sim_params_list: Iterable[Any],
    batch_size: int = 32,
    max_workers: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Run many simulations in parallel using ProcessPoolExecutor with chunking.
    Returns: List[{"value": np.ndarray, "shape": tuple}]
    """
    int_list = list(int_params_list)
    ext_list = list(ext_params_list)
    sim_list = list(sim_params_list)
    n = len(int_list)
    assert n == len(ext_list) == len(sim_list), "Parameter lists must have same length"

    # Create chunks
    chunks: List[Tuple[List[Any], List[Any], List[Any]]] = []
    for i in range(0, n, batch_size):
        chunks.append((int_list[i : i + batch_size], ext_list[i : i + batch_size], sim_list[i : i + batch_size]))

    results: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_batch_worker, model_cls, c[0], c[1], c[2]) for c in chunks]
        for fut in futures:
            results.extend(fut.result())
    return results