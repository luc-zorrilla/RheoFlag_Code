from pathlib import Path
import dill as pickle
import json
from datetime import datetime
import shutil
from typing import Any, Optional, Dict
from joblib import Parallel, delayed
from dataclasses import dataclass, field
from Models import Model, ModelList, SimpleModel
from Inferences import Inference, InferencePipeline, PipelinePass
from scipy.optimize import minimize, OptimizeResult
import numpy as np

# =========== #
# Checkpoints #
# =========== #

@dataclass
class CheckpointEntry:
    """Metadata for a single completed task."""
    task_key: str
    completed: bool
    timestamp: str
    filepath: str  # relative path to saved artifact


@dataclass
class WorkflowCheckpoint:
    """High-level checkpoint tracking the entire workflow."""
    simulation_entries: Dict[str, CheckpointEntry] = field(default_factory=dict)
    inference_entries: Dict[str, CheckpointEntry] = field(default_factory=dict)
    stage: str = "simulation"
    timestamp: str = ""

class CheckpointManager:
    """Unified checkpoint system for simulations and inferences."""
    
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.checkpoint_dir / "checkpoint.json"
        self.artifacts_dir = self.checkpoint_dir / "artifacts"
        self.artifacts_dir.mkdir(exist_ok=True)
    
    # ========================================================================
    # Artifact Storage
    # ========================================================================
    
    def save_artifact(self, artifact_type: str, key: str, artifact: Any) -> str:
        """
        Save an artifact (ModelList, InferenceResult, etc.) and return relative path.
        
        Args:
            artifact_type: "models" or "inference"
            key: Unique identifier (e.g., "int_0_pair_2" or "inference_task_5")
            artifact: Object to pickle
        
        Returns:
            Relative filepath
        """
        type_dir = self.artifacts_dir / artifact_type
        type_dir.mkdir(exist_ok=True)
        filename = f"{key}.pkl"
        filepath = type_dir / filename
        with open(filepath, "wb") as f:
            pickle.dump(artifact, f)
        return str(filepath.relative_to(self.checkpoint_dir))
    
    def load_artifact(self, artifact_type: str, key: str) -> Optional[Any]:
        """Load an artifact by type and key."""
        type_dir = self.artifacts_dir / artifact_type
        filepath = type_dir / f"{key}.pkl"
        if filepath.exists():
            with open(filepath, "rb") as f:
                return pickle.load(f)
        return None
    
    def artifact_exists(self, artifact_type: str, key: str) -> bool:
        """Check if artifact exists without loading."""
        type_dir = self.artifacts_dir / artifact_type
        filepath = type_dir / f"{key}.pkl"
        return filepath.exists()
    
    # ========================================================================
    # Checkpoint Persistence
    # ========================================================================
    
    def save_checkpoint(self, checkpoint: WorkflowCheckpoint):
        """Save checkpoint metadata to JSON."""
        data = {
            "stage": checkpoint.stage,
            "timestamp": checkpoint.timestamp,
            "simulation_entries": {
                k: {
                    "task_key": v.task_key,
                    "completed": v.completed,
                    "timestamp": v.timestamp,
                    "filepath": v.filepath,
                }
                for k, v in checkpoint.simulation_entries.items()
            },
            "inference_entries": {
                k: {
                    "task_key": v.task_key,
                    "completed": v.completed,
                    "timestamp": v.timestamp,
                    "filepath": v.filepath,
                }
                for k, v in checkpoint.inference_entries.items()
            },
        }
        with open(self.metadata_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def load_checkpoint(self) -> Optional[WorkflowCheckpoint]:
        """Load checkpoint from file."""
        if not self.metadata_file.exists():
            return None
        
        with open(self.metadata_file) as f:
            data = json.load(f)
        
        simulation_entries = {
            k: CheckpointEntry(**v)
            for k, v in data.get("simulation_entries", {}).items()
        }
        
        inference_entries = {
            k: CheckpointEntry(**v)
            for k, v in data.get("inference_entries", {}).items()
        }
        
        return WorkflowCheckpoint(
            simulation_entries=simulation_entries,
            inference_entries=inference_entries,
            stage=data.get("stage", "simulation"),
            timestamp=data.get("timestamp", ""),
        )
    
    # ========================================================================
    # Query Methods
    # ========================================================================
    
    def is_simulation_done(self, int_idx: int, pair_idx: int) -> bool:
        """Check if simulation task is completed."""
        checkpoint = self.load_checkpoint()
        if not checkpoint:
            return False
        task_key = f"int_{int_idx}_pair_{pair_idx}"
        return checkpoint.simulation_entries.get(task_key, CheckpointEntry(task_key, False, "", "")).completed
    
    def is_inference_done(self, task_key: str) -> bool:
        """Check if inference task is completed."""
        checkpoint = self.load_checkpoint()
        if not checkpoint:
            return False
        return checkpoint.inference_entries.get(task_key, CheckpointEntry(task_key, False, "", "")).completed

# =========== #
# Simulations #
# =========== #

@dataclass
class SimulationTask:
    """A single simulation: one int_params with one (ext_params, sim_params) pair."""
    int_idx: int
    pair_idx: int  # Index within ext/sim pairs
    int_params: dict
    ext_params: dict
    sim_params: dict


def run_single_simulation(
    int_idx: int,
    pair_idx: int,
    int_params: dict,
    ext_params_list: list,
    sim_params_list: list,
    model_class: type,
    checkpoint_mgr: CheckpointManager,
) -> ModelList:
    """
    Simulate one internal parameter with all zipped (ext, sim) pairs.
    
    Args:
        int_idx: Internal parameter index
        pair_idx: Index in the ext/sim pair list (unused here, kept for consistency)
        int_params: Single internal parameter dict
        ext_params_list: List of external parameter dicts
        sim_params_list: List of simulation parameter dicts (same length as ext)
        model_class: The Model subclass to instantiate
        checkpoint_mgr: CheckpointManager for checkpointing
    
    Returns:
        ModelList with all (ext, sim) combinations
    """
    task_key = f"int_{int_idx}_pair_all"  # All pairs for this int_idx
    
    # Check if already completed
    if checkpoint_mgr.artifact_exists("models", task_key):
        model_list = checkpoint_mgr.load_artifact("models", task_key)
        if model_list:
            return model_list
    
    # Create ModelList with batch of models (one per ext/sim pair)
    model_list = ModelList.from_params(
        model_class=model_class,
        int_params_batch=[int_params] * len(ext_params_list),  # Repeat for each pair
        ext_params_batch=ext_params_list,
        sim_params_batch=sim_params_list,
    )

    # Run simulation in parallel
    model_list.simulate(n_jobs=-1, parallel=True)
    
    # Save and checkpoint
    artifact_path = checkpoint_mgr.save_artifact("models", task_key, model_list)
    checkpoint = checkpoint_mgr.load_checkpoint() or WorkflowCheckpoint()
    checkpoint.simulation_entries[task_key] = CheckpointEntry(
        task_key=task_key,
        completed=True,
        timestamp=str(datetime.now()),
        filepath=artifact_path,
    )
    checkpoint.stage = "simulation"
    checkpoint.timestamp = str(datetime.now())
    checkpoint_mgr.save_checkpoint(checkpoint)
    
    return model_list

def parallel_simulate_nested(
    int_params_list: list,
    ext_params_list: list,
    sim_params_list: list,
    model_class: type,
    checkpoint_mgr: CheckpointManager,
    n_jobs: int = -1,
) -> Dict[int, ModelList]:
    """
    Outer loop over int_params, inner loop over zipped (ext, sim) pairs.
    
    Each int_params creates one ModelList with multiple models (one per pair).
    
    Args:
        int_params_list: List of internal parameter dicts
        ext_params_list: List of external parameter dicts
        sim_params_list: List of simulation parameter dicts (must equal len(ext_params_list))
        model_class: The Model subclass to instantiate
        checkpoint_mgr: CheckpointManager for resuming
        n_jobs: Parallelization across int_idx (outer loop)
    
    Returns:
        Dict mapping int_idx -> ModelList (with len(ext_params_list) models each)
    """
    assert len(ext_params_list) == len(sim_params_list), \
        "ext_params_list and sim_params_list must have same length"
    
    n_int = len(int_params_list)
    n_pairs = len(ext_params_list)
    
    print(f"Simulation structure:")
    print(f"  Internal params: {n_int}")
    print(f"  Ext/Sim pairs: {n_pairs}")
    print(f"  Total ModelLists: {n_int}")
    print(f"  Total models: {n_int * n_pairs}")
    
    # Check which int_idx tasks need recomputation
    pending_int_idx = [
        i for i in range(n_int)
        if not checkpoint_mgr.artifact_exists("models", f"int_{i}_pair_all")
    ]
    
    if pending_int_idx:
        print(f"Running {len(pending_int_idx)} pending int_idx tasks...")
        Parallel(n_jobs=n_jobs)(
            delayed(run_single_simulation)(
                int_idx,
                0,  # pair_idx unused
                int_params_list[int_idx],
                ext_params_list,
                sim_params_list,
                model_class,
                checkpoint_mgr,
            )
            for int_idx in pending_int_idx
        )
    
    # Load all ModelLists
    model_lists = {}
    for int_idx in range(n_int):
        model_list = checkpoint_mgr.load_artifact("models", f"int_{int_idx}_pair_all")

        if model_list and model_list.results:
            for model, result_dict in zip(model_list.models, model_list.results):
                if model.sim_output is None:
                    model.sim_output = result_dict

        model_lists[int_idx] = model_list
    
    print(f"✓ Simulations complete: {len(model_lists)} ModelLists")
    return model_lists

# ========== #
# Inferences #
# ========== #

@dataclass
class InferenceTask:
    """Single inference job on models with same int_params."""
    task_key: str  # Unique identifier
    int_idx: int  # Which ModelList to use
    pair_indices: Optional[List[int]] = None  # If None, use all models in ModelList
    make_pipeline_fn: Optional[callable] = None  # Factory function for pipeline
    pipeline_kwargs: dict = field(default_factory=dict)
    initial_guesses: list = field(default_factory=list)

def run_single_inference(
    task: InferenceTask,
    model_lists: Dict[int, ModelList],
    checkpoint_mgr: CheckpointManager,
) -> Any:
    """
    Run inference on a ModelList (or subset of models within it).
    
    Args:
        task: InferenceTask defining the inference
        model_lists: Dict of int_idx -> ModelList
        checkpoint_mgr: CheckpointManager for checkpointing
    
    Returns:
        Inference result
    """
    if checkpoint_mgr.is_inference_done(task.task_key):
        return checkpoint_mgr.load_artifact("inference", task.task_key)
    
    # Get the ModelList
    model_list = model_lists[task.int_idx]
    
    # Filter to specific pairs if requested
    if task.pair_indices is not None:
        filtered_model_list = ModelList(
            models=[model_list.models[i] for i in task.pair_indices]
        )
    else:
        filtered_model_list = model_list
    
    # Create pipeline via factory
    pipeline = InferencePipeline.from_factory(
        task.make_pipeline_fn,
        model_list=filtered_model_list,
        **task.pipeline_kwargs,
    )
    
    # Filter initial_guesses per pass based on param_keys_to_infer
    initial_guesses_per_pass = []
    for pass_idx, pipeline_pass in enumerate(pipeline.passes):
        param_keys = pipeline_pass.param_keys_to_infer
        
        # Filter each initial guess to only include keys for this pass
        filtered_guesses = [
            {key: guess[key] for key in param_keys if key in guess}
            for guess in task.initial_guesses
        ]
        initial_guesses_per_pass.append(filtered_guesses)

    # Run pipeline with filtered initial_guesses
    result = pipeline.run(
        initial_guesses_per_pass=initial_guesses_per_pass,
        verbose=True,
    )
    
    # Checkpoint
    artifact_path = checkpoint_mgr.save_artifact("inference", task.task_key, result)
    checkpoint = checkpoint_mgr.load_checkpoint() or WorkflowCheckpoint()
    checkpoint.inference_entries[task.task_key] = CheckpointEntry(
        task_key=task.task_key,
        completed=True,
        timestamp=str(datetime.now()),
        filepath=artifact_path,
    )
    checkpoint.stage = "inference"
    checkpoint.timestamp = str(datetime.now())
    checkpoint_mgr.save_checkpoint(checkpoint)
    
    return result

def parallel_infer_nested(
    inference_tasks: List[InferenceTask],
    model_lists: Dict[int, ModelList],
    checkpoint_mgr: CheckpointManager,
    n_jobs: int = -1,
) -> Dict[str, Any]:
    """
    Run all inference tasks in parallel.
    
    Returns:
        Dict mapping task_key -> inference result
    """
    print(f"Total inference tasks: {len(inference_tasks)}")
    
    pending_tasks = [
        t for t in inference_tasks
        if not checkpoint_mgr.is_inference_done(t.task_key)
    ]
    
    if pending_tasks:
        
        # Debug: Show which tasks are about to run
        for task in pending_tasks:
            print(f"  → {task.task_key}")

        print(f"Running {len(pending_tasks)} pending inferences...")
        Parallel(n_jobs=n_jobs)(
            delayed(run_single_inference)(task, model_lists, checkpoint_mgr)
            for task in pending_tasks
        )
    
    # Load all results
    results = {}
    for task in inference_tasks:
        result = checkpoint_mgr.load_artifact("inference", task.task_key)
        results[task.task_key] = result
    
    print(f"✓ Inferences complete: {len(results)} results")
    return results

# ============ #
# Orchestrator #
# ============ #

class SimulationInferenceWorkflow:
    """Orchestrates nested simulation → inference pipeline with checkpoints."""
    
    def __init__(self, checkpoint_dir: Path = Path("./checkpoints")):
        self.checkpoint_mgr = CheckpointManager(checkpoint_dir)
    
    def run_simulations(
        self,
        int_params_list: list,
        ext_params_list: list,
        sim_params_list: list,
        model_class: type,
        n_jobs: int = -1,
    ) -> Dict[int, ModelList]:
        """
        Run nested simulations: outer loop over int_params, inner loop over ext/sim pairs.
        
        Returns:
            Dict mapping int_idx -> ModelList
        """
        return parallel_simulate_nested(
            int_params_list,
            ext_params_list,
            sim_params_list,
            model_class,
            self.checkpoint_mgr,
            n_jobs=n_jobs,
        )
    
    def run_inferences(
        self,
        inference_tasks: List[InferenceTask],
        model_lists: Dict[int, ModelList],
        n_jobs: int = -1,
    ) -> Dict[str, Any]:
        """Run inference tasks."""
        return parallel_infer_nested(
            inference_tasks,
            model_lists,
            self.checkpoint_mgr,
            n_jobs=n_jobs,
        )
    
    def get_model_list(self, int_idx: int) -> Optional[ModelList]:
        """Retrieve a ModelList by int_idx."""
        return self.checkpoint_mgr.load_artifact("models", f"int_{int_idx}_pair_all")
    
    def get_inference_result(self, task_key: str) -> Optional[Any]:
        """Retrieve an inference result by task_key."""
        return self.checkpoint_mgr.load_artifact("inference", task_key)
    
    def get_checkpoint_status(self) -> Optional[WorkflowCheckpoint]:
        """Get current checkpoint status."""
        return self.checkpoint_mgr.load_checkpoint()
    
    def clear_checkpoints(self):
        """Clear all checkpoints (start fresh)."""
        import shutil
        if self.checkpoint_mgr.checkpoint_dir.exists():
            shutil.rmtree(self.checkpoint_mgr.checkpoint_dir)
        self.checkpoint_mgr.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_mgr.artifacts_dir.mkdir(exist_ok=True)


# ========== Test: Workflow ==========

def test_workflow():
    """Test models in the SimulationInferenceWorkflow. """

    print(f"Test workflow")
    # Clean up any previous checkpoints
    checkpoint_dir = Path("./test_checkpoints")
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)

    # Create workflow
    workflow = SimulationInferenceWorkflow(checkpoint_dir=checkpoint_dir)

    # Define simple parameters
    int_params_list = [2.0, 3.0, 4.0]  # 3 internal parameter values
    ext_params_list = [10.0, 20.0]      # 2 external parameter values
    sim_params_list = [None, None]      # 2 simulation parameters (not used in SimpleModel)

    print("="*70)
    print("SIMPLE MODEL SIMULATION TEST")
    print("="*70)

    print(f"\nInput structure:")
    print(f"  Internal params: {len(int_params_list)} values = {int_params_list}")
    print(f"  Ext/Sim pairs: {len(ext_params_list)} pairs")
    print(f"  Expected: {len(int_params_list)} ModelLists × {len(ext_params_list)} models = {len(int_params_list) * len(ext_params_list)} total models")

    # Run simulations
    print(f"int_params_list, ext_params_list, sim_params_list = {int_params_list}, {ext_params_list}, {sim_params_list}")
    model_lists = workflow.run_simulations(
        int_params_list=int_params_list,
        ext_params_list=ext_params_list,
        sim_params_list=sim_params_list,
        model_class=SimpleModel,
        n_jobs=1,  # Use serial execution for easier debugging
    )

    print(f"\nResults:")
    print(f"  ModelLists created: {len(model_lists)}")
    for int_idx, model_list in model_lists.items():
        print(f"\n  int_idx={int_idx} (int_params={int_params_list[int_idx]}):")
        for model_idx, model in enumerate(model_list.models):
            print(f"int_params, ext_params, sim_params = {model.int_params}, {model.ext_params}, {model.sim_params}")
            ext_val = ext_params_list[model_idx]
            result = model.sim_output['value'][0]
            expected = int_params_list[int_idx] * ext_val
            print(f"    model[{model_idx}] (ext_params={ext_val}): result={result}, expected={expected}, match={result == expected}")

    print("\n✓ Test complete!")

def test_composition_with_workflow():
    """Test composed models in the SimulationInferenceWorkflow."""
    
    print("\n" + "="*70)
    print("COMPOSED MODEL SIMULATION TEST")
    print("="*70)
    
    # Clean up any previous checkpoints
    checkpoint_dir = Path("./test_checkpoints_composed")
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    
    # --- Create a composed model: scale internal params by 2 ---
    def compose_int_params(int_p, ext_p, sim_p):
        """Scale internal parameters by 2."""
        return int_p * 2
    
    ComposedSimpleModel = SimpleModel.compose(
        compose_int_params=compose_int_params
    )
    
    # Create workflow
    workflow = SimulationInferenceWorkflow(checkpoint_dir=checkpoint_dir)
    
    # Define parameters
    int_params_list = [2.0, 3.0, 4.0]  # Will be scaled by 2 during composition
    ext_params_list = [10.0, 20.0]
    sim_params_list = [None, None]
    
    print(f"\nInput structure:")
    print(f"  Internal params (before composition): {int_params_list}")
    print(f"  Internal params (after composition): {[p * 2 for p in int_params_list]}")
    print(f"  Ext/Sim pairs: {len(ext_params_list)}")
    
    # Run simulations with composed model
    model_lists = workflow.run_simulations(
        int_params_list=int_params_list,
        ext_params_list=ext_params_list,
        sim_params_list=sim_params_list,
        model_class=ComposedSimpleModel,
        n_jobs=1,
    )
    
    # Restore sim_output from results
    for int_idx in range(len(model_lists)):
        model_list = model_lists[int_idx]
        if model_list and model_list.results:
            for model, result_dict in zip(model_list.models, model_list.results):
                print(f"int_params, ext_params, sim_params = {model.int_params}, {model.ext_params}, {model.sim_params}")
                if model.sim_output is None:
                    model.sim_output = result_dict
    
    print(f"\nResults:")
    print(f"  ModelLists created: {len(model_lists)}")
    
    all_correct = True
    for int_idx, model_list in model_lists.items():
        print(f"\n  int_idx={int_idx} (original int_params={int_params_list[int_idx]}):")
        for model_idx, model in enumerate(model_list.models):
            ext_val = ext_params_list[model_idx]
            composed_int_val = int_params_list[int_idx] * 2  # After composition
            result = model.sim_output['value'][0]
            expected = composed_int_val * ext_val
            match = result == expected
            all_correct = all_correct and match
            print(f"    model[{model_idx}] (ext_params={ext_val}): "
                f"result={result}, expected={expected}, match={match}")
            print(f"      int_params after composition: {model.int_params}")
    
    if all_correct:
        print("\n✓ All composed model simulations passed!")
    else:
        print("\n✗ Some results did not match!")
    
    print("="*70)
    return all_correct

def test_double_composition_with_workflow():
    """Test doubly-composed models in the workflow."""
    
    print("\n" + "="*70)
    print("DOUBLE COMPOSED MODEL SIMULATION TEST")
    print("="*70)
    
    checkpoint_dir = Path("./test_checkpoints_double_composed")
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    
    # --- Composition 1: Scale internal by 2 ---
    def compose_int_v1(int_p, ext_p, sim_p):
        return int_p * 2
    
    ComposedModel_v1 = SimpleModel.compose(
        compose_int_params=compose_int_v1
    )
    
    # --- Composition 2: Add offset to external ---
    def compose_ext_v2(int_p, ext_p, sim_p):
        return ext_p + 10
    
    DoubleComposedModel = ComposedModel_v1.compose(
        compose_ext_params=compose_ext_v2
    )
    
    workflow = SimulationInferenceWorkflow(checkpoint_dir=checkpoint_dir)
    
    int_params_list = [1.0, 2.0]
    ext_params_list = [5.0, 15.0]
    sim_params_list = [None, None]
    
    print(f"\nInput structure:")
    print(f"  Internal params (before): {int_params_list}")
    print(f"  Internal params (after compose v1): {[p * 2 for p in int_params_list]}")
    print(f"  External params (before): {ext_params_list}")
    print(f"  External params (after compose v2): {[p + 10 for p in ext_params_list]}")
    
    model_lists = workflow.run_simulations(
        int_params_list=int_params_list,
        ext_params_list=ext_params_list,
        sim_params_list=sim_params_list,
        model_class=DoubleComposedModel,
        n_jobs=1,
    )
    
    # Restore sim_output
    for int_idx in range(len(model_lists)):
        model_list = model_lists[int_idx]
        if model_list and model_list.results:
            for model, result_dict in zip(model_list.models, model_list.results):
                if model.sim_output is None:
                    model.sim_output = result_dict
    
    print(f"\nResults:")
    all_correct = True
    for int_idx, model_list in model_lists.items():
        original_int = int_params_list[int_idx]
        composed_int = original_int * 2
        print(f"\n  int_idx={int_idx} (original={original_int}, after compose v1={composed_int}):")
        for model_idx, model in enumerate(model_list.models):
            original_ext = ext_params_list[model_idx]
            composed_ext = original_ext + 10
            result = model.sim_output['value'][0]
            expected = composed_int * composed_ext
            match = result == expected
            all_correct = all_correct and match
            print(f"    model[{model_idx}] (ext: {original_ext} → {composed_ext}): "
                f"result={result}, expected={expected}, match={match}")
    
    if all_correct:
        print("\n✓ All double-composed model simulations passed!")
    else:
        print("\n✗ Some results did not match!")
    
    print("="*70)
    return all_correct

def test_composition_with_all_params():
    """Test composition affecting all three parameter types in workflow."""
    
    print("\n" + "="*70)
    print("COMPOSITION (ALL PARAMETERS) WITH WORKFLOW TEST")
    print("="*70)
    
    checkpoint_dir = Path("./test_checkpoints_all_params")
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    
    # --- Extended SimpleModel that uses sim_params ---
    @dataclass
    class ExtendedSimpleModel(Model):
        """SimpleModel that incorporates sim_params in computation."""
        
        def simulate_single(self) -> Dict[str, Any]:
            """Multiply int_params by ext_params, scaled by sim_params multiplier."""
            multiplier = self.sim_params.get('multiplier', 1.0) if self.sim_params else 1.0
            result = self.int_params * self.ext_params * multiplier
            self.sim_output = {
                'value': np.array([result]),
                'shape': (1,)
            }
            return self.sim_output
    
    # --- Composition: Modify all three parameters ---
    def compose_int(int_p, ext_p, sim_p):
        return int_p + 1
    
    def compose_ext(int_p, ext_p, sim_p):
        return ext_p * 2
    
    def compose_sim(int_p, ext_p, sim_p):
        if sim_p is None:
            sim_p = {}
        else:
            sim_p = dict(sim_p)
        sim_p['multiplier'] = 3.0
        return sim_p
    
    ComposedExtendedModel = ExtendedSimpleModel.compose(
        compose_int_params=compose_int,
        compose_ext_params=compose_ext,
        compose_sim_params=compose_sim
    )
    
    workflow = SimulationInferenceWorkflow(checkpoint_dir=checkpoint_dir)
    
    int_params_list = [2.0, 3.0]
    ext_params_list = [4.0, 5.0]
    sim_params_list = [None, None]
    
    print(f"\nInput structure:")
    print(f"  int_params (before): {int_params_list}")
    print(f"  int_params (after +1): {[p + 1 for p in int_params_list]}")
    print(f"  ext_params (before): {ext_params_list}")
    print(f"  ext_params (after *2): {[p * 2 for p in ext_params_list]}")
    print(f"  sim_params: None → {{'multiplier': 3.0}}")
    
    model_lists = workflow.run_simulations(
        int_params_list=int_params_list,
        ext_params_list=ext_params_list,
        sim_params_list=sim_params_list,
        model_class=ComposedExtendedModel,
        n_jobs=1,
    )
    
    # Restore sim_output
    for int_idx in range(len(model_lists)):
        model_list = model_lists[int_idx]
        if model_list and model_list.results:
            for model, result_dict in zip(model_list.models, model_list.results):
                if model.sim_output is None:
                    model.sim_output = result_dict
    
    print(f"\nResults:")
    all_correct = True
    for int_idx, model_list in model_lists.items():
        original_int = int_params_list[int_idx]
        composed_int = original_int + 1
        print(f"\n  int_idx={int_idx} (int: {original_int} → {composed_int}):")
        for model_idx, model in enumerate(model_list.models):
            original_ext = ext_params_list[model_idx]
            composed_ext = original_ext * 2
            result = model.sim_output['value'][0]
            expected = composed_int * composed_ext * 3.0
            match = result == expected
            all_correct = all_correct and match
            print(f"    model[{model_idx}] (ext: {original_ext} → {composed_ext}): "
                f"result={result}, expected={expected} (computation: {composed_int} × {composed_ext} × 3), match={match}")
    
    if all_correct:
        print("\n✓ All all-param composition tests passed!")
    else:
        print("\n✗ Some results did not match!")
    
    print("="*70)
    return all_correct

def test_workflow_with_inference():
    """Test SimulationInferenceWorkflow: simulations → one-pass inference."""
    
    # Clean up any previous checkpoints
    checkpoint_dir = Path("./test_checkpoints")
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    
    # Create workflow
    workflow = SimulationInferenceWorkflow(checkpoint_dir=checkpoint_dir)
    
    # Define parameters
    int_params_list = [
        {'int_params': 2.0},
        {'int_params': 3.0},
        {'int_params': 4.0},
    ]
    ext_params_list = [
        {'value': 5.0},
        {'value': 10.0},
    ]
    sim_params_list = [None, None]
    
    print("=" * 70)
    print("WORKFLOW: SIMULATION + ONE-PASS INFERENCE")
    print("=" * 70)
    
    # =========================================================================
    # PHASE 1: SIMULATIONS
    # =========================================================================
    print("\nPHASE 1: Running Simulations")
    print("-" * 70)
    print(f"Input structure:")
    print(f"  Internal params: {len(int_params_list)} values")
    print(f"  Ext/Sim pairs: {len(ext_params_list)} pairs")
    print(f"  Expected: {len(int_params_list)} ModelLists × {len(ext_params_list)} models")
    
    model_lists = workflow.run_simulations(
        int_params_list=int_params_list,
        ext_params_list=ext_params_list,
        sim_params_list=sim_params_list,
        model_class=SimpleModel,
        n_jobs=1,
    )
    
    print(f"\n✓ Simulations complete: {len(model_lists)} ModelLists created")
    
    # Verify simulation results
    print(f"\nSimulation Results:")
    for int_idx, model_list in model_lists.items():
        int_param_val = int_params_list[int_idx]['int_params']
        print(f"\n  int_idx={int_idx} (int_params={int_param_val}):")
        for model_idx, model in enumerate(model_list.models):
            ext_val = ext_params_list[model_idx]['value']
            result = model.sim_output['value'][0]
            expected = int_param_val * ext_val
            match = abs(result - expected) < 1e-10
            print(f"    model[{model_idx}] (ext={ext_val:>5}): result={result:>6.1f}, expected={expected:>6.1f} ✓" if match else f"    model[{model_idx}] (ext={ext_val:>5}): result={result:>6.1f}, expected={expected:>6.1f} ✗")
    
    # =========================================================================
    # PHASE 2: ONE-PASS INFERENCE
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 2: One-Pass Inference")
    print("-" * 70)
    
    def make_simple_inference_pipeline(model_list, **kwargs):
        """Factory function to create inference pipeline from ModelList."""
        
        # Extract ground truths from simulated models
        ground_truths = [model.sim_output['value'] for model in model_list.models]
        ext_params_batch = [{'value': model.ext_params['value']} for model in model_list.models]
        
        def loss_fn(predicted_list, ground_truth_list):
            """MSE loss aggregated across all models."""
            total_loss = 0.0
            for predicted, gt in zip(predicted_list, ground_truth_list):
                total_loss += np.mean((predicted - gt) ** 2)
            return total_loss / len(ground_truth_list)
        
        # Single pass: infer int_params
        pass_1 = PipelinePass(
            name="Pass_1_infer_int_params",
            model_class=SimpleModel,
            ground_truths=ground_truths,
            ext_params_list=ext_params_batch,
            sim_params_list=[None] * len(ground_truths),
            param_keys_to_infer=['int_params'],
            fixed_params={},
            product_or_zip="zip",
            optimizer=minimize,
            optimizer_kwargs={'method': 'Nelder-Mead'},
        )
        
        return InferencePipeline(
            passes=[pass_1],
            loss_fn=loss_fn,
            n_jobs_per_pass=-1,
        )
    
    # Create inference tasks: one for each int_idx
    inference_tasks = []
    for int_idx in range(len(int_params_list)):
        task = InferenceTask(
            task_key=f"infer_int_{int_idx}",
            int_idx=int_idx,
            pair_indices=None,  # Use all models in the ModelList
            make_pipeline_fn=make_simple_inference_pipeline,
            pipeline_kwargs={},
            initial_guesses=[{'int_params': 1.0}],
        )
        inference_tasks.append(task)
    
    print(f"\nCreated {len(inference_tasks)} inference task(s)")
    for task in inference_tasks:
        print(f"  - {task.task_key}: int_idx={task.int_idx}")
    
    # Run inferences
    inference_results = workflow.run_inferences(
        inference_tasks=inference_tasks,
        model_lists=model_lists,
        n_jobs=1,
    )
    
    print(f"\n✓ Inferences complete: {len(inference_results)} result(s)")
    
    # =========================================================================
    # PHASE 3: VERIFY INFERENCE RESULTS
    # =========================================================================
    print("\n" + "=" * 70)
    print("PHASE 3: Verification")
    print("-" * 70)
    print(f"\nInference Results:")
    
    all_success = True
    for int_idx in range(len(int_params_list)):
        task_key = f"infer_int_{int_idx}"
        result = inference_results[task_key]
        true_int_params = int_params_list[int_idx]['int_params']
        inferred_int_params = result[0].params['int_params']
        error = abs(inferred_int_params - true_int_params)
        success = error < 0.01
        
        print(f"\n  {task_key}:")
        print(f"    True int_params: {true_int_params}")
        print(f"    Inferred int_params: {inferred_int_params:.4f}")
        print(f"    Error: {error:.6f}")
        print(f"    Converged: {result[0].success}")
        print(f"    Final loss: {result[0].loss:.6e}")
        print(f"    Status: {'✓ PASS' if success else '✗ FAIL'}")
        
        all_success = all_success and success
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    checkpoint_status = workflow.get_checkpoint_status()
    print(f"Checkpoint Status:")
    print(f"  Stage: {checkpoint_status.stage}")
    print(f"  Simulation entries: {len(checkpoint_status.simulation_entries)}")
    print(f"  Inference entries: {len(checkpoint_status.inference_entries)}")
    
    if all_success:
        print("\n✓ WORKFLOW TEST PASSED!")
    else:
        print("\n✗ WORKFLOW TEST FAILED!")
    print("=" * 70)
    
    assert all_success, "All inference results should match true parameters"

def test_workflow_with_single_composition_inference():
    """Test workflow: composed simulations → one-pass inference to recover original params."""
    
    checkpoint_dir = Path("./test_checkpoints_composed_inference")
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    
    workflow = SimulationInferenceWorkflow(checkpoint_dir=checkpoint_dir)
    
    # Original parameters before composition
    int_params_list = [
        {'int_params': 2.0},
        {'int_params': 3.0},
        {'int_params': 4.0},
    ]
    ext_params_list = [
        {'value': 5.0},
        {'value': 10.0},
    ]
    sim_params_list = [None, None]
    
    print("=" * 70)
    print("WORKFLOW: SINGLE COMPOSITION + ONE-PASS INFERENCE")
    print("=" * 70)
    
    # Composition: scale int_params by 2
    def compose_int_v1(int_p, ext_p, sim_p):
        # Extract scalar from dict
        int_val = int_p['int_params'] if isinstance(int_p, dict) else int_p
        return int_val * 2
    
    ComposedSimpleModel = SimpleModel.compose(
        compose_int_params=compose_int_v1
    )
    
    print("\nPHASE 1: Running Composed Simulations")
    print("-" * 70)
    print(f"Composition: int_params → int_params * 2")
    print(f"Input structure:")
    print(f"  Original int_params: {[p['int_params'] for p in int_params_list]}")
    print(f"  After composition: {[p['int_params'] * 2 for p in int_params_list]}")
    print(f"  Ext params: {[p['value'] for p in ext_params_list]}")
    
    model_lists = workflow.run_simulations(
        int_params_list=int_params_list,
        ext_params_list=ext_params_list,
        sim_params_list=sim_params_list,
        model_class=ComposedSimpleModel,
        n_jobs=1,
    )
    
    print(f"\n✓ Simulations complete: {len(model_lists)} ModelLists created")
    
    print(f"\nSimulation Results:")
    for int_idx, model_list in model_lists.items():
        original_int = int_params_list[int_idx]['int_params']
        composed_int = original_int * 2
        print(f"\n  int_idx={int_idx} (original={original_int}, composed={composed_int}):")
        for model_idx, model in enumerate(model_list.models):
            ext_val = ext_params_list[model_idx]['value']
            result = model.sim_output['value'][0]
            expected = composed_int * ext_val
            match = abs(result - expected) < 1e-10
            status = "✓" if match else "✗"
            print(f"    model[{model_idx}] (ext={ext_val:>5}): result={result:>6.1f}, expected={expected:>6.1f} {status}")
    
    print("\n" + "=" * 70)
    print("PHASE 2: One-Pass Inference (Recover Original Parameters)")
    print("-" * 70)
    print("Goal: Infer the original int_params before composition")
    print("Strategy: Use composed model in inference to recover original values\n")
    
    def make_composed_inference_pipeline(model_list, **kwargs):
        """Factory for inference pipeline with composed model."""
        
        ground_truths = [model.sim_output['value'] for model in model_list.models]
        ext_params_batch = [model.ext_params for model in model_list.models]
        
        def loss_fn(predicted_list, ground_truth_list):
            """MSE loss aggregated across all models."""
            total_loss = 0.0
            for predicted, gt in zip(predicted_list, ground_truth_list):
                total_loss += np.mean((predicted - gt) ** 2)
            return total_loss / len(ground_truth_list)
        
        # Use the composed model in the inference pipeline
        pass_1 = PipelinePass(
            name="Pass_1_infer_original_int_params",
            model_class=ComposedSimpleModel,
            ground_truths=ground_truths,
            ext_params_list=ext_params_batch,
            sim_params_list=[None] * len(ground_truths),
            param_keys_to_infer=['int_params'],
            fixed_params={},
            product_or_zip="zip",
            optimizer=minimize,
            optimizer_kwargs={'method': 'Nelder-Mead'},
        )
        
        return InferencePipeline(
            passes=[pass_1],
            loss_fn=loss_fn,
            n_jobs_per_pass=-1,
        )
    
    inference_tasks = []
    for int_idx in range(len(int_params_list)):
        task = InferenceTask(
            task_key=f"infer_composed_int_{int_idx}",
            int_idx=int_idx,
            pair_indices=None,
            make_pipeline_fn=make_composed_inference_pipeline,
            pipeline_kwargs={},
            initial_guesses=[{'int_params': 1.0}],
        )
        inference_tasks.append(task)
    
    print(f"Created {len(inference_tasks)} inference task(s)")
    for task in inference_tasks:
        print(f"  - {task.task_key}: int_idx={task.int_idx}")
    
    inference_results = workflow.run_inferences(
        inference_tasks=inference_tasks,
        model_lists=model_lists,
        n_jobs=1,
    )
    
    print(f"\n✓ Inferences complete: {len(inference_results)} result(s)")
    
    print("\n" + "=" * 70)
    print("PHASE 3: Verification")
    print("-" * 70)
    print(f"\nInference Results:")
    
    all_success = True
    for int_idx in range(len(int_params_list)):
        task_key = f"infer_composed_int_{int_idx}"
        result = inference_results[task_key]
        true_original_int = int_params_list[int_idx]['int_params']
        inferred_int = result[0].params['int_params']
        error = abs(inferred_int - true_original_int)
        success = error < 0.01
        
        print(f"\n  {task_key}:")
        print(f"    True original int_params: {true_original_int}")
        print(f"    Inferred int_params: {inferred_int:.4f}")
        print(f"    Error: {error:.6f}")
        print(f"    Converged: {result[0].success}")
        print(f"    Final loss: {result[0].loss:.6e}")
        print(f"    Status: {'✓ PASS' if success else '✗ FAIL'}")
        
        all_success = all_success and success
    
    print("\n" + "=" * 70)
    checkpoint_status = workflow.get_checkpoint_status()
    print(f"Checkpoint Status:")
    print(f"  Stage: {checkpoint_status.stage}")
    print(f"  Simulation entries: {len(checkpoint_status.simulation_entries)}")
    print(f"  Inference entries: {len(checkpoint_status.inference_entries)}")
    
    if all_success:
        print("\n✓ SINGLE COMPOSITION WORKFLOW TEST PASSED!")
    else:
        print("\n✗ SINGLE COMPOSITION WORKFLOW TEST FAILED!")
    print("=" * 70)
    
    assert all_success, "All inference results should match original parameters"

def test_workflow_with_double_composition_inference():
    """Test workflow: double-composed simulations → one-pass inference to recover original params."""
    
    checkpoint_dir = Path("./test_checkpoints_double_composed_inference")
    if checkpoint_dir.exists():
        shutil.rmtree(checkpoint_dir)
    
    workflow = SimulationInferenceWorkflow(checkpoint_dir=checkpoint_dir)
    
    int_params_list = [
        {'int_params': 2.0},
        {'int_params': 3.0},
        {'int_params': 4.0},
    ]
    ext_params_list = [
        {'value': 5.0},
        {'value': 10.0},
    ]
    sim_params_list = [None, None]
    
    print("=" * 70)
    print("WORKFLOW: DOUBLE COMPOSITION + ONE-PASS INFERENCE")
    print("=" * 70)
    
    # Composition 1: Scale internal by 2
    def compose_int_v1(int_p, ext_p, sim_p):
        int_val = int_p['int_params'] if isinstance(int_p, dict) else int_p
        return int_val * 2
    
    ComposedSimpleModel_v1 = SimpleModel.compose(
        compose_int_params=compose_int_v1
    )
    
    # Composition 2: Add offset to external
    def compose_ext_v2(int_p, ext_p, sim_p):
        ext_val = ext_p['value'] if isinstance(ext_p, dict) else ext_p
        return ext_val + 10
    
    DoubleComposedModel = ComposedSimpleModel_v1.compose(
        compose_ext_params=compose_ext_v2
    )
    
    print("\nPHASE 1: Running Double-Composed Simulations")
    print("-" * 70)
    print(f"Composition 1: int_params → int_params * 2")
    print(f"Composition 2: ext_params → ext_params + 10")
    print(f"Input structure:")
    print(f"  Original int_params: {[p['int_params'] for p in int_params_list]}")
    print(f"  After comp 1: {[p['int_params'] * 2 for p in int_params_list]}")
    print(f"  Original ext_params: {[p['value'] for p in ext_params_list]}")
    print(f"  After comp 2: {[p['value'] + 10 for p in ext_params_list]}")
    
    model_lists = workflow.run_simulations(
        int_params_list=int_params_list,
        ext_params_list=ext_params_list,
        sim_params_list=sim_params_list,
        model_class=DoubleComposedModel,
        n_jobs=1,
    )
    
    print(f"\n✓ Simulations complete: {len(model_lists)} ModelLists created")
    
    print(f"\nSimulation Results:")
    for int_idx, model_list in model_lists.items():
        original_int = int_params_list[int_idx]['int_params']
        composed_int = original_int * 2
        print(f"\n  int_idx={int_idx} (original_int={original_int}, composed_int={composed_int}):")
        for model_idx, model in enumerate(model_list.models):
            original_ext = ext_params_list[model_idx]['value']
            composed_ext = original_ext + 10
            result = model.sim_output['value'][0]
            expected = composed_int * composed_ext
            match = abs(result - expected) < 1e-10
            status = "✓" if match else "✗"
            print(f"    model[{model_idx}] (ext: {original_ext} → {composed_ext}): result={result:>6.1f}, expected={expected:>6.1f} {status}")
    
    print("\n" + "=" * 70)
    print("PHASE 2: One-Pass Inference (Recover Original Parameters)")
    print("-" * 70)
    print("Goal: Infer original int_params by using double-composed model")
    print("Strategy: Optimize original int_params against observations from double-composed model\n")
    
    def make_double_composed_inference_pipeline(model_list, **kwargs):
        """Factory for inference pipeline with double-composed model."""
        
        ground_truths = [model.sim_output['value'] for model in model_list.models]
        ext_params_batch = [model.ext_params for model in model_list.models]
        
        def loss_fn(predicted_list, ground_truth_list):
            """MSE loss aggregated across all models."""
            total_loss = 0.0
            for predicted, gt in zip(predicted_list, ground_truth_list):
                total_loss += np.mean((predicted - gt) ** 2)
            return total_loss / len(ground_truth_list)
        
        pass_1 = PipelinePass(
            name="Pass_1_infer_original_int_params",
            model_class=DoubleComposedModel,
            ground_truths=ground_truths,
            ext_params_list=ext_params_batch,
            sim_params_list=[None] * len(ground_truths),
            param_keys_to_infer=['int_params'],
            fixed_params={},
            product_or_zip="zip",
            optimizer=minimize,
            optimizer_kwargs={
                'method': 'BFGS',
            },            
        )
        
        return InferencePipeline(
            passes=[pass_1],
            loss_fn=loss_fn,
            n_jobs_per_pass=-1,
        )
    
    initial_guesses = [
        {'int_params': 0.5},
        {'int_params': 2.0},
        {'int_params': 5.0},
        {'int_params': 8.0},
    ]

    inference_tasks = []
    for int_idx in range(len(int_params_list)):
        task = InferenceTask(
            task_key=f"infer_double_composed_int_{int_idx}",
            int_idx=int_idx,
            pair_indices=None,
            make_pipeline_fn=make_double_composed_inference_pipeline,
            pipeline_kwargs={},
            initial_guesses=initial_guesses,
        )
        inference_tasks.append(task)
    
    print(f"Created {len(inference_tasks)} inference task(s)")
    for task in inference_tasks:
        print(f"  - {task.task_key}: int_idx={task.int_idx}")
    
    inference_results = workflow.run_inferences(
        inference_tasks=inference_tasks,
        model_lists=model_lists,
        n_jobs=1,
    )
    
    print(f"\n✓ Inferences complete: {len(inference_results)} result(s)")
    
    print("\n" + "=" * 70)
    print("PHASE 3: Verification")
    print("-" * 70)
    print(f"\nInference Results:")
    
    all_success = True
    for int_idx in range(len(int_params_list)):
        task_key = f"infer_double_composed_int_{int_idx}"
        result = inference_results[task_key]
        true_original_int = int_params_list[int_idx]['int_params']
        inferred_int = result[0].params['int_params']
        error = abs(inferred_int - true_original_int)
        success = error < 0.01
        
        print(f"\n  {task_key}:")
        print(f"    True original int_params: {true_original_int}")
        print(f"    Inferred int_params: {inferred_int:.4f}")
        print(f"    Error: {error:.6f}")
        print(f"    Converged: {result[0].success}")
        print(f"    Final loss: {result[0].loss:.6e}")
        print(f"    Status: {'✓ PASS' if success else '✗ FAIL'}")
        
        all_success = all_success and success
    
    print("\n" + "=" * 70)
    checkpoint_status = workflow.get_checkpoint_status()
    print(f"Checkpoint Status:")
    print(f"  Stage: {checkpoint_status.stage}")
    print(f"  Simulation entries: {len(checkpoint_status.simulation_entries)}")
    print(f"  Inference entries: {len(checkpoint_status.inference_entries)}")
    
    if all_success:
        print("\n✓ DOUBLE COMPOSITION WORKFLOW TEST PASSED!")
    else:
        print("\n✗ DOUBLE COMPOSITION WORKFLOW TEST FAILED!")
    print("=" * 70)
    
    assert all_success, "All inference results should match original parameters"


# ========== Run all workflow tests ==========
if __name__ == "__main__":
    test_workflow()
    test_composition_with_workflow()
    test_double_composition_with_workflow()
    test_composition_with_all_params()
    test_workflow_with_inference()
    test_workflow_with_single_composition_inference()
    test_workflow_with_double_composition_inference()    
    print("\n" + "="*70)
    print("ALL COMPOSITION TESTS COMPLETED")
    print("="*70)