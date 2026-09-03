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
    model_class_list: list,
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
                model_class_list[int_idx],
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
        model_class_list: list,
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
            model_class_list,
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