from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from joblib import Parallel, delayed
from datetime import datetime
import json
import dill as pickle

from Models import Model, ModelList
from Inferences import Inference, InferencePipeline, InferenceResult


@dataclass
class CheckpointEntry:
    """Metadata for a single completed task."""
    task_key: str  # "{int_idx}_{ext_idx}_{sim_idx}" for simulations
    completed: bool
    timestamp: str
    filepath: str  # relative path to saved artifact


@dataclass
class WorkflowCheckpoint:
    """High-level checkpoint tracking the entire workflow."""
    simulation_entries: Dict[str, CheckpointEntry] = field(default_factory=dict)
    inference_entries: Dict[int, CheckpointEntry] = field(default_factory=dict)
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
    
    # ============================================================================
    # Artifact Storage (Models, Inference Results)
    # ============================================================================
    
    def save_artifact(self, artifact_type: str, key: str, artifact: Any) -> str:
        """
        Save an artifact (ModelList, InferenceResult, etc.) and return relative path.
        
        Args:
            artifact_type: "models" or "inference"
            key: Unique identifier (e.g., "0_1_2" for (int_idx, ext_idx, sim_idx))
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
    
    # ============================================================================
    # Checkpoint Persistence
    # ============================================================================
    
    def save_checkpoint(self, checkpoint: WorkflowCheckpoint):
        """Save checkpoint metadata to JSON."""
        data = {
            "stage": checkpoint.stage,
            "timestamp": checkpoint.timestamp,
            "simulation_entries": {
                k: {"task_key": v.task_key, "completed": v.completed, "timestamp": v.timestamp, "filepath": v.filepath}
                for k, v in checkpoint.simulation_entries.items()
            },
            "inference_entries": {
                str(k): {"task_key": v.task_key, "completed": v.completed, "timestamp": v.timestamp, "filepath": v.filepath}
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
            int(k): CheckpointEntry(**v)
            for k, v in data.get("inference_entries", {}).items()
        }
        
        return WorkflowCheckpoint(
            simulation_entries=simulation_entries,
            inference_entries=inference_entries,
            stage=data.get("stage", "simulation"),
            timestamp=data.get("timestamp", ""),
        )
    
    # ============================================================================
    # Query Methods
    # ============================================================================
    
    def is_simulation_done(self, int_idx: int, ext_idx: int, sim_idx: int) -> bool:
        """Check if simulation task is completed."""
        checkpoint = self.load_checkpoint()
        if not checkpoint:
            return False
        task_key = f"{int_idx}_{ext_idx}_{sim_idx}"
        return checkpoint.simulation_entries.get(task_key, CheckpointEntry(task_key, False, "", "")).completed
    
    def is_inference_done(self, task_id: int) -> bool:
        """Check if inference task is completed."""
        checkpoint = self.load_checkpoint()
        if not checkpoint:
            return False
        return checkpoint.inference_entries.get(task_id, CheckpointEntry("", False, "", "")).completed


# ============================================================================
# SIMULATION STAGE
# ============================================================================

def run_single_simulation(
    int_idx: int,
    ext_idx: int,
    sim_idx: int,
    int_params: dict,
    ext_params: dict,
    sim_params: dict,
    model_class: type,
    checkpoint_mgr: CheckpointManager,
) -> ModelList:
    """
    Simulate a single parameter set using ModelList.
    
    Args:
        int_idx, ext_idx, sim_idx: Indices for nested loops
        int_params, ext_params, sim_params: Parameter dicts
        model_class: The Model subclass to instantiate
        checkpoint_mgr: CheckpointManager for resuming
    
    Returns:
        ModelList with results populated
    """
    task_key = f"{int_idx}_{ext_idx}_{sim_idx}"
    
    # Check if already completed
    if checkpoint_mgr.is_simulation_done(int_idx, ext_idx, sim_idx):
        model_list = checkpoint_mgr.load_artifact("models", task_key)
        if model_list:
            return model_list
    
    # Create ModelList with single Model (or batch if needed)
    model_list = ModelList.from_params(
        model_class=model_class,
        int_params_batch=[int_params],
        ext_params_batch=[ext_params],
        sim_params_batch=[sim_params],
    )
    
    # Run simulation (serial since it's already one model)
    model_list.simulate(n_jobs=1, parallel=False)
    
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


def parallel_simulate(
    int_params_list: List[dict],
    ext_params_list: List[dict],
    sim_params_list: List[dict],
    model_class: type,
    checkpoint_mgr: CheckpointManager,
    n_jobs: int = -1,
) -> Dict[Tuple[int, int, int], ModelList]:
    """
    Simulate all parameter combinations in parallel (nested loops).
    
    Returns:
        Dict mapping (int_idx, ext_idx, sim_idx) -> ModelList
    """
    tasks = [
        (int_idx, ext_idx, sim_idx, int_params_list[int_idx], ext_params_list[ext_idx], sim_params_list[sim_idx])
        for int_idx in range(len(int_params_list))
        for ext_idx in range(len(ext_params_list))
        for sim_idx in range(len(sim_params_list))
    ]
    
    print(f"Total simulation tasks: {len(tasks)}")
    
    pending_tasks = [
        t for t in tasks
        if not checkpoint_mgr.is_simulation_done(t[0], t[1], t[2])
    ]
    
    if pending_tasks:
        print(f"Running {len(pending_tasks)} pending simulations...")
        Parallel(n_jobs=n_jobs)(
            delayed(run_single_simulation)(
                t[0], t[1], t[2], t[3], t[4], t[5], model_class, checkpoint_mgr
            )
            for t in pending_tasks
        )
    
    # Load all completed models
    models = {}
    for int_idx, ext_idx, sim_idx, _, _, _ in tasks:
        task_key = f"{int_idx}_{ext_idx}_{sim_idx}"
        model_list = checkpoint_mgr.load_artifact("models", task_key)
        models[(int_idx, ext_idx, sim_idx)] = model_list
    
    print(f"✓ Simulations complete: {len(models)} ModelLists")
    return models


# ============================================================================
# INFERENCE STAGE
# ============================================================================

@dataclass
class InferenceTask:
    """A single inference job definition."""
    task_id: int
    model_indices: Tuple[Tuple[int, int, int], ...]  # Which ModelLists to use
    make_pipeline_fn: Callable[..., InferencePipeline]  # Factory function
    pipeline_kwargs: Dict[str, Any] = field(default_factory=dict)  # Args to factory
    initial_guesses: List[Dict[str, float]] = field(default_factory=list)  # For Inference


def run_single_inference_task(
    task: InferenceTask,
    model_lists: Dict[Tuple[int, int, int], ModelList],
    checkpoint_mgr: CheckpointManager,
) -> Any:
    """
    Run a single inference task with pipeline factory.
    
    Args:
        task: InferenceTask defining the inference job
        model_lists: Dict of available ModelLists
        checkpoint_mgr: CheckpointManager for checkpointing
    
    Returns:
        Inference result (depends on pipeline)
    """
    if checkpoint_mgr.is_inference_done(task.task_id):
        return checkpoint_mgr.load_artifact("inference", str(task.task_id))
    
    # Gather ModelLists for this task
    task_model_lists = [model_lists[idx] for idx in task.model_indices]
    
    # Create pipeline via factory function
    pipeline = InferencePipeline.from_factory(
        task.make_pipeline_fn,
        model_lists=task_model_lists,
        **task.pipeline_kwargs,
    )
    
    # Run pipeline (handles multi-pass inference internally)
    result = pipeline.run()
    
    # Checkpoint
    artifact_path = checkpoint_mgr.save_artifact("inference", str(task.task_id), result)
    checkpoint = checkpoint_mgr.load_checkpoint() or WorkflowCheckpoint()
    checkpoint.inference_entries[task.task_id] = CheckpointEntry(
        task_key=str(task.task_id),
        completed=True,
        timestamp=str(datetime.now()),
        filepath=artifact_path,
    )
    checkpoint.stage = "inference"
    checkpoint.timestamp = str(datetime.now())
    checkpoint_mgr.save_checkpoint(checkpoint)
    
    return result


def parallel_infer(
    inference_tasks: List[InferenceTask],
    model_lists: Dict[Tuple[int, int, int], ModelList],
    checkpoint_mgr: CheckpointManager,
    n_jobs: int = -1,
) -> Dict[int, Any]:
    """
    Run all inference tasks in parallel.
    
    Returns:
        Dict mapping task_id -> inference result
    """
    print(f"Total inference tasks: {len(inference_tasks)}")
    
    pending_tasks = [
        t for t in inference_tasks
        if not checkpoint_mgr.is_inference_done(t.task_id)
    ]
    
    if pending_tasks:
        print(f"Running {len(pending_tasks)} pending inferences...")
        Parallel(n_jobs=n_jobs)(
            delayed(run_single_inference_task)(task, model_lists, checkpoint_mgr)
            for task in pending_tasks
        )
    
    # Load all results
    results = {}
    for task in inference_tasks:
        result = checkpoint_mgr.load_artifact("inference", str(task.task_id))
        results[task.task_id] = result
    
    print(f"✓ Inferences complete: {len(results)} results")
    return results


# ============================================================================
# HIGH-LEVEL WORKFLOW ORCHESTRATOR
# ============================================================================

class SimulationInferenceWorkflow:
    """Orchestrates simulation → inference pipeline with checkpoint support."""
    
    def __init__(self, checkpoint_dir: Path = Path("./checkpoints")):
        self.checkpoint_mgr = CheckpointManager(checkpoint_dir)
    
    def run(
        self,
        int_params_list: List[dict],
        ext_params_list: List[dict],
        sim_params_list: List[dict],
        model_class: type,
        inference_tasks: List[InferenceTask],
        n_jobs_simulation: int = -1,
        n_jobs_inference: int = -1,
    ) -> Tuple[Dict[Tuple[int, int, int], ModelList], Dict[int, Any]]:
        """
        Execute simulation → inference pipeline with resume capability.
        
        Args:
            int_params_list: List of internal parameter dicts
            ext_params_list: List of external parameter dicts
            sim_params_list: List of simulation parameter dicts
            model_class: Model subclass to instantiate
            inference_tasks: List of InferenceTask objects
            n_jobs_simulation: Parallelization for simulations
            n_jobs_inference: Parallelization for inferences
        
        Returns:
            Tuple of (model_lists_dict, inference_results_dict)
        """
        # Stage 1: Simulate all parameter combinations
        model_lists = parallel_simulate(
            int_params_list,
            ext_params_list,
            sim_params_list,
            model_class,
            self.checkpoint_mgr,
            n_jobs=n_jobs_simulation,
        )
        
        # Stage 2: Run inferences
        inference_results = parallel_infer(
            inference_tasks,
            model_lists,
            self.checkpoint_mgr,
            n_jobs=n_jobs_inference,
        )
        
        return model_lists, inference_results
    
    # ========================================================================
    # Result Retrieval
    # ========================================================================
    
    def get_inference_result(self, task_id: int) -> Optional[Any]:
        """Retrieve a specific inference result by task_id."""
        return self.checkpoint_mgr.load_artifact("inference", str(task_id))
    
    def get_model_list(self, int_idx: int, ext_idx: int, sim_idx: int) -> Optional[ModelList]:
        """Retrieve a specific ModelList by indices."""
        task_key = f"{int_idx}_{ext_idx}_{sim_idx}"
        return self.checkpoint_mgr.load_artifact("models", task_key)
    
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
