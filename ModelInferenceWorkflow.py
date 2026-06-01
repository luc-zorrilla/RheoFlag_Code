from dataclasses import dataclass, asdict
from pathlib import Path
import json
import dill as pickle
from typing import Any, List, Tuple, Dict, Optional
from joblib import Parallel, delayed
from datetime import datetime

from Models import Model, ModelList
from Inferences import Inference, InferencePipeline, InferenceResult

# ============================================================================
# 2. ENHANCED CHECKPOINT SYSTEM
# ============================================================================

@dataclass
class SimulationTaskResult:
    """Metadata for a single simulation task."""
    int_params_key: str
    ext_params_key: str
    sim_params_key: str
    completed: bool
    timestamp: str
    model_file: str


@dataclass
class InferenceTaskResult:
    """Metadata for a single inference task."""
    task_id: int
    int_params_key: str
    indices: Tuple[int, ...]
    inference_mode: str
    completed: bool
    timestamp: str
    result_file: str


@dataclass
class WorkflowCheckpoint:
    """High-level checkpoint tracking the entire workflow."""
    simulation_tasks: Dict[str, SimulationTaskResult]  # key: f"{int_idx}_{ext_idx}_{sim_idx}"
    inference_tasks: Dict[int, InferenceTaskResult]   # key: task_id
    stage: str  # "simulation" or "inference"
    timestamp: str


class CheckpointManager:
    """Enhanced checkpoint system with nested loop support."""
    
    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.checkpoint_dir / "checkpoint.json"
        self.models_dir = self.checkpoint_dir / "models"
        self.results_dir = self.checkpoint_dir / "results"
        self.models_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
    
    # ========== MODEL SAVING/LOADING ==========
    
    def save_model(self, int_idx: int, ext_idx: int, sim_idx: int, model: Model) -> str:
        """Save a Model instance, return relative filename."""
        filename = f"model_{int_idx}_{ext_idx}_{sim_idx}.pkl"
        filepath = self.models_dir / filename
        with open(filepath, "wb") as f:
            pickle.dump(model, f)
        return filename
    
    def load_model(self, int_idx: int, ext_idx: int, sim_idx: int) -> Optional[Model]:
        """Load a Model instance."""
        filename = f"model_{int_idx}_{ext_idx}_{sim_idx}.pkl"
        filepath = self.models_dir / filename
        if filepath.exists():
            with open(filepath, "rb") as f:
                return pickle.load(f)
        return None
    
    # ========== INFERENCE RESULT SAVING/LOADING ==========
    
    def save_inference_result(self, task_id: int, result: Any) -> str:
        """Save inference result (InferenceResult or list of InferenceResults)."""
        filename = f"inference_task_{task_id}.pkl"
        filepath = self.results_dir / filename
        with open(filepath, "wb") as f:
            pickle.dump(result, f)
        return filename
    
    def load_inference_result(self, task_id: int) -> Optional[Any]:
        """Load inference result."""
        filename = f"inference_task_{task_id}.pkl"
        filepath = self.results_dir / filename
        if filepath.exists():
            with open(filepath, "rb") as f:
                return pickle.load(f)
        return None
    
    # ========== CHECKPOINT MANAGEMENT ==========
    
    def save_checkpoint(self, checkpoint: WorkflowCheckpoint):
        """Save checkpoint with full metadata."""
        # Convert to serializable format
        data = {
            "stage": checkpoint.stage,
            "timestamp": checkpoint.timestamp,
            "simulation_tasks": {
                k: {
                    "int_params_key": v.int_params_key,
                    "ext_params_key": v.ext_params_key,
                    "sim_params_key": v.sim_params_key,
                    "completed": v.completed,
                    "timestamp": v.timestamp,
                    "model_file": v.model_file,
                }
                for k, v in checkpoint.simulation_tasks.items()
            },
            "inference_tasks": {
                str(k): {
                    "task_id": v.task_id,
                    "int_params_key": v.int_params_key,
                    "indices": v.indices,
                    "inference_mode": v.inference_mode,
                    "completed": v.completed,
                    "timestamp": v.timestamp,
                    "result_file": v.result_file,
                }
                for k, v in checkpoint.inference_tasks.items()
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
        
        simulation_tasks = {
            k: SimulationTaskResult(**v)
            for k, v in data.get("simulation_tasks", {}).items()
        }
        
        inference_tasks = {
            int(k): InferenceTaskResult(**v)
            for k, v in data.get("inference_tasks", {}).items()
        }
        
        return WorkflowCheckpoint(
            simulation_tasks=simulation_tasks,
            inference_tasks=inference_tasks,
            stage=data.get("stage", "simulation"),
            timestamp=data.get("timestamp", ""),
        )
    
    def is_simulation_done(self, int_idx: int, ext_idx: int, sim_idx: int) -> bool:
        """Check if a specific simulation task is completed."""
        checkpoint = self.load_checkpoint()
        if not checkpoint:
            return False
        task_key = f"{int_idx}_{ext_idx}_{sim_idx}"
        return checkpoint.simulation_tasks.get(task_key, SimulationTaskResult(
            "", "", "", False, "", ""
        )).completed
    
    def is_inference_done(self, task_id: int) -> bool:
        """Check if a specific inference task is completed."""
        checkpoint = self.load_checkpoint()
        if not checkpoint:
            return False
        return checkpoint.inference_tasks.get(task_id, InferenceTaskResult(
            -1, "", (), "", False, "", ""
        )).completed


# ============================================================================
# 3. SIMULATION STAGE (with nested loop support)
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
    ) -> Model:
    """
    Simulate a single parameter set. Uses checkpoint to skip if already done.
    
    Args:
        int_idx, ext_idx, sim_idx: Indices for nested loops
        int_params, ext_params, sim_params: Parameter dicts
        model_class: The Model subclass to instantiate
        checkpoint_mgr: CheckpointManager for resuming
    
    Returns:
        Model instance with output_data populated
    """
    # Check if already computed
    if checkpoint_mgr.is_simulation_done(int_idx, ext_idx, sim_idx):
        model = checkpoint_mgr.load_model(int_idx, ext_idx, sim_idx)
        if model:
            return model
    
    # Create and simulate model
    model = model_class(int_params, ext_params, sim_params)
    model.simulate()
    
    # Save model and update checkpoint
    model_file = checkpoint_mgr.save_model(int_idx, ext_idx, sim_idx, model)
    checkpoint = checkpoint_mgr.load_checkpoint() or WorkflowCheckpoint({}, {}, "simulation", "")
    task_key = f"{int_idx}_{ext_idx}_{sim_idx}"
    checkpoint.simulation_tasks[task_key] = SimulationTaskResult(
        int_params_key=str(int_idx),
        ext_params_key=str(ext_idx),
        sim_params_key=str(sim_idx),
        completed=True,
        timestamp=str(datetime.now()),
        model_file=model_file,
    )
    checkpoint.stage = "simulation"
    checkpoint.timestamp = str(datetime.now())
    checkpoint_mgr.save_checkpoint(checkpoint)
    
    return model

def parallel_simulate(
    int_params_list: List[dict],
    ext_params_list: List[dict],
    sim_params_list: List[dict],
    model_class: type,
    checkpoint_mgr: CheckpointManager,
    n_jobs: int = -1,
    ) -> Dict[Tuple[int, int, int], Model]:
    """
    Simulate all parameter combinations in parallel (nested loops).
    
    Args:
        int_params_list, ext_params_list, sim_params_list: Lists of params
        model_class: Model subclass to use
        checkpoint_mgr: CheckpointManager
        n_jobs: joblib parallelization
    
    Returns:
        Dict mapping (int_idx, ext_idx, sim_idx) -> Model instance
    """
    # Generate all tasks
    tasks = [
        (int_idx, ext_idx, sim_idx, int_params_list[int_idx], 
        ext_params_list[ext_idx], sim_params_list[sim_idx])
        for int_idx in range(len(int_params_list))
        for ext_idx in range(len(ext_params_list))
        for sim_idx in range(len(sim_params_list))
    ]
    
    print(f"Total simulation tasks: {len(tasks)}")
    
    # Filter out completed tasks
    pending_tasks = [
        t for t in tasks
        if not checkpoint_mgr.is_simulation_done(t[0], t[1], t[2])
    ]
    
    if pending_tasks:
        print(f"Running {len(pending_tasks)} pending simulations...")
        results = Parallel(n_jobs=n_jobs)(
            delayed(run_single_simulation)(
                t[0], t[1], t[2], t[3], t[4], t[5], model_class, checkpoint_mgr
            )
            for t in pending_tasks
        )
    
    # Load all models (cached or newly computed)
    models = {}
    for int_idx, ext_idx, sim_idx, _, _, _ in tasks:
        model = checkpoint_mgr.load_model(int_idx, ext_idx, sim_idx)
        models[(int_idx, ext_idx, sim_idx)] = model
    
    print(f"✓ Simulations complete: {len(models)} models")
    return models

# ============================================================================
# 4. INFERENCE STAGE (with pipeline abstraction)
# ============================================================================

@dataclass
class InferenceTask:
    """A single inference job definition."""
    task_id: int
    int_idx: int
    indices: Tuple[Tuple[int, int, int], ...]  # Which model indices to combine
    inference_mode: str  # "single" or "cumulative"
    int_params_key: str
    ext_params_key: str
    sim_params_key: str


def run_single_inference_pipeline(
    task: InferenceTask,
    models: Dict[Tuple[int, int, int], Model],
    checkpoint_mgr: CheckpointManager,
    ) -> Any:
    """
    Run inference pipeline for a single task.
    
    Returns:
        InferenceResult (single mode) or List[InferenceResult] (cumulative mode)
    """
    # Check if already computed
    if checkpoint_mgr.is_inference_done(task.task_id):
        return checkpoint_mgr.load_inference_result(task.task_id)
    
    # Get models for this task
    task_models = [models[idx] for idx in task.indices]
    
    if task.inference_mode == "single":
        # Single inference from one model
        model = task_models[0]
        
        # Create pipeline (subclass-specific)
        pipeline = model.create_inference_pipeline()
        
        # Run inference
        result = pipeline.infer(
            ground_truth=model.output_data,
            int_params=model.int_params,
            ext_params=model.ext_params,
            sim_params=model.sim_params,
        )
    
    elif task.inference_mode == "cumulative":
        # Cumulative inference from multiple models
        first_model = task_models[0]
        
        # Create pipeline (subclass-specific)
        pipeline = first_model.create_inference_pipeline()
        
        # Prepare lists
        ground_truth_list = [m.output_data for m in task_models]
        ext_params_list = [m.ext_params for m in task_models]
        sim_params_list = [m.sim_params for m in task_models]
        
        # Run inference
        result = pipeline.infer( # TODO: replace by infer_batch?
            ground_truth_list=ground_truth_list,
            int_params=first_model.int_params,
            ext_params_list=ext_params_list,
            sim_params_list=sim_params_list,
        )
    
    # Save result and update checkpoint
    result_file = checkpoint_mgr.save_inference_result(task.task_id, result)
    checkpoint = checkpoint_mgr.load_checkpoint() or WorkflowCheckpoint({}, {}, "inference", "")
    checkpoint.inference_tasks[task.task_id] = InferenceTaskResult(
        task_id=task.task_id,
        int_params_key=task.int_params_key,
        indices=task.indices,
        inference_mode=task.inference_mode,
        completed=True,
        timestamp=str(datetime.now()),
        result_file=result_file,
    )
    checkpoint.stage = "inference"
    checkpoint.timestamp = str(datetime.now())
    checkpoint_mgr.save_checkpoint(checkpoint)
    
    return result


def parallel_infer(
    inference_tasks: List[InferenceTask],
    models: Dict[Tuple[int, int, int], Model],
    checkpoint_mgr: CheckpointManager,
    n_jobs: int = -1,
) -> Dict[int, Any]:
    """
    Run all inference tasks in parallel, with checkpoint support.
    
    Returns:
        Dict mapping task_id -> InferenceResult(s)
    """
    print(f"Total inference tasks: {len(inference_tasks)}")
    
    # Filter out completed tasks
    pending_tasks = [
        t for t in inference_tasks
        if not checkpoint_mgr.is_inference_done(t.task_id)
    ]
    
    if pending_tasks:
        print(f"Running {len(pending_tasks)} pending inferences...")
        Parallel(n_jobs=n_jobs)(
            delayed(run_single_inference_pipeline)(task, models, checkpoint_mgr)
            for task in pending_tasks
        )
    
    # Load all results (cached or newly computed)
    results = {}
    for task in inference_tasks:
        result = checkpoint_mgr.load_inference_result(task.task_id)
        results[task.task_id] = result
    
    print(f"✓ Inferences complete: {len(results)} results")
    return results


# ============================================================================
# 5. ORCHESTRATOR
# ============================================================================

class InferenceWorkflow:
    """High-level orchestrator for two-stage pipeline."""
    
    def __init__(self, checkpoint_dir: Path = Path("./checkpoints")):
        self.checkpoint_mgr = CheckpointManager(checkpoint_dir)
    
    def run(
        self,
        int_params_list: List[dict],
        ext_params_list: List[dict],
        sim_params_list: List[dict],
        model_class: type,
        inference_tasks: List[InferenceTask],
        n_jobs: int = -1,
    ):
        """
        Execute simulation → inference pipeline with resume capability.
        
        Args:
            int_params_list: List of internal parameter dicts
            ext_params_list: List of external parameter dicts
            sim_params_list: List of simulation parameter dicts
            model_class: Model subclass to instantiate
            inference_tasks: List of InferenceTask objects
            n_jobs: joblib parallelization (-1 for all cores)
        
        Returns:
            Tuple of (models_dict, inference_results_dict)
        """
        # Stage 1: Simulate all parameter combinations
        models = parallel_simulate(
            int_params_list,
            ext_params_list,
            sim_params_list,
            model_class,
            self.checkpoint_mgr,
            n_jobs=n_jobs,
        )
        
        # Stage 2: Run inferences
        inference_results = parallel_infer(
            inference_tasks,
            models,
            self.checkpoint_mgr,
            n_jobs=n_jobs,
        )
        
        return models, inference_results
    
    def get_results(self, task_id: int) -> Any:
        """Retrieve a specific inference result by task_id."""
        return self.checkpoint_mgr.load_inference_result(task_id)
    
    def get_model(self, int_idx: int, ext_idx: int, sim_idx: int) -> Optional[Model]:
        """Retrieve a specific model by indices."""
        return self.checkpoint_mgr.load_model(int_idx, ext_idx, sim_idx)
    
    def get_checkpoint_status(self) -> Optional[WorkflowCheckpoint]:
        """Get current checkpoint status."""
        return self.checkpoint_mgr.load_checkpoint()
    
    def clear_checkpoint(self):
        """Clear all checkpoints (start fresh)."""
        import shutil
        if self.checkpoint_mgr.checkpoint_dir.exists():
            shutil.rmtree(self.checkpoint_mgr.checkpoint_dir)
        self.checkpoint_mgr.checkpoint_dir.mkdir(parents=True, exist_ok=True)

# ============================================================================
# 6. EXAMPLE USAGE & SUBCLASS IMPLEMENTATION
# ============================================================================

if __name__ == "__main__":
    from pathlib import Path
    
    # Define parameter lists
    int_params_list = [
        {"a": 1.0},
        {"a": 1.5},
        {"a": 2.0},
    ]
    
    ext_params_list = [
        {"b": 0.5},
        {"b": 1.0},
        {"b": 1.5},
    ]
    
    sim_params_list = [
        {"c": 0.01},
        {"c": 0.05},
    ]
    
    # Define inference tasks
    inference_tasks = [
        # Single inferences: one model each
        InferenceTask(
            task_id=0,
            int_idx=0,
            indices=((0, 0, 0),),
            inference_mode="single",
            int_params_key="0",
            ext_params_key="0",
            sim_params_key="0",
        ),
        InferenceTask(
            task_id=1,
            int_idx=1,
            indices=((1, 1, 1),),
            inference_mode="single",
            int_params_key="1",
            ext_params_key="1",
            sim_params_key="1",
        ),
        # Cumulative inference: combine multiple models
        InferenceTask(
            task_id=2,
            int_idx=0,
            indices=((0, 0, 0), (0, 1, 0), (0, 2, 1)),
            inference_mode="cumulative",
            int_params_key="0",
            ext_params_key="multi",
            sim_params_key="multi",
        ),
        InferenceTask(
            task_id=3,
            int_idx=2,
            indices=((2, 0, 0), (2, 1, 1), (2, 2, 0)),
            inference_mode="cumulative",
            int_params_key="2",
            ext_params_key="multi",
            sim_params_key="multi",
        ),
    ]
    
    # Create workflow
    workflow = InferenceWorkflow(checkpoint_dir=Path("./my_inference_checkpoints"))
    
    # Run workflow (can be interrupted and resumed)
    models, inference_results = workflow.run(
        int_params_list=int_params_list,
        ext_params_list=ext_params_list,
        sim_params_list=sim_params_list,
        model_class=MyModel,
        inference_tasks=inference_tasks,
        n_jobs=4,
    )
    
    print("\n" + "="*70)
    print("WORKFLOW COMPLETE")
    print("="*70)
    
    # Retrieve and inspect results
    for task_id in range(len(inference_tasks)):
        result = workflow.get_results(task_id)
        print(f"\nTask {task_id}:")
        print(f"  Type: {type(result).__name__}")
        print(f"  Result: {result}")
    
    # Inspect checkpoint status
    checkpoint = workflow.get_checkpoint_status()
    print(f"\nCheckpoint Status:")
    print(f"  Stage: {checkpoint.stage}")
    print(f"  Simulations completed: {len(checkpoint.simulation_tasks)}")
    print(f"  Inferences completed: {len(checkpoint.inference_tasks)}")
    print(f"  Last updated: {checkpoint.timestamp}")