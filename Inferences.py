from typing import Type, Optional, Any, Callable, List, Dict, Union, Tuple, ClassVar
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Inference:
    """
    Base class for model inference.

    Attributes:
        optimizer: Optimizer instance for parameter search.
        loss: Loss function instance for evaluating model performance.
        ground_truth: Dictionary with keys:
            - 'y': target output as np.ndarray
            - 'ext_params': external parameters
            - 'output_shape': expected shape of model output
        model: Model class to be inferred (set as a class attribute in subclasses).
        inferred_model: Model instance with inferred parameters.
    """

    model: ClassVar[Type["Model"]]  # To be set by subclasses

    def __init__(
        self,
        optimizer: Any,
        loss: Callable,
        ground_truth: Dict[str, Any],  # {'y': np.ndarray, 'ext_params': Any, 'output_shape': tuple}
        **kwargs: Any,
    ):
        self.optimizer = optimizer
        self.loss = loss
        self.ground_truth = ground_truth
        self.inferred_model = None
        self.kwargs = kwargs

    def _check_output_shape(self, sim_output: np.ndarray) -> None:
        """Check that the model output shape matches the ground truth."""
        if sim_output.shape != self.ground_truth["output_shape"]:
            raise ValueError(
                f"Model output shape {sim_output.shape} does not match ground truth shape {self.ground_truth['output_shape']}"
            )

    def simple_infer(
        self,
        initial_int_params: np.ndarray,
        sim_params: Any = None,
    ) -> "Model":
        """
        Simple inference: use the optimizer to find the best internal parameters.

        Args:
            initial_int_params: Initial guess for internal parameters.
            sim_params: Simulation parameters (optional).

        Returns:
            Model instance with inferred parameters.
        """
        def objective(int_params: np.ndarray) -> float:
            """Objective function for the optimizer."""
            m = self.model(int_params, self.ground_truth["ext_params"], sim_params)
            sim_output = m.simulate_single()
            self._check_output_shape(sim_output)
            return self.loss(sim_output, self.ground_truth["y"])

        result = self.optimizer.minimize(objective, initial_int_params)
        self.inferred_model = self.model(result.x, self.ground_truth["ext_params"], sim_params)
        return self.inferred_model

    def infer(self, *args, **kwargs) -> "Model":
        """Subclasses should override this method for custom inference logic."""
        raise NotImplementedError

class Square_Inference(Inference):
    """
    Inference class specialized for the Square model.
    """
    model = Square  # Class attribute: this inference is for Square models

    def infer(
        self,
        initial_int_params: np.ndarray,
        sim_params: Any = None,
    ) -> "Square":
        """
        Infer the best internal parameter for the Square model.

        Args:
            initial_int_params: Initial guess for internal parameters.
            sim_params: Simulation parameters (optional).

        Returns:
            Square model instance with inferred parameters.
        """
        logger.info("Running Square model inference...")
        return self.simple_infer(initial_int_params, sim_params)

# class ViscoElasticFilament_Inference(Inference):
#     """
#     Inference class specialized for the ViscoElasticFilament model.
#     """
#     model = ViscoElasticFilament  # Class attribute: this inference is for ViscoElasticFilament models

#     def infer(
#         self,
#         initial_int_params: np.ndarray,
#         sim_params: Any = None,
#     ) -> "ViscoElasticFilament":
#         """
#         Infer the best internal parameters for the ViscoElasticFilament model.

#         Args:
#             initial_int_params: Initial guess for internal parameters.
#             sim_params: Simulation parameters (optional).

#         Returns:
#             ViscoElasticFilament model instance with inferred parameters.
#         """
#         logger.info("Running ViscoElasticFilament model inference...")
#         return self.simple_infer(initial_int_params, sim_params)