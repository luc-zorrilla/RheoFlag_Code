""" Libraries """
import numpy as np
import types
from concurrent.futures import ProcessPoolExecutor # Parallel computations

""" Classes """
class Model:

    def __init__(self, int_params, ext_params, sim_params, simulate):
        """ Basic constructor of the class. """
        self.int_params = int_params
        self.ext_params = ext_params
        self.sim_params = sim_params

        # Bind methods to the instance
        self.simulate = types.MethodType(simulate, self) # This method allows to simulate a model given its internal, external and simulation parameters

""" Functions """

def _simulate_single(int_params, ext_params, sim_params, simulate):
    """
    Helper function to create a Model instance and run simulate.
    This is needed because ProcessPoolExecutor can't pickle bound methods directly.
    """
    model_instance = Model(int_params, ext_params, sim_params, simulate)
    return model_instance.simulate(int_params, ext_params, sim_params)

def parallel_simulate(int_params_list, ext_params_list, sim_params_list, simulate):
    """ 
    Simulate various instances of a model given varying internal, external, and simulation parameters.
    Each iteration creates a new Model instance and runs simulate in parallel.

    Args:
        int_params_list: List of internal parameters
        ext_params_list: List of external parameters
        sim_params_list: List of simulation parameters
        simulate: Function defining the simulation logic

    Returns:
        List of simulation results
    """
    # Ensure all lists are of the same length
    assert len(int_params_list) == len(ext_params_list) == len(sim_params_list)

    args = zip(int_params_list, ext_params_list, sim_params_list)

    # Run simulations in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(_simulate_single, int_params, ext_params, sim_params, simulate)
            for int_params, ext_params, sim_params in args
        ]
        results = [future.result() for future in futures]

    return results


if __name__ == "__main__":
    """ The main code in this script will only be used for testing functions defined above."""
    print(None)