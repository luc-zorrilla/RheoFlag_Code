import numpy as np
from typing import Any, Dict, List, Optional, Sequence
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import time


## --- Differential system AQX_dot = B --- ##
    
def g(t, X, Sp4, k0, bool_EI, Beta, taus_b, tau_s = 0, gamma = 2, n_L=[0,0], m_L=0, Lambdas=0, Zetas=0, InterpFlow = 0):

    """ Returns the non-dimensionalized equation X_tilde_dot = g(X_tilde; t; parameters). 
    The difference with f(t,X) is that X is extended to add theta_0_dot, giving X_tilde. 
    Since a second order equation in time is perscribed at the base, it can be turned 
    into a first order equation and added to the matricial system.
    """

    # Boundary conditions (basal hinge, free distal end)
    n_0 = n_L # No displacement at the base
    m_0 = k0*X[2] # Rotation at the base is allowed

    ##################################################################
    ###### Solve the linear system with infinite basal stiffness #####
    N = X.shape[0]-2
    X_3N = X3N(X)

    A = AA(X_3N, gamma)

    Q = QQ(X_3N)

    A_DB = int(bool_EI) * ADB(taus_b, N)

    A_DS = tau_s * ADS(N)

    if InterpFlow == 0: # No external flow is given
        X_dot_flow = Flow(X_3N)
    else: # External flow is given
        X_flow = InterpFlow(t) 
        X_dot_flow = Flow(X_3N, X_flow)

    B = int(bool_EI) * BB(X_3N) + BC_L(X_3N, n_L, m_L) + BC_0(X_3N, n_0, m_0) + Beta * BS(X_3N) - BF(X_3N, Lambdas) - BM(Zetas) + ActiveBending(X) - Sp4 * BFlow(X_3N, X_dot_flow, gamma)

    A_tilde = (Sp4 * A - Beta * A_DS ) @ Q - A_DB

    X_dot = (np.linalg.inv(A_tilde) @ B).ravel()

    ##################################################################

    # Enforce \dot(x0) = \dot(y0) = 0, because error propagation breaks it.
    X_dot[0] = 0
    X_dot[1] = 0
    
    return X_dot

def ViscoElasticFilament_Simulate(int_params, ext_params, sim_params):
    """
    Run forward simulation for a single instance of the viscoelastic filament.
    This method should populate sim_output with the result before returning.
    Returns: {"value": np.ndarray, "shape": tuple} # Is shape necessary?
    """

    X_0, N, taus_b, tau_s, gamma, n_L, m_L, Sp4, k0 = int_params # Change taus_b for tau_b? Or tau_s for taus_s?
    Lambdas, Zetas, A, w0, X_flow_field_string, X_flow_field, InterpFlow = ext_params
    T_span, T_eval, T_sim_max, method = sim_params

    # --- COMPLETE THIS PART --- #
    # If the flow has been interpolated, use it as is.
    # Else, if the flow field is not existent (X_flow_field_string = "NO FLOW"), do without it.
    #    # If the flow field is not given and not constructed yet, construct it with (A,w0).
    #    # If the flow field is already constructed or given, interpolate it.
    # --- COMPLETE THIS PART --- #
    
    # Create an interpolation function for the flow field if necessary
    if X_flow_field_string != "NO FLOW":
        InterpFlow = interp1d(np.array(T_eval).reshape(len(T_eval),), X_flow_field, axis=1, fill_value="extrapolate")
    else:
        InterpFlow = 0
    Args = (Sp4, k0, True, gamma, taus_b, tau_s, gamma, n_L, m_L, Lambdas, Zetas, InterpFlow)

    # Define the time limiter
    class StopOnTime:
        def __init__(self, max_simulation_time):
            self.max_simulation_time = max_simulation_time
            self.start_time = time.time()

        def terminate_integration(self, t, y):
            if time.time() - self.start_time > self.max_simulation_time:
                return 0
            return 1
    time_limiter = StopOnTime(T_sim_max)

    # Run the simulation
    try:
        sol = solve_ivp(fun = g, t_span = T_span, y0 = X_0, args=Args, t_eval=T_eval, method = method, events=time_limiter.terminate_integration)
        T_sim = time.time() - time_limiter.start_time

        if sol.t_events[0].size > 0:
            T_sim = np.inf
            mistake = np.array(["Solving aborted: too long."])
            print(mistake)
            sim_output = {"value": None, "shape": None}
        else:
            sim_output = {"value": sol.y, "shape": sol.y.shape}
    except Exception as ex:
        T_sim = np.inf
        mistake = np.array([str(ex)])
        print(mistake)
        sim_output = {"value": None, "shape": None}

    return sim_output

class ViscoElasticFilament(Model):
    def __init__(self, int_params: np.ndarray, ext_params: Any, sim_params: Any):
        super().__init__(int_params, ext_params, sim_params)
        self.int_params = int_params
        self.ext_params = ext_params
        self.sim_params = sim_params

    def simulate_single(self) -> Dict[str, Any]:
        """
        Run forward simulation for a single instance of the viscoelastic filament.
        This method should populate self.sim_output with the result before returning.
        Returns: {"value": np.ndarray, "shape": tuple} # Is shape necessary?
        """

        return ViscoElasticFilament_Simulate(self.int_params, self.ext_params, self.sim_params)
