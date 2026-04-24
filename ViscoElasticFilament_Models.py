import numpy as np
from typing import Any, Dict, List, Optional, Sequence
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.linalg import solve
from scipy.optimize import root, approx_fprime
from Models import Model, compose_model, parallel_simulate_batch
import time

## --- Initial conditions --- ##

def StraightLine(N):
    """ A straight line """
    X_0 = np.zeros(N+2, dtype = np.double)
    return X_0

def Bend(N, k = 2, phi = np.pi / 4):
    """ A bend (of the kth segment). k = 2 and phi = pi/4 corresponds to ProximalBend """
    X_0 = np.zeros(N+2, np.double)
    X_0[k] = phi
    return X_0

def ProximalBend(N):
    """ A proximal bend (of the first segment) """
    X_0 = Bend(N, k = 2, phi = np.pi/4)
    return X_0    

def SecondBend(N):
    """ A proximal bend (of the second segment) """
    X_0 = Bend(N, k = 3, phi = np.pi/1024)
    return X_0

## --- Parameter functions --- ##

def Theta(X, k):
    """ Returns Theta_k from X_Np2. """

    theta_k = np.sum(X[2:k+3])
    return theta_k

def X2(X, i):
    """ Returns non-dimensional position vector X_i from X_Np2. """

    X_2 = np.zeros(2)
    # print("X[0]", X[0])
    # print("X_2[0]", X_2[0])
    X_2[0] = X[0]
    X_2[1] = X[1]
    for k in range(i):
        theta_k = Theta(X, k)
        X_2[0] += np.cos(theta_k)
        X_2[1] += np.sin(theta_k)
    return X_2

def X3N(X):
    """ Returns adimensional X_3N vector from X_Np2. """

    N = X.shape[0]-2
    X_3N = np.zeros((3*N,1), dtype=np.double)

    for i in range(N):
        X_2 = X2(X, i) # X_2 is adimensional
        theta_i = Theta(X, i)
        X_3N[i,0] = X_2[0]
        X_3N[N+i, 0] = X_2[1]
        X_3N[2*N+i, 0] = theta_i

    return X_3N

def XNp2(X_3N):
    """ Returns adimensional X_Np2 vector from X_3N. """

    N = X_3N.shape[0]//3
    X_Np2 = np.zeros((N+2,), dtype=np.double)

    X_Np2[0] = X_3N[0,0] # x_0
    X_Np2[1] = X_3N[N,0] # y_0
    X_Np2[2] = X_3N[2*N,0] # alpha_0 = theta_0
    X_Np2[3:] = X_3N[2*N+1:,0] - X_3N[2*N:-1,0] # alpha_i = theta_i - theta_{i-1}
    return X_Np2

# Unit test
# N = 5
# X = ProximalBend(N)
# X_3N = X3N(X)
# print("X = ", X)
# print("X_3N = ", X_3N)

def QQ(X_3N):
    """ Non-dimensional Transfert matrix from X_Np2_dot to X_3N_dot. 
    It is shape-dependent while the opposite transfert matrix is not. """

    N = X_3N.shape[0]//3
    Q = np.zeros((3*N, N+2))
    Q[:N,0] = np.ones((N,1)).ravel()
    Q[N:2*N,1] = np.ones((N,1)).ravel()

    Q_x = np.zeros((N,N))
    Q_y = np.zeros((N,N))
    Q_theta = np.tri(N)
    
    for i in range(1,N):
        theta_im1 = X_3N[2*N+i-1,0]
        Q_x[i,:i] = Q_x[i-1,:i] - np.sin(theta_im1)*np.ones((1,i))
        Q_y[i,:i] = Q_y[i-1,:i] + np.cos(theta_im1)*np.ones((1,i))
    Q[:N,2:] = Q_x
    Q[N:2*N,2:] = Q_y
    Q[2*N:3*N,2:] = Q_theta
    return Q

## --- Fluid drag --- ##

def Eta(mu, r, L, h = -1):
    """ Returns the parallel drag coefficient (RFT). """
    if h<0:
        return 2 * np.pi * mu / (np.log(L/r) - 0.807)
    else:
        return 2 * np.pi * mu / np.log(2*h/r)

def Xi(mu, r, L, h = -1):
    """ Returns the perpendicular drag coefficient (RFT). """
    if h<0:
        return 4 * np.pi * mu / (np.log(L/r) + 0.193)
    else:
        return 2 * Eta(mu, r, L, h)

def GG(theta, gamma):
    """ Computes the matrix G of fluid drag for RFT computation. """

    G = np.zeros((3,5))
    cos = np.cos(theta)
    sin = np.sin(theta)

    G[0,0] = (gamma-1)*cos*sin
    G[1,1] = - G[0,0]
    G[0,1] = - sin**2 - gamma*cos**2
    G[1,0] = cos**2 + gamma*sin**2
    G[0,2] = - gamma*cos/2
    G[2,1] = G[0,2]
    G[1,2] = - gamma*sin/2
    G[2,0] = - G[1,2]
    G[2,2] = - gamma/3

    G[0,3] = 0
    G[0,4] = 0
    G[1,3] = 0
    G[1,4] = 0    

    # Components used for external flow only
    G[2,3] = - gamma * sin
    G[2,4] = gamma * cos

    return G

def UU(X_3N, k, gamma):
    """ Non-dimensional operator G @ [X_dot] """
    N = X_3N.shape[0]//3
    theta_k = X_3N[2*N+k, 0]
    G = GG(theta_k, gamma)
    U = np.zeros((3, 3*N))
    U[:,k] = G[:,0]
    U[:,N+k] = G[:,1]
    U[:,2*N+k] = G[:,2]
    return U

def DD(X_3N, k, i):
    """ Returns non-dimensional (x_k-x_j, y_k-y_j, 1) """
    D = np.zeros((1,3))
    N = X_3N.shape[0]//3
    D[0,0] = X_3N[k,0] - X_3N[i,0]
    D[0,1] = X_3N[N+k,0] - X_3N[N+i,0]
    D[0,2] = 1
    return D

def AA(X_3N, gamma):
    """ Computes and returns non-dimensional A(X_3N) such that A @ X_3N_dot = A @ [Q @ X_dot] = B. """

    N = X_3N.shape[0]//3
    A = np.zeros((N+2,3*N))

    A[0,0] = 1 # 1*x0_dot = b_0
    A[1,N] = 1 # 1*y0_dot = b_1
    # A[2,2*N] = 1 # 1*theta_0_dot = b_2
    for j in range(0, N):
        for i in range(j, N):
            A[j+2,:] = A[j+2,:] + DD(X_3N, i, j) @ UU(X_3N, i, gamma)
    return A

## A dashpots
def ADB(taus_b, N):
    """ Returns the matrix used to model bending dashpots all along the axoneme.
    taus_b is a list of non-dimensional internal bending viscosities. """

    if len(taus_b)==(N-1):
        A_DB = np.diag([0,0,0] + taus_b)
    else:
        A_DB = np.zeros((N+2, N+2))
    return A_DB

def ADS(N):
    """ Returns the matrix A_DS used to model shear dissipation all along the axoneme.
    taus_s is a list of non-dimensional internal shearing viscosity.
    Remark: A_DS applies to X_3N_dot and not directly to X_Np2_dot """

    A_DS = np.zeros((N+2,3*N))
    A_DS[2:, 2*N:] = np.triu(np.ones((N, N))) # upper triangular matrix
    A_DS[2:, 2*N] = np.arange(-N, 0, 1) # change first column
    return A_DS

# N = 10
# A_DS = ADS(N)
# print("A_DS = ", A_DS)
# exit()

## --- External flow --- ##

def CreateFlowField(A = 0., w0 = 0., psi = 0., T_meas = [], filename = ""):

    """ Creates a non-dimensional flow field and returns a string and an array representing resp. the type and data
    - 1. If filename = "" and if A = 0 or T_meas = [], there is no flow field: return 0
    - 2. If filemame = "" and if A > 0 and if w0 = 0, returns a constant homogeneous flow # TODO: change so the time dimension disappears
    - 3. If filemame = "" and if A > 0 and if w0 > 0, returns [A*sin(t)] for t in T (homogeneous flow)
    - 4. If a filename is given, import flow field from the file (PIV);
    Returned flow field in the 2 last cases is an array of shape (2x|T|)
    psi is the angle between the x-axis and the flow.

    Note: for a non-dimensional flow, A should be chosen in non-dimensional units and t is non-dimensional (i.e., counted in w0 units)

    """

    X_flow_field = 0
    return_string = "NO FLOW"

    if filename == "": # Cases 1,2,3
        if len(T_meas)==0 or A == 0: # Case 1
            return return_string, X_flow_field
        else: # Cases 2,3
            
            if w0==0 and len(T_meas)==1: # Case 2 # TODO: test this case
                X_flow_field = A * np.array([np.cos(psi), np.sin(psi)]).reshape((2,1))
                return_string = "CONSTANT FLOW: (psi, A) = (" + str(psi) + ", " + str(A) + ")"
                return return_string, X_flow_field

            else: # Case 3
                if w0 == 0:
                    X_flow_field = A * np.array([np.cos(psi), np.sin(psi)]).reshape((2,1)) @ np.ones((1, len(T_meas)))
                else:
                    X_flow_field = A * np.array([np.cos(psi), np.sin(psi)]).reshape((2,1)) @ np.sin(w0*T_meas[:]).reshape((1,-1))
                return_string = "SINE FLOW: (psi, A, w0) = (" + str(psi) + ", " + str(A) + ", " + str(w0) + ")"

                return return_string, X_flow_field

    else: # Case 4
        # Import field from filename (Change later) # TODO: include experimental data
        return_string = "PIV-IMPORTED from " + filename
        return return_string, X_flow_field 

def Flow(X_3N, X_flow_field = np.array([0]) ):

    """ Computes non-dimensional average flow speed and 1st moment of flow speed on each axoneme segment
    given a flow vector field X_flow_field. There are N segments, numerated from 0 to N-1.
    If a homogeneous flow is imposed, the flow is supposed to be constant within the same segment and
    the first moment is a simple average. """

    N = X_3N.shape[0]//3
    X_dot_flow = np.zeros((4*N,1))

    # No flow is imposed
    if np.shape(X_flow_field)[0] == 1:
        return X_dot_flow

    # A homogeneous flow is imposed
    elif np.shape(X_flow_field)[0] == 2:
        X_dot_flow[:N, 0] = X_flow_field[0] # Flow velocity on x axis
        X_dot_flow[N:2*N, 0] = X_flow_field[1] # Flow velocity on y axis
        X_dot_flow[2*N:3*N, 0] = (1 / 2) * X_flow_field[0] # First moment of flow velocity on x axis
        X_dot_flow[3*N:, 0] = (1 / 2) * X_flow_field[1] # First moment of flow velocity on y axis
        return X_dot_flow

    # An inhomogeneous flow is imposed, e.g. with PIV experiments
    else: 
        # Add things here later
        return X_dot_flow

def TT_flow(X_dot_flow, k):
    """ Non-dimensional operator [X_dot_flow]_k where X_dot_flow is of shape (4*N x 1). 
    Returns the external flow components that contribute to the hydrodynamic drag, i.e.,
        - the flow speed on each segment (2 scalars),
        - the first moment of flow speed on each segment (2 scalars),
        - there is no direct contribution for the angle (see equations) (1 scalar). 
    """

    T_flow_k = np.zeros((5,1))
    N = X_dot_flow.shape[0]//4
    T_flow_k[2] = 0 # No (direct!) contribution for the filament angle

    T_flow_k[0] = - X_dot_flow[k] # Opposite of Flow velocity on x axis
    T_flow_k[1] = - X_dot_flow[N+k] # Opposite of Flow velocity on y axis
    T_flow_k[3] = X_dot_flow[2*N+k] # First moment of flow velocity on x axis
    T_flow_k[4] = X_dot_flow[3*N+k] # First moment of flow velocity on y axis

    return T_flow_k

## --- Right-hand side --- ##

def BC_L(X_3N, n_L=[0,0], m_L=0):
    """Returns non-dimensional B_C representing boundary conditions at the distal end. 
    Zero is default for a free end. n_L and m_L are chosen adimensionally."""

    N = X_3N.shape[0]//3
    B_C = np.zeros((N+2,1))

    ## point force and point moment at distal end.
    x_L = X_3N[N-1, 0] + np.cos(X_3N[-1, 0])
    y_L = X_3N[2*N-1, 0] + np.sin(X_3N[-1, 0])
    
    B_C[0] = - n_L[0]
    B_C[1] = - n_L[1]
    B_C[2:] = (y_L - X_3N[N:2*N])*n_L[0] - (x_L - X_3N[:N])*n_L[1] - m_L

    return B_C

def BC_0(X_3N, n_0 = [0,0], m_0 = 0):
    """ Returns non-dimensional right-hand side of the differential system for boundary conditions at s = 0 (proximal end). """

    N = X_3N.shape[0]//3
    B_C = np.zeros((N+2,1))

    B_C[0] = n_0[0] # force equation on x axis
    B_C[1] = n_0[1] # force equation y axis
    B_C[2] = m_0
    # Partial filament torque balances (B_C[3:]) does not depend on torque or force at s = 0.
    
    return B_C

def BB(X_3N): # Argument X_3N could be replaced by X
    """ Returns non-dimensional right-hand side of the differential system for bending elasticity. """

    N = X_3N.shape[0]//3
    B = np.zeros((N+2,1))

    B[0] = 0 # force equation (here on x axis) is not affected by elasticity
    B[1] = 0 # force equation (here on y axis) is not affected by elasticity    
    B[2] = 0 # total torque equation does not depend on bending resistance

    # Bending resistance (constitutive equations)
    B[3:] = (X_3N[2*N+1:, :] - X_3N[2*N:-1, :]) # Bending resistance
    
    return B

def BS(X_3N):
    """ Returns non-dimensional right-hand side of the differential system for shear elasticity. """

    N = X_3N.shape[0]//3
    B = np.zeros((N+2,1))

    # Boundary conditions at proximal end
    B[0] = 0 # No force on x axis due to shear at s = 0
    B[1] = 0 # No force on y axis due to shear at s = 0
    B[2] = 0 # No torque due to shear at s = 0
    for i in range(N-1):
        B[3+i] = np.sum(X_3N[2*N+i+1:, 0]) - (N-i-1)*X_3N[2*N, 0] # Sliding resistance        
    return B

def BFlow(X_3N, X_dot_flow, gamma):
    """ Returns non-dimensional B_flow representing moments due to background flow. This is similar to computations on the left-hand side of the differential equation.
    Importantly, when put on the right-hand side of the equation one should add a minus sign. """

    N = X_3N.shape[0]//3
    B_flow = np.zeros((N+2, 1))

    B_flow[0,0] = 0
    B_flow[1,0] = 0

    for j in range(N):
        for i in range(j, N):
            theta_i = X_3N[2*N+i, 0]
            B_flow[j+2,0] += np.squeeze(DD(X_3N, i, j) @ GG(theta_i, gamma) @ TT_flow(X_dot_flow, i))
    return B_flow

def BF(X_3N, Lambdas):
    """ returns B_F representing non-dimensional moments of uniform density forces on each segment. """
    N = len(Lambdas)
    B_F = np.zeros((N+2,1))
    if Lambdas == [0]*N:
        return B_F
    else:
        for j in range(N):
            for i in range(j, N):
                Lambda_i = Lambdas[i]
                B_F[j+2,0] = B_F[j+2,0] + Lambda_i[1] * (X_3N[i, 0] - X_3N[j, 0] + np.cos(X_3N[2*N+i, 0])/2) - Lambda_i[0] * (X_3N[N+i, 0] - X_3N[N+j, 0] + np.sin(X_3N[2*N+i, 0])/2)
        return B_F

def BM(Zetas):
    """returns B_M representing torques between each segment. """
    N = len(Zetas)
    B_M = np.zeros((N+2,1))
    if Zetas == [0]*N:
        return B_M
    else:
        for j in range(N):
            B_M[j+2,0] = np.sum(Zetas[j:])
        return B_M

### Active bending moments

def ActiveBending(X):
    # try with local (one node or more) constant + dissipation ()
    N = X.shape[0]-2
    B_active = np.zeros((N+2,1))
    return B_active

## --- Differential system AQX_dot = B --- ##

def g(
    t, 
    X, 
    Sp4, k0, bool_EI, Beta, 
    taus_b, tau_s=0,
    gamma=2, 
    n_L=[0,0], m_L=0,
    Lambdas=0, Zetas=0, InterpFlow=0,
):

    # --- Setup ---
    N = X.shape[0] - 2
    X_3N = X3N(X)

    x = X_3N[:N, 0]
    y = X_3N[N:2*N, 0]
    theta = X_3N[2*N:, 0]

    # --- Boundary conditions ---
    n_0 = n_L
    m_0 = k0 * X[2]

    # --- Precompute GG ---
    G_all = [GG(theta[i], gamma) for i in range(N)]

    # --- Build A (optimized, no U_i) ---
    A = np.zeros((N+2, 3*N))
    A[0, 0] = 1
    A[1, N] = 1

    for j in range(N):
        row = A[j+2]

        for i in range(j, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]

            D = np.array([dx, dy, 1.0])

            G_i = G_all[i]
            dG = D @ G_i   # shape (5,)

            row[i]     += dG[0]
            row[N+i]   += dG[1]
            row[2*N+i] += dG[2]

    # --- Q matrix ---
    Q = QQ(X_3N)

    # --- Dissipation ---
    A_DB = int(bool_EI) * ADB(taus_b, N)
    A_DS = tau_s * ADS(N)

    # --- Flow --- #
    if callable(InterpFlow):
        X_flow = InterpFlow(t) # Only t-dependent component
    else:
        if type(InterpFlow) == float:
            X_flow = np.array([0])
        elif InterpFlow.size == 2: 
            X_flow = InterpFlow
        
    X_dot_flow = Flow(X_3N, X_flow)

    # --- BFlow ---
    B_flow = np.zeros((N+2, 1))

    for j in range(N):
        for i in range(j, N):
            dx = x[i] - x[j]
            dy = y[i] - y[j]

            D = np.array([[dx, dy, 1.0]])

            T = TT_flow(X_dot_flow, i)
            B_flow[j+2, 0] += (D @ G_all[i] @ T).item()

    # --- RHS ---
    B = (
        int(bool_EI) * BB(X_3N)
        + BC_L(X_3N, n_L, m_L)
        + BC_0(X_3N, n_0, m_0)
        + Beta * BS(X_3N)
        - BF(X_3N, Lambdas)
        - BM(Zetas)
        + ActiveBending(X)
        - Sp4 * B_flow
    )

    # --- System matrix ---
    A_tilde = (Sp4 * A - Beta * A_DS) @ Q - A_DB

    # --- Solve ---
    try:
        X_dot = np.linalg.solve(A_tilde, B).ravel()
    except np.linalg.LinAlgError:
        X_dot = (np.linalg.pinv(A_tilde) @ B).ravel()

    # --- Enforce base constraints: this is necessary to avoid error propagation. ---
    X_dot[0] = 0
    X_dot[1] = 0

    return X_dot

def ViscoElasticFilament_Simulate(int_params, ext_params, sim_params):
    """
    Run forward simulation for a single instance of the viscoelastic filament.
    This method should populate sim_output with the result before returning.
    Returns: {"value": np.ndarray, "shape": tuple} # Is shape necessary?
    """

    Args = (
        # Internal parameters
        int_params['Sp4'],
        int_params['k0'],
        int_params['bool_EI'],
        int_params['Beta'],
        int_params['taus_b'],
        int_params['tau_s'],
        int_params['gamma'],
        int_params['n_L'],
        int_params['m_L'],
        # External parameters
        ext_params['Lambdas'],
        ext_params['Zetas'], 
        ext_params['InterpFlow'],
    )

    # Define the time limiter
    class StopOnTime:
        def __init__(self, max_simulation_time):
            self.max_simulation_time = max_simulation_time
            self.start_time = time.time()

        def terminate_integration(self, t, y, *args):
            if time.time() - self.start_time > self.max_simulation_time:
                return 0
            return 1
    time_limiter = StopOnTime(sim_params['T_sim_max'])

    # Run the simulation
    try:
        if callable(ext_params["InterpFlow"]):
            sol = solve_ivp(
                fun = g, 
                t_span = sim_params['T_span'], 
                y0 = int_params['X_0'],
                args=Args, 
                t_eval=sim_params['T_eval'], 
                method = sim_params['method'], 
                events=time_limiter.terminate_integration,
            )

            T_sim = time.time() - time_limiter.start_time
            if sol.t_events[0].size > 0:
                T_sim = np.inf
                mistake = np.array(["Solving aborted: too long."])
                print(mistake)
                sim_output = {"value": None, "shape": None}
            else:
                sim_output = {"value": sol.y, "shape": sol.y.shape}            
        else:
            sol = root(lambda x:g(0, x, *Args), int_params['X_0'], method=sim_params["method"])
            J = approx_fprime(sol.x, lambda x: g(0, x, *Args), epsilon=1e-8)
            eigenvalues = np.linalg.eigvals(J)
            is_stable = np.all(np.real(eigenvalues) <= 0)  # (Meta-)stable if all Re(λ) <(=) 0
            assert is_stable, f"Not meta-stable equilibrium: np.real(eigenvalues) <= 0 {np.real(eigenvalues)}"
            sim_output = {"value": sol.x, "shape": sol.x.shape}   

    except Exception as ex:
        T_sim = np.inf
        mistake = np.array([str(ex)])
        print(mistake)
        sim_output = {"value": None, "shape": None}

    return sim_output

class ViscoElasticFilament(Model):
    def __init__(self, int_params: Any, ext_params: Any, sim_params: Any):
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

        res = ViscoElasticFilament_Simulate(self.int_params, self.ext_params, self.sim_params)
        self.sim_output = res
        return res

    @classmethod
    def simulate_batch( # TODO: vectorize this function
        cls,
        int_params_batch: Sequence[Any],
        ext_params_batch: Sequence[Any],
        sim_params_batch: Sequence[Any],
    ) -> List[Dict[str, Any]]:
        """
        Vectorized batch implementation (if possible).
        Returns: List[{"value": np.ndarray, "shape": tuple}]
        """
        
        results = []
        for ip, ep, sp in zip(int_params_batch, ext_params_batch, sim_params_batch):
            instance = cls(ip, ep, sp)
            output = instance.simulate_single()
            results.append(output)
        return results

# Define the pre-determined function to transform (A, w0, psi) into InterpFlow
def FlowParams_to_InterpFlow(int_params, ext_params, sim_params):
    """ Transform external parameters (Lambdas, Zetas, A, w0, psi) -> (Lambdas, Zetas, InterpFlow). """
    
    ## Intermediate parameters
    T_eval = sim_params['T_eval']
    X_flow_field_string, X_flow_field = CreateFlowField(ext_params['A'], ext_params['w0'], ext_params['psi'], T_eval)

    ## Final parameters # TODO: modify so that one gets an interpolated flow in time except when "CONSTANT FLOW" (len(T_eval)==1)
    if ("NO FLOW" in X_flow_field_string):
        InterpFlow = 0
    elif ("CONSTANT FLOW" in X_flow_field_string):
        InterpFlow = X_flow_field # TODO: To be completed
    else:         
        InterpFlow = interp1d(np.array(T_eval).reshape(len(T_eval),), X_flow_field, axis=1, fill_value="extrapolate")

    return {'Lambdas': ext_params['Lambdas'], 'Zetas': ext_params['Zetas'], 'InterpFlow': InterpFlow}
    
# Define the ViscoElasticFilament_FlowParams class by composing the ViscoElasticFilament class
ViscoElasticFilament_FlowParams = compose_model(
    ViscoElasticFilament,
    compose_ext_params=FlowParams_to_InterpFlow,
)

# Define the pre-determined function to transform (A, w0, psi) into InterpFlow
def tau_b_to_taus_b(int_params, ext_params, sim_params):
    """ Transform internal parameters (..., tau_b, ...) -> (..., taus_b, ...). """
    new_int_params = int_params.copy()
    new_int_params['taus_b'] = [new_int_params['tau_b']]*(new_int_params['N']-1) # Making uniform distribution of taus_b.
    new_int_params.pop('tau_b')
    return new_int_params

ViscoElasticFilament_FlowParams_ScalarBending = compose_model(
    ViscoElasticFilament_FlowParams,
    compose_int_params=tau_b_to_taus_b,
)