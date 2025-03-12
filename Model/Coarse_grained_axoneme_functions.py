""" This file contains all functions used to simulate a viscoelastic filament. """

### libraries ###
from turtle import color
from matplotlib import markers
import numpy as np
import pandas as pd
from regex import E
from scipy.integrate import solve_ivp
from scipy import interpolate
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import time
from datetime import datetime

import json
import codecs

#############################
### ----- Functions ----- ###
#############################

################################
## --- Metadata as a dict --- ##

def write_dict_to_json_file(dictionary, file_name):
    """
    Writes a Python dictionary to a JSON file in a user-friendly format.
    Converts NumPy arrays to lists for JSON serialization.

    Args:
        dictionary (dict): The dictionary to write to the JSON file.
        file_name (str): The name of the JSON file to write to.
    """
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert NumPy array to list
        raise TypeError(f"Type {type(obj)} is not JSON serializable")

    try:
        with open(file_name, 'w') as json_file:
            json.dump(dictionary, json_file, indent=4, default=convert_numpy)

    except Exception as e:
        print(f"An error occurred: {e}")

def read_dict_from_json_file(file_name):
    """
    Reads a Python dictionary from a JSON file.
    Converts lists back to NumPy arrays where applicable.

    Args:
        file_name (str): The name of the JSON file to read from.

    Returns:
        dict: The dictionary read from the JSON file, with lists converted back to NumPy arrays.
    """
    def convert_back_to_numpy(obj):
        if isinstance(obj, list):
            return np.array(obj)  # Convert lists back to NumPy arrays
        return obj

    try:
        with open(file_name, 'r') as json_file:
            dictionary = json.load(json_file)

        # Recursively convert lists back to NumPy arrays
        for key, value in dictionary.items():
            if isinstance(value, list):
                dictionary[key] = convert_back_to_numpy(value)

        print(f"Dictionary with NumPy arrays has been successfully read from {file_name}")
        return dictionary
    except FileNotFoundError:
        print(f"The file {file_name} does not exist.")
    except json.JSONDecodeError:
        print("Error decoding JSON from file.")
    except Exception as e:
        print(f"An error occurred: {e}")

################################

##############################
## --- Data as csv file --- ##

def write_array_to_csv(array, filename):
    """
    Write a NumPy array of any shape to a CSV file.
    The shape is saved in the first line of the CSV file.

    Parameters:
    - array: NumPy array of any shape
    - filename: the name of the CSV file (without extension)
    """
    array = np.array(array)
    shape = array.shape

    # Open the file in write mode
    with open(f'{filename}', 'w') as f:
        # Write the shape in the first line
        f.write(','.join(map(str, shape)) + '\n')

        # Flatten the array and write the data
        np.savetxt(f, array.flatten(), delimiter=',')

    print(f"Array successfully written to {filename}")

def read_array_from_csv(filename):
    """
    Read a NumPy array of any shape from a CSV file.
    The shape is obtained from the first line of the CSV file.

    Parameters:
    - filename: the name of the CSV file (without extension)

    Returns:
    - NumPy array reshaped to its original dimensions
    """
    # Open the CSV file and read the first line (shape)
    with open(f'{filename}', 'r') as f:
        # Read the first line which contains the shape
        shape_line = f.readline().strip()
        shape = tuple(map(int, shape_line.split(',')))

        # Read the flattened array data
        flat_array = np.loadtxt(f, delimiter=',')

    # Reshape it back to its original shape
    array = flat_array.reshape(shape)
    
    print(f"Array successfully read from {filename}.csv")
    return array

##############################

################################################################################
## --- Reading metadata and data

def get_metadata(metadata_filename):

    solver_dict = read_dict_from_json_file(metadata_filename) # contains '.json' in the name
    return solver_dict

def get_data(data_filename):

    sol = read_array_from_csv(data_filename) # contains '.csv' in the name
    t_sim = sol[0,:]
    sol_y = sol[1:,:]

    return t_sim, sol_y
################################################################################

################################
## --- Initial conditions --- ##

def StraightLine(N):
    """ A straight line """
    X_0 = np.zeros(N+2, dtype = np.double)
    return X_0

def ProximalBend(N):
    """ A bend after the first segment """
    X_0 = np.zeros(N+2, np.double)
    X_0[3] = np.pi/4
    return X_0

def SmoothCurve(N):
    """ A curve with constant curvature, so that the total shear angle is pi/2 """
    X_0 = np.zeros(N+2, np.double)
    X_0[3:] = (np.pi/2)/N
    return X_0

# Add reading the last position from a file and start from there? e.g. for changing integration method.

## --- Initial conditions --- ##
################################


#################################
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
        X_3N[i] = X_2[0]
        X_3N[N+i] = X_2[1]
        X_3N[2*N+i] = theta_i

    return X_3N

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
        theta_im1 = X_3N[2*N+i-1]
        Q_x[i,:i] = Q_x[i-1,:i] - np.sin(theta_im1)*np.ones((1,i))
        Q_y[i,:i] = Q_y[i-1,:i] + np.cos(theta_im1)*np.ones((1,i))
    Q[:N,2:] = Q_x
    Q[N:2*N,2:] = Q_y
    Q[2*N:3*N,2:] = Q_theta
    return Q

## --- Parameter functions --- ##
#################################


########################
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

    cos = np.cos(theta)
    sin = np.sin(theta)

    G = np.zeros((3,5))
    G[0,0] = (gamma-1)*cos*sin
    G[1,1] = - G[0,0]
    G[0,1] = - sin**2 - gamma*cos**2
    G[1,0] = cos**2 + gamma*sin**2
    G[0,2] = - gamma*cos/2
    G[2,1] = G[0,2]
    G[1,2] = - gamma*sin/2
    G[2,0] = - G[1,2]
    G[2,2] = - gamma/3

    # Components used for external flow only
    G[2,3] = - gamma * sin
    G[2,4] = gamma * cos

    return G

def UU(X_3N, k, gamma):
    """ Non-dimensional operator G @ [X_dot] """
    N = X_3N.shape[0]//3
    theta_k = X_3N[2*N+k]
    G = GG(theta_k, gamma)
    U = np.zeros((3, 3*N))
    U[:,k] = G[:,0]
    U[:,N+k] = G[:,1]
    U[:,2*N+k] = G[:,2]
    return U

def DD(X_3N, k, i):
    """ Returns non-dimensional (x_i-x_j, y_i-y_j, 1) """
    D = np.zeros((1,3))
    N = X_3N.shape[0]//3
    D[0,0] = X_3N[k] - X_3N[i]
    D[0,1] = X_3N[N+k] - X_3N[N+i]
    D[0,2] = 1
    return D

def AA(X_3N, gamma):
    """ Computes and returns non-dimensional A(X_3N) such that A @ X_3N_dot = A @ [Q @ X_dot] = B. """

    N = X_3N.shape[0]//3
    A = np.zeros((N+2,3*N))
    A[0,0] = 1
    A[1,N] = 1
    A[2,2*N] = 1
    for j in range(1, N):
        for i in range(j, N):
            A[j+2,:] = A[j+2,:] + DD(X_3N, i, j) @ UU(X_3N, i, gamma)
        # A[j+2,:] = A[j+2,:]
    return A

## A dashpots
def ADB(taus_b, N):
    """ Returns the matrix used to model bending dashpots all along the axoneme.
    taus_b is a list of non-dimensional internal bending viscosities. """

    if len(taus_b)==N:
        A_DB = np.diag([0,0] + taus_b)
    else:
        A_DB = np.zeros((N+2, N+2))
    return A_DB

def ADS(N):
    """ Returns the matrix used to model shear dissipation all along the axoneme.
    taus_s is a list of non-dimensional internal shearing viscosity. """
    taus_s = [1]*N
    A_DS = np.tril(np.tile([0,0] + taus_s, (N+2,1)))
    return A_DS

# Nus_b = [1,1,1,1,1]
# A_D = AD(Nus_b)
# print("A_D = ", A_D)
# exit()

# ------------------- #
# -- External flow -- #

def CreateFlowField(A = 0., w0 = 0., psi = 0., T_meas = [], filename = ""):

    """ Creates a non-dimensional flow field and returns a string and an array representing resp. the type and data
    - 1. If a filename is given, import flow field from the file (PIV);
    - 2. If filename = "" and if A = 0 or T_meas = [], there is no flow field: return 0
    - 3. If filemame = "" and if A > 0 and if w0 = 0, returns a constant homogeneous flow
    - 4. If filemame = "" and if A > 0 and if w0 > 0, returns [A*sin(t)] for t in T (homogeneous flow)
    Returned flow field in the 2 last cases is an array of shape (2x|T|)
    psi is the angle between the x-axis and the flow.

    Note: for a non-dimensional flow, A should be chosen in non-dimensional units and t is non-dimensional (i.e., counted in w0 units)

    """

    X_flow_field = 0
    return_string = "NO FLOW"

    if filename == "": # Cases 2,3,4
        if len(T_meas)==0 or A == 0: # Case 2
            return return_string, X_flow_field
        else: # Cases 3,4
            X_flow_field = np.zeros((2,len(T_meas)))
            if w0==0: # Case 3
                X_flow_field[0,:] = A*np.cos(psi)
                X_flow_field[1,:] = A*np.sin(psi)
                return_string = "CONSTANT FLOW: (psi, A) = (" + str(psi) + ", " + str(A) + ")"
                return return_string, X_flow_field

            else: # Case 4
                for t in range(len(T_meas)):
                    X_flow_field[0,t] = A*np.sin(w0*T_meas[t])*np.cos(psi)
                    X_flow_field[1,t] = A*np.sin(w0*T_meas[t])*np.sin(psi)
                return_string = "SINE FLOW: (psi, A, w0) = (" + str(psi) + ", " + str(A) + ", " + str(w0) + ")"

                return return_string, X_flow_field

    else: # Case 1
        # Import field from filename (Change later)   
        return_string = "PIV-IMPORTED from " + filename
        return return_string, X_flow_field 

def Flow(X_3N, X_flow_field = np.array([0]) ):

    """ Computes non-dimensional average flow speed and 1st moment of flow speed on each axoneme segment
    given a flow vector field X_flow_field. There are N segments, numerated from 0 to N-1."""

    N = X_3N.shape[0]//3
    X_dot_flow = np.zeros((4*N,1))

    # No flow is imposed
    if np.shape(X_flow_field)[0] == 1:
        return X_dot_flow

    # A homogeneous flow is imposed
    elif np.shape(X_flow_field)[0] == 2:
        X_dot_flow[:N, 0] = X_flow_field[0]
        X_dot_flow[N:2*N, 0] = X_flow_field[1]
        X_dot_flow[2*N:3*N, 0] = (1 / 2) * X_flow_field[0]
        X_dot_flow[3*N:, 0] = (1 / 2) * X_flow_field[1]
        return X_dot_flow

    # A inhomogeneous flow is imposed, e.g. with PIV experiments
    else: 
        # Add things here later
        return X_dot_flow

def TT_flow(X_dot_flow, k):
    """ Non-dimensional operator [X_dot_flow]_k where X_dot_flow is of shape (4*N x 1). """

    T_flow_k = np.zeros((5,1))
    N = X_dot_flow.shape[0]//4

    T_flow_k[0] = - X_dot_flow[k]
    T_flow_k[1] = - X_dot_flow[N+k]
    T_flow_k[3] = X_dot_flow[2*N+k]
    T_flow_k[4] = X_dot_flow[3*N+k]

    return T_flow_k

# Test
# Delta_S = 0.1
# N = 10
# X = ProximalBend(N)
# X_dot_flow_test = Flow(X, Delta_S,-2)
# print("X_dot_flow_test: ", X_dot_flow_test)
# for k in range(8):
#     T_flow_k = TT_flow(X_dot_flow_test, k, Delta_S)
#     print("T_flow_k for k = ", k, " is: ", T_flow_k)
# exit()

# -- External flow -- #
# ------------------- #

## --- Fluid drag --- ##
########################


#############################
## --- Right-hand side --- ##

def BC(X_3N, n_L=[0,0], m_L=0):
    """Returns non-dimensional B_C representing boundary conditions at the distal end. 
    Zero is default for a free end. n_L and m_L are chosen adimensionally too."""

    N = X_3N.shape[0]//3
    B_C = np.zeros((N+2,1))

    ## point force and point moment at distal end.
    x_L = X_3N[N-1] + np.cos(X_3N[-1])
    y_L = X_3N[2*N-1] + np.sin(X_3N[-1])

    B_C[3:] = (y_L - X_3N[N+1:2*N])*n_L[0] - (x_L - X_3N[1:N])*n_L[1] - m_L

    return B_C

def BB(X_3N):
    """ Returns non-dimensional right-hand side of the differential system for bending elasticity. """

    N = X_3N.shape[0]//3
    B = np.zeros((N+2,1))
    # Boundary conditions at proximal end
    B[0] = 0
    B[1] = 0
    B[2] = 0
    # Bending resistance (constitutive equations)
    B[3:] = B[3:] + (X_3N[2*N+1:]-X_3N[2*N:-1]) # Bending resistance
    
    return B

def BS(X_3N):
    """ Returns non-dimensional right-hand side of the differential system for shear elasticity. """

    N = X_3N.shape[0]//3
    B = np.zeros((N+2,1))
    # Boundary conditions at proximal end
    B[0] = 0
    B[1] = 0
    B[2] = 0
    for i in range(2,N+1):
        B[i+1] = B[i+1] + (np.sum(X_3N[2*N+i-1:]) - (N-i+1)*X_3N[2*N] ) # Sliding resistance
    return B

def BFlow(X_3N, X_dot_flow, gamma):
    """ Returns non-dimensional B_flow representing moments due to background flow. This is similar to computations
    on the left-hand side of the differential equation.
    Importantly, when put on the right-hand side of the equation one should add a minus sign. """

    N = X_3N.shape[0]//3
    B_flow = np.zeros((N+2, 1))

    B_flow[0,0] = 0
    B_flow[1,0] = 0
    B_flow[2,0] = 0

    for j in range(1, N):
        for i in range(j, N):
            theta_i = X_3N[2*N+i]
            B_flow[j+2,0] = B_flow[j+2,0] + DD(X_3N, i, j) @ GG(theta_i, gamma) @ TT_flow(X_dot_flow, i)
        # B_flow[j+2,0] = B_flow[j+2,0]
    return B_flow

# # Unit test
# N = 4
# Delta_S = 1
# X = StraightLine(N)
# X_3N = X3N(X, Delta_S)
# print("X_3N = ", X_3N)
# X_dot_flow_test = Flow(X, Delta_S, -2)
# print("X_dot_flow_test: ", X_dot_flow_test)
# eta = 1
# gamma = 1
# B_Flow = BFlow(X_3N, X_dot_flow_test, eta, gamma, Delta_S)
# print("B_Flow = ", B_Flow)
# exit()

def BF(X_3N, Lambdas):
    """returns B_F representing non-dimensional moments of uniform density forces on each segment. """
    N = len(Lambdas)
    B_F = np.zeros((N+2,1))
    if Lambdas == [0]*N:
        return B_F
    else:
        for j in range(1, N):
            for i in range(j, N):
                Lambda_i = Lambdas[i]
                B_F[j+2,0] = B_F[j+2,0] + Lambda_i[1] * (X_3N[i] - X_3N[j] + 2*np.cos(X_3N[2*N+i])) - Lambda_i[0] * (X_3N[N+i] - X_3N[N+j] + 2*np.sin(X_3N[2*N+i]))
        return B_F

def BM(Zetas):
    """returns B_M representing torques between each segment. """
    N = len(Zetas)
    B_M = np.zeros((N+2,1))
    if Zetas == [0]*N:
        return B_M
    else:
        for j in range(1, N):
            B_M[j+2,0] = np.sum(Zetas[j:])
        return B_M

### Active bending moments

def ActiveBending(X):
    # try with local (one node or more) constant + dissipation ()
    N = X.shape[0]-2
    B_active = np.zeros((N+2,1))
    return B_active

## --- Right-hand side --- ##
#############################


#############################################
## --- Differential system AQX_dot = B --- ##

def f(t, X, Sp4, Beta, taus_b, gamma, n_L=[0,0], m_L=0, Lambdas=0, Zetas=0, InterpFlow = 0):

    """ Returns the non-dimensionalized equation X_dot = f(X; t; parameters). """

    N = X.shape[0]-2

    X_3N_time = time.time()
    # X_3N = X3N(X, Delta_S)
    X_3N = X3N(X)
    X_3N_time = time.time() - X_3N_time
    if X_3N_time>1:
        print("Getting X_3N from X took %s seconds." % (X_3N_time))

    A_time = time.time()
    A = AA(X_3N, gamma)
    A_time = time.time() - A_time
    if A_time>1:
        print("Getting A took %s seconds." % (A_time))

    Q_time = time.time()
    Q = QQ(X_3N)
    Q_time = time.time() - Q_time
    if Q_time>1:
        print("Getting Q took %s seconds." % (Q_time))

    ADB_time = time.time()
    A_DB = ADB(taus_b, N)
    ADB_time = time.time() - ADB_time
    if ADB_time>1:
        print("Getting A_DB took %s seconds." % (ADB_time))

    ADS_time = time.time()
    A_DS = ADS(N)
    ADS_time = time.time() - ADS_time
    if ADS_time>1:
        print("Getting A_DS took %s seconds." % (ADS_time))

    if InterpFlow == 0:
        X_dot_flow = Flow(X_3N)
    else:
        # print("Time t = ", t)
        X_flow = InterpFlow(t)
        X_dot_flow = Flow(X_3N, X_flow)

    B_time = time.time()
    B = BB(X_3N) + BC(X_3N, n_L, m_L) + Beta * BS(X_3N) - BF(X_3N, Lambdas) - BM(Zetas) + ActiveBending(X) - Sp4 * BFlow(X_3N, X_dot_flow, gamma)
    B_time = time.time() - B_time
    if B_time>1:
        print("Getting B took %s seconds." % (B_time))

    X_dot_time = time.time()

    X_dot = (np.linalg.inv(Sp4 * A @ Q - A_DB - Beta * A_DS) @ B).ravel()
    X_dot_time = time.time() - X_dot_time
    if X_dot_time>1:
        print("Inverting to get X_dot took %s seconds." % (X_dot_time))

    # Result
    time_list = [X_3N_time, A_time, ADB_time, ADS_time, Q_time, B_time, X_dot_time]
    time_dict = ["X_3N","A", "A_D", "Q", "B", "X_dot"]
    max_time = np.max(time_list)
    if max_time>1:
        print("The longest computation took %s seconds and was" %max_time , time_dict[time_list.index(max_time)])
    
    return X_dot

## --- Differential system AQX_dot = B --- ##
#############################################

# Define a custom event function to stop based on time
class StopOnTime:
    def __init__(self, max_simulation_time):
        self.start_time = time.time()  # Track the start time
        self.max_simulation_time = max_simulation_time

    def event(self, t, y, *args):
        # Check elapsed time
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.max_simulation_time:
            return 0  # Event triggers when this returns zero
        return 1      # Otherwise, continue integration
    
    def terminate_integration(self, t, y, *args):
        # This ensures we only terminate the integration when crossing zero
        return np.sign(self.event(t, y, *args))

## --- Test --- ##
# def g(x, a, b, c):
#     print(x,a,b,c)
#     return 

######################################################
## --- Solves and saves the differential system --- ##
def Solve(f, taus_b, Beta, gamma, n_L, m_L, A, w0, Sp4, Lambdas, Zetas, X_flow_field_string, T_span, T_eval, T_sim_max, X_flow_field, X_0, method = 'LSODA'):
    """Solves the linear system for a set of parameters and returns the solution. """

    time_limiter = StopOnTime(max_simulation_time=T_sim_max)

    # Creates an interpolation function of the flow field to inject it in the solver
    if X_flow_field_string != "NO FLOW":
        InterpFlow = interpolate.interp1d(np.array(T_eval).reshape(len(T_eval),), X_flow_field, axis=1, fill_value="extrapolate") # Beware of that extrapolation option - might be due to the period being much higher than actual time step
        Args = (Sp4, Beta, taus_b, gamma, n_L, m_L, Lambdas, Zetas, InterpFlow)

    else: # no flow
        Args = (Sp4, Beta, taus_b, gamma, n_L, m_L, Lambdas, Zetas)

    start_time = time.time()
    try:
        sol = solve_ivp(fun = f, t_span = T_span, y0 = X_0, args=Args, t_eval=T_eval, method = method, events=time_limiter.terminate_integration).y
        solving_time = time.time() - start_time

    except ValueError:
        print("ValueError")
        res = False
    except np.linalg.LinAlgError:
        print("LinAlgError")
        res = False

    if sol.t_events[0].size > 0:
        # If an event triggered, i.e., the time limit was exceeded
        res = np.inf
        print("Solving aborted: too long.")
    else:
        res = solving_time, sol
        print("Solving took %s seconds." % solving_time)

    return res

def SolveAndSave(output_folder, N, taus_b, init_conf, Beta, gamma, n_L, m_L, A, w0, Sp4, Lambdas, Zetas, X_flow_field_string, T_span, T_eval, T_sim_max, X_flow_field, X_0, method = 'LSODA'):
    
    """ Solves the linear system for a set of parameters and saves the resulting dynamics in a file. 
    Returns True if the algorithm converges, False otherwise. 
    
    INPUTS
    - output_folder: where the metadata+data file is saved
    - N: number of segments
    - taus_b: bending characteristic time
    - init_conf: initial spatial configuration of the filament (string)
    - Beta: ratio of the shear elasticity over the bending elasticity. 0 means only bending elasticity is present
    - gamma: RFT parameter
    - n_L: point force at s = L
    - m_L: point torque at s = L
    - A: flow amplitude
    - w0: flow frequency
    - Sp4: Sperm number^4, i.e., ratio of fluid viscosity over bending elasticity
    - Lambdas: ad hoc force on filament segments
    - Zetas: ad hoc torque on filament segments
    - X_flow_field_string: flow field metadata
    - T_span: simulation time
    - T_eval: time points to evaluate the dynamical system
    - T_sim_max: maximum simulation time before abort (in seconds).
    - X_flow_field: prescribed flow field
    - X_0: initial position of the filament
    - method: solving method for solve_ivp. Can be any of ["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"]
        Explicit Runge-Kutta methods (‘RK23’, ‘RK45’, ‘DOP853’) should be used for non-stiff problems and implicit methods (‘Radau’, ‘BDF’) for stiff problems [9]. Among Runge-Kutta methods, ‘DOP853’ is recommended for solving with high precision (low values of rtol and atol).

        If not sure, first try to run ‘RK45’. If it makes unusually many iterations, diverges, or fails, your problem is likely to be stiff and you should use ‘Radau’ or ‘BDF’. ‘LSODA’ can also be a good universal choice, but it might be somewhat less convenient to work with as it wraps old Fortran code.
    """

    # print('Solving...')

    ############################################################################
    #### Metadata
    # print("Writing metadata...")

    date = datetime.now().strftime("%Y%m%d-%I%M%S%f")
    metadata_filename = output_folder + "metadata_" + str(date) + ".json"
    data_filename = output_folder + "data_" + str(date) + ".csv"

    solver_values = [output_folder, N, taus_b, str(init_conf), Beta, gamma, n_L, m_L, A, w0, Sp4, Lambdas, Zetas, X_flow_field_string, T_span, T_eval, T_sim_max, X_flow_field, X_0, method]

    solver_keys = ["output_folder", "N", "taus_b", "init_conf", "Beta", "gamma", "n_L", "m_L", "A", "w0", "Sp4", "Lambdas", "Zetas", "X_flow_field_string", "T_span", "T_eval", "T_sim_max", "X_flow_field", "X_0", "method"]
    solver_dict = {f"{solver_keys[k]}": solver_values[k] for k in range(len(solver_values))}

    write_dict_to_json_file(solver_dict, metadata_filename)

    # print("Metadata written.")
    ############################################################################

    ############################################################################
    #### Flow field

    # print("Interpolating flow field...")

    # Creates an interpolation function of the flow field to inject it in the solver
    if X_flow_field_string != "NO FLOW":

        InterpFlow = interpolate.interp1d(np.array(T_eval).reshape(len(T_eval),), X_flow_field, axis=1, fill_value="extrapolate") # Beware of that extrapolation option - might be due to the period being much higher than actual time step
        
        Args = (Sp4, Beta, taus_b, gamma, n_L, m_L, Lambdas, Zetas, InterpFlow)
    else:
        Args = (Sp4, Beta, taus_b, gamma, n_L, m_L, Lambdas, Zetas)

    # print("Flow field interpolated. ")
    ############################################################################

    ############################################################################
    #### Solving and writing solution

    time_limiter = StopOnTime(max_simulation_time=T_sim_max)
    start_time = time.time()
    try:
        sol = solve_ivp(fun = f, t_span = T_span, y0 = X_0, args=Args, t_eval=T_eval, method = method, events=time_limiter.terminate_integration)
        solving_time = time.time() - start_time

        if sol.t_events[0].size > 0:
            print("Solving aborted: too long.")
            solving_time = np.inf
            res = False
        else:
            print("Solving took %s seconds." % solving_time)
            res = True

        array = np.vstack(( np.ones((1, (sol.y).shape[1])) * solving_time, sol.y ))
        write_array_to_csv(array, data_filename)

        print("Solving took %s seconds." % (time.time() - start_time))
        return res

    except ValueError:
        print("ValueError")
        sol = np.array(['ValueError'])
        solving_time = np.inf
        array = np.hstack(( np.ones((1, sol.shape[1])) * solving_time, sol ))        
        write_array_to_csv(array, data_filename)
        res = False
        return res

    except np.linalg.LinAlgError:
        print("LinAlgError")
        sol = np.array(['LinAlgError'])
        solving_time = np.inf
        array = np.hstack(( np.ones((1, sol.shape[1])) * solving_time, sol ))
        write_array_to_csv(array, data_filename)
        res = False
        return res

    except Exception as ex:
        print(ex)
        res = False
        return res

    ############################################################################



def SolveAndSave_callback(result):
    """ Callback function to use pool.apply_async to SolveAndSave. """
    # result += 1
    # global results
    # results.append(result)
    return

if __name__ == "__main__":

    theta = np.linspace(0, 2*np.pi, 1000)
    gamma = np.array([1,2])

    G_max_eigenvalue = np.zeros((theta.shape[0], gamma.shape[0]))
    for k in range(G_max_eigenvalue.shape[0]):
        for l in range(G_max_eigenvalue.shape[1]):
                G = GG(theta[k], gamma[l])
                G_max_eigenvalue[k,l] = np.linalg.norm(G, ord = np.inf)
    G_max_theta = np.max(G_max_eigenvalue, axis = 0)
    print("G_max = ", G_max_theta)