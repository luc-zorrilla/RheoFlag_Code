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
import csv

#############################
### ----- Functions ----- ###
#############################

import webbrowser
# Set default web browser for webbrowser as VSCode (can also be done manually)
VS_path = "C:\\Users\\Luc\\AppData\\Local\\Programs\\Microsoft VS Code\\Code.exe"
webbrowser.register('VS', None, webbrowser.BackgroundBrowser(VS_path))
web = webbrowser.get('VS')
# This scripts adds a method to go.Figure class so that one can plot figures in html format inside VS code.
def vs_show(self):
    temp_dir = "C:\\Users\\Luc\\Documents\\MEGASync\\PhD\\RheoFlag\\Results\\Temp\\"
    temp_file_number = round(datetime.now().timestamp())
    save_url = temp_dir + "temp_" + str(temp_file_number) + "_.html"

    self.write_html(save_url, include_mathjax = 'cdn')
    web.open(save_url)

    return save_url
go.Figure.vs_show = vs_show


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
    - array: NumPy array of any shape (can contain strings or numbers)
    - filename: the name of the CSV file (without extension)
    """
    array = np.array(array)  # Ensure the input is a NumPy array
    shape = array.shape

    # Open the file in write mode (with .csv extension)
    with open(f'{filename}', 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Write the shape in the first line
        writer.writerow(shape)
        
        # Check if the array contains strings
        if np.issubdtype(array.dtype, np.number):
            # Use np.savetxt for numerical arrays
            np.savetxt(f, array.flatten(), delimiter=',')
        else:
            # Use csv.writer for arrays containing strings
            for item in array.flatten():
                writer.writerow([item])

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
        reader = csv.reader(f)
        
        # Read the first line which contains the shape
        shape_line = next(reader)
        shape = tuple(map(int, shape_line))
        
        # Read the flattened array data (as strings initially)
        flat_data = [row[0] for row in reader]

    # Check if the data is numeric or not
    try:
        # Attempt to convert the data to floats
        flat_array = np.array(flat_data, dtype=float)
        # Reshape it back to its original shape if successful
        array = flat_array.reshape(shape)
    except ValueError:
        # If the data contains strings, keep it as a string array
        array = np.array(flat_data, dtype=object).reshape(shape)
    
    return array

##############################

################################################################################
## --- Reading metadata and data

def get_metadata(metadata_filename):

    solver_dict = read_dict_from_json_file(metadata_filename) # contains '.json' in the name
    return solver_dict

def get_data(data_filename):

    sol = read_array_from_csv(data_filename) # contains '.csv' in the name

    return sol
################################################################################

################################
## --- Initial conditions --- ##

def StraightLine(N):
    """ A straight line """
    X_0 = np.zeros(N+3, dtype = np.double)
    return X_0

def ProximalBend(N):
    """ A proximal bend (of the first segment) """
    X_0 = np.zeros(N+3, np.double)
    X_0[2] = np.pi/4
    return X_0

def SmoothCurve(N):
    """ A curve with constant curvature, so that the total shear angle is pi/2 """
    X_0 = np.zeros(N+3, np.double)
    X_0[2:-1] = np.arange(1,N+1) * (np.pi/4) / N
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

    A[0,0] = 1 # 1*x0_dot = b_0
    A[1,N] = 1 # 1*y0_dot = b_1
    A[2,2*N] = 1 # 1*theta_0_dot = b_2
    for j in range(1, N):
        for i in range(j, N):
            A[j+2,:] = A[j+2,:] + DD(X_3N, i, j) @ UU(X_3N, i, gamma)
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

def BC_L(X_3N, n_L=[0,0], m_L=0):
    """Returns non-dimensional B_C representing boundary conditions at the distal end. 
    Zero is default for a free end. n_L and m_L are chosen adimensionally."""

    N = X_3N.shape[0]//3
    B_C = np.zeros((N+2,1))

    ## point force and point moment at distal end.
    x_L = X_3N[N-1] + np.cos(X_3N[-1])
    y_L = X_3N[2*N-1] + np.sin(X_3N[-1])
    
    B_C[0] = - n_L[0]
    B_C[1] = - n_L[1]
    B_C[2:] = (y_L - X_3N[N:2*N])*n_L[0] - (x_L - X_3N[:N])*n_L[1] - m_L

    return B_C

def BH(X_3N, k0):
    """ Returns non-dimensional right-hand side of the differential system for boundary conditions
    at s = 0 (proximal end). The boundary condition  """

    B = np.zeros((N+2,1))
    if k0 == np.inf:
        return B_H
    else:
        B_H[0] = 0 # force equation (here on x axis) is not affected by elasticity
        B_H[1] = 0 # force equation (here on y axis) is not affected by elasticity   
        B_H[2] = -k0*(X_3N[2*N])
        return B_H

def BB(X_3N):
    """ Returns non-dimensional right-hand side of the differential system for bending elasticity. """

    N = X_3N.shape[0]//3
    B = np.zeros((N+2,1))

    B[0] = 0 # force equation (here on x axis) is not affected by elasticity
    B[1] = 0 # force equation (here on y axis) is not affected by elasticity    
    B[2] = 0 # total torque equation does not depend on bending resistance

    # Bending resistance (constitutive equations)
    B[3:] = (X_3N[2*N+1:]-X_3N[2*N:-1]) # Bending resistance
    
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
    B = BB(X_3N) + BC_L(X_3N, n_L, m_L) + Beta * BS(X_3N) - BF(X_3N, Lambdas) - BM(Zetas) + ActiveBending(X) - Sp4 * BFlow(X_3N, X_dot_flow, gamma)
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

    
def g(t, X_tilde, Sp4, k0, Beta, taus_b, gamma, n_L=[0,0], m_L=0, Lambdas=0, Zetas=0, InterpFlow = 0):

    """ Returns the non-dimensionalized equation X_tilde_dot = g(X_tilde; t; parameters). 
    The difference with f(t,X) is that X is extended to add theta_0_dot, giving X_tilde. 
    Since a second order equation in time is perscribed at the base, it can be turned 
    into a first order equation and added to the matricial system.
    """

    # Apply the spring equation first
    # theta_0_dot_dot = -k0*X_tilde[2]
    # theta_0_dot = -X_tilde[2] * np.sin(np.sqrt(k0)*t)
    # theta_0 = X_tilde[2] * np.cos(np.sqrt(k0)*t)
    # n_0 = 0 # No displacement at the base
    # m_0 = -k0*X_tilde[2] # Rotation at the base is allowed

    ##################################################################
    ###### Solve the linear system with infinite basal stiffness #####
    X = X_tilde[:-1]
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
    B = BB(X_3N) + BC_L(X_3N, n_L, m_L) + Beta * BS(X_3N) - BF(X_3N, Lambdas) - BM(Zetas) + ActiveBending(X) - Sp4 * BFlow(X_3N, X_dot_flow, gamma)
    B_time = time.time() - B_time
    if B_time>1:
        print("Getting B took %s seconds." % (B_time))

    X_dot_time = time.time()

    X_dot = (np.linalg.inv(Sp4 * A @ Q - A_DB - Beta * A_DS) @ B).ravel()
    X_dot_time = time.time() - X_dot_time
    if X_dot_time>1:
        print("Inverting to get X_dot took %s seconds." % (X_dot_time))
    ##################################################################
    
    # Extend the linear system to the basal hinge conditions
    X_dot[2] = X_tilde[-1] # \dot(theta_0) = theta_0_dot
    # theta_0_dot_dot = -k0*X[2] + 0
    X_tilde_dot = np.hstack((X_dot, [theta_0_dot_dot]))

    # Result
    time_list = [X_3N_time, A_time, ADB_time, ADS_time, Q_time, B_time, X_dot_time]
    time_dict = ["X_3N","A", "A_D", "Q", "B", "X_dot"]
    max_time = np.max(time_list)
    if max_time>1:
        print("The longest computation took %s seconds and was" %max_time , time_dict[time_list.index(max_time)])
    
    return X_tilde_dot


## --- Differential system AQX_dot = B --- ##
#############################################

# Class to track total simulation time and trigger an event when time limit is exceeded
class StopOnTime:
    def __init__(self, max_simulation_time):
        self.max_simulation_time = max_simulation_time
        self.start_time = time.time()
    
    # This event function will check the total simulation time elapsed and stop if necessary
    def event(self, t, y, *args):
        elapsed_time = time.time() - self.start_time
        # Return 0 to trigger the event when time exceeds the limit
        return self.max_simulation_time - elapsed_time

    # Ensure the event is terminal (integration stops when the event triggers)
    def terminate_integration(self, t, y, *args):
        return self.event(t, y, *args)
    
    # Set event properties
    terminate_integration.terminal = True
    terminate_integration.direction = 0

## --- Test --- ##
# def g(x, a, b, c):
#     print(x,a,b,c)
#     return 

def SolveAndSave(output_folder, N, taus_b, init_conf, Beta, gamma, n_L, m_L, A, w0, Sp4, k0, Lambdas, Zetas, X_flow_field_string, T_span, T_eval, T_sim_max, X_flow_field, X_0, method = 'LSODA'):
    
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
    - k0: elasticity at the base (s = 0)
    - Lambdas: ad hoc force on filament segments
    - Zetas: ad hoc torque on filament segments
    - X_flow_field_string: flow field metadata
    - T_span: simulation time
    - T_eval: time points to evaluate the dynamical system
    - T_sim_max: maximum simulation time before abort (in seconds).
    - X_flow_field: prescribed flow field
    - X_0: initial position of the filament (now including theta_0_dot)
    - method: solving method for solve_ivp. Can be any of ["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"]
        Explicit Runge-Kutta methods (‘RK23’, ‘RK45’, ‘DOP853’) should be used for non-stiff problems and implicit methods (‘Radau’, ‘BDF’) for stiff problems [9]. Among Runge-Kutta methods, ‘DOP853’ is recommended for solving with high precision (low values of rtol and atol).

        If not sure, first try to run ‘RK45’. If it makes unusually many iterations, diverges, or fails, your problem is likely to be stiff and you should use ‘Radau’ or ‘BDF’. ‘LSODA’ can also be a good universal choice, but it might be somewhat less convenient to work with as it wraps old Fortran code.
    """

    # print('Solving...')

    ############################################################################
    #### Prepare metadata and data files
    date = datetime.now().strftime("%Y%m%d-%I%M%S%f")
    metadata_filename = output_folder + "metadata_" + str(date) + ".json"
    data_filename = output_folder + "data_" + str(date) + ".csv"
    ############################################################################

    ############################################################################
    #### Flow field

    # print("Interpolating flow field...")

    # Creates an interpolation function of the flow field to inject it in the solver
    if X_flow_field_string != "NO FLOW":

        InterpFlow = interpolate.interp1d(np.array(T_eval).reshape(len(T_eval),), X_flow_field, axis=1, fill_value="extrapolate") # Beware of that extrapolation option - might be due to the period being much higher than actual time step
        
        Args = (Sp4, k0, Beta, taus_b, gamma, n_L, m_L, Lambdas, Zetas, InterpFlow)
    else:
        Args = (Sp4, k0, Beta, taus_b, gamma, n_L, m_L, Lambdas, Zetas)

    # print("Flow field interpolated. ")
    ############################################################################

    ############################################################################
    #### Solving and writing solution and metadata

    try:
        time_limiter = StopOnTime(max_simulation_time=T_sim_max)
        sol = solve_ivp(fun = g, t_span = T_span, y0 = X_0, args=Args, t_eval=T_eval, method = method, events=time_limiter.terminate_integration)
        T_sim = time.time() - time_limiter.start_time

        if sol.t_events[0].size > 0:
            T_sim = np.inf
            mistake = np.array(["Solving aborted: too long."])
            print(mistake)
            write_array_to_csv(mistake, data_filename)
            res = False
        else:
            print("Solving took %s seconds." % T_sim)
            write_array_to_csv(sol.y, data_filename)
            res = True

    except BaseException as ex:
        T_sim = np.inf
        mistake = np.array([str(ex)])
        print(mistake)
        write_array_to_csv(mistake, data_filename)
        res = False
    
    ############################################################################
    # Write metadata
    # print("Writing metadata...")
    solver_values = [output_folder, N, taus_b, str(init_conf), Beta, gamma, n_L, m_L, A, w0, Sp4, k0, Lambdas, Zetas, X_flow_field_string, T_span, T_eval, T_sim_max, T_sim, X_flow_field, X_0, method]
    solver_keys = ["output_folder", "N", "taus_b", "init_conf", "Beta", "gamma", "n_L", "m_L", "A", "w0", "Sp4", "k0", "Lambdas", "Zetas", "X_flow_field_string", "T_span", "T_eval", "T_sim_max", "T_sim", "X_flow_field", "X_0", "method"]
    solver_dict = {f"{solver_keys[k]}": solver_values[k] for k in range(len(solver_values))}
    write_dict_to_json_file(solver_dict, metadata_filename)
    # print("Metadata written.")
    ############################################################################
    
    return res

def SolveAndSave_callback(result):
    """ Callback function to use pool.apply_async to SolveAndSave. """    
    # global results
    # results += 1
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