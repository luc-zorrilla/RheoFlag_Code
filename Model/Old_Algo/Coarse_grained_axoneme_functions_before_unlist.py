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


#############################
### ----- Functions ----- ###
#############################

################################
## --- Initial conditions --- ##
def StraightLine(N):
    X_0 = np.zeros(N+2, dtype = np.double)
    return X_0

def ProximalBend(N):
    X_0 = np.zeros(N+2, np.double)
    X_0[2] = np.pi/4
    return X_0

def SmoothCurve(N):
    X_0 = np.zeros(N+2, np.double)
    X_0[2] = np.pi/2
    X_0[3:] = -np.pi/20
    return X_0

## --- Initial conditions --- ##
################################


#################################
## --- Parameter functions --- ##

# def PP(N):
#     """ Transfert matrix from X_3N to X_Np2. Not in use right now. """

#     P = np.zeros((N+2, 3*N))
#     P[0,0] = 1
#     P[1, N] = 1
#     P[2, 2*N+1] = 1
#     for i in range(3,N+2):
#         P[i,i]=1
#         P[i,i-1]=-1
#     return P

def Theta(X, k):
    """ Returns Theta_k from X_Np2. """

    theta_k = np.sum(X[2:k+3])
    return theta_k

def X2(X, i, Delta_S):
    """ Returns position vector X_i from X_Np2. """

    X_2 = np.zeros(2)
    # print("X[0]", X[0])
    # print("X_2[0]", X_2[0])
    X_2[0] = X[0]
    X_2[1] = X[1]
    for k in range(i):
        theta_k = Theta(X, k)
        X_2[0] += Delta_S*np.cos(theta_k)
        X_2[1] += Delta_S*np.sin(theta_k)
    return X_2

def X3N(X, Delta_S):
    N = X.shape[0]-2
    X_3N = np.zeros((3*N,1), dtype=np.double)
    for i in range(N):
        X_2 = X2(X, i, Delta_S)
        theta_i = Theta(X, i)
        X_3N[i] = X_2[0]
        X_3N[N+i] = X_2[1]
        X_3N[2*N+i] = theta_i

    return X_3N

def QQ(X_3N, Delta_S):
    """ Transfert matrix from X_Np2_dot to X_3N_dot. 
    It is shape-dependent while the opposite transfert matrix is not. """

    N = X_3N.shape[0]//3
    Q = np.zeros((3*N, N+2))
    Q[:N,0] = np.ones((N,1)).ravel()
    Q[N:2*N,1] = np.ones((N,1)).ravel()

    Q_x = np.zeros((N,N))
    Q_y = np.zeros((N,N))
    Q_theta = np.eye(N)
    
    for i in range(1,N):
        theta_im1 = X_3N[2*N+i-1]
        Q_x[i,:i] = Q_x[i-1,:i] - Delta_S*np.sin(theta_im1)*np.ones((1,i))
        Q_y[i,:i] = Q_y[i-1,:i] + Delta_S*np.cos(theta_im1)*np.ones((1,i))
        for j in range(i):
            Q_theta[i,j] = 1

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
    
def UU(X_3N, k, gamma, Delta_S):
    """ Operator G @ [X_dot] """
    N = X_3N.shape[0]//3
    theta_k = X_3N[2*N+k]
    G = GG(theta_k, gamma)
    U = np.zeros((3, 3*N))
    U[:,k] = G[:,0]
    U[:,N+k] = G[:,1]
    U[:,2*N+k] = Delta_S*G[:,2]
    return U

def DD(X_3N, k, i, Delta_S):
    """ Returns (x_i-x_j, y_i-y_j, Delta_S) """
    D = np.zeros((1,3))
    N = X_3N.shape[0]//3
    D[0,0] = X_3N[k] - X_3N[i]
    D[0,1] = X_3N[N+k] - X_3N[N+i]
    D[0,2] = Delta_S
    return D

## A
def AA(X_3N, eta, gamma, Delta_S):
    """ Computes and returns A(X_3N) such that A @ X_Np2_dot = A @ [Q @ X_3N_dot] = B. """
    N = X_3N.shape[0]//3
    A = np.zeros((N+2,3*N))
    A[0,0] = 1
    A[1,N] = 1
    A[2,2*N] = 1
    for j in range(1, N):
        for i in range(j, N):
            A[j+2,:] = A[j+2,:] + DD(X_3N, i, j, Delta_S) @ UU(X_3N, i, gamma, Delta_S)
        A[j+2,:] = A[j+2,:] * eta * Delta_S
    return A

## A dashpot
def AD(Nus, N):
    """ Returns the matrix used to model dashpots all along the axoneme.
    Nus is a list of internal viscosities. """

    if len(Nus)==N:
        A_D = np.diag([0,0] + Nus)
    else:
        A_D = np.zeros((N+2, N+2))
    return A_D

# Nus = [1,1,1,1,1]
# A_D = AD(Nus)
# print("A_D = ", A_D)
# exit()

# ------------------- #
# -- External flow -- #

def Flow(X, Delta_S, X_flow_field = 0):

    """ Computes average flow speed and 1st moment of flow speed on each axoneme segment
    given a flow vector field X_flow_field. There are N segments, numerated from 0 to N-1."""

    N = X.shape[0]-2
    X_dot_flow = np.zeros((4*N,1))

    # No flow is imposed
    if np.shape(X_flow_field)[0] == 1:
        return X_dot_flow

    # A homogeneous flow is imposed
    elif np.shape(X_flow_field)[0] == 2:
        X_dot_flow[:N, 0] = X_flow_field[0]
        X_dot_flow[N:2*N, 0] = X_flow_field[1]
        X_dot_flow[2*N:3*N, 0] = (1 / 2) * Delta_S * X_flow_field[0]
        X_dot_flow[3*N:, 0] = (1 / 2) * Delta_S * X_flow_field[1]
        return X_dot_flow

    # A inhomogeneous flow is imposed, e.g. with PIV experiments
    else: 
        # Add things here later
        return X_dot_flow

def TT_flow(X_dot_flow, k, Delta_S):
    """ Operator [X_dot_flow]_k where X_dot_flow is of shape (4*N x 1). """

    T_flow_k = np.zeros((5,1))
    N = X_dot_flow.shape[0]//4

    T_flow_k[0] = - X_dot_flow[k]
    T_flow_k[1] = - X_dot_flow[N+k]
    T_flow_k[3] = X_dot_flow[2*N+k] / Delta_S
    T_flow_k[4] = X_dot_flow[3*N+k] / Delta_S

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

def BC(X_3N, Delta_S, n_L=[0,0], m_L=0):
    """Returns B_C representing boundary conditions at the distal end. 
    Zero is default for a free end."""

    N = X_3N.shape[0]//3
    B_C = np.zeros((N+2,1))

    ## point force and point moment at distal end.
    x_L = X_3N[N-1] + Delta_S*np.cos(X_3N[-1])
    y_L = X_3N[2*N-1] + Delta_S*np.sin(X_3N[-1])

    B_C[3:] = (y_L - X_3N[N+1:2*N])*n_L[0] - (x_L - X_3N[1:N])*n_L[1] - m_L

    return B_C

def BB(X_3N, K_b, Delta_S):
    """ Returns right-hand side of the differential system for constitutive equations.
    It includes bending elasticity. """

    N = X_3N.shape[0]//3
    B = np.zeros((N+2,1))
    # Boundary conditions at proximal end
    B[0] = 0
    B[1] = 0
    B[2] = 0
    # Bending and sliding resistance (constitutive equations)

    if K_b!=0:
        B[3:] = B[3:] + K_b*(X_3N[2*N+1:]-X_3N[2*N:-1]) # Bending resistance

    return B

def BS(X_3N, K_s, Delta_S):
    """ Returns right-hand side of the differential system for constitutive equations.
    It includes shear elasticity. """

    N = X_3N.shape[0]//3
    B = np.zeros((N+2,1))
    # Boundary conditions at proximal end
    B[0] = 0
    B[1] = 0
    B[2] = 0
    if K_s!=0:
        for i in range(2,N+1):
            B[i+1] = B[i+1] + K_s*Delta_S*( np.sum(X_3N[2*N+i-1:]) - (N-i+1)*X_3N[2*N] ) # Sliding resistance
    return B

def BFlow(X_3N, X_dot_flow, eta, gamma, Delta_S):
    """ Returns B_flow representing moments due to background flow. This is similar to computations
    on the left side of the differential equation and therefore could be optimized.
    Importantly, when put on the right-hand side of the equation one should add a minus sign. """

    N = X_3N.shape[0]//3
    B_flow = np.zeros((N+2, 1))

    B_flow[0,0] = 0
    B_flow[1,0] = 0
    B_flow[2,0] = 0

    for j in range(1, N):
        for i in range(j, N):
            theta_i = X_3N[2*N+i]
            B_flow[j+2,0] = B_flow[j+2,0] + DD(X_3N, i, j, Delta_S) @ GG(theta_i, gamma) @ TT_flow(X_dot_flow, i, Delta_S)
        B_flow[j+2,0] = B_flow[j+2,0] * eta * Delta_S
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


def BF(X_3N, Delta_S, Lambdas):
    """returns B_F representing moments of uniform density forces on each segment. """
    N = len(Lambdas)
    B_F = np.zeros((N+2,1))
    if Lambdas == [0]*N:
        return B_F
    else:
        for j in range(1, N):
            for i in range(j, N):
                Lambda_i = Lambdas[i]
                B_F[j+2,0] = B_F[j+2,0] + Lambda_i[1]*Delta_S * (X_3N[i] - X_3N[j] + Delta_S/2*np.cos(X_3N[2*N+i])) - Lambda_i[0]*Delta_S * (X_3N[N+i] - X_3N[N+j] + Delta_S/2*np.sin(X_3N[2*N+i]))
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


def f(t, X, eta, gamma, K_s, K_b, Delta_S, Nus = [], n_L=[0,0], m_L=0, Lambdas=0, Zetas=0, InterpFlow = 0):

    N = X.shape[0]-2

    X_3N_time = time.time()
    X_3N = X3N(X, Delta_S)
    X_3N_time = time.time() - X_3N_time
    if X_3N_time>1:
        print("Getting X_3N from X took %s seconds." % (X_3N_time))

    A_time = time.time()
    A = AA(X_3N, eta, gamma, Delta_S)
    A_time = time.time() - A_time
    if A_time>1:
        print("Getting A took %s seconds." % (A_time))

    Q_time = time.time()
    Q = QQ(X_3N, Delta_S)
    Q_time = time.time() - Q_time
    if Q_time>1:
        print("Getting Q took %s seconds." % (Q_time))

    AD_time = time.time()
    A_D = AD(Nus, N)
    AD_time = time.time() - AD_time
    if AD_time>1:
        print("Getting A_D took %s seconds." % (AD_time))

    if InterpFlow == 0:
        X_flow = -1
    else:
        X_flow = InterpFlow(t)
    X_dot_flow = Flow(X_3N, Delta_S, X_flow) # Updates with X_3N

    B_time = time.time()
    B = BC(X_3N, Delta_S, n_L, m_L) + BB(X_3N, K_b, Delta_S) + BS(X_3N, K_s, Delta_S) - BFlow(X_3N, X_dot_flow, eta, gamma, Delta_S) - BF(X_3N, Delta_S, Lambdas) - BM(Zetas) + ActiveBending(X)
    B_time = time.time() - B_time
    if B_time>1:
        print("Getting B took %s seconds." % (B_time))

    X_dot_time = time.time()
    X_dot = (np.linalg.inv(A @ Q - A_D) @ B).ravel()
    X_dot_time = time.time() - X_dot_time
    if X_dot_time>1:
        print("Inverting to get X_dot took %s seconds." % (X_dot_time))

    # Result
    time_list = [X_3N_time, A_time, AD_time, Q_time, B_time, X_dot_time]
    time_dict = ["X_3N","A", "A_D", "Q", "B", "X_dot"]
    max_time = np.max(time_list)
    if max_time>1:
        print("The longest computation took %s seconds and was" %max_time , time_dict[time_list.index(max_time)])
    
    return X_dot

## --- Differential system AQX_dot = B --- ##
#############################################



