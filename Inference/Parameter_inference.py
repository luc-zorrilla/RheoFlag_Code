""" This script aims at inferring viscoelastic parameters from a model of viscoelastic filament, given "truth" data (which in this case is simulation output data). For now only the grid search is implemented. """

import numpy as np
from Coarse_grained_axoneme_functions import *
from Coarse_grained_analysis_functions import *
from scipy.integrate import solve_ivp

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from datetime import datetime

#######################################
### ----- Read data from file ----- ###

folder_name = "C:/Users/Luc/Documents/PhD_Large_files/RheoFlag/Model/Output/renormalized/pure_bending/"
filename = "data_20221124-010008948918.dat"
parameters, X_data = ExtractParametersData(folder_name + filename)

N, taus_b, init_conf, Beta, gamma, n_L, m_L, A, w0, Sp4_data, Lambdas, Zetas, X_flow_field_type, X_flow_field_params, T_span, T_eval = parameters
tau_b = taus_b[0]
print("N = ", N)
print("A = ", A)
print("w0 = ", w0)
print("Bending elasticity / drag timescale: Sp4 = ", Sp4_data)
print("Shear/bending elasticity ratio: Beta = ", Beta)
print("tau_b = ", tau_b)

### ----- Read data from file ----- ###
#######################################

###########################################
###########################################
### ----- Optimization algorithms ----- ###

print("shape of data: ", X_data.shape)

# Known parameters:
print("N = ", N)
print("A = ", A)
print("w0 = ", w0)
print("T_span = ", T_span)
X_flow_field_string, X_flow_field = CreateFlowField(A = X_flow_field_params[1], w0 = X_flow_field_params[2], psi = X_flow_field_params[0], T_meas = T_eval)

#######################
### 1 - Grid search ###

### Choose a norm
def L2_spacetime_norm(x,y):
    """ This norm is the Froebenius norm of vector fields seen as matrices of dimensions Space x Time."""
    return  np.linalg.norm(x-y)

### Choose grid to evaluate in parameter space
log_Sp4_min = -0.01
log_Sp4_max = 0.01
P = 50
Sp4_grid = np.logspace(start = log_Sp4_min, stop = log_Sp4_max, num = P) # Log-grid in base 10!!
error_grid = np.zeros(Sp4_grid.shape)

for Sp4_index in range(Sp4_grid.shape[0]):
    
    X_0 = X_data[:,0]
    Sp4 = Sp4_grid[Sp4_index]
    print("Sp4 = ", Sp4)

    if X_flow_field_type != "NO FLOW":
        InterpFlow = interpolate.interp1d(np.array(T_eval).reshape(len(T_eval),), X_flow_field, axis=1, fill_value="extrapolate") # Beware of that extrapolation option - might be due to the period being much higher than actual time step
        Args = (Sp4, Beta, taus_b, gamma, n_L, m_L, Lambdas, Zetas, InterpFlow)

    else:
        Args = (Sp4, Beta, taus_b, gamma, n_L, m_L, Lambdas, Zetas)  

    sol = solve_ivp(fun = f, t_span = T_span, y0 = X_0, args=Args, t_eval=T_eval, method = 'LSODA').y
    error_grid[Sp4_index] = L2_spacetime_norm(sol, X_data)
    print("Absolute error = ", error_grid[Sp4_index])

fig = go.Figure()
fig.add_scatter(x = Sp4_grid, y = error_grid, mode = "markers", name = "absolute error")
fig.add_scatter(x = Sp4_grid, y = error_grid / L2_spacetime_norm(0, X_data), mode = "markers", name = "relative error")
fig.show()

# refine grid
Sp4_index = np.argmin(error_grid)
Sp4_min = Sp4_grid[np.max((0, Sp4_index-1))]
Sp4_max = Sp4_grid[np.min((Sp4_index+1, Sp4_grid.shape[0]-1))]
Sp4_grid = np.linspace(start = Sp4_min, stop = Sp4_max, num = P)
error_grid = np.zeros(Sp4_grid.shape)

for Sp4_index in range(Sp4_grid.shape[0]):
    
    X_0 = X_data[:,0]
    Sp4 = Sp4_grid[Sp4_index]
    print("Sp4 = ", Sp4)

    if X_flow_field_type != "NO FLOW":
        InterpFlow = interpolate.interp1d(np.array(T_eval).reshape(len(T_eval),), X_flow_field, axis=1, fill_value="extrapolate") # Beware of that extrapolation option - might be due to the period being much higher than actual time step
        Args = (Sp4, Beta, taus_b, gamma, n_L, m_L, Lambdas, Zetas, InterpFlow)

    else:
        Args = (Sp4, Beta, taus_b, gamma, n_L, m_L, Lambdas, Zetas)  

    sol = solve_ivp(fun = f, t_span = T_span, y0 = X_0, args=Args, t_eval=T_eval, method = 'LSODA').y
    error_grid[Sp4_index] = L2_spacetime_norm(sol, X_data)
    print("Absolute error = ", error_grid[Sp4_index])

Sp4_index = np.argmin(error_grid)
Sp4_inferred = Sp4_grid[Sp4_index]
rel_error = error_grid[Sp4_index] / L2_spacetime_norm(X_data, 0)
print("The closest parameters to real parameters, using grid search, are Sp4 = ", Sp4_inferred, "while Sp4_real = ", Sp4_data)
print("There is a relative precision of ", rel_error)

fig = go.Figure()
fig.add_scatter(x = Sp4_grid, y = error_grid, mode = "markers", name = "absolute error")
fig.add_scatter(x = Sp4_grid, y = error_grid / L2_spacetime_norm(0, X_data), mode = "markers", name = "relative error")
fig.show()

### 1 - Grid search: log-scale + lin-scale refinement ###
#######################

########################################
### 2 - Discretized gradient descent ###

### 2 - Discretized gradient descent ###
########################################

### ----- Optimization algorithms ----- ###
###########################################
###########################################