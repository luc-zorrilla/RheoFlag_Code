
from cmath import phase
from socketserver import DatagramRequestHandler
import pandas as pd
import os
import glob

from Coarse_grained_axoneme_functions import *
from Coarse_grained_analysis_functions import *

import numpy as np
from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

from prettytable import PrettyTable

import plotly.express as px

##############################################
##### --- Read datafiles in a folder --- #####

## Specify folder name
folder_path = "C:/Users/Luc/Documents/PhD_Large_files/RheoFlag/Model/Output/renormalized/bending_elasticity_viscosity/periodic_response/"

################### Temp #####################

# creating the parameter table
Param_names = ["datafile", "N", "taus_b", "init_conf", "Beta", "gamma", "n_L", "m_L", "A", "w0", "Sp4", "Lambdas", "Zetas", "X_flow_field_type", "X_flow_field_params", "T_span"] # "T_eval"
Param_table = [Param_names] # list of lists
################## Temp ######################

## Read all datafiles in that folder and store parameters in a table (or list of lists)
os.chdir(folder_path)
dat_files = glob.glob("data*.dat")
for d in dat_files:
    P = ExtractParameters(d)
    # Param_table.add_row([d] + P[:-1]) # -1 --> Putting out T_eval, which is too long here.
    Param_table.append([d] + P[:-1])
# print("Number of data files = ", len(Param_table))
# print("first datafile, parameters = ", Param_table[1])

## Choose which parameters one wants. If None, that parameter is ignored.
search_files = True
# w0_list = [22.570197196339205, 23.644894126454084, 24.770763559917114]
if search_files:
    datafile_list = [None]
    N = 10
    taus_b = None
    init_conf = None
    Beta = 0
    gamma = 2
    n_L = None
    m_L = None
    A = 1e-2
    w0 = 1e0
    Sp4 = 1e-1
    Lambdas = None
    Zetas = None
    X_flow_field_type = "SINE FLOW: (psi, A, w0)"
    X_flow_field_params = None
    T_span = None

    matching_param_table = []
    matching_data = []

    for datafile in datafile_list:

        Wanted_parameters = [datafile, N, taus_b, init_conf, Beta, gamma, n_L, m_L, A, w0, Sp4, Lambdas, Zetas, X_flow_field_type, X_flow_field_params, T_span]
        ## Extract from the table the datafiles corresponding to wanted parameters

        for x in(Param_table[1:]): # From i=1 to avoid field names
            # print("list IsEqual ", list(map(IsEqual, x, Wanted_parameters)))
            # print("count of False in list ", list(map(IsEqual, x, Wanted_parameters)).count(False))
            is_equal = list(map(IsEqual, x, Wanted_parameters)).count(False) == 0 # If number of False is not 0, then is_equal = False
            if is_equal:
                data = ExtractData(x[0])
                if np.count_nonzero(data)>0: # Avoiding any ErrorValue, i.e., no data.
                    matching_param_table.append(x)
                    matching_data.append(data)
    
    # Sort data and parameter table by w0
    # matching_param_table.sort(key = lambda elem: elem[8])
    sorted_indices = sorted(range(len(matching_param_table)), key = lambda k: matching_param_table[k][9])
    # print("sorted indices = ", sorted_indices)
    matching_param_table = [matching_param_table[i] for i in sorted_indices]
    matching_data = [matching_data[i] for i in sorted_indices]
    print("w0 span: ", [matching_param_table[k][9] for k in range(len(matching_param_table))])
    print(len(matching_data), "matching datafiles. ")
    if len(matching_data)<3:
        print([matching_param_table[i][0] for i in range(len(matching_param_table))])
    print("first matching file: ", matching_param_table[0][0])
    print("last matching file: ", matching_param_table[-1][0])
    # print("before last matching file: ", matching_param_table[-2][0])

exit()
##### --- Read datafiles in a folder --- #####
##############################################

##########################################
##### --- Tangent angle analysis --- #####

###################
### --- PCA --- ###

# phase_PCA = np.zeros((len(matching_param_table),))
# w0_array = np.zeros((len(matching_param_table),))
# for k in range(len(matching_param_table)):
#     print("Computing phase for simulation #", k)
#     [N, taus_b, init_conf, Beta, gamma, n_L, m_L, A, w0, Sp4, Lambdas, Zetas, X_flow_field_type, X_flow_field_params, T_span, T_eval], X = ExtractParametersData(matching_param_table[k][0])
#     if k==0:
#         show_PCA = True
#     else:
#         show_PCA = False
#     phase_PCA[k] = analyze_shape(X, X_flow_field_params, T_eval, show_PCA = show_PCA, show_Fourier = False)
#     w0_array[k] = w0
# print("w0_array", w0_array)
# print("PCA_phase", phase_PCA)
# fig_phase_frequency = px.scatter(x = w0_array, y = phase_PCA/np.pi)
# fig_phase_frequency.show()

### --- PCA --- ###
###################

###############################
### --- Spatial Fourier --- ###

phases_Fourier= np.zeros((len(matching_param_table), (N-1)//2 +1))
w0_array = np.zeros((len(matching_param_table),))
for k in range(len(matching_param_table)):
    [N, taus_b, init_conf, Beta, gamma, n_L, m_L, A, w0, Sp4, Lambdas, Zetas, X_flow_field_type, X_flow_field_params, T_span, T_eval], X = ExtractParametersData(matching_param_table[k][0])
    X_flow = A*np.sin(w0*np.array(T_eval))
    # print("[N, taus_b, Beta, gamma, n_L, m_L, A, w0, Sp4, Lambdas, Zetas, X_flow_field_type, X_flow_field_params, T_span] = ", [N, taus_b, Beta, gamma, n_L, m_L, A, w0, Sp4, Lambdas, Zetas, X_flow_field_type, X_flow_field_params, T_span])
    
    T_eval = np.array(T_eval)
    w0_array[k] = w0

    Theta, Theta_0 = Kymograph(X, False)
    # Popping out the first angle, fixed because of clamped BC
    Theta = Theta[:,1:]
    Theta_0 = Theta_0[:,1:]

    Theta_q, spatial_modes = SpatialFourier(np.transpose(Theta), T_eval, w0, False)
    Fourier_phases, polar_coeffs_ellipses, fig_ellipse = SpatialFourier_vs_Flow(Theta_q, spatial_modes, X_flow, w0, Theta_q.shape[0], True)
    # fig_ellipse.show()
    phases_Fourier[k,:] = Fourier_phases.reshape((phases_Fourier.shape[1],))

fig_phase_frequency_Fourier = make_subplots(
    rows=phases_Fourier.shape[1], cols=1, 
    subplot_titles=("Real fourier phase", "Imaginary Fourier phase")
    )
for l in range(phases_Fourier.shape[1]):
    trace = go.Scatter(x=w0_array, y=phases_Fourier[:,l]/np.pi)
    fig_phase_frequency_Fourier.add_trace(trace, row=l+1, col=1)
fig_phase_frequency_Fourier.update_layout(title_text = "Phases")
fig_phase_frequency_Fourier.show()

### --- Spatial Fourier --- ###
###############################



##### --- Tangent angle analysis --- #####
##########################################