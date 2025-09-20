""" This script aims at visualizing multiple simulation outputs at the same time, 
from the model of viscoelastic filament. """


# from audioop import mul
import multiprocessing
from re import A

from regex import R
from A01_Coarse_grained_axoneme_functions import *
from B01_simulations_analysis import *

import numpy as np
from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime

from plotting_functions import *
pio.templates.default = "figure_template"

#######################################
### ----- Read data from file ----- ###

relaxation_folder = "C:\\Users\\Luc\\Documents\\PhD_Large_files\\RheoFlag\\Model\\Output\\SecondBend_Relaxation\\BendingElasticity_Clamped_VaryingBendingViscosity\\"
relaxation_list = ["20250526-032830406021_N_10_tau_s_0_taus_b_0_bool_EI_True_Beta_0_gamma_2_A_0_w0_0_Sp4_1_k0_10000000000000.0", "20250526-032855177061_N_10_tau_s_0_taus_b_1000.0_bool_EI_True_Beta_0_gamma_2_A_0_w0_0_Sp4_1_k0_10000000000000.0"]
harmonic_folder = "C:\\Users\\Luc\\Documents\\PhD_Large_files\\RheoFlag\\Model\\Output\\StraightLine_PeriodicFlow\\BendingElasticity_Clamped_VaryingBendingViscosity\\VaryingFrequencyAmplitude\\"
# harmonic_list = ["20250425-065432009531_N_10_tau_s_0_taus_b_1.0_bool_EI_True_Beta_0_gamma_2_A_1e-05_w0_1e-08_Sp4_1_k0_10000000000000.0", "20250425-094213855897_N_10_tau_s_0_taus_b_1000.0_bool_EI_True_Beta_0_gamma_2_A_1e-05_w0_1e-08_Sp4_1_k0_10000000000000.0"]
harmonic_list = ["20250425-065622977713_N_10_tau_s_0_taus_b_1.0_bool_EI_True_Beta_0_gamma_2_A_2.0433597178569438e-05_w0_0.001_Sp4_1_k0_10000000000000.0",
"20250425-094106662156_N_10_tau_s_0_taus_b_1000.0_bool_EI_True_Beta_0_gamma_2_A_1e-05_w0_0.001_Sp4_1_k0_10000000000000.0"]

relaxation_range = [0, 0.025]
harmonic_range = [-0.025, 0.025]
range = harmonic_range

folder_name = harmonic_folder
filename_list = harmonic_list
X_list = []
for id_filename in filename_list:
    metadata_filename = folder_name + 'metadata_' + id_filename +'.json'
    data_filename = folder_name + 'data_' + id_filename + '.csv'
    solver_dict = get_metadata(metadata_filename)
    output_folder, N, taus_b, tau_s, init_conf, bool_EI, Beta, gamma, n_L, m_L, A, w0, Sp4, k0, Lambdas, Zetas, X_flow_field_string, T_span, T_eval, T_sim_max, T_sim, X_flow_field, X_0, method = list(solver_dict.values())
    tau_b = taus_b[0]
    X = get_data(data_filename) # s, t
    X_3N_final = X3N(X[:,-1])
    

    keys_toprint = ['N', 'init_conf', 'A', 'w0', 'Sp4', 'k0', 'Beta', 'taus_b', 'tau_s', 'n_L', 'm_L', 'Lambdas', 'Zetas']
    for key in keys_toprint:
        print(key, solver_dict[key])

    T_eval = np.array(T_eval)
    if (A > 0) & (w0 > 0):
        T_eval_norm = T_eval * w0 / (2*np.pi)
    else:
        T_eval_norm = T_eval
    X_flow = A*np.sin(2 * np.pi * T_eval_norm)

    ##################################
    ### ----- Shape analysis ----- ###

    X_list.append(X)

# Animated shape
fig = AnimatedShapes(X_list[0], X_list[1], X_flow, N, w0, Sp4, Beta, tau_b, T_eval)
fig.update_yaxes(range = range)
fig.update_layout(showlegend = False, width = 1000, height = 1000)
fig.vs_show()
