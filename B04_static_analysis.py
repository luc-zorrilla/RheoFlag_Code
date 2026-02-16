""" This script is used to analyze harmonic responses of the model. """

################################################################################
### Libraries

import os
import glob
import time

# Path
from pathlib import Path # To work with relative path
working_path = (Path(__file__).resolve().parent.parent).resolve()
universal_code_path = ((working_path.parent.parent) / 'Miscellaneous' / 'Code').resolve()

import sys
# Plotting functions
sys.path.append(str(universal_code_path.resolve()))
from plotting_functions import * # type: ignore
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.graph_objects as go
pio.templates.default = "figure_template"

# import multiprocessing as mp
from datetime import datetime
from A01_Coarse_grained_axoneme_functions import *
from B01_simulations_analysis import *

import numpy as np
from scipy.signal import find_peaks

################################################################################

if __name__ == '__main__':

    folder_name = Path('..').joinpath('Model').joinpath('Output').resolve()
    folder_name /= "StraightLine_PeriodicFlow"
    folder_name /= "ShearElasticity_Clamped_VaryingShearViscosity"
    folder_name /= "StaticTest"

    id_filenames = [
        "20250424-124112473627_N_10_tau_s_0_taus_b_0_bool_EI_False_Beta_1000.0_gamma_2_A_1.0_w0_0_Sp4_1_k0_10000000000000.0", 
        "20250424-124112516164_N_10_tau_s_0_taus_b_0_bool_EI_False_Beta_1000.0_gamma_2_A_250.0_w0_0_Sp4_1_k0_10000000000000.0",
        "20250424-124112496404_N_10_tau_s_0_taus_b_0_bool_EI_False_Beta_1000.0_gamma_2_A_10.0_w0_0_Sp4_1_k0_10000000000000.0",
        "20250424-124112496404_N_10_tau_s_0_taus_b_0_bool_EI_False_Beta_1000.0_gamma_2_A_100.0_w0_0_Sp4_1_k0_10000000000000.0",
        ]

    # Equilibrium solution - stroboscopic view and analytical solution (N = 35) + kinetic energy
    id_filename = id_filenames[-1]
    metadata_filename = folder_name + 'metadata_' + id_filename +'.json'
    data_filename = folder_name + 'data_' + id_filename + '.csv'
    solver_dict = get_metadata(metadata_filename) 
    output_folder, N, taus_b, tau_s, init_conf, bool_EI, Beta, gamma, n_L, m_L, A, w0, Sp4, k0, Lambdas, Zetas, X_flow_field_string, T_span, T_eval, T_sim_max, T_sim, X_flow_field, X_0, method = list(solver_dict.values())
    X = get_data(data_filename) # s, t
    X_3N_final = X3N(X[:,-1])

    T_eval = np.array(T_eval)
    if (A > 0) & (w0 > 0):
        T_eval_norm = T_eval * w0 / (2*np.pi)
    else:
        T_eval_norm = T_eval
    X_flow = A*np.sin(w0*T_eval)            

    # Stroboscopic view
    n_strobes = 50
    condition = (T_eval_norm >= 0)
    min_index = np.arange(T_eval_norm.shape[0])[condition][0]
    max_index = np.arange(T_eval_norm.shape[0])[condition][-1]
    indices_s = StroboscopicView(T_eval_norm[min_index:max_index], n_strobes=n_strobes)
    c = sample_colorscale('matter_r', np.linspace(0, 1, num = indices_s.shape[0]))[::-1]

    # Equilbrium Profile
    fig = go.Figure()
    for k in range(indices_s.shape[0]):
        fig.add_scatter(x = X3N(X[:,indices_s[k]])[:N, 0]/N, y = X3N(X[:,indices_s[k]])[N:2*N, 0], marker_color = c[k], line_width = 1, showlegend = (k in [0, indices_s.shape[0]-1]), name = [r"$\huge{\boldsymbol{y}_0}$", r"$\huge{\boldsymbol{y}_\text{eq}}$"][k > 0])
    fig.vs_show()
    time.sleep(1)

    # Convergence to equilibrium - kinetic energy decays (log-log plot)
    # Kinetic energy
    K = KineticEnergy(X, N, T_eval) # t
    fig = go.Figure()
    fig.add_scatter(x = T_eval, y = K, line_width = 2)
    for k in range(indices_s.shape[0]):
        fig.add_scatter(x = [T_eval[indices_s[k]]], y = [K[indices_s[k]]], marker_color = c[k], mode = 'markers', marker_size = 6)
    fig.update_yaxes(type = 'log')
    fig.vs_show()