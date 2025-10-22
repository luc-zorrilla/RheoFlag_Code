""" This script ___ """

## Libraries

from C01_Inference_Viscoelastic_Model import *
from misc_func import *

import glob
import pickle
from pathlib import Path
writing_path = (Path('..') / 'Inference' / 'FromSimulationData' / 'BendingElasticity_BendingViscosity_Clamped')
import numpy as np
import pandas as pd

from plotly.subplots import make_subplots
import plotly.graph_objects as go

## Main

bool_test = False
if __name__ == "__main__":

    A_list = []
    w0_list = []
    p_inf_list = []
    IE_list = []
    Hm1_list = []

    filepaths = list(writing_path.glob('**/*.pkl')) # List of path of .pkl files in the writing path
    filenames = [str(filepath.resolve()) for filepath in filepaths] # Convert to strings in the corresponding OS
    print("filenames:", len(filenames))

    for filename in filenames:

        pkl_file = open(filename, 'rb')
        VI_dict = pickle.load(pkl_file)
        pkl_file.close()

        exp_variable_params = VI_dict["exp_variable_params"]
        guess_variable_params = VI_dict["args"]["guess_variable_params"]
        flow_params = VI_dict["flow_params"]
        A = flow_params["A"]
        w0 = flow_params["w0"]
        fixed_params = VI_dict["args"]["fixed_params"]

        p_star = np.array(list(exp_variable_params.values()))
        inferred_variable_params = VI_dict["output"].x
        p_inf = np.array(list(inferred_variable_params.values()))

        IE = L2_relative_error(p_inf, p_star)
        Hm1 = VI_dict['output']['lowest_optimization_result']['hess_inv'].todense()
        if not VI_dict['output']['success']: # If convergence failed, error is infinite
            Hm1 = np.ones_like(Hm1) * np.inf

        A_list.append(A)
        w0_list.append(w0)
        p_inf_list.append(p_inf)
        IE_list.append(IE)
        Hm1_list.append(Hm1)

        ## Inferred parameters with uncertainty
        # fig = go.Figure()
        # fig.add_scatter(x = list(guess_variable_params.keys()), y = list(guess_variable_params.values()), mode = "markers", marker_color = "green", marker_size = 10, name = "p_0", opacity = 0.7)
        # fig.add_scatter(x = list(inferred_variable_params.keys()), y = list(inferred_variable_params.values()), error_y = dict(type = "data", array = np.sqrt(np.diag(Hm1))), mode = "markers", marker_color = "black", name = "p_inf", opacity = 0.7)
        # fig.add_scatter(x = list(exp_variable_params.keys()), y = list(exp_variable_params.values()), mode = "markers", marker_color = "red", marker_size = 10, name = "p_star", opacity = 0.7)
        # fig.show()

    variable_keys = list(exp_variable_params.keys())

    df = pd.DataFrame()
    df["A"] = A_list
    df["w0"] = w0_list
    df["p_inf"] = p_inf_list
    df["IE"] = IE_list
    df["Hm1"] = Hm1_list # Covariance matrix
    df["Sigma"] = df.apply(lambda x: np.sqrt(np.diag(x["Hm1"])), axis = 1)

    print("df[p_inf]", df["p_inf"])
    print("df[IE]", df["IE"])
    print("df[hm1]", df["Hm1"])

    n_vars = p_inf_list[0].shape[0]
    for k_vars in range(n_vars):
        df["p_inf_" + str(k_vars)] = df.apply(lambda x: x['p_inf'][k_vars], axis = 1)
        df["sigma_p_inf_" + str(k_vars)] = df.apply(lambda x: x['Sigma'][k_vars], axis = 1)

    p_combined = [] # Combine parameter estimates (using BLC function)
    for j in range(n_vars):
        Z_vector_list_j = [np.array([df["p_inf"][k][j], df["Sigma"][k][j]]) for k in range(df["p_inf"].shape[0])]
        Z_combined_vector_j = BLC(Z_vector_list_j)
        p_combined.append(Z_combined_vector_j)
    print("p_combined", p_combined)

    # Plot IE heatmap for each (A, w0)-point
    fig = go.Figure(data = go.Heatmap(x = df['A'], y = df['w0'], z = np.log10(df['IE']), colorscale = 'RdPu_r'))
    fig.update_xaxes(title = "w0", type = "linear")
    fig.update_yaxes(title = "A", type = "linear")
    fig.update_layout(title = "Inference Error (for each external parameter)")
    fig.show()

    # Plot inferred params for each (A,w0)-point
    subplot_titles = []
    for k in range(n_vars):
        subplot_titles += variable_keys[k]
        subplot_titles += ["sigma_" + variable_keys[k]]
    fig = make_subplots(rows = n_vars, cols = 2, subplot_titles=subplot_titles)

    for k_vars in range(n_vars):
        hm_p_inf_k_vars = go.Heatmap(x = df['A'], y = df['w0'], z = df['p_inf_' + str(k_vars)], colorscale = 'RdPu_r')
        hm_sigma_p_inf_k_vars = go.Heatmap(x = df['A'], y = df['w0'], z = np.log10(df['sigma_p_inf_' + str(k_vars)]), colorscale = 'RdPu_r')
        fig.add_trace(hm_p_inf_k_vars, row = 1 + k_vars, col = 1)
        fig.add_trace(hm_sigma_p_inf_k_vars, row = 1 + k_vars, col = 2)
        
    fig.update_xaxes(title = "w0", type = "log")
    fig.update_yaxes(title = "A", type = "log")
    fig.update_layout(title = "Inferred parameters (for each external parameter)")
    fig.show()

    # Plot inferred params for all (A,w0)-point combined (only works for 1 or 2 variables) --> to generalize to N params, one needs to plot individual histograms
    if n_vars == 1:
        nbins = 20
        fig = go.Figure(go.Histogram(
            x=df['p_inf_0'],
            nbinsx = nbins))
        fig.update_layout(title = "Histogram of inferred parameters (all external parameters combined)")
        fig.show()        
    elif n_vars == 2:
        nbinsx = 100
        nbinsx = 50
        fig = go.Figure(go.Histogram2d(
            x=df['p_inf_0'],
            y=df['p_inf_1'], 
            nbinsx=100, nbinsy=50))
        fig.update_layout(title = "Histogram of inferred parameters (all external parameters combined)")
        fig.show()

    # Inference error for all (A, w0)-point combined, for each set of internal parameters
    # To be filled


