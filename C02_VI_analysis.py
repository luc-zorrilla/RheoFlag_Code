""" This script ___ """

## Libraries

from C01_Inference_Viscoelastic_Model import *
from misc_func import *

import glob
import pickle
from pathlib import Path
writing_path = (Path(__file__).resolve().parent.parent / 'Inference' / 'FromSimulationData' / 'QuarterPeriod' / 'BendingShearElasticity_NoViscosity_Clamped')
import numpy as np
import pandas as pd

from plotly.subplots import make_subplots
import plotly.graph_objects as go

## Main

bool_test = False
if __name__ == "__main__":

    A_list = []
    w0_list = []
    p_star_list = []
    p_inf_list = []
    guess_list = []
    IE_list = []
    Hm1_list = []
    X_local_list = []
    F_local_list = []
    X_global_list = []
    F_global_list = []
    accept_global_list = []
    ret_list = []

    filepaths = list(writing_path.glob('*.pkl')) # List of path of .pkl files in the writing path
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
        ret = VI_dict["output"][0]
        inferred_variable_params = ret.x
        p_inf = np.array(list(inferred_variable_params.values()))
        guess = np.array(list(guess_variable_params.values()))

        X_local, F_local, X_global, F_global, accept_global = VI_dict["output"][1:]
        for l in range(len(VI_dict["output"][1:])):
            V = [X_local, F_local, X_global, F_global, accept_global][l] # This will need to be changed to take into account the fact that X_local is a list of arrays. The easiest would be to concatenate it and now where it should be divided, or to turn it into a list of arrays for real
            if l in [2,3]:
                V = np.array(V).squeeze()
            V_list = [X_local_list, F_local_list, X_global_list, F_global_list, accept_global_list][l]
            V_list.append(V)

        IE = L2_relative_error(p_inf, p_star)
        Hm1 = ret['lowest_optimization_result']['hess_inv'].todense()
        if not ret['success']: # If convergence failed, error is infinite
            Hm1 = np.ones_like(Hm1) * np.inf

        A_list.append(A)
        w0_list.append(w0)
        p_star_list.append(p_star)
        p_inf_list.append(p_inf)
        guess_list.append(guess)
        IE_list.append(IE)
        Hm1_list.append(Hm1)
        ret_list.append(ret)

        ## Inferred parameters with uncertainty
        # fig = go.Figure()
        # fig.add_scatter(x = list(guess_variable_params.keys()), y = list(guess_variable_params.values()), mode = "markers", marker_color = "green", marker_size = 10, name = "p_0", opacity = 0.7)
        # fig.add_scatter(x = list(inferred_variable_params.keys()), y = list(inferred_variable_params.values()), error_y = dict(type = "data", array = np.sqrt(np.diag(Hm1))), mode = "markers", marker_color = "black", name = "p_inf", opacity = 0.7)
        # fig.add_scatter(x = list(exp_variable_params.keys()), y = list(exp_variable_params.values()), mode = "markers", marker_color = "red", marker_size = 10, name = "p_star", opacity = 0.7)
        # fig.show()

    variable_keys = list(exp_variable_params.keys())

    # Make dataframe
    df = pd.DataFrame()

    df["A"] = A_list
    df["w0"] = w0_list
    df["p_star"] = p_star_list
    df["p_inf"] = p_inf_list
    df["guess"] = guess_list
    df["IE"] = IE_list
    df["Hm1"] = Hm1_list # Covariance matrix
    df["Sigma"] = df.apply(lambda x: np.sqrt(np.diag(x["Hm1"])), axis = 1)
    for key in ["X_local", "F_local", "X_global", "F_global", "accept_global"]:
        df[key] = eval(key + "_list")
    df["ret"] = ret_list

    n_vars = p_inf_list[0].shape[0]
    for k_vars in range(n_vars):
        df["p_inf_" + str(k_vars)] = df.apply(lambda x: x['p_inf'][k_vars], axis = 1)
        df["sigma_p_inf_" + str(k_vars)] = df.apply(lambda x: x['Sigma'][k_vars], axis = 1)

    # Select files

    ## Select for a specific guess
    # Sp4_guess = 10
    # Beta_guess = 1e-3
    # df = df[df['guess'] == Sp4_guess].reset_index()

    ## Select for a specific exp filament
    Sp4_exp = 1
    Beta_exp = 1e0
    tau_b_exp = 1e0
    target = np.array([Sp4_exp, Beta_exp]) # np.array([Sp4_exp, tau_b_exp]) # 
    df2 = df[df['p_star'].apply(lambda x: np.array_equal(x, target))].reset_index(drop=True)
    print(df2)

    # Select for specific external parameters
    df_Aw0 = df2[(df2['A'] == 1e-8) & (df2['w0'] == 1e5)].reset_index()

    # Combine inferred parameters

    p_combined = [] # Combine parameter estimates (using BLC function)
    p_mean = []
    for j in range(n_vars):
        Z_vector_list_j = [np.array([df2["p_inf"][k][j], df2["Sigma"][k][j]]) for k in range(df2["p_inf"].shape[0])]
        Z_combined_vector_j = BLC(Z_vector_list_j)
        Z_mean_vector_j = np.array([np.nanmean(np.array(Z_vector_list_j), where = np.array(Z_vector_list_j) < np.inf, axis = 0)[0], (np.nanstd(np.array(Z_vector_list_j), where = np.array(Z_vector_list_j) < np.inf, axis = 0, ddof = 1) / np.sqrt(len(Z_vector_list_j)))[0]])
        p_combined.append(Z_combined_vector_j)
        p_mean.append(Z_mean_vector_j)
    print("p_combined", p_combined)
    print("p_mean", p_mean)

    # Plots

    # Plot X, F evolution for each A, w0
    ## Evolution of the global optimizer (showing successive minima)
    fig = go.Figure()
    fig.add_scatter(x = df_Aw0["X_global"][0], y = df_Aw0["F_global"][0], name = "Global (X,F)", mode = "markers")
    fig.update_xaxes(title = "X")
    fig.update_yaxes(title = "F(X)")
    fig.vs_show()

    ## Evolution of local optimizers
    fig = make_subplots(rows = len(df_Aw0["X_local"][0]), cols = n_vars + 1)
    for k_global in range(len(df_Aw0["X_local"][0])):
        for k_var in range(n_vars):
            fig.add_scatter(x = np.arange(len(df_Aw0["X_local"][0][k_global])), y = np.array(df_Aw0["X_local"][0][k_global]).reshape(-1, n_vars)[:,k_var], name = "X for k = " + str(k_global), row = 1+k_global, col = 1+k_var)
        fig.add_scatter(x = np.arange(len(df_Aw0["F_local"][0][k_global])), y = np.array(df_Aw0["F_local"][0][k_global]).squeeze(), name = "F for k = " + str(k_global), row = 1+k_global, col = 1+n_vars)
    fig.update_yaxes(type = "log", col = n_vars + 1)
    fig.vs_show()     

    ## Plot IE heatmap for each (A, w0)-point
    fig = go.Figure(data = go.Heatmap(x = np.log10(df2['A']), y = np.log10(df2['w0']), z = np.log10(df2['IE']), colorscale = 'RdPu_r'))
    fig.update_xaxes(title = "log A", type = "linear")
    fig.update_yaxes(title = "log w0", type = "linear")
    fig.update_layout(title = "Inference Error (for each external parameter)")
    fig.vs_show()

    ## Plot inferred params for each (A,w0)-point
    subplot_titles = []
    for k in range(n_vars):
        subplot_titles += [variable_keys[k]]
        subplot_titles += ["sigma_" + variable_keys[k]]

    fig = make_subplots(rows = n_vars, cols = 2, subplot_titles=subplot_titles)
    for k_vars in range(n_vars):
        hm_p_inf_k_vars = go.Heatmap(x = np.log10(df2['A']), y = np.log10(df2['w0']), z = df2['p_inf_' + str(k_vars)], colorscale = 'RdPu_r')
        hm_sigma_p_inf_k_vars = go.Heatmap(x = np.log10(df2['A']), y = np.log10(df2['w0']), z = np.log10(df2['sigma_p_inf_' + str(k_vars)]), colorscale = 'RdPu_r')
        fig.add_trace(hm_p_inf_k_vars, row = 1 + k_vars, col = 1)
        fig.add_trace(hm_sigma_p_inf_k_vars, row = 1 + k_vars, col = 2)
    fig.update_xaxes(title = "log A", type = "linear")
    fig.update_yaxes(title = "log w0", type = "linear")
    fig.update_layout(title = "Inferred parameters (for each external parameter)")
    fig.vs_show()

    ## Plot inferred params for all (A,w0)-point combined (only works for 1 or 2 variables) --> to generalize to N params, one needs to plot individual histograms
    if n_vars == 1:
        nbins = 20
        fig = go.Figure(go.Histogram(
            x=df2['p_inf_0'],
            nbinsx = nbins))
        fig.update_layout(title = "Histogram of inferred parameters (all external parameters combined)")
        fig.vs_show()     
    elif n_vars == 2:
        nbinsx = 100
        nbinsx = 50
        fig = go.Figure(go.Histogram2d(
            x=df2['p_inf_0'],
            y=df2['p_inf_1'], 
            nbinsx=100, nbinsy=50))
        fig.update_layout(title = "Histogram of inferred parameters (all external parameters combined)")
        fig.vs_show()

    ## Inference error for all (A, w0)-point combined, for each set of internal parameters
    # To be filled


