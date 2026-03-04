""" This script ___ """

## Libraries

from C01_Inference_Viscoelastic_Model import *
from misc_func import *

import glob
import pickle
from pathlib import Path
writing_path = (Path(__file__).resolve().parent.parent / 'Inference' / 'FromSimulationData' / 'MultiplePeriods' / 'LastPeriod' / 'BendingElasticity_BendingViscosity_Clamped' / 'FixedSp4')
print("writing_path", writing_path)
import numpy as np
import pandas as pd

from plotly.subplots import make_subplots
import plotly.graph_objects as go

## Main

bool_test = False
if __name__ == "__main__":

    # Import dataframe written after second pass, i.e., "df.pkl".
    reading_filename = str((writing_path / "df.pkl").resolve())
    df = pd.read_pickle(reading_filename)
    print("df", df)

    # Global averages

    p_mean = custom_average(df['p_inf'], df['sigma'], type = "mean") # type: "mean", "median", "combined"
    p_median = custom_average(df['p_inf'], df['sigma'], type = "median")
    p_combined = custom_average(df['p_inf'], df['sigma'], type = "combined")

    new_p_mean = custom_average(df['new_p_inf'], df['new_sigma'], type = "mean") # type: "mean", "median", "combined"
    new_p_median = custom_average(df['new_p_inf'], df['new_sigma'], type = "median")
    new_p_combined = custom_average(df['new_p_inf'], df['new_sigma'], type = "combined")

    print("p_mean, new_p_mean", p_mean, new_p_mean)
    print("p_median, new_p_median", p_median, new_p_median)
    print("p_combined, new_p_combined", p_combined, new_p_combined)

    exit()

    # Select for specific external parameters
    # df_Aw0 = df[(df['A'] == 1e-8) & (df['w0'] == 1e-3)].reset_index()
    
    # Plots

    ## Plot IE heatmap for each (A, w0)-point
    fig = go.Figure(data = go.Heatmap(x = np.log10(df['A']), y = np.log10(df['w0']), z = np.log10(df['IE']), colorscale = 'RdPu_r'))
    fig.update_xaxes(title = "log A", type = "linear")
    fig.update_yaxes(title = "log w0", type = "linear")
    fig.update_layout(title = "Inference Error (for each external parameter)")
    fig.vs_show()

    ## Plot F_global (final) heatmap for each (A,w0)-point
    fig = go.Figure(data = go.Heatmap(x = np.log10(df['A']), y = np.log10(df['w0']), z = np.log10(df['F_inf']), colorscale = 'RdPu_r'))
    fig.update_xaxes(title = "log A", type = "linear")
    fig.update_yaxes(title = "log w0", type = "linear")
    fig.update_layout(title = "F_inf (for each external parameter)")
    fig.vs_show()    

    ## Plot inferred params for each (A,w0)-point
    subplot_titles = []
    for k in range(n_vars):
        subplot_titles += [variable_keys[k]]
        subplot_titles += ["sigma_" + variable_keys[k]]

    fig = make_subplots(rows = n_vars, cols = 2, subplot_titles=subplot_titles)
    for k_vars in range(n_vars):
        hm_p_inf_k_vars = go.Heatmap(x = np.log10(df['A']), y = np.log10(df['w0']), z = df['p_inf_' + str(k_vars)], colorscale = 'RdPu_r')
        hm_sigma_p_inf_k_vars = go.Heatmap(x = np.log10(df['A']), y = np.log10(df['w0']), z = np.log10(df['sigma_p_inf_' + str(k_vars)]), colorscale = 'RdPu_r')
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
            x=df['p_inf_0'],
            nbinsx = nbins))
        fig.update_layout(title = "Histogram of inferred parameters (all external parameters combined)")
        fig.vs_show()     
    elif n_vars == 2:
        nbinsx = 100
        nbinsy = 50
        fig = go.Figure(go.Histogram2d(
            x=df['p_inf_0'],
            y=df['p_inf_1'], 
            nbinsx=100, nbinsy=50))
        fig.update_layout(title = "Histogram of inferred parameters (all external parameters combined)")
        fig.vs_show()

    # Plot X, F evolution for each A, w0
    ## Evolution of the global optimizer (showing successive minima)
    fig = make_subplots(rows = n_vars + 1, cols = 1)
    for k_var in range(n_vars):
        fig.add_scatter(x = np.arange(len(df_Aw0["F_global"][0])), y = np.array(df_Aw0["X_global"][0]).reshape(-1, n_vars)[:,k_var], row = 1+k_var, col = 1, name = "X_k global for k = " + str(k_var))
    fig.add_scatter(x = np.arange(len(df_Aw0["F_local"][0])), y = np.array(df_Aw0["F_global"][0]), row = 1+n_vars, col = 1, name = "F(X) global")
    fig.update_yaxes(type = "log", row = n_vars + 1)
    fig.vs_show()

    ## Evolution of local optimizers
    fig = make_subplots(rows = len(df_Aw0["X_local"][0]), cols = n_vars + 1)
    for k_global in range(len(df_Aw0["X_local"][0])):
        for k_var in range(n_vars):
            fig.add_scatter(x = np.arange(len(df_Aw0["X_local"][0][k_global])), y = np.array(df_Aw0["X_local"][0][k_global]).reshape(-1, n_vars)[:,k_var], name = "X for k = " + str(k_global), row = 1+k_global, col = 1+k_var)
        fig.add_scatter(x = np.arange(len(df_Aw0["F_local"][0][k_global])), y = np.array(df_Aw0["F_local"][0][k_global]).squeeze(), name = "F for k = " + str(k_global), row = 1+k_global, col = 1+n_vars)
    fig.update_yaxes(type = "log", col = n_vars + 1)
    fig.vs_show()     
