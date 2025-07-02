# Libraries
import numpy as np
import scipy.optimize as so
from optimparallel import minimize_parallel # L-BFGS-B parallel implementation
import pandas as pd

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.express.colors import sample_colorscale
import plotly.io as pio

import glob
import pickle
import pprint
writing_dir = "C:\\Users\\Luc\\Documents\\PhD_Large_files\\RheoFlag\\Inference\\FromSimulationData\\"

from A02_Inference_Viscoelastic_Model import *
from misc_func import *

bool_test = False
if __name__ == "__main__":

    A_list = []
    w0_list = []
    p_inf_list = []
    IE_list = []
    Hm1_list = []

    filenames = glob.glob(writing_dir + "*.pkl")
    for filename in filenames:

        pkl_file = open(filename, 'rb')
        VI_dict = pickle.load(pkl_file)
        pkl_file.close()

        # pprint.pprint(VI_dict)
        # Hessian
        # print(VI_dict['output']['lowest_optimization_result']['hess_inv'].todense())

        exp_variable_params = VI_dict["exp_variable_params"]
        guess_variable_params = VI_dict["args"]["guess_variable_params"]
        flow_params = VI_dict["flow_params"]
        A = flow_params["A"]
        w0 = flow_params["w0"]
        fixed_params = VI_dict["args"]["fixed_params"]

        p_star = np.array(list(exp_variable_params.values()))
        inferred_variable_params = VI_dict["output"].x
        p = np.array(list(inferred_variable_params.values()))

        IE = L2_relative_error(p, p_star)
        Hm1 = VI_dict['output']['lowest_optimization_result']['hess_inv'].todense()
        # print("A, w0, L2 Relative Inference Error:", A, w0, IE)

        A_list.append(A)
        w0_list.append(w0)
        p_inf_list.append(p)
        IE_list.append(IE)
        Hm1_list.append(Hm1)

        ## Inferred parameters with uncertainty
        # fig = go.Figure()
        # fig.add_scatter(x = list(guess_variable_params.keys()), y = list(guess_variable_params.values()), mode = "markers", marker_color = "green", marker_size = 10, name = "p_0", opacity = 0.7)
        # fig.add_scatter(x = list(inferred_variable_params.keys()), y = list(inferred_variable_params.values()), error_y = dict(type = "data", array = np.sqrt(np.diag(Hm1))), mode = "markers", marker_color = "black", name = "p_inf", opacity = 0.7)
        # fig.add_scatter(x = list(exp_variable_params.keys()), y = list(exp_variable_params.values()), mode = "markers", marker_color = "red", marker_size = 10, name = "p_star", opacity = 0.7)
        # fig.show()

    
    # Plot transects
    ## Inference Error
    # fig = go.Figure()
    # fig.add_scatter(x = w0_list, y = IE_list, mode = "markers")
    # fig.update_xaxes(type = "log")
    # fig.show()
    # exit()

    df = pd.DataFrame()
    df["A"] = A_list
    df["w0"] = w0_list
    df["p_inf"] = p_inf_list
    df["IE"] = IE_list
    df["Hm1"] = Hm1_list # Covariance matrix
    df["Sigma"] = df.apply(lambda x: np.diag(x["Hm1"]), axis = 1)

    # Plot heatmap
    # df = df.pivot(index='A', columns='w0', values = 'IE') # .fillna(0)
    # print(df.values.shape)
    # print(df)
    # exit()

    # fig = go.Figure(data = go.Heatmap(x = df.columns, y = df.index, z = np.log10(df.values), colorscale = 'RdPu_r'))
    # fig.update_xaxes(title = "w0", type = "log")
    # fig.update_yaxes(title = "A", type = "log")
    # fig.show()

    # Combine parameter estimates (using BLC function)
    p_combined = []
    for j in range(len(list(exp_variable_params.keys()))):
        Z_vector_list_j = [np.array([df["p_inf"][k][j], df["Sigma"][k][j]]) for k in range(df["p_inf"].shape[0])]
        Z_combined_vector_j = BLC(Z_vector_list_j)
        p_combined.append(Z_combined_vector_j)
    print("p_combined", p_combined)