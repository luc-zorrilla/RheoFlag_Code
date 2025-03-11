from audioop import mul
import multiprocessing
from re import A

from regex import R
from Coarse_grained_axoneme_functions import *
from Coarse_grained_analysis_functions import *

import numpy as np
from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime

#######################################
### ----- Read data from file ----- ###
# exit()

folder_name = "C:/Users/Luc/Documents/PhD_Large_files/RheoFlag/Model/Output/renormalized/bending_elasticity_viscosity/periodic_response/"
# A = 1e-2
# w0 = 1: tau = 0: "data_20240710-072407387487.dat", tau = 1e-2: "data_20240710-072434872817.dat", tau = 1e-1: "solver error", tau = 1e0: "data_20240710-072510078570.dat", tau = 1e1: "data_20240710-072538423593.dat"
# w0 = 10: tau = 0: "data_20240710-072407436877.dat", tau = 1e-2: "data_20240710-072439597364.dat", tau = 1e-1: "solver error, tau = 1e0: "data_20240710-072511265361.dat", tau = 1e1: "data_20240710-072542375262.dat"

X_list = []
filename_list = ["data_20240710-072407436877.dat", "data_20240710-072542375262.dat"]
for filename in filename_list:
    parameters, X = ExtractParametersData(folder_name + filename)

    N, taus_b, init_conf, Beta, gamma, n_L, m_L, A, w0, Sp4, Lambdas, Zetas, X_flow_field_type, X_flow_field_params, T_span, T_eval = parameters
    tau_b = taus_b[0]
    print("N = ", N)
    print("A = ", A)
    print("w0 = ", w0)
    print("Bending elasticity / drag timescale: Sp4 = ", Sp4)
    print("Shear/bending elasticity ratio: Beta = ", Beta)
    print("tau_b = ", tau_b)

    # print(X_flow_field_type)
    # exit()
    # print("shape(X) = ", X.shape)
    # print("X[0] = ", X[:,0])
    # print("parameters = ", parameters)


    ##################################
    ### ----- Shape analysis ----- ###
    T_eval = np.array(T_eval)
    X_flow = A*np.sin(w0*T_eval)

    X_list.append(X)

# Animated shape
overlap_fig = AnimatedShapes(X_list[0], X_list[1], X_flow, N, w0, Sp4, Beta, tau_b, T_eval)
overlap_fig.show()
