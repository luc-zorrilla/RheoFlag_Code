from audioop import mul
import multiprocessing

from regex import R
from Coarse_grained_axoneme_functions import *
import numpy as np
import plotly.express as px
from datetime import datetime

#######################################
### ----- Read data from file ----- ###

def ExtractDataNoFlow(filename):
    file = open(filename, "r")

    ## Metadata extraction
    file.readline()
    N_list =  list(map(int, file.readline().strip('][\n').split(', ')))
    init_conf_list = file.readline().strip('][\n').split(', ')
    K_b_list = list(map(float, file.readline().strip('][\n').split(', ')))
    K_s_list = list(map(float, file.readline().strip('][\n').split(', '))) 
    eta_list = list(map(float, file.readline().strip('][\n').split(', '))) 
    xi_list = list(map(float, file.readline().strip('][\n').split(', ')))

    n_L_list = file.readline().strip('\n').split(sep = ', ')
    n_L_list = [list(map(float,n_L.strip('][').split('; '))) for n_L in n_L_list]

    m_L_list = list(map(float, file.readline().strip('][\n').split(', ')))
    Lambda_list = list(map(float, file.readline().strip('][\n').split(', ')))
    T_span = list(map(float, file.readline().strip('][\n').split(', ')))
    T_eval = list(map(float, file.readline().strip('][\n').split(', ')))
    Delta_S = float(file.readline().strip('\n'))

    ## Create data arrays in functions of the metadata
    X_list = [np.zeros((len(T_eval), N+2, len(init_conf_list), len(K_b_list), len(K_s_list), len(eta_list), len(xi_list), len(n_L_list), len(m_L_list), len(Lambda_list))) for N in N_list]
    file.readline()
    file.readline()
    for N_index in range(len(N_list)):
        for init_conf_index in range(len(init_conf_list)):
            for K_b_index in range(len(K_b_list)):
                for K_s_index in range(len(K_s_list)):
                    for eta_index in range(len(eta_list)):
                        for xi_index in range(len(xi_list)):
                            for n_L_index in range(len(n_L_list)):
                                for m_L_index in range(len(m_L_list)):
                                    for Lambda_index in range(len(Lambda_list)):
                                        file.readline()
                                        for t_index in range(len(T_eval)):
                                            X_list[N_index][t_index, :, init_conf_index, K_b_index, K_s_index, eta_index, xi_index, n_L_index, m_L_index, Lambda_index] = list(map(float, file.readline().strip('][\n').lstrip().replace('  ', ' ').split(' ')))
                                        file.readline()
    
    parameters_list = [N_list, init_conf_list, K_b_list, K_s_list, eta_list, xi_list, n_L_list, m_L_list, Lambda_list, T_span, T_eval, Delta_S]
    return parameters_list, X_list

filename = "Output/" + "Pure_bending/Uniform_force/" + "pure_bending_benchmark_uniform" + "_0010" + ".dat" # Put here the name of the file you want to analyze
parameters_list, X_list = ExtractDataNoFlow(filename)
# print("len(X_list) = ", len(X_list))
# print("X_list[0] = ", X_list[0][:,:,0,0,0,0,0,0,0])
# print("parameters = ", parameters)

### ----- Read data from file ----- ###
#######################################



#################################
### ----- Visualisation ----- ###

red = "#FF0000"
black = "#000000"
light_orange = "#ffbc40"

def ParamAndShapeNoFlow(param_indices, param_list, X_list):

    param = [param_list[k][param_indices[k]] for k in range(len(param_indices))] + param_list[-3:]
    X = X_list[param_indices[0]][:,:, param_indices[1], param_indices[2], param_indices[3], param_indices[4], param_indices[5], param_indices[6], param_indices[7], param_indices[8]]
    return param, X

##################
## --- Y(X) --- ##

# Parameter choice from the chosen data file
no_flow = False
if no_flow:
    N_index = parameters_list[0].index(20)
    init_conf_index = 0
    K_b_index = 0
    K_s_index = 0
    eta_index = 0
    xi_index = 0
    n_L_index = 0
    m_L_index = 0
    Lambda_index = parameters_list[8].index(0.00005)
    parameters_indices = [N_index, init_conf_index, K_b_index, K_s_index, eta_index, xi_index, n_L_index, m_L_index, Lambda_index]
    parameters, X = ParamAndShapeNoFlow(parameters_indices, parameters_list, X_list)
    N, init_conf, K_b, K_s, eta, xi, n_L, m_L, Lambda, T_span, T_eval, Delta_s = parameters

    ## Analytical equilibrium solution
    X_eq = np.zeros((2*N))
    L = (N-1)*Delta_s
    # For a point force at distal end
    # for k in range(N):
    #     X_eq[k] = Delta_s * k
    #     X_eq[N+k] = 3 * (X_eq[k]/L)**2 - (X_eq[k]/L)**3
    #     X_eq[N+k] = X_eq[N+k] * n_L[1]*(L**3) / 6 / K_b
    # For a uniform vertical force
    for k in range(N):
        X_eq[k] = Delta_s * k
        X_eq[N+k] = (X_eq[k]/L)**4 - 4*(X_eq[k]/L)**3 + 6*(X_eq[k]/L)**2
        X_eq[N+k] = X_eq[N+k] * Lambda*(L**4) / 24 / K_b


    fig_one_set = go.Figure()
    new_trace_eq = go.Scatter(x = X_eq[:N], y = X_eq[N:], name = "Analytical solution", line = dict(color=red))
    fig_one_set.add_trace(new_trace_eq)
    for t in range(X.shape[0]):
        X_3N_t = X3N(X[t,:], Delta_s)
        if t==0 or t==X.shape[0]-1:
            new_trace_t = go.Scatter(x=X_3N_t[:N,0], y=X_3N_t[N:2*N,0], name="t = " + str(T_eval[t]), line=dict(color=black))
            fig_one_set.add_trace(new_trace_t)
        else:
            new_trace_t = go.Scatter(x=X_3N_t[:N,0], y=X_3N_t[N:2*N,0], name="t = " + str(T_eval[t]), visible=False, line=dict(color=light_orange))
            fig_one_set.add_trace(new_trace_t)


    fig_one_set.update_layout(
        legend_title_text='Beam shapes',
        title = "Beam shapes reach analytical solution for pure bending at long time.",
        xaxis_title = str(parameters),
        yaxis_title = "Y(X)",
        )

    fig_one_set.update_xaxes(
        showgrid=False,
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor="black",
        titlefont_size = 8
    )

    fig_one_set.update_yaxes(
        scaleanchor = "x",
        scaleratio = 1,
        showgrid=False,
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor="black"
    )

    fig_one_set.show()
    ## --- Y(X) --- ##
    ##################

    ###################
    ## --- DY/Dt --- ##
    N_index = parameters_list[0].index(20)
    init_conf_index = 0
    K_b_index = 0
    K_s_index = 0
    eta_index = 0
    xi_index = 0
    n_L_index = 0
    m_L_index = 0
    Lambda_index = parameters_list[8].index(0.00005)

    parameters_indices = [N_index, init_conf_index, K_b_index, K_s_index, eta_index, xi_index, n_L_index, m_L_index, Lambda_index]
    parameters, X = ParamAndShapeNoFlow(parameters_indices, parameters_list, X_list)
    # parameters = [parameters_list[k][parameters_indices[k]] for k in range(len(parameters_indices))] + parameters_list[-3:]
    N, init_conf, K_b, K_s, eta, xi, n_L, m_L, Lambda, T_span, T_eval, Delta_s = parameters

    # Kinetic energy calculation
    X_dot = np.zeros((len(T_eval)-1, N+2))
    K = np.zeros((len(T_eval)-1)) # Kinetic energy
    for t_index in range(1, len(T_eval)):
        t = T_eval[t_index]
        X_dot[t_index -1, :] = (X[t_index,:] - X[t_index-1, :]) / (T_eval[t_index] - T_eval[t_index-1]) 
        K[t_index-1] = np.linalg.norm(X_dot[t_index-1, 2:]) / 2

    fig_X_dot = go.Figure()
    new_trace = go.Scatter(x = T_eval[1:], y = K, line=dict(color=light_orange))
    fig_X_dot.add_trace(new_trace)

    fig_X_dot.update_xaxes(
        showgrid=False,
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor="black"
    )

    fig_X_dot.update_yaxes(
        # scaleanchor = "x",
        # scaleratio = 1,
        showgrid=False,
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor="black"
    )

    fig_X_dot.update_layout(
        title = "Kinetic energy decays to zero: static equilibrium is reached in finite time",
        xaxis_title = "Time",
        yaxis_title = "Kinetic energy"
    )

    fig_X_dot.show()


    ## --- dY/dt --- ##
    ###################

    ### ----- Visualisation ----- ###
    #################################
