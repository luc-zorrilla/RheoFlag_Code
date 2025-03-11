from audioop import mul
import multiprocessing

from regex import R
from Coarse_grained_axoneme_functions import *
import numpy as np
import plotly.express as px
from datetime import datetime

#############################
### ----- Functions ----- ###

def ExtractData(filename):
    file = open(filename, "r")

    ## Metadata extraction
    file.readline()
    L_list =  list(map(float, file.readline().strip('][\n').split(', ')))
    N_list =  list(map(int, file.readline().strip('][\n').split(', ')))
    init_conf_list = file.readline().strip('][\n').split(', ')
    E_b_list = list(map(float, file.readline().strip('][\n').split(', ')))
    K_s_list = list(map(float, file.readline().strip('][\n').split(', ')))

    Nus_list = file.readline().strip('\n').split(sep = ', ')
    # Replace ":"
    Nus_list = [list(map(float, Nus.strip('][').split('; '))) for Nus in Nus_list]

    n_L_list = file.readline().strip('\n').split(sep = ', ')
    n_L_list = [list(map(float,n_L.strip('][').split('; '))) for n_L in n_L_list]

    m_L_list = list(map(float, file.readline().strip('][\n').split(', ')))

    Lambda_list = file.readline().strip('\n').split(sep = ', ')
    # Replace ":"
    Lambda_list = [list(map(str, Lambdas.strip('][').split(': '))) for Lambdas in Lambda_list]
    for k in range(len(N_list)):
        Lambdas = Lambda_list[k]
        # Replace ";"
        Lambdas = [list(map(float, Lambda.strip('][').split('; '))) for Lambda in Lambdas]
        Lambda_list[k] = Lambdas

    Zeta_list = file.readline().strip('\n').split(sep = ', ')
    # Replace ":"
    Zeta_list = [list(map(float, Zetas.strip('][').split('; '))) for Zetas in Zeta_list]

    X_flow_field_list = list(map(float, file.readline().strip('][\n').split(', ')))

    mu = float(file.readline().strip('\n'))
    r = float(file.readline().strip('\n'))
    h = float(file.readline().strip('\n'))
        
    T_span = list(map(float, file.readline().strip('][\n').split(', ')))
    T_eval = list(map(float, file.readline().strip('][\n').split(', ')))

    ## Create data arrays in functions of the metadata
    X_list = [np.zeros((len(T_eval), len(L_list), N+2, len(init_conf_list), len(E_b_list), len(K_s_list), len(n_L_list), len(m_L_list), len(X_flow_field_list))) for N in N_list]
    file.readline()
    file.readline()
    for L_index in range(len(L_list)):
        for N_index in range(len(N_list)):
            for init_conf_index in range(len(init_conf_list)):
                for E_b_index in range(len(E_b_list)):
                    for K_s_index in range(len(K_s_list)):
                        for n_L_index in range(len(n_L_list)):
                            for m_L_index in range(len(m_L_list)):
                                    for X_flow_field_index in range(len(X_flow_field_list)):
                                        file.readline()
                                        for t_index in range(len(T_eval)):
                                            X_list[N_index][t_index, L_index, :, init_conf_index, E_b_index, K_s_index, n_L_index, m_L_index, X_flow_field_index] = list(map(float, file.readline().strip('][\n').lstrip().rstrip().replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').split(' ')))
                                        file.readline()
    
    parameters_list = [L_list, N_list, init_conf_list, E_b_list, K_s_list, n_L_list, m_L_list, X_flow_field_list, mu, r, h, T_span, T_eval, Lambda_list, Zeta_list, Nus_list]
    return parameters_list, X_list


def ParamAndShape(param_indices, param_list, X_list):

    param = [param_list[k][param_indices[k]] for k in range(len(param_indices))] + param_list[-8:-3] +  [param_list[-3][param_indices[1]]] + [param_list[-2][param_indices[1]]] + [param_list[-1][param_indices[1]]]
    # X_list[N_index][t_index, L_index, N_index, init_conf_index, E_b_index, K_s_index, n_L_index, m_L_index, X_flow_field_index] = list(map(float, file.readline().strip('][\n').lstrip().replace('  ', ' ').split(' ')))
    X = X_list[param_indices[1]][:, param_indices[0], :, param_indices[2], param_indices[3], param_indices[4], param_indices[5], param_indices[6], param_indices[7]]
    return param, X

### ----- Functions ----- ###
#############################