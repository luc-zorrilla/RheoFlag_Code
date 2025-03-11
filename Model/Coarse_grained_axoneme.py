""" This file """

### libraries ###
# from Coarse_grained_analysis import *
from Coarse_grained_axoneme_functions import *
import numpy as np
import plotly.express as px
from datetime import datetime
import warnings
import multiprocessing as mp

if __name__ == "__main__":
    ###########################################
    ### ----- Adimensional Parameters ----- ###

    # RFT parameters
    gamma_list = [2] #1.5834

    ## Coarse-graining parameters
    N_list = [10]

    ## Constitutive parameters
    # Sperm number
    Sp4_list = [10**(-1)]
    # Shear / bending elasticity ratio
    Beta_list = [0]
    Tau_b_list = [0, 10**(-2), 10**(-1), 10**(0), 10**(1), 10**(2), 10**(3)]
    taus_b_list = [[[tau_b]*N for tau_b in Tau_b_list] for N in N_list]

    ## Boundary conditions
    n_L_list = [[0, 0]]
    m_L_list = [0]

    ## External forces: ad hoc or external flow

    # - Ad hoc - #
    # Lambda_List = [[[Lambda_0_x, Lambda_0_y], ..., [Lambda_Nm1_x, Lambda_Nm1_y]]]
    Lambda = 0
    # Uniform constant vertical force
    Lambdas_list = [[[0, Lambda] for k in range(N)] for N in N_list]

    # Uniform vertical force on middle segment
    # Lambdas_list = []
    # for N in N_list:
    #     Lambdas_list.append([[0,0]]*N)
    #     for k in range(N):
    #         if k==N//2:
    #             Lambdas_list[-1][k] = [0, Lambda]

    # Uniform horizontal force on segment at L/2
    # Lambdas_list = []
    # for N in N_list:
    #     Lambdas_list.append([[0,0]]*N)
    #     for k in range(N):
    #         if k==N//2:
    #             Lambdas_list[-1][k] = [-Lambda, 0]

    # Uniform vertical force on last segment
    # Lambdas_list = []
    # for N in N_list:
    #     Lambdas_list.append([[0,0]]*N)
    #     for k in range(N):
    #         if k==N-1:
    #             Lambdas_list[-1][k] = [0, Lambda]
    # print("Lambdas_list: ", Lambdas_list)

    # Torque between the two first segments
    # Zetas_list = []
    # for N in N_list:
    #     Zetas_list.append([0]*N)
    #     for k in range(N):
    #         if k==1:
    #             Zetas_list[-1][k] = Zeta
    # print("Zetas_list: ", Zetas_list)

    # Torque between the two middle segments
    Zeta = 0
    Zetas_list = []
    for N in N_list:
        Zetas_list.append([Zeta]*N)
        # for k in range(N):
        #     if k==N//2:
        #         Zetas_list[-1][k] = Zeta
    # print("Zetas_list: ", Zetas_list)

    ## Initial conditions
    init_conf_list = [StraightLine]
    # Eq_vertical_boundary_force = np.array([0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 5.877222997995371e-04, 5.532385567299844e-04, 5.184944574676916e-04, 4.842262606778126e-04, 4.492680707413920e-04, 4.151914704003446e-04, 3.800651712615969e-04, 3.461225606184570e-04, 3.108923972790254e-04, 2.770099861024614e-04, 2.417517203724261e-04, 2.078481007731863e-04, 1.726401011219769e-04, 1.386356075533087e-04, 1.035495750841326e-04, 6.937505495678696e-05, 3.446260135614057e-05])

    ## Flow field
    # No flow
    # X_flow_field_list = [-1]
    # Constant vertical flow
    # X_flow_field_list = [np.array([0, 10**(-6)])]
    # Periodic vertical flow of amplitude ( max velocity) A and frequency w0: A*sin(t)
    A_list = [10**(-3), 10**(-2), 10**(-1)] # [0]
    w0_list = [10**(-1), 10**(0), 10**(1)]
    w0 = 0 # 0 for constant flow, otherwise sinusoidal flow of period w0 in w_s units.
    psi = np.pi/2

    ## Simulation time parameters, counted in tau_s

    # dT = 10**(-1) # 1*10**(0)
    # T_max = 10**(2) # 5*10**(-3)
    # T_span = [0, T_max]
    # T_eval = [dT*i for i in range(int(T_max/dT)+1)]
    dT_list = [2*np.pi/(100*w0) for w0 in w0_list]
    T_max_list = [2*np.pi*20/w0 for w0 in w0_list]
    T_span_list = [[0, T_max] for T_max in T_max_list]
    T_eval_list = [[dT_list[l]*i for i in range(int(T_max_list[l]/dT_list[l])+1)] for l in range(len(w0_list))]

    Flow_field_filename = ""
    X_flow_field_list = []
    X_flow_field_string_list = []
    for k in range(len(A_list)):
        for l in range(len(w0_list)):
            X_flow_field_string, X_flow_field = CreateFlowField(A_list[k], w0_list[l], psi, T_eval_list[l], filename = Flow_field_filename)
            X_flow_field_list.append(X_flow_field)
            X_flow_field_string_list.append(X_flow_field_string)
    ### ----- Adimensional parameters ----- ###
    ###########################################


    ##############################
    ### ----- Data files ----- ###

    # File with data on parameter loops
    date = datetime.now().strftime("%Y%m%d-%I%M%S%f")
    output_folder = "C:/Users/Luc/Documents/PhD_Large_files/RheoFlag/Model/Output/renormalized/bending_elasticity_viscosity/periodic_response/"
    list_file = open(output_folder + "list_data_" + str(date) + ".dat", "w")

    ## Write metadata content

    list_file.write("METADATA:" + "\n")
    list_file.write("N_list = " + str(N_list) + "\n")
    list_file.write("init_conf_List = " + str(init_conf_list) + "\n")
    list_file.write("Beta_list = " + str(Beta_list) + "\n")
    list_file.write("Tau_b_list = " + str(taus_b_list) + "\n")
    list_file.write("n_L_list = " + str(n_L_list) + "\n")
    list_file.write("m_L_List = " + str(m_L_list) + "\n")
    list_file.write("Lambdas_list = " + str(Lambdas_list) + "\n")
    list_file.write("Zetas_list = " + str(Zetas_list) + "\n")

    X_flow_field_list_string = "X_flow_field_list = " + "["
    for p in range(len(X_flow_field_string_list)):
        X_flow_field_list_string += X_flow_field_string_list[p]
        if p<len(X_flow_field_string_list)-1:
            X_flow_field_list_string += ", "
    X_flow_field_list_string += "]" + "\n"
    list_file.write(X_flow_field_list_string)

    list_file.write("Sp4_list = " + str(Sp4_list) + "\n")
    list_file.write("gamma_list = " + str(gamma_list) + "\n")
    list_file.write("T_span_list = " + str(T_span_list)+"\n")
    list_file.write("T_eval_list = " + str(T_eval_list)+"\n")
    list_file.write("\n")
    list_file.close()
    ### ----- Data files ----- ###
    #############################


    ############################################
    ### ----- Adimensional Computation ----- ###

    # Parameters of interest:
    # - Internal parameters: Sp4, Beta, Tau_s, Tau_b, K_b
    # - External forcing parameters: A, w0
    # - Simulation parameters: N or Delta_S

    # Number of systems to integrate = 
    files_number = len(N_list)*len(init_conf_list)*len(Beta_list)*len(n_L_list)*len(m_L_list)*len(A_list)*len(w0_list)*len(Sp4_list)*len(gamma_list)
    print(files_number, "different problems will be integrated")
    progression_number = 0

    pool = mp.Pool(mp.cpu_count())

    for n in range(len(N_list)):
        N = N_list[n]
        taus_b_real_list = taus_b_list[n]
        Lambdas = Lambdas_list[n]
        Zetas = Zetas_list[n]
        for taus_b in taus_b_real_list:
            for init_conf in init_conf_list:
                X_0 = init_conf(N)
                for Beta in Beta_list:
                    for n_L in n_L_list:
                        for m_L in m_L_list:
                            for k in range(len(A_list)):
                                A = A_list[k]
                                for l in range(len(w0_list)):
                                    w0 = w0_list[l]
                                    T_span = T_span_list[l]
                                    T_eval = T_eval_list[l]
                                    X_flow_field = X_flow_field_list[k*len(w0_list)+l] ## 
                                    X_flow_field_string = X_flow_field_string_list[k*len(w0_list)+l]

                                    for Sp4 in Sp4_list:
                                        for gamma in gamma_list:
                                            # Write individual data

                                            args_SolveAndSave = (output_folder, N, taus_b, init_conf, Beta, gamma, n_L, m_L, A, w0, Sp4, Lambdas, Zetas, X_flow_field_string, T_span, T_eval, X_flow_field, X_0)
                                            # args_g = (N, Sp4, w0, A)
                                            # print("Before pool.apply_async")
                                            res = pool.apply_async(func = SolveAndSave, args = args_SolveAndSave, callback = SolveAndSave_callback)
                                            # progression_number += 1
                                            # print("progression: ", progression_number/files_number, "%% of the problems have been solved.")
                                        
    pool.close()
    pool.join() # postpones the execution of next line of code until all processes in the queue are done.
                                 
    ### ----- Adimensional Computation ----- ###
    ############################################