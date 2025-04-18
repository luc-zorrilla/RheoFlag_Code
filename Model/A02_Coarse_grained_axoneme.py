""" This file aims at simulating a viscoelastic filament (coarse-grained) on a subset of 
input parameters. Then each simulation is saved individually. """

### libraries ###
from A01_Coarse_grained_axoneme_functions import *
import numpy as np
import plotly.express as px
from datetime import datetime
import warnings
import multiprocessing as mp

# Folder in which simulation outputs are stored
output_folder = "C:/Users/Luc/Documents/PhD_Large_files/RheoFlag/Model/Output/"

if __name__ == "__main__":
    ###########################################
    ### ----- Adimensional Parameters ----- ###

    print("Preparing parameter space...")

    ####################
    ## RFT parameters ##
    gamma_list = [2] #1.5834
    ####################


    ################################
    ## Coarse-graining parameters ##
    N_list = [10]
    ################################

    #############################
    ## Constitutive parameters ##
    
    ################
    # Sperm number #
    Sp4_list = [1e0]
    ################

    ###############################
    # basal hinge spring constant #
    k0_list = [1e10]
    ###############################
    
    #################################
    # Bending elasticity activation #
    #################################
    bool_EI = False # Default is True
    #################################

    ####################################
    # Shear / bending elasticity ratio #
    Beta_list = [1e2, 1e3, 1e4]
    ####################################

    ###############################
    # Bending viscosity timescale #
    # remark : it is in tau_s units
    Tau_b_list = [0]
    taus_b_list = [[[tau_b]*(N-1) for tau_b in Tau_b_list] for N in N_list]
    ###############################

    ##############################
    # Shear viscosity activation #
    tau_s_list = [1e3] #, 1e1, 1e2, 1e3]
    ##############################

    ####################################

    #############################

    #########################
    ## Boundary conditions ##

    ######################
    # Spatial conditions #
    n_L_list = [[0, 0]] # Force at s = L
    m_L_list = [0] # Torque at s = L
    ######################
    print("Parameter space prepared.")

    #######################
    # Temporal conditions #

    print("Preparing initial conditions...")

    
    init_conf_list = [SecondBend] ## Initial conditions in [StraightLine, ProximalBend, SecondBend]

    print("Initial conditions prepared. ")
    #######################  

    #########################

    print("External forces and torques...")
    #############################
    ## External forces: ad hoc ##

    #########
    # Force #
    # Lambda_List = [[[Lambda_0_x, Lambda_0_y], ..., [Lambda_Nm1_x, Lambda_Nm1_y]]]

    # Negative horizontal force at s = L/2
    Lambda = [0,0]
    Lambdas_list = [[[0,0]]*((N-1)//2) + [Lambda] + [[0,0]]*(N//2) for N in N_list]

    # Uniform force along length
    # Lambda = [0,0]
    # Lambdas_list = [[Lambda for k in range(N)] for N in N_list]
    
    
    #########

    ##########
    # Torque #
    Zeta = 0
    Zetas_list = []
    for N in N_list:
        Zetas_list.append([Zeta]*N)    
    ##########

    #########
    # Force #

    # Uniform constant vertical force
    # Lambdas_list = [[[0, Lambda] for k in range(N)] for N in N_list]        

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
    #########

    ##########
    # Torque #

    # Torque between the two first segments
    # Zetas_list = []
    # for N in N_list:
    #     Zetas_list.append([0]*N)
    #     for k in range(N):
    #         if k==1:
    #             Zetas_list[-1][k] = Zeta
    # print("Zetas_list: ", Zetas_list)
    
    # Torque between the two middle segments
    # Zeta = 0
    # Zetas_list = []
    # for N in N_list:
    #     Zetas_list.append([Zeta]*N)
        # for k in range(N):
        #     if k==N//2:
        #         Zetas_list[-1][k] = Zeta
    # print("Zetas_list: ", Zetas_list)
    ##########

    print("External forces and torques ready.")
    #############################


    ################
    ## Flow field ##

    # No flow
    # X_flow_field_list = [-1]

    # Constant vertical flow
    # X_flow_field_list = [np.array([0, 10**(-6)])]
    
    # Periodic vertical flow of amplitude ( max velocity) A and frequency w0: A*sin(t)
    A_list = [0]
    w0_list = [0]
    w0 = 0 # 0 for constant flow, otherwise sinusoidal flow of period w0 in w_s units.
    psi = np.pi/2 # Angle of the flow w.r.t. the horizontal axis

    ################

    ###########################
    ## Simulation parameters ##

    ######################
    # Integration scheme #
    method_list = ['BDF'] # ["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"]
    ######################

    ################################
    # Time, counted in tau_s units #
    print("Setting time...")

    # time determined by the flow timescale
    # dT_list = [2*np.pi/(100*w0) for w0 in w0_list]
    # T_max_list = [2*np.pi*100/w0 for w0 in w0_list]

    # Same time for all simulations
    dT_list = [5e-1 for w0 in w0_list]
    T_max_list = [5e3 for w0 in w0_list]

    T_span_list = [[0, T_max] for T_max in T_max_list]
    T_eval_list = [[dT_list[l]*i for i in range(int(T_max_list[l]/dT_list[l]))] for l in range(len(w0_list))]

    ################################

    ########################################
    # Maximum simulation time (s) per step #
    T_sim_max = 600 # 12h
    ########################################

    print("Time set.")
    ###########################

    ### ----- Adimensional parameters ----- ###
    ########################################### 


    ###################################
    ### ---- Create flow field ---- ###
    print('Creating flow field...')

    Flow_field_filename = ""
    X_flow_field_list = []
    X_flow_field_string_list = []
    s = 0
    for k in range(len(A_list)):
        for l in range(len(w0_list)):
            s += 1
            print('s = ', s, 'flow fields have been created out of', len(A_list)*len(w0_list), '(', 100 * s / (len(A_list) * len(w0_list)), 'percent).')
            X_flow_field_string, X_flow_field = CreateFlowField(A_list[k], w0_list[l], psi, T_eval_list[l], filename = Flow_field_filename)
            X_flow_field_list.append(X_flow_field)
            X_flow_field_string_list.append(X_flow_field_string)

    print('Flow field created.')
    ### ---- Create flow field ---- ###
    ###################################


    ############################################
    ### ----- Adimensional Computation ----- ###

    print("Computations will start...")

    # Parameters of interest:
    # - Initial conditions: X_0, init_conf
    # - Boundary conditions: n_L, m_L
    # - Internal parameters: Sp4, Beta, Tau_s, Tau_b, K_b
    # - External force and torque parameters: Lambdas, Zetas
    # - External flow parameters: A, w0, X_flow_field_string, X_flow_field
    # - Simulation parameters: N, T_span, T_eval, method

    # Number of systems to integrate
    files_number = len(N_list)*len(init_conf_list)*len(Tau_b_list)*len(tau_s_list)*len(Beta_list)*len(n_L_list)*len(m_L_list)*len(A_list)*len(w0_list)*len(Sp4_list)*len(k0_list)*len(gamma_list)*len(method_list)
    print(files_number, "problems will be integrated")

    # Start parallel computation
    pool = mp.Pool(mp.cpu_count())

    # Loop over all parameters
    for n in range(len(N_list)):
        N = N_list[n]
        taus_b_real_list = taus_b_list[n]
        Lambdas = Lambdas_list[n]
        Zetas = Zetas_list[n]
        for taus_b in taus_b_real_list:
            for tau_s in tau_s_list:
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
                                            for k0 in k0_list:
                                                for gamma in gamma_list:
                                                    for method in method_list:

                                                        #########################################
                                                        ### ---- Gather solver arguments ---- ###
                                                        solver_dict = dict(output_folder = output_folder, N = N, taus_b = taus_b, tau_s = tau_s, init_conf = init_conf, bool_EI = bool_EI, Beta = Beta, gamma = gamma, n_L = n_L, m_L = m_L, A = A, w0 = w0, Sp4 = Sp4, k0 = k0, Lambdas = Lambdas, Zetas = Zetas, X_flow_field_string = X_flow_field_string, T_span = T_span, T_eval = T_eval, T_sim_max = T_sim_max, X_flow_field = X_flow_field, X_0 = X_0, method = method)
                                                        #########################################

                                                        res = pool.apply_async(func = SolveAndSave, args = list(solver_dict.values()), callback = SolveAndSave_callback)
                                                
    pool.close()
    pool.join() # postpones the execution of next line of code until all processes in the queue are done.
    print("All problems have been solved (one way or another!). ")
    
    ### ----- Adimensional Computation ----- ###
    ############################################