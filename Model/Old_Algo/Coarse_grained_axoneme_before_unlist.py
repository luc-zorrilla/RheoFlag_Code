from Coarse_grained_axoneme_functions import *
import numpy as np
import plotly.express as px
from datetime import datetime

##############################
### ----- Parameters ----- ###

## Coarse-graining parameters
# L = 10**(-5) # 10 microns
# N = 10 # Number of points (number of segments is N-1)
# Delta_S = L/(N-1) # Length of segments
L_list = [10**(-5)] # 10 microns
N_list = [10]

## Constitutive parameters
# E_b = 1 # Bending resistance
# K_b = E_b / Delta_S # Bending resistance per unit length
# K_s = 0 # Sliding resistance
E_b_list = [10**(-23)]
K_s_list = [0]
Nus_list = [[0]*N for N in N_list]

## Fluid drag and axoneme dimensions
mu = 10**(-3) # Viscosity of water
r = 100 * 10**(-9) # Radius of axoneme
h = -1 # average distance between axoneme and a hypothetic wall
# RFT drag components when there is no wall effect
# eta_list = [Eta(mu, r, L) for L in L_list]
# xi_list = [Xi(mu, r, L) for L in L_list] --> These should not be used, but mu and r yes.

## Boundary conditions
n_L_list = [[0, 0]]
m_L_list = [0]

## External forces: ad hoc or external flow

# - Ad hoc - #
# Lambda_List = [[[Lambda_0_x, Lambda_0_y], ..., [Lambda_Nm1_x, Lambda_Nm1_y]]]

Lambda = 0
Zeta = 0

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
Zetas_list = []
for N in N_list:
    Zetas_list.append([0]*N)
    # for k in range(N):
    #     if k==N//2:
    #         Zetas_list[-1][k] = Zeta
print("Zetas_list: ", Zetas_list)

## Initial conditions
init_conf_list = [StraightLine]
# Eq_vertical_boundary_force = np.array([0.000000000000000e+00, 0.000000000000000e+00, 0.000000000000000e+00, 5.877222997995371e-04, 5.532385567299844e-04, 5.184944574676916e-04, 4.842262606778126e-04, 4.492680707413920e-04, 4.151914704003446e-04, 3.800651712615969e-04, 3.461225606184570e-04, 3.108923972790254e-04, 2.770099861024614e-04, 2.417517203724261e-04, 2.078481007731863e-04, 1.726401011219769e-04, 1.386356075533087e-04, 1.035495750841326e-04, 6.937505495678696e-05, 3.446260135614057e-05])

## Simulation time parameters
dT = 2*10**(-3)
T_max = 2*10**(-1)
T_span = [0, T_max]
T_eval = [dT*i for i in range(int(T_max/dT)+1)]

## Flow field
# No flow
# X_flow_field_list = [-1] l
# Constant vertical flow
# X_flow_field_list = [np.array([0, 10**(-6)])]
# Periodic vertical flow of amplitude ( max velocity) A and frequency w0: A*sin(w0*t)
A = 10**(-3)
w0 = 2 * np.pi * 50 # 50 Hz similar to flagellar beating
T_meas = T_eval # T_meas represents the time points at which one measures the flow field. Here it is arbitrary since there is no experiment.
X_flow_field_0 = np.zeros((2,len(T_meas)))
for t in range(len(T_meas)):
    X_flow_field_0[1,t] = A*np.sin(w0 * T_meas[t])
X_flow_field_list = [X_flow_field_0]

### ----- Parameters ----- ###
##################################


#############################
### ----- Data file ----- ###

date = datetime.now().strftime("%Y%m%d-%I%M%S_%p")
file = open("Output/data_" + str(date) + ".dat", "a")

## Write metadata content

file.write("METADATA: L_list, N_list, init_conf_list, E_b_list, K_s_list, Nus_list, n_L_list, m_L_list, Lambda_list, Zeta_list, X_flow_field_list, mu, r, h, T_span, T_eval\n")
file.write(str(L_list)+"\n")
file.write(str(N_list)+"\n")
file.write(str(init_conf_list)+"\n")
file.write(str(E_b_list)+"\n")
file.write(str(K_s_list)+"\n")

Nu_string = "["
for p in range(len(Nus_list)):
    Nu_N_list = Nus_list[p]
    Nu_string += str(Nu_N_list).replace(',', ';')
    if p<len(Nus_list)-1:
        Nu_string += ", "
Nu_string += "]\n"
file.write(Nu_string)

n_L_string = "["
for n_L in n_L_list:
    n_L_string += str(n_L).replace(',',';')
    if n_L!=n_L_list[-1]:
        n_L_string += ", "
n_L_string += "]\n"
file.write(n_L_string)

file.write(str(m_L_list)+"\n")

# file.write(str(Lambda_list)+"\n") #Old version
Lambda_string = "["
for p in range(len(Lambdas_list)):
    Lambda_N_list = Lambdas_list[p]
    Lambda_string += "["
    for q in range(len(Lambda_N_list)):
        Lambda_vector = Lambda_N_list[q]
        Lambda_string += str(Lambda_vector).replace(',',';')
        if q<len(Lambda_N_list)-1:
            Lambda_string += ": "
    Lambda_string += "]"
    if p<len(Lambdas_list)-1:
        Lambda_string += ", "
Lambda_string += "]\n"
file.write(Lambda_string)

Zeta_string = "["
for p in range(len(Zetas_list)):
    Zeta_N_list = Zetas_list[p]
    Zeta_string += str(Zeta_N_list).replace(',', ';')
    if p<len(Zetas_list)-1:
        Zeta_string += ", "
Zeta_string += "]\n"
file.write(Zeta_string)

# X_flow_string = "["
# for p in range(len(X_flow_field_list)):
#     X_flow_field = X_flow_field_list[p]
#     X_flow_string += np.array2string(X_flow_field_list, precision = 15, floatmode="maxprec").replace('\n','')
# file.write(+"\n")

file.write(str(mu)+"\n")
file.write(str(r)+"\n")
file.write(str(h)+"\n")
file.write(str(T_span)+"\n")
file.write(str(T_eval)+"\n")
file.write("\n")
file.write("DATA:\n")

### ----- Data file ----- ###
#############################


###############################
### ----- Computation ----- ###

for L in L_list:
    eta = Eta(mu, r, L, h)
    xi = Xi(mu, r, L, h)
    if xi!=0:
        gamma = xi/eta
    else:
        print("Error: xi = 0")
    for k in range(len(N_list)):
        N = N_list[k]
        Nus = Nus_list[k]
        Lambdas = Lambdas_list[k]
        Zetas = Zetas_list[k]
        Delta_S = L/(N-1)
        for init_conf in init_conf_list:
            X_0 = init_conf(N)
            # X_0 = Eq_vertical_boundary_force

            for E_b in E_b_list:
                K_b = E_b / Delta_S
                for K_s in K_s_list:
                    # for eta in eta_list:
                    # for xi in xi_list:
                    for n_L in n_L_list:
                        for m_L in m_L_list:
                                for X_flow_field in X_flow_field_list:

                                    # Creates an interpolation function of the flow field to inject it in the solver
                                    InterpFlow = interpolate.interp1d(T_meas, X_flow_field, axis=1)

                                    Args = (eta, gamma, K_s, K_b, Delta_S, Nus, n_L, m_L, Lambdas, Zetas, InterpFlow)
                                    print("eta, gamma, K_s, K_b, Delta_S, Nus, n_L, m_L, Lambdas, Zetas, X_flow_field: ", Args)

                                    start_time = time.time()
                                    sol = solve_ivp(fun = f, t_span = T_span, y0 = X_0, args=Args, t_eval=T_eval).y
                                    print("Solving took %s seconds." % (time.time() - start_time))

                                    ### --------- ###
                                    ### SAVE DATA ###
                                    file.write("L = " + str(L) + ", N = " + str(N) + ", init_conf = " + str(init_conf) + ", K_b = " + str(K_b) + ", K_s = " + str(K_s) + ", eta = " + str(eta) + ", xi = " + str(xi) + ", Nus = " + str(Nus) + ", n_L = " + str(n_L) + ", m_L = " + str(m_L) + ", Lambdas = " + str(Lambdas)+ ", Zetas = " + str(Zetas) + ", T_span = " + str(T_span) + ", T_eval = " + str(T_eval) + ", Delta_S = " + str(Delta_S) + "\n") # Write parameters used for this loop
                                    for t in range(len(T_eval)):
                                        # file.write(str(sol[:,t]).replace('\n','')+"\n")
                                        file.write(np.array2string(sol[:,t], precision = 15, floatmode="maxprec").replace('\n','')+"\n")
                                    file.write("\n")
                                
                                ### SAVE DATA ###
                                ### --------- ###
file.close()
### ----- Computation ----- ###
###############################