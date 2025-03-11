from audioop import mul
import multiprocessing
from re import A

from regex import R
from Coarse_grained_axoneme_functions import *
from Coarse_grained_analysis_functions import *

import numpy as np
from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

import plotly.express as px
from datetime import datetime


#######################################
### ----- Read data from file ----- ###

filename = "Output/" + "Periodic_flow/" + "test" + "_0005" + ".dat" # Put here the name of the file you want to analyze
parameters_list, X_list = ExtractData(filename)
# print("len(X_list) = ", len(X_list))
# print("X_list[0] = ", X_list[0][:,:,0,0,0,0,0,0,0])
# print("parameters = ", parameters)

### ----- Read data from file ----- ###
#######################################


##############################################################
### ----- Parameter choice from the chosen data file ----- ###
L_index = 0
N_index = 0
init_conf_index = 0
E_b_index = 0
K_s_index = 0
n_L_index = 0
m_L_index = 0
parameters_indices = [L_index, N_index, init_conf_index, E_b_index, K_s_index, n_L_index, m_L_index]

parameters, X = ParamAndShape(parameters_indices, parameters_list, X_list)
L, N, init_conf, E_b, K_s, n_L, m_L, mu, r, h, T_span, T_eval, Lambdas, Zetas, Nus = parameters
print("Lambdas = ", Lambdas)
print("Zetas = ", Zetas)
Delta_s = L/(N-1) 
eta = Eta(mu, r, L, h)
xi = Xi(mu, r, L, h)
K_b = E_b / Delta_s
gamma = 0
if xi!=0:
    gamma = xi / eta

### ----- Parameter choice from the chosen data file ----- ###
##############################################################


#################################
### ----- Visualisation ----- ###

#############################
## --- Plot parameters --- ##

red = "#d62728"
black = "#000000"
curry = "#bcbd22"
orange = "#ff7f0e"
blue = '#17becf'
purple = '#9467bd'

## --- Plot parameters --- ##
#############################

##################
## --- Y(X) --- ##

## Analytical equilibrium solution
# X_eq = np.zeros((2*N))
# L = (N-1)*Delta_s
# For a point force at distal end
# for k in range(N):
#     X_eq[k] = Delta_s * k
#     X_eq[N+k] = 3 * (X_eq[k]/L)**2 - (X_eq[k]/L)**3
#     X_eq[N+k] = X_eq[N+k] * n_L[1]*(L**3) / 6 / E_b
# For a uniform force at distal segment
# F = Lambdas[-1][1]*Delta_s
# for k in range(N):
#     X_eq[k] = Delta_s * k
#     X_eq[N+k] = 3 * (X_eq[k]/L)**2 - (X_eq[k]/L)**3
#     X_eq[N+k] = X_eq[N+k] * F*(L**3) / 6 / E_b
# For a uniform vertical force
# Lambda = Lambdas[0][1]
# for k in range(N):
#     X_eq[k] = Delta_s * k
#     X_eq[N+k] = (X_eq[k]/L)**4 - 4*(X_eq[k]/L)**3 + 6*(X_eq[k]/L)**2
#     X_eq[N+k] = X_eq[N+k] * Lambda*(L**4) / 24 / E_b
# For a uniform small vertical flow
# U = 10**(-6)
# for k in range(N):
#     X_eq[k] = Delta_s * k
#     X_eq[N+k] = (X_eq[k]/L)**4 - 4*(X_eq[k]/L)**3 + 6*(X_eq[k]/L)**2
#     X_eq[N+k] = X_eq[N+k] * U * xi *(L**4) / 24 / E_b

fig_one_set = go.Figure()
# new_trace_eq = go.Scatter(x = X_eq[:N], y = X_eq[N:], name = "Analytical solution", line = dict(color=red))
# fig_one_set.add_trace(new_trace_eq)
for t in range(X.shape[0]):
    X_3N_t = X3N(X[t,:], Delta_s)
    if t==0 or t==X.shape[0]-1:
        new_trace_t = go.Scatter(x=X_3N_t[:N,0], y=X_3N_t[N:2*N,0], name="t = " + str(T_eval[t]), line=dict(color=black))
        force_trace_t = go.Scatter(x=X_3N_t[N//2:N//2+2,0], y=X_3N_t[N + N//2:N + N//2+2,0], name="force application at t = " + str(T_eval[t]), line=dict(color=red))
        fig_one_set.add_trace(new_trace_t)
        fig_one_set.add_trace(force_trace_t)
    else:
        new_trace_t = go.Scatter(x=X_3N_t[:N,0], y=X_3N_t[N:2*N,0], name="t = " + str(T_eval[t]), visible="legendonly", line=dict(color=orange))
        force_trace_t = go.Scatter(x=X_3N_t[N//2:N//2+2,0], y=X_3N_t[N + N//2:N + N//2+2,0], name="force application at t = " + str(T_eval[t]), visible="legendonly", line=dict(color=red))
        fig_one_set.add_trace(new_trace_t)
        fig_one_set.add_trace(force_trace_t)


fig_one_set.update_layout(
    legend_title_text='Beam shapes',
    title = "Beam shapes at different times",
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

# Kinetic energy calculation
X_dot = np.zeros((len(T_eval), N+2))
K = np.zeros((len(T_eval))) # Kinetic energy
left_K = np.zeros((len(T_eval)))
for t_index in range(1, len(T_eval)):
    t = T_eval[t_index]
    X_dot[t_index, :] = (X[t_index,:] - X[t_index-1, :]) / (T_eval[t_index] - T_eval[t_index-1]) 

    K[t_index] = np.linalg.norm(X_dot[t_index, 2:])**2 / 2
    left_K[t_index] = np.linalg.norm(X_dot[t_index, 2:2+N//2])**2 / 2

# Shear torque calculation
S = np.zeros((len(T_eval), N+2))
B = np.zeros((len(T_eval), N+2))
M = np.zeros((len(T_eval), N+2))
sum_S = np.zeros((len(T_eval)))
sum_B = np.zeros((len(T_eval)))
sum_M = np.zeros((len(T_eval)))
left_S = np.zeros((len(T_eval)))
left_B = np.zeros((len(T_eval)))
left_M = np.zeros((len(T_eval)))


for t in range(X.shape[0]):
    X_3N_t = X3N(X[t,:], Delta_s)
    B[t, :] = BB(X_3N_t, K_b, Delta_s)[:,0]
    sum_B[t] = np.sum(B[t,:])
    left_B[t] = np.sum(B[t,2:2+N//2])
    S[t, :] = BS(X_3N_t, K_s, Delta_s)[:,0]
    sum_S[t] = np.sum(S[t,:])
    left_S[t] = np.sum(S[t,2:2+N//2])
    M[t,:] = BF(X_3N_t, Delta_s, Lambdas)[:,0]
    sum_M[t] = np.sum(M[t,:])
    left_M[t] = np.sum(M[t,2:2+N//2])

fig_X_dot = go.Figure()
K_trace = go.Scatter(x = T_eval, y = K, line=dict(color=orange), name ="Total kinetic energy")
fig_X_dot.add_trace(K_trace)
left_K_trace = go.Scatter(x = T_eval, y = left_K, line=dict(color=purple), name="Kinetic energy on left side")
fig_X_dot.add_trace(left_K_trace)
right_K_trace = go.Scatter(x = T_eval, y = K-left_K, line=dict(color=blue), name="Kinetic energy on right side")
fig_X_dot.add_trace(right_K_trace)

M_trace = go.Scatter(x = T_eval, y = sum_M, line=dict(color=orange), name ="Total torque")
fig_X_dot.add_trace(M_trace)
left_M_trace = go.Scatter(x = T_eval, y = left_M, line=dict(color=purple), name="Torque on left side")
fig_X_dot.add_trace(left_M_trace)
right_M_trace = go.Scatter(x = T_eval, y = sum_M-left_M, line=dict(color=blue), name="Torque on right side")
fig_X_dot.add_trace(right_M_trace)

B_trace = go.Scatter(x = T_eval, y = sum_B, line=dict(color=orange), name ="Total bending moment")
fig_X_dot.add_trace(B_trace)
left_B_trace = go.Scatter(x = T_eval, y = left_B, line=dict(color=purple), name="Bending moment on left side")
fig_X_dot.add_trace(left_B_trace)
right_B_trace = go.Scatter(x = T_eval, y = sum_B-left_B, line=dict(color=blue), name="Bending moment on right side")
fig_X_dot.add_trace(right_B_trace)

S_trace = go.Scatter(x = T_eval, y = sum_S, line=dict(color=orange), name ="Total shear moment")
fig_X_dot.add_trace(S_trace)
left_S_trace = go.Scatter(x = T_eval, y = left_S, line=dict(color=purple), name="Shear moment on left side")
fig_X_dot.add_trace(left_S_trace)
right_S_trace = go.Scatter(x = T_eval, y = sum_S-left_S, line=dict(color=blue), name="Shear moment on right side")
fig_X_dot.add_trace(right_S_trace)


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
    title = "Kinetic energy over time",
    xaxis_title = "Time",
    yaxis_title = "Kinetic energy"
)

fig_X_dot.show()

## --- dY/dt --- ##
###################

##################
## --- Y(t) --- ##

# Simulation relaxation vector ( i.e., tip deflection over time)
Y_tip = np.array([X3N(X[t,:], Delta_s)[2*N-1,0] for t in range(len(T_eval))])
# Analytical relaxation solution
print("r = ", r, "L = ", L, "E_b = ", E_b)
tau = 11 * np.pi * mu * L**4 / (60 * E_b * np.log(L/(4*r)) )
tau_new = 11*np.pi * mu * L**4 / (30 * E_b * (np.log(L/r) + 0.193))
print("tau = ", tau, "tau_new = ", tau_new, "tau/tau_new = ", tau/tau_new)
Y_t = np.zeros((len(T_eval),1))
Y_t[0,0] = Y_tip[0]
for t_index in range(1,len(T_eval)):
    Y_t[t_index,0] = Y_t[0,0]*np.exp(-T_eval[t_index] / tau)

fig_Y_t = go.Figure()
Y_tip_trace = go.Scatter(x = T_eval, y = Y_tip, line=dict(color=orange), name ="Simulated tip deflection over time")
fig_Y_t.add_trace(Y_tip_trace)
Y_t_analytical_trace = go.Scatter(x = T_eval, y = Y_t[:,0], line=dict(color=blue), name ="Analytical tip deflection over time")
fig_Y_t.add_trace(Y_t_analytical_trace)
fig_Y_t.update_layout(
    title = "Tip deflection over time during relaxation after bending",
    xaxis_title = "Time",
    yaxis_title = "Tip deflection"
)
fig_Y_t.show()

## --- Y(t) --- ##
##################

### ----- Visualisation ----- ###
#################################

############################################
### ----- Drag - bending timescale ----- ###

def Linear_Regression(X_train, Y_train):
    reg = linear_model.LinearRegression().fit(X_train, Y_train)
    Y_pred = reg.intercept_ + reg.coef_ * X_train
    print("Mean squared error: %.2f" % mean_squared_error(Y_train, Y_pred))

    return reg.coef_

# T_eval_reg = np.array(T_eval).reshape(-1, 1)
# a = Linear_Regression(T_eval_reg, np.log(Y_tip))
# tau = -1 /a
# print("Slope = ", a, "and tau = ", tau, "for N = ", N)

tau5 = 0.3878805
tau10 = 0.30547176
tau15 = 0.283917
Tau = [tau5, tau10, tau15]
NN = [5, 10, 15]
fig_tau_N = px.scatter(x=NN, y=Tau)
fig_tau_N.add_trace(go.Scatter(x=NN, y=[tau]*len(NN), mode ='lines'))
fig_tau_N.show()

### ----- Drag - bending timescale ----- ###
############################################


#######################
### ----- PCA ----- ###

Theta = np.reshape(np.array([X3N(X[t,:], Delta_s)[2*N:] for t in range(len(T_eval))]), newshape = (len(T_eval), N)) #Kymograph matrix
# Kymograph = px.imshow(Theta, labels=dict(x="Arclength", y="Time", color="Tangent angle"))
# Kymograph.show()

# Compute mean (careful when angles go over 2pi!!)
Theta_0 = np.repeat(np.reshape(np.mean(Theta, axis = 0), newshape=(1,Theta.shape[1])), Theta.shape[0], axis=0)
# mean_Kymograph = px.imshow(Theta_0, labels=dict(x="Arclength", y="Time", color="Mean Tangent angle"))
# mean_Kymograph.show()

# Center data to the mean
Delta_Theta = Theta - Theta_0
Delta_Kymograph = px.imshow(Delta_Theta, labels=dict(x="Arclength", y="Time", color="Centered Tangent angle"))
Delta_Kymograph.show()

# Make covariance matrix

C = np.transpose(Delta_Theta) @ Delta_Theta
Covariance_figure = px.imshow(C, labels=dict(x="Arclength", y="Arclength", color="Covariance"))
Covariance_figure.show()

# Eigenvalue spectrum

w, v = np.linalg.eig(C)
w = np.array([eigenvalue for eigenvalue in w if np.isreal(eigenvalue)])
Eigenspectrum = px.scatter(x=np.arange(w.shape[0]), y=np.sort(np.real(w))[::-1], title="Eigenvalues of covariance matrix")
Eigenspectrum.show()

# Principal component extraction



exit()

# X[t,i]
pca_0 = PCA(n_components=4)
pca_0.fit(X_PCA[:,:])
print(pca_0.explained_variance_ratio_)

