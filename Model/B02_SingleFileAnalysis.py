################################################################################
""" This file will be used to analyze a single simulation. """
################################################################################
### Libraries

import sys
plot_functions_folder = "C:/Users/Luc/Documents/MEGAsync/Code"
sys.path.insert(0, plot_functions_folder)

import multiprocessing
from datetime import datetime
from A01_Coarse_grained_axoneme_functions import *
from B01_simulations_analysis import *

import numpy as np
from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

from plotting_functions import * 
pio.templates.default = "custom_template"
cyclic_color = ['Twilight', 'IceFire', 'Edge', 'Phase', 'HSV', 'mrybm', 'mygbm'][3]
diverging_color = [reverted_Tealrose, 'Fall', 'Geyser', 'Temps', 'Tealrose', 'Tropic'][0]
################################################################################

temp_folder = "C:/Users/Luc/Documents/MEGAsync/PhD/RheoFlag/Results/Temp/"

################################################################################
### Read metadata and data

folder_name = "C:/Users/Luc/Documents/PhD_Large_files/RheoFlag/Model/Output/" 
# folder_name += "AnalyticalComparisons/PureBending_Clamped_Relaxation/"
# folder_name += "AnalyticalComparisons/PureBending_Clamped_UniformVerticalFlow/"    
# folder_name += "AnalyticalComparisons/PureBending_Clamped_TipVerticalPointForce/"
# folder_name += "ProximalBend_NoFlow/BendingElasticity_Clamped_VaryingShearBending/"
# folder_name += "StraightLine_PeriodicFlow/PureBending_Clamped_NoViscosity/"
folder_name += "StraightLine_PeriodicFlow/BendingShear_Clamped_NoViscosity/"

################################################################################
id_filename = "20250324-054729163116_N_10_bool_tau_s_False_taus_b_0_Beta_1000.0_gamma_2_A_1.0_w0_1.0_Sp4_1.0_k0_1000000000000.0"
################################################################################

metadata_filename = folder_name + 'metadata_' + id_filename +'.json'
data_filename = folder_name + 'data_' + id_filename + '.csv'

solver_dict = get_metadata(metadata_filename)
output_folder, N, taus_b, bool_tau_s, init_conf, Beta, gamma, n_L, m_L, A, w0, Sp4, k0, Lambdas, Zetas, X_flow_field_string, T_span, T_eval, T_sim_max, T_sim, X_flow_field, X_0, method = list(solver_dict.values())

X = get_data(data_filename) # s, t
X_3N_final = X3N(X[:,-1])

if T_sim == np.inf:
    print('Not solved. Error: ', X)
    exit()

tau_b = taus_b[0]

keys_toprint = ['N', 'init_conf', 'A', 'w0', 'Sp4', 'k0', 'Beta', 'taus_b', 'bool_tau_s', 'n_L', 'm_L', 'Lambdas', 'Zetas']
for key in keys_toprint:
    print(key, solver_dict[key])

################################################################################
################################################################################
### Visualisation

##################################
### ----- Shape analysis ----- ###
T_eval = np.array(T_eval)
if (A > 0) & (w0 > 0):
    T_eval_norm = T_eval * w0 / (2*np.pi)
else:
    T_eval_norm = T_eval
X_flow = A*np.sin(2 * np.pi * T_eval_norm)

# # Fourier of the flow
# X_flow_f = np.fft.rfft(X_flow, axis=0)
# f = np.fft.rfftfreq(X_flow.shape[0])
# f /= T_eval_norm[1] - T_eval_norm[0]

# fig = go.Figure()
# fig.add_scatter(x = f, y = np.abs(X_flow_f))
# fig.vs_show()

# time.sleep(1)
# fig = go.Figure()
# fig.add_scatter(x = f, y = np.angle(X_flow_f) / (2 * np.pi))
# fig.vs_show()

# exit()

# Crop part of the data
# n_max = 101
# X = X[:, :n_max]
# T_eval_norm = T_eval_norm[:n_max]
# c = sample_colorscale('BuPu', np.linspace(0, 1, num = T_eval_norm.shape[0]))[::-1]

# fig = go.Figure()
# for t in range(X.shape[1]):
    
#     fig.add_scatter(x = X3N(X[:,t])[:N, 0], y = X3N(X[:,t])[N:2*N, 0], marker_color = c[t])

# fig.update_layout(showlegend = True)
# fig.vs_show()

# time.sleep(1)

# fig = go.Figure()
# fig.add_scatter(x = np.arange(T_eval_norm.shape[0]), y = T_eval_norm)
# fig.vs_show()
# exit()

# Stroboscopic view
eps = 1/1e-12 # -1, np.inf
if np.abs(w0 - 0) < eps:
    # Static view: divided all dynamics in n_strobes points equally distant
    n_strobes = T_eval_norm.shape[0]//2 #10000
    condition = (T_eval_norm >= T_eval_norm[-1]/2)
else:
    # Dynamic view: divided permanent regime in n_strobes points equally distant within one flow period
    n_strobes = 101
    condition = (T_eval_norm >= round(T_eval_norm[-1]) - 2) & (T_eval_norm <= round(T_eval_norm[-1]) - 1)
    # condition = (T_eval_norm >= 0) & (T_eval_norm <= 1)

indices_s = StroboscopicView(T_eval_norm[condition], n_strobes = n_strobes)
c = sample_colorscale('BuPu', np.linspace(0, 1, num = indices_s.shape[0]))[::-1]

# Kinetic energy
# K = KineticEnergy(X, N, T_eval_norm) # t
# fig = go.Figure()
# for k in range(indices_s.shape[0]):
#     fig.add_scatter(x = [T_eval_norm[condition][indices_s[k]]], y = [K[condition][indices_s[k]]], marker_color = c[k], mode = 'markers')
# fig.update_xaxes(type = 'linear')
# fig.update_yaxes(type = 'log')
# fig.vs_show()

# time.sleep(1)

# fig = go.Figure()
# for k in range(indices_s.shape[0]):
#     fig.add_scatter(x = X3N(X[:, condition][:,indices_s[k]])[:N, 0], y = X3N(X[:, condition][:,indices_s[k]])[N:2*N, 0], marker_color = c[k])
# # fig.add_scatter(x = X_3N_eq[:n_eq,0], y = X_3N_eq[n_eq:2*n_eq,0], marker_color = cb_orange)
# # fig.update_xaxes()
# # fig.update_yaxes()
# fig.update_layout(showlegend = True)
# fig.vs_show()

# time.sleep(1)

################
## Kymographs ##
################

# Kymograph for alpha
Alpha = np.transpose(X[2:, condition][:,indices_s]) # t, s
fig = go.Figure(data = go.Heatmap(
    x = T_eval_norm[condition][indices_s],
    y = np.linspace(start = 0, stop = 1, num = Alpha.shape[1]),
    z = np.transpose(Alpha),
    colorscale = 'BuPu',
    ))

fig.update_yaxes(title = 's')
fig.update_xaxes(title = 'w0 * t if w0>0, t otherwise')
fig.update_layout(title = "alpha")
fig.vs_show()

time.sleep(1)

# Kymograph for theta
Theta = Kymograph(X[:, condition][:,indices_s]) # t, s

fig = go.Figure(data = go.Heatmap(
    x = T_eval_norm[condition][indices_s],
    y = np.linspace(start = 0, stop = 1, num = Theta.shape[1]),
    z = np.transpose(Theta),
    colorscale = 'BuPu',
    ))

fig.update_yaxes(title = 's')
fig.update_xaxes(title = 'w0 * t if w0>0, t otherwise')
fig.update_layout(title = "theta")
fig.vs_show()

time.sleep(1)

# Spatial Fourier
Alpha_q, q_alpha = SpatialFourier(np.transpose(Alpha)) # Alpha_q: q, t
Theta_q, q_theta = SpatialFourier(np.transpose(Theta)) # Theta_q: q, t

delta_s = 1/N
q_alpha *= 1/delta_s
q_theta *= 1/delta_s

trace = [go.Heatmap(
    x= T_eval_norm[condition][indices_s],
    y= q_alpha,
    z= np.abs(Alpha_q),
    colorscale='BuPu',
    )]
layout = go.Layout(
    title = 'Spatial Spectrogram of Alpha',
    yaxis = dict(title = 'Wave number'), # y-axis label
    xaxis = dict(title = 'Time [# Flow periods]'), # x-axis label
    )
fig = go.Figure(data=trace, layout=layout)
fig.vs_show()

time.sleep(1)

trace = [go.Heatmap(
    x= T_eval_norm[condition][indices_s],
    y= q_theta,
    z= np.abs(Theta_q),
    colorscale='BuPu',
    )]
layout = go.Layout(
    title = 'Spatial Spectrogram of Theta',
    yaxis = dict(title = 'Wave number'), # y-axis label
    xaxis = dict(title = 'Time [# Flow periods]'), # x-axis label
    )
fig = go.Figure(data=trace, layout=layout)
fig.vs_show()

time.sleep(1)

# Temporal Fourier
Alpha_f, f_alpha = TemporalFourier(np.transpose(Alpha)) # Alpha_f: s, f
Theta_f, f_theta = TemporalFourier(np.transpose(Theta)) # Theta_f: s, f

delta_t_norm = T_eval_norm[condition][indices_s[1]] - T_eval_norm[condition][indices_s[0]]
print("Delta_t = ", delta_t_norm)
f_alpha *= 1/delta_t_norm
f_theta *= 1/delta_t_norm

trace = [go.Heatmap(
    x= f_alpha,
    y= np.linspace(0, 1, num = Alpha_f.shape[0]),
    z= np.abs(Alpha_f),
    colorscale='BuPu',
    )]
layout = go.Layout(
    title = 'abs(Alpha(s,f))',
    yaxis = dict(title = 'Arclength'),
    xaxis = dict(title = 'Frequency [in flow frequency units]'), 
    )
fig = go.Figure(data=trace, layout=layout)
fig.vs_show()

time.sleep(1)

trace = [go.Heatmap(
    x= f_alpha,
    y= np.linspace(0, 1, num = Alpha_f.shape[0]),
    z= np.angle(Alpha_f) / (2 * np.pi),
    colorscale=diverging_color,
    )]
layout = go.Layout(
    title = 'phi(Alpha(s, f))',
    yaxis = dict(title = 'Arclength'),
    xaxis = dict(title = 'Frequency [in flow frequency units]'), 
    )
fig = go.Figure(data=trace, layout=layout)
fig.vs_show()

time.sleep(1)

trace = [go.Heatmap(
    x= f_theta,
    y= np.linspace(0, 1, num = Theta_f.shape[0]),
    z= np.abs(Theta_f),
    colorscale='BuPu',
    )]
layout = go.Layout(
    title = 'abs(Theta(s,f))',
    yaxis = dict(title = 'Arclength'),
    xaxis = dict(title = 'Frequency [in flow frequency units]'), 
    )
fig = go.Figure(data=trace, layout=layout)
fig.vs_show()

time.sleep(1)

trace = [go.Heatmap(
    x= f_theta,
    y= np.linspace(0, 1, num = Theta_f.shape[0]),
    z= np.angle(Theta_f)/(2*np.pi),
    colorscale=diverging_color,
    )]
layout = go.Layout(
    title = 'phi(Theta(s,f))',
    yaxis = dict(title = 'Arclength'),
    xaxis = dict(title = 'Frequency [in flow frequency units]'), 
    )
fig = go.Figure(data=trace, layout=layout)
fig.vs_show()

time.sleep(1)

# (Spatial + Temporal) Fourier

Alpha_q, q_alpha = SpatialFourier(np.transpose(Alpha)) # Alpha_q: q, t
Theta_q, q_theta = SpatialFourier(np.transpose(Theta)) # Theta_q: q, t
Alpha_fq, f_alpha = TemporalFourier(Alpha_q) # q, f
Theta_fq, f_theta = TemporalFourier(Theta_q) # q, f

# print("indices_s", indices_s[:50])
delta_t_norm = T_eval_norm[condition][indices_s[1]] - T_eval_norm[condition][indices_s[0]]
# print("delta_t_norm: ", delta_t_norm)
f_alpha *= 1/delta_t_norm
f_theta *= 1/delta_t_norm

# Plots in (q, f)
trace = [go.Heatmap(
    x= f_alpha,
    y= q_alpha,
    z= np.log(np.abs(Alpha_fq)),
    colorscale='BuPu',
    )]
layout = go.Layout(
    title = 'log|F_t(F_s(Alpha))|',
    yaxis = dict(title = 'Wave number'), # y-axis label
    xaxis = dict(title = 'Frequency (in flow frequency units)'), # x-axis label
    )
fig = go.Figure(data=trace, layout=layout)
fig.vs_show()

time.sleep(1)

# Plots in (q, f) 

# Modulus
trace = [go.Heatmap(
    x= f_theta,
    y= q_theta,
    z= np.log(np.abs(Theta_fq)),
    colorscale='BuPu',
    )]
layout = go.Layout(
    title = 'log|F_t(F_s(Theta))|',
    yaxis = dict(title = 'Wave number'), # y-axis label
    xaxis = dict(title = 'Frequency (in flow frequency units)'), # x-axis label
    )
fig = go.Figure(data=trace, layout=layout)
fig.vs_show()

time.sleep(1)

# Phase
trace = [go.Heatmap(
    x= f_alpha,
    y= q_alpha,
    z= np.angle(Alpha_fq)/(2*np.pi),
    colorscale=diverging_color,
    )]
layout = go.Layout(
    title = 'phi(F_t(F_s(Alpha)))',
    yaxis = dict(title = 'Wave number'), # y-axis label
    xaxis = dict(title = 'Frequency (in flow frequency units)'), # x-axis label
    )
fig = go.Figure(data=trace, layout=layout)
fig.vs_show()

time.sleep(1)
# Phase
trace = [go.Heatmap(
    x= f_theta,
    y= q_theta,
    z= np.angle(Theta_fq)/(2*np.pi),
    colorscale=diverging_color,
    )]
layout = go.Layout(
    title = 'phi(F_t(F_s(Theta)))',
    yaxis = dict(title = 'Wave number'), # y-axis label
    xaxis = dict(title = 'Frequency (in flow frequency units)'), # x-axis label
    )
fig = go.Figure(data=trace, layout=layout)
fig.vs_show()

# stroboscopic Animation with flow
# fig_shape = AnimatedShape(X[:, condition][:,indices_s], X_flow[indices_s], N, w0, Sp4, Beta, tau_b, T_eval_norm[indices_s])
# fig_shape.show()

# PCA
# psi = np.pi/2
# P, Lambda = PCA(Theta, bool_from_scratch=False)
# print("Explained variance of first and second PCA components: ", Lambda[0] / np.sum(Lambda), Lambda[1] / np.sum(Lambda))
# fig_eigenspectrum.vs_show()

### ----- Shape analysis ----- ###
##################################



