""" This file aims at analyzing a single simulation of a viscoelastic filament. 
In particular, one can visualize the waveform in space-time as a video. """

# from audioop import mul
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
# periodic_pure_bending_filename = "data_20240710-072407387487.dat"
# periodic_bending_all_filename = "data_20240710-072612423428.dat"

filename = "data_20250311-034227331518.dat"
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

### ----- Read data from file ----- ###
#######################################

##############################################################
### ----- Parameter choice from the chosen data file ----- ###

# Empty #

### ----- Parameter choice from the chosen data file ----- ###
##############################################################


#################################
### ----- Visualisation ----- ###

#############################
## --- Plot parameters --- ##


## --- Plot parameters --- ##
#############################

##################################
### ----- Shape analysis ----- ###
T_eval = np.array(T_eval)
X_flow = A*np.sin(w0*T_eval)

# Animated shape
fig_shape = AnimatedShape(X, X_flow, N, w0, Sp4, Beta, tau_b, T_eval)
fig_shape.show()

# Kymograph
Theta, Theta_0, fig_kymograph = Kymograph(X, True)
# Popping out the first angle, fixed because of clamped BC
Theta = Theta[:,1:]
Theta_0 = Theta_0[:,1:]
print("Kymograph shape: ", Theta.shape) # Should be M x (N-1) now
fig_kymograph.show()

# PCA
psi = np.pi/2
P, Lambda, fig_eigenspectrum = PCA(Theta, False, True)
print("Explained variance of first and second PCA components: ", Lambda[0] / np.sum(Lambda), Lambda[1] / np.sum(Lambda))
fig_eigenspectrum.show()

# Phase between PCA and flow
PCA_phase, polar_coeffs_ellipse, figs_ellipse = PCA_vs_Flow(Theta, Theta_0, P, X_flow, 1, True)
# figs_ellipse[0].show()

# Spatial Fourier
# Xq, spatial_modes, fig_spatial_modes = SpatialFourier(X, N, T_eval, w0, True)
# fig_spatial_modes.show()
Theta_q, spatial_modes, fig_spatial_modes = SpatialFourier(np.transpose(Theta), T_eval, w0, True)
# fig_spatial_modes.show()

# Phases between Spatial Fourier and flow
# Fourier_phases, polar_coeffs_ellipses, fig_ellipses = SpatialFourier_vs_Flow(Theta_q, spatial_modes, X_flow, w0, Theta_q.shape[0], True)
# fig_ellipses.show()
# print(Fourier_phases)

# Spectrogram heat map: x-axis is time, y-axis is frequency, z-axis (color) is the power
# f, bins, Pxx = scipy.signal.spectrogram(Theta[:,0], 1, axis=0)
# # Plot with plotly
# trace = [go.Heatmap(
#     x= bins,
#     y= f,
#     z= 10*np.log10(Pxx),
#     colorscale='Jet',
#     )]
# layout = go.Layout(
#     title = 'Spectrogram with plotly',
#     yaxis = dict(title = 'Frequency'), # y-axis label
#     xaxis = dict(title = 'Time'), # x-axis label
#     )
# fig = go.Figure(data=trace, layout=layout)
# fig.show()

Pxx_array = np.zeros((spatial_modes.shape[0], T_eval.shape[0]))
for t in range(len(T_eval)):
    q, bins, Pxx = scipy.signal.spectrogram(Theta[t,:], 1, axis=0)
    Pxx_array[:,t] = Pxx.reshape(spatial_modes.shape[0],)
# print("Pxx_array.shape = ", Pxx_array.shape)
# print("Spatial modes shape = ", spatial_modes.shape)
# # Plot with plotly
trace = [go.Heatmap(
    x= T_eval * w0 / (2*np.pi),
    y= spatial_modes,
    z= Pxx_array, #10*np.log10(Pxx_array),
    colorscale='Jet',
    )]
layout = go.Layout(
    title = 'Spatial Spectrogram over time',
    yaxis = dict(title = 'Wave number'), # y-axis label
    xaxis = dict(title = 'Time [# Flow periods]'), # x-axis label
    )
fig = go.Figure(data=trace, layout=layout)
fig.show()


# exit()

### ----- Shape analysis ----- ###
##################################



