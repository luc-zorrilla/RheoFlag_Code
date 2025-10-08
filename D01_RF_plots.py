""" This file will be used to verify adequation of the model to analytical results. 
Paper-quality figures will also be made to illustrate that. """

################################################################################
### Libraries

import os
import sys
from pathlib import Path
plot_functions_folder = Path.home() / 'Work' / 'Miscellaneous' / 'Code'
sys.path.insert(0, plot_functions_folder)
import glob

import multiprocessing
from datetime import datetime
from A01_Coarse_grained_axoneme_functions import *
from B01_simulations_analysis import *

import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy import special

from plotting_functions import * 
pio.templates.default = "figure_template"

# Linear Fit
def d_exp(t, tau, A):
    return A*np.exp(-t/tau)
################################################################################

temp_folder = Path.cwd().joinpath('Model').joinpath('Results').joinpath('Temp')
writing_dir = temp_folder

fig_nbr = 14
panel_nbr = 0

################################
# Model chapter - benchmarking #
################################ 
# Figure 0: analytical results for pure bending, clamped axoneme, with point force at the tip
    # Panel a - stroboscopic view of the filament along with analytical solution
    # Panel b - kinetic energy vs time, showing equilibrium is reached
    # Panel c - L2(Equilibrium - Analytical solution) vs 1/N, showing convergence to the solution

# Figure 1: analytical results for pure bending, clamped axoneme, with uniform vertical flow
    # Panel a - stroboscopic view of the filament along with analytical solution
    # Panel b - kinetic energy vs time, showing equilibrium is reached
    # Panel c - L2(Equilibrium - Analytical solution) vs 1/N, showing convergence to the solution

# Figure 2: simulations for shear elasticity + bending elasticity, no viscosity, clamped axoneme
    # Panel a - stroboscopic view of the filament in 3 different regimes of shear / bending.

# Figure 3: simulations for shear elasticity + bending elasticity, no viscosity, clamped axoneme
    # Panel a - Counterbend for two different regimes

##############################
# Model chapter - relaxation #
##############################

# Figure 4: simulations for bending elasticity + bending viscosity, clamped axoneme
    # Panel a - relaxation for varying bending viscosity
    # Panel b - fit timescale against internal bending timescale

# Figure 5: simulations for shear elasticity + shear viscosity, clamped axoneme
    # Panel a - relaxation for varying shear viscosities
    # Panel b - fit timescale against internal shear timescale

#####################################
# Model chapter - harmonic response #
#####################################

# Figure 6: simulations for a periodic flow, for a clamped axoneme with bending elasticity + bending viscosity
    # Panel a - tip movement for varying bending viscosity and flow frequency: phase
    # Panel b - tip movement for varying bending viscosity and flow frequency: max amplitude
    # Panels c and d: trials with varying amplitude. Not useful here. 

# Figure 7: simulations for a periodic flow, for a clamped axoneme with bending elasticity + bending viscosity
    # Panel a - transect for tau_b (<<, >>) tau_{b,f} with phase and max amplitude

# Figure 8: simulations for a periodic flow, for a clamped axoneme with shear elasticity + shear viscosity
    # Panel a - tip movement for varying shear viscosity and flow frequency: phase
    # Panel b - tip movement for varying shear viscosity and flow frequency: max amplitude

# Figure 9: simulations for a periodic flow, for a clamped axoneme with shear elasticity + shear viscosity
    # Panel a - transects for tau_s (<<, >>) tau_{s,f} with phase and max amplitude

#########################################
# Inference chapter - harmonic response #
#########################################

# Figure 10 - simulations for a periodic flow, clamped axoneme with bending (elasticity + viscosity)
    # Panel a - phase diagrams

# Figure 11 - simulations for a periodic flow, clamped axoneme with bending (elasticity + viscosity)
    # Panel a - phase response

# Figure 12 - simulations for a periodic flow, clamped axoneme with bending (elasticity + viscosity)
    # Panel a - amplitude response

# Figure 13 -  simulations for a periodic flow, clamped axoneme with bending (elasticity + viscosity)
    # Panel a - 3 errors
    # Panel b - inset for best error

# Figure 14 - simulations for a periodic flow, clamped axoneme with shear (elasticity + viscosity)
    # Panel a - phase diagrams

# Figure 15 - simulations for a periodic flow, clamped axoneme with shear (elasticity + viscosity)
    # Panel a - phase response

# Figure 16 - simulations for a periodic flow, clamped axoneme with shear (elasticity + viscosity)
    # Panel a - amplitude response

# Figure 17 - simulations for a periodic flow, clamped axoneme with shear (elasticity + viscosity)
    # Panel a - 3 errors
    # Panel b - inset for best error

# Figure 18 - bending axoneme, varying bending viscosity
    # Panel a - k(B, w0) heatmap with inverted analytical solution - 3 plots at varying viscosities
    # Panel b - gamma(B, w0) heatmap with inverted analytical solution - 3 plots at varying viscosities

# Figure 19 - bending axoneme, varying bending viscosity
    # Panel a - k(B, w0) heatmap with inverted analytical solution - 3 plots at varying viscosities
    # Panel b - gamma(B, w0) heatmap with inverted analytical solution - 3 plots at varying viscosities

if __name__ == '__main__':

    fig_filename = writing_dir + "fig" + "_" + str(fig_nbr) + "_" + "panel" + "_" + str(panel_nbr) + ".pdf"

    # Pure Bending, vertical point force at the tip
    if fig_nbr == 0:

        folder_name = Path('..').joinpath('Model').joinpath('Output').resolve()
        folder_name /= "AnalyticalComparisons"
        folder_name /= "PureBending_Clamped_TipVerticalPointForce"

        id_filenames = ["20250415-120922564262_N_5_tau_s_0_taus_b_0_Beta_0_gamma_2_A_0_w0_0_Sp4_1.0_k0_10000000000.0", "20250415-120922601564_N_10_tau_s_0_taus_b_0_Beta_0_gamma_2_A_0_w0_0_Sp4_1.0_k0_10000000000.0", "20250415-120922621337_N_15_tau_s_0_taus_b_0_Beta_0_gamma_2_A_0_w0_0_Sp4_1.0_k0_10000000000.0", "20250415-120922659220_N_20_tau_s_0_taus_b_0_Beta_0_gamma_2_A_0_w0_0_Sp4_1.0_k0_10000000000.0", "20250415-120922662966_N_25_tau_s_0_taus_b_0_Beta_0_gamma_2_A_0_w0_0_Sp4_1.0_k0_10000000000.0", "20250415-120922679084_N_30_tau_s_0_taus_b_0_Beta_0_gamma_2_A_0_w0_0_Sp4_1.0_k0_10000000000.0", "20250415-120922716650_N_35_tau_s_0_taus_b_0_Beta_0_gamma_2_A_0_w0_0_Sp4_1.0_k0_10000000000.0", "20250415-120922805353_N_40_tau_s_0_taus_b_0_Beta_0_gamma_2_A_0_w0_0_Sp4_1.0_k0_10000000000.0", "20250415-122526334494_N_45_tau_s_0_taus_b_0_Beta_0_gamma_2_A_0_w0_0_Sp4_1.0_k0_10000000000.0"]

        # Equilibrium solution - stroboscopic view and analytical solution (N = 35) + kinetic energy
        if panel_nbr in [0, 1]:

            id_filename = id_filenames[-1]
            metadata_filename = folder_name + 'metadata_' + id_filename +'.json'
            data_filename = folder_name + 'data_' + id_filename + '.csv'
            solver_dict = get_metadata(metadata_filename) 
            output_folder, N, taus_b, tau_s, init_conf, Beta, gamma, n_L, m_L, A, w0, Sp4, k0, Lambdas, Zetas, X_flow_field_string, T_span, T_eval, T_sim_max, T_sim, X_flow_field, X_0, method = list(solver_dict.values()) # bool_EI is missing in this metadata
            X = get_data(data_filename) # s, t 
            X_3N_final = X3N(X[:,-1])

            T_eval = np.array(T_eval)
            if (A > 0) & (w0 > 0):
                T_eval_norm = T_eval * w0 / (2*np.pi)
            else:
                T_eval_norm = T_eval
            X_flow = A*np.sin(w0*T_eval)            

            # Stroboscopic view
            n_strobes = 50

            condition = (T_eval_norm >= 0)
            min_index = np.arange(T_eval_norm.shape[0])[condition][0]
            max_index = np.arange(T_eval_norm.shape[0])[condition][-1]
            indices_s = StroboscopicView(T_eval_norm[min_index:max_index], n_strobes=n_strobes)
            c = sample_colorscale('matter_r', np.linspace(0, 1, num = indices_s.shape[0]))[::-1]

            # Analytical Equilibrium Profile
            if panel_nbr == 0:
                
                n_eq = 1000
                X_3N_eq = CheckEquilibrium(N, A, gamma, Sp4, n_L = n_L, Lambdas=Lambdas, conditions = "vertical_point_tip", n_eq = n_eq)

                fig = go.Figure()
                fig.add_scatter(x = X3N(X[:,0])[:N, 0]/N, y = X3N(X[:,0])[N:2*N, 0], marker_color = "black", name = r"$\huge{\boldsymbol{y}_0}$", line_width = 6)
                fig.add_scatter(x = X3N(X[:,-1])[:N, 0]/N, y = X3N(X[:,-1])[N:2*N, 0], marker_color = cb_orange, name = r"$\huge{\boldsymbol{y}_\text{eq}}$", line_dash = "dash", line_width = 6)
                fig.add_scatter(x = X_3N_eq[:n_eq,0][X_3N_eq[:n_eq,0]<=N-1]/N, y = X_3N_eq[n_eq:2*n_eq,0][X_3N_eq[:n_eq,0]<=N-1], marker_color = cb_dark_purple, name = r"$\huge{\boldsymbol{y}^\star}$", line_dash = "dot", line_width = 6)

                x_ticks = np.round(np.linspace(0,1,11),2)
                x_ticks_text = [r"$\huge{" + str(x_tick) + "}$" for x_tick in x_ticks]
                fig.update_xaxes(
                    title = r'$\huge{X(s)/N}$', 
                    range = [0, 1], 
                    tickmode = 'array',
                    tickvals = x_ticks,
                    ticktext = x_ticks_text
                    )
                y_ticks = np.round(np.linspace(0,3e-2,7),3)
                y_ticks_text = [r"$\huge{" + str(y_tick) + "}$" for y_tick in y_ticks]
                fig.update_yaxes(
                    title = r'$\huge{Y(s)}$',
                    range = [-1e-3,3e-2],
                    tickmode = 'array',
                    tickvals = y_ticks,
                    ticktext = y_ticks_text
                    )
                fig.update_layout(
                    showlegend = True, 
                    legend = dict(
                        xref = 'paper',
                        yref = 'paper',
                        xanchor = 'left', 
                        yanchor = 'top',
                        x = 0.1, 
                        y = 0.8,
                        itemwidth = 50,
                        bgcolor= 'rgba(0,0,0,0)',
                        ),
                    margin = dict(
                        l = 200,
                        r = 200,
                        b = 200,
                        t = 200,
                        pad = 0,
                        ),
                    width = 1000 + 200 + 200,
                    height = 500 + 200 + 200,
                    )

            # Convergence to equilibrium - kinetic energy decays (log-log plot)
            elif panel_nbr == 1:
                
                # Kinetic energy
                K = KineticEnergy(X, N, T_eval) # t
                fig = go.Figure()
                fig.add_scatter(x = T_eval, y = K, line_width = 6)

                x_ticks = 10 ** np.arange(10) + 1e-9 # needed to add 1e-3 due to approximation in sci_notation.
                x_ticks_text = [r"$\huge{" + sci_notation(x_tick, -1, -1) + "}$" for x_tick in x_ticks] 
                x_minor_ticks = np.hstack([np.arange(10) * x_tick for x_tick in x_ticks])
                fig.update_xaxes(
                    title = r'$\huge{t}$', 
                    type = 'log',
                    range = [3, 7], 
                    tickmode = 'array',
                    tickvals = x_ticks,
                    ticktext = x_ticks_text,
                    minor=dict(ticks = "outside", ticklen = 6, tickwidth = 3, tickvals = x_minor_ticks),
                    )
                y_ticks = np.float_power(10, np.arange(-16, -31, -1))
                y_ticks_text = [r"$\huge{" + sci_notation(y_tick, -1) + "}$" for y_tick in y_ticks]
                for k in range(len(y_ticks_text)):
                    if k % 2 == 1:
                        y_ticks_text[k] = r"$\huge{}$" # Get rid of half the tick text
                y_minor_ticks = np.hstack([np.arange(10) * y_tick for y_tick in y_ticks])
                fig.update_yaxes(
                    title = r'$\huge{K(t)}$',
                    type = 'log',
                    range = [-30,-16],
                    tickmode = 'array',
                    tickvals = y_ticks,
                    ticktext = y_ticks_text,
                    minor=dict(ticks = "outside", ticklen = 6, tickwidth = 3, tickvals = y_minor_ticks),
                    )
                fig.update_layout(
                    showlegend = False,
                    margin = dict(
                        l = 200,
                        r = 200,
                        b = 200,
                        t = 200,
                        pad = 0,
                        ),
                    width = 500 + 200 + 200,
                    height = 500 + 200 + 200,
                    )

        # L2 Error (Solution - analytical solution) for varying N (1/N, log(relative L2 error))
        elif panel_nbr == 2:

            L2_error_array = np.zeros((len(id_filenames)))
            for l in range(len(id_filenames)):
                id_filename = id_filenames[l]
                metadata_filename = folder_name + 'metadata_' + id_filename +'.json'
                data_filename = folder_name + 'data_' + id_filename + '.csv'
                solver_dict = get_metadata(metadata_filename)
                output_folder, N, taus_b, tau_s, init_conf, Beta, gamma, n_L, m_L, A, w0, Sp4, k0, Lambdas, Zetas, X_flow_field_string, T_span, T_eval, T_sim_max, T_sim, X_flow_field, X_0, method = list(solver_dict.values()) # bool_EI is not present in this metadata
                tau_b = taus_b[0]
                if T_sim == np.inf:
                    print('Not solved. Error: ', X)
                    exit()

                # Equilibrium profile
                X = get_data(data_filename) # s, t
                X_3N_final = X3N(X[:,-1])

                # Analytical Equilbrium Profile
                X_3N_eq = CheckEquilibrium(N, A, gamma, Sp4, n_L = n_L, Lambdas=Lambdas, conditions = "vertical_flow_uniform", n_eq = N)

                step = N // 5
                L2_error = np.linalg.norm(X_3N_final[::step] - X_3N_eq[::step]) / np.linalg.norm(X_3N_final[::step])
                L2_error_array[l] = L2_error

            fig = go.Figure()
            fig.add_scatter(x = 1 / np.arange(5, 50, 5), y = L2_error_array, mode = 'markers', marker_color = cb_dark_red, marker_size = 12)

            x_range = np.log10(np.array([0.02, 0.12]))
            x_ticks = np.array([1,2,3,4,5,6,7,8,9,10,20]) * 1e-2
            x_ticks_text = [r"$\huge{" + sci_notation(x_tick, 0, 0, -2) + "}$" for x_tick in x_ticks]
            fig.update_xaxes(
                title = r'$\huge{1/N}$', 
                type = 'log',
                range = x_range, 
                tickmode = 'array',
                tickvals = x_ticks,
                ticktext = x_ticks_text,
                )
            y_range = np.log10(np.array([0.02, 0.12]))
            y_ticks = np.array([1,2,3,4,5,6,7,8,9,10,20]) * 1e-2
            y_ticks_text = [r"$\huge{" + sci_notation(y_tick, 0, 0, -2) + "}$" for y_tick in y_ticks]
            fig.update_yaxes(
                title = r'$\huge{\epsilon}$',
                type = 'log',
                range = y_range,
                tickmode = 'array',
                tickvals = y_ticks,
                ticktext = y_ticks_text,
                )
            fig.update_layout(
                showlegend = False,
                margin = dict(
                    l = 200,
                    r = 200,
                    b = 200,
                    t = 200,
                    pad = 0,
                    ),
                width = 500 + 200 + 200,
                height = 500 + 200 + 200,
                )

    # Pure Bending, uniform vertical flow.
    elif fig_nbr == 1:

        folder_name = Path('..').joinpath('Model').joinpath('Output').resolve()
        folder_name /= "AnalyticalComparisons"
        folder_name /= "PureBending_Clamped_UniformVerticalFlow"

        id_filenames = ["20250415-111946800826_N_5_tau_s_0_taus_b_0_Beta_0_gamma_2_A_1e-08_w0_0_Sp4_1.0_k0_10000000000.0", "20250415-111946825696_N_10_tau_s_0_taus_b_0_Beta_0_gamma_2_A_1e-08_w0_0_Sp4_1.0_k0_10000000000.0", "20250415-111946863118_N_15_tau_s_0_taus_b_0_Beta_0_gamma_2_A_1e-08_w0_0_Sp4_1.0_k0_10000000000.0", "20250415-111946894785_N_20_tau_s_0_taus_b_0_Beta_0_gamma_2_A_1e-08_w0_0_Sp4_1.0_k0_10000000000.0", "20250415-111946920750_N_25_tau_s_0_taus_b_0_Beta_0_gamma_2_A_1e-08_w0_0_Sp4_1.0_k0_10000000000.0", "20250415-111946929903_N_30_tau_s_0_taus_b_0_Beta_0_gamma_2_A_1e-08_w0_0_Sp4_1.0_k0_10000000000.0", "20250415-111946930900_N_35_tau_s_0_taus_b_0_Beta_0_gamma_2_A_1e-08_w0_0_Sp4_1.0_k0_10000000000.0", "20250415-112025796860_N_40_tau_s_0_taus_b_0_Beta_0_gamma_2_A_1e-08_w0_0_Sp4_1.0_k0_10000000000.0", "20250415-112025855976_N_45_tau_s_0_taus_b_0_Beta_0_gamma_2_A_1e-08_w0_0_Sp4_1.0_k0_10000000000.0", "20250415-112025990187_N_50_tau_s_0_taus_b_0_Beta_0_gamma_2_A_1e-08_w0_0_Sp4_1.0_k0_10000000000.0", "20250415-112025990187_N_55_tau_s_0_taus_b_0_Beta_0_gamma_2_A_1e-08_w0_0_Sp4_1.0_k0_10000000000.0"]

        # Equilibrium solution - stroboscopic view and analytical solution (N = 35) + kinetic energy
        if panel_nbr in [0, 1]:

            id_filename = id_filenames[-1]
            metadata_filename = folder_name + 'metadata_' + id_filename +'.json'
            data_filename = folder_name + 'data_' + id_filename + '.csv'
            solver_dict = get_metadata(metadata_filename)
            output_folder, N, taus_b, tau_s, init_conf, Beta, gamma, n_L, m_L, A, w0, Sp4, k0, Lambdas, Zetas, X_flow_field_string, T_span, T_eval, T_sim_max, T_sim, X_flow_field, X_0, method = list(solver_dict.values()) # bool_EI is not present in this metadata
            X = get_data(data_filename) # s, t
            X_3N_final = X3N(X[:,-1])

            T_eval = np.array(T_eval)
            if (A > 0) & (w0 > 0):
                T_eval_norm = T_eval * w0 / (2*np.pi)
            else:
                T_eval_norm = T_eval
            X_flow = A*np.sin(w0*T_eval)            

            # Stroboscopic view
            n_strobes = 50

            condition = (T_eval_norm >= 0)
            min_index = np.arange(T_eval_norm.shape[0])[condition][0]
            max_index = np.arange(T_eval_norm.shape[0])[condition][-1]
            indices_s = StroboscopicView(T_eval_norm[min_index:max_index], n_strobes = n_strobes)
            c = sample_colorscale('matter_r', np.linspace(0, 1, num = indices_s.shape[0]))[::-1]

            # Analytical Equilbrium Profile
            if panel_nbr == 0:
                
                n_eq = 1000
                X_3N_eq = CheckEquilibrium(N, A, gamma, Sp4, n_L = n_L, Lambdas=Lambdas, conditions = "vertical_flow_uniform", n_eq = n_eq)

                fig = go.Figure()
                fig.add_scatter(x = X3N(X[:,0])[:N, 0]/N, y = X3N(X[:,0])[N:2*N, 0], marker_color = "black", line_width = 6, name = r"$\huge{\boldsymbol{y}_0}$")
                fig.add_scatter(x = X3N(X[:,-1])[:N, -1]/N, y = X3N(X[:,-1])[N:2*N, -1], marker_color = cb_orange, line_width = 6, name = r"$\huge{\boldsymbol{y}_\text{eq}}$", line_dash = 'dash')                
                fig.add_scatter(x = X_3N_eq[:n_eq,0][X_3N_eq[:n_eq,0]<=N-1]/N, y = X_3N_eq[n_eq:2*n_eq,0][X_3N_eq[:n_eq,0]<=N-1], marker_color = cb_dark_purple, line_width = 6, name = r"$\huge{\boldsymbol{y}^\star}$", line_dash = 'dot')

                x_ticks = np.round(np.linspace(0,1,11),2)
                x_ticks_text = [r"$\huge{" + str(x_tick) + "}$" for x_tick in x_ticks]
                fig.update_xaxes(
                    title = r'$\huge{X(s)/N}$', 
                    range = [0, 1], 
                    tickmode = 'array',
                    tickvals = x_ticks,
                    ticktext = x_ticks_text
                    )
                y_ticks = np.round(np.linspace(0,3e-2,7),3)
                y_ticks_text = [r"$\huge{" + str(y_tick) + "}$" for y_tick in y_ticks]
                fig.update_yaxes(
                    title = r'$\huge{Y(s)}$',
                    range = [-1e-3,3e-2],
                    tickmode = 'array',
                    tickvals = y_ticks,
                    ticktext = y_ticks_text
                    )
                fig.update_layout(
                    showlegend = True, 
                    legend = dict(
                        xref = 'paper',
                        yref = 'paper',
                        xanchor = 'left', 
                        yanchor = 'top',
                        x = 0.1, 
                        y = 0.8,
                        itemwidth = 50,
                        bgcolor= 'rgba(0,0,0,0)',
                        ),
                    margin = dict(
                        l = 200,
                        r = 200,
                        b = 200,
                        t = 200,
                        pad = 0,
                        ),
                    width = 1000 + 200 + 200,
                    height = 500 + 200 + 200,
                    )

            # Convergence to equilibrium - kinetic energy decays (log-log plot)
            elif panel_nbr == 1:
                
                # Kinetic energy
                K = KineticEnergy(X, N, T_eval) # t
                fig = go.Figure()
                fig.add_scatter(x = T_eval, y = K, line_width = 6)

                x_ticks = 10 ** np.arange(10) + 1e-9 # needed to add 1e-3 due to approximation in sci_notation.
                x_ticks_text = [r"$\huge{" + sci_notation(x_tick, -1, -1) + "}$" for x_tick in x_ticks] 
                x_minor_ticks = np.hstack([np.arange(10) * x_tick for x_tick in x_ticks])
                fig.update_xaxes(
                    title = r'$\huge{t}$', 
                    type = 'log',
                    range = [3, 7], 
                    tickmode = 'array',
                    tickvals = x_ticks,
                    ticktext = x_ticks_text,
                    minor=dict(ticks = "outside", ticklen = 6, tickwidth = 3, tickvals = x_minor_ticks),
                    )
                y_ticks = np.float_power(10, np.arange(-16, -31, -1))
                y_ticks_text = [r"$\huge{" + sci_notation(y_tick, -1) + "}$" for y_tick in y_ticks]
                for k in range(len(y_ticks_text)):
                    if k % 2 == 1:
                        y_ticks_text[k] = r"$\huge{}$" # Get rid of half the tick text
                y_minor_ticks = np.hstack([np.arange(10) * y_tick for y_tick in y_ticks])
                fig.update_yaxes(
                    title = r'$\huge{K(t)}$',
                    type = 'log',
                    range = [-27,-18],
                    tickmode = 'array',
                    tickvals = y_ticks,
                    ticktext = y_ticks_text,
                    minor=dict(ticks = "outside", ticklen = 6, tickwidth = 3, tickvals = y_minor_ticks),
                    )
                fig.update_layout(
                    showlegend = False,
                    margin = dict(
                        l = 200,
                        r = 200,
                        b = 200,
                        t = 200,
                        pad = 0,
                        ),
                    width = 500 + 200 + 200,
                    height = 500 + 200 + 200,
                    )

        # L2 Error (Solution - analytical solution) for varying N (1/N, log(relative L2 error))
        elif panel_nbr == 2:

            L2_error_array = np.zeros((len(id_filenames)))
            for l in range(len(id_filenames)):
                id_filename = id_filenames[l]
                metadata_filename = folder_name + 'metadata_' + id_filename +'.json'
                data_filename = folder_name + 'data_' + id_filename + '.csv'

                solver_dict = get_metadata(metadata_filename)
                output_folder, N, taus_b, tau_s, init_conf, Beta, gamma, n_L, m_L, A, w0, Sp4, k0, Lambdas, Zetas, X_flow_field_string, T_span, T_eval, T_sim_max, T_sim, X_flow_field, X_0, method = list(solver_dict.values()) # bool_EI is not present in this metadata
                tau_b = taus_b[0]
                if T_sim == np.inf:
                    print('Not solved. Error: ', X)
                    exit()

                # Equilibrium profile
                X = get_data(data_filename) # s, t
                X_3N_final = X3N(X[:,-1])

                # Analytical Equilbrium Profile
                X_3N_eq = CheckEquilibrium(N, A, gamma, Sp4, n_L = n_L, Lambdas=Lambdas, conditions = "vertical_flow_uniform", n_eq = N)

                step = N // 5
                L2_error = np.linalg.norm(X_3N_final[::step] - X_3N_eq[::step]) / np.linalg.norm(X_3N_final[::step])
                L2_error_array[l] = L2_error

            fig = go.Figure()
            fig.add_scatter(x = 1/np.arange(5, 60, 5), y = L2_error_array, mode = 'markers', marker_color = cb_dark_red, marker_size = 12)

            x_range = np.log10(np.array([0.017, 0.12]))
            x_ticks = np.array([1,2,3,4,5,6,7,8,9,10,20]) * 1e-2
            x_ticks_text = [r"$\huge{" + sci_notation(x_tick, 0, 0, -2) + "}$" for x_tick in x_ticks]
            fig.update_xaxes(
                title = r'$\huge{1/N}$', 
                type = 'log',
                range = x_range, 
                tickmode = 'array',
                tickvals = x_ticks,
                ticktext = x_ticks_text,
                )
            y_range = np.log10(np.array([0.017, 0.12]))
            y_ticks = np.array([1,2,3,4,5,6,7,8,9,10,20]) * 1e-2
            y_ticks_text = [r"$\huge{" + sci_notation(y_tick, 0, 0, -2) + "}$" for y_tick in y_ticks]
            fig.update_yaxes(
                title = r'$\huge{\epsilon}$',
                type = 'log',
                range = y_range,
                tickmode = 'array',
                tickvals = y_ticks,
                ticktext = y_ticks_text,
                )
            fig.update_layout(
                showlegend = False,
                margin = dict(
                    l = 200,
                    r = 200,
                    b = 200,
                    t = 200,
                    pad = 0,
                    ),
                width = 500 + 200 + 200,
                height = 500 + 200 + 200,
                )

    # Bending + Shear elasticity, no viscosity, clamped axoneme - stroboscope
    elif fig_nbr == 2:

        if panel_nbr == 0:

            folder_name = Path('..').joinpath('Model').joinpath('Output').resolve()
            folder_name /= "ProximalBend_NoFlow"
            folder_name /= "BendingElasticity_Clamped_VaryingShearBending"

            id_filenames = ["20250416-052857586460_N_10_tau_s_0_taus_b_0_Beta_0.001_gamma_2_A_0_w0_0_Sp4_1.0_k0_10000000000.0", "20250416-052952250898_N_10_tau_s_0_taus_b_0_Beta_0.1_gamma_2_A_0_w0_0_Sp4_1.0_k0_10000000000.0", "20250416-054051471468_N_10_tau_s_0_taus_b_0_Beta_1.0_gamma_2_A_0_w0_0_Sp4_1.0_k0_10000000000.0", "20250416-053616003820_N_10_tau_s_0_taus_b_0_Beta_1000.0_gamma_2_A_0_w0_0_Sp4_1.0_k0_10000000000.0"] 
            

            fig = make_subplots(rows = len(id_filenames), cols = 1, subplot_titles = [r"$\huge{\beta = 10^{-3}}$", r"$\huge{\beta = 10^{-1}}$", r"$\huge{\beta = 1}$", r"$\huge{\beta = 10^3}$"], shared_xaxes=True)

            for l in range(len(id_filenames)):

                id_filename = id_filenames[l]
                metadata_filename = folder_name + 'metadata_' + id_filename +'.json'
                data_filename = folder_name + 'data_' + id_filename + '.csv'
                solver_dict = get_metadata(metadata_filename)
                output_folder, N, taus_b, tau_s, init_conf, bool_EI, Beta, gamma, n_L, m_L, A, w0, Sp4, k0, Lambdas, Zetas, X_flow_field_string, T_span, T_eval, T_sim_max, T_sim, X_flow_field, X_0, method = list(solver_dict.values())
                X = get_data(data_filename) # s, t
                X_3N_final = X3N(X[:,-1])

                T_eval = np.array(T_eval)
                if (A > 0) & (w0 > 0):
                    T_eval_norm = T_eval * w0 / (2*np.pi)
                else:
                    T_eval_norm = T_eval
                X_flow = A*np.sin(w0*T_eval)            

                # Stroboscopic view
                n_strobes = 20
                t_s = T_eval[-1] / n_strobes                
                
                condition = (T_eval_norm >= 0)
                min_index = np.arange(T_eval_norm.shape[0])[condition][0]
                max_index = np.arange(T_eval_norm.shape[0])[condition][-1]
                indices_s = StroboscopicView(T_eval_norm[min_index:max_index], n_strobes = n_strobes)
                c = sample_colorscale(colorscale = dark_purple_scale[::-1], samplepoints = np.linspace(0, 1, num = indices_s.shape[0]))[::-1]        

                for k in range(indices_s.shape[0]):
                    fig.add_scatter(x = X3N(X[:,indices_s[k]])[:N, 0], y = X3N(X[:,indices_s[k]])[N:2*N, 0], marker_color = c[k], row = 1 + l, col = 1)
                # fig.update_xaxes()
                # fig.update_yaxes()
            fig.update_layout(width = 800, height = 300 * len(id_filenames), showlegend = False)

    # Bending + Shear elasticity, no viscosity, clamped axoneme - counterbend
    elif fig_nbr == 3:

        if panel_nbr == 0:
            folder_name = Path('..').joinpath('Model').joinpath('Output').resolve()       
            folder_name /= "ProximalBend_NoFlow"
            folder_name /= "BendingShearElasticity_Clamped_Counterbend"
            id_filenames = ["20250416-012526174965_N_10_tau_s_0_taus_b_0_Beta_1.0_gamma_2_A_0_w0_0_Sp4_1.0_k0_10000000000.0","20250416-011925928066_N_10_tau_s_0_taus_b_0_Beta_0.1_gamma_2_A_0_w0_0_Sp4_1.0_k0_10000000000.0"]

            fig = make_subplots(rows = len(id_filenames), cols = 1, subplot_titles = [r"$\huge{\beta = 1, \lambda = 2}$", r"$\huge{\beta = 0.1, \lambda = 0.5}$"], shared_xaxes=True)

            for l in range(len(id_filenames)):

                id_filename = id_filenames[l]
                metadata_filename = folder_name + 'metadata_' + id_filename +'.json'
                data_filename = folder_name + 'data_' + id_filename + '.csv'
                solver_dict = get_metadata(metadata_filename)
                output_folder, N, taus_b, tau_s, init_conf, bool_EI, Beta, gamma, n_L, m_L, A, w0, Sp4, k0, Lambdas, Zetas, X_flow_field_string, T_span, T_eval, T_sim_max, T_sim, X_flow_field, X_0, method = list(solver_dict.values())
                X = get_data(data_filename) # s, t
                X_3N_final = X3N(X[:,-1])

                T_eval = np.array(T_eval)
                if (A > 0) & (w0 > 0):
                    T_eval_norm = T_eval * w0 / (2*np.pi)
                else:
                    T_eval_norm = T_eval
                X_flow = A*np.sin(w0*T_eval)            

                # Stroboscopic view
                n_strobes = 5
                t_s = T_eval[-1] / n_strobes                
                
                condition = (T_eval_norm >= 0)
                min_index = np.arange(T_eval_norm.shape[0])[condition][0]
                max_index = np.arange(T_eval_norm.shape[0])[condition][-1]
                indices_s = StroboscopicView(T_eval_norm[min_index:max_index], n_strobes = n_strobes)
                c = sample_colorscale(dark_purple_scale, np.linspace(0, 1, num = indices_s.shape[0]))[::-1]        

                for k in range(indices_s.shape[0]):
                    fig.add_scatter(x = X3N(X[:,indices_s[k]])[:N, 0], y = X3N(X[:,indices_s[k]])[N:2*N, 0], marker_color = c[k], row = 1 + l, col = 1)
                # fig.update_xaxes()
                # fig.update_yaxes()
            fig.update_layout(width = 800, height = 300 * len(id_filenames), showlegend = False)

    # Bending elasticity, varying bending viscosity, clamped axoneme - relaxation
    elif fig_nbr == 4:
        
        # Relaxation curves for varying bending viscosity
        if panel_nbr == 0:

            folder_name = Path('..').joinpath('Model').joinpath('Output').resolve()    
            folder_name /= "SecondBend_Relaxation"
            folder_name /= "BendingElasticity_Clamped_VaryingBendingViscosity/"

            filenames = glob.glob(folder_name + '*.json')
            id_filenames = [os.path.basename(filename).removeprefix("metadata_").removesuffix(".json") for filename in filenames]
            
            fig = go.Figure()

            for l in range(len(id_filenames)):

                id_filename = id_filenames[l]
                metadata_filename = folder_name + 'metadata_' + id_filename +'.json'
                data_filename = folder_name + 'data_' + id_filename + '.csv'
                solver_dict = get_metadata(metadata_filename)
                output_folder, N, taus_b, tau_s, init_conf, bool_EI, Beta, gamma, n_L, m_L, A, w0, Sp4, k0, Lambdas, Zetas, X_flow_field_string, T_span, T_eval, T_sim_max, T_sim, X_flow_field, X_0, method = list(solver_dict.values())                
                tau_b = taus_b[0]
                X = get_data(data_filename) # s, t
                X_3N_final = X3N(X[:,-1])

                delta_t = T_eval[1] - T_eval[0]
                T_eval = np.array(T_eval)
                if (A > 0) & (w0 > 0):
                    T_eval_norm = T_eval * w0 / (2*np.pi)
                else:
                    T_eval_norm = T_eval
                X_flow = A*np.sin(w0*T_eval)            

                # Stroboscopic view
                n_strobes = 20
                t_s = T_eval[-1] / n_strobes                
                
                condition = (T_eval_norm >= 0)
                min_index = np.arange(T_eval_norm.shape[0])[condition][0]
                max_index = np.arange(T_eval_norm.shape[0])[condition][-1]
                indices_s = StroboscopicView(T_eval_norm[min_index:max_index], n_strobes = n_strobes)
                c = sample_colorscale(colorscale = dark_purple_scale[::-1], samplepoints = np.linspace(0, 1, num = indices_s.shape[0]))[::-1]        

                # Figure
                if tau_b < 1e6:
                    marker_color = 'black'
                    xaxis_name = "x1"
                    if tau_b == 0:
                        line_dash = 'dash'
                    elif tau_b == 1:
                        line_dash = 'dot'
                    elif tau_b == 1e3:
                        line_dash = 'solid'
                else:
                    xaxis_name = "x2"        
                    marker_color = cb_orange
                    line_dash = 'solid'

                X_3N = np.array([X3N(X[:,t]) for t in range(X.shape[1])]).squeeze().transpose()
                x_tip = np.array([X_3N[N-1,:], X_3N[2*N-1,:]])
                fig.add_scatter(
                    x = np.arange(x_tip.shape[1])*delta_t, 
                    y = x_tip[1,:]/x_tip[1,0], 
                    name = r"$\huge{\tau_b = " + sci_notation(tau_b, 0,0) + "}$", 
                    marker_color = marker_color, 
                    xaxis = xaxis_name,
                    line_dash = line_dash,
                    line_width = 6,
                    )

                # Fit to exponentially decreasing function
                popt, pcov = curve_fit(f = d_exp, xdata = T_eval/delta_t, ydata = x_tip[1,:]/x_tip[1,0])
                popt[0] *= delta_t
                pcov[0,0] *= delta_t**2
                pcov[0,1] *= delta_t
                pcov[1,0] *= delta_t
                print("tau_b, popt, pcov: ", tau_b, popt, pcov)

            x_ticks_1 = np.arange(0,125,25)*1e2
            x_ticks_text_1 = [r"$\huge{" + sci_notation(x_tick_1, 1, 1) + "}$" for x_tick_1 in x_ticks_1]
            x_ticks_2 = np.arange(0,4,1)*1e6
            x_ticks_text_2 = [r"$\huge{" + sci_notation(x_tick_2, 0, 0) + "}$" for x_tick_2 in x_ticks_2]   
            fig.update_layout(
                xaxis = dict(
                    zeroline = True, 
                    title = r"$\huge{t}$",
                    side = 'bottom',
                    overlaying = 'x',
                    range = [0, 1e4],
                    tickmode = "array",
                    tickvals = x_ticks_1,
                    ticktext = x_ticks_text_1,
                ),
                xaxis2 = dict(
                    zeroline = True, 
                    title = r"$\huge{t}$",
                    side = 'top',
                    overlaying = 'x',                    
                    range = [0, 3e6],
                    tickmode = "array",
                    tickvals = x_ticks_2,
                    ticktext = x_ticks_text_2,
                    linecolor = cb_orange,
                    tickcolor = cb_orange,
                ),
                )             
            y_ticks = np.float_power(10, (np.arange(-2,1)))
            y_ticks_text = [r"$\huge{" + str(y_tick) + "}$" for y_tick in y_ticks]
            fig.update_yaxes(
                range = [-2,0.2],
                title = r"$\huge{y(t)/y(0)}$",
                type = 'log',
                tickmode = "array",
                tickvals = y_ticks,
                ticktext = y_ticks_text                  
            )            
            fig.update_layout(
                margin = dict(l = 200, r = 200, t = 200, b = 200),
                width = 300 + 400, 
                height = 300 + 400, 
                showlegend = True,
                )

        # Fit timescale against bending viscosity
        elif panel_nbr == 1:

            tau_b_list = [0, 1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6]
            tau_fit_list = [1.32370907e+03, 1.32917591e+03, 1.33100003e+03, 1.43848476e+03, 2.38949445e+03, 1.14066673e+04, 1.01144517e+05, 1.00163941e+06]
            delta_tau_fit_list = [9.91664631e-01, 1.06888231e+00, 1.17750208e+00, 1.89812829e+00, 1.07628131e+00, 1.60957857e+02, 4.27311246e+04, 2.06367158e+03]

            # Linear relation
            npoints = 1000
            x_tau = np.float_power(10, np.linspace(-0.2,6.2,npoints))

            # Constant at origin
            tau_f_b = tau_fit_list[0]
            y_tau_constant = np.ones_like(x_tau)*tau_f_b

            fig = go.Figure()
            fig.add_scatter(x = x_tau, y = x_tau, mode = 'lines', marker_color = cb_purple, name = r"$\huge{\tau = \tau_b}$", line_dash = 'dash', line_width = 6)
            fig.add_scatter(x = x_tau, y = y_tau_constant, mode = 'lines', marker_color = cb_blue, name = r"$\huge{\tau = \tau_{f,b}}$", line_dash = 'dash', line_width = 6)
            fig.add_scatter(x = x_tau, y = x_tau + y_tau_constant, mode = 'lines', marker_color = cb_red, name = r"$\huge{\tau = \tau_{f,b} + \tau_b}$", line_width = 6)
            fig.add_scatter(
                x = tau_b_list[1:],
                y = tau_fit_list[1:], 
                error_y = dict(type = 'data', visible = True, array = delta_tau_fit_list[1:]),
                marker_color = 'black',
                marker_size = 12,
                mode = 'markers',
                name = r"$\huge{\tau_\text{fit}}$",
                )

            x_ticks = np.float_power(10, np.arange(0,7,2))
            x_ticks_text = [r"$\huge{" + sci_notation(x_tick, 0, 0) + "}$" for x_tick in x_ticks]   
            fig.update_xaxes(
                type = 'log',
                range = [-0.1,6.1],
                tickmode = "array",
                tickvals = x_ticks,
                ticktext = x_ticks_text,
                title = r"$\huge{\tau_b}$",               
                ) 
            y_ticks = np.float_power(10, np.arange(0,7,1))
            y_ticks_text = [r"$\huge{" + sci_notation(y_tick, 0, 0) + "}$" for y_tick in y_ticks]                   
            fig.update_yaxes(
                type = 'log',
                range = [3,6.1],
                tickmode = "array",
                tickvals = y_ticks,
                ticktext = y_ticks_text,    
                )
            fig.update_layout(
                margin = dict(l = 400, r = 400, b = 400, t = 400),
                width = 300 + 800,
                height = 300 + 800,
            )

    # Shear elasticity, varying shear viscosity, clamped axoneme - relaxation
    elif fig_nbr == 5:

        # Relaxation curves for varying bending viscosity
        if panel_nbr == 0:

            folder_name = Path('..').joinpath('Model').joinpath('Output').resolve()
            folder_name /= "SecondBend_Relaxation"
            folder_name /= "ShearElasticity_Clamped_VaryingShearViscosity"

            filenames = glob.glob(folder_name + '*.json')
            id_filenames = [os.path.basename(filename).removeprefix("metadata_").removesuffix(".json") for filename in filenames]
            
            fig = go.Figure()

            for l in range(len(id_filenames)):

                id_filename = id_filenames[l]
                metadata_filename = folder_name + 'metadata_' + id_filename +'.json'
                data_filename = folder_name + 'data_' + id_filename + '.csv'
                solver_dict = get_metadata(metadata_filename)
                output_folder, N, taus_b, tau_s, init_conf, bool_EI, Beta, gamma, n_L, m_L, A, w0, Sp4, k0, Lambdas, Zetas, X_flow_field_string, T_span, T_eval, T_sim_max, T_sim, X_flow_field, X_0, method = list(solver_dict.values())                
                tau_b = taus_b[0]
                X = get_data(data_filename) # s, t
                X_3N_final = X3N(X[:,-1])

                delta_t = T_eval[1] - T_eval[0]
                T_eval = np.array(T_eval)
                if (A > 0) & (w0 > 0):
                    T_eval_norm = T_eval * w0 / (2*np.pi)
                else:
                    T_eval_norm = T_eval
                X_flow = A*np.sin(w0*T_eval)            

                # Stroboscopic view
                n_strobes = 20
                t_s = T_eval[-1] / n_strobes                
                
                condition = (T_eval_norm >= 0)
                min_index = np.arange(T_eval_norm.shape[0])[condition][0]
                max_index = np.arange(T_eval_norm.shape[0])[condition][-1]
                indices_s = StroboscopicView(T_eval_norm[min_index:max_index], n_strobes = n_strobes)
                c = sample_colorscale(colorscale = dark_purple_scale[::-1], samplepoints = np.linspace(0, 1, num = indices_s.shape[0]))[::-1]        

                # Figure
                if tau_s < 6e4:
                    marker_color = 'black'
                    xaxis_name = "x1"
                    if tau_s == 0:
                        line_dash = 'dash'
                    elif tau_s == 6e-2:
                        line_dash = 'dot'
                    elif tau_s == 6e1:
                        line_dash = 'solid'
                else:
                    xaxis_name = "x2"        
                    marker_color = cb_orange
                    line_dash = 'solid'

                X_3N = np.array([X3N(X[:,t]) for t in range(X.shape[1])]).squeeze().transpose()
                x_tip = np.array([X_3N[N-1,:], X_3N[2*N-1,:]])
                fig.add_scatter(
                    x = np.arange(x_tip.shape[1])*delta_t, 
                    y = x_tip[1,:]/x_tip[1,0], 
                    name = r"$\huge{\tau_s = " + sci_notation(tau_s, 0,0) + "}$", 
                    marker_color = marker_color, 
                    xaxis = xaxis_name,
                    line_dash = line_dash,
                    line_width = 6,
                    )

                # Fit to exponentially decreasing function
                popt, pcov = curve_fit(f = d_exp, xdata = T_eval/delta_t, ydata = x_tip[1,:]/x_tip[1,0])
                popt[0] *= delta_t
                pcov[0,0] *= delta_t**2
                pcov[0,1] *= delta_t
                pcov[1,0] *= delta_t
                print("tau_s, popt, pcov: ", tau_s, popt, pcov)

            x_ticks_1 = np.arange(0,75,15)*1e1
            x_ticks_text_1 = [r"$\huge{" + sci_notation(x_tick_1, 1, 1) + "}$" for x_tick_1 in x_ticks_1]
            x_ticks_2 = np.arange(0,25,5)*1e4
            x_ticks_text_2 = [r"$\huge{" + sci_notation(x_tick_2, 1, 1) + "}$" for x_tick_2 in x_ticks_2]
            fig.update_layout(
                xaxis = dict(
                    zeroline = True, 
                    title = r"$\huge{t}$",
                    side = 'bottom',
                    overlaying = 'x',
                    range = [0, 6e2],
                    tickmode = "array",
                    tickvals = x_ticks_1,
                    ticktext = x_ticks_text_1,
                ),
                xaxis2 = dict(
                    zeroline = True, 
                    title = r"$\huge{t}$",
                    side = 'top',
                    overlaying = 'x',                    
                    range = [0, 2e5],
                    tickmode = "array",
                    tickvals = x_ticks_2,
                    ticktext = x_ticks_text_2,
                    linecolor = cb_orange,
                    tickcolor = cb_orange,
                ),
                )             
            y_ticks = np.float_power(10, (np.arange(-2,1)))
            y_ticks_text = [r"$\huge{" + str(y_tick) + "}$" for y_tick in y_ticks]
            fig.update_yaxes(
                range = [-2,0.2],
                title = r"$\huge{y(t)/y(0)}$",
                type = 'log',
                tickmode = "array",
                tickvals = y_ticks,
                ticktext = y_ticks_text                  
            )            
            fig.update_layout(
                margin = dict(l = 400, r = 400, t = 400, b = 400),
                width = 300 + 800, 
                height = 300 + 800, 
                showlegend = True,
                )

        # Fit timescale against bending viscosity
        elif panel_nbr == 1:

            tau_s_list = [0, 6e-2, 6e-1, 6e0, 6e1, 6e2, 6e3, 6e4]
            tau_fit_list = [62.35186524, 62.39079242, 62.89711295, 67.5452019, 119.97670424, 659.71284223, 6.06251470e+03, 6.00903861e+04]
            delta_tau_fit_list = [1.49066945e-01, 1.47079188e-01, 1.46723593e-01, 1.16766543e-01, 2.20731722e-02, 1.12625598e-01, 8.84565313e+00, 8.49033480e+02]

            # Linear relation
            npoints = 1000
            x_tau = np.float_power(10, np.linspace(-3,5,npoints))

            # Constant at origin
            tau_f_s = tau_fit_list[0]
            y_tau_constant = np.ones_like(x_tau)*tau_f_s

            fig = go.Figure()
            fig.add_scatter(x = x_tau, y = x_tau, mode = 'lines', marker_color = cb_purple, name = r"$\huge{\tau = \tau_s}$", line_dash = 'dash', line_width = 6)
            fig.add_scatter(x = x_tau, y = y_tau_constant, mode = 'lines', marker_color = cb_blue, name = r"$\huge{\tau = \tau_{f,s}}$", line_dash = 'dash', line_width = 6)
            fig.add_scatter(x = x_tau, y = x_tau + y_tau_constant, mode = 'lines', marker_color = cb_red, name = r"$\huge{\tau = \tau_{f,s} + \tau_s}$", line_width = 6)
            fig.add_scatter(
                x = tau_s_list[1:],
                y = tau_fit_list[1:], 
                error_y = dict(type = 'data', visible = True, array = delta_tau_fit_list[1:]),
                marker_color = 'black',
                marker_size = 12,
                mode = 'markers',
                name = r"$\huge{\tau_\text{fit}}$",
                )

            x_ticks = np.float_power(10, np.arange(-3,7,2))
            x_ticks_text = [r"$\huge{" + sci_notation(x_tick, 0, 0) + "}$" for x_tick in x_ticks]   
            fig.update_xaxes(
                type = 'log',
                range = [np.log10(6e-2) - 0.1, np.log10(6e4) + 0.1],
                tickmode = "array",
                tickvals = x_ticks,
                ticktext = x_ticks_text,
                title = r"$\huge{\tau_s}$",               
                ) 
            y_ticks = np.float_power(10, np.arange(-3,7,1))
            y_ticks_text = [r"$\huge{" + sci_notation(y_tick, 0, 0) + "}$" for y_tick in y_ticks]                   
            fig.update_yaxes(
                type = 'log',
                range = [np.log10(6e1) - 0.1, np.log10(6e4) + 0.1],
                tickmode = "array",
                tickvals = y_ticks,
                ticktext = y_ticks_text,    
                )
            fig.update_layout(
                margin = dict(l = 400, r = 400, b = 400, t = 400),
                width = 300 + 800,
                height = 300 + 800,
            )

    # Bending elasticity, varying bending viscosity, flow frequency and amplitude; clamped axoneme
    elif fig_nbr == 6: # Heatmaps

        folder_name = Path('..').joinpath('Model').joinpath('Output').resolve()
        folder_name /= "StraightLine_PeriodicFlow"
        folder_name /= "BendingElasticity_Clamped_VaryingBendingViscosity"
        folder_name /= "VaryingFrequencyAmplitude"
        dataframe_filename = folder_name + "maxdev" + ".csv"
        N = 10

        df = pd.read_csv(dataframe_filename)

        # Select only for one value of the amplitude
        if panel_nbr in [0,1]:
            eps = 1e-6
            A = 1e-2
            df = df.loc[np.abs(df['A']- A) < eps]
            df['log_w0'] = df.apply(lambda x: x['log_w0']/np.log(10), axis = 1)

        df['log_tau_b_m1'] = df.apply(lambda x: x['log_tau_b_m1']/np.log(10), axis = 1)
        df['max_y_tip'] = df.apply(lambda x: x['max_y_tip']/N, axis = 1)
        df['log_max_y_tip'] = df.apply(lambda x: np.log10(x['max_y_tip']), axis = 1)
        
    
        if panel_nbr == 0: # Phase

            # # Plot f_tip and phi_tip against tau_b, w0
            fig = plot_heatmap(df, 'log_w0', 'log_tau_b_m1', 'phi_max_y_tip')
            fig.update_layout(
                xaxis_title = r"$\huge{\log{\omega_0}}$",
                yaxis_title = r"$\huge{\log{\tau_b^{-1}}}$", 
                coloraxis_colorbar = dict(
                    title = r"$\huge{\phi_\text{max}}$",
                    orientation = "v",
                    tickmode="array",
                    tickcolor = 'black',
                    tickvals = [0, 0.25, 0.5, 0.75, 1],
                    ticktext = ["0", "0.25", "0.5", "0.75", "1"],
                    ticks = "outside",
                    tickwidth = 3,
                    ticklen = 12,
                ),
                coloraxis_colorscale = dark_purple_scale,
                margin = dict(l = 200, r = 200, t = 200, b = 200),
                width = 800, height = 800)
            
        elif panel_nbr == 1: # max_y_tip

            fig = plot_heatmap(df, 'log_w0', 'log_tau_b_m1', 'max_y_tip')
            fig.update_layout(
                xaxis_title = r"$\huge{\log{\omega_0}}$",
                yaxis_title = r"$\huge{\log{\tau_b^{-1}}}$",
                coloraxis_colorbar = dict(
                    title = r"$\huge{\frac{\displaystyle y_\text{max}}{\displaystyle N}}$",
                    orientation = "v",
                    tickcolor = 'black',
                    ticks = "outside",
                    tickwidth = 3,
                    ticklen = 12,
                ),                    
                margin = dict(l = 200, r = 200, t = 200, b = 200),
                width = 800, height = 800)

        if panel_nbr in [2,3]:
            # Select only for one value of the amplitude: A = 1e-2
            eps = 1e-6
            w0 = 1e-2
            df = df.loc[np.abs(df['w0']- w0) < eps]
            df['log_A'] = df.apply(lambda x: x['log_A']/np.log(10), axis = 1)
    
        if panel_nbr == 2: # Phase

            # # Plot f_tip and phi_tip against tau_b, w0
            fig = plot_heatmap(df, 'log_A', 'log_tau_b_m1', 'phi_max_y_tip')
            fig.update_layout(
                xaxis_title = r"$\huge{\log{A}}$",
                yaxis_title = r"$\huge{\log{\tau_b^{-1}}}$",
                # coloraxis_cmin = 0.25,
                # coloraxis_cmax = 0.5,                
                coloraxis_colorbar = dict(
                    title = r"$\huge{\phi_\text{max}}$",
                    orientation = "h",
                    tickmode="array",
                    tickcolor = 'black',
                    tickvals = [0, 0.25, 0.5, 0.75, 1],
                    ticktext = ["0", "0.25", "0.5", "0.75", "1"],
                    ticks = "outside",
                    tickwidth = 3,
                    ticklen = 12,
                ),
                coloraxis_colorscale = dark_purple_scale,
                margin = dict(l = 200, r = 200, t = 200, b = 200),
                width = 800, height = 800)
            
        elif panel_nbr == 3: # max_y_tip

            fig = plot_heatmap(df, 'log_A', 'log_tau_b_m1', 'max_y_tip')
            fig.update_layout(
                xaxis_title = r"$\huge{\log{A}}$",
                yaxis_title = r"$\huge{\log{\tau_b^{-1}}}$",
                coloraxis_colorbar = dict(
                    title = r"$\huge{\frac{\displaystyle y_\text{max}}{N}}$",
                    orientation = "h",
                    # tickmode="array",
                    tickcolor = 'black',
                    # tickvals = [0, 0.25, 0.5, 0.75, 1],
                    # ticktext = ["0", "0.25", "0.5", "0.75", "1"],
                    ticks = "outside",
                    tickwidth = 3,
                    ticklen = 12,
                ),                    
                margin = dict(l = 200, r = 200, t = 200, b = 200),
                width = 800, height = 800)
        
    # Bending elasticity, varying bending viscosity, flow frequency and amplitude; clamped axoneme
    elif fig_nbr == 7:

        # 2 Transects at tau_b = cte
        if panel_nbr == 0:

            folder_name = Path('..').joinpath('Model').joinpath('Output').resolve()
            folder_name /= "StraightLine_PeriodicFlow"
            folder_name /= "BendingElasticity_Clamped_VaryingBendingViscosity"
            folder_name /= "VaryingFrequencyAmplitude"
            dataframe_filename = folder_name + "maxdev" + ".csv"

            df = pd.read_csv(dataframe_filename)

            # Select only for one value of the amplitude: A = 1e-2
            eps = 1e-12
            A = 1e-2
            df = df.loc[np.abs(df['A']- A) < eps]
            df['log_w0'] = df.apply(lambda x: x['log_w0']/np.log(10), axis = 1)
            df['log_tau_b_m1'] = df.apply(lambda x: x['log_tau_b_m1']/np.log(10), axis = 1)
            df['log_max_y_tip'] = df.apply(lambda x: np.log10(x['max_y_tip']), axis = 1)
            
            # Make transects dataframes
            tau_b_m1_small = 1e-6
            df_small = df.loc[np.abs(df['tau_b_m1']- tau_b_m1_small) < eps]
            tau_b_m1_large = 1e0
            df_large = df.loc[np.abs(df['tau_b_m1']- tau_b_m1_large) < eps]

            fig = make_subplots(
                rows = 2, cols = 1, 
                subplot_titles = [r"$\huge{\tau_b = 10^6}$", r"$\huge{\tau_b = 1}$"], 
                shared_xaxes = True,
                specs = [[{"secondary_y": True}], [{"secondary_y": True}]],
                )
            for k_df in range(2):
                df_k = [df_small, df_large][k_df]
                fig.add_scatter(x = df_k['log_w0'], y = df_k['phi_max_y_tip'], row = 1 + k_df, col = 1, mode = "markers", marker_color = "black", name = "phi_y_max", secondary_y = False)
                fig.add_scatter(x = df_k['log_w0'], y = np.log10(df_k['max_y_tip']/np.max(df_k['max_y_tip'])), row = 1 + k_df, col = 1, mode = "markers", marker_color = cb_orange, name = "y_max", secondary_y = True)
            
            x_ticks = np.arange(-9, 1, 3)
            x_ticks_text = [r"$\huge{" + str(x_tick) + "}$" for x_tick in x_ticks]            
            fig.update_xaxes(
                title = r"$\huge{\log{\omega_0}}$",
                range = [-9.1,0.1],
                tickmode = "array",
                tickvals = x_ticks,
                ticktext = x_ticks_text,
            )
            y_ticks = np.arange(250,600,125)/1000
            y_ticks_text = [r"$\huge{" + str(y_tick) + "}$" for y_tick in y_ticks]
            fig.update_yaxes(
                title = r"$\huge{\phi_\text{max}}$",
                range = [0.24, 0.51],
                secondary_y = False,
                tickmode = "array",
                tickvals = y_ticks,
                ticktext = y_ticks_text,
            )
            y_ticks = np.arange(-6,1,2)
            y_ticks_text = [r"$\huge{" + str(y_tick) + "}$" for y_tick in y_ticks]            
            fig.update_yaxes(
                title = r"$\huge{\log{y_\text{max}^\circ}}$",
                range = [-6.1, 0.1],
                secondary_y = True,
                tickmode = "array",
                tickvals = y_ticks,
                ticktext = y_ticks_text,
                row = 1,
                col = 1,
            )
            y_ticks = np.arange(-3,1,1)
            y_ticks_text = [r"$\huge{" + str(y_tick) + "}$" for y_tick in y_ticks]            
            fig.update_yaxes(
                title = r"$\huge{\log{y_\text{max}^\circ}}$",
                range = [-2.5, 0.1],
                secondary_y = True,
                tickmode = "array",
                tickvals = y_ticks,
                ticktext = y_ticks_text,
                row = 2,
                col = 1,
            )                           
            fig.update_layout(
                margin = dict(l = 200, r = 200, t = 200, b = 200),
                width = 400 + 400,
                height = 300*2 + 400,
                showlegend = False,
            )

    # Shear elasticity, varying shear viscosity and flow frequency, clamped axoneme
    elif fig_nbr == 8:

        folder_name = Path('..').joinpath('Model').joinpath('Output').resolve()
        folder_name /= "StraightLine_PeriodicFlow"
        folder_name /= "ShearElasticity_Clamped_VaryingShearViscosity"
        folder_name /= "VaryingFrequencyAmplitude/"
        dataframe_filename = folder_name + "maxdev" + ".csv"

        df = pd.read_csv(dataframe_filename)
        N = 10

        # Select only for one value of the amplitude: A = 1e-2
        eps = 1e-2
        A = 1e0
        df = df.loc[np.abs(df['A']- A) < eps]
        df['log_w0'] = df.apply(lambda x: x['log_w0']/np.log(10), axis = 1)
        df['log_tau_s_m1'] = df.apply(lambda x: x['log_tau_s_m1']/np.log(10), axis = 1)
        df['max_y_tip'] = df.apply(lambda x: x['max_y_tip']/N, axis = 1)
        df['log_max_y_tip'] = df.apply(lambda x: np.log10(x['max_y_tip']), axis = 1)

        if panel_nbr == 0:

            fig = plot_heatmap(df, 'log_w0', 'log_tau_s_m1', 'phi_max_y_tip')
            fig.update_layout(
                xaxis_title = r"$\huge{\log{\omega_0}}$",
                yaxis_title = r"$\huge{\log{\tau_s^{-1}}}$",
                # coloraxis_cmin = 0.25,
                # coloraxis_cmax = 0.5,
                coloraxis_colorbar = dict(
                    title = r"$\huge{\phi_\text{max}}$",
                    orientation = 'v',
                    tickmode="array",
                    tickcolor = 'black',
                    tickvals = [0, 0.25, 0.5, 0.75, 1],
                    ticktext = ["0", "0.25", "0.5", "0.75", "1"],
                    ticks = "outside",
                    tickwidth = 3,
                    ticklen = 12,
                ),
                coloraxis_colorscale = dark_purple_scale,
                margin = dict(l = 200, r = 200, t = 200, b = 200),
                width = 800, height = 800)

        elif panel_nbr == 1:

            fig = plot_heatmap(df, 'log_w0', 'log_tau_s_m1', 'max_y_tip')
            fig.update_layout(
                xaxis_title = r"$\huge{\log{\omega_0}}$",
                yaxis_title = r"$\huge{\log{\tau_s^{-1}}}$",
                coloraxis_colorbar = dict(
                    title = r"$\huge{\frac{\displaystyle y_\text{max}}{\displaystyle N}}$",
                    orientation = "v",
                    # tickmode="array",
                    tickcolor = 'black',
                    # tickvals = [0, 0.25, 0.5, 0.75, 1],
                    # ticktext = ["0", "0.25", "0.5", "0.75", "1"],
                    ticks = "outside",
                    tickwidth = 3,
                    ticklen = 12,
                ),                
                margin = dict(l = 200, r = 200, t = 200, b = 200),
                width = 800, height = 800)    

    # Shear elasticity, varying shear viscosity and flow frequency, clamped axoneme
    elif fig_nbr == 9:

        # 2 Transects at tau_s = cte
        if panel_nbr == 0:

            folder_name = Path('..').joinpath('Model').joinpath('Output').resolve()
            folder_name /= "StraightLine_PeriodicFlow"
            folder_name /= "ShearElasticity_Clamped_VaryingShearViscosity"
            folder_name /= "VaryingFrequencyAmplitude"
            dataframe_filename = folder_name + "maxdev" + ".csv"

            df = pd.read_csv(dataframe_filename)

            # Select only for one value of the amplitude: A = 1e-2
            eps = 1e-5
            A = 1e0
            df = df.loc[np.abs(df['A']- A) < eps]
            df['log_w0'] = df.apply(lambda x: x['log_w0']/np.log(10), axis = 1)
            df['log_tau_s_m1'] = df.apply(lambda x: x['log_tau_s_m1']/np.log(10), axis = 1)
            df['log_max_y_tip'] = df.apply(lambda x: np.log10(x['max_y_tip']), axis = 1)
            
            # Make transects dataframes
            tau_s_m1_small = 1e-4
            df_small = df.loc[np.abs(df['tau_s_m1']- tau_s_m1_small) < eps]
            tau_s_m1_large = 1e1
            df_large = df.loc[np.abs(df['tau_s_m1']- tau_s_m1_large) < eps]

            fig = make_subplots(
                rows = 2, cols = 1, 
                subplot_titles = [r"$\huge{\tau_s = 10^{-4}}$", r"$\huge{\tau_s = 10}}$"], 
                shared_xaxes = True,
                specs = [[{"secondary_y": True}], [{"secondary_y": True}]],
                )
            for k_df in range(2):
                df_k = [df_small, df_large][k_df]
                fig.add_scatter(x = df_k['log_w0'], y = df_k['phi_max_y_tip'], row = 1 + k_df, col = 1, mode = "markers", marker_color = "black", name = "phi_y_max", secondary_y = False)
                fig.add_scatter(x = df_k['log_w0'], y = np.log10(df_k['max_y_tip']/np.max(df_k['max_y_tip'])), row = 1 + k_df, col = 1, mode = "markers", marker_color = cb_orange, name = "y_max", secondary_y = True)
            
            x_ticks = np.arange(-7, 3, 3)
            x_ticks_text = [r"$\huge{" + str(x_tick) + "}$" for x_tick in x_ticks]            
            fig.update_xaxes(
                title = r"$\huge{\log{\omega_0}}$",
                range = [-7.1,2.1],
                tickmode = "array",
                tickvals = x_ticks,
                ticktext = x_ticks_text,
            )
            y_ticks = np.arange(250,600,125)/1000
            y_ticks_text = [r"$\huge{" + str(y_tick) + "}$" for y_tick in y_ticks]
            fig.update_yaxes(
                title = r"$\huge{\phi_\text{max}}$",
                range = [0.2, 0.55],
                secondary_y = False,
                tickmode = "array",
                tickvals = y_ticks,
                ticktext = y_ticks_text,
            )
            y_ticks = np.arange(-6,1,2)
            y_ticks_text = [r"$\huge{" + str(y_tick) + "}$" for y_tick in y_ticks]            
            fig.update_yaxes(
                title = r"$\huge{\log{y_\text{max}^\circ}}$",
                range = [-4.8, 0.1],
                secondary_y = True,
                tickmode = "array",
                tickvals = y_ticks,
                ticktext = y_ticks_text,
                row = 1,
                col = 1,
            )
            y_ticks = np.arange(-3,1,1)
            y_ticks_text = [r"$\huge{" + str(y_tick) + "}$" for y_tick in y_ticks]            
            fig.update_yaxes(
                title = r"$\huge{\log{y_\text{max}^\circ}}$",
                range = [-2.7, 0.1],
                secondary_y = True,
                tickmode = "array",
                tickvals = y_ticks,
                ticktext = y_ticks_text,
                row = 2,
                col = 1,
            )                           
            fig.update_layout(
                margin = dict(l = 200, r = 200, t = 200, b = 200),
                width = 400 + 400,
                height = 300*2 + 400,
                showlegend = False,
            )

    # Inference Bending (elasticity + viscosity): phase diagrams
    elif fig_nbr == 10:

        folder_name = Path('..').joinpath('Model').joinpath('Output').resolve()
        folder_name /= "StraightLine_PeriodicFlow"
        folder_name /= "BendingElasticity_Clamped_VaryingBendingViscosity"
        folder_name /= "VaryingFrequencyAmplitude"
        dataframe_filename = folder_name + "maxdev" + ".csv"
        N = 10

        df = pd.read_csv(dataframe_filename)

        eps = 1e-6
        df['log_w0'] = df.apply(lambda x: x['log_w0']/np.log(10), axis = 1)
        df['log_A'] = df.apply(lambda x: x['log_A']/np.log(10), axis = 1)
        df['log_tau_b_m1'] = df.apply(lambda x: x['log_tau_b_m1']/np.log(10), axis = 1)
        df['max_y_tip'] = df.apply(lambda x: x['max_y_tip']/N, axis = 1)
        df['log_max_y_tip'] = df.apply(lambda x: np.log10(x['max_y_tip']), axis = 1)

        tau_b_m1_list = list(np.unique(df['tau_b_m1']))
        # tau_b_m1_list = [tau_b_m1_list[0], tau_b_m1_list[2], tau_b_m1_list[-1]]
        print('tau_b_m1_list', tau_b_m1_list)

        # Heatmaps
        if panel_nbr == 0:

            fig = make_subplots(rows = 2, cols = len(tau_b_m1_list), subplot_titles = [r"$\huge{\tau_b =" + sci_notation(1/tau_b_m1, 1, 1) + "}$" for tau_b_m1 in tau_b_m1_list])

            for l in range(len(tau_b_m1_list)):
                tau_b_m1 = tau_b_m1_list[l]
                df_tau_b_m1 =  df.loc[np.abs(df['tau_b_m1']- tau_b_m1) < eps]
                fig_phi_tau_b_m1 = plot_heatmap(df_tau_b_m1, 'log_w0', 'log_A', 'phi_max_y_tip')
                fig_phi_tau_b_m1.update_traces(coloraxis = 'coloraxis')
                fig_B_tau_b_m1 = plot_heatmap(df_tau_b_m1, 'log_w0', 'log_A', 'max_y_tip')
                fig_B_tau_b_m1.update_traces(coloraxis = 'coloraxis2')
                fig.add_trace(fig_phi_tau_b_m1.data[0], row = 1, col = 1+l)
                fig.add_trace(fig_B_tau_b_m1.data[0], row = 2, col = 1+l)

            fig.update_xaxes(
                title = r"$\huge{\log \omega_0}$",
            )
            fig.update_yaxes(
                title = r"$\huge{\log B}$",
            )            

            fig.update_layout(
                margin = dict(l = 400, r = 400, t = 400, b = 400),
                width = 400 * len(tau_b_m1_list) + 400*2,
                height = 500 * 2 + 400*2,
                coloraxis=dict(
                    colorbar = dict(
                        title = r"$\huge{\phi_\text{max}}$",
                        x = 1, yanchor = 'top', y = 1, 
                        len = 1/2 - 1/10, 
                        tickmode="array",
                        tickcolor = 'black',
                        tickvals = [0, 0.2, 0.4, 0.6, 0.8, 1],
                        ticktext = ["0", "0.2", "0.4", "0.6", "0.8", "1"],
                        ticks = "outside",
                        tickwidth = 3,
                        ticklen = 12,
                        ),
                    colorscale = dark_purple_scale,
                ),
                coloraxis2=dict(
                    colorbar = dict(
                        title = r"$\huge{\frac{\displaystyle y_\text{max}}{N}}$",
                        x = 1, yanchor = 'bottom', y = 0, 
                        len = 1/2 - 1/10,
                        tickcolor = 'black',
                        # dtick = 1,
                        ticks = "outside",
                        tickwidth = 3,
                        ticklen = 12,                        
                        ),
                ),
                )

    # Inference Bending (elasticity + viscosity): phase response
    elif fig_nbr == 11:

        folder_name = Path('..').joinpath('Model').joinpath('Output').resolve()
        folder_name /= "StraightLine_PeriodicFlow"
        folder_name /= "BendingElasticity_Clamped_VaryingBendingViscosity"
        folder_name /= "VaryingFrequencyAmplitude"
        dataframe_filename = folder_name + "maxdev" + ".csv"

        df = pd.read_csv(dataframe_filename)

        eps = 1e-6
        df['log_w0'] = df.apply(lambda x: x['log_w0']/np.log(10), axis = 1)
        df['log_A'] = df.apply(lambda x: x['log_A']/np.log(10), axis = 1)
        df['log_tau_b_m1'] = df.apply(lambda x: x['log_tau_b_m1']/np.log(10), axis = 1)
        df['log_max_y_tip'] = df.apply(lambda x: np.log10(x['max_y_tip']), axis = 1)

        tau_b_m1_list = list(np.unique(df['tau_b_m1']))
        # tau_b_m1_list = [tau_b_m1_list[0], tau_b_m1_list[2], tau_b_m1_list[-1]]
        print('tau_b_m1_list', tau_b_m1_list)

        # Transects phi(w0) and phi(A)
        if panel_nbr == 0:
            
            def f_arctan(w_0, a, b):
                return b + np.arctan(a*w_0)/(2*np.pi)

            # Select only for one value of the amplitude
            eps = 1e-12
            A = 1e-5
            df_A = df.loc[np.abs(df['A'] - A) < eps]

            # Select only for one value of the frequency
            eps = 1e-12
            w0 = 1e-8
            df_w0 = df.loc[np.abs(df['w0'] - w0) < eps]

            fig = make_subplots(rows = 2, cols = len(tau_b_m1_list), subplot_titles = [r"$\huge{\tau_b =" + sci_notation(1/tau_b_m1, 1, 1) + "}$" for tau_b_m1 in tau_b_m1_list])
    
            for l in range(len(tau_b_m1_list)):
                tau_b_m1 = tau_b_m1_list[l]
                df_tau_b_m1 =  df_A.loc[np.abs(df_A['tau_b_m1'] - tau_b_m1) < eps]

                fig.add_scatter(x = df_tau_b_m1['w0'], y = df_tau_b_m1['phi_max_y_tip'], row = 1, col = 1 + l, mode = 'markers', marker_color = "black")

                if l == 0: #tau_b = 1e6
                    w0_max = 1e-5
                elif l == 1: # tau_b = 1e5
                    w0_max = 1e-4
                elif l == 2: # tau_b = 1e4
                    w0_max = 1e-3
                elif l == 3: # tau_b = 1e3
                    w0_max = 6e-3
                elif l == 4: # tau_b = 1e2
                    w0_max = 1e-2
                elif l == 5: # tau_b = 1e1
                    w0_max = 1e-2
                else: # tau_b = 1e0
                    w0_max = 1e-2
                df_fit = df_tau_b_m1.loc[df_tau_b_m1['w0'] - (3/2)*w0_max < 0]
                popt, pcov = curve_fit(f = f_arctan, xdata = df_fit['w0'], ydata = df_fit['phi_max_y_tip'])
                
                # n_samples = 1e3
                x_samples = np.sort(df_tau_b_m1['w0'])
                y_fit = f_arctan(x_samples, *popt)
                fig.add_scatter(x = x_samples, y = y_fit, row = 1, col = 1 + l, mode = 'lines', marker_color = cb_dark_orange)
                
                x_ticks = [0, w0_max, 2*w0_max, 3*w0_max]
                x_ticks_text = [r"$\huge{" + sci_notation(x_tick, decimal_digits = 0) + "}$" for x_tick in x_ticks]  
                fig.update_xaxes(
                    range = [0, 3*w0_max], 
                    tickmode="array",
                    tickvals = x_ticks,
                    ticktext = x_ticks_text,
                    row = 1, 
                    col = 1+l,
                    )

                # gamma/k = a
                gamma_over_k = popt[0]
                sd_gamma_over_k = np.sqrt(np.diag(pcov))[0]
                print("gamma/k = ", gamma_over_k, sd_gamma_over_k)

            for l in range(len(tau_b_m1_list)):
                tau_b_m1 = tau_b_m1_list[l]
                df_tau_b_m1 =  df_w0.loc[np.abs(df_w0['tau_b_m1'] - tau_b_m1) < eps]

                fig.add_scatter(x = df_tau_b_m1['A'], y = df_tau_b_m1['phi_max_y_tip'], row = 2, col = 1 + l, mode = 'markers', marker_color = "black")

            fig.update_xaxes(
                title = r"$\huge{\omega_0}$",
                row = 1,
            )
            x_ticks = [0, 5e-3, 1e-2]
            x_ticks_text = [r"$\huge{" + str(x_tick) + "}$" for x_tick in x_ticks]                  
            fig.update_xaxes(
                title = r"$\huge{A}$",
                tickmode="array",
                tickvals = x_ticks,
                ticktext = x_ticks_text,                
                row = 2,
            )            
            fig.update_yaxes(
                title = r"$\huge{\phi_\text{max}}$",
                col = 1,
            )
            y_ticks = [0.25, 0.375, 0.5]
            y_ticks_text = [r"$\huge{" + str(y_tick) + "}$" for y_tick in y_ticks]
            fig.update_yaxes(
                range = [0.24, 0.61],
                tickmode="array",
                tickvals = y_ticks,
                ticktext = y_ticks_text,
                row = 1,
            )
            y_ticks = [0, 0.25, 0.5]
            y_ticks_text = [r"$\huge{" + str(y_tick) + "}$" for y_tick in y_ticks]            
            fig.update_yaxes(
                range = [-0.01, 0.51],
                tickmode="array",
                tickvals = y_ticks,
                ticktext = y_ticks_text,                
                row = 2,
            )
            for col_m1 in range(1,len(tau_b_m1_list)):
                fig.update_yaxes(showticklabels = False, col = col_m1 + 1)                        

            fig.update_layout(
                margin = dict(l = 400, r = 400, t = 400, b = 400),
                width = 400 * len(tau_b_m1_list) + 400*2,
                height = 500 + 400*2,
                )

    # Inference Bending (elasticity + viscosity): amplitude response
    elif fig_nbr == 12:

        folder_name = Path('..').joinpath('Model').joinpath('Output').resolve()
        folder_name /= "StraightLine_PeriodicFlow"
        folder_name /= "BendingElasticity_Clamped_VaryingBendingViscosity"
        folder_name /= "VaryingFrequencyAmplitude"
        dataframe_filename = folder_name + "maxdev" + ".csv"

        df = pd.read_csv(dataframe_filename)

        eps = 1e-6
        df['log_w0'] = df.apply(lambda x: x['log_w0']/np.log(10), axis = 1)
        df['log_A'] = df.apply(lambda x: x['log_A']/np.log(10), axis = 1)
        df['log_tau_b_m1'] = df.apply(lambda x: x['log_tau_b_m1']/np.log(10), axis = 1)
        df['log_max_y_tip'] = df.apply(lambda x: np.log10(x['max_y_tip']), axis = 1)

        tau_b_m1_list = list(np.unique(df['tau_b_m1']))
        # tau_b_m1_list = [tau_b_m1_list[0], tau_b_m1_list[2], tau_b_m1_list[-1]]
        print('tau_b_m1_list', tau_b_m1_list)

        # Transects B(w0) and B(A)
        if panel_nbr == 0:

            def f_sqm1(x, alpha, beta):
                return beta * (alpha * x**2 + 1)**(-1/2)

            def f_lin(x, a, b):
                return b + a*x

            # Select only for one value of the amplitude
            eps = 1e-12
            A = 1e-5
            df_A = df.loc[np.abs(df['A'] - A) < eps]

            # Select only for one value of the frequency
            eps = 1e-12
            w0 = 1e-8
            df_w0 = df.loc[np.abs(df['w0'] - w0) < eps]

            fig = make_subplots(rows = 2, cols = len(tau_b_m1_list), subplot_titles = [r"$\huge{\tau_b =" + sci_notation(1/tau_b_m1, 1, 1) + "}$" for tau_b_m1 in tau_b_m1_list])
    
            for l in range(len(tau_b_m1_list)):
                tau_b_m1 = tau_b_m1_list[l]
                df_tau_b_m1 =  df_A.loc[np.abs(df_A['tau_b_m1'] - tau_b_m1) < eps]

                fig.add_scatter(x = df_tau_b_m1['w0'], y = df_tau_b_m1['max_y_tip'], row = 1, col = 1 + l, mode = 'markers', marker_color = "black")

                popt, pcov = curve_fit(f = f_sqm1, xdata = df_tau_b_m1['w0'], ydata = df_tau_b_m1['max_y_tip'])
                
                # n_samples = 1e3
                x_samples = np.sort(df_tau_b_m1['w0'])
                y_fit = f_sqm1(x_samples, *popt)
                fig.add_scatter(x = x_samples, y = y_fit, row = 1, col = 1 + l, mode = 'lines', marker_color = cb_dark_orange)

                # gamma/k = np.sqrt(alpha)
                gamma_over_k = np.sqrt(popt[0])
                sd_gamma_over_k = np.sqrt(np.diag(pcov))[0] / (2 *  np.sqrt(popt[0]))
                # k = A/beta, A = 1e-2
                k = A/popt[1]
                sd_k = A * np.sqrt(np.diag(pcov))[1] / (popt[1]**2)
                # gamma = np.sqrt(alpha) * k
                gamma = np.sqrt(popt[0])*k
                sd_gamma = A * np.abs(-np.sqrt(np.diag(pcov))[0] * popt[1] / (2 * np.sqrt(popt[0])) - np.sqrt(np.diag(pcov))[1]*np.sqrt(popt[0])) / (popt[1]**2)
                # sqrt(w0^2gamma^2 + k^2)
                sqrt_ = np.sqrt((w0*gamma)**2 + k**2)
                sd_sqrt_ = (w0**2 * gamma * sd_gamma + k * sd_k) / sqrt_

                print("tau_b: ", 1/tau_b_m1)
                print("k", k, sd_k)
                print("gamma", gamma, sd_gamma)
                print("gamma/k", gamma_over_k, sd_gamma_over_k)
                print("sqrt_", sqrt_, sd_sqrt_)

            for l in range(len(tau_b_m1_list)):
                tau_b_m1 = tau_b_m1_list[l]
                df_tau_b_m1 =  df_w0.loc[np.abs(df_w0['tau_b_m1'] - tau_b_m1) < eps]

                fig.add_scatter(x = df_tau_b_m1['A'], y = df_tau_b_m1['max_y_tip'], row = 2, col = 1 + l, mode = 'markers', marker_color = "black")

                max_A = 1e-3
                df_fit = df_tau_b_m1.loc[df_tau_b_m1['A'] - max_A <= 0]
                popt, pcov = curve_fit(f = f_lin, xdata = df_fit['A'], ydata = df_fit['max_y_tip'])
                # n_samples = 1e3
                x_samples = np.sort(df_tau_b_m1['A'])
                y_fit = f_lin(x_samples, *popt)
                fig.add_scatter(x = x_samples, y = y_fit, row = 2, col = 1 + l, mode = 'lines', marker_color = cb_dark_orange)

                # (w0^2*gamma^2 + k^2)^(-1/2)
                sqrt_ = 1 / popt[0]
                sd_sqrt_ = np.sqrt(np.diag(pcov))[0] / (popt[0]**2)

                print("tau_b: ", 1/tau_b_m1)
                print("sqrt_", sqrt_, sd_sqrt_)

                y_fit_max = y_fit[-1]
                if y_fit_max <= 1:
                    d_r = 1
                else:
                    d_r = 0
                y_ticks = [0, np.round(y_fit_max/3, d_r), 2*np.round(y_fit_max/3, d_r), np.round(3*np.round(y_fit_max/3, d_r), d_r)]
                if y_fit_max > 1:
                    y_ticks = [int(y_tick) for y_tick in y_ticks]
                y_ticks_text = [r"$\huge{" + str(y_tick) + "}$" for y_tick in y_ticks] 
                fig.update_yaxes(
                    tickmode="array",
                    tickvals = y_ticks,
                    ticktext = y_ticks_text,        
                    row = 2, col = 1 + l,
                )

            x_log_list = np.arange(-8,1,4)
            x_ticks = [np.float_power(10, m) for m in x_log_list]
            x_ticks_text = [r"$\huge{" + sci_notation(x_tick, decimal_digits=-1) + "}$" for x_tick in x_ticks]                 
            fig.update_xaxes(
                title = r"$\huge{\omega_0}$",
                type = 'log',
                tickmode="array",
                tickvals = x_ticks,
                ticktext = x_ticks_text,   
                row = 1,
            )
            x_ticks = [0, 5e-3, 1e-2]
            x_ticks_text = [r"$\huge{" + str(x_tick) + "}$" for x_tick in x_ticks]              
            fig.update_xaxes(
                title = r"$\huge{A}$",
                tickmode="array",
                tickvals = x_ticks,
                ticktext = x_ticks_text,
                row = 2,
            )
            y_log_list = np.arange(-8,-1,2)
            y_ticks = [np.float_power(10, m) for m in y_log_list]
            y_ticks_text = [r"$\huge{" + sci_notation(y_tick, decimal_digits=-1) + "}$" for y_tick in y_ticks]                 
            fig.update_yaxes(
                type = 'log',
                tickmode="array",
                tickvals = y_ticks,
                ticktext = y_ticks_text,                     
                row = 1,
            )
            fig.update_yaxes(
                title = r"$\huge{y_\text{max}}$",
                col = 1,
            )
            fig.update_layout(
                margin = dict(l = 400, r = 400, t = 400, b = 400),
                width = 400 * len(tau_b_m1_list) + 400*2,
                height = 500 + 400*2,
                )
        
    # Inference Bending (elasticity + viscosity): fit values    
    elif fig_nbr == 13:

        tau_b = 10**np.arange(0,7,1)[::-1]
        tau_f_b = 1.3e3 # Approximate value
        gamma_over_k = np.array([0.99e6, 1.00e5, 1.12e4, 2.29e3, 1.40e3, 1.30e3, 1.29e3])
        sd_gamma_over_k = np.array([0.01e6, 0.01e5, 0.01e4, 0.02e3, 0.01e3, 0.01e3, 0.01e3])
        eps_r_tau_b = (gamma_over_k - tau_b) / tau_b + 1
        sd_eps_r_tau_b = sd_gamma_over_k / tau_b
        eps_r_tau_fb =(gamma_over_k - tau_f_b) / tau_f_b + 1
        sd_eps_r_tau_fb = sd_gamma_over_k / tau_f_b      
        eps_r_tau_bfb = (gamma_over_k - (tau_b + tau_f_b)) / (tau_b+tau_f_b) + 1
        sd_eps_r_tau_bfb = sd_gamma_over_k / (tau_b + tau_f_b)        

        if panel_nbr == 0:

            fig = go.Figure()
            fig.add_scatter(x = tau_b, y = eps_r_tau_b, 
                error_y = dict(type = 'data', visible = True, array = sd_eps_r_tau_b),
                mode = 'markers+lines', marker_color = cb_dark_orange,
                name = r"$\huge{\frac{\gamma}{k \tau_{b}}}$",
                )
            fig.add_scatter(x = tau_b, y = eps_r_tau_fb, 
                error_y = dict(type = 'data', visible = True, array = sd_eps_r_tau_fb),
                mode = 'markers+lines', marker_color = cb_dark_purple,
                name = r"$\huge{\frac{\gamma}{k \tau_{f,b}}}$",
                )            
            fig.add_scatter(x = tau_b, y = eps_r_tau_bfb, 
                error_y = dict(type = 'data', visible = True, array = sd_eps_r_tau_bfb),
                mode = 'markers', marker_color = cb_red,
                name = r"$\huge{\frac{\gamma}{k (\tau_b + \tau_{f,b})}}$",
                )

            x_ticks = np.float_power(10, np.arange(0,10,1)) 
            x_ticks_text = [r"$\huge{" + sci_notation(x_tick, decimal_digits=-1) + "}$" for x_tick in x_ticks]   
            for k in range(len(x_ticks_text)):
                if k%2 == 1:
                    x_ticks_text[k] = ""
            fig.update_xaxes(
                title = r"$\huge{\tau_b}$", 
                type = 'log',
                tickmode="array",
                tickvals = x_ticks,
                ticktext = x_ticks_text,               
                )
            y_ticks = np.float_power(10, np.arange(0,4,1))
            y_ticks_text = [r"$\huge{" + sci_notation(y_tick, decimal_digits=-1) + "}$" for y_tick in y_ticks]
            fig.update_yaxes(
                type = 'log',
                tickmode="array",
                tickvals = y_ticks,
                ticktext = y_ticks_text,             
                )
            fig.update_layout(
                margin = dict(l = 200, r = 200, t = 200, b = 200),
                width = 500 + 400,
                height = 500 + 400,
            )

        # Plots of relative error of best error functional
        elif panel_nbr == 1:

            fig = go.Figure()
            fig.add_scatter(x = tau_b, y = 100*(eps_r_tau_bfb - 1), 
                error_y = dict(type = 'data', visible = True, array = sd_eps_r_tau_bfb),
                mode = 'markers', marker_color = cb_red,
                )

            x_ticks = np.float_power(10, np.arange(0,10,1)) 
            x_ticks_text = [r"$\huge{" + sci_notation(x_tick, decimal_digits=-1) + "}$" for x_tick in x_ticks]   
            for k in range(len(x_ticks_text)):
                if k%2 == 1:
                    x_ticks_text[k] = ""                     
            fig.update_xaxes(
                title = r"$\huge{\tau_b}$",
                tickmode="array",
                tickvals = x_ticks,
                ticktext = x_ticks_text,                  
                type = 'log',
                )
            y_ticks = np.arange(-1,2,1)
            y_ticks_text = [r"$\huge{" + str(y_tick) + "\%" + "}$" for y_tick in y_ticks]                    
            fig.update_yaxes(
                range = [-1.5, 1.5],
                title = r"$\huge{\frac{\gamma/k - (\tau_b - \tau_{f,b})}{\tau_b - \tau_{f,b}}}$",
                tickmode="array",
                tickvals = y_ticks,
                ticktext = y_ticks_text,                        
                type = 'linear',
                )
            fig.update_layout(
                margin = dict(l = 200, r = 200, t = 200, b = 200),
                width = 500 + 400,
                height = 250 + 400,
                showlegend = False,                
            )

    # Inference shear (elasticity + viscosity): phase diagrams
    elif fig_nbr == 14:

        folder_name = Path('..').joinpath('Model').joinpath('Output').resolve()
        folder_name /= "StraightLine_PeriodicFlow"
        folder_name /= "ShearElasticity_Clamped_VaryingShearViscosity"
        folder_name /= "VaryingFrequencyAmplitude"
        dataframe_filename = folder_name + "maxdev" + ".csv"
        N = 10

        df = pd.read_csv(dataframe_filename)

        eps = 1e-6
        df['log_w0'] = df.apply(lambda x: x['log_w0']/np.log(10), axis = 1)
        df['log_A'] = df.apply(lambda x: x['log_A']/np.log(10), axis = 1)
        df['log_tau_s_m1'] = df.apply(lambda x: x['log_tau_s_m1']/np.log(10), axis = 1)
        df['max_y_tip'] = df.apply(lambda x: x['max_y_tip']/N, axis = 1)
        df['log_max_y_tip'] = df.apply(lambda x: np.log10(x['max_y_tip']), axis = 1)

        tau_s_m1_list = list(np.unique(df['tau_s_m1']))
        print('tau_s_m1_list', tau_s_m1_list)

        # Heatmaps
        if panel_nbr == 0:

            fig = make_subplots(rows = 2, cols = len(tau_s_m1_list), subplot_titles = [r"$\huge{\tau_s =" + sci_notation(1/tau_s_m1, 1, 1) + "}$" for tau_s_m1 in tau_s_m1_list])

            for l in range(len(tau_s_m1_list)):
                tau_s_m1 = tau_s_m1_list[l]
                df_tau_s_m1 =  df.loc[np.abs(df['tau_s_m1']- tau_s_m1) < eps]
                fig_phi_tau_s_m1 = plot_heatmap(df_tau_s_m1, 'log_w0', 'log_A', 'phi_max_y_tip')
                fig_phi_tau_s_m1.update_traces(coloraxis = 'coloraxis')
                fig_B_tau_s_m1 = plot_heatmap(df_tau_s_m1, 'log_w0', 'log_A', 'max_y_tip')
                fig_B_tau_s_m1.update_traces(coloraxis = 'coloraxis2')
                fig.add_trace(fig_phi_tau_s_m1.data[0], row = 1, col = 1+l)
                fig.add_trace(fig_B_tau_s_m1.data[0], row = 2, col = 1+l)

            fig.update_xaxes(
                title = r"$\huge{\log \omega_0}$",
            )
            fig.update_yaxes(
                title = r"$\huge{\log B}$",
            )            

            fig.update_layout(
                margin = dict(l = 400, r = 400, t = 400, b = 400),
                width = 400 * len(tau_s_m1_list) + 400*2,
                height = 500 * 2 + 400*2,
                coloraxis=dict(
                    colorbar = dict(
                        title = r"$\huge{\phi_\text{max}}$",
                        x = 1, yanchor = 'top', y = 1, 
                        len = 1/2 - 1/10, 
                        tickmode="array",
                        tickcolor = 'black',
                        tickvals = [0, 0.2, 0.4, 0.6, 0.8, 1],
                        ticktext = ["0", "0.2", "0.4", "0.6", "0.8", "1"],
                        ticks = "outside",
                        tickwidth = 3,
                        ticklen = 12,
                        ),
                    colorscale = dark_purple_scale,
                ),
                coloraxis2=dict(
                    colorbar = dict(
                        title = r"$\huge{\frac{\displaystyle y_\text{max}}{\displaystyle N}}$",
                        x = 1, yanchor = 'bottom', y = 0, 
                        len = 1/2 - 1/10,
                        tickcolor = 'black',
                        # dtick = 1,
                        ticks = "outside",
                        tickwidth = 3,
                        ticklen = 12,                        
                        ),
                ),
                )

    # Inference Shear (elasticity + viscosity): phase response
    elif fig_nbr == 15:

        folder_name = Path('..').joinpath('Model').joinpath('Output').resolve()
        folder_name /= "StraightLine_PeriodicFlow"
        folder_name /= "ShearElasticity_Clamped_VaryingShearViscosity"
        folder_name /= "VaryingFrequencyAmplitude"
        dataframe_filename = folder_name + "maxdev" + ".csv"

        df = pd.read_csv(dataframe_filename)

        eps = 1e-6
        df['log_w0'] = df.apply(lambda x: x['log_w0']/np.log(10), axis = 1)
        df['log_A'] = df.apply(lambda x: x['log_A']/np.log(10), axis = 1)
        df['log_tau_s_m1'] = df.apply(lambda x: x['log_tau_s_m1']/np.log(10), axis = 1)
        df['log_max_y_tip'] = df.apply(lambda x: np.log10(x['max_y_tip']), axis = 1)

        tau_s_m1_list = list(np.unique(df['tau_s_m1']))
        print('tau_s_m1_list', tau_s_m1_list)

        # Transects phi(w0) and phi(A)
        if panel_nbr == 0:
            
            def f_arctan(w_0, a, b):
                return b + np.arctan(a*w_0)/(2*np.pi)

            # Select only for one value of the amplitude
            eps = 1e-12
            A = 1e-3
            df_A = df.loc[np.abs(df['A'] - A) < eps]

            # Select only for one value of the frequency
            eps = 1e-12
            w0 = 1e-7
            df_w0 = df.loc[np.abs(df['w0'] - w0) < eps]

            fig = make_subplots(rows = 2, cols = len(tau_s_m1_list), subplot_titles = [r"$\huge{\tau_s =" + sci_notation(1/tau_s_m1, 1, 1) + "}$" for tau_s_m1 in tau_s_m1_list])
    
            for l in range(len(tau_s_m1_list)):
                tau_s_m1 = tau_s_m1_list[l]
                df_tau_s_m1 =  df_A.loc[np.abs(df_A['tau_s_m1'] - tau_s_m1) < eps]

                fig.add_scatter(x = df_tau_s_m1['w0'], y = df_tau_s_m1['phi_max_y_tip'], row = 1, col = 1 + l, mode = 'markers', marker_color = "black")

                if l == 0: #tau_s = 1e5
                    w0_max = 1e-4
                elif l == 1: # tau_s = 1e4
                    w0_max = 1e-3
                elif l == 2: # tau_s = 1e3
                    w0_max = 1e-2
                elif l == 3: # tau_s = 1e2
                    w0_max = 6e-2
                elif l == 4: # tau_s = 1e1
                    w0_max = 1e-1
                elif l == 5: # tau_s = 1e0
                    w0_max = 1e-1
                else: # tau_s = 1e0
                    w0_max = 1e-1
                df_fit = df_tau_s_m1.loc[df_tau_s_m1['w0'] - (3/2)*w0_max < 0]
                popt, pcov = curve_fit(f = f_arctan, xdata = df_fit['w0'], ydata = df_fit['phi_max_y_tip'])
                
                # n_samples = 1e3
                x_samples = np.sort(df_tau_s_m1['w0'])
                y_fit = f_arctan(x_samples, *popt)
                fig.add_scatter(x = x_samples, y = y_fit, row = 1, col = 1 + l, mode = 'lines', marker_color = cb_dark_orange)
                
                x_ticks = [0, w0_max, 2*w0_max, 3*w0_max]
                x_ticks_text = [r"$\huge{" + sci_notation(x_tick, decimal_digits = 0) + "}$" for x_tick in x_ticks]  
                fig.update_xaxes(
                    range = [0, 3*w0_max], 
                    tickmode="array",
                    tickvals = x_ticks,
                    ticktext = x_ticks_text,
                    row = 1, 
                    col = 1+l,
                    )

                # gamma/k = a
                gamma_over_k = popt[0]
                sd_gamma_over_k = np.sqrt(np.diag(pcov))[0]
                print("gamma/k = ", gamma_over_k, sd_gamma_over_k)

            for l in range(len(tau_s_m1_list)):
                tau_s_m1 = tau_s_m1_list[l]
                df_tau_s_m1 =  df_w0.loc[np.abs(df_w0['tau_s_m1'] - tau_s_m1) < eps]

                fig.add_scatter(x = df_tau_s_m1['A'], y = df_tau_s_m1['phi_max_y_tip'], row = 2, col = 1 + l, mode = 'markers', marker_color = "black")

            fig.update_xaxes(
                title = r"$\huge{\omega_0}$",
                row = 1,
            )
            x_ticks = [0, 5e-1, 1e0]
            x_ticks_text = [r"$\huge{" + str(x_tick) + "}$" for x_tick in x_ticks]                  
            fig.update_xaxes(
                title = r"$\huge{A}$",
                tickmode="array",
                tickvals = x_ticks,
                ticktext = x_ticks_text,                
                row = 2,
            )            
            fig.update_yaxes(
                title = r"$\huge{\phi_\text{max}}$",
                col = 1,
            )
            y_ticks = [0.25, 0.375, 0.5]
            y_ticks_text = [r"$\huge{" + str(y_tick) + "}$" for y_tick in y_ticks]
            fig.update_yaxes(
                range = [0.24, 0.61],
                tickmode="array",
                tickvals = y_ticks,
                ticktext = y_ticks_text,
                row = 1,
            )
            y_ticks = [0, 0.25, 0.5]
            y_ticks_text = [r"$\huge{" + str(y_tick) + "}$" for y_tick in y_ticks]            
            fig.update_yaxes(
                range = [-0.01, 0.51],
                tickmode="array",
                tickvals = y_ticks,
                ticktext = y_ticks_text,                
                row = 2,
            )
            for col_m1 in range(1,len(tau_s_m1_list)):
                fig.update_yaxes(showticklabels = False, col = col_m1 + 1)                        

            fig.update_layout(
                margin = dict(l = 400, r = 400, t = 400, b = 400),
                width = 400 * len(tau_s_m1_list) + 400*2,
                height = 500 + 400*2,
                )

    # Inference Shear (elasticity + viscosity): amplitude response
    elif fig_nbr == 16:

        folder_name = Path('..').joinpath('Model').joinpath('Output').resolve()
        folder_name /= "StraightLine_PeriodicFlow"
        folder_name /= "ShearElasticity_Clamped_VaryingShearViscosity"
        folder_name /= "VaryingFrequencyAmplitude"
        dataframe_filename = folder_name + "maxdev" + ".csv"

        df = pd.read_csv(dataframe_filename)

        eps = 1e-6
        df['log_w0'] = df.apply(lambda x: x['log_w0']/np.log(10), axis = 1)
        df['log_A'] = df.apply(lambda x: x['log_A']/np.log(10), axis = 1)
        df['log_tau_s_m1'] = df.apply(lambda x: x['log_tau_s_m1']/np.log(10), axis = 1)
        df['log_max_y_tip'] = df.apply(lambda x: np.log10(x['max_y_tip']), axis = 1)

        tau_s_m1_list = list(np.unique(df['tau_s_m1']))
        print('tau_s_m1_list', tau_s_m1_list)

        # Transects B(w0) and B(A)
        if panel_nbr == 0:

            def f_sqm1(x, alpha, beta):
                return beta * (alpha * x**2 + 1)**(-1/2)

            def f_lin(x, a, b):
                return b + a*x

            # Select only for one value of the amplitude
            eps = 1e-12
            A = 1e-3
            df_A = df.loc[np.abs(df['A'] - A) < eps]

            # Select only for one value of the frequency
            eps = 1e-12
            w0 = 1e-7
            df_w0 = df.loc[np.abs(df['w0'] - w0) < eps]

            fig = make_subplots(rows = 2, cols = len(tau_s_m1_list), subplot_titles = [r"$\huge{\tau_s =" + sci_notation(1/tau_s_m1, 1, 1) + "}$" for tau_s_m1 in tau_s_m1_list])
    
            for l in range(len(tau_s_m1_list)):
                tau_s_m1 = tau_s_m1_list[l]
                df_tau_s_m1 =  df_A.loc[np.abs(df_A['tau_s_m1'] - tau_s_m1) < eps]

                fig.add_scatter(x = df_tau_s_m1['w0'], y = df_tau_s_m1['max_y_tip'], row = 1, col = 1 + l, mode = 'markers', marker_color = "black")

                popt, pcov = curve_fit(f = f_sqm1, xdata = df_tau_s_m1['w0'], ydata = df_tau_s_m1['max_y_tip'])
                
                # n_samples = 1e3
                x_samples = np.sort(df_tau_s_m1['w0'])
                y_fit = f_sqm1(x_samples, *popt)
                fig.add_scatter(x = x_samples, y = y_fit, row = 1, col = 1 + l, mode = 'lines', marker_color = cb_dark_orange)

                # gamma/k = np.sqrt(alpha)
                gamma_over_k = np.sqrt(popt[0])
                sd_gamma_over_k = np.sqrt(np.diag(pcov))[0] / (2 *  np.sqrt(popt[0]))
                # k = A/beta, A = 1e-2
                k = A/popt[1]
                sd_k = A * np.sqrt(np.diag(pcov))[1] / (popt[1]**2)
                # gamma = np.sqrt(alpha) * k
                gamma = np.sqrt(popt[0])*k
                sd_gamma = A * np.abs(-np.sqrt(np.diag(pcov))[0] * popt[1] / (2 * np.sqrt(popt[0])) - np.sqrt(np.diag(pcov))[1]*np.sqrt(popt[0])) / (popt[1]**2)
                # sqrt(w0^2gamma^2 + k^2)
                sqrt_ = np.sqrt((w0*gamma)**2 + k**2)
                sd_sqrt_ = (w0**2 * gamma * sd_gamma + k * sd_k) / sqrt_

                print("tau_s: ", 1/tau_s_m1)
                print("k", k, sd_k)
                print("gamma", gamma, sd_gamma)
                print("gamma/k", gamma_over_k, sd_gamma_over_k)
                print("sqrt_", sqrt_, sd_sqrt_)

            for l in range(len(tau_s_m1_list)):
                tau_s_m1 = tau_s_m1_list[l]
                df_tau_s_m1 =  df_w0.loc[np.abs(df_w0['tau_s_m1'] - tau_s_m1) < eps]

                fig.add_scatter(x = df_tau_s_m1['A'], y = df_tau_s_m1['max_y_tip'], row = 2, col = 1 + l, mode = 'markers', marker_color = "black")

                max_A = 1e-2
                df_fit = df_tau_s_m1.loc[df_tau_s_m1['A'] - max_A <= 0]
                popt, pcov = curve_fit(f = f_lin, xdata = df_fit['A'], ydata = df_fit['max_y_tip'])
                # n_samples = 1e3
                x_samples = np.sort(df_tau_s_m1['A'])
                y_fit = f_lin(x_samples, *popt)
                fig.add_scatter(x = x_samples, y = y_fit, row = 2, col = 1 + l, mode = 'lines', marker_color = cb_dark_orange)

                # (w0^2*gamma^2 + k^2)^(-1/2)
                sqrt_ = 1 / popt[0]
                sd_sqrt_ = np.sqrt(np.diag(pcov))[0] / (popt[0]**2)

                print("tau_s: ", 1/tau_s_m1)
                print("sqrt_", sqrt_, sd_sqrt_)

                y_fit_max = y_fit[-1]
                if y_fit_max <= 1:
                    d_r = 1
                else:
                    d_r = 0
                y_ticks = [0, np.round(y_fit_max/3, d_r), 2*np.round(y_fit_max/3, d_r), np.round(3*np.round(y_fit_max/3, d_r), d_r)]
                if y_fit_max > 1:
                    y_ticks = [int(y_tick) for y_tick in y_ticks]
                y_ticks_text = [r"$\huge{" + str(y_tick) + "}$" for y_tick in y_ticks] 
                fig.update_yaxes(
                    tickmode="array",
                    tickvals = y_ticks,
                    ticktext = y_ticks_text,        
                    row = 2, col = 1 + l,
                )

            x_log_list = np.arange(-8,1,4)
            x_ticks = [np.float_power(10, m) for m in x_log_list]
            x_ticks_text = [r"$\huge{" + sci_notation(x_tick, decimal_digits=-1) + "}$" for x_tick in x_ticks]                 
            fig.update_xaxes(
                title = r"$\huge{\omega_0}$",
                type = 'log',
                tickmode="array",
                tickvals = x_ticks,
                ticktext = x_ticks_text,   
                row = 1,
            )
            x_ticks = [0, 5e-1, 1e0]
            x_ticks_text = [r"$\huge{" + str(x_tick) + "}$" for x_tick in x_ticks]              
            fig.update_xaxes(
                title = r"$\huge{A}$",
                tickmode="array",
                tickvals = x_ticks,
                ticktext = x_ticks_text,
                row = 2,
            )
            y_log_list = np.arange(-8,-1,2)
            y_ticks = [np.float_power(10, m) for m in y_log_list]
            y_ticks_text = [r"$\huge{" + sci_notation(y_tick, decimal_digits=-1) + "}$" for y_tick in y_ticks]                 
            fig.update_yaxes(
                type = 'log',
                tickmode="array",
                tickvals = y_ticks,
                ticktext = y_ticks_text,                     
                row = 1,
            )
            fig.update_yaxes(
                title = r"$\huge{y_\text{max}}$",
                col = 1,
            )
            fig.update_layout(
                margin = dict(l = 400, r = 400, t = 400, b = 400),
                width = 400 * len(tau_s_m1_list) + 400*2,
                height = 500 + 400*2,
                )
        
    # Inference Shear (elasticity + viscosity): fit values    
    elif fig_nbr == 17:

        tau_s = np.float_power(10, np.arange(-1,6,1))[::-1]
        tau_f_s = 6.6e1 # Approximate value
        gamma_over_k = np.array([9.95e4, 1.00e4, 1.06e3, 1.66e2, 7.61e1, 6.69e1, 6.60e1])
        sd_gamma_over_k = np.array([0.04e4, 0.01e4, 0.01e3, 0.01e2, 0.03e1, 0.02e1, 0.02e1])
        eps_r_tau_s = (gamma_over_k - tau_s) / tau_s + 1
        sd_eps_r_tau_s = sd_gamma_over_k / tau_s
        eps_r_tau_fb =(gamma_over_k - tau_f_s) / tau_f_s + 1
        sd_eps_r_tau_fb = sd_gamma_over_k / tau_f_s      
        eps_r_tau_sfb = (gamma_over_k - (tau_s + tau_f_s)) / (tau_s+tau_f_s) + 1
        sd_eps_r_tau_sfb = sd_gamma_over_k / (tau_s + tau_f_s)        

        if panel_nbr == 0:

            fig = go.Figure()
            fig.add_scatter(x = tau_s, y = eps_r_tau_s, 
                error_y = dict(type = 'data', visible = True, array = sd_eps_r_tau_s),
                mode = 'markers+lines', marker_color = cb_dark_orange,
                name = r"$\huge{\frac{\gamma}{k \tau_{s}}}$",
                )
            fig.add_scatter(x = tau_s, y = eps_r_tau_fb, 
                error_y = dict(type = 'data', visible = True, array = sd_eps_r_tau_fb),
                mode = 'markers+lines', marker_color = cb_dark_purple,
                name = r"$\huge{\frac{\gamma}{k \tau_{f,s}}}$",
                )            
            fig.add_scatter(x = tau_s, y = eps_r_tau_sfb, 
                error_y = dict(type = 'data', visible = True, array = sd_eps_r_tau_sfb),
                mode = 'markers', marker_color = cb_red,
                name = r"$\huge{\frac{\gamma}{k (\tau_s + \tau_{f,s})}}$",
                )

            x_ticks = np.float_power(10, np.arange(-1,9,1))
            x_ticks_text = [r"$\huge{" + sci_notation(x_tick, decimal_digits=-1) + "}$" for x_tick in x_ticks]   
            for k in range(len(x_ticks_text)):
                if k%2 == 1:
                    x_ticks_text[k] = ""
            fig.update_xaxes(
                title = r"$\huge{\tau_s}$", 
                type = 'log',
                tickmode="array",
                tickvals = x_ticks,
                ticktext = x_ticks_text,               
                )
            y_ticks = np.float_power(10, np.arange(0,4,1))
            y_ticks_text = [r"$\huge{" + sci_notation(y_tick, decimal_digits=-1) + "}$" for y_tick in y_ticks]
            fig.update_yaxes(
                type = 'log',
                tickmode="array",
                tickvals = y_ticks,
                ticktext = y_ticks_text,             
                )
            fig.update_layout(
                margin = dict(l = 200, r = 200, t = 200, b = 200),
                width = 400 + 400,
                height = 350 + 400,
            )

        # Plots of relative error of best error functional
        elif panel_nbr == 1:

            fig = go.Figure()
            fig.add_scatter(x = tau_s, y = 100*(eps_r_tau_sfb - 1), 
                error_y = dict(type = 'data', visible = True, array = sd_eps_r_tau_sfb),
                mode = 'markers', marker_color = cb_red,
                )

            x_ticks = np.float_power(10, np.arange(-1,9,1))
            x_ticks_text = [r"$\huge{" + sci_notation(x_tick, decimal_digits=-1) + "}$" for x_tick in x_ticks]   
            for k in range(len(x_ticks_text)):
                if k%2 == 1:
                    x_ticks_text[k] = ""
            fig.update_xaxes(
                title = r"$\huge{\tau_s}$",
                tickmode="array",
                tickvals = x_ticks,
                ticktext = x_ticks_text,                  
                type = 'log',
                )
            y_ticks = np.arange(-1,2,1)
            y_ticks_text = [r"$\huge{" + str(y_tick) + "\%" + "}$" for y_tick in y_ticks]                    
            fig.update_yaxes(
                range = [-1.5, 1.5],
                title = r"$\huge{\frac{\gamma/k - (\tau_s - \tau_{f,s})}{\tau_s - \tau_{f,s}}}$",
                tickmode="array",
                tickvals = y_ticks,
                ticktext = y_ticks_text,                        
                type = 'linear',
                )
            fig.update_layout(
                margin = dict(l = 200, r = 200, t = 200, b = 200),
                width = 400 + 400,
                height = 250 + 400,
                showlegend = False,                
            )

    # Direct Inference Bending (elasticity + viscosity): k, gamma
    elif fig_nbr == 18:

        folder_name = Path('..').joinpath('Model').joinpath('Output').resolve()
        folder_name /= "StraightLine_PeriodicFlow"
        folder_name /= "BendingElasticity_Clamped_VaryingBendingViscosity"
        folder_name /= "VaryingFrequencyAmplitude"
        dataframe_filename = folder_name + "maxdev" + ".csv"

        df = pd.read_csv(dataframe_filename)

        eps = 1e-6
        df['log_w0'] = df.apply(lambda x: x['log_w0']/np.log(10), axis = 1)
        df['log_A'] = df.apply(lambda x: x['log_A']/np.log(10), axis = 1)
        df['log_tau_b_m1'] = df.apply(lambda x: x['log_tau_b_m1']/np.log(10), axis = 1)
        df['log_max_y_tip'] = df.apply(lambda x: np.log10(x['max_y_tip']), axis = 1)

        # Add k to df
        df['k'] = df.apply(lambda x: x['A'] / x['max_y_tip'] / np.sqrt(1 + np.tan((x['phi_max_y_tip']-1/4)*2*np.pi)**2), axis = 1)
        # df['log_k'] = df.apply(lambda x: np.log10(x['k']), axis = 1)

        # Add gamma to df
        df['gamma'] = df.apply(lambda x: x['A'] / x['max_y_tip'] * np.tan((x['phi_max_y_tip']-1/4)*2*np.pi) / x['w0'] / np.sqrt(1 + np.tan((x['phi_max_y_tip']-1/4)*2*np.pi)**2), axis = 1)
        # df['log_gamma'] = df.apply(lambda x: np.log10(x['gamma']), axis = 1)

        # Add k/gamma to df
        df['gamma_over_k'] = df.apply(lambda x: np.tan((x['phi_max_y_tip']-1/4)*2*np.pi) / x['w0'], axis = 1)
        # df['log_gamma_over_k'] = df.apply(lambda x: np.log10(x['gamma_over_k']), axis = 1)

        tau_b_m1_list = list(np.unique(df['tau_b_m1']))
        # tau_b_m1_list = [tau_b_m1_list[0], tau_b_m1_list[2], tau_b_m1_list[-1]]
        print('tau_b_m1_list', tau_b_m1_list)

        if panel_nbr == 0:
            fig = make_subplots(rows = len(tau_b_m1_list), cols = 3) # ((tau_b), (k, gamma))
            
            for l in range(len(tau_b_m1_list)):
                tau_b_m1 = tau_b_m1_list[l]
                df_tau_b_m1 =  df.loc[np.abs(df['tau_b_m1'] - tau_b_m1) < eps]

                fig_k_tau_b_m1 = plot_heatmap(df_tau_b_m1, 'log_w0', 'log_A', 'k')
                fig_k_tau_b_m1.update_traces(coloraxis = 'coloraxis')
                fig_g_tau_b_m1 = plot_heatmap(df_tau_b_m1, 'log_w0', 'log_A', 'gamma')
                fig_g_tau_b_m1.update_traces(coloraxis = 'coloraxis2')
                fig_gk_tau_b_m1 = plot_heatmap(df_tau_b_m1, 'log_w0', 'log_A', 'gamma_over_k')
                fig_gk_tau_b_m1.update_traces(coloraxis = 'coloraxis3')

                fig.add_trace(fig_k_tau_b_m1.data[0], row = 1+l, col = 1)
                fig.add_trace(fig_g_tau_b_m1.data[0], row = 1+l, col = 2)
                fig.add_trace(fig_gk_tau_b_m1.data[0], row = 1+l, col = 3)

            fig.update_xaxes(
                title = r"$\huge{\log \omega_0}$",
            )
            fig.update_yaxes(
                title = r"$\huge{\log B}$",
            )            

            fig.update_layout(
                margin = dict(l = 400, r = 400, t = 400, b = 400),
                width = 400 * 3 + 400*2,
                height = 400 * len(tau_b_m1_list) + 400*2,
                coloraxis=dict(
                    colorbar = dict(
                        # x = 1, yanchor = 'top', y = 1, 
                        # len = 1/2 - 1/10, 
                        # tickmode="array",
                        tickcolor = 'black',
                        # tickvals = [0, 0.2, 0.4, 0.6, 0.8, 1],
                        # ticktext = ["0", "0.2", "0.4", "0.6", "0.8", "1"],
                        ticks = "outside",
                        tickwidth = 3,
                        ticklen = 12,
                        ),
                    colorscale = dark_purple_scale,
                ),
                coloraxis2=dict(
                    colorbar = dict(
                        # x = 1, yanchor = 'bottom', y = 0, 
                        # len = 1/2 - 1/10,
                        tickcolor = 'black',
                        # dtick = 1,
                        ticks = "outside",
                        tickwidth = 3,
                        ticklen = 12,                        
                        ),
                ),
                coloraxis3=dict(
                    colorbar = dict(
                        # x = 1, yanchor = 'bottom', y = 0, 
                        # len = 1/2 - 1/10,
                        tickcolor = 'black',
                        # dtick = 1,
                        ticks = "outside",
                        tickwidth = 3,
                        ticklen = 12,                        
                        ),
                ),                
                )
            
        elif panel_nbr == 1:

            # Select only for one value of the amplitude
            eps = 1e-12
            A = 1e-5
            df_A = df.loc[np.abs(df['A'] - A) < eps]

            # Select only for one value of the frequency
            eps = 1e-12
            w0 = 1e-8
            df_w0 = df.loc[np.abs(df['w0'] - w0) < eps]

            fig = make_subplots(rows = 2, cols = len(tau_b_m1_list))
    
            # Vary w0, fix A
            for l in range(len(tau_b_m1_list)):
                tau_b_m1 = tau_b_m1_list[l]
                df_tau_b_m1 =  df_A.loc[np.abs(df_A['tau_b_m1'] - tau_b_m1) < eps]

                fig.add_scatter(x = df_tau_b_m1['w0'], y = df_tau_b_m1['k'], row = 1, col = 1 + l, mode = 'markers', marker_color = "black")
                fig.add_scatter(x = df_tau_b_m1['w0'], y = df_tau_b_m1['gamma'], row = 1, col = 1 + l, mode = 'markers', marker_color = cb_orange)
                fig.add_scatter(x = df_tau_b_m1['w0'], y = df_tau_b_m1['gamma_over_k'], row = 1, col = 1 + l, mode = 'markers', marker_color = cb_purple)      

            # Vary A, fix w0
            for l in range(len(tau_b_m1_list)):
                tau_b_m1 = tau_b_m1_list[l]
                df_tau_b_m1 =  df_w0.loc[np.abs(df_w0['tau_b_m1'] - tau_b_m1) < eps]

                fig.add_scatter(x = df_tau_b_m1['A'], y = df_tau_b_m1['k'], row = 2, col = 1 + l, mode = 'markers', marker_color = "black")
                fig.add_scatter(x = df_tau_b_m1['A'], y = df_tau_b_m1['gamma'], row = 2, col = 1 + l, mode = 'markers', marker_color = cb_orange)
                fig.add_scatter(x = df_tau_b_m1['A'], y = df_tau_b_m1['gamma_over_k'], row = 2, col = 1 + l, mode = 'markers', marker_color = cb_purple)                                              

            fig.update_layout(
                margin = dict(l = 400, r = 400, t = 400, b = 400),
                width = 400*len(tau_b_m1_list) + 400*2,
                height = 400 + 400*2,
            )

    # Direct Inference Shear (elasticity + viscosity): k, gamma
    elif fig_nbr == 19:

        if panel_nbr == 0:
            print("")

    fig.write_image(fig_filename)
    fig.vs_show()
