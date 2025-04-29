""" This file will be used to verify adequation of the model to analytical results. 
Paper-quality figures will also be made to illustrate that. """

################################################################################
### Libraries

import os
import sys
plot_functions_folder = "C:/Users/Luc/Documents/MEGAsync/Code"
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

temp_folder = "C:/Users/Luc/Documents/MEGAsync/PhD/RheoFlag/Results/Temp/"
writing_dir = temp_folder

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

# Figure 4: simulations for bending elasticity + bending viscosity, clamped axoneme
    # Panel a - relaxation for varying bending viscosity

# Figure 5: simulations for shear elasticity + shear viscosity, clamped axoneme
    # Panel a - relaxation for varying shear viscosities

# Figure 6: simulations for a periodic flow, for a clamped axoneme with bending elasticity + bending viscosity
    # Panel a - tip movement for varying bending viscosity and flow frequency: phase
    # Panel b - tip movement for varying bending viscosity and flow frequency: max amplitude

# Figure 7: simulations for a periodic flow, for a clamped axoneme with bending elasticity + bending viscosity
    # Panel a - transect for tau_b (<<, >>) tau_{b,f} with phase and max amplitude

# Figure 8: simulations for a periodic flow, for a clamped axoneme with shear elasticity + shear viscosity
    # Panel a - tip movement for varying shear viscosity and flow frequency: phase
    # Panel b - tip movement for varying shear viscosity and flow frequency: max amplitude

# Figure 9: simulations for a periodic flow, for a clamped axoneme with shear elasticity + shear viscosity
    # Panel a - transects for tau_s (<<, >>) tau_{s,f} with phase and max amplitude

fig_nbr = 7
panel_nbr = 0
if __name__ == '__main__':

    fig_filename = writing_dir + "fig" + "_" + str(fig_nbr) + "_" + "panel" + "_" + str(panel_nbr) + ".pdf"

    # Pure Bending, vertical point force at the tip
    if fig_nbr == 0:

        folder_name = "C:/Users/Luc/Documents/PhD_Large_files/RheoFlag/Model/Output/"
        folder_name += "AnalyticalComparisons/PureBending_Clamped_TipVerticalPointForce/"

        id_filenames = ["20250415-120922564262_N_5_tau_s_0_taus_b_0_Beta_0_gamma_2_A_0_w0_0_Sp4_1.0_k0_10000000000.0", "20250415-120922601564_N_10_tau_s_0_taus_b_0_Beta_0_gamma_2_A_0_w0_0_Sp4_1.0_k0_10000000000.0", "20250415-120922621337_N_15_tau_s_0_taus_b_0_Beta_0_gamma_2_A_0_w0_0_Sp4_1.0_k0_10000000000.0", "20250415-120922659220_N_20_tau_s_0_taus_b_0_Beta_0_gamma_2_A_0_w0_0_Sp4_1.0_k0_10000000000.0", "20250415-120922662966_N_25_tau_s_0_taus_b_0_Beta_0_gamma_2_A_0_w0_0_Sp4_1.0_k0_10000000000.0", "20250415-120922679084_N_30_tau_s_0_taus_b_0_Beta_0_gamma_2_A_0_w0_0_Sp4_1.0_k0_10000000000.0", "20250415-120922716650_N_35_tau_s_0_taus_b_0_Beta_0_gamma_2_A_0_w0_0_Sp4_1.0_k0_10000000000.0", "20250415-120922805353_N_40_tau_s_0_taus_b_0_Beta_0_gamma_2_A_0_w0_0_Sp4_1.0_k0_10000000000.0", "20250415-122526334494_N_45_tau_s_0_taus_b_0_Beta_0_gamma_2_A_0_w0_0_Sp4_1.0_k0_10000000000.0"]

        # Equilibrium solution - stroboscopic view and analytical solution (N = 35) + kinetic energy
        if panel_nbr in [0, 1]:

            id_filename = id_filenames[-1]
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
            n_strobes = 50

            condition = (T_eval_norm >= 0)
            min_index = np.arange(T_eval_norm.shape[0])[condition][0]
            max_index = np.arange(T_eval_norm.shape[0])[condition][-1]
            indices_s = StroboscopicView(T_eval_norm[min_index:max_index], n_strobes=n_strobes)
            c = sample_colorscale('matter_r', np.linspace(0, 1, num = indices_s.shape[0]))[::-1]

            # Analytical Equilbrium Profile
            if panel_nbr == 0:
                
                n_eq = 1000
                X_3N_eq = CheckEquilibrium(N, A, gamma, Sp4, n_L = n_L, Lambdas=Lambdas, conditions = "vertical_point_tip", n_eq = n_eq)
                # ["vertical_point_tip", "vertical_density_tip", "vertical_density_uniform", "vertical_flow_uniform"]

                fig = go.Figure()
                for k in range(indices_s.shape[0]):
                    fig.add_scatter(x = X3N(X[:,indices_s[k]])[:N, 0]/N, y = X3N(X[:,indices_s[k]])[N:2*N, 0], marker_color = c[k], line_width = 1, showlegend = (k in [0, indices_s.shape[0]-1]), name = [r"$\huge{\boldsymbol{y}_0}$", r"$\huge{\boldsymbol{y}_\text{eq}}$"][k > 0])
                fig.add_scatter(x = X_3N_eq[:n_eq,0][X_3N_eq[:n_eq,0]<=N-1]/N, y = X_3N_eq[n_eq:2*n_eq,0][X_3N_eq[:n_eq,0]<=N-1], marker_color = cb_dark_red, line_width = 2, name = r"$\huge{\boldsymbol{y}^\star}$")

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
                    range = [0,3e-2],
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
                fig.add_scatter(x = T_eval, y = K, line_width = 2)
                for k in range(indices_s.shape[0]):
                    fig.add_scatter(x = [T_eval[indices_s[k]]], y = [K[indices_s[k]]], marker_color = c[k], mode = 'markers', marker_size = 6)

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
                output_folder, N, taus_b, tau_s, init_conf, bool_EI, Beta, gamma, n_L, m_L, A, w0, Sp4, k0, Lambdas, Zetas, X_flow_field_string, T_span, T_eval, T_sim_max, T_sim, X_flow_field, X_0, method = list(solver_dict.values())
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
            fig.add_scatter(x = 1 / np.arange(5, 50, 5), y = L2_error_array, mode = 'markers', marker_color = cb_dark_red, marker_size = 6)

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

        folder_name = "C:/Users/Luc/Documents/PhD_Large_files/RheoFlag/Model/Output/"
        folder_name += "AnalyticalComparisons/PureBending_Clamped_UniformVerticalFlow/"

        id_filenames = ["20250415-111946800826_N_5_tau_s_0_taus_b_0_Beta_0_gamma_2_A_1e-08_w0_0_Sp4_1.0_k0_10000000000.0", "20250415-111946825696_N_10_tau_s_0_taus_b_0_Beta_0_gamma_2_A_1e-08_w0_0_Sp4_1.0_k0_10000000000.0", "20250415-111946863118_N_15_tau_s_0_taus_b_0_Beta_0_gamma_2_A_1e-08_w0_0_Sp4_1.0_k0_10000000000.0", "20250415-111946894785_N_20_tau_s_0_taus_b_0_Beta_0_gamma_2_A_1e-08_w0_0_Sp4_1.0_k0_10000000000.0", "20250415-111946920750_N_25_tau_s_0_taus_b_0_Beta_0_gamma_2_A_1e-08_w0_0_Sp4_1.0_k0_10000000000.0", "20250415-111946929903_N_30_tau_s_0_taus_b_0_Beta_0_gamma_2_A_1e-08_w0_0_Sp4_1.0_k0_10000000000.0", "20250415-111946930900_N_35_tau_s_0_taus_b_0_Beta_0_gamma_2_A_1e-08_w0_0_Sp4_1.0_k0_10000000000.0", "20250415-112025796860_N_40_tau_s_0_taus_b_0_Beta_0_gamma_2_A_1e-08_w0_0_Sp4_1.0_k0_10000000000.0", "20250415-112025855976_N_45_tau_s_0_taus_b_0_Beta_0_gamma_2_A_1e-08_w0_0_Sp4_1.0_k0_10000000000.0", "20250415-112025990187_N_50_tau_s_0_taus_b_0_Beta_0_gamma_2_A_1e-08_w0_0_Sp4_1.0_k0_10000000000.0", "20250415-112025990187_N_55_tau_s_0_taus_b_0_Beta_0_gamma_2_A_1e-08_w0_0_Sp4_1.0_k0_10000000000.0"]

        # Equilibrium solution - stroboscopic view and analytical solution (N = 35) + kinetic energy
        if panel_nbr in [0, 1]:

            id_filename = id_filenames[-1]
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
                # ["vertical_point_tip", "vertical_density_tip", "vertical_density_uniform", "vertical_flow_uniform"]

                fig = go.Figure()
                for k in range(indices_s.shape[0]):
                    fig.add_scatter(x = X3N(X[:,indices_s[k]])[:N, 0]/N, y = X3N(X[:,indices_s[k]])[N:2*N, 0], marker_color = c[k], line_width = 1, showlegend = (k in [0, indices_s.shape[0]-1]), name = [r"$\huge{\boldsymbol{y}(0)}$", r"$\huge{\boldsymbol{y}_{eq}}$"][k > 0])
                fig.add_scatter(x = X_3N_eq[:n_eq,0][X_3N_eq[:n_eq,0]<=N-1]/N, y = X_3N_eq[n_eq:2*n_eq,0][X_3N_eq[:n_eq,0]<=N-1], marker_color = cb_dark_red, line_width = 2, name = r"$\huge{\boldsymbol{y}^\star}$")

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
                    range = [0,3e-2],
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
                fig.add_scatter(x = T_eval, y = K, line_width = 2)
                for k in range(indices_s.shape[0]):
                    fig.add_scatter(x = [T_eval[indices_s[k]]], y = [K[indices_s[k]]], marker_color = c[k], mode = 'markers', marker_size = 6)

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
                output_folder, N, taus_b, tau_s, init_conf, bool_EI, Beta, gamma, n_L, m_L, A, w0, Sp4, k0, Lambdas, Zetas, X_flow_field_string, T_span, T_eval, T_sim_max, T_sim, X_flow_field, X_0, method = list(solver_dict.values())
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
            fig.add_scatter(x = 1/np.arange(5, 60, 5), y = L2_error_array, mode = 'markers', marker_color = cb_dark_red, marker_size = 6)

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

            folder_name = "C:/Users/Luc/Documents/PhD_Large_files/RheoFlag/Model/Output/"
            folder_name += "ProximalBend_NoFlow/BendingElasticity_Clamped_VaryingShearBending/"

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

            folder_name = "C:/Users/Luc/Documents/PhD_Large_files/RheoFlag/Model/Output/"
            folder_name += "ProximalBend_NoFlow/BendingShearElasticity_Clamped_Counterbend/"
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
        
        if panel_nbr == 0:

            folder_name = "C:/Users/Luc/Documents/PhD_Large_files/RheoFlag/Model/Output/"
            folder_name += "SecondBend_Relaxation/BendingElasticity_Clamped_VaryingBendingViscosity/"

            filenames = glob.glob(folder_name + '*.json')
            id_filenames = [os.path.basename(filename).removeprefix("metadata_").removesuffix(".json") for filename in filenames]

            fig = make_subplots(rows = len(id_filenames), cols = 1, shared_xaxes=False)

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

                # Figure 2
                X_3N = np.array([X3N(X[:,t]) for t in range(X.shape[1])]).squeeze().transpose()
                x_tip = np.array([X_3N[N-1,:], X_3N[2*N-1,:]])
                fig.add_scatter(x = np.arange(x_tip.shape[1])*delta_t, y = x_tip[1,:]/x_tip[1,0], row = 1+l, col =1, name = r"$\huge{\tau_b = " + sci_notation(tau_b, 0,0) + "}$", marker_color = "black")

                # Fit to exponentially decreasing function
                popt, pcov = curve_fit(f = d_exp, xdata = T_eval/delta_t, ydata = x_tip[1,:]/x_tip[1,0])
                popt[0] *= delta_t
                pcov[0,0] *= delta_t**2
                pcov[0,1] *= delta_t
                pcov[1,0] *= delta_t
                print("tau_b, popt, pcov: ", tau_b, popt, pcov)

            fig.update_xaxes(zeroline = True, title = r"$\huge{t}$")
            x_ticks = np.arange(0,6)*1e3
            x_ticks_text = [r"$\huge{" + sci_notation(x_tick, 0, 0) + "}$" for x_tick in x_ticks]       
            for row in [1,2,3]:
                fig.update_xaxes(
                    range = [0,5e3],
                    row = row,
                    col = 1,
                    tickmode = "array",
                    tickvals = x_ticks,
                    ticktext = x_ticks_text
                )
            x_ticks = np.arange(0,6)*1e6
            x_ticks_text = [r"$\huge{" + sci_notation(x_tick, 0, 0) + "}$" for x_tick in x_ticks] 
            fig.update_xaxes(
                range = [0,5e6],
                row = 4,
                col = 1,
                tickmode = "array",
                tickvals = x_ticks,
                ticktext = x_ticks_text                
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
                height = 300 * len(id_filenames) + 400, 
                showlegend = True,
                )

    # Shear elasticity, varying shear viscosity, clamped axoneme - relaxation
    elif fig_nbr == 5:

        if panel_nbr == 0:

            folder_name = "C:/Users/Luc/Documents/PhD_Large_files/RheoFlag/Model/Output/"
            folder_name += "SecondBend_Relaxation/ShearElasticity_Clamped_VaryingShearViscosity/"

            id_filenames = [
                "20250427-083206365151_N_10_tau_s_0_taus_b_0_bool_EI_False_Beta_1.0_gamma_2_A_0_w0_0_Sp4_1_k0_10000000000000.0",
                "20250427-083206379138_N_10_tau_s_0.06_taus_b_0_bool_EI_False_Beta_1.0_gamma_2_A_0_w0_0_Sp4_1_k0_10000000000000.0",
                "20250427-083206383138_N_10_tau_s_60.0_taus_b_0_bool_EI_False_Beta_1.0_gamma_2_A_0_w0_0_Sp4_1_k0_10000000000000.0",
                "20250427-083228620175_N_10_tau_s_60000.0_taus_b_0_bool_EI_False_Beta_1.0_gamma_2_A_0_w0_0_Sp4_1_k0_10000000000000.0",
                ]

            fig = make_subplots(rows = len(id_filenames), cols = 1, shared_xaxes=False)

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

                # Figure 2
                X_3N = np.array([X3N(X[:,t]) for t in range(X.shape[1])]).squeeze().transpose()
                x_tip = np.array([X_3N[N-1,:], X_3N[2*N-1,:]])
                fig.add_scatter(x = np.arange(x_tip.shape[1])*delta_t, y = x_tip[1,:]/x_tip[1,0], row = 1+l, col =1, name = r"$\huge{\tau_s = " + sci_notation(tau_s, 0,0) + "}$", marker_color = "black")

                # Fit to exponentially decreasing function
                popt, pcov = curve_fit(f = d_exp, xdata = T_eval/delta_t, ydata = x_tip[1,:]/x_tip[1,0])
                popt[0] *= delta_t
                pcov[0,0] *= delta_t**2
                pcov[0,1] *= delta_t
                pcov[1,0] *= delta_t
                print("tau_s, popt, pcov: ", tau_s, popt, pcov)

            fig.update_xaxes(zeroline = True, title = r"$\huge{t}$")
            x_ticks = np.arange(0,35,40)*1e1
            x_ticks_text = [r"$\huge{" + sci_notation(x_tick,2,0,2) + "}$" for x_tick in x_ticks]
            for row in [1,2,3]:
                fig.update_xaxes(
                    range = [0,3e2],
                    row = row,
                    col = 1,
                    tickmode = "array",
                    tickvals = x_ticks,
                    ticktext = x_ticks_text
                )
            x_ticks = np.arange(0,40,10)*1e4
            x_ticks_text = [r"$\huge{" + sci_notation(x_tick,2,0,5) + "}$" for x_tick in x_ticks] 
            fig.update_xaxes(
                range = [0,3e5],
                row = 4,
                col = 1,
                tickmode = "array",
                tickvals = x_ticks,
                ticktext = x_ticks_text                
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
                width = 350 + 400, 
                height = 300 * len(id_filenames) + 400, 
                showlegend = True,
                )

    # Bending elasticity, varying bending viscosity, flow frequency and amplitude; clamped axoneme
    elif fig_nbr == 6: # Heatmaps

        folder_name = "C:/Users/Luc/Documents/PhD_Large_files/RheoFlag/Model/Output/"
        folder_name += "StraightLine_PeriodicFlow/BendingElasticity_Clamped_VaryingBendingViscosity/VaryingFrequencyAmplitude/"
        dataframe_filename = folder_name + "maxdev" + ".csv"

        df = pd.read_csv(dataframe_filename)

        # Select only for one value of the amplitude: A = 1e-2
        eps = 1e-6
        A = 1e-2
        df = df.loc[np.abs(df['A']- A) < eps]
        df['log_w0'] = df.apply(lambda x: x['log_w0']/np.log(10), axis = 1)
        df['log_tau_b_m1'] = df.apply(lambda x: x['log_tau_b_m1']/np.log(10), axis = 1)
        df['log_max_y_tip'] = df.apply(lambda x: np.log10(x['max_y_tip']), axis = 1)
    
        if panel_nbr == 0: # Phase

            # # Plot f_tip and phi_tip against tau_b, w0
            fig = plot_heatmap(df, 'log_w0', 'log_tau_b_m1', 'phi_max_y_tip')
            fig.update_layout(
                xaxis_title = r"$\huge{\log{\omega_0}}$",
                yaxis_title = r"$\huge{\log{\tau_b^{-1}}}$",
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
            
        elif panel_nbr == 1: # max_y_tip

            fig = plot_heatmap(df, 'log_w0', 'log_tau_b_m1', 'max_y_tip')
            fig.update_layout(
                xaxis_title = r"$\huge{\log{\omega_0}}$",
                yaxis_title = r"$\huge{\log{\tau_b^{-1}}}$",
                coloraxis_colorbar = dict(
                    title = r"$\huge{y_\text{max}}$",
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

            folder_name = "C:/Users/Luc/Documents/PhD_Large_files/RheoFlag/Model/Output/"
            folder_name += "StraightLine_PeriodicFlow/BendingElasticity_Clamped_VaryingBendingViscosity/VaryingFrequencyAmplitude/"
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
                subplot_titles = [r"$\huge{\tau_b = 10^{-6}}$", r"$\huge{\tau_b = 1}$"], 
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
                title = r"$\huge{\log{y_\text{max}}}$",
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
                title = r"$\huge{\log{y_\text{max}}}$",
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

        folder_name = "C:/Users/Luc/Documents/PhD_Large_files/RheoFlag/Model/Output/"
        folder_name += "StraightLine_PeriodicFlow/ShearElasticity_Clamped_VaryingShearViscosity/"
        folder_name += "VaryingFrequencyAmplitude/"
        dataframe_filename = folder_name + "maxdev" + ".csv"

        df = pd.read_csv(dataframe_filename)

        # Select only for one value of the amplitude: A = 1e-2
        eps = 1e-2
        A = 1e0
        df = df.loc[np.abs(df['A']- A) < eps]
        df['log_w0'] = df.apply(lambda x: x['log_w0']/np.log(10), axis = 1)
        df['log_tau_s_m1'] = df.apply(lambda x: x['log_tau_s_m1']/np.log(10), axis = 1)
        df['log_max_y_tip'] = df.apply(lambda x: np.log10(x['max_y_tip']), axis = 1)

        if panel_nbr == 0:

            fig = plot_heatmap(df, 'log_w0', 'log_tau_s_m1', 'phi_max_y_tip')
            fig.update_layout(
                xaxis_title = r"$\huge{\log{\omega_0}}$",
                yaxis_title = r"$\huge{\log{\tau_s^{-1}}}$",
                coloraxis_colorbar = dict(
                    title = r"$\huge{\phi_\text{max}}$",
                    orientation = 'h',
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
                    title = r"$\huge{y_\text{max}}$",
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

    # Shear elasticity, varying shear viscosity and flow frequency, clamped axoneme
    elif fig_nbr == 9:

        # 2 Transects at tau_s = cte
        if panel_nbr == 0:

            folder_name = "C:/Users/Luc/Documents/PhD_Large_files/RheoFlag/Model/Output/"
            folder_name += "StraightLine_PeriodicFlow/ShearElasticity_Clamped_VaryingShearViscosity/"
            folder_name += "VaryingFrequencyAmplitude/"
            dataframe_filename = folder_name + "maxdev" + ".csv"

            df = pd.read_csv(dataframe_filename)

            # Select only for one value of the amplitude: A = 1e-2
            eps = 1e-3
            A = 1e0
            df = df.loc[np.abs(df['A']- A) < eps]
            df['log_w0'] = df.apply(lambda x: x['log_w0']/np.log(10), axis = 1)
            df['log_tau_s_m1'] = df.apply(lambda x: x['log_tau_s_m1']/np.log(10), axis = 1)
            df['log_max_y_tip'] = df.apply(lambda x: np.log10(x['max_y_tip']), axis = 1)
            
            # Make transects dataframes
            tau_s_m1_small = 1e-3
            df_small = df.loc[np.abs(df['tau_s_m1']- tau_s_m1_small) < eps]
            tau_s_m1_large = 1e3
            df_large = df.loc[np.abs(df['tau_s_m1']- tau_s_m1_large) < eps]

            fig = make_subplots(
                rows = 2, cols = 1, 
                subplot_titles = [r"$\huge{\tau_s = 10^{-3}}$", r"$\huge{\tau_s = 10^3}}$"], 
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
            y_ticks = np.arange(0,6,1)/10
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
                title = r"$\huge{\log{y_\text{max}}}$",
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
                title = r"$\huge{\log{y_\text{max}}}$",
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

            

    fig.write_image(fig_filename)
    fig.vs_show()
