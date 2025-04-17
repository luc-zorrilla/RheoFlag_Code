""" This file will be used to verify adequation of the model to analytical results. 
Paper-quality figures will also be made to illustrate that. """

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
pio.templates.default = "figure_template"
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
    # Panel b - Counterbend for two different regimes

# Figure 3: simulations for bending elasticity + bending viscosity, clamped axoneme
    # Panel a - relaxation
# Figure 4: simulations for shear elasticity + shear viscosity, clamped axoneme
    # Panel a - relaxation for varying shear viscosities, Sp4 = 1
    # Panel b - relaxation for varying Sp4, tau_s = 1e3

fig_nbr = 4
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
            output_folder, N, taus_b, tau_s, init_conf, Beta, gamma, n_L, m_L, A, w0, Sp4, k0, Lambdas, Zetas, X_flow_field_string, T_span, T_eval, T_sim_max, T_sim, X_flow_field, X_0, method = list(solver_dict.values())
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
                output_folder, N, taus_b, tau_s, init_conf, Beta, gamma, n_L, m_L, A, w0, Sp4, k0, Lambdas, Zetas, X_flow_field_string, T_span, T_eval, T_sim_max, T_sim, X_flow_field, X_0, method = list(solver_dict.values())
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
            output_folder, N, taus_b, tau_s, init_conf, Beta, gamma, n_L, m_L, A, w0, Sp4, k0, Lambdas, Zetas, X_flow_field_string, T_span, T_eval, T_sim_max, T_sim, X_flow_field, X_0, method = list(solver_dict.values())
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
                output_folder, N, taus_b, tau_s, init_conf, Beta, gamma, n_L, m_L, A, w0, Sp4, k0, Lambdas, Zetas, X_flow_field_string, T_span, T_eval, T_sim_max, T_sim, X_flow_field, X_0, method = list(solver_dict.values())
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

    # Bending + Shear elasticity, no viscosity, clamped axoneme.
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
                output_folder, N, taus_b, tau_s, init_conf, Beta, gamma, n_L, m_L, A, w0, Sp4, k0, Lambdas, Zetas, X_flow_field_string, T_span, T_eval, T_sim_max, T_sim, X_flow_field, X_0, method = list(solver_dict.values())
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

        # Counterbend
        if panel_nbr == 1:

            folder_name = "C:/Users/Luc/Documents/PhD_Large_files/RheoFlag/Model/Output/"
            folder_name += "ProximalBend_NoFlow/BendingShearElasticity_Clamped_Counterbend/"
            id_filenames = ["20250416-012526174965_N_10_tau_s_0_taus_b_0_Beta_1.0_gamma_2_A_0_w0_0_Sp4_1.0_k0_10000000000.0","20250416-011925928066_N_10_tau_s_0_taus_b_0_Beta_0.1_gamma_2_A_0_w0_0_Sp4_1.0_k0_10000000000.0"]

            fig = make_subplots(rows = len(id_filenames), cols = 1, subplot_titles = [r"$\huge{\beta = 1, \lambda = 2}$", r"$\huge{\beta = 0.1, \lambda = 0.5}$"], shared_xaxes=True)

            for l in range(len(id_filenames)):

                id_filename = id_filenames[l]
                metadata_filename = folder_name + 'metadata_' + id_filename +'.json'
                data_filename = folder_name + 'data_' + id_filename + '.csv'
                solver_dict = get_metadata(metadata_filename)
                output_folder, N, taus_b, tau_s, init_conf, Beta, gamma, n_L, m_L, A, w0, Sp4, k0, Lambdas, Zetas, X_flow_field_string, T_span, T_eval, T_sim_max, T_sim, X_flow_field, X_0, method = list(solver_dict.values())
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

    # Bending elasticity, varying bending viscosity, clamped axoneme
    elif fig_nbr == 3:
        
        if panel_nbr == 0:

            folder_name = "C:/Users/Luc/Documents/PhD_Large_files/RheoFlag/Model/Output/"
            folder_name += "SecondBend_Relaxation/BendingElasticity_Clamped_VaryingBendingViscosity/"

            id_filenames = ["20250416-040627112877_N_10_tau_s_0_taus_b_0_Beta_0_gamma_2_A_0_w0_0_Sp4_1.0_k0_10000000000.0", "20250416-040627112877_N_10_tau_s_0_taus_b_0.001_Beta_0_gamma_2_A_0_w0_0_Sp4_1.0_k0_10000000000.0", "20250416-040627112877_N_10_tau_s_0_taus_b_1.0_Beta_0_gamma_2_A_0_w0_0_Sp4_1.0_k0_10000000000.0", "20250416-040627114882_N_10_tau_s_0_taus_b_1000.0_Beta_0_gamma_2_A_0_w0_0_Sp4_1.0_k0_10000000000.0", "20250416-045354068702_N_10_tau_s_0_taus_b_1000000.0_Beta_0_gamma_2_A_0_w0_0_Sp4_1.0_k0_10000000000.0"]             

            fig = make_subplots(rows = len(id_filenames), cols = 1, subplot_titles = [""], shared_xaxes=True)
            fig_2 = make_subplots(rows = len(id_filenames), cols = 1, shared_xaxes=True)

            for l in range(len(id_filenames)):

                id_filename = id_filenames[l]
                metadata_filename = folder_name + 'metadata_' + id_filename +'.json'
                data_filename = folder_name + 'data_' + id_filename + '.csv'
                solver_dict = get_metadata(metadata_filename)
                output_folder, N, taus_b, tau_s, init_conf, Beta, gamma, n_L, m_L, A, w0, Sp4, k0, Lambdas, Zetas, X_flow_field_string, T_span, T_eval, T_sim_max, T_sim, X_flow_field, X_0, method = list(solver_dict.values())
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

                for k in range(indices_s.shape[0]):
                    fig.add_scatter(x = X3N(X[:,indices_s[k]])[:N, 0], y = X3N(X[:,indices_s[k]])[N:2*N, 0], marker_color = c[k], row = 1 + l, col = 1)
                # fig.update_xaxes()
                # fig.update_yaxes()

                # Figure 2
                X_3N = np.array([X3N(X[:,t]) for t in range(X.shape[1])]).squeeze().transpose()
                # print("X_3N.shape", X_3N.shape)
                x_tip = np.array([X_3N[N-1,:], X_3N[2*N-1,:]])
                # print("x_tip.shape", x_tip.shape)
                # exit()
                fig_2.add_scatter(x = np.arange(x_tip.shape[1])*delta_t, y = x_tip[1,:]/x_tip[1,0], row = 1+l, col =1, name = "tau_b = " + str(tau_b))
            fig_2.update_xaxes(zeroline = True)
            fig_2.update_yaxes(zeroline = True)
            fig_2.vs_show()


            fig.update_layout(width = 800, height = 300 * len(id_filenames), showlegend = False)

    # Shear elasticity, varying shear viscosity, clamped axoneme
    elif fig_nbr == 4:

        if panel_nbr == 0:

            folder_name = "C:/Users/Luc/Documents/PhD_Large_files/RheoFlag/Model/Output/"
            folder_name += "SecondBend_Relaxation/ShearElasticity_Clamped_VaryingShearViscosity/"

            id_filenames = [
                "20250417-032313300149_N_10_tau_s_0.001_taus_b_0_Beta_1000.0_gamma_2_A_0_w0_0_Sp4_1.0_k0_10000000000.0", 
                "20250417-032313302154_N_10_tau_s_0.01_taus_b_0_Beta_1000.0_gamma_2_A_0_w0_0_Sp4_1.0_k0_10000000000.0", 
                "20250417-032313302154_N_10_tau_s_0.1_taus_b_0_Beta_1000.0_gamma_2_A_0_w0_0_Sp4_1.0_k0_10000000000.0", 
                "20250417-032215850350_N_10_tau_s_1.0_taus_b_0_Beta_1000.0_gamma_2_A_0_w0_0_Sp4_1.0_k0_10000000000.0", 
                "20250417-031556517975_N_10_tau_s_10.0_taus_b_0_Beta_1000.0_gamma_2_A_0_w0_0_Sp4_1.0_k0_10000000000.0", 
                "20250417-031611475771_N_10_tau_s_100.0_taus_b_0_Beta_1000.0_gamma_2_A_0_w0_0_Sp4_1.0_k0_10000000000.0", 
                "20250417-031641350954_N_10_tau_s_1000.0_taus_b_0_Beta_1000.0_gamma_2_A_0_w0_0_Sp4_1.0_k0_10000000000.0", 
                "20250417-031709623421_N_10_tau_s_1000000.0_taus_b_0_Beta_1000.0_gamma_2_A_0_w0_0_Sp4_1.0_k0_10000000000.0"]             

            fig = make_subplots(rows = len(id_filenames), cols = 1, subplot_titles = [""], shared_xaxes=True)
            fig_2 = make_subplots(rows = len(id_filenames), cols = 1, shared_xaxes=False)

            for l in range(len(id_filenames)):

                id_filename = id_filenames[l]
                metadata_filename = folder_name + 'metadata_' + id_filename +'.json'
                data_filename = folder_name + 'data_' + id_filename + '.csv'
                solver_dict = get_metadata(metadata_filename)
                output_folder, N, taus_b, tau_s, init_conf, Beta, gamma, n_L, m_L, A, w0, Sp4, k0, Lambdas, Zetas, X_flow_field_string, T_span, T_eval, T_sim_max, T_sim, X_flow_field, X_0, method = list(solver_dict.values())
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

                for k in range(indices_s.shape[0]):
                    fig.add_scatter(x = X3N(X[:,indices_s[k]])[:N, 0], y = X3N(X[:,indices_s[k]])[N:2*N, 0], marker_color = c[k], row = 1 + l, col = 1)
                # fig.update_xaxes()
                # fig.update_yaxes()

                # Figure 2
                X_3N = np.array([X3N(X[:,t]) for t in range(X.shape[1])]).squeeze().transpose()
                # print("X_3N.shape", X_3N.shape)
                x_tip = np.array([X_3N[N-1,:], X_3N[2*N-1,:]])
                # print("x_tip.shape", x_tip.shape)
                # exit()
                fig_2.add_scatter(x = np.arange(x_tip.shape[1])*delta_t, y = x_tip[1,:]/x_tip[1,0], row = 1+l, col =1, name = "tau_s = " + str(tau_s))
            fig_2.update_xaxes(zeroline = True)
            fig_2.update_yaxes(zeroline = True)
            fig_2.update_layout(width = 800, height = 300 * len(id_filenames), showlegend = True)    
            fig_2.vs_show()


            fig.update_layout(width = 800, height = 300 * len(id_filenames), showlegend = False)

        # Varying Sp4 should change the shape of the relaxation curve
        if panel_nbr == 1:

            folder_name = "C:/Users/Luc/Documents/PhD_Large_files/RheoFlag/Model/Output/"
            folder_name += "SecondBend_Relaxation/ShearElasticity_Clamped_VaryingShearViscosity/VaryingSp4/"
            
            id_filenames = [
                # "20250417-051505285138_N_10_tau_s_1000.0_taus_b_0_Beta_1000.0_gamma_2_A_0_w0_0_Sp4_0.001_k0_10000000000.0",
                # "20250417-051505377790_N_10_tau_s_1000.0_taus_b_0_Beta_1000.0_gamma_2_A_0_w0_0_Sp4_0.01_k0_10000000000.0",
                # "20250417-051505377790_N_10_tau_s_1000.0_taus_b_0_Beta_1000.0_gamma_2_A_0_w0_0_Sp4_0.1_k0_10000000000.0",
                "20250417-051505442897_N_10_tau_s_1000.0_taus_b_0_Beta_1000.0_gamma_2_A_0_w0_0_Sp4_1.0_k0_10000000000.0",
                # "20250417-051505480475_N_10_tau_s_1000.0_taus_b_0_Beta_1000.0_gamma_2_A_0_w0_0_Sp4_10.0_k0_10000000000.0",
                # "20250417-051505687573_N_10_tau_s_1000.0_taus_b_0_Beta_1000.0_gamma_2_A_0_w0_0_Sp4_100.0_k0_10000000000.0",
                "20250417-051505760460_N_10_tau_s_1000.0_taus_b_0_Beta_1000.0_gamma_2_A_0_w0_0_Sp4_1000.0_k0_10000000000.0",
                "20250417-052302018931_N_10_tau_s_1000.0_taus_b_0_Beta_1000.0_gamma_2_A_0_w0_0_Sp4_10000.0_k0_10000000000.0",
                "20250417-052302018931_N_10_tau_s_1000.0_taus_b_0_Beta_1000.0_gamma_2_A_0_w0_0_Sp4_100000.0_k0_10000000000.0",
                "20250417-052302018931_N_10_tau_s_1000.0_taus_b_0_Beta_1000.0_gamma_2_A_0_w0_0_Sp4_1000000.0_k0_10000000000.0",
                ]                          

            fig = make_subplots(rows = len(id_filenames), cols = 1, subplot_titles = [""], shared_xaxes=True)
            fig_2 = make_subplots(rows = len(id_filenames), cols = 1, shared_xaxes=False)

            for l in range(len(id_filenames)):

                id_filename = id_filenames[l]
                metadata_filename = folder_name + 'metadata_' + id_filename +'.json'
                data_filename = folder_name + 'data_' + id_filename + '.csv'
                solver_dict = get_metadata(metadata_filename)
                output_folder, N, taus_b, tau_s, init_conf, Beta, gamma, n_L, m_L, A, w0, Sp4, k0, Lambdas, Zetas, X_flow_field_string, T_span, T_eval, T_sim_max, T_sim, X_flow_field, X_0, method = list(solver_dict.values())
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

                for k in range(indices_s.shape[0]):
                    fig.add_scatter(x = X3N(X[:,indices_s[k]])[:N, 0], y = X3N(X[:,indices_s[k]])[N:2*N, 0], marker_color = c[k], row = 1 + l, col = 1)
                # fig.update_xaxes()
                # fig.update_yaxes()

                # Figure 2
                X_3N = np.array([X3N(X[:,t]) for t in range(X.shape[1])]).squeeze().transpose()
                # print("X_3N.shape", X_3N.shape)
                x_tip = np.array([X_3N[N-1,:], X_3N[2*N-1,:]])
                # print("x_tip.shape", x_tip.shape)
                # exit()
                fig_2.add_scatter(x = np.arange(x_tip.shape[1])*delta_t, y = x_tip[1,:]/x_tip[1,0], row = 1+l, col =1, name = "Sp4 = " + str(Sp4))
            fig_2.update_xaxes(zeroline = True)
            fig_2.update_yaxes(zeroline = True)
            fig_2.update_layout(width = 800, height = 300 * len(id_filenames), showlegend = True)    
            fig_2.vs_show()


            fig.update_layout(width = 800, height = 300 * len(id_filenames), showlegend = False)



    fig.write_image(fig_filename)
    fig.vs_show()
