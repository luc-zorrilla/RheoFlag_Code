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

bool_figures = True # Figures for paper
fig_nbr = 1
panel_nbr = 0

if bool_figures:

    fig_filename = writing_dir + "fig" + "_" + str(fig_nbr) + "_" + "panel" + "_" + str(panel_nbr) + ".pdf"

    # Pure Bending, uniform vertical flow.
    if fig_nbr == 0:

        folder_name = "C:/Users/Luc/Documents/PhD_Large_files/RheoFlag/Model/Output/"
        folder_name += "AnalyticalComparisons/PureBending_Clamped_UniformVerticalFlow/"

        id_filenames = ["20250318_N_5_035950022795", "20250318_N_10_025957138534", "20250318_N_15_035950205972", "20250318_N_20_030248179364", "20250318_N_25_035950257314", "20250318_N_30_030512701827", "20250318_N_35_041039114888"]

        # Equilibrium solution - stroboscopic view and analytical solution (N = 35) + kinetic energy
        if panel_nbr in [0, 1]:

            id_filename = id_filenames[-1]
            metadata_filename = folder_name + 'metadata_' + id_filename +'.json'
            data_filename = folder_name + 'data_' + id_filename + '.csv'
            solver_dict = get_metadata(metadata_filename)
            output_folder, N, taus_b, init_conf, Beta, gamma, n_L, m_L, A, w0, Sp4, k0, Lambdas, Zetas, X_flow_field_string, T_span, T_eval, T_sim_max, T_sim, X_flow_field, X_0, method = list(solver_dict.values())
            X = get_data(data_filename) # s, t
            X_3N_final = X3N(X[:,-1])

            T_eval = np.array(T_eval)
            if (A > 0) & (w0 > 0):
                T_eval_norm = T_eval * w0 / (2*np.pi)
            else:
                T_eval_norm = T_eval
            X_flow = A*np.sin(w0*T_eval)            

            # Stroboscopic view
            if N == 10:
                t_s = 500
            elif N == 20:
                t_s = 5000
            elif N in [5, 15, 25, 30]:
                t_s = 50000
            elif N in [35]:
                t_s = 250000
            condition = (T_eval_norm >= 0)
            min_index = np.arange(T_eval_norm.shape[0])[condition][0]
            max_index = np.arange(T_eval_norm.shape[0])[condition][-1]
            indices_s = StroboscopicView(T_eval_norm[min_index:max_index], t_s = t_s)
            c = sample_colorscale('BuPu', np.linspace(0, 1, num = indices_s.shape[0]))[::-1]

            # Analytical Equilbrium Profile
            if panel_nbr == 0:
                
                n_eq = 1000
                X_3N_eq = CheckEquilibrium(N, A, gamma, Sp4, n_L = n_L, Lambdas=lambdas, conditions = "vertical_flow_uniform", n_eq = n_eq)
                # ["vertical_point_tip", "vertical_density_tip", "vertical_density_uniform", "vertical_flow_uniform"]

                fig = go.Figure()
                for k in range(indices_s.shape[0]):
                    fig.add_scatter(x = X3N(X[:,indices_s[k]])[:N, 0], y = X3N(X[:,indices_s[k]])[N:2*N, 0], marker_color = c[k])
                fig.add_scatter(x = X_3N_eq[:n_eq,0], y = X_3N_eq[n_eq:2*n_eq,0], marker_color = cb_orange)
                # fig.update_xaxes()
                # fig.update_yaxes()
                fig.update_layout(showlegend = True)            

            # Convergence to equilibrium - kinetic energy decays (log-log plot)
            elif panel_nbr == 1:
                
                # Kinetic energy
                K = KineticEnergy(X, N, T_eval) # t
                fig = go.Figure()
                fig.add_scatter(x = T_eval, y = K)
                for k in range(indices_s.shape[0]):
                    fig.add_scatter(x = [T_eval[indices_s[k]]], y = [K[indices_s[k]]], marker_color = c[k], mode = 'markers')
                fig.update_xaxes(type = 'log')
                fig.update_yaxes(type = 'log')             

        # L2 Error (Solution - analytical solution) for varying N (1/N, log(relative L2 error))
        elif panel_nbr == 2:

            L2_error_array = np.zeros((len(id_filenames)))
            for l in range(len(id_filenames)):
                id_filename = id_filenames[l]
                metadata_filename = folder_name + 'metadata_' + id_filename +'.json'
                data_filename = folder_name + 'data_' + id_filename + '.csv'

                solver_dict = get_metadata(metadata_filename)
                output_folder, N, taus_b, init_conf, Beta, gamma, n_L, m_L, A, w0, Sp4, k0, Lambdas, Zetas, X_flow_field_string, T_span, T_eval, T_sim_max, T_sim, X_flow_field, X_0, method = list(solver_dict.values())
                if T_sim == np.inf:
                    print('Not solved. Error: ', X)
                    exit()
                tau_b = taus_b[0]

                # Equilibrium profile
                X = get_data(data_filename) # s, t
                X_3N_final = X3N(X[:,-1])

                # Analytical Equilbrium Profile
                X_3N_eq = CheckEquilibrium(N, A, gamma, Sp4, n_L = n_L, Lambdas=Lambdas, conditions = "vertical_flow_uniform", n_eq = N)

                step = N // 5
                L2_error = np.linalg.norm(X_3N_final[::step] - X_3N_eq[::step]) / np.linalg.norm(X_3N_final[::step])
                L2_error_array[l] = L2_error

            fig = go.Figure()
            fig.add_scatter(x = 1 / np.array([5, 10, 15, 20, 25, 30, 35]), y = L2_error_array, mode = 'markers', marker_color = cb_dark_red)
            fig.update_xaxes(title = '1/N')
            fig.update_yaxes(title = 'log L2(sim, analytical) / L2(analytical)', type = 'log')
            
    # Pure Bending, vertical point force at the tip
    elif fig_nbr == 1:

        folder_name = "C:/Users/Luc/Documents/PhD_Large_files/RheoFlag/Model/Output/"
        folder_name += "AnalyticalComparisons/PureBending_Clamped_TipVerticalPointForce/"

        id_filenames = ["20250318_N_5_050718197323", "20250318_N_10_050718279494", "20250318_N_15_050546109420", "20250318_N_20_050718330565", "20250318_N_25_050718350355", "20250318_N_30_051643250389", "20250318_N_35_052122084524"]

        # Equilibrium solution - stroboscopic view and analytical solution (N = 35) + kinetic energy
        if panel_nbr in [0, 1]:

            id_filename = id_filenames[-1]
            metadata_filename = folder_name + 'metadata_' + id_filename +'.json'
            data_filename = folder_name + 'data_' + id_filename + '.csv'
            solver_dict = get_metadata(metadata_filename)
            output_folder, N, taus_b, init_conf, Beta, gamma, n_L, m_L, A, w0, Sp4, k0, Lambdas, Zetas, X_flow_field_string, T_span, T_eval, T_sim_max, T_sim, X_flow_field, X_0, method = list(solver_dict.values())
            X = get_data(data_filename) # s, t
            X_3N_final = X3N(X[:,-1])

            T_eval = np.array(T_eval)
            if (A > 0) & (w0 > 0):
                T_eval_norm = T_eval * w0 / (2*np.pi)
            else:
                T_eval_norm = T_eval
            X_flow = A*np.sin(w0*T_eval)            

            # Stroboscopic view
            if N == 10:
                t_s = 500
            elif N == 20:
                t_s = 5000
            elif N in [5, 15, 25, 30]:
                t_s = 50000
            elif N in [35]:
                t_s = 250000
            condition = (T_eval_norm >= 0)
            min_index = np.arange(T_eval_norm.shape[0])[condition][0]
            max_index = np.arange(T_eval_norm.shape[0])[condition][-1]
            indices_s = StroboscopicView(T_eval_norm[min_index:max_index], t_s = t_s)
            c = sample_colorscale('BuPu', np.linspace(0, 1, num = indices_s.shape[0]))[::-1]

            # Analytical Equilbrium Profile
            if panel_nbr == 0:
                
                n_eq = 1000
                X_3N_eq = CheckEquilibrium(N, A, gamma, Sp4, n_L = n_L, Lambdas=Lambdas, conditions = "vertical_point_tip", n_eq = n_eq)
                # ["vertical_point_tip", "vertical_density_tip", "vertical_density_uniform", "vertical_flow_uniform"]

                fig = go.Figure()
                for k in range(indices_s.shape[0]):
                    fig.add_scatter(x = X3N(X[:,indices_s[k]])[:N, 0], y = X3N(X[:,indices_s[k]])[N:2*N, 0], marker_color = c[k])
                fig.add_scatter(x = X_3N_eq[:n_eq,0], y = X_3N_eq[n_eq:2*n_eq,0], marker_color = cb_orange)
                # fig.update_xaxes()
                # fig.update_yaxes()
                fig.update_layout(showlegend = True)            

            # Convergence to equilibrium - kinetic energy decays (log-log plot)
            elif panel_nbr == 1:
                
                # Kinetic energy
                K = KineticEnergy(X, N, T_eval) # t
                fig = go.Figure()
                fig.add_scatter(x = T_eval, y = K)
                for k in range(indices_s.shape[0]):
                    fig.add_scatter(x = [T_eval[indices_s[k]]], y = [K[indices_s[k]]], marker_color = c[k], mode = 'markers')
                fig.update_xaxes(type = 'log')
                fig.update_yaxes(type = 'log')             

        # L2 Error (Solution - analytical solution) for varying N (1/N, log(relative L2 error))
        elif panel_nbr == 2:

            L2_error_array = np.zeros((len(id_filenames)))
            for l in range(len(id_filenames)):
                id_filename = id_filenames[l]
                metadata_filename = folder_name + 'metadata_' + id_filename +'.json'
                data_filename = folder_name + 'data_' + id_filename + '.csv'

                solver_dict = get_metadata(metadata_filename)
                output_folder, N, taus_b, init_conf, Beta, gamma, n_L, m_L, A, w0, Sp4, k0, Lambdas, Zetas, X_flow_field_string, T_span, T_eval, T_sim_max, T_sim, X_flow_field, X_0, method = list(solver_dict.values())
                if T_sim == np.inf:
                    print('Not solved. Error: ', X)
                    exit()
                tau_b = taus_b[0]

                # Equilibrium profile
                X = get_data(data_filename) # s, t
                X_3N_final = X3N(X[:,-1])

                # Analytical Equilbrium Profile
                X_3N_eq = CheckEquilibrium(N, A, gamma, Sp4, n_L = n_L, Lambdas=Lambdas, conditions = "vertical_flow_uniform", n_eq = N)

                step = N // 5
                L2_error = np.linalg.norm(X_3N_final[::step] - X_3N_eq[::step]) / np.linalg.norm(X_3N_final[::step])
                L2_error_array[l] = L2_error

            fig = go.Figure()
            fig.add_scatter(x = 1 / np.array([5, 10, 15, 20, 25, 30, 35]), y = L2_error_array, mode = 'markers', marker_color = cb_dark_red)
            fig.update_xaxes(title = '1/N')
            fig.update_yaxes(title = 'log L2(sim, analytical) / L2(analytical)', type = 'log')
            

    fig.write_image(fig_filename)
    fig.vs_show()
