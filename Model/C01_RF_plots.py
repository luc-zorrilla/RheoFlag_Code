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

# Figure 0: analytical results for pure bending, clamped axoneme, with uniform vertical flow
    # Panel a - stroboscopic view of the filament along with analytical solution
    # Panel b - kinetic energy vs time, showing equilibrium is reached
    # Panel c - L2(Equilibrium - Analytical solution) vs 1/N, showing convergence to the solution

# Figure 1: analytical results for pure bending, clamped axoneme, with point force at the tip
    # Panel a - stroboscopic view of the filament along with analytical solution
    # Panel b - kinetic energy vs time, showing equilibrium is reached
    # Panel c - L2(Equilibrium - Analytical solution) vs 1/N, showing convergence to the solution

# Figure 2: simulations for shear elasticity + bending elasticity, no viscosity, clamped axoneme
    # Panel a - stroboscopic view of the filament in 3 different regimes of shear / bending.
    # Panel b - Counterbend 

fig_nbr = 0
panel_nbr = 0

if __name__ == '__main__':

    fig_filename = writing_dir + "fig" + "_" + str(fig_nbr) + "_" + "panel" + "_" + str(panel_nbr) + ".pdf"

    # Pure Bending, uniform vertical flow.
    if fig_nbr == 0:

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
                    fig.add_scatter(x = X3N(X[:,indices_s[k]])[:N, 0], y = X3N(X[:,indices_s[k]])[N:2*N, 0], marker_color = c[k], line_width = 1)
                fig.add_scatter(x = X_3N_eq[:n_eq,0][X_3N_eq[:n_eq,0]<=N-1], y = X_3N_eq[n_eq:2*n_eq,0][X_3N_eq[:n_eq,0]<=N-1], marker_color = cb_dark_red, line_width = 2)
                # fig.update_xaxes()
                # fig.update_yaxes()
                fig.update_layout(showlegend = True, margin = dict(l = 100, r = 100, t = 100, b = 100))

            # Convergence to equilibrium - kinetic energy decays (log-log plot)
            elif panel_nbr == 1:
                
                # Kinetic energy
                K = KineticEnergy(X, N, T_eval) # t
                fig = go.Figure()
                fig.add_scatter(x = T_eval, y = K, line_width = 2)
                for k in range(indices_s.shape[0]):
                    fig.add_scatter(x = [T_eval[indices_s[k]]], y = [K[indices_s[k]]], marker_color = c[k], mode = 'markers', marker_size = 6)
                fig.update_xaxes(type = 'log')
                fig.update_yaxes(type = 'log')
                fig.update_layout(showlegend = False, margin = dict(l = 100, r = 100, t = 100, b = 100)) 

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
            fig.add_scatter(x = 1 / np.arange(5, 60, 5), y = L2_error_array, mode = 'markers', marker_color = cb_dark_red, marker_size = 6)
            fig.update_xaxes(title = 'log(1/N)', type = 'log')
            fig.update_yaxes(title = 'log L2(sim, analytical) / L2(analytical)', type = 'log')
            fig.update_layout(margin = dict(l = 100, r = 100, t = 100, b = 100))
            
    # Pure Bending, vertical point force at the tip
    elif fig_nbr == 1:

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
            n_strobes = 20

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
                output_folder, N, taus_b, tau_s, init_conf, Beta, gamma, n_L, m_L, A, w0, Sp4, k0, Lambdas, Zetas, X_flow_field_string, T_span, T_eval, T_sim_max, T_sim, X_flow_field, X_0, method = list(solver_dict.values())
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
            fig.add_scatter(x = 1 / np.arange(5, 50, 5), y = L2_error_array, mode = 'markers', marker_color = cb_dark_red)
            fig.update_xaxes(title = 'log(1/N)', type = 'log')
            fig.update_yaxes(title = 'log L2(sim, analytical) / L2(analytical)', type = 'log')
            fig.update_layout(margin = dict(l = 100, r = 100, t = 100, b = 100))
            
    # Bending + Shear elasticity, no viscosity, clamped axoneme.
    elif fig_nbr == 2:

        if panel_nbr == 0:

            folder_name = "C:/Users/Luc/Documents/PhD_Large_files/RheoFlag/Model/Output/"
            folder_name += "ProximalBend_NoFlow/BendingElasticity_Clamped_VaryingShearBending/"

            # id_filenames = ["20250319_Beta_1e-3_123447509552", "20250319_Beta_1e-2_123447503793", "20250319_Beta_1e-1_123447444425", "20250318_Beta_1e0_073258315126", "20250318_Beta_1e1_073258408813", "20250318_Beta_1e2_073258444692", "20250318_Beta_1e3_073258451669"]
            id_filenames = ["20250319_Beta_1e-3_123447509552", "20250318_Beta_1e0_073258315126", "20250318_Beta_1e3_073258451669"]            

            fig = make_subplots(rows = len(id_filenames), cols = 1, subplot_titles = [r"$\text{Bending _ } \beta = 10^{-3}$", r"$\text{Bending + shear _ } \beta = 1$", r"$\text{Shear _ } \beta = 10^{-3}$"], shared_xaxes=True)

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
                n_strobes = 200
                t_s = T_eval[-1] / n_strobes                
                
                condition = (T_eval_norm >= 0)
                min_index = np.arange(T_eval_norm.shape[0])[condition][0]
                max_index = np.arange(T_eval_norm.shape[0])[condition][-1]
                indices_s = StroboscopicView(T_eval_norm[min_index:max_index], t_s = t_s)
                c = sample_colorscale('matter_r', np.linspace(0, 1, num = indices_s.shape[0]))[::-1]        

                for k in range(indices_s.shape[0]):
                    fig.add_scatter(x = X3N(X[:,indices_s[k]])[:N, 0], y = X3N(X[:,indices_s[k]])[N:2*N, 0], marker_color = c[k], row = 1 + l, col = 1)
                # fig.update_xaxes()
                # fig.update_yaxes()
            fig.update_layout(width = 800, height = 300 * len(id_filenames), showlegend = False)

        # Counterbend
        if panel_nbr == 1:

            folder_name = "C:/Users/Luc/Documents/PhD_Large_files/RheoFlag/Model/Output/"
            folder_name += "ProximalBend_NoFlow/BendingShearElasticity_Clamped_Counterbend/"
            id_filenames = ["20250319_beta_1e0_lambda_1e0_021057191004", "20250319_beta_1e-1_lambda_5e-1_025640335717"]


            fig = make_subplots(rows = len(id_filenames), cols = 1, subplot_titles = [r"$\beta = 1, \lambda = 1$", r"$\beta = 0.1, \lambda = 0.5$"], shared_xaxes=True)

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
                n_strobes = 100
                t_s = T_eval[-1] / n_strobes                
                
                condition = (T_eval_norm >= 0)
                min_index = np.arange(T_eval_norm.shape[0])[condition][0]
                max_index = np.arange(T_eval_norm.shape[0])[condition][-1]
                indices_s = StroboscopicView(T_eval_norm[min_index:max_index], t_s = t_s)
                c = sample_colorscale('matter_r', np.linspace(0, 1, num = indices_s.shape[0]))[::-1]        

                for k in range(indices_s.shape[0]):
                    fig.add_scatter(x = X3N(X[:,indices_s[k]])[:N, 0], y = X3N(X[:,indices_s[k]])[N:2*N, 0], marker_color = c[k], row = 1 + l, col = 1)
                # fig.update_xaxes()
                # fig.update_yaxes()
            fig.update_layout(width = 800, height = 300 * len(id_filenames), showlegend = False)



    fig.write_image(fig_filename)
    fig.vs_show()
