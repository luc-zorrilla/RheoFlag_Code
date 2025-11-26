""" This script is used to analyze harmonic responses of the model. """

################################################################################
### Libraries

import os
import sys
from pathlib import Path
plot_functions_folder = Path.home() / 'Work' / 'Miscellaneous' / 'Code'
sys.path.insert(0, plot_functions_folder)
import glob

import multiprocessing as mp
from datetime import datetime
from A01_Coarse_grained_axoneme_functions import *
from B01_simulations_analysis import *

import numpy as np
from scipy.signal import find_peaks

from plotting_functions import * 
pio.templates.default = "figure_template"
################################################################################

temp_folder = Path.cwd().joinpath('Model').joinpath('Results').joinpath('Temp')
writing_dir = temp_folder

def get_tip_frequency(solver_dict, X):
    """ Compute the fourier transform of the vertical position of the tip 
    of the filament subject to a periodic vertical flow, and extract the 
    main frequency of the signal as the position of the maximum of the 
    power spectrum. """

    # Get T_eval
    output_folder, N, taus_b, tau_s, init_conf, bool_EI, Beta, gamma, n_L, m_L, A, w0, Sp4, k0, Lambdas, Zetas, X_flow_field_string, T_span, T_eval, T_sim_max, T_sim, X_flow_field, X_0, method = list(solver_dict.values())
    tau_b = taus_b[0]
    
    # Normalize time in flow periods
    delta_t = T_eval[1] - T_eval[0]
    T_eval = np.array(T_eval)
    if (A > 0) & (w0 > 0):
        T_eval_norm = T_eval * w0 / (2*np.pi)
    else:
        T_eval_norm = T_eval

    # TIP values
    X_3N = np.array([X3N(X[:,t]) for t in range(X.shape[1])]).squeeze().transpose()
    x_tip = np.array([X_3N[N-1,:], X_3N[2*N-1,:]])

    # Fourier transform of the tip (y position)
    y_tip_f = np.fft.rfft(x_tip[1,:])
    f_y_tip = np.fft.rfftfreq(x_tip[1,:].shape[0])
    delta_t_norm = T_eval_norm[1] - T_eval_norm[0]
    f_y_tip *= 1/delta_t_norm

    # Power spectrum (square magnitude)
    P_tip = np.abs(y_tip_f)**2 
    
    # Get frequency peak
    peaks, peaks_heights = find_peaks(P_tip, height = 0)
    if len(peaks)>0:
        max_peak = peaks[np.argmax(peaks_heights)]
        f_tip = f_y_tip[max_peak]
    else: # No peak was found
        f_tip = -1
    
    print("f_tip = ", f_tip)
    
    return f_tip

def get_tip_frequency_and_phase(solver_dict, X):
    """ Compute the fourier transform of the vertical position of the tip 
    of the filament subject to a periodic vertical flow, and extract the 
    main frequency of the signal as the position of the maximum of the 
    power spectrum. Extract the phase as defined by the argument of the 
    fourier transform for the main frequency. """

    # Get T_eval
    output_folder, N, taus_b, tau_s, init_conf, bool_EI, Beta, gamma, n_L, m_L, A, w0, Sp4, k0, Lambdas, Zetas, X_flow_field_string, T_span, T_eval, T_sim_max, T_sim, X_flow_field, X_0, method = list(solver_dict.values())
    tau_b = taus_b[0]
    
    # Normalize time in flow periods
    delta_t = T_eval[1] - T_eval[0]
    T_eval = np.array(T_eval)
    if (A > 0) & (w0 > 0):
        T_eval_norm = T_eval * w0 / (2*np.pi)
    else:
        T_eval_norm = T_eval

    # TIP values
    X_3N = np.array([X3N(X[:,t]) for t in range(X.shape[1])]).squeeze().transpose()
    x_tip = np.array([X_3N[N-1,:], X_3N[2*N-1,:]])

    # Fourier transform of the tip (y position)
    y_tip_f = np.fft.rfft(x_tip[1,:])
    f_y_tip = np.fft.rfftfreq(x_tip[1,:].shape[0])
    delta_t_norm = T_eval_norm[1] - T_eval_norm[0]
    f_y_tip *= 1/delta_t_norm

    # Power spectrum (square magnitude)
    P_y_tip = np.abs(y_tip_f)**2
    # Phase
    phi_y_tip = np.angle(y_tip_f) / (2*np.pi)

    max_peak = np.argmax(P_y_tip)
    f_tip = f_y_tip[max_peak]
    
    npoints = 1 # corresponds here to 0.01Hz precision on both sides of the peak.
    min_index = np.max((0, max_peak-npoints))
    max_index = np.min((phi_y_tip.shape[0], max_peak+1+npoints))
    phi_tip = np.mean(phi_y_tip[min_index:max_index])
    delta_phi_tip = np.std(phi_y_tip[min_index:max_index], ddof = 1)/np.sqrt(max_index - min_index + 1)

    ## TEST ##
    eps = 1e-20
    if (np.abs(w0-1e-9)**2 + np.abs(tau_b - 1e0)**2) < eps:
        fig = make_subplots(rows = 2, cols = 1, subplot_titles = ["Power spectrum", "Phase spectrum"], shared_xaxes=True)
        fig.add_scatter(x = f_y_tip, y = P_y_tip, row = 1, col = 1, name = "f_tip = " + str(np.round(f_tip,2)))
        fig.add_scatter(x = f_y_tip, y = phi_y_tip, row = 2, col = 1, name = "phi_tip = " + str(np.round(phi_tip,2)))
        fig.update_layout(
            title = "(w0, tau_b) = " + str(np.round(w0,2)) + " " + str(np.round(tau_b,2)), 
            margin = dict(l = 200, r = 200, t = 200, b = 200),
            width = 500 + 400,
            height = 500 + 400)
        fig.vs_show()
    ##########
    
    print("f_tip = ", f_tip)
    print("phi_tip = ", phi_tip)
    print("delta_phi_tip = ", delta_phi_tip)
    
    return [f_tip, phi_tip, delta_phi_tip]

def get_tip_maxdeviation_phase(solver_dict, X):
    """ Get the vertical position of the tip of the filament subject to a 
    periodic vertical flow, and extract the phase at which the maximal 
    absolute vertical position is reached. Also returns the max deviation value. 
    """

    # Get T_eval
    output_folder, N, taus_b, tau_s, init_conf, bool_EI, Beta, gamma, n_L, m_L, A, w0, Sp4, k0, Lambdas, Zetas, X_flow_field_string, T_span, T_eval, T_sim_max, T_sim, X_flow_field, X_0, method = list(solver_dict.values())
    tau_b = taus_b[0]
    
    # Normalize time in flow periods
    delta_t = T_eval[1] - T_eval[0]
    T_eval = np.array(T_eval)
    if (A > 0) & (w0 > 0):
        T_eval_norm = T_eval * w0 / (2*np.pi)
    else:
        T_eval_norm = T_eval

    try:
        # TIP values
        X_3N = np.array([X3N(X[:,t]) for t in range(X.shape[1])]).squeeze().transpose()
        x_tip = np.array([X_3N[N-1,:], X_3N[2*N-1,:]])

        # Look at the vertical position
        y_tip = x_tip[1,:]

        # Look at the second half of the signal (starting at a multiple of the flow period)
        y_tip = y_tip[T_eval_norm >= T_eval_norm[-1] // 2]
        T_eval_norm = T_eval_norm[T_eval_norm >= T_eval_norm[-1] // 2]

        # Get values at max, min and zero
        min_y_tip = np.min(y_tip)
        max_y_tip = np.max(y_tip)
        zero_y_tip = np.min(np.abs(y_tip))

        # Get phases at these values and average to get a single phase

        # Define the right tolerance to find relevant points
        eps = 1e-10
        count = np.count_nonzero((np.abs((y_tip-min_y_tip)) / np.abs(min_y_tip)) < eps)
        n_periods = np.round(T_eval_norm[-1] - T_eval_norm[0])
        while count > n_periods + 1:
            eps /= 10
            count = np.count_nonzero((np.abs((y_tip-min_y_tip)) / np.abs(min_y_tip)) < eps)
        while count <= n_periods-1:
            eps *= 1.1
            count = np.count_nonzero((np.abs((y_tip-min_y_tip)) / np.abs(min_y_tip)) < eps)

        min_y_tip_index = (np.abs((y_tip-min_y_tip)) / np.abs(min_y_tip)) < eps
        max_y_tip_index = (np.abs((y_tip-max_y_tip)) / np.abs(max_y_tip)) < eps
        zerom_y_tip_index = (np.abs(y_tip)<eps) & (np.hstack((np.diff(y_tip),0))<0)
        zerop_y_tip_index = (np.abs(y_tip)<eps) & (np.hstack((np.diff(y_tip),0))>0)

        phi_min_y_tip = np.mean(T_eval_norm[min_y_tip_index]%1)
        phi_max_y_tip = np.mean(T_eval_norm[max_y_tip_index]%1)
        phi_zerom_y_tip = np.mean(T_eval_norm[zerom_y_tip_index]%1)
        phi_zerop_y_tip = np.mean(T_eval_norm[zerop_y_tip_index]%1)

        delta_phi_min_y_tip = np.std(T_eval_norm[min_y_tip_index]%1, ddof = 1) / T_eval_norm[min_y_tip_index].shape[0]
        delta_phi_max_y_tip = np.std(T_eval_norm[max_y_tip_index]%1, ddof = 1) / T_eval_norm[max_y_tip_index].shape[0]
        delta_phi_zerom_y_tip = np.std(T_eval_norm[zerom_y_tip_index]%1, ddof = 1) / T_eval_norm[zerom_y_tip_index].shape[0]
        delta_phi_zerop_y_tip = np.std(T_eval_norm[zerop_y_tip_index]%1, ddof = 1) / T_eval_norm[zerop_y_tip_index].shape[0]

        print("phi_min_y_tip: ", phi_min_y_tip)

        return [min_y_tip, max_y_tip, phi_min_y_tip, phi_max_y_tip, phi_zerom_y_tip, phi_zerop_y_tip, delta_phi_min_y_tip, delta_phi_max_y_tip, delta_phi_zerom_y_tip, delta_phi_zerop_y_tip]
    
    except IndexError:
        return [np.nan]*10

if __name__ == "__main__":

    Bending_EV = False # Whether to look at bending elasticity + viscosity
    Shear_EV = True # Whether to look at shear elasticity + viscosity

    if Bending_EV:

        folder_name = Path('..').joinpath('Model').joinpath('Output').resolve()
        folder_name /= "StraightLine_PeriodicFlow"
        folder_name /= "BendingElasticity_Clamped_VaryingBendingViscosity" #KnownBehaviourTest/"
        folder_name /= "VaryingFrequencyAmplitude"

        filenames = glob.glob(folder_name + '*.json')
        id_filenames = [os.path.basename(filename).removeprefix("metadata_").removesuffix(".json") for filename in filenames]

        # Do the previous steps globally
        columns = ['w0', 'taus_b', 'A']

        # df = observable_dataframe(folder_name, id_filenames, columns, get_tip_frequency_and_phase, obs_type = 'both')
        # df.columns = [*df.columns[:-1], 'f_and_phi_tip']
        # df['f_tip'] = df.apply(lambda x: x['f_and_phi_tip'][0], axis = 1)
        # df['phi_tip'] = df.apply(lambda x: x['f_and_phi_tip'][1], axis = 1)
        # df['delta_phi_tip'] = df.apply(lambda x: x['f_and_phi_tip'][2], axis = 1)
        # df['tau_b_m1'] = df.apply(lambda x: 1/x['taus_b'][0], axis = 1)
        # for o in ['tau_b_m1', 'w0']:
        #     df['log_'+o] = df.apply(lambda x: np.log(x[o]), axis = 1)

        # dataframe_filename = folder_name + "fourier" + ".csv"
        # df.to_csv(dataframe_filename)

        df = observable_dataframe(folder_name, id_filenames, columns, get_tip_maxdeviation_phase, obs_type = 'both')
        df.columns = [*df.columns[:-1], 'maxdev_and_phi_tip']
        Lo = ['min_y_tip', 'max_y_tip', 'phi_min_y_tip', 'phi_max_y_tip', 'phi_zerom_y_tip', 'phi_zerop_y_tip', 'delta_phi_min_y_tip', 'delta_phi_max_y_tip', 'delta_phi_zerom_y_tip', 'delta_phi_zerop_y_tip']
        for k in range(len(Lo)):
            df[Lo[k]] = df.apply(lambda x: x['maxdev_and_phi_tip'][k], axis = 1)
        df['tau_b_m1'] = df.apply(lambda x: 1/x['taus_b'][0], axis = 1)
        for o in ['tau_b_m1', 'w0', 'A']:
            df['log_'+o] = df.apply(lambda x: np.log(x[o]), axis = 1)

        dataframe_filename = folder_name + "maxdev" + ".csv"
        df.to_csv(dataframe_filename)
    
    if Shear_EV:

        folder_name = folder_name = Path('..').joinpath('Model').joinpath('Output').resolve()
        folder_name /= "StraightLine_PeriodicFlow"
        folder_name /= "ShearElasticity_Clamped_VaryingShearViscosity"
        folder_name /= "LargeAmplitude"
        filenames = glob.glob(folder_name + '*.json')
        id_filenames = [os.path.basename(filename).removeprefix("metadata_").removesuffix(".json") for filename in filenames]

        # Do the previous steps globally
        columns = ['w0', 'tau_s', 'A']

        # df = observable_dataframe(folder_name, id_filenames, columns, get_tip_frequency_and_phase, obs_type = 'both')
        # df.columns = [*df.columns[:-1], 'f_and_phi_tip']
        # df['f_tip'] = df.apply(lambda x: x['f_and_phi_tip'][0], axis = 1)
        # df['phi_tip'] = df.apply(lambda x: x['f_and_phi_tip'][1], axis = 1)
        # df['delta_phi_tip'] = df.apply(lambda x: x['f_and_phi_tip'][2], axis = 1)
        # df['tau_b_m1'] = df.apply(lambda x: 1/x['taus_b'][0], axis = 1)
        # for o in ['tau_b_m1', 'w0']:
        #     df['log_'+o] = df.apply(lambda x: np.log(x[o]), axis = 1)

        # dataframe_filename = folder_name + "fourier" + ".csv"
        # df.to_csv(dataframe_filename)

        df = observable_dataframe(folder_name, id_filenames, columns, get_tip_maxdeviation_phase, obs_type = 'both')
        df.columns = [*df.columns[:-1], 'maxdev_and_phi_tip']
        Lo = ['min_y_tip', 'max_y_tip', 'phi_min_y_tip', 'phi_max_y_tip', 'phi_zerom_y_tip', 'phi_zerop_y_tip', 'delta_phi_min_y_tip', 'delta_phi_max_y_tip', 'delta_phi_zerom_y_tip', 'delta_phi_zerop_y_tip']
        for k in range(len(Lo)):
            df[Lo[k]] = df.apply(lambda x: x['maxdev_and_phi_tip'][k], axis = 1)
        df['tau_s_m1'] = df.apply(lambda x: 1/x['tau_s'], axis = 1)
        for o in ['tau_s_m1', 'w0', 'A']:
            df['log_'+o] = df.apply(lambda x: np.log(x[o]), axis = 1)

        dataframe_filename = folder_name + "maxdev" + ".csv"
        df.to_csv(dataframe_filename)
    