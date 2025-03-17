""" 
This script gathers all computed simulations complying with some conditions and 
creates a simulation time matrix in parameter space, which can then be plotted.
In fact, this can be put into a function and even generalized to any functional 
f(data) or f(metadata), or even more generally to any function f.
"""

from Coarse_grained_axoneme_functions import * 

import numpy as np
import pandas as pd

import glob
import os

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

def fetch_files(directory, metadata_condition, data_condition = None):
    """ Fetches all the filename ids in directory which comply with some conditions, 
    which can be either on the data or the metadata. Returns a list of ids. 
    
    Note that metadata_condition and data_condition are functions of respectively 
    the metadata (as a dict) and the data (as a numpy array). 
    
    Example:
    def metadata_condition_0(solver_dict):
        output_folder, N, taus_b, init_conf, Beta, gamma, n_L, m_L, A, w0, Sp4, Lambdas, Zetas, X_flow_field_string, T_span, T_eval, T_sim_max, T_sim, X_flow_field, X_0, method = list(solver_dict.values())

        bool_condition = (N == 10) & ("SmoothCurve" in init_conf) & (gamma == 2) & (method == 'RK45')

        return bool_condition
    
    """

    ids_list = []

    # Go through metadata files in directory
    metadata_files = glob.glob(directory + "*.json")
    for metadata_file in metadata_files:
        base_name = os.path.basename(metadata_file)
        base_id = base_name.strip('metadata_.json')

        # Get metadata
        solver_dict = get_metadata(metadata_file)
        
        # Conditions on metadata
        bool_metadata_condition = metadata_condition(solver_dict)

        if data_condition == None:
            if bool_metadata_condition:
                ids_list.append(base_id)
        else:
            # Get data
            data_file = 'data_' + base_id + '.csv'
            X = get_data(data_file)

            # Conditions on data
            bool_data_condition = data_condition(X)

            if (bool_metadata_condition & bool_data_condition):
                ids_list.append(base_id)

    return ids_list

def observable_1D_dataframe(directory, ids_list, columns, observable, obs_type = 'metadata'):
    """ Takes as input a list of ids corresponding to simulations and makes a 
    dataframe in parameter space specified by columns where the value of the dataframe 
    is an observable, which will simply be for now a number. 
    Remark: this function could be generalized to observable_ND_dataframe() where the 
    observable is not a necessarily a number but could be anything. 
    
    - observable is a function of metadata (as a dict), of the data (as 
    a numpy array), or of both and can be specified in obs_type: ['metadata', 
    'data', 'both'].

        Example:
        def observable_0(solver_dict, None):
            output_folder, N, taus_b, init_conf, Beta, gamma, n_L, m_L, A, w0, Sp4, Lambdas, Zetas, X_flow_field_string, T_span, T_eval, T_sim_max, T_sim, X_flow_field, X_0, method = list(solver_dict.values())

            return T_sim

    - parameters specified by axes will form the axes of the matrix, in the form 
    of a list of strings such as ['Sp4', 'Beta'] or ['Sp4', 'Beta', 'tau_b', 'A', 'w0']
    
    """

    # Create dataframe
    df = pd.DataFrame(columns = columns + [str(observable)])

    for base_id in ids_list:
        
        # Metadata
        metadata_file = directory + 'metadata_' + base_id + '.json'
        solver_dict = get_metadata(metadata_file)

        # Get column values (except observable)
        col_values = [solver_dict[column] for column in columns]

        # Get observable
        if obs_type in ['metadata']:
            obs = observable(solver_dict, None)
        else:
            data_file = 'data_' + base_id + '.csv'
            X = get_data(data_file)            
            if obs_type in ['data']:
                obs = observable(None, X)
            elif obs_type in ['both']:
                obs = observable(solver_dict, X)
        col_values.append(obs)

        # Put into table
        df.loc[len(df)] = col_values

    return df

def plot_2D(df, column_0, column_1):
    """ Plot two columns of a dataframe in a 2D axis. """

    fig = go.Figure()
    fig.add_scatter(x = df[column_0], y = df[column_1], mode = 'markers')
    fig.update_xaxes(title = column_0)
    fig.update_yaxes(title = column_1)
    # fig.vs_show()

    return fig

def plot_heatmap(df, column_0, column_1, column_2):
    """ Plot two columns of a dataframe in a 2D axis and a third column as a 
    heatmap. """

    fig = go.Figure(data = go.Heatmap(
        x = df[column_0],
        y = df[column_1],
        z = df[column_2],
        ))
    
    fig.update_xaxes(title = column_0)
    fig.update_yaxes(title = column_1)
    # fig.vs_show()

    return fig


if __name__ == '__main__':

    # Fetch files satisfying required conditions

    sim_directory = "C:/Users/Luc/Documents/PhD_Large_files/RheoFlag/Model/Output/" # "StraightLine_PeriodicFlow_Radau/"

    def metadata_condition_0(solver_dict):
        output_folder, N, taus_b, init_conf, Beta, gamma, n_L, m_L, A, w0, Sp4, k0, Lambdas, Zetas, X_flow_field_string, T_span, T_eval, T_sim_max, T_sim, X_flow_field, X_0, method = list(solver_dict.values())

        # bool_condition = (N == 10) & (Beta == 0) & ("SmoothCurve" in init_conf) & (gamma == 2) & ((A < 1e-2) & (w0 < 1e-2)) & (method == 'Radau')
        eps = 1e-6
        bool_condition = (N == 10) & (np.abs(taus_b[0] - 0) < eps) & (np.abs(Beta - 0) < eps) & ("ProximalBend" in init_conf) & (gamma == 2) & ((np.abs(A - 0) < eps) & (np.abs(w0 - 0) < eps)) & (np.abs(Sp4 - 1e0) < eps) & (np.abs(k0 - 1e1) < eps) & (method == 'Radau')

        return bool_condition

    ids_list = fetch_files(sim_directory, metadata_condition_0, None)
    print("# files: ", len(ids_list))

    print(ids_list[0])
    exit()

    # Compute observable on these files and put this new data into a dataframe

    columns = ['Sp4', 'taus_b']

    def simulation_time(solver_dict, X = None):
        output_folder, N, taus_b, init_conf, Beta, gamma, n_L, m_L, A, w0, Sp4, Lambdas, Zetas, X_flow_field_string, T_span, T_eval, T_sim_max, T_sim, X_flow_field, X_0, method = list(solver_dict.values())
        return T_sim    
    
    df = observable_1D_dataframe(sim_directory, ids_list, columns, simulation_time, obs_type = 'metadata')
    df.columns = [*df.columns[:-1], 'T_sim']
    df['tau_b'] = df.apply(lambda x: x['taus_b'][0], axis = 1)

    # Plot T_sim against Sp4
    plot_2D(df, 'tau_b', 'T_sim').show()
    plot_heatmap(df, 'Sp4', 'tau_b', 'T_sim').show()