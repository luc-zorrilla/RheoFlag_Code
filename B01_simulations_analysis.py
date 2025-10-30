""" 
This script contains functions useful to analyze single or multiple simulations.

In particular, it gathers all computed simulations complying with some conditions and 
creates a simulation time matrix in parameter space, which can then be plotted.
In fact, this can be put into a function and even generalized to any functional 
f(data) or f(metadata), or even more generally to any function f.
"""

from datetime import datetime
import glob
import os
from pathlib import Path

from A01_Coarse_grained_axoneme_functions import * 
import multiprocessing as mp

import math
import numpy as np
import pandas as pd
import scipy.signal

from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.express.colors import sample_colorscale
import plotly.io as pio

def fetch_files(directory_path, metadata_condition, data_condition = None):
    """ Fetches all the filename ids in directory which comply with some conditions, 
    which can be either on the data or the metadata. Returns a list of ids. 
    
    Note that metadata_condition and data_condition are functions of respectively 
    the metadata (as a dict) and the data (as a numpy array). 
    
    Example:
    def metadata_condition_0(solver_dict):
        output_folder, N, taus_b, tau_s, init_conf, bool_EI, Beta, gamma, n_L, m_L, A, w0, Sp4, k0, Lambdas, Zetas, X_flow_field_string, T_span, T_eval, T_sim_max, T_sim, X_flow_field, X_0, method = list(solver_dict.values())

        bool_condition = (N == 10) & ("SmoothCurve" in init_conf) & (gamma == 2) & (method == 'RK45')

        return bool_condition
    
    """

    ids_list = []

    # Go through metadata files in directory
    metadata_filepaths = directory_path.glob("*.json")
    metadata_files = [str(metadata_filepath) for metadata_filepath in metadata_filepaths]
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

def nondimensionalize(eta, E_b, nu_b, K_s, nu_s, L, N):
    """ Returns non-dimensional parameters (directly used in the model) from physical parameters """
    
    Sp4 = eta * (L/N)**4 / E_b
    tau_b = nu_b / E_b
    beta = K_s * (L/N)**2 / E_b
    tau_s = nu_s/E_s

    return Sp4, tau_b, beta, tau_s

def dimensionalize(Sp4, tau_b, beta, tau_s, eta, N, L):
    """ Returns physical dimensional parameters from non-dimensional parameters(directly used in the model). Requires the knowledge of the filament length L and the drag coefficient eta. """

    E_b = eta * (L/N)**4 / Sp4
    nu_b = tau_b * E_b
    K_s = beta * E_b * (N/L)**2
    nu_s = tau_s * K_s

    return E_b, nu_b, K_s, nu_s

def observable_singlefile(directory_path, base_id, columns, observable, obs_type):
    
    """ Applies an observable to a single file """

    # Metadata
    metadata_file = str(directory_path / ('metadata_' + base_id + '.json'))
    solver_dict = get_metadata(metadata_file)

    # Get column values (except observable)
    col_values = [solver_dict[column] for column in columns]

    # Get observable
    if obs_type in ['metadata']:
        obs = observable(solver_dict, None)
    else:
        data_file =  str(directory_path / ('data_' + base_id + '.csv'))
        X = get_data(data_file)            
        if obs_type in ['data']:
            obs = observable(None, X)
        elif obs_type in ['both']:
            obs = observable(solver_dict, X)
    col_values.append(obs)

    return col_values

def observable_dataframe(directory_path, ids_list, columns, observable, obs_type = 'metadata'):
    """ Takes as input a list of ids corresponding to simulations and makes a 
    dataframe in parameter space specified by columns where the value of the dataframe 
    is an observable, which will simply be for now a number.
    
    - observable is a function of metadata (as a dict), of the data (as 
    a numpy array), or of both and can be specified in obs_type: ['metadata', 
    'data', 'both'].

        Example:
        def observable_0(solver_dict, None):
            output_folder, N, taus_b, init_conf, Beta, gamma, n_L, m_L, A, w0, Sp4, Lambdas, Zetas, X_flow_field_string, T_span, T_eval, T_sim_max, T_sim, X_flow_field, X_0, method = list(solver_dict.values())

            return T_sim

    - parameters specified by columns will form the axes of the matrix, in the form 
    of a list of strings such as ['Sp4', 'Beta'] or ['Sp4', 'Beta', 'tau_b', 'A', 'w0']
    
    """

    # Create dataframe
    df = pd.DataFrame(columns = columns + [str(observable)])

    # Start parallel computation
    pool = mp.Pool(mp.cpu_count())

    for base_id in ids_list:
        
        # Apply in parallel observable for each file
        col_values = pool.apply_async(func = observable_singlefile, args = (directory_path, base_id, columns, observable, obs_type)).get()

        # Put into table
        df.loc[len(df)] = col_values

    pool.close()
    pool.join() # postpones the execution of next line of code until all processes in the queue 

    return df

def observable_list_dataframe(directory_path, ids_list, columns, observable_list, obs_type_list):
    """ Takes as input a list of ids corresponding to simulations and makes a 
    dataframe in parameter space specified by columns where the value of the dataframe 
    is an observable, which will simply be for now a number.
    
    - observable_list is a list of functions of metadata (as a dict), of the data (as 
    a numpy array), or of both and can be specified in obs_type: ['metadata', 
    'data', 'both'].

        Example of an element of observable_list:
        def observable_0(solver_dict, None):
            output_folder, N, taus_b, init_conf, Beta, gamma, n_L, m_L, A, w0, Sp4, Lambdas, Zetas, X_flow_field_string, T_span, T_eval, T_sim_max, T_sim, X_flow_field, X_0, method = list(solver_dict.values())

            return T_sim

    - parameters specified by axes will form the axes of the matrix, in the form 
    of a list of strings such as ['Sp4', 'Beta'] or ['Sp4', 'Beta', 'tau_b', 'A', 'w0']
    
    """

    # Create dataframe
    df = pd.DataFrame(columns = columns + [str(observable)])

    # Start parallel computation
    pool = mp.Pool(mp.cpu_count())

    for base_id in ids_list:
        
        # Metadata
        metadata_file =  str(directory_path / ('netadata_' + base_id + '.json'))
        solver_dict = get_metadata(metadata_file)

        # Get column values (except observable)
        col_values = [solver_dict[column] for column in columns]

        for observable, obs_type in zip(observable_list, obs_type_list):
                
            # Apply in parallel observable for each file
            col_values = pool.apply_async(func = observable_singlefile, args = (directory_path, base_id, columns, observable, obs_type)).get()

            # Put into table
            df.loc[len(df)] = col_values

            pool.close()
            pool.join() # postpones the execution of next line of code until all processes in the queue 

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
        coloraxis = 'coloraxis',
        )) 
    fig.update_xaxes(title = column_0)
    fig.update_yaxes(title = column_1)

    return fig

# def animated_shapes(X, N, T_eval, Sp4):

#     """ Makes videos of shapes (X(t), Y(t)) with time cursor and flow orientation marker. 
#     Time is counted in Sp4 units, which always exist - as long as there is bending elasticity. 
#     Returns the figure ready to be plotted. """

#     fig_dict = dict(data = [], layout = {}, frames = [])

#     # data
#     data_dict = go.Scatter(x=X3N(X[:,0])[:N,0], y=X3N(X[:,0])[N:2*N,0])
#     fig_dict["data"].append(data_dict)

#     # generic layout
#     fig_dict["layout"]["xaxis"] = dict(range = [0, N+1], title = "X")
#     fig_dict["layout"]["yaxis"] = dict(range = [-N, N], title = "Y")
#     fig_dict["layout"]["hovermode"] = "closest"
#     # Buttons and animation updates
#     fig_dict["layout"]["updatemenus"] = [
#         {
#             "buttons": [
#                 {
#                     "args": [None, {"frame": {"duration": 50, "redraw": False},
#                                     "fromcurrent": True, "transition": {"duration": 0}}],
#                     "label": "Play",
#                     "method": "animate"
#                 },
#                 {
#                     "args": [[None], {"frame": {"duration": 0, "redraw": False},
#                                     "mode": "immediate",
#                                     "transition": {"duration": 0}}],
#                     "label": "Pause",
#                     "method": "animate"
#                 }
#             ],
#             "direction": "left",
#             "pad": {"r": 10, "t": 50},
#             "showactive": False,
#             "type": "buttons",
#             "x": 0.1,
#             "xanchor": "right",
#             "y": 0,
#             "yanchor": "top"
#         }
#     ]

#     # Sliders
#     sliders_dict = {
#         "active": 0,
#         "yanchor": "top",
#         "xanchor": "left",
#         "currentvalue": {
#             "font": {"size": 20},
#             "prefix": "Time:",
#             "visible": True,
#             "xanchor": "right"
#         },
#         "transition": {"duration": 0, "easing": "cubic-in-out"},
#         "pad": {"b": 10, "t": 50},
#         "len": 0.9,
#         "x": 0.1,
#         "y": 0,
#         "steps": [] # steps should reflect each frame, so cfilled within time
#     }

#     for k in range(X.shape[1]):
#         # frames
#         frame_dict = dict(data = go.Scatter(x = X3N(X[:,k])[:N, 0], y = X3N(X[:,k])[N:2*N, 0]), name = str(k))
#         fig_dict["frames"].append(frame_dict)

#         # frame-specific layout: slider steps
#         slider_step = dict(
#         args = [[str(k)], dict(frame = dict(duration = 50, redraw = False), mode = "immediate", transition= dict(duration = 0))],
#         label = str(np.round(T_eval[k]/Sp4, 0)) + "in Sp4 units",
#         method = "animate")
#         sliders_dict["steps"].append(slider_step)

#     # Sliders
#     fig_dict["layout"]["sliders"] = [sliders_dict]
#     # Construct final animated figure
#     fig_shapes = go.Figure(fig_dict)
#     return fig_shapes

def KineticEnergy(X, N, T_eval):

    """ Computes approximate kinetic energy from current shape, to check equilibrium.
    Returns a figure of K(t), ready to be plotted. """

    # Kinetic energy calculation
    X_dot = np.zeros((N+2, len(T_eval)))
    K = np.zeros((len(T_eval))) # Kinetic energy
    left_K = np.zeros((len(T_eval)))
    for t_index in range(1, len(T_eval)):
        t = T_eval[t_index]
        X_dot[:, t_index] = (X[:, t_index] - X[:, t_index-1]) / (T_eval[t_index] - T_eval[t_index-1]) 
        K[t_index] = np.linalg.norm(X_dot[2:, t_index])**2 / 2

    return K

def CheckEquilibrium(N, A, gamma, Sp4, n_L = [0,0], Lambdas=[[0,0]], conditions = "None", n_eq = 1000):

    """ Returns a figure with equilibrium position of a shape dynamics and computes analytical solution for small deflection: 
    - vertical point force at tip - "vertical_point_tip"
    - vertical density force on tip - "vertical_density_tip"
    - vertical uniform density force along the beam - "vertical_density_uniform"
    - vertical uniform flow - "vertical_flow_uniform"

    All analytical solutions are computed in the case of an initial horizontal beam, assuming equilibrium and small deflection.
    References for analytical solutions: [Felgner et. al. , Journal of Cell Science, 1996]
    
    Remark: total filament length = N but only N-1 segments are represented (N first points). 

    Returns X_eq with 2 * n_eq points:
        - n_eq points for x = s (small deflection approximation), such that s in [0,N] -> [0,L]
        - n_eq points for y, 
    """

    ## Analytical equilibrium solution
    X_eq = np.zeros((2*n_eq))
    theta_eq = np.zeros((n_eq))

    X_eq[:n_eq] = np.linspace(start = 0, stop = N, num = n_eq) # Arclength
    x_eq_neq = N

    # For a vertical point force at distal end
    if conditions == "vertical_point_tip":
        F = n_L[1]
        X_eq[n_eq:] = (3 * (X_eq[:n_eq]/(N))**2 - (X_eq[:n_eq]/(N))**3 ) * F * ((N)**3) / 6
        y_eq_neq = (3 * (x_eq_neq/(N))**2 - (x_eq_neq/(N))**3 ) * F * ((N)**3) / 6

    # For a density force at distal segment
    elif conditions == "vertical_density_tip":
        F = Lambdas[-1][1]
        X_eq[n_eq:] = (3 * (X_eq[:n_eq]/N)**2 - (X_eq[:n_eq]/N)**3) * F * (N**3) / 6
        y_eq_neq = (3 * (x_eq_neq/N)**2 - (x_eq_neq/N)**3) * F * (N**3) / 6

    # For a uniform vertical force
    elif conditions == "vertical_density_uniform":
        f = Lambdas[0][1]
        X_eq[n_eq:] = ( (X_eq[:n_eq]/N)**4 - 4*(X_eq[:n_eq]/N)**3 + 6*(X_eq[:n_eq]/N)**2 ) * f * (N**4) / 24
        y_eq_neq = ( (x_eq_neq/N)**4 - 4*(x_eq_neq/N)**3 + 6*(x_eq_neq/N)**2 ) * f * (N**4) / 24

    # For a uniform small vertical flow
    elif conditions == "vertical_flow_uniform":
        X_eq[n_eq:] = ( (X_eq[:n_eq]/N)**4 - 4*(X_eq[:n_eq]/N)**3 + 6*(X_eq[:n_eq]/N)**2 ) * A * gamma * Sp4 *(N**4) / 24
        y_eq_neq = ( (x_eq_neq/N)**4 - 4*(x_eq_neq/N)**3 + 6*(x_eq_neq/N)**2 ) * A * gamma * Sp4 *(N**4) / 24
        
    else:
        print("No condition for exact solution has been specified.")
        return NameError
    
    theta_eq[:-1] = np.arctan2(X_eq[n_eq+1:], X_eq[1:n_eq])
    theta_eq[-1] = np.arctan2(y_eq_neq, x_eq_neq)
    X_3N_eq = np.vstack((X_eq.reshape((-1,1)), theta_eq.reshape(-1,1))) # Thetas are filled to zero here!
    return X_3N_eq

def Kymograph(X):

    """ Computes Kymograph of X and mean angle."""

    N = X.shape[0]-2
    M = X.shape[1] # Number of time samples 
    
    Theta = np.zeros((M,N))
    for k in range(N):
        Theta[:,k] = np.sum(X[2:k+3,:], axis=0)

    return Theta

def StroboscopicView(T_eval, n_strobes):
    """ Return n_s regularly spaced indices of T_eval"""
    
    indices_s = np.linspace(start = 0, stop = T_eval.shape[0] - 1, num = n_strobes, dtype = int)
    return indices_s

def Covariance(Theta, Theta_0, bool_fig = False):

    """ Computes the covariance matrix associated with a Kymograph Theta. """

    N = Theta.shape[1]

    ## Center data to the mean
    Delta_Theta = Theta - Theta_0
    # Delta_Kymograph = px.imshow(Delta_Theta, labels=dict(x="Arclength", y="Time", color="Centered Tangent angle"))
    # Delta_Kymograph.show()

    ## Reduce standard deviation to one
    SD = np.reshape(np.std(Theta, axis=0), newshape = (1, N))
    Delta_Theta = Delta_Theta / SD

    ## Make covariance matrix
    C = np.transpose(Delta_Theta) @ Delta_Theta / Theta.shape[0]
    if bool_fig:
        Covariance_figure = px.imshow(C, labels=dict(x="Arclength", y="Arclength", color="Covariance"))
        return C, Covariance_figure
    else:
        return C

def cart_to_pol(coeffs):
    """

    Convert the cartesian conic coefficients, (a, b, c, d, e, f), to the
    ellipse parameters, where F(x, y) = ax^2 + bxy + cy^2 + dx + ey + f = 0.
    The returned parameters are x0, y0, ap, bp, e, phi, where (x0, y0) is the
    ellipse centre; (ap, bp) are the semi-major and semi-minor axes,
    respectively; e is the eccentricity; and phi is the rotation of the semi-
    major axis from the x-axis.

    """

    if coeffs == [None, None, None, None, None, None]:
        return coeffs

    # We use the formulas from ttps://mathworld.wolfram.com/Ellipse.html
    # which assumes a cartesian form ax^2 + 2bxy + cy^2 + 2dx + 2fy + g = 0.
    # Therefore, rename and scale b, d and f appropriately.
    a = coeffs[0]
    b = coeffs[1] / 2
    c = coeffs[2]
    d = coeffs[3] / 2
    f = coeffs[4] / 2
    g = coeffs[5]

    den = b**2 - a*c
    if den > 0:
        raise ValueError('coeffs do not represent an ellipse: b^2 - 4ac must'
                    ' be negative!')

    # The location of the ellipse centre.
    x0, y0 = (c*d - b*f) / den, (a*f - b*d) / den

    num = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)
    fac = np.sqrt((a - c)**2 + 4*b**2)
    # The semi-major and semi-minor axis lengths (these are not sorted).
    ap = np.sqrt(num / den / (fac - a - c))
    bp = np.sqrt(num / den / (-fac - a - c))

    # Sort the semi-major and semi-minor axis lengths but keep track of
    # the original relative magnitudes of width and height.
    width_gt_height = True
    if ap < bp:
        width_gt_height = False
        ap, bp = bp, ap

    # The eccentricity.
    r = (bp/ap)**2
    if r > 1:
        r = 1/r
    e = np.sqrt(1 - r)

    # The angle of anticlockwise rotation of the major-axis from x-axis.
    if b == 0:
        phi = 0 if a < c else np.pi/2
    else:
        phi = np.arctan((2.*b) / (a - c)) / 2
        if a > c:
            phi += np.pi/2
    if not width_gt_height:
        # Ensure that phi is the angle to rotate to the semi-major axis.
        phi += np.pi/2
    phi = phi % np.pi

    return [x0, y0, ap, bp, e, phi]

def PCA(Theta, bool_from_scratch = False):

    """ Computes PCA from a Kymograph Theta.
    Returns Principal components (eigenvectors) with associated eigenvalues. """

    ########################
    ### PCA from scratch ###
    # Warning: it is not computing orthogonal eigenvectors for now
    if bool_from_scratch: 

        # Covariance matrix
        C = Covariance(Theta) # This function standardizes Theta (mean and std)
        # Covariance_figure = px.imshow(C, labels=dict(x="Arclength", y="Arclength", color="Covariance"))
        # Covariance_figure.show()

        # Eigenvalue spectrum (! Not orthogonal !)
        w, v = np.linalg.eig(C)
        idx = w.argsort()[::-1]
        w = w[idx]
        v = v[:,idx]

    ### PCA from scratch ###
    ########################

    else:

    ###########################
    ### PCA Using libraries ###

        M = Theta.shape[0]

        # Standardize data
        sc = StandardScaler()
        sc.fit(Theta)
        Theta_std = sc.transform(Theta) # Mean becomes zero, Standard deviation becomes 1 for each column. Statistics are made for each feature, i.e over time.
        # print("Mean and std after standardization: ", np.mean(Theta_std, axis=0), np.std(Theta_std, axis=0))

        # Perform SVD
        u, s, vh = np.linalg.svd(Theta_std, full_matrices = True)

        # Retrieve PCA from SVD
        v = np.diag((np.diag(s) @ np.diag(s)) / M) # Eigenvalues of covariance matrix
        w = np.transpose(vh) # Matrix whose columns are principal components
        # print("Eigenvalues of covariance matrix: ", Lambda)

    # fig_Eigenspectrum = px.scatter(x=np.arange(w.shape[0]), y=np.real(w), title="Eigenvalues of covariance matrix")

    return w, v

def SpatialFourier(X):

    """ Performs Fourier transform on the spatial axis (axis #0) and returns Fourier transform and corresponding wavenumber modes.
    This assumes that X = X(s, _) where s is the spatial coordinate. """

    Xq = np.fft.rfft(X, axis=0)
    modes = np.fft.rfftfreq(X.shape[0])

    return Xq, modes

def TemporalFourier(X):
    """ Performs Fourier transform on the temporal axis (axis #1) and returns Fourier transform and corresponding frequency modes.
    This assumes that X = X(_, t) where t is the temporal coordinate. """

    Xf = np.fft.rfft(X, axis=1)
    modes = np.fft.rfftfreq(X.shape[1])

    return Xf, modes

def SpatialFourier_vs_Flow(Xq, modes, flow_field, w0, k=1, bool_fig=False):

    """ Computes the phases between k first components of Spatial Fourier and Flow field.
    Possibility to show Fourier k first components against flow field. """

    if k>Xq.shape[0]:
        print("Wrong value for k: too large. k has been redefined as Xq.shape[0] - 1")
        k = Xq.shape[0]

    # Fit ellipse on Fourier_vs_flow curves
    Tf = 100
    t_start = 15*Tf # starts after 15 periods of the flow field
    Fourier_amplitudes = np.zeros((k,1))
    Fourier_phases = np.zeros((k,1))
    polar_coeffs_ellipses = np.zeros((k,1,6))
    fig_ellipse = make_subplots(rows=k, cols=1)
    for q in range(k):
        sign_q = np.sign(np.real(Xq[q,:])) # keep sign information
        mod_q = np.abs(Xq[q,:])
        signed_mod_q = np.multiply(mod_q, sign_q)
        polar_coeffs_ellipses[q,0,:], Fourier_phases[q,0] = LissajousPhase(flow_field, mod_q, t_start, True) # 5 last periods out of 20 are used for limit cycle computations
        # if np.abs(Fourier_phases[q,0])<np.pi/4:
        #     polar_coeffs_ellipses[q,0,:], Fourier_phases[q,0] = LissajousPhase(np.roll(flow_field, int(Tf/2)), np.real(Xq[q,:]), t_start, True)
        #     Fourier_phases[q,0] = Fourier_phases[q,0] + np.pi / 2
        
        # polar_coeffs_ellipses[q,1,:], Fourier_phases[q,1] = LissajousPhase(flow_field, np.imag(Xq[q,:]), t_start, True)
        # if np.abs(Fourier_phases[q,1])<np.pi/4:
        #     polar_coeffs_ellipses[q,1,:], Fourier_phases[q,1] = LissajousPhase(np.roll(flow_field, int(Tf/2)), np.imag(Xq[q,:]), t_start, True)
        #     Fourier_phases[q,1] = Fourier_phases[q,1] + np.pi / 2
        
        # else:
            # polar_coeffs_ellipses[q,1,:], Fourier_phases[q,1] = [None]*6, None
        # print("phase from spatial Fourier mode q = ", q, "is phi = ", Fourier_phase[q])

        # Plot the least squares ellipse
        if Fourier_phases[q,0] != None:
            x, y = make_ellipse_pts(polar_coeffs_ellipses[q,0], npts = 100, tmin = 0, tmax = 2*np.pi)
            fig_ellipse.add_scatter(x=x, y=y, mode = 'lines', name = "ellipse - Least square fit", row=q+1, col=1)
        fig_ellipse.add_scatter(x=flow_field[:t_start], y=mod_q[t_start:], name = "Phase cycle - starting points", row=q+1, col=1)
        fig_ellipse.add_scatter(x=flow_field[t_start:], y=mod_q[t_start:], name = "Phase cycle - simulation points", row=q+1, col=1)

        # if Fourier_phases[q,1] != None:
        #     x, y = make_ellipse_pts(polar_coeffs_ellipses[q,1,:], npts = 100, tmin = 0, tmax = 2*np.pi)
        #     fig_ellipse.add_scatter(x=x, y=y, mode = 'lines', name = "ellipse - Least square fit", row=q+1, col=2)
        # fig_ellipse.add_scatter(x=flow_field[:t_start], y=np.imag(Xq[q, :t_start]), name = "Phase cycle - starting points", row=q+1, col=2)
        # fig_ellipse.add_scatter(x=flow_field[t_start:], y=np.imag(Xq[q, t_start:]), name = "Phase cycle - simulation points", row=q+1, col=2)

        # fig_ellipse.update_layout(width =600, height=525)

    if bool_fig:
        return Fourier_phases, polar_coeffs_ellipses, fig_ellipse
    else:
        return Fourier_phases, polar_coeffs_ellipses

###################################
### Plot animated shape in time ###

def AnimatedShape(X, X_flow, N, w0, Sp4, Beta, tau_b, T_eval):
    """ Animation of a viscoelastic filament"""    
    fig_dict = dict(data = [], layout = {}, frames = [])

    # data
    data_dict = go.Scatter(x=X3N(X[:,0])[:N,0], y=X3N(X[:,0])[N:2*N,0])
    fig_dict["data"].append(data_dict)
    # flow direction
    data_flow_dict = go.Scatter(x = [X3N(X[:,0])[N-1,0], X3N(X[:,0])[N-1,0]], y = [X3N(X[:,0])[2*N-1,0], X3N(X[:,0])[2*N-1,0] + np.sign(X_flow[0])])
    fig_dict["data"].append(data_flow_dict)

    # generic layout
    fig_dict["layout"]["xaxis"] = dict(range = [0, N+1], title = "X")
    fig_dict["layout"]["yaxis"] = dict(range = [-N, N], title = "Y")
    fig_dict["layout"]["hovermode"] = "closest"
    fig_dict["layout"]["title"] = "Filament shape over time for w0 = " + str(w0) + " and Sp^4 = " + str(Sp4) + " and Beta = " + str(Beta) + " and tau_b = " + str(tau_b)
    # Buttons and animation updates
    fig_dict["layout"]["updatemenus"] = [
        {
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 50, "redraw": False},
                                    "fromcurrent": True, "transition": {"duration": 0}}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 50},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }
    ]

    # Sliders
    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Time:",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 0, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": [] # steps should reflect each frame, so cfilled within time
    }

    for k in range(X.shape[1]):
        # frames
        # frame_dict = dict(data = go.Scatter(x = X3N(X[:,k])[:N, 0], y = X3N(X[:,k])[N:2*N, 0]), name = str(k))
        # fig_dict["frames"].append(frame_dict)
        frame_dict = go.Scatter(x = X3N(X[:,k])[:N, 0], y = X3N(X[:,k])[N:2*N, 0])
        frame_flow_dict = go.Scatter(x = [X3N(X[:,k])[N-1,0], X3N(X[:,k])[N-1,0]], y = [X3N(X[:,k])[2*N-1,0], X3N(X[:,k])[2*N-1,0] + np.sign(X_flow[k])])
        fig_dict["frames"].append(dict(data = [frame_dict, frame_flow_dict], name = str(k)))
        # frame-specific layout: slider steps
        slider_step = dict(
        args = [[str(k)], dict(frame = dict(duration = 50, redraw = False), mode = "immediate", transition= dict(duration = 0))],
        label = np.round(T_eval[k], 0),
        method = "animate")
        sliders_dict["steps"].append(slider_step)

    # Sliders
    fig_dict["layout"]["sliders"] = [sliders_dict]
    # Construct final animated figure
    fig_shapes = go.Figure(fig_dict)
    # fig_shapes.show()

    fig_shapes.update_yaxes(scaleanchor="x", scaleratio=1)

    return fig_shapes

def AnimatedTwoShapes(X, Y, X_flow, N, w0, Sp4, Beta, tau_b, T_eval):
    """ Plot two overlapping shapes. To be generalized to N_s shapes. """
    fig_dict = dict(data = [], layout = {}, frames = [])

    # data
    data_dict = go.Scatter(x=X3N(X[:,0])[:N,0], y=X3N(X[:,0])[N:2*N,0])
    data_dict_2 = go.Scatter(x=X3N(Y[:,0])[:N,0], y=X3N(Y[:,0])[N:2*N,0])
    fig_dict["data"].append(data_dict)
    fig_dict["data"].append(data_dict_2)

    # flow direction
    data_flow_dict = go.Scatter(x = [X3N(X[:,0])[N-1,0], X3N(X[:,0])[N-1,0]], y = [X3N(X[:,0])[2*N-1,0], X3N(X[:,0])[2*N-1,0] + np.sign(X_flow[0])])
    fig_dict["data"].append(data_flow_dict)

    # generic layout
    fig_dict["layout"]["xaxis"] = dict(range = [0, N+1], title = "X")
    fig_dict["layout"]["yaxis"] = dict(range = [-N, N], title = "Y")
    fig_dict["layout"]["hovermode"] = "closest"
    fig_dict["layout"]["title"] = "Filament shape over time for w0 = " + str(w0) + " and Sp^4 = " + str(Sp4) + " and Beta = " + str(Beta) + " and tau_b = " + str(tau_b)
    # Buttons and animation updates
    fig_dict["layout"]["updatemenus"] = [
        {
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 50, "redraw": False},
                                    "fromcurrent": True, "transition": {"duration": 0}}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 50},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }
    ]

    # Sliders
    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Time:",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 0, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": [] # steps should reflect each frame, so cfilled within time
    }

    for k in range(X.shape[1]):
        # frames
        # frame_dict = dict(data = go.Scatter(x = X3N(X[:,k])[:N, 0], y = X3N(X[:,k])[N:2*N, 0]), name = str(k))
        # fig_dict["frames"].append(frame_dict)
        frame_dict = go.Scatter(x = X3N(X[:,k])[:N, 0], y = X3N(X[:,k])[N:2*N, 0])
        frame_dict_2 = go.Scatter(x = X3N(Y[:,k])[:N, 0], y = X3N(Y[:,k])[N:2*N, 0])
        frame_flow_dict = go.Scatter(x = [X3N(X[:,k])[N-1,0], X3N(X[:,k])[N-1,0]], y = [X3N(X[:,k])[2*N-1,0], X3N(X[:,k])[2*N-1,0] + np.sign(X_flow[k])])
        fig_dict["frames"].append(dict(data = [frame_dict, frame_dict_2, frame_flow_dict], name = str(k)))
        # frame-specific layout: slider steps
        slider_step = dict(
        args = [[str(k)], dict(frame = dict(duration = 50, redraw = False), mode = "immediate", transition= dict(duration = 0))],
        label = np.round(T_eval[k], 0),
        method = "animate")
        sliders_dict["steps"].append(slider_step)

    # Sliders
    fig_dict["layout"]["sliders"] = [sliders_dict]
    # Construct final animated figure
    fig_shapes = go.Figure(fig_dict)
    # fig_shapes.show()

    return fig_shapes


def AnimatedShapes(X_list, X_flow, N, T_eval):
    """ Plot N_s overlapping shapes stored in X_ilist.

    --> To be updated so that the flow is plotted in a circle with radius one.
    --> Only show first filament at the beginning, and display legend
    """
    n_decimals=1 # Number of decimals for the time slider

    fig_dict = dict(data = [], layout = {}, frames = [])

    # data: initial frame
    for p in range(len(X_list)):
        X = X_list[p]
        # Only add the first element to the layout initially.
        data_dict = go.Scatter(x=X3N(X[:,0])[:N,0], y=X3N(X[:,0])[N:2*N,0])
        fig_dict["data"].append(data_dict)

    # flow direction
    # data_flow_dict = go.Scatter(x = [X3N(X[:,0])[N-1,0], X3N(X[:,0])[N-1,0]], y = [X3N(X[:,0])[2*N-1,0], X3N(X[:,0])[2*N-1,0] + np.sign(X_flow[0])])
    # fig_dict["data"].append(data_flow_dict)

    # generic layout
    fig_dict["layout"]["xaxis"] = dict(range = [0, N+1], title = "X")
    fig_dict["layout"]["yaxis"] = dict(range = [-N, N], title = "Y")
    fig_dict["layout"]["hovermode"] = "closest"
    # Buttons and animation updates
    fig_dict["layout"]["updatemenus"] = [
        {
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 50, "redraw": False},
                                    "fromcurrent": True, "transition": {"duration": 0}}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 50},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }
    ]

    # Sliders
    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Time:",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 0, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": [] # steps should reflect each frame, so cfilled within time
    }

    # data: following frames
    for k in range(len(T_eval)):
        frame_dict_list = []
        for p in range(len(X_list)):
            X = X_list[p]
            frame_dict = go.Scatter(x = X3N(X[:,k])[:N, 0], y = X3N(X[:,k])[N:2*N, 0])
            frame_dict_list.append(frame_dict)
        
        # frame_flow_dict = go.Scatter(x = [X3N(X[:,k])[N-1,0], X3N(X[:,k])[N-1,0]], y = [X3N(X[:,k])[2*N-1,0], X3N(X[:,k])[2*N-1,0] + np.sign(X_flow[k])])
        # frame_dict_list.append(frame_flow_dict)

        fig_dict["frames"].append(dict(data = frame_dict_list, name = str(k)))
        
        # frame-specific layout: slider steps
        slider_step = dict(
        args = [[str(k)], dict(frame = dict(duration = 50, redraw = False), mode = "immediate", transition= dict(duration = 0))],
        label = np.round(T_eval[k], n_decimals),
        method = "animate")
        sliders_dict["steps"].append(slider_step)

    # Sliders
    fig_dict["layout"]["sliders"] = [sliders_dict]
    # Construct final animated figure
    fig_shapes = go.Figure(fig_dict)
    fig_shapes.update_layout(showlegend = True)
    # fig_shapes.show()

    return fig_shapes



### ----- Plotting animated shape in time ----- ###
######################################


if __name__ == '__main__':

    # Fetch files satisfying required conditions

    sim_path = (Path(__file__).resolve().parent.parent / 'Model' / 'Output' / 'Inference_Examples' / 'VarySp4Npoints' / 'npoints_10000').resolve()

    def metadata_condition_0(solver_dict, eps = 1e-6):
        """ This functions computes the boolean corresponding to the condition of a clamped purely bending filament, in a harmonic vertical flow. """

        output_folder, N, taus_b, tau_s, init_conf, bool_EI, Beta, gamma, n_L, m_L, A, w0, Sp4, k0, Lambdas, Zetas, X_flow_field_string, T_span, T_eval, T_sim_max, T_sim, X_flow_field, X_0, method = list(solver_dict.values())

        bool_condition = (N == 10) & (np.abs(taus_b[0] - 0) < eps) & (np.abs(Beta - 0) < eps) & ("StraightLine" in init_conf) & (gamma == 2) & (np.abs(A - 1e-7) < eps) & (np.abs(w0 - 1e-6) < eps) & (np.abs(k0 - 1e13) < eps) & (method == 'BDF')

        return bool_condition

    ids_list = fetch_files(sim_path, metadata_condition_0, None)
    ids_list.sort()
    print("# files: ", len(ids_list))
    print(ids_list[0])
    
    X_list = []
    for base_id in ids_list:
        data_file = str(sim_path / ('data_' + base_id + '.csv'))
        X = get_data(data_file)
        X_list.append(X)

    metadata_file = str(sim_path / ('metadata_' + base_id + '.json'))
    solver_dict = get_metadata(metadata_file)
    output_folder, N, taus_b, tau_s, init_conf, bool_EI, Beta, gamma, n_L, m_L, A, w0, Sp4, k0, Lambdas, Zetas, X_flow_field_string, T_span, T_eval, T_sim_max, T_sim, X_flow_field, X_0, method = list(solver_dict.values())  
    T_eval *= w0 / (2*np.pi)

    # Visualize filaments
    Sp4 = 1
    X_3_Nm1_eq = CheckEquilibrium(N-1, A, gamma, Sp4, n_L = n_L, Lambdas=Lambdas, conditions = "vertical_flow_uniform", n_eq = N-1) # Go to N-1 only to take into account clamping at the base later
    X_3N_eq = np.zeros((3*N, 1))
    X_3N_eq[0] = 0 # Clamped x
    X_3N_eq[1:N] = X_3_Nm1_eq[:N-1] + 1
    X_3N_eq[N] = 0 # Clamped y
    X_3N_eq[N+1:2*N] = X_3_Nm1_eq[N-1:2*(N-1)] + 0
    X_3N_eq[2*N] = 0 # Clamped theta
    X_3N_eq[2*N+1:] = X_3_Nm1_eq[2*(N-1):]

    X_Np2_eq = XNp2(X_3N_eq)
    X_3N_eq_from_Np2 = X3N(X_Np2_eq)

    # Check that equilibrium is right
    fig = go.Figure()
    fig.add_scatter(x = X_3N_eq[:N, 0], y = X_3N_eq[N:2*N, 0], name = "Eq using x,y")
    fig.add_scatter(x = X_3N_eq_from_Np2[:N, 0], y = X_3N_eq_from_Np2[N:2*N, 0], name = "Eq using theta")
    fig.vs_show()
    
    X_eq = X_Np2_eq.repeat(T_eval.shape[0], axis = 1)
    X_list.append(X_eq)

    ## 3D animation
    fig_anim = AnimatedShapes(X_list = X_list, X_flow = X_flow_field, N = N, T_eval = T_eval)
    # fig_anim.add_scatter(x=X_3N_eq[:N], y=X_3N_eq[N:2*N], marker_color = "red", mode = "lines", name = "Analytical solution")
    fig_anim.vs_show()

    ## 2D heatmap plot
    # X_list = [X_list[0]] # To be removed
    # for X in X_list:
    #     data = go.Heatmap
    #     fig = go.Figure(data=go.Heatmap(
    #                 x = T_eval,
    #                 y = np.linspace(0,1,N),
    #                 z = X[2:, :],
    #                 ))
    #     fig.vs_show()

    # # Compute observable on these files and put this new data into a dataframe
    # columns = ['Sp4', 'taus_b']

    # def simulation_time(solver_dict, X = None):
    #     output_folder, N, taus_b, init_conf, Beta, gamma, n_L, m_L, A, w0, Sp4, Lambdas, Zetas, X_flow_field_string, T_span, T_eval, T_sim_max, T_sim, X_flow_field, X_0, method = list(solver_dict.values())
    #     return T_sim    
    
    # df = observable_dataframe(sim_directory, ids_list, columns, simulation_time, obs_type = 'metadata')
    # df.columns = [*df.columns[:-1], 'T_sim']
    # df['tau_b'] = df.apply(lambda x: x['taus_b'][0], axis = 1)

    # # Plot T_sim against Sp4
    # plot_2D(df, 'tau_b', 'T_sim').show()
    # plot_heatmap(df, 'Sp4', 'tau_b', 'T_sim').show()