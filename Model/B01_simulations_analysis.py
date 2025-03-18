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

from A01_Coarse_grained_axoneme_functions import * 
from audioop import mul
import multiprocessing

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

def animated_shapes(X, N, T_eval, Sp4):

    """ Makes videos of shapes (X(t), Y(t)) with time cursor and flow orientation marker. 
    Time is counted in Sp4 units, which always exist - as long as there is bending elasticity. 
    Returns the figure ready to be plotted. """

    fig_dict = dict(data = [], layout = {}, frames = [])

    # data
    data_dict = go.Scatter(x=X3N(X[:,0])[:N,0], y=X3N(X[:,0])[N:2*N,0])
    fig_dict["data"].append(data_dict)

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

    for k in range(X.shape[1]):
        # frames
        frame_dict = dict(data = go.Scatter(x = X3N(X[:,k])[:N, 0], y = X3N(X[:,k])[N:2*N, 0]), name = str(k))
        fig_dict["frames"].append(frame_dict)

        # frame-specific layout: slider steps
        slider_step = dict(
        args = [[str(k)], dict(frame = dict(duration = 50, redraw = False), mode = "immediate", transition= dict(duration = 0))],
        label = str(np.round(T_eval[k]/Sp4, 0)) + "in Sp4 units",
        method = "animate")
        sliders_dict["steps"].append(slider_step)

    # Sliders
    fig_dict["layout"]["sliders"] = [sliders_dict]
    # Construct final animated figure
    fig_shapes = go.Figure(fig_dict)
    return fig_shapes

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
    
    Remark: total filament length = N but only N-1 segments are represented (N first points). 
    Returns X_eq with 3 * n_eq points:
        - n_eq points for x and n_eq points for y and n_eq points for theta, such that s in [0, N-1 = L-1]
    """

    ## Analytical equilibrium solution
    X_eq = np.zeros((2*n_eq))
    X_eq[:n_eq] = np.linspace(start = 0, stop = N, num = n_eq)

    # For a vertical point force at distal end
    if conditions == "vertical_point_tip":
        F = n_L[1]
        X_eq[n_eq:] = (3 * (X_eq[:n_eq]/(N))**2 - (X_eq[:n_eq]/(N))**3 ) * F * ((N)**3) / 6

    # For a density force at distal segment
    elif conditions == "vertical_density_tip":
        F = Lambdas[-1][1]
        X_eq[n_eq:] = (3 * (X_eq[:n_eq]/N)**2 - (X_eq[:n_eq]/N)**3) * F * (N**3) / 6

    # For a uniform vertical force
    elif conditions == "vertical_density_uniform":
        f = Lambdas[0][1]
        X_eq[n_eq:] = ( (X_eq[:n_eq]/N)**4 - 4*(X_eq[:n_eq]/N)**3 + 6*(X_eq[:n_eq]/N)**2 ) * f * (N**4) / 24

    # For a uniform small vertical flow
    elif conditions == "vertical_flow_uniform":
        X_eq[n_eq:] = ( (X_eq[:n_eq]/N)**4 - 4*(X_eq[:n_eq]/N)**3 + 6*(X_eq[:n_eq]/N)**2 ) * A * gamma * Sp4 *(N**4) / 24
        
    else:
        print("No condition for exact solution has been specified.")
        return NameError
    
    X_3N_eq = np.vstack((X_eq.reshape((-1,1)), np.zeros((n_eq)).reshape(-1,1)))
    return X_3N_eq

def Kymograph(X):

    """ Computes Kymograph of X and mean angle."""

    N = X.shape[0]-2
    M = X.shape[1] # Number of time samples 
    
    Theta = np.zeros((M,N))
    for k in range(N):
        Theta[:,k] = np.sum(X[2:k+3,:], axis=0)

    return Theta

def StroboscopicView(T_eval, t_s):
    """ Return values of X at t = k * T_s.
    Remark: X is of the form (s,t) """
    
    l = int(T_eval[-1] // t_s)
    indices_s = np.zeros((l+1,), dtype = int)
    
    for k in range(l+1):
        k_s = np.argmin(np.abs(T_eval - k * t_s))
        indices_s[k] = k_s
        
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

def PCA(Theta, bool_from_scratch = False, bool_fig=False):

    """ Computes PCA from a Kymograph Theta.
    Returns Principal components (eigenvectors) with associated eigenvalues.
    Possibility to return figure of the eigenspectrum """

    ########################
    ### PCA from scratch ###
    # Warning: it is not computing orthogonal eigenvectors for now
    if bool_from_scratch: 

        # Covariance matrix
        C = Covariance(Theta) # This function standardizes Theta (mean and std)
        print("Covariance shape: ", C.shape)
        # Covariance_figure = px.imshow(C, labels=dict(x="Arclength", y="Arclength", color="Covariance"))
        # Covariance_figure.show()

        # Eigenvalue spectrum (! Not orthogonal !)
        w, v = np.linalg.eig(C)
        idx = w.argsort()[::-1]
        w = w[idx]
        v = v[:,idx]
        if bool_fig:
            fig_Eigenspectrum = px.scatter(x=np.arange(w.shape[0]), y=np.real(w), title="Eigenvalues of covariance matrix")
            return w, v, fig_Eigenspectrum
        else:
            return w, v

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
        Lambda = np.diag((np.diag(s) @ np.diag(s)) / M) # Eigenvalues of covariance matrix
        P = np.transpose(vh) # Matrix whose columns are principal components
        # print("Eigenvalues of covariance matrix: ", Lambda)
        if bool_fig:
            fig_Eigenspectrum = px.scatter(x=range(np.shape(Lambda)[0]), y=Lambda, title = "Eigenspectrum of covariance matrix from SVD")
            return P, Lambda, fig_Eigenspectrum
        else:
            return P, Lambda

def PCA_vs_Flow(Theta, Theta_0, P, flow_field, k=1, bool_fig=False):

    """ Computes the phase between first k components of PCA and Flow field.
    Possibility to show PCA first k components against flow field. """

    # Extract shape scores
    shape_scores = (Theta-Theta_0) @ P

    # shape_scores_figure = px.scatter(x=B_0, y=B_1, title="Shape scores 1 over 0")
    # shape_scores_figure.show()

    # Shape score 0 versus Flow field - normalized to angle dimension
    # print("Type of flow field: ", X_flow_field_type)
    # print("Parameters of flow field: ", X_flow_field_params)
    # w0 = 0
    # normalized_flow_field = np.array([A * xi * Delta_s / K_b ] * len(T_eval)) # Assuming vertical flow

    # Fit ellipse on PCA curve
    
    t_start = 750
    polar_coeffs_ellipses = []
    PCA_phases = []
    figs_ellipse = []
    for l in range(k):
        B_l = shape_scores[:,l]
        polar_coeffs_ellipse_l, PCA_phase_l = LissajousPhase(flow_field, B_l, t_start, True) # 5 last periods are used for limit cycle computations
        polar_coeffs_ellipses.append(polar_coeffs_ellipse_l)
        PCA_phases.append(PCA_phase_l)

        # Plot the least squares ellipse
        x, y = make_ellipse_pts(polar_coeffs_ellipse_l, npts = 100, tmin = 0, tmax = 2*np.pi)
        fig_ellipse_l = go.Figure()
        fig_ellipse_l.add_scatter(x=x, y=y, mode = 'lines', name = "ellipse - Least square fit")
        fig_ellipse_l.add_scatter(x=flow_field[:t_start], y=B_l[:t_start], name = "Phase cycle - starting points")
        fig_ellipse_l.add_scatter(x=flow_field[t_start:], y=B_l[t_start:], name = "Phase cycle - simulation points")
        # fig_ellipse.update_layout(width =600, height=525)
        figs_ellipse.append(fig_ellipse_l)

    if bool_fig:
        return PCA_phases, polar_coeffs_ellipses, figs_ellipse
    else:
        return PCA_phases, polar_coeffs_ellipses

def SpatialFourier(X, T_eval, w0, bool_fig=False):

    """ Performs Fourier transform on the spatial axis (axis #0) and returns Fourier transform and corresponding spatial modes.
    Possibility to return figure of spatial spectrum."""

    Xq = np.fft.rfft(X, axis=0)
    # Xs = np.fft.irfft(Xq, axis=0)
    # print("Xq.shape = ", Xq.shape)
    modes = np.fft.rfftfreq(X.shape[0])
    # wavelength = [np.inf] + N+2 / modes[1:]
    fig_modes_temporal = make_subplots(
        rows=1, cols=2, 
        subplot_titles=("Amplitude of fourier modes", "Phase of Fourier modes")
        )
    for q in range(Xq.shape[0]):
        sign_q = np.sign(np.real(Xq[q,:])) # keep sign information
        mod_q = np.abs(Xq[q,:])
        signed_mod_q = np.multiply(np.abs(Xq[q,:]), sign_q)
        angle_q = np.angle(np.multiply(np.exp(np.angle(Xq[q,:])*1j), sign_q))
        test_trace = go.Scatter(x=T_eval*w0/(2*np.pi), y=mod_q, name="signed Modulus(Mode q = " + str(q) + ")")
        fig_modes_temporal.add_trace(test_trace, row=1, col=1)
        test_trace = go.Scatter(x=T_eval*w0/(2*np.pi), y=angle_q, name="Phase(Mode q = " + str(q) + ") between -pi/2 and pi/2")
        fig_modes_temporal.add_trace(test_trace, row=1, col=2)

    fig_modes_temporal.update_layout(title_text="Fourier spatial modes against time - normalized by flow period.")
    if bool_fig:
        return Xq, modes, fig_modes_temporal
    else:
        return Xq, modes

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
    """ What is the difference with animated_shapes ?"""    
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

def AnimatedShapes(X, Y, X_flow, N, w0, Sp4, Beta, tau_b, T_eval):
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

### ----- Plotting animated shape in time ----- ###
######################################


if __name__ == '__main__':

    # Fetch files satisfying required conditions

    sim_directory = "C:/Users/Luc/Documents/PhD_Large_files/RheoFlag/Model/Output/"
    sim_directory += "AnalyticalComparisons/PureBending_Clamped_Relaxation/"
    # "StraightLine_PeriodicFlow"

    def metadata_condition_0(solver_dict, eps = 1e-6):
        output_folder, N, taus_b, init_conf, Beta, gamma, n_L, m_L, A, w0, Sp4, k0, Lambdas, Zetas, X_flow_field_string, T_span, T_eval, T_sim_max, T_sim, X_flow_field, X_0, method = list(solver_dict.values())

        bool_condition = (N == 10) & (np.abs(taus_b[0] - 0) < eps) & (np.abs(Beta - 0) < eps) & ("ProximalBend" in init_conf) & (gamma == 2) & ((np.abs(A - 0) < eps) & (np.abs(w0 - 0) < eps)) & (np.abs(Sp4 - 1e0) < eps) & (np.abs(k0 - 1e3) < eps) & (method == 'Radau')

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