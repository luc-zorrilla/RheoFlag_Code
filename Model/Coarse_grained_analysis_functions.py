""" This file compiles all functions used for analysis of an individual 
simulation of a viscoelastic filament, which takes the form of a file. """

from audioop import mul
import multiprocessing

from regex import R
from Coarse_grained_axoneme_functions import *
import math
import numpy as np
import scipy.signal

from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from datetime import datetime

########################################
### ----- Extracting Functions ----- ###

def ExtractParametersData(filename):

    file = open(filename, "r")

    ## Metadata extraction
    file.readline()
    N = int(file.readline()[4:])
    taus_b = file.readline()[9:]
    taus_b = list(map(float, taus_b.strip('][\n').split(', ')))
    init_conf = file.readline()[12:]
    Beta = float(file.readline()[7:])
    gamma = float(file.readline()[8:])

    n_L = file.readline()[6:] ##
    n_L = list(map(float,n_L.strip('][\n').split(', ')))

    m_L = float(file.readline()[6:]) ##

    A = float(file.readline()[4:])

    w0 = float(file.readline()[5:])

    Sp4 = float(file.readline()[6:])

    Lambdas = file.readline()[10:] ##
    Lambdas = list(map(str, Lambdas.strip('][\n').split(', ')))
    for k in range(len(Lambdas)):
        Lambdas[k] = list(map(float, Lambdas[k].strip('][').split('; ')))

    Zetas = file.readline()[8:] ##
    Zetas = list(map(float, Zetas.strip('][\n').split(', ')))

    X_flow_field = file.readline()[15:]
    if X_flow_field[:9] == "SINE FLOW":
        X_flow_field_type = "SINE FLOW: (psi, A, w0)"
        X_flow_field_params = list(map(float, X_flow_field[26:].strip(")(\n").split(", ")))
    elif X_flow_field[:13] == "CONSTANT FLOW":
        X_flow_field_type = "CONSTANT FLOW: (psi, A)"
        X_flow_field_params = list(map(float, X_flow_field[26:].strip(")(\n").split(", ")))
    elif X_flow_field[:12] == "PIV-IMPORTED":
        X_flow_field_type = "PIV-IMPORTED: filename"
        X_flow_field_params = X_flow_field[18:] # Filename of the PIV import
    else: # X_flow_field == "NO FLOW":
        X_flow_field_type = "NO FLOW"
        X_flow_field_params = 0
        
    T_span = file.readline()[9:] ##
    T_span = list(map(float, T_span.strip('][\n').split(', ')))
    T_eval = file.readline()[9:] ##
    T_eval = list(map(float, T_eval.strip('][\n').split(', ')))

    method = file.readline()

    ## Create data arrays in functions of the metadata
    X = np.zeros((N+2, len(T_eval)))
    file.readline()
    file.readline()

    first_line = file.readline()
    if first_line[:10] == "ValueError":
        for t in range(1, len(T_eval)):
            file.readline()
    else:
        X[:,0] = list(map(float, first_line.strip('][\n').lstrip().rstrip().replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').split(' ')))
        for t in range(1, len(T_eval)):
            X[:,t] = list(map(float, file.readline().strip('][\n').lstrip().rstrip().replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').split(' ')))

    file.close()
    parameters = [N, taus_b, init_conf, Beta, gamma, n_L, m_L, A, w0, Sp4, Lambdas, Zetas, X_flow_field_type, X_flow_field_params, T_span, T_eval, method]
    return parameters, X

def ExtractParameters(filename):

    file = open(filename, "r")

    ## Metadata extraction
    file.readline()
    N = int(file.readline()[4:])
    taus_b = file.readline()[9:]
    taus_b = list(map(float, taus_b.strip('][\n').split(', ')))
    init_conf = file.readline()[12:]
    Beta = float(file.readline()[7:])
    gamma = float(file.readline()[8:])

    n_L = file.readline()[6:] ##
    n_L = list(map(float,n_L.strip('][\n').split(', ')))

    m_L = float(file.readline()[6:]) ##

    A = float(file.readline()[4:])

    w0 = float(file.readline()[5:])

    Sp4 = float(file.readline()[6:])

    Lambdas = file.readline()[10:] ##
    Lambdas = list(map(str, Lambdas.strip('][\n').split(', ')))
    for k in range(len(Lambdas)):
        Lambdas[k] = list(map(float, Lambdas[k].strip('][').split('; ')))

    Zetas = file.readline()[8:] ##
    Zetas = list(map(float, Zetas.strip('][\n').split(', ')))

    X_flow_field = file.readline()[15:]
    if X_flow_field[:9] == "SINE FLOW":
        X_flow_field_type = "SINE FLOW: (psi, A, w0)"
        X_flow_field_params = list(map(float, X_flow_field[26:].strip(")(\n").split(", ")))
    elif X_flow_field[:13] == "CONSTANT FLOW":
        X_flow_field_type = "CONSTANT FLOW: (psi, A)"
        X_flow_field_params = list(map(float, X_flow_field[26:].strip(")(\n").split(", ")))
    elif X_flow_field[:12] == "PIV-IMPORTED":
        X_flow_field_type = "PIV-IMPORTED: filename"
        X_flow_field_params = X_flow_field[18:] # Filename of the PIV import
    else: # X_flow_field == "NO FLOW":
        X_flow_field_type = "NO FLOW"
        X_flow_field_params = 0
        
    T_span = file.readline()[9:] ##
    T_span = list(map(float, T_span.strip('][\n').split(', ')))
    T_eval = file.readline()[9:] ##
    T_eval = list(map(float, T_eval.strip('][\n').split(', '))) # T_eval = list(map(float, T_eval.strip('][\n').lstrip().rstrip().replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').split(' ')))

    method = file.readline()

    file.close()
    parameters = [N, taus_b, init_conf, Beta, gamma, n_L, m_L, A, w0, Sp4, Lambdas, Zetas, X_flow_field_type, X_flow_field_params, T_span, T_eval, method]
    return parameters

def ExtractData(filename):

    file = open(filename, "r")

    ## Metadata extraction
    file.readline()

    N = int(file.readline()[4:])

    file.readline()
    file.readline()
    file.readline()
    file.readline()
    file.readline()
    file.readline()
    file.readline()
    file.readline()
    file.readline()
    file.readline()
    file.readline()
    file.readline()
    file.readline()

    T_eval = file.readline()[9:] ##
    T_eval = list(map(float, T_eval.strip('][\n').split(', ')))#list(map(float, T_eval.strip('][\n').lstrip().rstrip().replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').split(' ')))

    file.readline()
    
    ## Create data arrays in functions of the metadata
    X = np.zeros((N+2, len(T_eval)))
    file.readline()
    file.readline()


    first_line = file.readline()
    if first_line[:10] == "ValueError":
        for t in range(1, len(T_eval)):
            file.readline()
    else:
        X[:,0] = list(map(float, first_line.strip('][\n').lstrip().rstrip().replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').split(' ')))
        for t in range(1, len(T_eval)):
            X[:,t] = list(map(float, file.readline().strip('][\n').lstrip().rstrip().replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').split(' ')))

    file.close()
    return X

### ----- Extracting Functions ----- ###
########################################

######################################
### ----- Plotting functions ----- ###

# HEX colors for plotting
red = "#d62728"
black = "#000000"
curry = "#bcbd22"
orange = "#ff7f0e"
blue = '#17becf'
purple = '#9467bd'

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

    fig_X_dot = go.Figure()
    K_trace = go.Scatter(x = T_eval, y = np.log(K), line=dict(color=orange), name ="Total kinetic energy")
    fig_X_dot.add_trace(K_trace)


    fig_X_dot.update_xaxes(
        showgrid=False,
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor="black"
    )

    fig_X_dot.update_yaxes(
        # scaleanchor = "x",
        # scaleratio = 1,
        showgrid=False,
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor="black"
    )

    fig_X_dot.update_layout(
        title = "Kinetic energy over time",
        xaxis_title = "Time",
        yaxis_title = "Kinetic energy"
    )
    return fig_X_dot

def CheckEquilibrium(X, N, T_eval, A, gamma, Sp4, n_L = [0,0], Lambdas=[[0,0]], conditions = "None"):

    """ Returns a figure with equilibrium position of a shape dynamics and computes analytical solution for small deflection: 
    - vertical point force at tip - "vertical_point_tip"
    - vertical density force on tip - "vertical_density_tip"
    - vertical uniform density force along the beam - "vertical_density_uniform"
    - vertical uniform flow - "vertical_flow_uniform"
    All analytical solutions are computed in the case of an initial horizontal beam, assuming equilibrium and small deflection."""

    ## Analytical equilibrium solution
    X_eq = np.zeros((2*N))

    # For a vertical point force at distal end
    if conditions == "vertical_point_tip":
        F = n_L[1]
        for k in range(N):
            X_eq[k] = k
            X_eq[N+k] = 3 * (X_eq[k]/(N))**2 - (X_eq[k]/(N))**3
            X_eq[N+k] = X_eq[N+k] * F * ((N)**3) / 6

    # For a density force at distal segment
    elif conditions == "vertical_density_tip":
        F = Lambdas[-1][1]
        for k in range(N):
            X_eq[k] = k
            X_eq[N+k] = 3 * (X_eq[k]/N)**2 - (X_eq[k]/N)**3
            X_eq[N+k] = X_eq[N+k] * F * (N**3) / 6

    # For a uniform vertical force
    elif conditions == "vertical_density_uniform":
        f = Lambdas[0][1]
        for k in range(N):
            X_eq[k] = k
            X_eq[N+k] = (X_eq[k]/N)**4 - 4*(X_eq[k]/N)**3 + 6*(X_eq[k]/N)**2
            X_eq[N+k] = X_eq[N+k] * f * (N**4) / 24

    # For a uniform small vertical flow
    elif conditions == "vertical_flow_uniform":
        for k in range(N):
            X_eq[k] = k
            X_eq[N+k] = (X_eq[k]/N)**4 - 4*(X_eq[k]/N)**3 + 6*(X_eq[k]/N)**2
            X_eq[N+k] = X_eq[N+k] * A * gamma * Sp4 *(N**4) / 24
    else:
        print("No condition for exact solution has been specified.")
        return NameError

    fig_one_set = go.Figure()
    new_trace_eq = go.Scatter(x = X_eq[:N], y = X_eq[N:], name = "Analytical solution", line = dict(color=red))
    fig_one_set.add_trace(new_trace_eq)
    for t in range(0, X.shape[1], 100):
        X_3N_t = X3N(X[:,t])
        if t==0 or t==X.shape[1]-1:
            new_trace_t = go.Scatter(x=X_3N_t[:N,0], y=X_3N_t[N:2*N,0], name="t = " + str(T_eval[t]), line=dict(color=black))
            # force_trace_t = go.Scatter(x=X_3N_t[N//2:N//2+2,0], y=X_3N_t[N + N//2:N + N//2+2,0], name="force application at t = " + str(T_eval[t]), line=dict(color=red))
            fig_one_set.add_trace(new_trace_t)
            # fig_one_set.add_trace(force_trace_t)
        else:
            new_trace_t = go.Scatter(x=X_3N_t[:N,0], y=X_3N_t[N:2*N,0], name="t = " + str(T_eval[t]), visible="legendonly", line=dict(color=orange))
            # force_trace_t = go.Scatter(x=X_3N_t[N//2:N//2+2,0], y=X_3N_t[N + N//2:N + N//2+2,0], name="force application at t = " + str(T_eval[t]), visible="legendonly", line=dict(color=red))
            fig_one_set.add_trace(new_trace_t)
            # fig_one_set.add_trace(force_trace_t)

    fig_one_set.update_layout(
        legend_title_text='Beam shapes',
        title = "Beam shapes at different times",
        yaxis_title = "Y(X)",
        )

    fig_one_set.update_xaxes(
        showgrid=False,
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor="black",
        titlefont_size = 8
    )

    fig_one_set.update_yaxes(
        scaleanchor = "x",
        scaleratio = 1,
        showgrid=False,
        zeroline=True,
        zerolinewidth=2,
        zerolinecolor="black"
    )

    return fig_one_set

def Kymograph(X, bool_fig=False):

    """ Computes Kymograph of X and mean angle. 
    Returns the associated figure if desired. """

    N = X.shape[0]-2
    M = X.shape[1] # Number of time samples 
    # Theta = np.reshape(np.array([X3N(X[t,:])[2*N:] for t in range(M)]), newshape = (M, N))
    Theta = np.zeros((M,N))
    for k in range(N):
        Theta[:,k] = np.sum(X[2:k+3,:], axis=0)

    # Mean angle
    # Careful with 2*pi limit!
    Theta_0 = np.repeat(np.reshape(np.mean(Theta, axis = 0), newshape=(1,N)), M, axis=0)

    if bool_fig:
        fig_Kymograph = px.imshow(Theta, labels=dict(x="Arclength from second segment", y="Time", color="Tangent angle"), aspect='auto')
        return Theta, Theta_0, fig_Kymograph
    else:
        return Theta, Theta_0

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

def fit_ellipse(x, y):
    """

    Fit the coefficients a,b,c,d,e,f, representing an ellipse described by
    the formula F(x,y) = ax^2 + bxy + cy^2 + dx + ey + f = 0 to the provided
    arrays of data points x=[x1, x2, ..., xn] and y=[y1, y2, ..., yn].

    Based on the algorithm of Halir and Flusser, "Numerically stable direct
    least squares fitting of ellipses'.


    """

    D1 = np.vstack([x**2, x*y, y**2]).T
    D2 = np.vstack([x, y, np.ones(len(x))]).T
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    try:
        T = -np.linalg.inv(S3) @ S2.T
        M = S1 + S2 @ T
        C = np.array(((0, 0, 2), (0, -1, 0), (2, 0, 0)), dtype=float)
        M = np.linalg.inv(C) @ M
        eigval, eigvec = np.linalg.eig(M)
        con = 4 * eigvec[0]* eigvec[2] - eigvec[1]**2
        ak = eigvec[:, np.nonzero(con > 0)[0]]
        return list(np.concatenate((ak, T @ ak)).ravel())
    except:
        # print("Cannot fit this ellipse. Most probably signal is zero.")
        return [None, None, None, None, None, None]
    
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

def make_ellipse_pts(params, npts=100, tmin=0, tmax=2*np.pi):
    """
    Return npts points on the ellipse described by the params = x0, y0, ap,
    bp, e, phi for values of the parametric variable t between tmin and tmax.

    """

    x0, y0, ap, bp, e, phi = params
    # A grid of the parametric variable, t.
    t = np.linspace(tmin, tmax, npts)
    x = x0 + ap * np.cos(t) * np.cos(phi) - bp * np.sin(t) * np.sin(phi)
    y = y0 + ap * np.cos(t) * np.sin(phi) + bp * np.sin(t) * np.cos(phi)
    return x, y

    # We use the formulas from https://mathworld.wolfram.com/Ellipse.html

def LissajousPhase(x, y, t_start=0, bool_coeffs = False):

    """ Returns phase extracted from the Lissajous curve fitted to an ellipse. 
    Possibility to return ellipse coefficients and to choose when the limit cycle starts. 
    """

    cartesian_coeffs_ellipse = fit_ellipse(x[t_start:], y[t_start:])
    polar_coeffs_ellipse = cart_to_pol(cartesian_coeffs_ellipse)
    x0, y0, ap, bp, e, phi = polar_coeffs_ellipse

    if polar_coeffs_ellipse == [None, None, None, None, None, None]:
        phase_modulus = None
        phase_sign = None

    # y_max = np.max(y[t_start:])-y0
    # print("y_max = ", y_max)
    else:
        # Create ellipse and extract y_max
        Ell_x, Ell_y = make_ellipse_pts(polar_coeffs_ellipse, npts=10000)
        y_max = np.max(Ell_y)-y0
        x_0_index = np.where(np.abs(Ell_x-x0) == np.min(np.abs(Ell_x-x0)))[0][0]
        y_x_0 = np.abs(Ell_y[x_0_index]-y0)
        if phi < np.pi/2:
            phase_modulus = np.arcsin(y_x_0/y_max)
        elif phi < np.pi:
            phase_modulus = np.pi - np.arcsin(y_x_0/y_max)
        m = 10 # Number of vertices chosen to calculate orientation
        d = 2 # distance between "successive" points considered
        Q_m = 0
        for k in range(m):
            det_k = (x[t_start+k+d]-x[t_start+k])*(y[t_start+k+2*d]-y[t_start+k]) - (x[t_start+k+2*d]-x[t_start+k])*(y[t_start+k+d]-y[t_start+k])
            sign_det_k = np.sign(det_k)
            # print("sign_det_k = ", sign_det_k)
            Q_m += sign_det_k
        Q_m /= m
        # print("Q_m = ", Q_m)
        phase_sign = -np.sign(Q_m)
    
    # print("phase_modulus = ", phase_modulus, ", phase_sign = ", phase_sign)

    if bool_coeffs:
        if phase_sign == None:
            return polar_coeffs_ellipse, phase_sign
        else:
            return polar_coeffs_ellipse, phase_sign * phase_modulus
    else:
        if phase_sign == None:
            return phase_sign
        else:
            return phase_sign * phase_modulus

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

    return fig_shapes

def AnimatedShapes(X, Y, X_flow, N, w0, Sp4, Beta, tau_b, T_eval):
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

def IsEqual(a,b):
    " Compares a and b, or does not compare them if at least one of them is None."
    if a == None or b == None:
        return True
    elif type(a) == float and type(b) == float:
        if '{:0.3e}'.format(a) == '{:0.3e}'.format(b): # Approximation here.
            return True
        else:
            return False
    elif a==b:
        return True
    else:
        return False

if __name__ == "__main__":

    # Test Lissajous
    t = np.arange(400)
    T = 100
    x = np.sin(2*np.pi*t/T+np.pi/4)
    phi = np.pi/4 + np.pi/10
    y = np.sin(2*np.pi*t/T + phi)
    polar_coeffs, phase = LissajousPhase(x,y, t_start = 100, bool_coeffs = True)
    print("polar coeffs = ", polar_coeffs)
    a, b = make_ellipse_pts(polar_coeffs, npts = 1000, tmin = 0, tmax = 2*np.pi)
    fig_ellipse = go.Figure()
    fig_ellipse.add_scatter(x=a, y=b, mode = 'lines', name = "ellipse - Least square fit")
    fig_ellipse.add_scatter(x=x[T:T+T//4], y=y[T:T+T//4], name = "Start of the cycle")
    fig_ellipse.add_scatter(x=x[T+T//4:], y=y[T+T//4:], name = "Rest of the cycle")
    fig_ellipse.show()
    print("phase [in pi units] = ", phase/np.pi)