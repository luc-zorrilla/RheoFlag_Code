### Libraries
import numpy as np
import scipy.optimize as so

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.express.colors import sample_colorscale
import plotly.io as pio

import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, 'c:\\Users\\Luc\\Documents\\MEGAsync\\PhD\\RheoFlag\\Code\\Model')
from A01_Coarse_grained_axoneme_functions import *

if __name__ == '__main__':
    print("")

    #--------------------------------
    # Fix parameters

    ## Filament properties
    gamma = 2
    N = 10
    Sp4 = 1
    k0 = 1e13
    bool_EI = True
    Beta = 0
    tau_b = 0
    taus_b = [tau_b]*(N-1)
    tau_s = 0

    ## Boundary conditions
    init_conf = StraightLine
    X0 = init_conf(N)

    ## External forcings
    n_L = [0,0]
    m_L = 0
    Lambda = [0,0]
    Lambdas = [Lambda for k in range(N)]
    Zeta = 0
    Zetas = [Zeta]*N

    ### Time-dependent Flow field
    A = 1e-5
    w0 = 1e-1
    psi = np.pi / 2
    Flow_field_filename = "" # Whether to use a measured flow field or not
    
    ## Integration and time
    method = 'BDF'
    dT = 2*np.pi/(10*w0)
    T_max = 2*np.pi*10/w0
    T_span = [0, T_max]
    T_eval = [dT*i for i in range(int(T_max/dT))]
    T_sim_max = 600

    ## Numerical Flow field and Interpolation
    X_flow_field_string, X_flow_field = CreateFlowField(A, w0, psi, T_eval, filename = Flow_field_filename)
    if X_flow_field_string != "NO FLOW":
        InterpFlow = interpolate.interp1d(np.array(T_eval).reshape(len(T_eval),), X_flow_field, axis=1, fill_value="extrapolate")
    else:         
        InterpFlow = 0

    ## Choose which parameters are relevant 

    ### Sp4 is the only varying parameter here
    def h(Sp4):
        return Solve_InterpFlow(gamma, N, Sp4, k0, bool_EI, Beta, taus_b, tau_s, X0, n_L, m_L, Lambdas, Zetas, InterpFlow, method, T_span, T_eval, T_sim_max).y
    #--------------------------------

    #--------------------------------
    # Minimization

    ## Set h_exp with simulation (or take from experimental data)
    Sp4 = 1
    h_exp = h(Sp4)
    print("h_exp found. ")

    ## Set a functional to be minimized

    def J(Sp4):
        """ Returns the L2-norm (squared) of the difference between the experimental output h_exp and the simulation output h_Sp4, computed as h(Sp4) and lying in the same space as h_exp.
        """

        h_Sp4 = h(Sp4)
        return (np.linalg.norm(h_Sp4-h_exp) / np.linalg.norm(h_exp))**2

    def callback_function(xk):
        print("xk: ", xk)
        return

    ## Minimize J

    ### Grid search
    # C mor frero

    ### BFGS (scipy)
    x0 = 5
    res = so.minimize(J, x0, method='BFGS', callback=callback_function, options={'disp': True} )

    ### Basin-hopping + BFGS (scipy)
    # minimizer_kwargs = {"method": "BFGS"}
    # x0 = 5
    # ret = basinhopping(J, x0, minimizer_kwargs=minimizer_kwargs, niter=20)
    # print("The global minimum is: ", ret.x, ret.fun)

    print("res.x", res.x)
    print("res.success", res.success)
    print("res.status", res.status)
    print("res.message", res.message)
    print("res.fun, res.jac, res.hess_inv", res.fun, res.jac, res.hess_inv)
    print("res.nfev, res.njev, res.nit", res.nfev, res.njev, res.nit)


    #--------------------------------
    