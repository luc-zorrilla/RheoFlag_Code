# Libraries
import numpy as np
import scipy.optimize as so
from optimparallel import minimize_parallel # L-BFGS-B parallel implementation

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.express.colors import sample_colorscale
import plotly.io as pio

import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, 'c:\\Users\\Luc\\Documents\\MEGAsync\\PhD\\RheoFlag\\Code\\Model')
from A01_Coarse_grained_axoneme_functions import *

# Functions

## Discrepancy functionals

def L2_relative_error(m1, m2):
    """ Computes the relative L2-norm of the discrepancy between m1 and m2. 
    Warning: this function will depend asymmetrically on m1 and m2, 
        m2 being the reference. """
    return (np.linalg.norm(m1 - m2) / np.linalg.norm(m2))**2

## Models

def Viscoelastic_Model(params):
    """ 
    This functions computes the output of the coarse-grained viscoelastic model 
    given a dictionary containing all its parameters.

    parameters: gamma, N, k0, bool_EI, Sp4, tau_b, Beta, tau_s,
                X0, n_L, m_L, Lambdas, Zetas, 
                InterpFlow, method, T_span, T_eval, T_sim_max.
    """

    params["taus_b"] = [params["tau_b"]]*(N-1)
    params.pop("tau_b")

    # INSERT HERE MODEL SIMULATION
    sol = Solve_InterpFlow(**params).y

    return sol

### Viscoelastic Model parameters

def Viscoelastic_model_parameters(gamma = 2, N = 10, Sp4 = 1, k0 = 1e13, bool_EI = True, Beta = 0, tau_b = 0, tau_s = 0, init_conf = StraightLine, n_L = [0,0], m_L = 0, Lambdas = None, Zetas = None, A = 1e-2, w0=1e0, method = "BDF", T_span = None, T_eval = None, T_sim_max = 60, filament_type = "bending_filament_no_viscosity"):
    """ 
    Gather all parameters of the viscoelastic model in a dictionary. If the parameter is None, give a default value.

    Physically intuitive filaments can be specified in filament_type as strings.
    Currently available filament types:
    - "bending_filament_no_viscosity"
    - "shearing_filament_no_viscosity"
    - "bending_and_shearing_filament_no_viscosity"
    - "bending_filament"
    - "shearing_filament"
    - "bending_and_shearing_filament"
    """

    # Check if param is None. If it is, give a default value.
    
    # Handle filament types
    if filament_type in ["bending_filament_no_viscosity"]:
        bool_EI = True
        Beta = 0
        tau_b = 0
        tau_s = 0
    
    elif filament_type in ["shearing_filament_no_viscosity"]:
        bool_EI = False
        Beta = 1
        tau_b = 0
        tau_s = 0

    elif filament_type in ["bending_and_shearing_filament_no_viscosity"]:
        bool_EI = True
        Beta = 1
        tau_b = 0
        tau_s = 0        

    elif filament_type in ["bending_filament"]:
        bool_EI = True
        Beta = 0
        tau_b = 1
        tau_s = 0  
    
    elif filament_type in ["shearing_filament"]:
        bool_EI = False
        Beta = 1
        tau_b = 0
        tau_s = 1          

    elif filament_type in ["bending_and_shearing_filament"]:
        bool_EI = True
        Beta = 1
        tau_b = 1
        tau_s = 1

    # param_list = [gamma, N, Sp4, k0, bool_EI, Beta, tau_b, tau_s, ...]
    # Make dictionary params from list param_list

    return params

## Functional Meta-functions

def Make_ModelExp_Functional(model, disc_func):
    """ 
    Makes a functional to minimize the experiment-model discrepancy,
    given a model and a discrepancy function.

    Example:
        - model = Viscoelastic_Model
        - disc_func = L2_relative_error
    
    Remark: this function could be used to make a functional that quantifies 
    biologically or evolutionarily meaningful quantity, as well. For example, it 
    could be the total dissipated energy.
    """

    def modelexp_functional(all_params, exp_data):
        """
        Takes model parameters and experimental data (or simulation outputs)
        as inputs and computes the model-experiment discrepancy.
        """

        # Compute model output from parameters
        m_p = model(all_params)
        # Calculate model-experiment discrepancy
        discrepancy = disc_func(m_p, exp_data)
        return discrepancy

    return modelexp_functional

# TEST: viscoelastic_modelexp_functional = Make_ModelExp_Functional(model = Viscoelastic_Model, discfunc = L2_relative_error)

## Optimization schemes

def Basinhopping_LBFGSB_Scheme(func, guess_params, bounds_params):
    """
    This function aims at minimizing a functional func given 
    an initial guess guess_params and a bound bounds_params.
    For that, it uses a global optimization method, the basin-hopping algorithm
    (scipy.optimize) with a local optimization method, the L-BFGS-B method, 
    which is a variant of the BFGS method with less memory usage and the 
    possibility to add box constraints.
    """
    method = "L-BFGS-B"
    niter = 5

    x0 = guess_params
    bounds = bounds_params
    minimizer_kwargs = {"method": method, "bounds": bounds, "options":{'disp': True},  "callback":callback_function}
    ret = so.basinhopping(func, x0, minimizer_kwargs=minimizer_kwargs, niter=niter)

    # Can be added here some reader-friendly outputs

    return ret

## Inference meta-function

def Infer(fixed_params, guess_variable_params, bounds_variable_params, functional, opt_scheme):
    """ 
    This function aims at inferring parameters that minimize a functional.

    Inputs:
        - fixed_params: a fixed set of parameters
        - guess_variable_params: initial guess for the variable parameters that we want to infer (None if unknown)
        - functional: the functional to be minimized
        - opt_scheme: optimization scheme

    Outputs:
        - inferred_variable_params (main result!)

    Example:
        - fixed_params: {"gamma":_, "N":_, "k0":_, "bool_EI":_, "Beta":_, "taus_b":_, "tau_s":_, "X0":_, "n_L":_, "m_L":_, "Lambdas":_, "Zetas":_, "InterpFlow":_, "method":_, "T_span":_, "T_eval":_, "T_sim_max":_}
        - guess_variable_params: {"Sp4":2.0}
        - bounds_variable_params: [(eps, 1/eps)] with eps = 1e-3
        - functional: viscoelastic_modelexp_functional
        - opt_scheme: Basinhopping_LBFGSB_Scheme
    """

    # Reduce functional
    def red_func(variable_params):
        """Reduce the functional to the variable parameters only. """
        all_params = fixed_params | variable_params
        return functional(all_params)
    
    # Minimize functional
    res = opt_scheme(red_func, guess_variable_params, bounds_variable_params)

    return res

# Main
if __name__ == '__main__':

    ## Function Unitary Tests

    ### Discrepancy functionals
    #### L2_relative_error(): OK
    # m1 = np.zeros((10,3))
    # m2 = np.ones((10, 3))
    # err = L2_relative_error(m1, m2)
    # print("err:", err)
    # exit()

    ### Models
    #### Viscoelastic_model_parameters()
    #### Viscoelastic_Model()

    ### Functional Meta-functions
    #### Make_ModelExp_Functional()

    ### Optimization schemes
    #### Basinhopping_LBFGSB_Scheme()

    ### Inference meta-function
    #### Infer()

    exit()

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
    A = 1e-2
    w0 = 1e0
    psi = np.pi / 2
    Flow_field_filename = "" # Whether to use a measured flow field or not
    
    ## Integration and time
    method = 'BDF'
    dT = 2*np.pi/(10*w0)
    T_max = 2*np.pi*1/w0
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
    # def h(Sp4):
    #     return Solve_InterpFlow(gamma, N, Sp4, k0, bool_EI, Beta, taus_b, tau_s, X0, n_L, m_L, Lambdas, Zetas, InterpFlow, method, T_span, T_eval, T_sim_max).y

    ### Sp4, taus_b
    # def h(p):
    #     global gamma, N, k0, bool_EI, taus_b, tau_s, X0, n_L, m_L, Lambdas, Zetas, InterpFlow, method, T_span, T_eval, T_sim_max

    #     Sp4, tau_b = p
    #     taus_b = [tau_b]*(N-1)
    #     sol = Solve_InterpFlow(gamma, N, Sp4, k0, bool_EI, Beta, taus_b, tau_s, X0, n_L, m_L, Lambdas, Zetas, InterpFlow, method, T_span, T_eval, T_sim_max).y
    #     return sol

    ### Sp4, Beta
    def h(p):
        global gamma, N, k0, bool_EI, taus_b, tau_s, X0, n_L, m_L, Lambdas, Zetas, InterpFlow, method, T_span, T_eval, T_sim_max
        Sp4, Beta = p
        sol = Solve_InterpFlow(gamma, N, Sp4, k0, bool_EI, Beta, taus_b, tau_s, X0, n_L, m_L, Lambdas, Zetas, InterpFlow, method, T_span, T_eval, T_sim_max).y
        return sol

    ### Sp4, tau_b, Beta
    def h(p):
        global gamma, N, k0, bool_EI, tau_s, X0, n_L, m_L, Lambdas, Zetas, InterpFlow, method, T_span, T_eval, T_sim_max
        Sp4, tau_b, Beta = p
        taus_b = [tau_b]*(N-1)
        sol = Solve_InterpFlow(gamma, N, Sp4, k0, bool_EI, Beta, taus_b, tau_s, X0, n_L, m_L, Lambdas, Zetas, InterpFlow, method, T_span, T_eval, T_sim_max).y
        return sol

    # # All parameters used to compute h
    # all_params = {"gamma":gamma, "N":N, "Sp4":Sp4, "k0":k0, "bool_EI":bool_EI, "Beta":Beta, "taus_b":taus_b, "tau_s":tau_s, "X0":X0, "n_L":n_L, "m_L":m_L, "Lambdas":Lambdas, "Zetas":Zetas, "InterpFlow":InterpFlow, "method":method, "T_span":Tspan, "T_eval":T_eval, "T_sim_max":T_sim_max}

    # # All parameters that could potentially be allowed to vary
    # all_variable_params = ["gamma", "k0", "Sp4", "tau_b", "Beta", "tau_s"]

    # ### Initial parameter guess
    # params_0 = {"gamma":2.0, "k0":1e13, "Sp4":1.0, "tau_b":0, "Beta":0, "tau_s":0}

    #--------------------------------

    #--------------------------------
    # Minimization

    ## Set h_exp with simulation (or take from experimental data)
    Sp4 = 1.0
    Beta = 1.0
    # tau_b = 0
    p = [Sp4, Beta]
    # h_exp = h(Sp4)
    h_exp = h(p)
    print("h_exp found: ", h_exp.shape)

    ## Set a functional to be minimized

    def J(p):
        """ Returns the L2-norm (squared) of the difference between the experimental output h_exp and the simulation output h_p, computed as h(p) and lying in the same space as h_exp.
        """
        h_p = h(p)
        return (np.linalg.norm(h_p-h_exp) / np.linalg.norm(h_exp))**2

    def callback_function(xk):
        print("xk: ", xk)
        return

    ## Minimize J

    ### Grid search [Homemade]
    # C mor frero

    ### BFGS [scipy]
    # x0 = 5
    # res = so.minimize(J, x0, method='BFGS', callback=callback_function, options={'disp': True} )

    ### L-BFGS-B [scipy]
    #### This method allows to specify "maxls", the max number of line search per iteration, can be parallelized (see below), and allows to add parameter bounds.
    # epsilon = 1e-2
    # x0 = 90 # 50 converges (to 1), but x0 >= 94 is stuck converging at x0.
    # bounds = [(epsilon, x0+epsilon)]
    # res = so.minimize(J, x0, method='L-BFGS-B', callback=callback_function, options={'disp': True}, bounds = bounds)

    ### L-BFGS-B (parallel version) [optimparallel] --> NOT WORKING FOR NOW
    # epsilon = 1e-2
    # x0 = 90 # 50 converges (to 1), but x0 >= 94 is stuck converging at x0.
    # bounds = [(epsilon, x0+epsilon)]
    # res = minimize_parallel(fun = J, x0 = x0, bounds = bounds)    

    ### Basin-hopping + BFGS (or any other local method) [scipy]
    # x0 = 90
    x0 = [1.0, 10.0]
    epsilon = 1e-2
    bounds = [(epsilon, 2*p[0]), (epsilon, 2*p[1])]
    minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds, "options":{'disp': True},  "callback":callback_function}
    ret = so.basinhopping(J, x0, minimizer_kwargs=minimizer_kwargs, niter=5)

    print("ret.x", ret.x)
    print("ret.success", ret.success)
    print("ret.message", ret.message)
    
    res = ret.lowest_optimization_result 
    print("res.x", res.x)
    print("res.success", res.success)
    print("res.status", res.status)
    print("res.message", res.message)
    print("res.fun, res.jac, res.hess_inv operator, res.hess_inv @ res.x", "ress.hess_inv.todense()", res.fun, res.jac, res.hess_inv, res.hess_inv @ res.x, res.hess_inv.todense())
    print("res.nfev, res.njev, res.nit", res.nfev, res.njev, res.nit)

    #--------------------------------
    