# Libraries
import numpy as np
import scipy.optimize as so
from optimparallel import minimize_parallel # L-BFGS-B parallel implementation

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.express.colors import sample_colorscale
import plotly.io as pio

import copy
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

### Viscoelastic Model parameters

def Viscoelastic_Model_Parameters(gamma = 2, N = 10, k0 = 1e13, bool_EI = True, Sp4 = 1, tau_b = 0, Beta = 0, tau_s = 0, init_conf = StraightLine, n_L = [0,0], m_L = 0, Lambdas = None, Zetas = None, InterpFlow = None, method = "BDF", T_span = None, T_eval = None, T_sim_max = 60, filament_type = "custom", flow_type = "sinusoidal"):
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
    
    Physically intuitive external flows can be specified in flow_type as strings, if InterpFlow is not None (or T_span or T_eval).
    Currently available flow types:
    - "constant": constant vertical flow
    - "sinusoidal": sinusoidal vertical flow

    Remark: One could replace:
        - InterpFlow -> (A, w0, psi) [Note: in general this cannot be done, i.e., if InterpFlow is from experimental data]
        - (T_span, T_eval) -> (dT, T_max) [Note: could this be recovered from InterpFlow directly?]
    But right now it is not suited because it would use memory and computational power to recompute these variables at every iteration.
    """

    # Gather all parameter names for simplicity
    param_names = ["gamma", "N", "k0", "bool_EI", "Sp4", "tau_b", "Beta", "tau_s", "X0", "n_L", "m_L", "Lambdas", "Zetas", "InterpFlow", "method", "T_span", "T_eval", "T_sim_max"]

    # Fill X0
    X0 = init_conf(N)

    # Fill None values for (Lambdas, Zetas, InterpFlow, T_span, T_eval)

    if Lambdas is None:
        ## Uniform force along length
        Lambda = [0,0]
        Lambdas = [Lambda for k in range(N)]

    if Zetas is None:
        Zeta = 0
        Zetas = [Zeta]*N

    if (InterpFlow is None) and (T_span is None) and (T_eval is None):

        # Flow types
        if flow_type in ["constant"]:

            ## Intermediate parameters
            A = 1e-2
            w0 = 0
            psi = np.pi / 2
            Flow_field_filename = ""      
            dT = 1e-1
            T_max = 1e1      

        elif flow_type in ["sinusoidal"]:

            ## Intermediate parameters
            A = 1e-2
            w0 = 1e0
            psi = np.pi / 2
            Flow_field_filename = ""      
            dT = 2*np.pi/(10*w0)
            T_max = 2*np.pi*10/w0    

        elif flow_type in ["custom"]:
            raise ValueError("Custom flow is required: (InterpFlow, T_span, T_eval) should be inserted. ")
            print("Custom external flow. ")    

        ## Final parameters
        T_span = [0, T_max]
        T_eval = [dT*i for i in range(int(T_max/dT))]

        ## Intermediate parameters
        X_flow_field_string, X_flow_field = CreateFlowField(A, w0, psi, T_eval, filename = Flow_field_filename)  

        ## Final parameters        
        if X_flow_field_string != "NO FLOW":
            InterpFlow = interpolate.interp1d(np.array(T_eval).reshape(len(T_eval),), X_flow_field, axis=1, fill_value="extrapolate")
        else:         
            InterpFlow = 0

    # In the case where (InterpFlow, T_span, T_eval) were constructed beforehand, the 3 variables should have been constructed together.
    elif not ((InterpFlow is not None) and (T_span is not None) and (T_eval is not None)):
        raise ValueError("(InterpFlow, T_span, T_eval) do not correspond to each other. One or two elements of the triplet are None. ")
    
    else:
        if flow_type in ["custom"]:
            print("Custom flow")
        else:
            print("Force custom flow.")

    # Filament types
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
    
    elif filament_type in ['custom']:
        print("Custom filament.")
    
    # Initialize dictionary
    params = {param_name:eval(param_name) for param_name in param_names}

    return params

def Viscoelastic_Model(params):
    """ 
    This functions computes the output of the coarse-grained viscoelastic model 
    given a dictionary containing all its parameters.

    parameters: gamma, N, k0, bool_EI, Sp4, tau_b, Beta, tau_s,
                X0, n_L, m_L, Lambdas, Zetas, 
                InterpFlow, method, T_span, T_eval, T_sim_max.

    Remark: Some input parameters could be changed in the future to:
        - (A, w0, psi) -> InterpFlow
        - (dT, T_max) -> (T_span, T_eval)
    But it would increase memory and computational power, as it needs to
    be evaluated at every call of the function.
    """

    # Transform scalar parameters into appropriate parameters for solving

    ## tau_b -> taus_b
    new_params = copy.deepcopy(params)
    new_params["taus_b"] = [new_params["tau_b"]]*(N-1)
    new_params.pop("tau_b")

    # INSERT HERE MODEL SIMULATION
    sol = Solve_InterpFlow(**new_params).y

    return sol

## Functional Meta-functions

def Make_Modelexpdisc_Func(disc_func, exp_data):
    """
    Returns the model-experiment discrepancy function, given a 
    discrepancy function and experimental data.
    """

    def modelexp_disc_func(model_output):
        """ 
        Return the model-experiment discrepancy, assuming fixed 
        experimental data. 
        """                

        return disc_func(model_output, exp_data)
    
    return modelexp_disc_func

def Make_Model_Functional(model, model_disc_func):
    """ 
    Makes a functional to minimize a functional linked to a model,
    given a model and a model discrepancy function.

    Example:
        - model = Viscoelastic_Model
        - model_disc_func = L2_relative_error(., exp_data)
    
    Remark: this function could be used to make a functional that quantifies 
    biologically or evolutionarily meaningful quantity, as well. For example, it 
    could be the total dissipated energy.
    """

    def model_functional(all_params):
        """
        Takes model parameters as input and computes the model discrepancy.
        """

        # Compute model output from parameters
        m_p = model(all_params)
        # Calculate model discrepancy
        discrepancy = model_disc_func(m_p)
        return discrepancy

    return model_functional

## Optimization schemes

def callback_function(xk):
    print("xk: ", xk)
    return

def Basinhopping_LBFGSB_Scheme(func, guess_variables, bounds, niter = 5):
    """
    This function aims at minimizing a functional func given 
    an initial guess guess_params and a bound bound_params.
    For that, it uses 
        - a global optimization method, the basin-hopping algorithm 
    (scipy.optimize) with 
        - a local optimization method, the L-BFGS-B method, which is a variant 
        of the BFGS method with less memory usage and the possibility to add box 
        constraints.

    Inputs:
        - func: a functional that takes a ndarray variable_params of shape 
        (n_v,1) as argument
        - guess_variables: a (n_v,1)-shaped ndarray
        - bound_params: a bound corresponding to the variable parameters and 
        according to the scipy.optimize syntax.
        - niter: number of global (basin-hopping) iterations
    """

    method = "L-BFGS-B"

    x0 = guess_variables
    minimizer_kwargs = {"method": method, "bounds": bounds, "options":{'disp': True},  "callback":callback_function}
    ret = so.basinhopping(func = func, x0 = x0, minimizer_kwargs = minimizer_kwargs, niter = niter)

    return ret

## Inference meta-function

### Reduce functional to a small set of variables
def Make_Red_Func(func, variable_keys, fixed_params):
    """
    Returns a functional that takes as input a np.ndarray of variables, 
    defined by variable_keys out of all the variables that func takes as
    input.

    Input:
        - func: a functional that takes as input many parameters
        - variable_keys: names of the variables we want to keep for the 
        reduced functional.
        - fixed_params: the rest of the parameters, which are fixed.
    Output:
        - red_func(variables = .), the reduced functional which 
        takes a np.ndarray as input, so that that functional can be 
        optimized by standard optimization solvers.
    """

    def red_func(variables):
        """
        The reduced functional.
        Warning! The variables np.ndarray should follow the same order as in the variable_params dictionary, otherwise it won't work.
        """
        # Check that variables and variable_keys are compatible
        if (len(list(variable_keys)) - variables.size) != 0:
            raise ValueError("Number of variables and number of keys is different.")

        variable_params = {key:variable for (key,variable) in zip(variable_keys, variables)}
        all_params = fixed_params | variable_params
        return func(all_params)
    
    return red_func

# Inference general function
def Infer(fixed_params, guess_variable_params, bounds, functional, opt_scheme, opt_args):
    """ 
    This function aims at inferring parameters that minimize a functional.

    Inputs:
        - fixed_params: a fixed set of parameters (dict)
        - guess_variable_params: initial guess for the variable parameters (dict)
        - bounds: optimization bounds for the variable parameters (bound Class instance)
        - functional: the functional to be minimized (function)
        - opt_scheme: optimization scheme (function)
        - opt_args: arguments for the optimization scheme (dict)

    Outputs:
        - inferred_variable_params

    Example:
        - fixed_params: {"gamma":_, "N":_, "k0":_, "bool_EI":_, "Beta":_, "taus_b":_, "tau_s":_, "X0":_, "n_L":_, "m_L":_, "Lambdas":_, "Zetas":_, "InterpFlow":_, "method":_, "T_span":_, "T_eval":_, "T_sim_max":_}
        - guess_variable_params: {"Sp4":2.0}
        - bounds_variable_params: [(eps, 1/eps)] with eps = 1e-3
        - functional: viscoelastic_modelexp_functional
        - opt_scheme: Basinhopping_LBFGSB_Scheme
        - opt_args: {"niter":5}
    """

    # Reduce functional
    variable_keys = list(guess_variable_params.keys())
    red_func = Make_Red_Func(func = functional, variable_keys = variable_keys, fixed_params = fixed_params)
    
    # Minimize functional
    guess_variables = np.array(list(guess_variable_params.values()))
    res = opt_scheme(func = red_func, guess_variables = guess_variables, bounds = bounds, **opt_args)

    # Transform back np.ndarray inferred variables into dictionary
    inferred_variables = res.x
    inferred_variable_params = copy.deepcopy(guess_variable_params)
    for k in range(len(inferred_variable_params)):
        key = list(inferred_variable_params.keys())[k]
        inferred_variable_params[key] = inferred_variables[k]
    res.x = inferred_variable_params

    return res

### Inference for the viscoelastic model
def ModelExp_Inference(exp_data, model, fixed_params, guess_variable_params, bounds, modelexpdisc_func, opt_scheme, opt_args):

    """ 
    This function infers parameters of a model, given an initial guess and 
    experimental data.

    Inputs:
        - exp_data: experimental data or simulation output. Must lie in the same space as the model output.
        - model: 
        - fixed_params:
        - guess_variable_params:
        - bounds:
        - modelexpdisc_func:
        - opt_scheme:
        - opt_args:
    Outputs:

    Example:
        - exp_data: ?
        - fixed_params: ?
        - guess_variable_params: {"Sp4":1, "tau_b":1}
        - bounds: ?
        - modelexpdisc_func: ?
        - opt_scheme: ?
        - opt_args: ?
    """

    # Set parameters
    # all_params = Viscoelastic_Model_Parameters(gamma = gamma, N = N, k0 = k0, bool_EI = bool_EI, Sp4 = Sp4, tau_b = tau_b, Beta = Beta, tau_s = tau_s, init_conf = init_conf, n_L = n_L, m_L = m_L, Lambdas = Lambdas, Zetas = Zetas, InterpFlow = InterpFlow, method = method, T_span = T_span, T_eval = T_eval, T_sim_max = T_sim_max, filament_type = "custom", flow_type = "custom")

    # Separate fixed and variable parameters
    variable_keys = ["Sp4", "tau_b"]
    variable_params = {key:all_params[key] for key in variable_keys}
    fixed_params = copy.deepcopy(all_params)
    for key in variable_keys:
        fixed_params.pop(key)


    # Inference
    guess_variable_params = variable_params
    functional = viscoelastic_modelexp_functional
    opt_scheme = Basinhopping_LBFGSB_Scheme
    niter = 1
    opt_args = {"niter":niter} # dictionary
    ret = Infer(fixed_params = fixed_params, guess_variable_params = guess_variable_params, bounds = bounds, functional = functional, opt_scheme = opt_scheme, opt_args=opt_args)
    print("inferred variables:", ret.x)

    return

# Main
if __name__ == '__main__':
    bool_test = True
    
    if bool_test:
            
        ## Function Unitary Tests

        ### Discrepancy functionals
        #### L2_relative_error(): OK
        # m1 = np.zeros((10,3))
        # m2 = np.ones((10, 3))
        # err = L2_relative_error(m1, m2)
        # print("err:", err)
        # exit()

        ### Models
        #### Viscoelastic_Model_Parameters(): OK [custom flow, custom filament]

        ##### Filament properties
        gamma = 2
        N = 10
        Sp4 = 1
        k0 = 1e13
        bool_EI = True
        Beta = 0
        tau_b = 0
        taus_b = [tau_b]*(N-1)
        tau_s = 0

        ##### Boundary conditions
        init_conf = StraightLine
        X0 = init_conf(N)

        ##### External forcings
        n_L = [0,0]
        m_L = 0
        Lambda = [0,0]
        Lambdas = [Lambda for k in range(N)]
        Zeta = 0
        Zetas = [Zeta]*N

        ##### Time-dependent Flow field
        A = 1e-2
        w0 = 1e0
        psi = np.pi / 2
        Flow_field_filename = "" # Whether to use a measured flow field or not
        
        ##### Integration and time
        method = 'BDF'
        dT = 2*np.pi/(10*w0)
        T_max = 2*np.pi*1/w0
        T_span = [0, T_max]
        T_eval = [dT*i for i in range(int(T_max/dT))]
        T_sim_max = 600

        ##### Numerical Flow field and Interpolation
        X_flow_field_string, X_flow_field = CreateFlowField(A, w0, psi, T_eval, filename = Flow_field_filename)
        if X_flow_field_string != "NO FLOW":
            InterpFlow = interpolate.interp1d(np.array(T_eval).reshape(len(T_eval),), X_flow_field, axis=1, fill_value="extrapolate")
        else:         
            InterpFlow = 0

        params = Viscoelastic_Model_Parameters(gamma = gamma, N = N, k0 = k0, bool_EI = bool_EI, Sp4 = Sp4, tau_b = tau_b, Beta = Beta, tau_s = tau_s, init_conf = init_conf, n_L = n_L, m_L = m_L, Lambdas = Lambdas, Zetas = Zetas, InterpFlow = InterpFlow, method = method, T_span = T_span, T_eval = T_eval, T_sim_max = T_sim_max, filament_type = "custom", flow_type = "custom")

        print("params:", params)
        
        #### Viscoelastic_Model(): OK 
        sol = Viscoelastic_Model(params)

        ### Functional Meta-functions
        
        #### Make_Modelexpdisc_Func: OK
        modelexp_disc_func = Make_Modelexpdisc_Func(disc_func = L2_relative_error, exp_data = sol)

        #### Make_Model_Functional(): OK
        model = Viscoelastic_Model
        viscoelastic_modelexp_functional = Make_Model_Functional(model = model, model_disc_func = modelexp_disc_func)

        all_params = params
        all_params['Sp4'] = 2 # Change one parameter to make simulation-experiment discrepancy non-zero.
        all_params['tau_b'] = 1 # Change one parameter to make simulation-experiment discrepancy non-zero.
        f = viscoelastic_modelexp_functional(all_params = all_params)
        print("Functional evaluated for a set of parameters all_params:", f)
        # exit()

        ### Optimization schemes

        #### Reduce functional input space: OK
        variable_keys = ["Sp4", "tau_b"]
        variable_params = {key:all_params[key] for key in variable_keys}
        fixed_params = copy.deepcopy(all_params)
        for key in variable_keys:
            fixed_params.pop(key)

        func = viscoelastic_modelexp_functional
        red_viscoelastic_modelexp_func = Make_Red_Func(func = func, variable_keys = variable_keys, fixed_params = fixed_params)
        
        # !The variables np.ndarray should follow the same order as in the variable_params dictionary, otherwise it won't work.
        variables = np.array(list(variable_params.values())) 
        f = red_viscoelastic_modelexp_func(variables)
        print("Reduced functional evaluated for a subset of parameters variables", f)
        # exit()

        #### Basinhopping_LBFGSB_Scheme(): OK for (1D, 2D, ...), OK with (no bounds, bounds)
        
        guess_variables = variables
        eps = 1e-3
        bound_Sp4 = [eps, 1e3]
        bound_tau_b = [0, 1e3]
        lb = [bound_Sp4[0], bound_tau_b[0]]
        ub = [bound_Sp4[1], bound_tau_b[1]]
        bounds = so.Bounds(lb,  ub)
        niter = 1 # Number of basin-hopping (global) iterations
        # ret = Basinhopping_LBFGSB_Scheme(func = red_viscoelastic_modelexp_func, guess_variables = guess_variables, bounds = bounds, niter = niter)
        # print("solution:", ret.x)
        # exit()

        ### Inference meta-function
        #### Infer(): OK for (viscoelastic model, L2-relative norm, Basin-hopping, L-BFGS-B)

        all_params = Viscoelastic_Model_Parameters(gamma = gamma, N = N, k0 = k0, bool_EI = bool_EI, Sp4 = Sp4, tau_b = tau_b, Beta = Beta, tau_s = tau_s, init_conf = init_conf, n_L = n_L, m_L = m_L, Lambdas = Lambdas, Zetas = Zetas, InterpFlow = InterpFlow, method = method, T_span = T_span, T_eval = T_eval, T_sim_max = T_sim_max, filament_type = "custom", flow_type = "custom")
        variable_keys = ["Sp4", "tau_b"]
        variable_params = {key:all_params[key] for key in variable_keys}
        fixed_params = copy.deepcopy(all_params)
        for key in variable_keys:
            fixed_params.pop(key)

        guess_variable_params = variable_params
        functional = viscoelastic_modelexp_functional
        opt_scheme = Basinhopping_LBFGSB_Scheme
        niter = 1
        opt_args = {"niter":niter} # dictionary
        ret = Infer(fixed_params = fixed_params, guess_variable_params = guess_variable_params, bounds = bounds, functional = functional, opt_scheme = opt_scheme, opt_args=opt_args)
        print("inferred variables:", ret.x)
        # exit()

        ### Inference (meta^2)-function
        #### Viscoelastic_Inference(): ?
        

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
    