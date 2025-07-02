# Libraries
import numpy as np
import scipy.optimize as so
from optimparallel import minimize_parallel # L-BFGS-B parallel implementation

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.express.colors import sample_colorscale
import plotly.io as pio

import pickle
writing_dir = "C:\\Users\\Luc\\Documents\\PhD_Large_files\\RheoFlag\\Inference\\FromSimulationData\\"
import copy
import sys
from misc_func import *
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
    res = Solve_InterpFlow(**new_params)
    sol = res.y

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

def Basinhopping_LBFGSB_Scheme(func, guess_variables, bounds, callback_function = callback_function, niter = 5):
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
        - callback_function: callback function
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
        - res (res.x is the inferred vector)

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
def ModelExp_Inference(exp_data, model, fixed_params, guess_variable_params, bounds, disc_func, opt_scheme, opt_args):

    """ 
    This function infers parameters of a model, given an initial guess and 
    experimental data.

    Inputs:
        - exp_data: experimental data or simulation output. Must lie in the same space as the model output.
        - model: the model (function)
        - fixed_params: fixed parameters (dict)
        - guess_variable_params: initial guess of the variable parameters (dict)
        - bounds: optimization bounds on the variable parameters (Bound Class instance)
        - disc_func: a discrepancy function between the model output and the experimental data (function)
        - opt_scheme: optimization scheme (function)
        - opt_args: arguments for the optimization scheme (dict)
    Outputs:
        - res (res.x is the inferred vector)

    Example:
        - exp_data: ?
        - fixed_params: ?
        - guess_variable_params: {"Sp4":1, "tau_b":1}
        - bounds: ?
        - modelexpdisc_func: ?
        - opt_scheme: ?
        - opt_args: ?
    """

    modelexpdisc_func = Make_Modelexpdisc_Func(disc_func = disc_func, exp_data = exp_data)
    modeldisc_func = Make_Model_Functional(model = model, model_disc_func = modelexpdisc_func)
    
    res = Infer(fixed_params = fixed_params, guess_variable_params = guess_variable_params, bounds = bounds, functional = modeldisc_func, opt_scheme = opt_scheme, opt_args=opt_args)

    return res

def Viscoelastic_Inference(exp_data, fixed_params, guess_variable_params, bounds, disc_func, opt_scheme, opt_args):
    return ModelExp_Inference(exp_data, Viscoelastic_Model, fixed_params, guess_variable_params, bounds, disc_func, opt_scheme, opt_args)

################################################################################
################################################################################

bool_test = False
if __name__ == '__main__':

    # Main # -------------------------------------------------------------------

    m1 = 8 # 11
    A_vec = np.float_power(10, np.linspace(-2, 5, num = m1)) # np.float_power(10, np.linspace(-5, 5, num = m1)) # np.array([1e-2])
    m2 = 11
    w0_vec = np.float_power(10, np.linspace(-5, 5, num = m2)) # np.array([1e0])
    m3 = 1 # 2
    psi_vec = np.array([np.pi/2]) # np.linspace(0, np.pi/2, num = m3)

    n1 = 1 # 14
    k0_vec = np.array([1e13]) # np.float_power(10, np.linspace(0, 13, num = n1))
    n2 = 1 # 11
    Sp4_vec = [1] # np.float_power(10, np.linspace(-5, 5, num = n2))
    n3 = 1 # 11
    tau_b_vec = [0] # np.float_power(10, np.linspace(-5, 5, num = n3))
    n4 = 1 # 11
    Beta_vec = [0] # np.float_power(10, np.linspace(-5, 5, num = n4))
    n5 = 1 # 11
    tau_s_vec = [0] # np.float_power(10, np.linspace(-5, 5, num = n5))

    m_tot = m1 * m2 * m3
    n_tot = n1 * n2 * n3 * n4 * n5
    mn_tot = m_tot * n_tot
    print("m_tot, n_tot, mn_tot:", m_tot, n_tot, mn_tot)
    IE_matrix = np.ones((m1,m2,m3,n1,n2,n3,n4,n5)) * np.nan

    ## Constructing experimental data

    ### Numerical properties
    N = 10

    #### Boundary conditions
    init_conf = StraightLine
    X0 = init_conf(N)

    #### External forcings
    n_L = [0,0]
    m_L = 0
    Lambda = [0,0]
    Lambdas = [Lambda for k in range(N)]
    Zeta = 0
    Zetas = [Zeta]*N

    ### Time-dependent flow field
    Flow_field_filename = "" # Whether to use a measured flow field or not

    for i1 in range(m1):
        A = A_vec[i1]
        for i2 in range(m2):
            w0 = w0_vec[i2]
            for i3 in range(m3):
                psi = psi_vec[i3]

                ### Integration and time
                method = 'BDF'
                dT = 2*np.pi/(10*w0)
                T_max = 2*np.pi*1/w0
                T_span = [0, T_max]
                T_eval = [dT*i for i in range(round(T_max/dT))]
                T_sim_max = 600

                ### Numerical Flow field and Interpolation
                X_flow_field_string, X_flow_field = CreateFlowField(A, w0, psi, T_eval, filename = Flow_field_filename)
                if X_flow_field_string != "NO FLOW":
                    InterpFlow = interpolate.interp1d(np.array(T_eval).reshape(len(T_eval),), X_flow_field, axis=1, fill_value="extrapolate")
                else:         
                    InterpFlow = 0
                
                print("Flow field created for (A,w0,psi) = ", A, w0, psi)
                flow_params = dict(A = A, w0 = w0, psi = psi)

                ### Filament properties
                gamma = 2
                bool_EI = True

                for j1 in range(n1):
                    k0 = k0_vec[j1]
                    for j2 in range(n2):
                        Sp4 = Sp4_vec[j2]
                        for j3 in range(n3):
                            tau_b = tau_b_vec[j3]
                            for j4 in range(n4):
                                Beta = Beta_vec[j4]
                                for j5 in range(n5):
                                    tau_s = tau_s_vec[j5]

                                    taus_b = [tau_b]*(N-1)

                                    exp_params = Viscoelastic_Model_Parameters(gamma = gamma, N = N, k0 = k0, bool_EI = bool_EI, Sp4 = Sp4, tau_b = tau_b, Beta = Beta, tau_s = tau_s, init_conf = init_conf, n_L = n_L, m_L = m_L, Lambdas = Lambdas, Zetas = Zetas, InterpFlow = InterpFlow, method = method, T_span = T_span, T_eval = T_eval, T_sim_max = T_sim_max, filament_type = "custom", flow_type = "custom")

                                    exp_data = Viscoelastic_Model(exp_params)

                                    ## Choose discrepancy function
                                    disc_func = L2_relative_error

                                    ## Choose initial guess (and fixed vs variable parameters)

                                    ### Initialize parameters perturbed around experimental parameters
                                    initial_params = copy.deepcopy(exp_params)
                                    initial_params["Sp4"] = 2.0
                                    initial_params["tau_b"] = 1.0

                                    ### Separate fixed and variable parameters
                                    variable_keys = ["Sp4", "tau_b"]
                                    exp_variable_params = {key:exp_params[key] for key in variable_keys}
                                    guess_variable_params = {key:initial_params[key] for key in variable_keys}
                                    fixed_params = initial_params
                                    for key in variable_keys:
                                        fixed_params.pop(key)

                                    ### Bounds
                                    eps = 1e-3
                                    bound_Sp4 = [eps, 1e3]
                                    bound_tau_b = [0, 1e3]
                                    lb = [bound_Sp4[0], bound_tau_b[0]]
                                    ub = [bound_Sp4[1], bound_tau_b[1]]
                                    bounds = so.Bounds(lb,  ub)

                                    ### Optimization schemes and arguments
                                    opt_scheme = Basinhopping_LBFGSB_Scheme
                                    niter = 1
                                    opt_args = {"niter":niter, "callback_function":callback_function}

                                    VI_args = dict(exp_data=exp_data, fixed_params=fixed_params, guess_variable_params=guess_variable_params, bounds = bounds, disc_func = disc_func, opt_scheme = opt_scheme, opt_args=opt_args)
                                    ret = Viscoelastic_Inference(**VI_args)

                                    ## Inference results

                                    ####################### MOVE THIS IN ANALYSIS

                                    ### Save results
                                    #### Make dictionary with all relevant information

                                    ###### Dictionary of the used function
                                    VI_dict = Make_Dict_From_Applied_Function(func = Viscoelastic_Inference, func_args = VI_args, func_output = ret)

                                    ###### Additional information
                                    VI_dict["exp_variable_params"] = exp_variable_params
                                    VI_dict["flow_params"] = flow_params
                                    
                                    # Pickle dictionary using the highest protocol available.
                                    
                                    ## Make flow part of the filename
                                    base_id = "_Flow"
                                    for key in list(flow_params.keys()):
                                        param = flow_params[key]
                                        base_id += "_" + key + "_" + f"{param:.2E}"                                    
                                    base_id += "_Fixed"
                                    for key in list(fixed_params.keys()):
                                        # Exclude non-scalar parameters
                                        if key in ["A", "w0", "psi", "gamma", "N", "k0", "Sp4", "tau_b", "Beta", "tau_s"]:
                                            param = fixed_params[key]
                                            base_id += "_" + key + "_" + f"{param:.2E}"
                                    base_id += "_Variable"
                                    for key in list(exp_variable_params.keys()):
                                        param = exp_variable_params[key]
                                        base_id += "_" + key + "_" + f"{param:.2E}"
                                    filename = writing_dir + "VI_dict" + base_id + ".pkl"
                                    output = open(filename, 'wb')
                                    pickle.dump(obj = VI_dict, file = output, protocol = -1)
                                    output.close()

    # ----------------------------------------------------------------- # Main # 

    
    # Tests # ------------------------------------------------------------------
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
        # f = viscoelastic_modelexp_functional(all_params = all_params)
        # print("Functional evaluated for a set of parameters all_params:", f)
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

        # guess_variable_params = variable_params
        # functional = viscoelastic_modelexp_functional
        # opt_scheme = Basinhopping_LBFGSB_Scheme
        # niter = 1
        # opt_args = {"niter":niter} # dictionary
        # ret = Infer(fixed_params = fixed_params, guess_variable_params = guess_variable_params, bounds = bounds, functional = functional, opt_scheme = opt_scheme, opt_args=opt_args)
        # print("inferred variables:", ret.x)
        # exit()

        #### ModelExp_Inference: OK (Viscoelastic model, simulation output data)

        # Include here sol = Viscoelastic_Model(...) if data = simulation output

        exp_data = sol

        # Set parameters
        all_params = Viscoelastic_Model_Parameters(gamma = gamma, N = N, k0 = k0, bool_EI = bool_EI, Sp4 = Sp4, tau_b = tau_b, Beta = Beta, tau_s = tau_s, init_conf = init_conf, n_L = n_L, m_L = m_L, Lambdas = Lambdas, Zetas = Zetas, InterpFlow = InterpFlow, method = method, T_span = T_span, T_eval = T_eval, T_sim_max = T_sim_max, filament_type = "custom", flow_type = "custom")

        model = Viscoelastic_Model
        disc_func = L2_relative_error

        # Separate fixed and variable parameters
        variable_keys = ["Sp4", "tau_b"]
        guess_variable_params = {key:all_params[key] for key in variable_keys}
        fixed_params = copy.deepcopy(all_params)
        for key in variable_keys:
            fixed_params.pop(key)

        eps = 1e-3
        bound_Sp4 = [eps, 1e3]
        bound_tau_b = [0, 1e3]
        lb = [bound_Sp4[0], bound_tau_b[0]]
        ub = [bound_Sp4[1], bound_tau_b[1]]
        bounds = so.Bounds(lb,  ub)

        opt_scheme = Basinhopping_LBFGSB_Scheme
        niter = 1
        opt_args = {"niter":niter, "callback_function":callback_function} # dictionary

        # res = ModelExp_Inference(exp_data=exp_data, model = Viscoelastic_Model, fixed_params=fixed_params, guess_variable_params=guess_variable_params, bounds = bounds, disc_func = disc_func, opt_scheme = opt_scheme, opt_args=opt_args)
        # print("Solution: ", res.x)
        # exit()

        #### Viscoelastic_Inference(): OK (Viscoelastic model, simulation output data)
        # res = Viscoelastic_Inference(exp_data=exp_data, fixed_params=fixed_params, guess_variable_params=guess_variable_params, bounds = bounds, disc_func = disc_func, opt_scheme = opt_scheme, opt_args=opt_args)
        # print("Solution: ", res.x)
        # exit()
        
    # ---------------------------------------------------------------- # Tests #
