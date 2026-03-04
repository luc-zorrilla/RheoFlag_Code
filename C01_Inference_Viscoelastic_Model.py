""" This file is 
    - gathering functions useful for inference at a high-level
    - """

### Libraries ###

from misc_func import *
from A01_Coarse_grained_axoneme_functions import *
import multiprocessing as mp

import dill as pickle # enhanced pickle library that handles function pickling as well
from pathlib import Path

writing_path = (Path(__file__).resolve().parent.parent / 'Inference' / 'FromSimulationData' / 'MultiplePeriods' / 'LastPeriod' / 'BendingElasticity_BendingViscosity_Clamped' / 'FixedSp4')
from datetime import datetime
import copy

import numpy as np
from scipy.optimize import minimize, Bounds
from _basinhopping_mod import *
import scipy.differentiate as sd

## Functions

### Discrepancy functionals

def L2_absolute_error(m1, m2):
    """ Computes the squared absolute L2-norm of the discrepancy between m1 and m2. 
    Remark: this function is symmetric w.r.t. m1 and m2. """
    return (np.linalg.norm(m1 - m2))**2

def L2_relative_error(m1, m2):
    """ Computes the squared relative L2-norm of the discrepancy between m1 and m2. 
    Warning: this function will depend asymmetrically on m1 and m2, 
        m2 being the reference. """
    return (np.linalg.norm(m1 - m2) / np.linalg.norm(m2))**2

### Functional Meta-functions

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

### Optimization schemes

class RandomDisplacementBounds(object):
    """random displacement with bounds:  see: https://stackoverflow.com/a/21967888/2320035
        Modified! (dropped acceptance-rejection sampling for a more specialized approach)
    """
    def __init__(self, bounds, stepsize=0.5):
        self.bounds = bounds
        self.stepsize = stepsize

    def __call__(self, x):
        """take a random step but ensure the new position is within the bounds """

        sl, sb = self.bounds.residual(x) # Lower and upper residual between x and the bounds
        min_step = np.maximum(-sl, -self.stepsize)
        max_step = np.minimum(sb, self.stepsize)

        random_step = np.random.uniform(low=min_step, high=max_step, size=x.shape)
        xnew = x + random_step

        return xnew

def Basinhopping_LBFGSB_Scheme(func, guess_variables, bounds, niter = 0, T = 0, stepsize = 5, tol = 1e-8, ftol = 1e-8, gtol = 1e-8, eps = 1e-8, jac = '3-point', finite_diff_rel_step = 1e-6, minimum_gradient = False, minimum_hessian = False):
    """
    This function aims at minimizing a functional func given an initial guess guess_params and a bound bounds.
    For that, it uses 
        - a global optimization method, the basin-hopping algorithm (scipy.optimize).
        The basin-hopping algorithm is iterative with each cycle composed of the following features
            - random perturbation of the coordinates
            - local minimization
            - accept or reject the new coordinates based on the minimized function value
        - a local optimization method, the L-BFGS-B method, which is a variant of the BFGS method with less memory usage and the possibility to add box  constraints.

    Inputs:
        - func: a functional that takes a ndarray variable_params of shape (n_v,1) as argument
        - guess_variables: a (n_v,1)-shaped ndarray
        - bound_params: a bound corresponding to the variable parameters and according to the scipy.optimize syntax.
        - niter: number of global (basin-hopping) iterations. The number of runs of the local minimizer will be niter+1. 
        If niter = 0, the basin-hopping algorithm simplifies into the local minimization scheme.
        - T: temperature of the basin-hopping algorithm, corresponding to the temperature of the metropolis algorithm for acceptance of a step. 
        If T = 0, then steps are only accepted if they minimize the functional.
        - stepsize: the maximum stepsize for the algorithm to randomly vary parameters.
        - tol: tolerance for the basin-hopping algorithm, using the global callback function
        - eps:
        - jac:
        - finite_diff_rel_step:
        - compute_minimum_gradient: whether to compute jacobian at found global minimum
        - compute_minimum_hessian: whether to compute hessian at found global minimum

    Outputs: 
        - ret is a Batch object containing all information resulting from the basinhopping algorithm -- including callbacks (still to be added).
    """

    method = "L-BFGS-B"
    x0 = guess_variables
    
    X_local = []
    F_local = []

    def wrapped_minimize(fun, x0, args=(), method=None, jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None):
        """ A wrapper of scipy.optimize.minimize used specifically for the basin-hopping algorithm, so that we obtain the starting point of each local minimization. """

        x = copy.deepcopy(x0)
        f = fun(x)
        X_local.append([x])
        F_local.append([f])

        print("x0: ", x)
        print("f(x0): ", f)

        return minimize(fun, x0, args=args, method=method, jac=jac, hess=hess, hessp=hessp, bounds=bounds, constraints=constraints, tol=tol, callback=callback, options=options)

    def local_callback_function(*, intermediate_result): # The star forces intermediate_result as a keyword argument
        """ Callback function for the local minimizer, i.e., the L-BFGS-B algorithm. """

        x_loc = copy.deepcopy(intermediate_result.x)
        f_loc = copy.deepcopy(intermediate_result.fun)

        X_local[-1].append(x_loc)
        F_local[-1].append(f_loc)

        k = len(X_local[-1]) - 1
        print("L-BFGS-B: (k, xk, f(xk)):", k, x_loc, f_loc)

        return    

    X_global = []
    F_global = []
    accept_global = []
    def global_callback_function(x,f,accept):
        """ Callback function for the global minimizer, i.e., the basin-hopping algorithm. """

        x_glob = copy.deepcopy(x)
        f_glob = copy.deepcopy(f)
        X_global.append(x_glob)
        F_global.append(f_glob)
        accept_global.append(accept)

        k = len(X_global) - 1
        print("Basin-hopping: (k,x_k,f_k,accept_k):", k, x_glob, f_glob, accept)

        # Stop if functional is lower than a tolerance threshold
        if f < tol:
            return True
        return
    
    # Local minimizer arguments
    minimizer_kwargs = {"method": method, 'jac':jac, "bounds": bounds, "options":{'disp': True ,'ftol':ftol, 'gtol':gtol, 'eps': eps, 'finite_diff_rel_step':finite_diff_rel_step}, "callback":local_callback_function}
    
    # Global minimizer arguments
    bounded_step = RandomDisplacementBounds(bounds)

    # Run global minimization algorithm
    ret = basinhopping(func = func, x0 = x0, minimizer_kwargs = minimizer_kwargs, niter = niter, stepsize = stepsize, T = T, callback = global_callback_function, minimize_wrapper = wrapped_minimize, take_step=bounded_step)

    x_final = ret.x
    
    ## Check if on boundary and choose direction of gradient approximation accordingly (Directed Backward Difference [+-1] or Central Difference [0])
    sl, sb = bounds.residual(x_final) # Lower and upper residual between x_final and the bounds
    if np.all(sl==0):
        step_direction = 1 # Lower boundary is reached
    elif np.all(sb==0):
        step_direction = -1 # Upper boundary is reached        
    elif np.all(sl >= 0) & np.all(sb >= 0): # Within domain
        step_direction = 0
    else:
        step_direction = np.inf
        raise ValueError("x_final is outside of the domain. sl, sb = ", sl, sb)

    # Compute gradient and hessian at convergence point (if convergence)
    
    if minimum_gradient or minimum_hessian:

        m = guess_variables.shape[0] # number of variables
        vec_func = Vectorize_Functional(func, m)
        if minimum_gradient:
            if ret['success']:
                g = sd.jacobian(f = vec_func, x = x_final, step_direction = step_direction).df
                print("Final gradient = ", g)
            else: # Convergence failed
                g = np.ones((m,))*np.inf
                print("(No convergence) Final gradient = ", g)
            ret.setdefault('jacobian', g)
        if minimum_hessian:
            if ret['success']:
                hess = sd.hessian(f = vec_func, x = x_final)
                if hess['success']:
                    h = hess.ddf
                else:
                    print("Hessian calculation failed. Status", hess.status)
                    h = np.zeros((m,m))
                print("Final hessian = ", h)
            else: # Convergence failed
                h = np.zeros((m,m))
                print("(No convergence) Final hessian = ", h)
            ret.setdefault('hessian', h)

            # Assign infinite error if convergence point is on the domain boundary
            if np.abs(np.abs(step_direction) - 1) == 0:
                h = np.zeros((m,m))
                ret['hessian'] = h

    return ret, X_local, F_local, X_global, F_global, accept_global

### Inference meta-function

#### Reduce functional to a small set of variables
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
        Warning! This functional is not vectorized.
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
        - X_local, F_local: iterations of the local optimisation algorithm
        - X_global, F_global, accept_global: iterations of the global optimisation algorithm
        - red_func: The reduced functional to be minimised (not vectorized!), taking only variable parameters as input

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
    res, X_local, F_local, X_global, F_global, accept_global = opt_scheme(func = red_func, guess_variables = guess_variables, bounds = bounds, **opt_args)

    # Transform back np.ndarray inferred variables into dictionary
    inferred_variables = res.x
    inferred_variable_params = copy.deepcopy(guess_variable_params)
    for k in range(len(inferred_variable_params)):
        key = list(inferred_variable_params.keys())[k]
        inferred_variable_params[key] = inferred_variables[k]
    res.x = inferred_variable_params

    return res, X_local, F_local, X_global, F_global, accept_global, red_func

### Inference for model-experiment optimization
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
    return Infer(fixed_params = fixed_params, guess_variable_params = guess_variable_params, bounds = bounds, functional = modeldisc_func, opt_scheme = opt_scheme, opt_args=opt_args)

### Inference for the viscoelastic model
def Viscoelastic_Inference(exp_data, fixed_params, guess_variable_params, bounds, disc_func, opt_scheme, opt_args):
    return ModelExp_Inference(exp_data, Viscoelastic_Model, fixed_params, guess_variable_params, bounds, disc_func, opt_scheme, opt_args)

def Viscoelastic_Inference_LP(exp_data, fixed_params, guess_variable_params, bounds, disc_func, opt_scheme, opt_args):
    """ Same as Viscoelastic_Inference but keeping the last 10th of the solution in time"""
    return ModelExp_Inference(exp_data, Viscoelastic_Model_LP, fixed_params, guess_variable_params, bounds, disc_func, opt_scheme, opt_args)

### Inference function in main loop
def Viscoelastic_inference_inloop(flow_params, exp_params, guess_variable_params, bounds, disc_func, opt_scheme, opt_args, writing_path):
    """ 
    This functions performs inference given a certain set of arguments.
    This function is used to parallelize computation within loops. 
    """

    ## Choose initial guess (and fixed vs variable parameters)

    ### Initialize parameters perturbed around experimental parameters

    initial_params = copy.deepcopy(exp_params)
    for key in guess_variable_params.keys():
        initial_params[key] = guess_variable_params[key]

    exp_variable_params = {key:exp_params[key] for key in guess_variable_params.keys()}
    
    ### Separate fixed and variable parameters
    fixed_params = initial_params
    for key in guess_variable_params.keys():
        fixed_params.pop(key)

    ### Compute filament and inference
    exp_data = Viscoelastic_Model_LP(exp_params)
    VI_args = dict(exp_data=exp_data, fixed_params=fixed_params, guess_variable_params=guess_variable_params, bounds = bounds, disc_func = disc_func, opt_scheme = opt_scheme, opt_args=opt_args)
    ret, X_local, F_local, X_global, F_global, accept_global, red_func = Viscoelastic_Inference_LP(**VI_args)

    ## Inference results

    ### Save results
    #### Make dictionary with all relevant information

    ###### Dictionary of the used function
    VI_dict = Make_Dict_From_Applied_Function(func = Viscoelastic_Inference, func_args = VI_args, func_output = [ret, X_local, F_local, X_global, F_global, accept_global, red_func])

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
    base_id += "_Guess"
    for key in list(guess_variable_params.keys()):
        param = guess_variable_params[key]
        base_id += "_" + key + "_" + f"{param:.2E}"     
    filename = str((writing_path / ("VI_dict" + base_id + ".pkl")).resolve())
    print("filename:", filename)
    output = open(filename, 'wb')
    pickle.dump(obj = VI_dict, file = output, protocol = -1)
    output.close()                                            

################################################################################
################################################################################

bool_main = True
if __name__ == '__main__':
    """ 
    In the main script, one makes a scan over parameter space of the viscoelastic filament. 
    For each point in parameter space, an inference is made in a subset of parameter space given an initial guess.
    This method allows to perform inference on the "same" viscoelastic filament but for various external forcings. 
    The inference follows a L-BFGS-B local optimization scheme combined with a Basin-Hopping global optimization scheme. 
    For each inference, relevant data is saved in a pkl (pickled) file, which will then be used in VI_analysis. 
    """

    if bool_main:
    ### Main

        # Optimization parameters

        ## Choose discrepancy function
        disc_func = L2_relative_error

        ## Optimization schemes and arguments
        opt_scheme = Basinhopping_LBFGSB_Scheme
        niter = 9 # number of iterations - 1 of the local minimizer in the Basin-hopping algorithm
        T = 0 # Temperature of the Basin-hopping algorithm. If T=0, only steps minimizing energy are accepted (apparently not...).
        stepsize = 5 # Step size of the Basin-hopping algorithm
        jac = '3-point'
        eps = 1e-8
        tol = 1e-10 # Tolerance threshold for the basin-hopping algorithm

        ftol = 1e-8 # Tolerance functional threshold for the local minimizer
        gtol = 1e-8 # Tolerance gradient threshold for the local minimizer        
        finite_diff_rel_step = 1e-6 # Maximum step size for finite difference calculation of the gradient
        minimum_gradient = False # Whether to compute gradient at found minimum
        minimum_hessian = True # Whether to compute hessian at found minimum
        opt_args = {"niter":niter, "T":T, "stepsize":stepsize, 'jac':jac, "ftol":ftol, "gtol":gtol, "eps":eps, "finite_diff_rel_step":finite_diff_rel_step, "minimum_gradient":minimum_gradient, "minimum_hessian":minimum_hessian, 'tol':tol}
        
        Sp4_guess = 1e1
        Beta_guess = 0
        tau_b_guess = 0
        tau_s_guess = 0
        for Sp4_guess in [1e1]:

            guess_variable_params = {'tau_b':tau_b_guess} # 'Beta':Beta_guess, 'tau_b':tau_b_guess, 'tau_s':tau_s_guess}

            ## Bounds 
            Sp4_min = np.double(1e-6)
            Sp4_max = np.double(1e6)
            bound_Sp4 = [Sp4_min, Sp4_max]

            Beta_min = 0
            Beta_max = np.double(1e9)
            bound_Beta = [Beta_min, Beta_max]

            tau_b_min = 0
            tau_b_max = np.double(1e9)
            bound_tau_b = [tau_b_min, tau_b_max]

            tau_s_min = 0
            tau_s_max = np.double(1e9)
            bound_tau_s = [tau_s_min, tau_s_max]            
            
            lb = [eval("bound_"+param)[0] for param in guess_variable_params.keys()]
            ub = [eval("bound_"+param)[1] for param in guess_variable_params.keys()]
            bounds = Bounds(lb,  ub)

            # Flow field
            m1 = 8
            A_vec = np.float_power(10, np.linspace(-10, -3, num = m1))
            m2 = 16
            w0_vec = np.float_power(10, np.linspace(-9, 6, num = m2))
            m3 = 1
            psi_vec = np.array([np.pi/2]) # np.linspace(0, np.pi/2, num = m3)

            ### Filament properties
            gamma = 2
            bool_EI = True

            # Viscoelastic properties
            n1 = 1 # 14
            k0_vec = np.array([1e13]) # np.float_power(10, np.linspace(0, 13, num = n1))
            n2 = 1 # 11
            Sp4_vec = [1] # np.float_power(10, np.linspace(-5, 5, num = n2))
            n3 = 1 # 11
            tau_b_vec = [1] # np.float_power(10, np.linspace(-5, 5, num = n3))
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

            # Start parallel computation
            pool = mp.Pool(mp.cpu_count()) # Use all cpu cores

            for i1 in range(m1):
                A = A_vec[i1]
                for i2 in range(m2):
                    w0 = w0_vec[i2]
                    for i3 in range(m3):
                        psi = psi_vec[i3]

                        ### Integration and time
                        method = 'BDF' # 'BDF'
                        dT = 2*np.pi/w0 * (1/10)
                        T_max = 2*np.pi/w0 * (10)
                        T_span = [0, T_max]
                        T_eval = [dT*i for i in range(round(T_max/dT))]
                        T_sim_max = 1*3600

                        ### Numerical Flow field and Interpolation
                        X_flow_field_string, X_flow_field = CreateFlowField(A, w0, psi, T_eval, filename = Flow_field_filename)
                        if X_flow_field_string != "NO FLOW":
                            InterpFlow = interpolate.interp1d(np.array(T_eval).reshape(len(T_eval),), X_flow_field, axis=1, fill_value="extrapolate")
                        else:         
                            InterpFlow = 0
                        
                        print("Flow field created for (A,w0,psi) = ", A, w0, psi)
                        flow_params = dict(A = A, w0 = w0, psi = psi)

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

                                            inloop_args = [flow_params, exp_params, guess_variable_params, bounds, disc_func, opt_scheme, opt_args, writing_path]                                        
                                            res = pool.apply_async(func = Viscoelastic_inference_inloop, args = inloop_args)

            pool.close()
            pool.join() # postpones the execution of next line of code until all processes in the queue are done.