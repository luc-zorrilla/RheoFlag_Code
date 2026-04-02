import numpy as np
import scipy.differentiate as sd
import dill as pickle

def write_array_to_csv(array, filename):
    """
    Write a NumPy array of any shape to a CSV file.
    
    The array is flattened before saving, and its shape is stored as metadata.
    
    Parameters:
    - array: NumPy array of any shape
    - filename: the name of the CSV file (without extension)
    """
    array = np.array(array)

    # Flatten the array and save to CSV
    np.savetxt(f'{filename}.csv', array.flatten(), delimiter=',')
    
    # Save the shape information in a separate file
    with open(f'{filename}_shape.txt', 'w') as shape_file:
        shape_file.write(','.join(map(str, array.shape)))

def read_array_from_csv(filename):
    """
    Read a NumPy array of any shape from a CSV file.
    
    The array is reshaped back to its original shape using metadata.
    
    Parameters:
    - filename: the name of the CSV file (without extension)
    
    Returns:
    - NumPy array reshaped to its original dimensions
    """
    # Load the shape information
    with open(f'{filename}_shape.txt', 'r') as shape_file:
        line = shape_file.read()
        if len(line) == 0:
            shape = 1
        else:
            shape = tuple(map(int, line.split(',')))
    
    # Read the flattened array from CSV
    flat_array = np.loadtxt(f'{filename}.csv', delimiter=',')
    
    # Reshape it back to its original shape
    return flat_array.reshape(shape)

def Make_Dict_From_Applied_Function(func, func_args, func_output):
    """
    This function makes a dictionary that keeps track of the function used, with
    its arguments and its ouput.
    
    Input:
        - func: the applied function (function)
        - func_args: the input arguments used for the function (dict)
        Note: If any func_args dictionary item is a function, keeps only its name (string).
        - func_output: the output of the function (unknown type)
    
    Update: save functions directly thanks to dill. Not used yet.
    """
    applied_func_dict = {}
    applied_func_dict["function"] = func #.__name__

    # for arg_key in list(func_args.keys()):
    #     arg = func_args[arg_key]
    #     if callable(arg): # Check if it's a function (or behaves like one)
    #         func_args[arg_key] = arg.__name__
    applied_func_dict["args"] = func_args

    applied_func_dict["output"] = func_output

    return applied_func_dict


def weighted_average(x, w):
    """ Performs the weighted average of x with weight vector w. Returns a float number """
    
    wa = np.sum(np.multiply(x,w))
    return wa


def linear_combination_estimators(vars):
    """ Computes the best linear combination of independant estimators of resp. variance vars. 
    Returns weights of the linear combination and total variance. """

    if vars.shape[0] == 1: # If there is only one measurement, there is nothing to do.
        if np.isnan(vars[0]) or vars[0]<0:
            w = np.array([0])
            tot_var = np.inf
        else:
            w = np.array([1])
            tot_var = vars[0]
        return w, tot_var
    
    # vars.shape[0] > 1
    if np.any(vars < 0):
        if np.all(vars < 0):
            return np.zeros(vars.shape), np.inf
        else:
            for k in range(vars.shape[0]):
                if vars[k] < 0:
                    vars[k] = np.inf

    elif not np.all(np.isfinite(vars)): # None or np.inf
        for k in range(vars.shape[0]):
            if not np.isfinite(vars[k]):
                vars[k] = np.inf

    elif np.any(vars == 0):
        if np.all(vars == 0):
            return np.ones(vars.shape) / vars.shape[0], 0
        else:
            w = np.zeros(vars.shape)
            for k in range(vars.shape[0]):
                if vars[k] == 0:
                    w[k] = 1
            w /= np.count_nonzero(w)
            return w, 0
    
    vars_finite = vars[np.where(np.isfinite(vars))]

    if vars_finite.shape[0] == 0:
        return np.zeros(vars.shape), np.inf

    else:
        inv_sigma = np.linalg.inv(np.diag(vars_finite))
        w_finite = inv_sigma @ np.ones(vars_finite.shape)
        tot_var = 1./ (np.transpose(np.ones(vars_finite.shape)) @ w_finite)
        w_finite = w_finite * tot_var

        w = np.zeros(vars.shape[0])
        w[np.where(np.isfinite(vars))] = w_finite

        return w, tot_var

def BLC(Z_vector_list):
    """ Assuming independence between estimators, computes the best linear combination of these estimators and return both the estimator and the variance
    Input: Z_vector_list is a list of estimators with their associated standard deviation (Z, std_Z) 
    Output: Z_combination_vector is the best linear estimate with its standard deviation: (Z_combination, std_Z_combination)
    """

    if len(Z_vector_list) == 0:
        Z_combination_vector = np.array([np.inf, np.inf])
    
    elif len(Z_vector_list) == 1:
        Z_combination_vector = Z_vector_list[0]
    
    else:
        if Z_vector_list[0].shape[0] < 2:
            raise ValueError('Z_vector_list[0].shape[0] < 2. Cannot use BLC without estimates of uncertainty. ')

        Z_vector = []
        var_vector = []
        for k in range(len(Z_vector_list)):
            
            if not (isinstance(Z_vector_list[k], list) and len(Z_vector_list[k]) == 0):
                if  np.isfinite(Z_vector_list[k][1]) and ~np.isnan(Z_vector_list[k][1]):
                    Z_vector.append(Z_vector_list[k][0])
                    var_vector.append(Z_vector_list[k][1] ** 2)
        Z_vector = np.array(Z_vector)
        var_vector = np.array(var_vector)

        if Z_vector.shape[0] == 0:
            Z_combination_vector = np.array([np.inf, np.inf])

        elif Z_vector.shape[0] == 1:
            Z_combination_vector = np.array([Z_vector[0], np.sqrt(var_vector[0])])

        else:
            w, tot_var = linear_combination_estimators(var_vector) # ¡Assumes independence of estimators in the same list!
            Z_combination = weighted_average(x = Z_vector, w = w)
            std_Z_combination = np.sqrt(tot_var) # Standard way of computing standard deviation of that estimator
            Z_combination_vector = np.array([Z_combination, std_Z_combination])
    
    return Z_combination_vector


def inv_mat(M):
    """ Invert matrix or set to infinity if not invertible. """
    try:
        Mm1 = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        Mm1 = np.ones_like(M) * np.inf
    return Mm1


def Vectorize_Functional(func, m):
    """ 
    This function vectorizes a functional with m input parameters, 
    by wrapping it inside another function.
    """

    def f_vec(x):

        x = np.array(x, copy=False)
        if x.ndim < 1 or x.shape[0] != m:
            raise ValueError(f"Expected first dim {m}, got {x.shape}")

        # Flatten extra dims
        extra_shape = x.shape[1:]
        p = int(np.prod(extra_shape, dtype=int)) if extra_shape else 1
        x_flat = x.reshape(m, p)

        # apply func to each column
        out = np.empty(p, dtype=float)
        for j in range(p):
            out[j] = func(x_flat[:, j])
        # reshape back to extra_shape
        return out.reshape(extra_shape)

    return f_vec


def custom_average(x, type = "mean"):
        """ 
        Performs a custom average of x.
        Inputs: 
            - x is a list of ndarrays of shape (2,), i.e., a list of (p, sigma_p)
            - type: a string corresponding to one element of L, 
                corresponding to the custom average type, where L = ["mean", "median", "combined"]
        Outputs:
            - avg_x: ndarray of shape (2,) where each element is an array of shape (nvars,). avg_x corresponds to (avg_p, sigma_avg_p)
        """
        p = np.array([x[k][0] for k in range(len(x))]).reshape((len(x), -1))
        sigma_p = np.array([x[k][1] for k in range(len(x))]).reshape((len(x), -1))

        n_samples = len(p)
        n_vars = p.shape[1] # Check shape of p and sigma_p
        avg_p = np.zeros((n_vars,))
        sigma_avg_p = np.zeros((n_vars,))

        for j in range(n_vars):
            Z_vector_list_j = np.array([np.array([p[k,j], sigma_p[k,j]]) for k in range(p.shape[0])]) # Error is measured from the hessian
            Z = Z_vector_list_j.reshape((-1,2))
            Z = Z[(Z[:,0] < np.inf) & (Z[:,1] < np.inf)]
            
            if type == 'mean':
                avg_p[j] = np.mean(Z, axis = 0)[0]
                sigma_avg_p[j] = (np.std(np.array(Z), axis = 0, ddof = 1) / np.sqrt(len(Z)))[0]
            elif type == 'median':
                avg_p[j] = np.median(Z, axis = 0)[0]
                sigma_avg_p[j] = np.nan # No std. for the median.
            elif type == "combined":
                avg_p[j], sigma_avg_p[j] = np.array(BLC(Z))
            else:
                raise ValueError("String does not correspond to available types of averages.")
        
        avg_x = np.array([avg_p, sigma_avg_p]).reshape(2,)
        return avg_x

def second_pass(): # This is a residual saving to to pickling scripts... no idea what happened but I need to leave this function here.
    return