import numpy as np


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
    """
    applied_func_dict = {}
    applied_func_dict["function"] = func.__name__

    for arg_key in list(func_args.keys()):
        arg = func_args[arg_key]
        if callable(arg): # Check if it's a function (or behaves like one)
            func_args[arg_key] = arg.__name__
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

            if np.isfinite(tot_var):
                M = w.shape[0]
                # Standard way of computing standard deviation of that estimator
                std_Z_combination = np.sqrt(tot_var)
                # Experimental way of computing standard deviation of that estimator
                # std_Z_combination = np.sqrt(np.sum(np.multiply(w, (Z_vector - Z_combination)**2))) / np.sqrt(M)
                # Another way (potentially biased) is the following. Notice the differences with the previous formula
                # std_Z_combination = np.sqrt(np.sum(np.multiply(w**2, (Z_vector - Z_combination)**2)))
            else:
                std_Z_combination = np.sqrt(tot_var)

            Z_combination_vector = np.array([Z_combination, std_Z_combination])
    
    return Z_combination_vector
