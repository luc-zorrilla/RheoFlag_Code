
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