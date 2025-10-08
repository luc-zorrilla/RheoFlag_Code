""" Optimization schemes from simple to complex, applied on minimal working examples. """

### Libraries
import numpy as np
import scipy.optimize as so

### Functions

def make_tuples(depth, n_array, start_array):
    """ This function creates a nested loop of a certain depth. 
        - Inputs: 
            - depth: integer
            - n_array: numpy array of integers, corresponding to the range of each loop
            - start: where each loop starts
    """
    if depth == 0:
        yield ()
    else:
        n = n_array[0]
        start = start_array[0]
        for x in range(start, n):
            n_array_new = n_array[1:]
            start_array_new = start_array[1:]
            for t in make_tuples(depth - 1, n_array_new, start_array_new):
                yield (x,) + t

# Test: OK
# depth = 2
# N = 5
# start = 0
# n_array = np.array([N, N])
# start_array =  np.array([start, start])
# for (i1, i2) in make_tuples(depth, n_array, start_array):
#     print("i1, i2 = ", i1, i2)

def gaussian(x, mu, sigma):
    """ Defines a functional that returns a non-normalized gaussian.
    - Input: 
        - x: position to be evaluated (scalar), 
        - mu: center of the gaussian (scalar), 
        - sigma: s.d. of the gaussian (scalar).
    - Output: numpy array f of shape (N,)"""

    f = np.exp(-(mu-x)**2/sigma**2)
    return f

# # Test: OK - for a scalar and a vector
# mu = 0
# sigma = 1
# x_min = -3
# x_max = 3
# N = 1000
# fig = go.Figure()
# # Point
# x = 0
# f = gaussian(x, mu, sigma)
# fig.add_scatter(x = [x], y = [f], marker_color = "red", name = "scalar")
# # Vector
# x = np.linspace(x_min, x_max, N)
# f = gaussian(x, mu, sigma)
# fig.add_scatter(x = x, y = f, marker_color = "black", name = "vector")
# fig.show()

def multigaussian(x, mu, sigma):
    """ Defines a function that returns a non-normalized multidimensional gaussian.
    - Input: 
        - x: numpy array of shape (D,)
        - mu: center of the gaussian, numpy array of shape (D,)
        - sigma: covariance matrix of the gaussian, numpy array of shape (D,D)
    - Output: numpy array of shape (N0, N1, ..., N(D-1))
    """
    
    x_c = x - mu
    f = np.exp(- (1/2) * np.transpose(x_c) @ np.linalg.inv(sigma) @ (x_c))
    return f

# # Test: OK for a point, OK for a vector
# D = 2
# mu = np.ones((D,)) * 0
# sigma = np.diag(np.ones(D,) * 1)
# x_min = -3
# x_max = 3
# N = 1000

# # Point
# x = np.ones((D,)) * 0
# f = multigaussian(x, mu, sigma)
# data_point = go.Heatmap(
#         x = [x[0]], 
#         y = [x[1]],
#         z = [f],
#         coloraxis = "coloraxis")
        
# # Point 2
# x = np.ones((D,)) * 1
# f = multigaussian(x, mu, sigma)
# data_point2 = go.Heatmap(
#         x = [x[0]], 
#         y = [x[1]],
#         z = [f],
#         coloraxis = "coloraxis")

# # vector
# x_one_dim = np.reshape(np.linspace(x_min, x_max, N), (1,N))
# x_grid = np.repeat(x_one_dim, D, axis=0) # shape = (D, N)
# f_grid = np.zeros([N for d in range(D)])

# depth = D
# start = 0
# n_array = (np.ones((D,)) * N).astype(int)
# start_array =  (np.ones((D,)) * start).astype(int)
# for tpl in make_tuples(depth, n_array, start_array):
#     x_point = np.array([x_grid[k,tpl[k]] for k in range(len(tpl))])
#     f_point = multigaussian(x_point, mu, sigma)
#     f_grid[tpl] = f_point

# data_vector = go.Heatmap(
#         x = x_grid[0,:], 
#         y = x_grid[1,:],
#         z = f_grid,
#         coloraxis = "coloraxis")

# fig = go.Figure(data = [data_point, data_point2, data_vector])
# fig.show()

def grid_search(f, x_grid):
    """ Computes a functional over a grid x and get the minimum. 
    Input: 
        - f is the functional, 
        - x_grid is the grid, of shape (D, N) where 
            - D is the dimension of the functional input and 
            - N is the number of grid points along each dimension
    Output: 
        - f_grid is a np array with values of f over x_grid,
    """

    depth = x_grid.shape[0]
    N = x_grid.shape[1]
    start = 0
    n_array = (np.ones((x_grid.shape[0],)) * N).astype(int)
    start_array = (np.ones((x_grid.shape[0],)) * start).astype(int)

    f_grid = np.zeros([N for d in range(depth)])

    for tpl in make_tuples(depth, n_array, start_array):
        x_point = np.array([x_grid[k,tpl[k]] for k in range(len(tpl))])
        f_point = f(x_point)
        f_grid[tpl] = f_point

    return f_grid

# Test: OK for 1D, OK for 2D
# N = 1000
# mu = 0
# sigma = 1
# x_min = -3
# x_max = 3
# def f(x):
#     return gaussian(x, mu, sigma)
# x_grid = np.linspace(x_min,x_max,N).reshape((1,N))
# f_grid = grid_search(f, x_grid)
# print(f_grid)
# exit()

# D = 2
# mu = np.ones((D,)) * 0
# sigma = np.diag(np.ones(D,) * 1)
# x_min = -3
# x_max = 3
# N = 1000
# def f(x): 
#     return multigaussian(x, mu, sigma)
# x_one_dim = np.reshape(np.linspace(x_min, x_max, N), (1,N))
# x_grid = np.repeat(x_one_dim, D, axis=0) # shape = (D, N)
# f_grid = grid_search(f, x_grid)

# data_vector = go.Heatmap(
#         x = x_grid[0,:], 
#         y = x_grid[1,:],
#         z = f_grid,
#         coloraxis = "coloraxis")

# fig = go.Figure(data = [data_vector])
# fig.show()

### Main

if __name__ == "__main__":

    ## Choose a functional and its parameter space.

    D = 2

    if D == 1:
        # 1D, gaussian
        N = 1000
        mu = 0
        sigma = 1
        x_min = -3
        x_max = 3
        def f(x):
            return gaussian(x, mu, sigma)
        x_grid = np.linspace(x_min,x_max,N).reshape((1,N))


    elif D == 2:
        # 2D, multivariate gaussian
        D = 2
        mu = np.ones((D,)) * 0
        sigma = np.diag(np.ones(D,) * 1)
        x_min = -3
        x_max = 3
        N = 1000
        def f(x): 
            return multigaussian(x, mu, sigma)
        x_one_dim = np.reshape(np.linspace(x_min, x_max, N), (1,N))
        x_grid = np.repeat(x_one_dim, D, axis=0) # shape = (D, N)

    ## Grid search 
    # f_grid = grid_search(f, x_grid)
    # argmin = np.unravel_index(np.argmin(-f_grid), f_grid.shape)
    # x_argmin = np.array([x_grid[d,argmin[d]] for d in range(D)])
    # print(x_argmin)

    ## Scipy - BFGS
    x0 = np.array([0.1]) #, -0.1])
    def g(x):
        return -f(x)
    res = so.minimize(g, x0, method='BFGS')
    print("res.x", res.x)
    print("res.success", res.success)
    print("res.status", res.status)
    print("res.message", res.message)
    print("res.fun, res.jac, res.hess_inv", res.fun, res.jac, res.hess_inv)
    print("res.nfev, res.njev, res.nit", res.nfev, res.njev, res.nit)
    

        