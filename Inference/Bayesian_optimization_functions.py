""" This file compiles all the functions to perform a bayesian-BFGS (change name) 
optimization of an inverse problem on a differential equation."""

import numpy as np
from scipy import special
from skfdiff import Model, Simulation
import scipy.integrate as integrate
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

def line_search(Omega, phi_star, C_primal, J_previous, J, Solve_Primal_Dual, s, p, p_bar, C_p, method = 'simple'):

    max_it = 10
    tau = 2

    if method == 'simple':
        print("J_previous =", J_previous)
        J_new = J_previous
        it = 0
        while J_new-J_previous>=0:
            if it >= max_it:
                print("max iteration number reached! ")
                return 0
            it+=1
            tau/=2
            p_new = p + tau * s
            print("p_new = ", p_new)

            primal_dual_new = Solve_Primal_Dual(Omega, phi_star, C_primal, p_new, skip = 1)
            J_new = J(Omega, phi_star, primal_dual_new[0], C_primal, p_new, p_bar, C_p)[0]
            print("J_new = ", J_new)
            
        return tau
    
    elif method == 'Wolfe':

    # def line_search(f,x,p,nabla):
    #     '''
    #     BACKTRACK LINE SEARCH WITH WOLFE CONDITIONS
    #     '''
    #     tau = 1
    #     c1 = 1e-4 
    #     c2 = 0.9 
    #     fx = f(x)
    #     x_new = x + a * p 
    #     nabla_new = grad(f,x_new)
    #     while f(x_new) >= fx + (c1*a*nabla.T@p) or nabla_new.T@p <= c2*nabla.T@p : 
    #         a *= 0.5
    #         x_new = x + a * p 
    #         nabla_new = grad(f,x_new)
    #     return a
        print("Woof!")
        return tau
    
    else:
        print("Miaou!")
        return None

def Exact_inverse_Hessian(Omega, Solve_Primal_Dual, phi_star, C_primal, nabla_previous, p_previous, s, p_bar, C_p, grad_J, tau = 2**(-10)):
    """ Computes the exact inverse Hessian for an exact Newton descent. Can be computationally costly! ç
    - tau is the step size such that p_previous + tau*s is the parameter at which the new gradient is computed. 
    Warning! Only works in 1D for now. """
    
    # Compute nabla_new, infinitesimal variation from nabla_previous in parameter space
    s_new = tau * s
    p_new = p_previous + s_new
    primal_dual_new = Solve_Primal_Dual(Omega, phi_star, C_primal, p_new, skip = -1)
    nabla_new = grad_J(Omega, primal_dual_new, p_previous + s_new, p_bar, C_p)[0]
    
    # Approximate Hessian inverse as an infinitesimal inverse rate of variation
    y = nabla_new - nabla_previous
    rr = 1 / (y.T @ s_new)
    H_exact = rr * (s_new@(s_new.T))

    return H_exact

def Quasi_inverse_Hessian(nabla_diff, s, H_previous, method = 'BFGS'):
    P = H_previous.shape[0]
    y = nabla_diff

    if method == 'BFGS':
        
        y = np.array([y])
        s = np.array([s])
        y = np.reshape(y,(P,1))
        s = np.reshape(s,(P,1))
        rr = 1/(y.T@s)
        li = (np.eye(P)-(rr*((s@(y.T)))))
        ri = (np.eye(P)-(rr*((y@(s.T)))))
        hess_inter = li@H_previous[-1]@ri
        H = hess_inter + (rr*((s@(s.T)))) # BFGS appeox
        return H
    
    else:
        print("Blblbl!")
        return H_previous

def BFGS_adjoint_optimization(Omega, phi_star, C_primal, Solve_Primal_Dual, J, Grad_J, p_0, p_bar, C_p, max_it = 100, eps_tol = 1e0):

    """ 
    # DESCRIPTION #

    Minimize data-model discrepancy [J] in parameter space of a model using a Quasi-Newton (BFGS) gradient descent
    - At every iteration in parameter space the solution [phi] of the model is computed,
    as well as the adjoint solution [psi] that respects the model space constraint.
    - Knowledge of the primal [phi] and dual [psi] solutions at one point in parameter space allows
    to compute the gradient in that space
    - Compute the new direction with the gradient and
        - a (BFGS) approximation of the inverse Hessian
        - with a line search (Simple or Wolfe)

    # INPUT #

    - Omega: space of application of functions phi and psi
    - phi_star: data (seen as a solution of the primal problem)
    - C_primal: Covariance operator quantifying uncertainty in primal solution space
    - Solve_Primal_Dual: specific function that solves simultaneously the predefined primal-dual problem
    - p_0: initial parameters
    - C_p: Covariance operator quantifying uncertainty in parameter space
    - max_it: maximum iterations allowed for the descent to converge
    - eps_tol: maximum error allowed before the descent has converged

    # OUTPUT #

    - ...

    """

    # Initialization
    d = p_0.shape[0]
    k = 0
    p_list = [p_0]
    print("p = ", p_list[-1])
    primal_dual_list = [Solve_Primal_Dual(Omega, phi_star, C_primal, p_list[-1])]
    J_list = [J(Omega, phi_star, primal_dual_list[-1][0], C_primal, p_list[-1], p_bar, C_p)]
    nabla_J_list = [Grad_J(Omega, primal_dual_list[-1], p_list[-1], p_bar, C_p)]
    H_list = [np.eye(d)] # [C_p]

    s_list = []
    tau_list = []

    try:
    # if True:
        print("Starting the while loop")
        while k < max_it and (J_list[-1][0] > eps_tol or (k==0 or np.abs(nabla_J_list[-1][0]) > np.abs(nabla_J_list[-2][0]))):

            # Find descent direction
            s_list.append(-H_list[-1] @ nabla_J_list[-1][0])
            print("Starting line search...")
            tau_list.append(line_search(Omega, phi_star, C_primal, J_list[-1][0], J, Solve_Primal_Dual, s_list[-1], p_list[-1], p_bar, C_p))
            if tau_list[-1] == 0:
                print("Line search failed.")
                raise ValueError
            print("Line search ended. ")
            s_list[-1] *= tau_list[-1]

            # Update
            p_list.append(p_list[-1] + s_list[-1])
            print("p = ", p_list[-1])
            primal_dual_list.append(Solve_Primal_Dual(Omega, phi_star, C_primal, p_list[-1], skip = -1))
            J_list.append(J(Omega, phi_star, primal_dual_list[-1][0], C_primal, p_list[-1], p_bar, C_p))
            nabla_J_list.append(Grad_J(Omega, primal_dual_list[-1], p_list[-1], p_bar, C_p))
            H_list.append(Quasi_inverse_Hessian(nabla_J_list[-1][0] - nabla_J_list[-2][0], s_list[-1], H_list[-1], method = 'BFGS')) # Comment to get simple gradient descent

            k += 1

        print("Algorithm is finished.")
        return p_list, primal_dual_list, J_list, nabla_J_list, H_list
    except:
        print("Something went wrong!")
        return p_list, primal_dual_list, J_list, nabla_J_list, H_list

# Uncertainty functions

def PCA(C):
        """ Perform a decomposition in orthonormal basis of a symmetric real matrix C.
        C can be interpreted as a covariance. """

        # Perform SVD
        u, L, vh = np.linalg.svd(C, full_matrices = True, hermitian = True)
        P = np.transpose(vh) # Matrix whose columns are principal components

        return P, L

def PCA_posterior_sampling(p, C):

    """ Samples posterior distribution of parameters p given a covariance matrix C.
    An orthogonal decomposition is performed and allows independent gaussian samplings. """

    n = p.shape[0]
    
    P, L = PCA(C) # Orthogonal decomposition
    l = np.transpose(np.sqrt(L))
    
    eta = np.transpose(np.random.normal(size=n))
    return p + P @ (l * eta)

