
"""

This file is to build and test parameter inference on simple models using
its analytically-derived weak form for regularizing the problem and
solving in parameter space using BFGS (quasi-Newton) to do a gradient
descent that requires knowledge of the analytical gradient, a BFGS
approximation of the Hessian. The analytical gradient in parameter space
is derived from the adjoint problem of the functional minimization.

    - Model 1: Euler-Bernoulli (pure bending) at equilibrium with charge q(s)
    is corresponds to the Coarse-grained model for beta = 0, tau_b = 0 and
    a flow field corresponding to q(s). This flow field should be taken so
    that we have an analytical solution, to check the solution is right.

"""
import numpy as np
from Coarse_grained_axoneme_functions import *
from Coarse_grained_analysis_functions import *

###############################
########## FUNCTIONS ##########

def solve_primal(Sp4, params):
    '''
    SOLVES THE PRIMAL PROBLEM FOR A SET OF INPUT PARAMETERS:
        dT, T_max, N, gamma
        Sp4, Beta, taus_b
        n_L, m_L, Lamdas, Zetas
    
    RETURNS X2N: 
        the filament final position in a (N,2) shape <-- IN Sp4 UNITS
    '''

    dT, T_max, N, Beta, taus_b, gamma, n_L, m_L, Lambdas, Zetas = params

    T_span= [0, T_max]
    T_eval = [dT*i for i in range(int(T_max/dT)+1)]
    X_0 = StraightLine(N)

    # External parameters
    X_flow_field_string = "NO FLOW"
    X_flow_field = 0
    A = 0
    w0 = 0

    sol = Solve(f, taus_b, Beta, gamma, n_L, m_L, A, w0, Sp4, Lambdas, Zetas, X_flow_field_string, T_span, T_eval, X_flow_field, X_0)

    # Args = (Sp4, Beta, taus_b, gamma, n_L, m_L, Lambdas, Zetas)
    # sol = solve_ivp(fun = f, t_span = T_span, y0 = X_0, args=Args, t_eval=T_eval, method = 'LSODA').y
    eq_sol = sol[:,-1]

    X_3N = X3N(eq_sol) * Sp4

    # Change dimensions of X: (3N * 1) --> N * 2
    X2N = np.zeros((N, 2))
    X2N[:,0] = X_3N[:N,0]
    X2N[:,1] = X_3N[N:2*N,0]
    return X2N

def J(X2N_star, X2N, Sp4, l):
    '''
    FUNCTION TO BE OPTIMISED w.r.t. Sp4
    '''
    dist = np.sum(np.linalg.norm(X2N_star-X2N, axis = 1)**2) / 2
    J = dist - np.sum(np.multiply(X2N_star-X2N, X2N)) - l*Sp4
    return J

def grad(Sp4, l, X2N_star, X2N): 
    '''
    ANALYTICAL GRADIENT CALCULATION
    '''
    d4Y2N = Sp4 * (X2N_star-X2N)
    nabla = np.sum(np.multiply(d4Y2N, X2N)) - l

    return nabla 

def line_search(J, Sp4, l, params, p, nabla):
    '''
    BACKTRACK LINE SEARCH WITH WOLFE CONDITIONS
    '''
    a = 1
    c1 = 1e-4 
    c2 = 0.9 
    X2N = solve_primal(Sp4, params)
    JSp4 = J(X2N_star, X2N, Sp4, l)

    Sp4_new = np.max((1e-6, Sp4 + a * p))
    print("Sp4_new = ", Sp4_new)
    X2N_new = solve_primal(Sp4_new, params)
    nabla_new = grad(Sp4_new, l, X2N_star, X2N_new)
    while J(X2N_star, X2N_new, Sp4, l) >= JSp4 + (c1 * a * nabla * p) or nabla_new * p <= c2 * nabla * p : 
        a *= 0.5
        Sp4_new = np.max((1e-6, Sp4 + a * p))
        print("Sp4_new = ", Sp4_new)
        X2N_new = solve_primal(Sp4_new, params)
        nabla_new = grad(Sp4_new, l, X2N_star, X2N_new)
    return a

def BFGS(X2N_star, Sp4_0, l, params, max_it):
    '''
    DESCRIPTION
    BFGS Quasi-Newton Method, implemented as described in Nocedal:
    Numerical Optimisation.


    INPUTS:
    J:      function to be optimised 
    Sp4_0:     intial guess
    max_it: maximum iterations
    l: regularization term for parameters (should be positive to impose positivity of parameters)

    OUTPUTS: 
    - if the algorithm converges:
        Sp4:      the optimal solution of the function J
    - else:
        Sp4_list, grad_Sp4_list, H_list, J_list, tau

    '''
    P = 1
    tau = 1
    s = 1
    print("Sp4 = ", Sp4_0)

    # Solve primal problem to get X2N(Sp4)
    X2N = solve_primal(Sp4_0, params = params)
    nabla = grad(Sp4_0, l, X2N_star, X2N) # initial gradient 
    print("nabla = ", nabla)
    H = np.eye(P) # initial hessian
    print("H = ", H[0][0])

    Sp4 = Sp4_0
    it = 2 

    Sp4_list = [Sp4]
    grad_Sp4_list = [nabla]
    H_list = [H[0][0]]
    J_list = [J(X2N_star = X2N_star, X2N = X2N, Sp4 = Sp4, l = l)]
    print("J = ", J_list[-1])
    try:
        while np.abs(J_list[-1])>1e-10 and np.abs(s)>1e-10: # while gradient is negative
            
            
            if it > max_it: 
                print('Maximum iterations reached!')
                break
            it += 1
            if P == 1:
                p = -H * nabla
            else:
                p = -H@nabla # search direction (Newton Method)
            tau = tau/2
            # print("Ongoing line search...")
            # tau = line_search(J, Sp4, l, params, p, nabla) # line search
            # print("Line search finished.")
            s = tau * p
            print("p, tau, s = ", p, tau, s)
            Sp4_new = np.max((1e-6, Sp4 + s[0][0]))
            print("Sp4 = ", Sp4_new)
            Sp4_list.append(Sp4_new)

            # Solve primal problem again (could linearize it)
            X2N_new = solve_primal(Sp4_new, params = params)
            J_list.append(J(X2N_star = X2N_star, X2N = X2N_new, Sp4 = Sp4, l = l))
            print("J = ", J_list[-1])

            # Compute new gradient
            nabla_new = grad(Sp4_new, l, X2N_star, X2N_new)
            print("nabla = ", nabla)
            grad_Sp4_list.append(nabla_new)

            #BFGS Hessian approximation
            y = nabla_new - nabla 

            if P == 1:
                r = 1/(y*s)
                li = (1-r*(s*y))
                ri = (1-r*(y*s)) 
                hess_inter = li * ri           
                H = hess_inter + (r*(s*s)) # BFGS Update

            else:    
                y = np.array([y])
                s = np.array([s])
                y = np.reshape(y,(P,1))
                s = np.reshape(s,(P,1))
                r = 1/(y.T@s)
                li = (np.eye(P)-(r*((s@(y.T)))))
                ri = (np.eye(P)-(r*((y@(s.T)))))
                hess_inter = li@H@ri
                H = hess_inter + (r*((s@(s.T)))) # BFGS Update

            print("H = ", H[0][0])
            H_list.append(H[0][0])

            if P == 1:
                nabla = nabla_new
                Sp4 = Sp4_new
            else:
                nabla = nabla_new[:]
                Sp4 = Sp4_new[:]

    except IndexError:
        print("Algorithm did not find primal solution? ")
        return [Sp4_list, grad_Sp4_list, H_list, J_list, tau]
    
    return [Sp4_list, grad_Sp4_list, H_list, J_list, tau]

########## FUNCTIONS ##########
###############################

if __name__ == "__main__":

    ##### Step 1: import the data
    print("Step 1: import data...")
    folder_name = "C:/Users/Luc/Documents/PhD_Large_files/RheoFlag/Model/Output/renormalized/pure_bending/static_responses/uniform_vertical_force/"
    filename = "data_20230825-031229082307.dat"
    parameters, X = ExtractParametersData(folder_name + filename)

    N, taus_b, init_conf, Beta, gamma, n_L, m_L, A, w0, Sp4, Lambdas, Zetas, X_flow_field_type, X_flow_field_params, T_span, T_eval = parameters
    X_star =  X[:, -1] # Equilibrium shape
    Sp4_star = Sp4
    print("Sp4_data = ", Sp4_star)

    ##### Step 2: initialize parameters
    print("Step 2: initialize parameters...")
    Sp4_0 = 1e0

    # Regularization
    l = 0 # Sp4 > 0 is enforced more strongly the bigger l is (l>0).

    grad_Sp4_list = []
    T_eval = np.array(T_eval)
    X_flow = A*np.sin(w0*T_eval)

    # Plot shape
    X3N_star = X3N(X_star) * Sp4_star
    # Change dimensions of X: (3N * 1) --> N * 2
    X2N_star = np.zeros((N, 2))
    X2N_star[:,0] = X3N_star[:N,0]
    X2N_star[:,1] = X3N_star[N:2*N,0]

    # print("plot equilibrium shape...")
    # fig = go.Figure()
    # fig.add_scatter(x = X3N_star[:N,0], y = X3N_star[N:2*N,0], mode = "markers")
    # fig.show()

    ##### Step 0: some tests
    # print("X2N.shape = ", X2N_star.shape)
    # a = J(X2N_star,X2N=X2N_star, Sp4 = Sp4, l = l)
    # print("a = ", a)
    # exit()

    ##### Step 3: BFGS optimization with primal and dual update
    dT = 1e0
    T_max = 1e3

    params = (dT, T_max, N, Beta, taus_b, gamma, n_L, m_L, Lambdas, Zetas)

    Sp4_guess = BFGS(X2N_star=X2N_star, Sp4_0 = Sp4_0, l = l, params = params, max_it = 100)

    print("Sp4_guess = ", Sp4_guess)
    print("[Sp4_list, grad_Sp4_list, H_list, J_list, tau]")