from Bayesian_optimization_functions import *

def Rho_uniform(Omega, p):
    """
    Returns a uniform density of charge on Omega, under the following parameters:
        - p[0] is the absolute total charge in Omega
        - p[1] is the sign of the charge
    """
    d = p.shape[0]
    if d == 1:
        Q = p[0]
        V = 4 / 3 * np.pi * (Omega[-1]**3 - Omega[0]**3)

    elif d == 2:
        Q = p[0]
        V = p[1]
        
    else:
        print("Dimension d of parameter space is wrong. ")
        raise ValueError
    
    rho = np.ones_like(Omega) * Q / V
    return rho

def analytical_potential(r, Q, sigma = 0):
    """ Returns the potential at given values of r,
        solution of the Poisson equation for either:
         - a gaussian charge of total charge Q and spread sigma, setting phi = 0 at infinity. 
         - a uniform density of charge in a spherical shell [a, A], setting phi = phi_A = 0 at r=A. """
    
    if sigma == 0: # uniform density of charge
        a = r[0]
        A = r[-1]
        rho = Q / (4/3*np.pi * (A**3 - a**3))
        phi = rho/3 * ( (A**2 - r**2) / 2 + a**3 * (1 / A - 1 / r) )

    else: # gaussian density of charge

        phi = Q / (4 * np.pi) * (1 / r) * special.erf(r / (sigma * np.sqrt(2)))
    return phi

def spherical_derivative(r_prime):
    """ Returns the surface of a sphere of radius r_prime,
    which corresponds to the volume element when the integrand
    only depends on the radius r. """
    return 4*np.pi*r_prime**2

def Solve_spherical_poisson(Omega, rho, dt = 50, t_max = 100):

    """ Make a model and solve it with the skdiff library
    Here the model is a spherical Poisson problem with:
        - 0 derivative on left boundary (Neumann)
        - 0 value on right boundary (Dirichlet)
        - a generic density of charge rho on Omega

    # INPUT #

    - Omega is the space of application of solutions of the model, and of rho
    - p is a set of parameters
    - rho is the density of charge in Omega
    - dt is the time step of the solver
    - t_max is the max time of the solver, i.e. the time at which the eq. solution is taken

    # OUTPUT #
    - phi_eq is the equilbrium (i.e., stationary) solution of the model

    """

    # Make model
    # print("Making model...")
    bc_spherical_poisson = {("phi", "r"): ("dr(phi)", "dirichlet"),}
    model_spherical_poisson = Model(["drr(phi) + 2 / r * dr(phi) + rho"],
            ["phi(r)"],
            parameters=["rho(r)"],
            boundary_conditions=bc_spherical_poisson)
    
    # Define and initialize fields with parameters p
    # print("Initializing fields...")
    phi_0 =  np.zeros_like(Omega)
    
    initial_fields_spherical = model_spherical_poisson.Fields(r=Omega, phi=phi_0,
                            rho=rho)
    
    # Create a simulation object, run and save
    # print("Making simulation object...")
    simulation_spherical = Simulation(model_spherical_poisson, initial_fields_spherical, dt=dt, tmax=t_max)
    container_spherical = simulation_spherical.attach_container()
    # print("Running simulation...")
    tmax, final_fields = simulation_spherical.run()
    phi_sol_spherical = container_spherical.data.phi
    # print("Done!")
    phi_eq_spherical = phi_sol_spherical[-1,:].to_numpy()
    
    return phi_eq_spherical

def Solve_Primal_Dual_spherical_Poisson(Omega, phi_star, C_phi, p, skip = -1, dt = [50,50], t_max = [100,100]):

    if skip in [-1, 0, 1]:

        if skip in [-1, 1]:
            
            rho_primal = Rho_uniform(Omega, p) # can be put as an input of the function
            print("Solving primal Poisson problem")
            phi = Solve_spherical_poisson(Omega = Omega, rho = rho_primal, dt = dt[0], t_max = t_max[0])
            if skip == 1: # skip dual
                psi = phi
        
        if skip in [-1, 0]:
            # dV = spherical_derivative(Omega)
            rho_dual = np.linalg.inv(C_phi) @ (phi - phi_star) / (np.transpose(phi_star) @ np.linalg.inv(C_phi) @ (phi_star)) # can be put as an input of the function
            print("Solving dual Poisson problem")
            psi = Solve_spherical_poisson(Omega = Omega, rho = rho_dual, dt = dt[1], t_max = t_max[1]) # by linearity of the Poisson equation, C_phi_m1 is put out of the equation to fasten computation
            if skip == 0: # skip primal
                phi = psi

    else:
        print("skibabop badop bop!")
        phi = None
        psi = None

    return [phi, psi]

def E_spherical_Poisson(Omega, phi, phi_star, C_phi):
    # dV = spherical_derivative(Omega)
    E_res = 1/2 * np.transpose(phi_star - phi) @ np.linalg.inv(C_phi) @ (phi_star - phi)
    E_res /= np.transpose(phi_star) @ np.linalg.inv(C_phi) @ (phi_star)
    return E_res

def R_spherical_Poisson(p, p_bar, C_p):
    R_res = np.transpose(p - p_bar) @ np.linalg.inv(C_p) @ (p - p_bar) / 2
    return R_res

def J_spherical_Poisson(Omega, phi_star, phi, C_phi, p, p_bar, C_p):

    """
    Computes the functional we wish to minimize. It consists of 
    - a data-model discrepancy term [E]
    - a regularization terms [R] with respect to parameter space
    - a regularization term [S] with respect to shape space (not done yet)

    # INPUT #

    - phi_star: the data
    - phi: the primal solution for a given p
    - C_phi: a covariance operator quantifying the error in primal solution space
    - p: parameters of the model
    - C_p: a covariance operator quantifying the error in parameter space

    # OUTPUT #

    - [J_res, ...]: the functional we wish to minimize computed at a given point in parameter space, 
    and potential other parameters such as E and R, in a list.

    """
    E_res = E_spherical_Poisson(Omega, phi, phi_star, C_phi)
    print("E = ", E_res)
    M_res = 0
    R_res = R_spherical_Poisson(p, p_bar, C_p)
    print("R = ", R_res)
    S_res = 0

    J_res = E_res + M_res + R_res + S_res
    return [J_res, E_res, R_res]

def Grad_J_spherical_Poisson(Omega, primal_dual, p, p_bar, C_p):
    """ Computes the parameter gradient from an analytical standpoint.
    # OUTPUT #

    - [nabla_J, ...]: the functional gradient we wish to minimize computed at a given point in parameter space, 
    and potential other parameters such as nabla_E and nabla_R, in a list.
    """

    d = p.shape[0]

    # Gradient in the data-model discrepancy
    nabla_E = 0

    # Gradient in the weak formulation
    rho_primal = Rho_uniform(Omega, p)
    dV = spherical_derivative(Omega)

    if d == 1:
        nabla_M = np.dot(rho_primal, primal_dual[1]*dV) * np.array([1 / p[0]])
    elif d == 2:
        nabla_M = np.dot(rho_primal, primal_dual[1]*dV) * np.transpose([1 / p[0], - 1 / p[1]])
    else:
        print("Dimension d of parameter space is wrong. ")
        raise ValueError

    # Gradient in the parameter priors
    nabla_R = np.linalg.inv(C_p) @ (p - p_bar)

    # Gradient in the signed distance field
    nabla_S = 0

    nabla_J = nabla_E + nabla_M + nabla_R + nabla_S
    # nabla_J = C_p @ nabla_J # Change scalar product in parameter space
    return [nabla_J, nabla_M, nabla_R]

# Make space Omega
a = 1e-2
A = 1e-1
N = 256
r = np.linspace(a, A, N)

# Make data phi_star
Q_star = 1e0
p_star = np.array([Q_star])
rho_star = Rho_uniform(Omega = r, p = p_star)
phi_star = Solve_spherical_poisson(Omega = r, rho = rho_star, dt = 50, t_max = 100)
phi_analytical = analytical_potential(r, Q_star, sigma = 0)

# Test solver performance
# fig = go.Figure()
# fig.add_scatter(x = r, y = rho_star, name = "Density of charge")
# fig.add_scatter(x = r, y = phi_star, name = "numerical solution")
# fig.add_scatter(x = r, y = phi_analytical, name = "analytical solution")
# fig.show()
# exit()

L = 1 # Number of samples
p_knot_list = np.zeros((L,1))
p_knot_mean_list = np.zeros((L,1))
p_knot_std_list = np.zeros((L,1))
np.random.seed(43)
for l in range(L):
    # Adding noise
    SNR = 100
    RMS = np.sqrt(np.mean(phi_star**2))
    sigma_phi_star = RMS / SNR
    print("For SNR = ", SNR, "sigma_phi_star =", sigma_phi_star)
    phi_star += np.random.normal(loc = 0, scale = sigma_phi_star, size = phi_star.shape)
    fig = px.line(phi_star)
    fig.show()
# 
    # Initialize parameters
    Q_0 = 1e2
    p_0 = np.array([Q_0])

    # Initialize priors
    C_phi = np.eye(phi_star.shape[0]) * sigma_phi_star**2
    p_bar = p_star
    sigma_p = 1e3
    C_p = np.eye(p_bar.shape[0]) * sigma_p**2

    # Optimization scheme parameters
    max_it = 100
    eps_tol = 4e-5 # Relative error between model and data

    # Optimization algorithm
    p_list, primal_dual_list, J_list, nabla_J_list, H_list = BFGS_adjoint_optimization(Omega = r, phi_star = phi_star, C_primal = C_phi, Solve_Primal_Dual = Solve_Primal_Dual_spherical_Poisson, J = J_spherical_Poisson, Grad_J = Grad_J_spherical_Poisson, p_0 = p_0, p_bar = p_bar, C_p = C_p, max_it = max_it, eps_tol = eps_tol)
    E_list = [J_list[k][1] for k in range(len(J_list))]
    R_list = [J_list[k][2] for k in range(len(J_list))]
    J_list = [J_list[k][0] for k in range(len(J_list))]
    nabla_M_list = [nabla_J_list[k][1] for k in range(len(nabla_J_list))]
    nabla_R_list = [nabla_J_list[k][2] for k in range(len(nabla_J_list))]
    nabla_J_list = [nabla_J_list[k][0] for k in range(len(nabla_J_list))]

    print("p_star = ", p_star)
    print("p_knot = ", p_list[-1])

    # Parameters
    fig = go.Figure()
    for i in range(p_0.shape[0]):
        fig.add_scatter(x = np.arange(len(p_list)), y = [p_list[k][i] for k in range(len(p_list))], name = "parameter i = " + str(i))
    fig.update_layout(title = "p(k)")
    fig.show()

    # Primal and dual
    fig = go.Figure()
    fig.add_scatter(x = r, y = phi_star, name = "phi_star")
    for k in range(len(primal_dual_list)):
        fig.add_scatter(x = r, y = primal_dual_list[k][0], name = "primal at step k = " + str(k))
        fig.add_scatter(x = r, y = primal_dual_list[k][1], name = "adjoint at step k = " + str(k))
    fig.show()

    # Functional
    fig = go.Figure()
    fig.add_scatter(x = np.arange(len(J_list)), y = np.log(J_list), name = "J(p)")
    fig.add_scatter(x = np.arange(len(E_list)), y = np.log(E_list), name = "E(p)")
    fig.add_scatter(x = np.arange(len(R_list)), y = np.log(R_list), name = "R(p)")
    fig.show()

    # Gradients
    fig = go.Figure()
    for i in range(p_0.shape[0]):
        fig.add_scatter(x = np.arange(len(nabla_J_list)), y = [np.log(nabla_J_list[k][i]) for k in range(len(nabla_J_list))], name = "grad J(p_i) for i = " + str(i))
        fig.add_scatter(x = np.arange(len(nabla_M_list)), y = [np.log(nabla_M_list[k][i]) for k in range(len(nabla_M_list))], name = "grad M(p_i) for i = " + str(i))
        fig.add_scatter(x = np.arange(len(nabla_R_list)), y = [np.log(nabla_R_list[k][i]) for k in range(len(nabla_R_list))], name = "grad R(p_i) for i = " + str(i))
    fig.show()

    # Hessian: compare BFGS approximation to analytical (exact Newton) Hessian
    H_exact_list = [Exact_inverse_Hessian(Omega = r, Solve_Primal_Dual = Solve_Primal_Dual_spherical_Poisson, phi_star = phi_star, C_primal = C_phi, nabla_previous = nabla_J_list[k], p_previous = p_list[k], s = np.array([[1]]), p_bar = p_bar, C_p = C_p, grad_J = Grad_J_spherical_Poisson) for k in range(len(H_list))]
    fig = go.Figure()
    fig.add_scatter(x = np.arange(len(H_list)), y = [H_list[k][0,0] for k in range(len(H_list))], name = "BFGS inverse Hessian approximation")
    fig.add_scatter(x = np.arange(len(H_exact_list)), y = [H_exact_list[k][0,0] for k in range(len(H_exact_list))], name = "Exact Hessian calculation")
    fig.show()

    # Data - model discrepancy
    # fig = go.Figure()
    ## Look at the paper to get inspiration!
    # fig.show()

    ### Uncertainty ###
    # Make uncertainty estimates
    H_knot = H_list[-1]
    # H_knot = H_exact_list[-1]
    p_knot = p_list[-1]

    M = 1000 # number of samples we want
    p_s_array= np.zeros((p_knot.shape[0], M))
    for m in range(M):
        p_s_array[:, m] = PCA_posterior_sampling(p_knot, H_knot)
    mean_p_s = np.mean(p_s_array, axis = 1)
    std_p_s = np.std(p_s_array, axis = 1)
    print("mean, std of posterior samples: ", mean_p_s, std_p_s)
    p_knot_list[l] = p_knot
    p_knot_mean_list[l] = mean_p_s
    p_knot_std_list[l] = std_p_s



