# import pytest
# import numpy as np
# from Models import Model, compose_model, parallel_simulate_batch
# from ViscoElasticFilament_Models import StraightLine, ViscoElasticFilament_create_params_list, ViscoElasticFilament_FlowParams_create_params_list, ViscoElasticFilament, FlowParams_to_InterpFlow, ViscoElasticFilament_FlowParams

# # --- Tests for ViscoElasticFilament ---
# def test_visco_elastic_filament():
#     """ Create an instance of the model. """
    
#     # int_params: # X_0, N, taus_b, tau_s, gamma, n_L, m_L, Sp4, k0, bool_EI
#     gamma = 2
#     N = 10
#     k0 = 1e13
#     bool_EI = True
#     Sp4 = 1
#     taus_b = [0]*(N-1)
#     Beta = 0
#     tau_s = 0
#     X_0 = StraightLine(N)
#     n_L = [0,0]
#     m_L = 0

#     # ext_params: # Lambdas, Zetas, InterpFlow
#     Lambdas = [[0,0] for k in range(N)]
#     Zetas = [0]*N
#     InterpFlow = 0

#     # sim_params: # T_span, T_eval, T_sim_max, method
#     T_span = [0,10]
#     N_T = 100 # Not in params
#     T_eval = np.linspace(T_span[0], T_span[1], N_T)
#     T_sim_max = 3600
#     method = 'BDF'

#     int_params = {'X_0':X_0, 'N':N, 'taus_b':taus_b, 'tau_s':tau_s, 'gamma':gamma, 'n_L':n_L, 'm_L':m_L, 'Sp4':Sp4, 'k0':k0, 'bool_EI':bool_EI}
#     ext_params = {'InterpFlow':InterpFlow, 'Lambdas': Lambdas, 'Zetas': Zetas}
#     sim_params = {'T_span':T_span, 'T_eval':T_eval, 'T_sim_max':T_sim_max, 'method':method}

#     viscoelastic_filament = ViscoElasticFilament(int_params, ext_params, sim_params)

#     # Simulate single
#     output = viscoelastic_filament.simulate_single()
#     assert "value" in output
#     assert "shape" in output
#     assert output["shape"] == (N+2, N_T)

# def test_visco_elastic_filament_batch():
#     """ Compute batches of models in parallel. """

#     int_params_keys = ["gamma",  "N", "k0", "bool_EI", "Sp4", "taus_b", "X_0", "Beta", "tau_s", "n_L", "m_L"]
#     ext_params_keys = ["Lambdas", "Zetas", "InterpFlow"]
#     sim_params_keys = ["T_span", "T_eval", "T_sim_max", "method"]
    
#     gamma_list = [1,2]
#     N_list = [10, 20]
#     k0_list = [1e13]
#     bool_EI_list = [True]
#     Sp4_list = [1,2]

#     taus_b_list = [[0]*(N-1) for N in N_list] # To be put in the N-loop
#     X_0_list = [StraightLine(N) for N in N_list] # To be put in the N-loop
    
#     Beta_list = [0,1]
#     tau_s_list = [0,1]
#     n_L_list = [[0,0]]
#     m_L_list = [0]

#     # ext_params: # Lambdas, Zetas, InterpFlow
#     Lambdas_list = [[[0,0] for k in range(N)] for N in N_list] # To be put in the N-loop
#     Zetas_list = [[0]*N for N in N_list] # To be put in the N-loop
#     InterpFlow_list = [0]

#     # sim_params: # T_span, T_eval, T_sim_max, method
#     T_span_list = [[0,10]]
#     N_T_list = [100, 10] # Not in params
#     T_eval_list = [[np.linspace(T_span[0], T_span[1], N_T) for T_span in T_span_list] for N_T in N_T_list]
#     T_sim_max_list = [3600]
#     method_list = ['BDF', 'Radau']

#     # Loop through parameters and simulate models
#     params_list_dict = {"gamma_list":gamma_list, "N_list":N_list, "k0_list":k0_list, "bool_EI_list":bool_EI_list, "Sp4_list":Sp4_list, "taus_b_list":taus_b_list, "X_0_list":X_0_list, "Beta_list":Beta_list, "tau_s_list":tau_s_list, "n_L_list":n_L_list, "m_L_list":m_L_list, "Lambdas_list":Lambdas_list, "Zetas_list":Zetas_list, "InterpFlow_list":InterpFlow_list, "T_span_list":T_span_list, "T_eval_list":T_eval_list, "T_sim_max_list":T_sim_max_list, "method_list":method_list}
#     params_list = ViscoElasticFilament_create_params_list(int_params_keys, ext_params_keys, sim_params_keys, params_list_dict)

#     # Simulate in parallel
#     outputs = parallel_simulate_batch(ViscoElasticFilament, params_list, n_jobs=4)
    
#     # Validate the output
#     for output in outputs:
#         assert "value" in output
#         assert "shape" in output
#         N = output['value'].shape[0] - 2
#         N_T = output['value'].shape[1]
#         assert output["shape"] == (N + 2, N_T)

# # --- Tests for ViscoElasticFilament_FlowParams ---
# def test_visco_elastic_filament_flow_params():
#     # Define the ViscoElasticFilament_FlowParams class by composing the ViscoElasticFilament class
#     ViscoElasticFilament_FlowParams = compose_model(
#         ViscoElasticFilament,
#         compose_ext_params=FlowParams_to_InterpFlow
#     )
#     # int_params: # X_0, N, taus_b, tau_s, gamma, n_L, m_L, Sp4, k0, bool_EI
#     gamma = 2
#     N = 10
#     k0 = 1e13
#     bool_EI = True
#     Sp4 = 1
#     taus_b = [0]*(N-1)
#     Beta = 0
#     tau_s = 0
#     X_0 = StraightLine(N)
#     n_L = [0,0]
#     m_L = 0

#     # sim_params: # T_span, T_eval, T_sim_max, method
#     T_span = [0,10]
#     N_T = 100 # Not in params
#     T_eval = np.linspace(T_span[0], T_span[1], N_T)
#     T_sim_max = 3600
#     method = 'BDF'

#     # ext_params: # Lambdas, Zetas, A, w0, psi
#     Lambdas = [[0,0] for k in range(N)]
#     Zetas = [0]*N
#     A = 1e-5
#     w0 = 1e0
#     psi = np.pi/2

#     int_params = {'X_0':X_0, 'N':N, 'taus_b':taus_b, 'tau_s':tau_s, 'gamma':gamma, 'n_L':n_L, 'm_L':m_L, 'Sp4':Sp4, 'k0':k0, 'bool_EI':bool_EI}
#     ext_params = {'A':A, 'w0':w0, 'psi':psi, 'Lambdas': Lambdas, 'Zetas': Zetas}
#     sim_params = {'T_span':T_span, 'T_eval':T_eval, 'T_sim_max':T_sim_max, 'method':method}

#     viscoelastic_filament_flowparams = ViscoElasticFilament_FlowParams(int_params, ext_params, sim_params)

#     # Simulate single
#     output = viscoelastic_filament_flowparams.simulate_single()
#     assert "value" in output
#     assert "shape" in output
#     assert output["shape"] == (N+2, N_T)


# def test_visco_elastic_filament_flow_params_batch():
#     """ Compute batches of models in parallel. """

#     int_params_keys = ["gamma",  "N", "k0", "bool_EI", "Sp4", "taus_b", "X_0", "Beta", "tau_s", "n_L", "m_L"]
#     ext_params_keys = ["Lambdas", "Zetas", "A", "w0", "psi"]
#     sim_params_keys = ["T_span", "T_eval", "T_sim_max", "method"]
    
#     gamma_list = [1,2]
#     N_list = [10, 20]
#     k0_list = [1e13]
#     bool_EI_list = [True]
#     Sp4_list = [1,2]

#     taus_b_list = [[0]*(N-1) for N in N_list] # To be put in the N-loop
#     X_0_list = [StraightLine(N) for N in N_list] # To be put in the N-loop
    
#     Beta_list = [0,1]
#     tau_s_list = [0,1]
#     n_L_list = [[0,0]]
#     m_L_list = [0]

#     # ext_params: # Lambdas, Zetas, InterpFlow
#     Lambdas_list = [[[0,0] for k in range(N)] for N in N_list] # To be put in the N-loop
#     Zetas_list = [[0]*N for N in N_list] # To be put in the N-loop
#     A_list = [1e-5, 1e-4, 0]
#     w0_list = [1e0, 0]
#     psi_list = [0]

#     # sim_params: # T_span, T_eval, T_sim_max, method
#     T_span_list = [[0,10]]
#     N_T_list = [100, 10] # Not in params
#     T_eval_list = [[np.linspace(T_span[0], T_span[1], N_T) for T_span in T_span_list] for N_T in N_T_list]
#     T_sim_max_list = [3600]
#     method_list = ['BDF', 'Radau']

#     # Loop through parameters and simulate models
#     # Get dictionary keys from ???
#     params_list_dict = {"gamma_list":gamma_list, "N_list":N_list, "k0_list":k0_list, "bool_EI_list":bool_EI_list, "Sp4_list":Sp4_list, "taus_b_list":taus_b_list, "X_0_list":X_0_list, "Beta_list":Beta_list, "tau_s_list":tau_s_list, "n_L_list":n_L_list, "m_L_list":m_L_list, "Lambdas_list":Lambdas_list, "Zetas_list":Zetas_list, "A_list":A_list, "w0_list":w0_list, "psi_list":psi_list, "T_span_list":T_span_list, "T_eval_list":T_eval_list, "T_sim_max_list":T_sim_max_list, "method_list":method_list}
#     params_list = ViscoElasticFilament_FlowParams_create_params_list(int_params_keys, ext_params_keys, sim_params_keys, params_list_dict)
    
#     # Simulate in parallel
#     outputs = parallel_simulate_batch(ViscoElasticFilament_FlowParams, params_list, n_jobs=4)
    
#     # Validate the output
#     for output in outputs:
#         assert "value" in output
#         assert "shape" in output
#         N = output['value'].shape[0] - 2
#         N_T = output['value'].shape[1]
#         assert output["shape"] == (N + 2, N_T)

# if __name__ == "__main__":
#     # Run pytest programmatically
#     pytest.main([__file__, "-v", "--tb=short"])

import numpy as np
import pytest
import ViscoElasticFilament_Models as vf

# ---------- BASIC BUILDING BLOCKS ---------- #

def test_straight_line_shape():
    N = 5
    X = vf.StraightLine(N)
    assert X.shape == (N + 2,)
    assert np.allclose(X, 0)


def test_theta_zero_case():
    X = np.zeros(10)
    theta = vf.Theta(X, 3)
    assert theta == 0


def test_theta_accumulation():
    X = np.zeros(10)
    X[2:6] = 1  # introduce angles
    theta = vf.Theta(X, 2)
    assert theta == np.sum(X[2:5])


# ---------- GEOMETRY TRANSFORMS ---------- #

def test_X2_straight_line():
    N = 5
    X = vf.StraightLine(N)

    pos = vf.X2(X, 3)
    # straight line → all cos(0)=1, sin(0)=0
    assert np.isclose(pos[0], 3)
    assert np.isclose(pos[1], 0)


def test_X3N_shape():
    N = 4
    X = vf.StraightLine(N)
    X3 = vf.X3N(X)

    assert X3.shape == (3 * N, 1)


def test_XNp2_inverse_consistency():
    N = 4
    X = vf.StraightLine(N)
    X3 = vf.X3N(X)
    X_back = vf.XNp2(X3)

    assert X_back.shape == (N + 2,)


# ---------- MATRICES ---------- #

def test_QQ_shape():
    N = 5
    X = vf.StraightLine(N)
    X3 = vf.X3N(X)

    Q = vf.QQ(X3)
    assert Q.shape == (3 * N, N + 2)


def test_AA_shape():
    N = 4
    X = vf.StraightLine(N)
    X3 = vf.X3N(X)

    A = vf.AA(X3, gamma=2)
    assert A.shape == (N + 2, 3 * N)


def test_ADB_valid():
    N = 5
    taus_b = [1] * (N - 1)
    A = vf.ADB(taus_b, N)

    assert A.shape == (N + 2, N + 2)


def test_ADS_shape():
    N = 5
    A = vf.ADS(N)

    assert A.shape == (N + 2, 3 * N)


# ---------- FLOW ---------- #

def test_create_flow_no_flow():
    label, flow = vf.CreateFlowField()

    assert label == "NO FLOW"
    assert flow == 0


def test_create_flow_constant():
    T = np.linspace(0, 1, 5)
    label, flow = vf.CreateFlowField(A=1, w0=0, psi=0, T_meas=T)

    assert flow.shape == (2, len(T))
    assert np.allclose(flow[1], 0)  # psi=0 → no y component


def test_flow_no_field():
    N = 4
    X = vf.StraightLine(N)
    X3 = vf.X3N(X)

    result = vf.Flow(X3)
    assert result.shape == (4 * N, 1)
    assert np.allclose(result, 0)


# ---------- RHS TERMS ---------- #

def test_BB_zero_straight_line():
    N = 5
    X = vf.StraightLine(N)
    X3 = vf.X3N(X)

    B = vf.BB(X3)
    assert np.allclose(B, 0)


def test_BS_shape():
    N = 5
    X = vf.StraightLine(N)
    X3 = vf.X3N(X)

    B = vf.BS(X3)
    assert B.shape == (N + 2, 1)


def test_BC_zero_defaults():
    N = 4
    X = vf.StraightLine(N)
    X3 = vf.X3N(X)

    B0 = vf.BC_0(X3)
    BL = vf.BC_L(X3)

    assert np.allclose(B0, 0)
    assert np.allclose(BL, 0)


# ---------- MAIN DYNAMICS ---------- #

def test_g_output_shape():
    N = 4
    X = vf.StraightLine(N)

    params = dict(
        Sp4=1,
        k0=1,
        bool_EI=True,
        Beta=1,
        taus_b=[1]*(N-1),
        tau_s=0,
        gamma=2,
        n_L=[0, 0],
        m_L=0,
        Lambdas=[0]*N,
        Zetas=[0]*N,
        InterpFlow=0
    )

    X_dot = vf.g(0, X, **params)
    assert X_dot.shape == X.shape


def test_g_fixed_base():
    N = 4
    X = vf.StraightLine(N)

    params = dict(
        Sp4=1,
        k0=1,
        bool_EI=True,
        Beta=1,
        taus_b=[1]*(N-1),
        tau_s=0,
        gamma=2,
        n_L=[0, 0],
        m_L=0,
        Lambdas=[0]*N,
        Zetas=[0]*N,
        InterpFlow=0
    )

    X_dot = vf.g(0, X, **params)

    # enforced in code
    assert X_dot[0] == 0
    assert X_dot[1] == 0


# ---------- INTEGRATION ---------- #

def test_simulation_runs():
    N = 3
    X0 = vf.StraightLine(N)

    int_params = {
        "Sp4": 1,
        "k0": 1,
        "bool_EI": True,
        "gamma": 2,
        "taus_b": [1]*(N-1),
        "tau_s": 0,
        "n_L": [0, 0],
        "m_L": 0,
        "X_0": X0
    }

    ext_params = {
        "Lambdas": [0]*N,
        "Zetas": [0]*N,
        "InterpFlow": 0
    }

    sim_params = {
        "T_span": (0, 0.1),
        "T_eval": np.linspace(0, 0.1, 5),
        "method": "RK45",
        "T_sim_max": 5
    }

    out = vf.ViscoElasticFilament_Simulate(int_params, ext_params, sim_params)

    assert out["value"] is not None

if __name__ == "__main__":
    # Run pytest programmatically
    pytest.main([__file__, "-v", "--tb=short"])