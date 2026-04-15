import pytest
import numpy as np
from Models import Model, merge_models, merge_multiple_models, compose_model, parallel_simulate_batch
from ViscoElasticFilament_Models import StraightLine, ViscoElasticFilament, FlowParams_to_InterpFlow

# --- Tests for ViscoElasticFilament ---
def test_visco_elastic_filament():
    """ Create an instance of the model. """
    
    # int_params: # X_0, N, taus_b, tau_s, gamma, n_L, m_L, Sp4, k0, bool_EI
    gamma = 2
    N = 10
    k0 = 1e13
    bool_EI = True
    Sp4 = 1
    taus_b = [0]*(N-1)
    Beta = 0
    tau_s = 0
    X_0 = StraightLine(N)
    n_L = [0,0]
    m_L = 0

    # ext_params: # Lambdas, Zetas, InterpFlow
    Lambdas = [[0,0] for k in range(N)]
    Zetas = [0]*N
    InterpFlow = 0

    # sim_params: # T_span, T_eval, T_sim_max, method
    T_span = [0,10]
    N_T = 100 # Not in params
    T_eval = np.linspace(T_span[0], T_span[1], N_T)
    T_sim_max = 3600
    method = 'BDF'

    int_params = {'X_0':X_0, 'N':N, 'taus_b':taus_b, 'tau_s':tau_s, 'gamma':gamma, 'n_L':n_L, 'm_L':m_L, 'Sp4':Sp4, 'k0':k0, 'bool_EI':bool_EI}
    ext_params = {'InterpFlow':InterpFlow, 'Lambdas': Lambdas, 'Zetas': Zetas}
    sim_params = {'T_span':T_span, 'T_eval':T_eval, 'T_sim_max':T_sim_max, 'method':method}

    viscoelastic_filament = ViscoElasticFilament(int_params, ext_params, sim_params)

    # Simulate single
    output = viscoelastic_filament.simulate_single()
    assert "value" in output
    assert "shape" in output
    assert output["shape"] == (N+2, N_T)


# --- Tests for ViscoElasticFilament_FlowParams ---
def test_visco_elastic_filament_flow_params():
    # Define the ViscoElasticFilament_FlowParams class by composing the ViscoElasticFilament class
    ViscoElasticFilament_FlowParams = compose_model(
        ViscoElasticFilament,
        compose_ext_params=FlowParams_to_InterpFlow
    )
    # int_params: # X_0, N, taus_b, tau_s, gamma, n_L, m_L, Sp4, k0, bool_EI
    gamma = 2
    N = 10
    k0 = 1e13
    bool_EI = True
    Sp4 = 1
    taus_b = [0]*(N-1)
    Beta = 0
    tau_s = 0
    X_0 = StraightLine(N)
    n_L = [0,0]
    m_L = 0

    # sim_params: # T_span, T_eval, T_sim_max, method
    T_span = [0,10]
    N_T = 100 # Not in params
    T_eval = np.linspace(T_span[0], T_span[1], N_T)
    T_sim_max = 3600
    method = 'BDF'

    # ext_params: # Lambdas, Zetas, A, w0, psi
    Lambdas = [[0,0] for k in range(N)]
    Zetas = [0]*N
    A = 1e-5
    w0 = 1e0
    psi = np.pi/2

    int_params = {'X_0':X_0, 'N':N, 'taus_b':taus_b, 'tau_s':tau_s, 'gamma':gamma, 'n_L':n_L, 'm_L':m_L, 'Sp4':Sp4, 'k0':k0, 'bool_EI':bool_EI}
    ext_params = {'A':A, 'w0':w0, 'psi':psi, 'Lambdas': Lambdas, 'Zetas': Zetas}
    sim_params = {'T_span':T_span, 'T_eval':T_eval, 'T_sim_max':T_sim_max, 'method':method}

    viscoelastic_filament_flowparams = ViscoElasticFilament_FlowParams(int_params, ext_params, sim_params)

    # Simulate single
    output = viscoelastic_filament_flowparams.simulate_single()
    assert "value" in output
    assert "shape" in output
    assert output["shape"] == (N+2, N_T)