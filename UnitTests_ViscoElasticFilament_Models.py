import numpy as np
import pytest
from scipy.integrate import solve_ivp
import ViscoElasticFilament_Models as vf

# -----------------------------
# Helpers
# -----------------------------

def make_base_params(N=4):
    return dict(
        Sp4=1.0,
        k0=1.0,
        bool_EI=True,
        Beta=1.0,
        taus_b=[1.0] * (N - 1),
        tau_s=0.0,
        gamma=2.0,
        n_L=[0.0, 0.0],
        m_L=0.0,
        Lambdas=[0.0] * N,
        Zetas=[0.0] * N,
        InterpFlow=0,
    )

# -----------------------------
# Base parameter builders
# -----------------------------
def make_flow_params(N=4):
    """
    Builds a fully consistent parameter set for simulation tests.

    IMPORTANT:
    - Ensures all N-dependent fields are aligned
    - Produces simulation-ready int/ext/sim params
    """

    # -----------------------------
    # Internal parameters
    # -----------------------------
    int_params = {
        "Sp4": 1.0,
        "k0": 1.0,
        "bool_EI": True,
        "gamma": 2.0,
        "taus_b": [1.0] * (N - 1),
        "tau_s": 0.0,
        "n_L": [0.0, 0.0],
        "m_L": 0.0,
        "X_0": vf.StraightLine(N),
    }

    # -----------------------------
    # External parameters (NO InterpFlow yet)
    # -----------------------------
    ext_params = {
        "A": 1.0,
        "w0": 2.0,
        "psi": 0.3,
        "Lambdas": [0.0] * N,
        "Zetas": [0.0] * N,
    }

    # -----------------------------
    # Simulation parameters
    # -----------------------------
    sim_params = {
        "T_span": (0.0, 0.1),
        "T_eval": np.linspace(0.0, 0.1, 10),
        "T_sim_max": 1.0,
        "method": "RK45",
    }

    return int_params, ext_params, sim_params

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