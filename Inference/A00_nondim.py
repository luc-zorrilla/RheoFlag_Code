""" 
This script is to write functions and tests for encoding physical parameters of the ViscoElastic Model into non-dimensional parameters, and vice-versa (when possible).
"""

# Physical parameters
L = 1e-5 # CR, wt
eta = 1 # ?
csi = 2 # ?
k_0 = 4.6e-12 # mCCD cells, primary cilium ## Range [0, 4.6e-12] --- TO CONVERT TO N.m/rad
E_b = 1e-21 # CR, wt, vanadate
nu_b = 1.6e-24 # CR, demembranated + ATP-reactivated ## Range [0, 1.6e-24]
K_s = 8e-11 # CR, wt, vanadate
nu_s = 1e-5 # CR, demembranated + ATP-reactivated ## Range [0, 1e-5]  --- TO CONVERT TO N.s/rad

# Expected Dimensional Units looking at the mathematical model
unit_dict = {}
unit_dict["L"] = "m"
unit_dict["eta"] = "N.s/m^2/rad"
unit_dict["csi"] = "N.s/m^2/rad"
unit_dict["k_0"] = "N.m/rad"
unit_dict["E_b"] = "N.m^2/rad"
unit_dict["nu_b"] = "N.m^2.s/rad"
unit_dict["K_s"] = "N/rad"
unit_dict["nu_s"] = "N.s/rad"

# Non-dimensional & Simulation parameters
N = 10
Delta_s = L/N
gamma = csi/eta
Sp4 = eta * (Delta_s ** 4) / E_b
tau_b = nu_b / E_b
Beta = K_s * (Delta_s ** 2) / E_b # For sea urchin sperm, demembranated: [3,8]e-14 m^{-2}
tau_s = nu_s / K_s

unit_dict = {}
unit_dict["N"] = "_"
unit_dict["Delta_s"] = "m"
unit_dict["gamma"] = "_"
unit_dict["Sp4"] = "_"
unit_dict["tau_b"] = "s"
unit_dict["Beta"] = "_"
unit_dict["tau_s"] = "s"

# External forcing parameters
A_dim = 1e-5 # Range [0, 70]e-6 (m)
A = A_dim / Delta_s
w0 = 50 # Range [0, 100] (Hz)
Psi = np.pi/2

unit_dict["A_dim"] = "m"
unit_dict["A"] = "_"
unit_dict["w0"] = "Hz"
unit_dict["Psi"] = "rad"
