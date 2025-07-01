# Libraries
import numpy as np
import scipy.optimize as so
from optimparallel import minimize_parallel # L-BFGS-B parallel implementation

import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from plotly.express.colors import sample_colorscale
import plotly.io as pio

import glob
import pickle
import pprint
writing_dir = "C:\\Users\\Luc\\Documents\\PhD_Large_files\\RheoFlag\\Inference\\FromSimulationData\\"

from A02_Inference_Viscoelastic_Model import *

bool_test = False
if __name__ == "__main__":

    filenames = glob.glob(writing_dir + "*.pkl")
    for filename in filenames:

        pkl_file = open(filename, 'rb')
        VI_dict = pickle.load(pkl_file)
        pkl_file.close()

        pprint.pprint(VI_dict)
        # Hessian
        # print(VI_dict['output']['lowest_optimization_result']['hess_inv'].todense())

        fixed_params = VI_dict["fixed_params"]
        
        exp_variable_params = VI_dict["exp_variable_params"]
        p_star = np.array(list(exp_variable_params.values()))
        inferred_variable_params = VI_dict["output"].x
        p = np.array(list(inferred_variable_params.values()))

        IE = L2_relative_error(p, p_star)
        print("L2 Relative Inference Error:", IE)