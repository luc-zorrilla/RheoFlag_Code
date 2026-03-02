""" 
This script does a second pass of inference for a single filament, by:

    1. Looping on all the inferred parameters for varying external parameters (A, w0). For each inferred parameter, do:
        
        - If the inferred parameter is sufficiently far from the weighted average of inferred parameters without this one,
        compute the functional at this weighted average and compare to its precedent value to update the parameter.

        Note: see if one does this pass multiple times or not.

    4. Save as "*_pass_x".
"""

## Libraries

from C01_Inference_Viscoelastic_Model import *
from misc_func import *

import glob
import pickle
from pathlib import Path
writing_path = (Path(__file__).resolve().parent.parent / 'Inference' / 'FromSimulationData' / 'MultiplePeriods' / 'LastPeriod' / 'BendingElasticity_NoViscosity_Clamped' / 'Test_020326')
print("writing_path", writing_path)
import numpy as np
import pandas as pd

from plotly.subplots import make_subplots
import plotly.graph_objects as go

def inv_mat(M):
    """ Invert matrix or set to infinity if not invertible. """
    try:
        Mm1 = np.linalg.inv(M)
    except:
        Mm1 = np.ones_like(M) * np.inf
    return Mm1

def custom_average(x, sigma_x, type = "mean"):
        """ 
        Performs a custom average of x.
        Inputs: 
            - x: ndarray of shape (n_samples, nvars)
            - sigma_x: ndarray of same shape as x. If unknown, set to infinite.
            - type: a string corresponding to one element of L, 
                corresponding to the custom average type, where L = ["mean", "median", "combined"]
        Outputs:
            - avg_x: ndarray of shape (nvars,)
            - sigma_avg_x: ndarray of shape (nvars,) corresponding to std. of the estimator avg_x
        """
        n_samples = len(x)
        x = np.array(x).reshape((n_samples,-1))
        sigma_x = np.array(sigma_x).reshape((n_samples,-1))
        n_vars = x.shape[1] # Check shape of x and sigma_x
        avg_x = np.zeros((n_vars,))
        sigma_avg_x = np.zeros((n_vars,))

        for j in range(n_vars):
            Z_vector_list_j = np.array([np.array([x[k,j], sigma_x[k,j]]) for k in range(x.shape[0])]) # Error is measured from the hessian
            Z = Z_vector_list_j.reshape((-1,2))
            Z = Z[(Z[:,0] < np.inf) & (Z[:,1] < np.inf)]
            
            if type == 'mean':
                avg_x[j] = np.mean(Z, axis = 0)[0]
                sigma_avg_x[j] = (np.std(np.array(Z), axis = 0, ddof = 1) / np.sqrt(len(Z)))[0]
            elif type == 'median':
                avg_x[j] = np.median(Z, axis = 0)[0]
                sigma_avg_x[j] = np.nan # No std. for the median.
            elif type == "combined":
                avg_x[j], sigma_avg_x[j] = np.array(BLC(Z))
            else:
                raise ValueError("String does not correspond to available types of averages.")
            
        return avg_x, sigma_avg_x

def second_pass(p_inf, sigma, H, F_inf, average_p_but_one, red_func):
        """ 
        1. Check if (p_inf +- sigma) and (average_p_but_one +- average_sigma_but_one) intersects.
        2. If there is no intersection, compute new_F = F_p_inf(average_p_but_one)
        3. If new_F < F_p_inf, new_p_inf = average_p_but_one and new_H = hessian(new_p_inf)
        4. Return
        """

        p_but_one = average_p_but_one[0]
        sigma_but_one = average_p_but_one[1]

        if not np.isfinite(sigma_but_one):
            return p_inf, sigma, H, F_inf

        # Check if error is infinite or confidence intervals don't intersect
        if (not np.isfinite(sigma)) or ((p_inf - sigma) > (p_but_one + sigma_but_one) or (p_inf + sigma) < (p_but_one - sigma_but_one)):
            
            new_F = red_func(p_but_one)
            if new_F < F_inf:

                print("Updating outlier parameter...")

                # Parameter
                new_p_inf = p_but_one

                # Hessian and sigma
                m = p_inf.shape[0] # number of variables
                vec_func = Vectorize_Functional(red_func, m)      
                print("Compute hessian...")        
                new_hess = sd.hessian(f = vec_func, x = new_p_inf)
                if new_hess['success']:
                    print("Hessian computed.")
                    new_H = new_hess.ddf
                    print("new_H = ", new_H)
                else:
                    print("Hessian calculation failed. Status", new_hess.status)
                    new_H = np.zeros((m,m))                
                new_sigma = np.sqrt(np.diag(inv_mat(new_H)))

                return new_p_inf, new_sigma, new_H, new_F
            else:
                return p_inf, sigma, H, F_inf
        else: 
            return p_inf, sigma, H, F_inf

## Main

bool_test = False
if __name__ == "__main__":

    variable_keys_list = []
    A_list = []
    w0_list = []
    p_star_list = []
    p_inf_list = []
    guess_list = []
    IE_list = []
    Hm1_list = []
    H_list = []
    X_local_list = []
    F_local_list = []
    X_global_list = []
    F_global_list = []
    F_inf_list = []
    accept_global_list = []
    red_func_list = []
    ret_list = []

    filepaths = list(writing_path.glob('VI_dict*.pkl')) # List of path of VI_dict_*.pkl files in the writing path
    filenames = [str(filepath.resolve()) for filepath in filepaths] # Convert to strings in the corresponding OS
    print("filenames:", len(filenames))

    for filename in filenames:

        pkl_file = open(filename, 'rb')
        VI_dict = pickle.load(pkl_file)
        pkl_file.close()

        exp_variable_params = VI_dict["exp_variable_params"]
        variable_keys = list(exp_variable_params.keys())
        variable_keys_list.append(variable_keys)        
        guess_variable_params = VI_dict["args"]["guess_variable_params"]
        flow_params = VI_dict["flow_params"]
        A = flow_params["A"]
        w0 = flow_params["w0"]
        fixed_params = VI_dict["args"]["fixed_params"]

        p_star = np.array(list(exp_variable_params.values()))
        ret = VI_dict["output"][0]
        inferred_variable_params = ret.x
        p_inf = np.array(list(inferred_variable_params.values()))
        guess = np.array(list(guess_variable_params.values()))

        X_local, F_local, X_global, F_global, accept_global, red_func = VI_dict["output"][1:]
        for l in range(len(VI_dict["output"][1:-1])):
            V = [X_local, F_local, X_global, F_global, accept_global][l] # This will need to be changed to take into account the fact that X_local is a list of arrays. The easiest would be to concatenate it and now where it should be divided, or to turn it into a list of arrays for real
            if l in [2,3]:
                V = np.array(V).squeeze()
            V_list = [X_local_list, F_local_list, X_global_list, F_global_list, accept_global_list][l]
            V_list.append(V)
        F_inf_list.append(F_global[-1])
        red_func_list.append(red_func)

        IE = L2_relative_error(p_inf, p_star)
        Hm1 = ret['lowest_optimization_result']['hess_inv'].todense()
        H = ret['hessian']
        if not ret['success']: # If convergence failed, error is infinite
            Hm1 = np.ones_like(Hm1) * np.inf
            H = np.zeros_like(H)
        
        A_list.append(A)
        w0_list.append(w0)
        p_star_list.append(p_star)
        p_inf_list.append(p_inf)
        guess_list.append(guess)
        IE_list.append(IE)
        Hm1_list.append(Hm1)
        H_list.append(H)
        ret_list.append(ret)

    variable_keys = list(exp_variable_params.keys())
    print("variable_keys", variable_keys)

    # Make dataframe
    df = pd.DataFrame()

    df["variable_keys"] = variable_keys_list
    df["A"] = A_list
    df["w0"] = w0_list
    df["p_star"] = p_star_list
    df["p_inf"] = p_inf_list
    df["guess"] = guess_list
    df["IE"] = IE_list
    df["Hm1"] = [inv_mat(H) for H in H_list] # Hm1_list # Covariance matrix
    df["H"] = H_list # Hessian matrix
    df["sigma"] = df.apply(lambda x: np.sqrt(np.diag(x["Hm1"])), axis = 1)
    for key in ["X_local", "F_local", "X_global", "F_global", "accept_global"]:
        df[key] = eval(key + "_list")
    df["F_inf"] = F_inf_list
    df['red_func'] = red_func_list
    df["ret"] = ret_list
    
    n_vars = p_inf_list[0].shape[0]
    for k_vars in range(n_vars):
        df["p_inf_" + str(k_vars)] = df.apply(lambda x: x['p_inf'][k_vars], axis = 1)
        df["sigma_p_inf_" + str(k_vars)] = df.apply(lambda x: x['sigma'][k_vars], axis = 1)
    print("df", df)

    # Select files

    ## Select for a specific guess
    # Sp4_guess = 10
    # Beta_guess = 1e-3
    # df = df[df['guess'] == Sp4_guess].reset_index()

    ## Select for a specific exp filament
    Sp4_exp = 1e0
    Beta_exp = 0e0
    tau_b_exp = 0e0
    tau_s_exp = 0e0
    target = np.array([eval(key + "_exp") for key in variable_keys])
    df2 = df[df['p_star'].apply(lambda x: np.array_equal(x, target))].reset_index(drop=True)

    print("df2", df2)

    ## Loop over inferred parameters

    ### Make lists of parameters except the current one
    all_p_but_one_list = [] # Each element is a list of all parameters p_j EXCEPT p_i,
    all_sigma_but_one_list = [] # along with their uncertainty
    for k in range(len(df2['p_inf'])):
        
        all_p_but_one = []
        all_sigma_but_one = []
        for l in range(len(df2['p_inf'])):
            if l != k:
                all_p_but_one.append(df2['p_inf'][l])
                all_sigma_but_one.append(df2['sigma'][l])
        all_p_but_one_list.append(all_p_but_one)
        all_sigma_but_one_list.append(all_sigma_but_one)

    df2['all_p_but_one'] = all_p_but_one_list
    df2['all_sigma_but_one'] = all_sigma_but_one_list

    ### Perform average on this modified lists
    
    df2['mean_p_but_one'] = df2.apply(lambda x: custom_average(x['all_p_but_one'], x['all_sigma_but_one'], type = "mean"), axis = 1) # type: "mean", "median", "combined"
    df2['median_p_but_one'] = df2.apply(lambda x: custom_average(x['all_p_but_one'], x['all_sigma_but_one'], type = "median"), axis = 1)
    df2['combined_p_but_one'] = df2.apply(lambda x: custom_average(x['all_p_but_one'], x['all_sigma_but_one'], type = "combined"), axis = 1)

    # print("df2_avg_p", df2[['mean_p_but_one', 'median_p_but_one', 'combined_p_but_one']])

    ### If p is far from average_p_but_one, compute F_p(average_p_but_one)    
    
    df2[['new_p_inf', 'new_sigma', 'new_H', 'new_F_inf']] = df2.apply(lambda x: second_pass(x['p_inf'], x['sigma'], x['H'], x['F_inf'], x['mean_p_but_one'], x['red_func']), axis = 1, result_type = "expand")
    print("df2", df2[['p_inf', 'new_p_inf', 'sigma', 'new_sigma', 'F_inf', 'new_F_inf']])

    # Save new data in the same folder
    df2['new_IE'] = df2.apply(lambda x: L2_relative_error(x["p_inf"], x["p_star"]), axis = 1)
    important_keys = ["A", "w0", "p_inf", "new_p_inf", "Hm1", "H", "new_H", "sigma", "new_sigma", "F_inf", "new_F_inf", "IE", "new_IE"]
    important_keys += ["variable_keys", "p_star", "guess", "ret"]
    df3 = df2[important_keys]
    writing_filename = str((writing_path / "df.pkl").resolve())
    df3.to_pickle(writing_filename)

    # Global averages - this is just a test

    p_mean = custom_average(df2['p_inf'], df2['sigma'], type = "mean") # type: "mean", "median", "combined"
    p_median = custom_average(df2['p_inf'], df2['sigma'], type = "median")
    p_combined = custom_average(df2['p_inf'], df2['sigma'], type = "combined")

    new_p_mean = custom_average(df2['new_p_inf'], df2['new_sigma'], type = "mean") # type: "mean", "median", "combined"
    new_p_median = custom_average(df2['new_p_inf'], df2['new_sigma'], type = "median")
    new_p_combined = custom_average(df2['new_p_inf'], df2['new_sigma'], type = "combined")

    print("p_mean, new_p_mean", p_mean, new_p_mean)
    print("p_median, new_p_median", p_median, new_p_median)
    print("p_combined, new_p_combined", p_combined, new_p_combined)