""" 
This script does a second pass of inference for a single filament, by:

    1. Looping on all the inferred parameters for varying external parameters (A, w0). For each inferred parameter, do:
        
        - If the inferred parameter is sufficiently far from the weighted average of inferred parameters without this one,
        compute the functional at this weighted average and compare to its precedent value to update the parameter.

        Note: see if one does this pass multiple times or not.

    4. Save as "*_pass_x".
"""

# Libraries

from C01_Inference_Viscoelastic_Model import *
from misc_func import *

import glob
import dill as pickle
from pathlib import Path
writing_path = (Path(__file__).resolve().parent.parent / 'Inference' / 'FromSimulationData' / 'MultiplePeriods' / 'LastPeriod' / 'BendingElasticity_BendingViscosity_Clamped/FixedSp4')
print("writing_path", writing_path)
import numpy as np
import pandas as pd
import scipy.differentiate as sd

from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Global variables
param_keys = ["gamma", "N", "k0", "bool_EI", "Sp4", "tau_b", "Beta", "tau_s", "X0", "n_L", "m_L", "Lambdas", "Zetas", "InterpFlow", "method", "T_span", "T_eval", "T_sim_max"]

def get_p_inf_and_sigma_inf(VI_dict):
    """ Self-explanatory. """

    ret = VI_dict['output'][0]
    p_inf = np.array(list(ret.x.values()))
    H_inf = ret['hessian']
    if not ret['success']: # If convergence failed, error is infinite
        H_inf = np.zeros_like(H_inf)
    sigma_inf = np.sqrt(np.diag(inv_mat(H_inf)))    

    return np.array([p_inf, sigma_inf]).reshape(2,)

def get_other_params(row, df):
        """ Get all (p_inf, sigma_inf) pairs except the current row's.
        Returns an array of shape (2,-1) whose size on axis=1 is the n_samples-1. """

        others = []
        for _, other_row in df.iterrows():
            if row.name != other_row.name:  # Skip the current row
                ret = other_row['VI_dict']['output'][0] 
                p_inf = np.array(list(ret.x.values()))
                H_inf = ret['hessian']
                if not ret['success']: # If convergence failed, error is infinite
                    H_inf = np.zeros_like(H_inf)
                sigma_inf = np.sqrt(np.diag(inv_mat(H_inf)))          
                others.append(np.array([p_inf, sigma_inf]).reshape((2,)))
        return others


def second_pass(VI_dict, avg_other_params, writing_path):
        """ 
        1. Check if (p_inf +- sigma) and (average_p_but_one +- average_sigma_but_one) intersects.
        2. If there is no intersection, compute new_F = F_p_inf(average_p_but_one)
        3. If new_F < F_p_inf, 
            a. new_p_guess = average_p_but_one
            b. new_p_inf = Inference with new_p_guess
        4. Save VI_dict outputs for each re-inference
        5. Just copy data for those who did not get re-inferenced (ongoing)

        ### UPDATE: rewrite second_pass so that it takes 
            - as input VI_dict, average_p_but_one, writing_path OK
            - as output new_VI_dict OK
            - and writes down new_VI_dict to a writing path OK
        """

        # Obtain p_inf, H_inf, sigma_inf, F_inf, red_func
        ret = VI_dict['output'][0]
        p_inf = np.array(list(ret.x.values()))
        H_inf = ret['hessian']
        if not ret['success']: # If convergence failed, error is infinite
            H_inf = np.zeros_like(H_inf)
        sigma_inf = np.sqrt(np.diag(inv_mat(H_inf)))
        F_inf = ret.fun # Verify this
        red_func = VI_dict["output"][-1]
        new_writing_path = (writing_path / "SecondPass").resolve()
        new_writing_path.mkdir(parents=True, exist_ok=True)


        # Obtain outlier-free averages
        p_but_one, sigma_but_one = np.array(avg_other_params[0]).reshape(1,), np.array(avg_other_params[1]).reshape(1,)

        if not np.isfinite(sigma_but_one):
            new_VI_dict = VI_dict
            bool_updated = False

        # Check if error is infinite or confidence intervals don't intersect
        elif (not np.isfinite(sigma_inf)) or ((p_inf - sigma_inf) > (p_but_one + sigma_but_one) or (p_inf + sigma_inf) < (p_but_one - sigma_but_one)):
            
            new_F = red_func(p_but_one)
            if new_F < F_inf:

                print("Updating outlier parameter...")

                # New inloop args

                ## flow_params 
                flow_params = VI_dict["flow_params"]
                
                ## exp_params
                global param_keys
                exp_variable_params = VI_dict["exp_variable_params"]
                fixed_params = VI_dict["args"]["fixed_params"]
                exp_params = {}
                for param_key in param_keys:
                    if param_key in exp_variable_params.keys():
                        exp_params[param_key] = exp_variable_params[param_key]
                    elif param_key in fixed_params.keys():
                        exp_params[param_key] = fixed_params[param_key]
                    else:
                        raise ValueError("A key is not present neither in exp_variable_params nor in fixed_params")                
                
                ## guess_variable_params
                new_p_guess = p_but_one.reshape((-1,))
                new_guess_variable_params = VI_dict["args"]["guess_variable_params"]
                for k in range(new_p_guess.shape[0]):
                    key = list(new_guess_variable_params.keys())[k]
                    new_guess_variable_params[key] = new_p_guess[k]
                                
                ## bounds, disc_func, opt_scheme, opt_args
                bounds = VI_dict["args"]["bounds"]
                disc_func = VI_dict["args"]["disc_func"]
                opt_scheme = VI_dict["args"]["opt_scheme"]
                opt_args = VI_dict["args"]["opt_args"]

                ## Note: new_writing_path is defined above, i.e., even when a new inference is not required
                
                # Inference with new guess
                inloop_args = [flow_params, exp_params, new_guess_variable_params, bounds, disc_func, opt_scheme, opt_args, new_writing_path]
                new_VI_dict = Viscoelastic_inference_inloop(*inloop_args)
                bool_updated = True

            else:
                print("Keep outlier parameter...")
                bool_updated = False

        else:
            print("Keep parameter...")
            bool_updated = False

        if not bool_updated:
            new_VI_dict = VI_dict
            filename = write_VI_dict_to_path(new_VI_dict, new_writing_path)

        return new_VI_dict

## Main

bool_test = False
if __name__ == "__main__":
    
    ### Read filenames from writing_path

    filepaths = list(writing_path.glob('VI_dict*.pkl')) # List of path of VI_dict_*.pkl files in the writing path
    filenames = [str(filepath.resolve()) for filepath in filepaths] # Convert to strings in the corresponding OS
    print("# filenames:", len(filenames))

    ### Loop through filenames

    VI_dict_list = []
    for filename in filenames:
        
        #### Obtain VI_dict for each filename
        pkl_file = open(filename, 'rb')
        VI_dict = pickle.load(pkl_file)
        pkl_file.close()

        VI_dict_list.append(VI_dict)
    
    ### Make dataframe from list of VI_dicts
    df_VI = pd.DataFrame()
    df_VI["VI_dict"] = VI_dict_list
    df_VI['writing_path'] = [writing_path] * len(VI_dict_list)

    ### Filter elements in dataframe (ONGOING)

    ### Obtain excluded average parameters for each element of the dataframe (p_but_one)

    #### Make list of parameters without current parameter
    
    ## TEST
    # df_VI["p_inf"] = df_VI['VI_dict'].apply(get_p_inf_and_sigma_inf)
    # print("print(df_VI[p_inf])", df_VI["p_inf"])
    # mean_p_inf = custom_average(df_VI['p_inf'], "mean")
    # median_p_inf = custom_average(df_VI['p_inf'], "median")
    # combined_p_inf = custom_average(df_VI['p_inf'], "combined")
    # print("mean_p_inf, median_p_inf, combined_p_inf", mean_p_inf, median_p_inf, combined_p_inf)    
    # exit()
    ## TEST
    
    df_VI['other_params'] = df_VI.apply(lambda row: get_other_params(row, df_VI), axis=1)

    #### Perform average on this modified lists

    df_VI['mean_other_params'] = df_VI['other_params'].apply(lambda x: custom_average(x, "mean")) # not working?
    df_VI['median_other_params'] = df_VI['other_params'].apply(lambda x: custom_average(x, "median"))
    df_VI['combined_other_params'] = df_VI['other_params'].apply(lambda x: custom_average(x, "combined"))
    print("df_VI[mean_other_params]", df_VI['mean_other_params'])

    ### Apply second pass to each element of the dataframe. This includes writing VI_dict_2 in new directory.
    df_VI['VI_dict_2'] = df_VI.apply(lambda row: second_pass(row['VI_dict'], row['mean_other_params'], row['writing_path']), axis=1)

    ### Compare old average and new average
    df_VI["p_inf"] = df_VI['VI_dict'].apply(get_p_inf_and_sigma_inf)
    mean_p_inf = custom_average(df_VI['p_inf'], "mean")
    median_p_inf = custom_average(df_VI['p_inf'], "median")
    combined_p_inf = custom_average(df_VI['p_inf'], "combined")

    df_VI["new_p_inf"] = df_VI['VI_dict_2'].apply(get_p_inf_and_sigma_inf)
    mean_new_p_inf = custom_average(df_VI['new_p_inf'], "mean")
    median_new_p_inf = custom_average(df_VI['new_p_inf'], "median")
    combined_new_p_inf = custom_average(df_VI['new_p_inf'], "combined")
    print("print(df_VI[new_p_inf])", df_VI["new_p_inf"])

    print("mean_p_inf, mean_new_p_inf:", mean_p_inf, mean_new_p_inf)
    print("median_p_inf, median_new_p_inf:", median_p_inf, median_new_p_inf)
    print("combined_p_inf, combined_new_p_inf:", combined_p_inf, combined_new_p_inf)

    ### Write new, information-compact dataframe in directory
    # compact_df_VI = Compact(df_VI) # To define
    # writing_filename = None # To update
    # compact_df_VI.to_pickle(writing_filename)

    exit()

    # Update: add those lists for second pass
    flow_params_list = []
    exp_params_list = []
    guess_variable_params_list= []
    bounds_list = []
    disc_func_list = []
    opt_scheme_list = []
    opt_args_list = []
    writing_path_list = []

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
    print("# filenames:", len(filenames))

    for filename in filenames:

        pkl_file = open(filename, 'rb')
        VI_dict = pickle.load(pkl_file)
        pkl_file.close()

        print("VI_dict example:", VI_dict)
        exit()

        exp_variable_params = VI_dict["exp_variable_params"]
        variable_keys = list(exp_variable_params.keys())
        variable_keys_list.append(variable_keys)        
        guess_variable_params = VI_dict["args"]["guess_variable_params"]
        flow_params = VI_dict["flow_params"]
        A = flow_params["A"]
        w0 = flow_params["w0"]
        fixed_params = VI_dict["args"]["fixed_params"]

        ## Update: add those variables to the lists
        exp_params = {}
        for param_key in param_keys:
            if param_key in exp_variable_params.keys():
                exp_params[param_key] = exp_variable_params[param_key]
            elif param_key in fixed_params.keys():
                exp_params[param_key] = fixed_params[param_key]
            else:
                raise ValueError("A key is not present neither in exp_variable_params nor in fixed_params")
        bounds = VI_dict["args"]["bounds"]
        disc_func = VI_dict["args"]["disc_func"]
        opt_scheme = VI_dict["args"]["opt_scheme"]
        opt_args = VI_dict["args"]["opt_args"]
        ##

        p_star = np.array(list(exp_variable_params.values()))
        ret = VI_dict["output"][0]
        inferred_variable_params = ret.x
        p_inf = np.array(list(inferred_variable_params.values()))
        F_inf = ret.fun
        guess = np.array(list(guess_variable_params.values()))

        X_local, F_local, X_global, F_global, accept_global, red_func = VI_dict["output"][1:]
        for l in range(len(VI_dict["output"][1:-1])):
            V = [X_local, F_local, X_global, F_global, accept_global][l] # This will need to be changed to take into account the fact that X_local is a list of arrays. The easiest would be to concatenate it and now where it should be divided, or to turn it into a list of arrays for real
            if l in [2,3]:
                V = np.array(V).squeeze()
            V_list = [X_local_list, F_local_list, X_global_list, F_global_list, accept_global_list][l]
            V_list.append(V)
        F_inf_list.append(F_inf)
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

        ## Update: add those lists
        flow_params_list.append(flow_params)
        exp_params_list.append(exp_params)
        guess_variable_params_list.append(guess_variable_params)
        bounds_list.append(bounds)
        disc_func_list.append(disc_func)
        opt_scheme_list.append(opt_scheme)
        opt_args_list.append(opt_args)
        writing_path_list.append(writing_path) # Could be replaced by turning "filename" into a PathLib path
        ##

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

    ## Update: add those to df
    df["flow_params"] = flow_params_list
    df["exp_params"] = exp_params_list
    df["guess_variable_params"] = guess_variable_params_list
    df["bounds"] = bounds_list
    df["disc_func"] = disc_func_list
    df["opt_scheme"] = opt_scheme_list
    df["opt_args"] = opt_args_list
    df['writing_path'] = writing_path_list
    df['inloop_args'] = df[['flow_params', 'exp_params', 'guess_variable_params', 'bounds', 'disc_func', 'opt_scheme', 'opt_args', 'writing_path']].apply(lambda row: row.values.tolist(), axis=1)
    print("df['inloop_args']", df['inloop_args'])
    
    ##

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

    ### If p is far from average_p_but_one, 
    #### 1. Compute F_p(average_p_but_one)
    #### 2. Do inference with new initial guess
    
    # df2[['new_p_inf', 'new_sigma', 'new_H', 'new_F_inf']] = df2.apply(lambda x: second_pass(x['p_inf'], x['sigma'], x['H'], x['F_inf'], x['mean_p_but_one'], x['red_func'], x['inloop_args']), axis = 1, result_type = "expand")
    df2.apply(lambda x: second_pass(x['p_inf'], x['sigma'], x['H'], x['F_inf'], x['mean_p_but_one'], x['red_func'], x['inloop_args']), axis = 1)
    exit()
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