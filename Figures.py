import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from ViscoElasticFilament_Inferences import workflow_elastic_viscous_general

import pickle
import json

def plot_sigma_vs_ext_param(workflow_output, int_params, ext_param_name, ax=None):
    """
    Plot sigma (Hessian-based standard error) for multiple internal parameters vs external parameter.
    All parameters plotted on same axis for comparison.
    
    Args:
        workflow_output: Dict from workflow_elastic_viscous_general
        int_params: List of internal parameter names (e.g., ['Sp4', 'Beta', 'eta'])
        ax: Matplotlib axis (creates new figure if None)
    
    Returns:
        Matplotlib axis object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    results_summary = workflow_output['results_summary']
    ext_param_ranges = workflow_output['ext_param_ranges']
    
    # Detect external parameter (assuming single external parameter)
    ext_values = ext_param_ranges[ext_param_name]
    
    if ext_values is None:
        raise ValueError(f"External parameter '{ext_param_name}' not found in workflow output")
    
    # Color palette for different parameters
    colors = plt.cm.tab10(np.linspace(0, 1, len(int_params)))
    
    # Plot each internal parameter
    for idx, int_param_name in enumerate(int_params):
        ext_list = []
        sigma_list = []
        
        for result in results_summary:
            task_key = result['task_key']
            
            # Parse task_key format: "int_{int_idx}_ext_{ext_idx}"
            parts = task_key.split('_')
            ext_idx = int(parts[-1])
            
            # Get sigma value for specified parameter
            sigma_key = f'{int_param_name}_sigma'
            sigma = result.get(sigma_key)
            
            if sigma is not None:
                # Handle both scalar and array values
                if isinstance(sigma, (list, np.ndarray)):
                    sigma_val = float(sigma[0]) if len(sigma) > 0 else None
                else:
                    sigma_val = float(sigma)
                
                if sigma_val is not None and np.isfinite(sigma_val):
                    ext_value = ext_values[ext_idx]
                    ext_list.append(ext_value)
                    sigma_list.append(sigma_val)
        
        if not sigma_list:
            print(f"Warning: No finite sigma values found for parameter '{int_param_name}'")
            continue
        
        # Scatter plot for this parameter
        ax.scatter(ext_list, sigma_list, alpha=0.6, s=50, 
                edgecolors='black', linewidth=0.5, 
                label=int_param_name, color=colors[idx])
    
    ax.set_xlabel(f'External Parameter {ext_param_name}', fontsize=12)
    ax.set_ylabel('Sigma (Hessian Std. Error)', fontsize=12)
    ax.set_title(f'Standard Error vs {ext_param_name} (Multiple Parameters)', fontsize=14, fontweight='bold')
    
    # Use log scale for common parameters (A, w0, etc.)
    if ext_param_name in ['A', 'w0', 'omega']:
        ax.set_xscale('log')
    
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    return ax


def plot_sigma_vs_ext_param_vec_size(workflow_outputs, int_params, ext_param_name, ax=None):
    """
    Plot sigma vs size of external parameter vector for cumulative inference runs.
    Multiple internal parameters plotted on same axis for comparison.
    
    Args:
        workflow_outputs: List of workflow output dicts from workflow_elastic_viscous_general
        int_params: List of internal parameter names (e.g., ['Sp4', 'Beta', 'eta'])
        ax: Matplotlib axis (creates new figure if None)
    
    Returns:
        Matplotlib axis object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    
    # Color palette for different parameters
    colors = plt.cm.tab10(np.linspace(0, 1, len(int_params)))
    
    # Plot each internal parameter
    for param_idx, int_param_name in enumerate(int_params):
        ext_vec_sizes = []
        sigma_all = []
        ext_vec_size_scatter = []
        
        for k, workflow_output in enumerate(workflow_outputs):
            results_summary = workflow_output['results_summary']
            ext_param_ranges = workflow_output['ext_param_ranges']
            
            ext_vec = ext_param_ranges[ext_param_name]
            ext_vec_size = len(ext_vec)
            
            # Extract all sigma values from this run
            sigmas_k = []
            for result in results_summary:
                sigma_key = f'{int_param_name}_sigma'
                sigma = result.get(sigma_key)
                
                if sigma is not None:
                    # Handle both scalar and array values
                    if isinstance(sigma, (list, np.ndarray)):
                        sigma_val = float(sigma[0]) if len(sigma) > 0 else None
                    else:
                        sigma_val = float(sigma)
                    
                    # Skip infinite values (inference failures)
                    if sigma_val is not None and np.isfinite(sigma_val):
                        sigmas_k.append(sigma_val)
                        sigma_all.append(sigma_val)
                        ext_vec_size_scatter.append(ext_vec_size)
            
            if sigmas_k:
                ext_vec_sizes.append(ext_vec_size)
        
        if not sigma_all:
            print(f"Warning: No finite sigma values found for parameter '{int_param_name}'")
            continue
        
        # Scatter plot of all individual points for this parameter
        ax.scatter(ext_vec_size_scatter, sigma_all, alpha=0.3, s=30, 
                edgecolors='none', color=colors[param_idx])
        
    ax.set_xlabel(f'Size of {ext_param_name} vector (number of points)', fontsize=12)
    ax.set_ylabel('Sigma (Hessian Std. Error)', fontsize=12)
    ax.set_title(f'Standard Error vs {ext_param_name} Vector Size (Cumulative, Multiple Parameters)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    return ax




# Usage
if __name__ == "__main__":

    # Bending Elasticity - Sp4
    int_param_ranges = {'Sp4': [1.0]}
    A_vec = np.pow(10, np.linspace(start=-6, stop=-1, num=6))
    ext_param_ranges = {'A': A_vec}
    elastic_params_list = ['Sp4']
    viscous_params_list = []

    inference_mode = "single_inference"
    checkpoint_str = "./Results/BendingElasticity/BendingElasticity"

    workflow_output = workflow_elastic_viscous_general(
        int_param_ranges=int_param_ranges,
        ext_param_ranges=ext_param_ranges,
        elastic_params_list = elastic_params_list,
        viscous_params_list = viscous_params_list,
        inference_mode = inference_mode,
        checkpoint_str=checkpoint_str,
        )

    ax = plot_sigma_vs_ext_param(workflow_output, int_params=['Sp4'], ext_param_name='A')
    plt.tight_layout()
    plt.show()

    inference_mode = "cumulative_inference"
    workflow_outputs = []
    for k in range(6):
        A_vec_k = np.pow(10, np.linspace(start = -6, stop = -6+k, num = k+1))
        ext_param_ranges = {'A': A_vec_k}
        checkpoint_str = f"./Results/BendingElasticity/BendingElasticity_{k}"

        workflow_outputs.append(workflow_elastic_viscous_general(
            int_param_ranges=int_param_ranges,
            ext_param_ranges=ext_param_ranges,
            elastic_params_list = elastic_params_list,
            viscous_params_list = viscous_params_list,
            inference_mode = inference_mode,
            checkpoint_str=checkpoint_str,
            ))
    
    ax = plot_sigma_vs_ext_param_vec_size(workflow_outputs, int_params=['Sp4'], ext_param_name='A')
    plt.tight_layout()
    plt.show()


    # Shear Elasticity - Beta

    int_param_ranges = {'Beta': [1.0]}
    A_vec = np.pow(10, np.linspace(start = -6, stop = -1, num = 6))
    ext_param_ranges = {'A': A_vec}
    elastic_params_list = ['Beta']
    viscous_params_list = []

    inference_mode = "single_inference"
    checkpoint_str = "./Results/ShearElasticity/ShearElasticity"

    workflow_output = workflow_elastic_viscous_general(
        int_param_ranges=int_param_ranges,
        ext_param_ranges=ext_param_ranges,
        elastic_params_list = elastic_params_list,
        viscous_params_list = viscous_params_list,
        inference_mode = inference_mode,
        checkpoint_str=checkpoint_str,
        )

    ax = plot_sigma_vs_ext_param(workflow_output, int_params=['Beta'], ext_param_name='A')
    plt.tight_layout()
    plt.show()        

    inference_mode = "cumulative_inference"   
    workflow_outputs = []
    for k in range(6):
        A_vec_k = np.pow(10, np.linspace(start = -6, stop = -6+k, num = k+1))
        ext_param_ranges = {'A': A_vec_k}
        checkpoint_str = f"./Results/ShearElasticity/ShearElasticity_{k}"

        workflow_outputs.append(workflow_elastic_viscous_general(
            int_param_ranges=int_param_ranges,
            ext_param_ranges=ext_param_ranges,
            elastic_params_list = elastic_params_list,
            viscous_params_list = viscous_params_list,
            inference_mode = inference_mode,
            checkpoint_str=checkpoint_str,
            ))

    ax = plot_sigma_vs_ext_param_vec_size(workflow_outputs, int_params=['Beta'], ext_param_name = 'A')
    plt.tight_layout()
    plt.show()

    # Bending & Shear Elasticities - Sp4, Beta

    int_param_ranges = {'Sp4': [1.0], 'Beta': [1.0]}
    A_vec = np.pow(10, np.linspace(start = -6, stop = -1, num = 6))
    ext_param_ranges = {'A': A_vec}
    elastic_params_list = ['Sp4', 'Beta']
    viscous_params_list = []

    inference_mode = "single_inference"
    checkpoint_str = "./Results/BendingShearElasticity/BendingShearElasticity"

    workflow_output = workflow_elastic_viscous_general(
        int_param_ranges=int_param_ranges,
        ext_param_ranges=ext_param_ranges,
        elastic_params_list = elastic_params_list,
        viscous_params_list = viscous_params_list,
        inference_mode = inference_mode,
        checkpoint_str=checkpoint_str,
        )

    ax = plot_sigma_vs_ext_param(workflow_output, int_params=['Sp4', 'Beta'], ext_param_name = 'A')
    plt.tight_layout()
    plt.show()    

    inference_mode = "cumulative_inference"   
    workflow_outputs = []
    for k in range(6):
        A_vec_k = np.pow(10, np.linspace(start = -6, stop = -6+k, num = k+1))
        ext_param_ranges = {'A': A_vec_k}
        checkpoint_str = f"./Results/BendingShearElasticity/BendingShearElasticity_{k}"

        workflow_outputs.append(workflow_elastic_viscous_general(
            int_param_ranges=int_param_ranges,
            ext_param_ranges=ext_param_ranges,
            elastic_params_list = elastic_params_list,
            viscous_params_list = viscous_params_list,
            inference_mode = inference_mode,
            checkpoint_str=checkpoint_str,
            ))
    
    ax = plot_sigma_vs_ext_param_vec_size(workflow_outputs, int_params=['Sp4', 'Beta'], ext_param_name='A')
    plt.tight_layout()
    plt.show()


    # Bending Viscosity (Fixed Bending Elasticity)

    int_param_ranges = {'tau_b': [1.0]}
    A_vec = [1e-6]
    w0_vec = np.pow(10, -np.linspace(start = -3, stop = 3, num = 7))
    ext_param_ranges = {'A': A_vec, 'w0':w0_vec}
    elastic_params_list = []
    viscous_params_list = ['tau_b']

    inference_mode = "single_inference"
    checkpoint_str = "./Results/BendingViscosity/BendingViscosity"

    workflow_output = workflow_elastic_viscous_general(
        int_param_ranges=int_param_ranges,
        ext_param_ranges=ext_param_ranges,
        elastic_params_list = elastic_params_list,
        viscous_params_list = viscous_params_list,
        inference_mode = inference_mode,
        checkpoint_str=checkpoint_str,
        )
    
    ax = plot_sigma_vs_ext_param(workflow_output, int_params=['tau_b'], ext_param_name='w0')
    plt.tight_layout()
    plt.show()        

    inference_mode = "cumulative_inference"  
    workflow_outputs = [] 
    for l in range(7):
        w0_vec_l = np.pow(10, -np.linspace(start = -3, stop = -3+l, num = l+1))
        ext_param_ranges = {'A': A_vec, 'w0':w0_vec_l}
        checkpoint_str = f"./Results/BendingViscosity/BendingViscosity_{l}"

        workflow_outputs.append(workflow_elastic_viscous_general(
            int_param_ranges=int_param_ranges,
            ext_param_ranges=ext_param_ranges,
            elastic_params_list = elastic_params_list,
            viscous_params_list = viscous_params_list,
            inference_mode = inference_mode,
            checkpoint_str=checkpoint_str,
            ))

    ax = plot_sigma_vs_ext_param_vec_size(workflow_outputs, int_params=['tau_b'], ext_param_name='w0')
    plt.tight_layout()
    plt.show()


    # Shear Viscosity (Fixed Bending Elasticity & Shear Elasticity)

    int_param_ranges = {'tau_s': [1.0], 'Beta':[1.0]}
    A_vec = [1e-6]
    w0_vec = np.pow(10, -np.linspace(start = -3, stop = 3, num = 7))
    ext_param_ranges = {'A': A_vec, 'w0':w0_vec}
    elastic_params_list = []
    viscous_params_list = ['tau_s']

    inference_mode = "single_inference"
    checkpoint_str = "./Results/ShearViscosity/ShearViscosity"

    workflow_output = workflow_elastic_viscous_general(
        int_param_ranges=int_param_ranges,
        ext_param_ranges=ext_param_ranges,
        elastic_params_list = elastic_params_list,
        viscous_params_list = viscous_params_list,
        inference_mode = inference_mode,
        checkpoint_str=checkpoint_str,
        )

    ax = plot_sigma_vs_ext_param(workflow_output, int_params=['tau_s'], ext_param_name='w0')
    plt.tight_layout()
    plt.show()    

    inference_mode = "cumulative_inference"   
    workflow_outputs = []
    for l in range(7):
        w0_vec_l = np.pow(10, -np.linspace(start = -3, stop = -3+l, num = l+1))
        ext_param_ranges = {'A': A_vec, 'w0':w0_vec_l}
        checkpoint_str = f"./Results/ShearViscosity/ShearViscosity_{l}"

        workflow_outputs.append(workflow_elastic_viscous_general(
            int_param_ranges=int_param_ranges,
            ext_param_ranges=ext_param_ranges,
            elastic_params_list = elastic_params_list,
            viscous_params_list = viscous_params_list,
            inference_mode = inference_mode,
            checkpoint_str=checkpoint_str,
            ))

    ax = plot_sigma_vs_ext_param_vec_size(workflow_outputs, int_params=['tau_s'], ext_param_name='w0')
    plt.tight_layout()
    plt.show()


    # Bending & Shear Viscosities (Fixed Bending & Shear Elasticities)

    int_param_ranges = {'tau_b': [1.0], 'tau_s':[1.0], 'Beta':[1.0]}
    A_vec = [1e-6]
    w0_vec = np.pow(10, -np.linspace(start = -3, stop = 3, num = 7))
    ext_param_ranges = {'A': A_vec, 'w0':w0_vec}
    elastic_params_list = []
    viscous_params_list = ['tau_b', 'tau_s']

    inference_mode = "single_inference"
    checkpoint_str = "./Results/BendingShearViscosity/BendingShearViscosity"

    workflow_output = workflow_elastic_viscous_general(
        int_param_ranges=int_param_ranges,
        ext_param_ranges=ext_param_ranges,
        elastic_params_list = elastic_params_list,
        viscous_params_list = viscous_params_list,
        inference_mode = inference_mode,
        checkpoint_str=checkpoint_str,
        )

    ax = plot_sigma_vs_ext_param(workflow_output, int_params=['tau_b', 'tau_s'], ext_param_name='w0')
    plt.tight_layout()
    plt.show()    

    inference_mode = "cumulative_inference"   
    workflow_outputs = []
    for l in range(7):
        w0_vec_l = np.pow(10, -np.linspace(start = -3, stop = -3+l, num = l+1))
        ext_param_ranges = {'A': A_vec, 'w0':w0_vec_l}
        checkpoint_str = f"./Results/BendingShearViscosity/BendingShearViscosity_{l}"

        workflow_outputs.append(workflow_elastic_viscous_general(
            int_param_ranges=int_param_ranges,
            ext_param_ranges=ext_param_ranges,
            elastic_params_list = elastic_params_list,
            viscous_params_list = viscous_params_list,
            inference_mode = inference_mode,
            checkpoint_str=checkpoint_str,
            ))

    ax = plot_sigma_vs_ext_param_vec_size(workflow_outputs, int_params=['tau_b', 'tau_s'], ext_param_name='w0')
    plt.tight_layout()
    plt.show()
