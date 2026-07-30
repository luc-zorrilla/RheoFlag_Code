import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from ViscoElasticFilament_Inferences import workflow_elastic_viscous_general

import pickle
import json

def plot_sigma_vs_ext_param(workflow_output, int_params, ext_param_name, metric='std', ax=None):
    """
    Plot metric (Hessian-based standard error or relative error) for multiple internal 
    parameters vs external parameter. Color-codes by internal parameter values when they vary.
    
    Args:
        workflow_output: Dict from workflow_elastic_viscous_general
        int_params: List of internal parameter names (e.g., ['tau_b', 'tau_s'])
        ext_param_name: Name of external parameter to plot against
        metric: 'std' for Hessian standard error, 'rel_error' for relative error (L2 norm)
        ax: Matplotlib axis (creates new figure if None)
    
    Returns:
        Matplotlib axis object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 7))
    
    results_summary = workflow_output['results_summary']
    ext_param_ranges = workflow_output['ext_param_ranges']
    int_param_ranges = workflow_output['int_param_ranges']
    
    # Detect external parameter
    ext_values = ext_param_ranges[ext_param_name]
    if ext_values is None:
        raise ValueError(f"External parameter '{ext_param_name}' not found in workflow output")
    
    # Identify which internal parameters vary (have >1 value)
    varying_params = {
        name: values for name, values in int_param_ranges.items()
        if name in int_params and len(values) > 1
    }
    fixed_params = {
        name: values for name, values in int_param_ranges.items()
        if name in int_params and len(values) == 1
    }
    
    # Build color palette for each varying parameter
    param_color_palettes = {}
    for param_name in varying_params:
        num_values = len(varying_params[param_name])
        param_color_palettes[param_name] = plt.cm.viridis(np.linspace(0, 1, num_values))
    
    # Color palette for fixed parameters and combined metric
    fixed_param_colors = plt.cm.tab20(np.linspace(0, 1, len(fixed_params) + 1))
    
    # Plot each internal parameter
    for param_idx, int_param_name in enumerate(int_params):
        ext_list = []
        metric_list = []
        color_list = []
        
        for result in results_summary:
            task_key = result['task_key']
            
            # Parse task_key format: "int_{int_idx}_ext_{ext_idx}"
            parts = task_key.split('_')
            int_idx = int(parts[1])
            ext_idx = int(parts[-1])
            
            # Get metric value for this parameter
            if metric == 'std':
                metric_key = f'{int_param_name}_sigma'
            elif metric == 'rel_error':
                metric_key = f'{int_param_name}_rel_error'
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            value = result.get(metric_key)
            if value is not None:
                # Handle both scalar and array values
                if isinstance(value, (list, np.ndarray)):
                    metric_val = float(value[0]) if len(value) > 0 else None
                else:
                    metric_val = float(value)
                
                if metric_val is not None and np.isfinite(metric_val):
                    ext_value = ext_values[ext_idx]
                    ext_list.append(ext_value)
                    metric_list.append(metric_val)
                    
                    # Determine color based on parameter type
                    if int_param_name in varying_params:
                        # Color by internal parameter value
                        param_values = varying_params[int_param_name]
                        color = param_color_palettes[int_param_name][int_idx]
                        param_value = param_values[int_idx]
                        # Format legend label with parameter value
                        if param_value >= 1e3 or (param_value < 1e-2 and param_value > 0):
                            label = f'{int_param_name}={param_value:.1e}'
                        else:
                            label = f'{int_param_name}={param_value}'
                    else:
                        # Fixed parameter—use fixed color
                        color = fixed_param_colors[param_idx]
                        param_value = fixed_params[int_param_name][0]
                        if param_value >= 1e3 or (param_value < 1e-2 and param_value > 0):
                            label = f'{int_param_name}={param_value:.1e}'
                        else:
                            label = f'{int_param_name}={param_value}'
                    
                    color_list.append((color, label))
        
        if not metric_list:
            print(f"Warning: No finite {metric} values found for parameter '{int_param_name}'")
            continue
        
        # Group by color to avoid duplicate legend entries
        color_to_data = {}
        for ext_val, metric_val, (color, label) in zip(ext_list, metric_list, color_list):
            color_tuple = tuple(color)  # Convert to tuple for hashing
            if color_tuple not in color_to_data:
                color_to_data[color_tuple] = {'ext': [], 'metric': [], 'label': label}
            color_to_data[color_tuple]['ext'].append(ext_val)
            color_to_data[color_tuple]['metric'].append(metric_val)
        
        # Plot each color group separately (one legend entry per color)
        for color_tuple, data in color_to_data.items():
            ax.scatter(
                data['ext'], data['metric'],
                alpha=0.6, s=70,
                color=np.array(color_tuple).reshape(1, -1)[0],
                edgecolors='black', linewidth=0.5,
                label=data['label'],
                zorder=2
            )
    
    # Plot combined metric if multiple parameters
    if len(int_params) > 1:
        combined_ext_list = []
        combined_metric_list = []
        combined_color_list = []
        
        for result in results_summary:
            task_key = result['task_key']
            
            # Parse task_key
            parts = task_key.split('_')
            int_idx = int(parts[1])
            ext_idx = int(parts[-1])
            
            # Collect all metric values for this result
            metrics_combined = []
            all_finite = True
            
            for int_param_name in int_params:
                if metric == 'std':
                    metric_key = f'{int_param_name}_sigma'
                elif metric == 'rel_error':
                    metric_key = f'{int_param_name}_rel_error'
                
                value = result.get(metric_key)
                
                if value is not None:
                    if isinstance(value, (list, np.ndarray)):
                        metric_val = float(value[0]) if len(value) > 0 else None
                    else:
                        metric_val = float(value)
                    
                    if metric_val is not None and np.isfinite(metric_val):
                        metrics_combined.append(metric_val)
                    else:
                        all_finite = False
                        break
                else:
                    all_finite = False
                    break
            
            # Compute combined metric (L2 norm)
            if all_finite and len(metrics_combined) == len(int_params):
                combined_metric = np.sqrt(np.sum(np.array(metrics_combined) ** 2))
                combined_metric_list.append(combined_metric)
                ext_value = ext_values[ext_idx]
                combined_ext_list.append(ext_value)
                
                # Determine color for combined metric
                # Priority: use first varying parameter, or first fixed parameter
                color_assigned = False
                
                if varying_params:
                    # Use first varying parameter for color
                    first_varying = list(varying_params.keys())[0]
                    param_values = varying_params[first_varying]
                    color = param_color_palettes[first_varying][int_idx]
                    color_assigned = True
                else:
                    # Use first fixed parameter (all have same value, so use gray)
                    color = 'gray'
                
                combined_color_list.append(color)
        
        if combined_metric_list:
            # Scatter plot of combined metric with distinct marker
            ax.scatter(
                combined_ext_list, combined_metric_list,
                alpha=0.7, s=120,
                c=combined_color_list,
                edgecolors='black', linewidth=1.5,
                marker='D',  # Diamond marker
                label='Combined (L2 norm)',
                zorder=3
            )
    
    # Axes and labels
    ax.set_xlabel(f'External Parameter: {ext_param_name}', fontsize=12, fontweight='bold')
    
    if metric == 'std':
        ax.set_ylabel('Sigma (Hessian Std. Error)', fontsize=12, fontweight='bold')
        title_suffix = 'Standard Error'
    else:
        ax.set_ylabel('Relative Error (L2 norm)', fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        title_suffix = 'Relative Error'
    
    ax.set_title(f'{title_suffix} vs {ext_param_name}', fontsize=14, fontweight='bold')
    
    # Use log scale for common parameters
    if ext_param_name in ['A', 'w0', 'omega']:
        ax.set_xscale('log')
    
    ax.grid(True, alpha=0.3, zorder=1)
    ax.legend(fontsize=10, loc='best', framealpha=0.95)
    
    return ax


def plot_sigma_vs_ext_param_vec_size(workflow_outputs, int_params, ext_param_name, metric='std', ax=None):
    """
    Plot metric vs size of external parameter vector for cumulative inference runs.
    Multiple internal parameters plotted on same axis for comparison.
    
    Plots all individual samples (no mean/std aggregation) with improved marker styling.
    Also plots combined metric (L2 norm) alongside individual metrics.
    
    Args:
        workflow_outputs: List of workflow output dicts from workflow_elastic_viscous_general
        int_params: List of internal parameter names (e.g., ['Sp4', 'Beta', 'eta'])
        metric: 'std' for Hessian standard error, 'rel_error' for relative error (L2 norm)
        ax: Matplotlib axis (creates new figure if None)
    
    Returns:
        Matplotlib axis object
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 7))
    
    # Color palette for different parameters
    colors = plt.cm.tab10(np.linspace(0, 1, len(int_params) + 1))  # +1 for combined metric
    
    # Detect external parameter name from first workflow
    ext_param_ranges = workflow_outputs[0]['ext_param_ranges']
    
    # Plot each internal parameter
    for param_idx, int_param_name in enumerate(int_params):
        ext_vec_size_scatter = []
        metric_all = []
        
        for k, workflow_output in enumerate(workflow_outputs):
            results_summary = workflow_output['results_summary']
            ext_param_ranges = workflow_output['ext_param_ranges']
            ext_vec = ext_param_ranges[ext_param_name]
            ext_vec_size = len(ext_vec)
            
            # Extract all metric values from this run
            for result in results_summary:
                if metric == 'std':
                    metric_key = f'{int_param_name}_sigma'
                elif metric == 'rel_error':
                    metric_key = f'{int_param_name}_rel_error'
                else:
                    raise ValueError(f"Unknown metric: {metric}")
                
                value = result.get(metric_key)
                
                if value is not None:
                    # Handle both scalar and array values
                    if isinstance(value, (list, np.ndarray)):
                        metric_val = float(value[0]) if len(value) > 0 else None
                    else:
                        metric_val = float(value)
                    
                    # Skip infinite values (inference failures)
                    if metric_val is not None and np.isfinite(metric_val):
                        metric_all.append(metric_val)
                        ext_vec_size_scatter.append(ext_vec_size)
        
        if not metric_all:
            print(f"Warning: No finite {metric} values found for parameter '{int_param_name}'")
            continue
        
        # Scatter plot of all individual points for this parameter
        # Use better marker styling: larger, semi-transparent, with subtle edges
        ax.scatter(ext_vec_size_scatter, metric_all, 
                    alpha=0.5, s=60, 
                    edgecolors=colors[param_idx], linewidth=0.5,
                    color=colors[param_idx], 
                    label=int_param_name, 
                    zorder=2)
    
    # Plot combined metric if multiple parameters
    if len(int_params) > 1:
        combined_metric_all = []
        combined_ext_vec_size_scatter = []
        
        for k, workflow_output in enumerate(workflow_outputs):
            results_summary = workflow_output['results_summary']
            ext_param_ranges = workflow_output['ext_param_ranges']
            ext_vec = ext_param_ranges[ext_param_name]
            ext_vec_size = len(ext_vec)
            
            for result in results_summary:
                # Collect all metric values for this result
                metrics_combined = []
                all_finite = True
                
                for int_param_name in int_params:
                    if metric == 'std':
                        metric_key = f'{int_param_name}_sigma'
                    elif metric == 'rel_error':
                        metric_key = f'{int_param_name}_rel_error'
                    
                    value = result.get(metric_key)
                    
                    if value is not None:
                        if isinstance(value, (list, np.ndarray)):
                            metric_val = float(value[0]) if len(value) > 0 else None
                        else:
                            metric_val = float(value)
                        
                        if metric_val is not None and np.isfinite(metric_val):
                            metrics_combined.append(metric_val)
                        else:
                            all_finite = False
                            break
                    else:
                        all_finite = False
                        break
                
                # Compute combined metric (L2 norm of individual metrics)
                if all_finite and len(metrics_combined) == len(int_params):
                    combined_metric = np.sqrt(np.sum(np.array(metrics_combined) ** 2))
                    combined_metric_all.append(combined_metric)
                    combined_ext_vec_size_scatter.append(ext_vec_size)
        
        if combined_metric_all:
            # Scatter plot of combined metric with distinct marker style
            ax.scatter(combined_ext_vec_size_scatter, combined_metric_all, 
                    alpha=0.6, s=100,  # Larger and more visible
                    edgecolors=colors[-1], linewidth=1,
                    marker='D',  # Diamond marker for distinction
                    color=colors[-1], 
                    label=f'Combined (L2 norm)',
                    zorder=3)
    
    ax.set_xlabel(f'Size of {ext_param_name} vector (number of points)', fontsize=12)
    
    if metric == 'std':
        ax.set_ylabel('Sigma (Hessian Std. Error)', fontsize=12)
        title_suffix = 'Standard Error'
    else:
        ax.set_ylabel('Relative Error (L2 norm)', fontsize=12)
        ax.set_yscale('log')
        title_suffix = 'Relative Error'
    
    ax.set_title(f'{title_suffix} vs {ext_param_name} Vector Size (All Samples)', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, zorder=1)
    ax.legend(fontsize=11, loc='best')
    
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

    ax = plot_sigma_vs_ext_param(workflow_output, int_params=['Sp4'], ext_param_name='A', metric='std')
    plt.tight_layout()
    plt.show()

    ax = plot_sigma_vs_ext_param(workflow_output, int_params=['Sp4'], ext_param_name='A', metric='rel_error')
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
    
    ax = plot_sigma_vs_ext_param_vec_size(workflow_outputs, int_params=['Sp4'], ext_param_name='A', metric = 'std')
    plt.tight_layout()
    plt.show()

    ax = plot_sigma_vs_ext_param_vec_size(workflow_outputs, int_params=['Sp4'], ext_param_name='A', metric = 'rel_error')
    plt.tight_layout()
    plt.show()

    # Shear Elasticity - Beta

    int_param_ranges = {'Beta': [1e-3, 1e0, 1e3]}
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

    ax = plot_sigma_vs_ext_param(workflow_output, int_params=['Beta'], ext_param_name='A', metric = 'std')
    plt.tight_layout()
    plt.show()    

    ax = plot_sigma_vs_ext_param(workflow_output, int_params=['Beta'], ext_param_name='A', metric = 'rel_error')
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

    ax = plot_sigma_vs_ext_param_vec_size(workflow_outputs, int_params=['Beta'], ext_param_name = 'A', metric = 'std')
    plt.tight_layout()
    plt.show()

    ax = plot_sigma_vs_ext_param_vec_size(workflow_outputs, int_params=['Beta'], ext_param_name = 'A', metric = 'rel_error')
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

    ax = plot_sigma_vs_ext_param(workflow_output, int_params=['Sp4', 'Beta'], ext_param_name = 'A', metric = 'std')
    plt.tight_layout()
    plt.show() 

    ax = plot_sigma_vs_ext_param(workflow_output, int_params=['Sp4', 'Beta'], ext_param_name = 'A', metric = 'rel_error')
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
    
    ax = plot_sigma_vs_ext_param_vec_size(workflow_outputs, int_params=['Sp4', 'Beta'], ext_param_name='A', metric = 'std')
    plt.tight_layout()
    plt.show()

    ax = plot_sigma_vs_ext_param_vec_size(workflow_outputs, int_params=['Sp4', 'Beta'], ext_param_name='A', metric = 'rel_error')
    plt.tight_layout()
    plt.show()


    # Bending Viscosity (Fixed Bending Elasticity)

    int_param_ranges = {'tau_b': [1.0]}
    A_vec = [1e-6]
    w0_vec = np.pow(10, -np.linspace(start = -4, stop = 2, num = 7))
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
    
    ax = plot_sigma_vs_ext_param(workflow_output, int_params=['tau_b'], ext_param_name='w0', metric = 'std')
    plt.tight_layout()
    plt.show()    

    ax = plot_sigma_vs_ext_param(workflow_output, int_params=['tau_b'], ext_param_name='w0', metric = 'rel_error')
    plt.tight_layout()
    plt.show()        

    inference_mode = "cumulative_inference"  
    workflow_outputs = [] 
    for l in range(7):
        w0_vec_l = np.pow(10, -np.linspace(start = -4, stop = -4+l, num = l+1))
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

    ax = plot_sigma_vs_ext_param_vec_size(workflow_outputs, int_params=['tau_b'], ext_param_name='w0', metric = 'std')
    plt.tight_layout()
    plt.show()

    ax = plot_sigma_vs_ext_param_vec_size(workflow_outputs, int_params=['tau_b'], ext_param_name='w0', metric = 'rel_error')
    plt.tight_layout()
    plt.show()

    # Shear Viscosity (Fixed Bending Elasticity & Shear Elasticity)

    int_param_ranges = {'tau_s': [1.0], 'Beta':[1.0]}
    A_vec = [1e-6]
    w0_vec = np.pow(10, -np.linspace(start = -4, stop = 2, num = 7))
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

    ax = plot_sigma_vs_ext_param(workflow_output, int_params=['tau_s'], ext_param_name='w0', metric = 'std')
    plt.tight_layout()
    plt.show()    


    ax = plot_sigma_vs_ext_param(workflow_output, int_params=['tau_s'], ext_param_name='w0', metric = 'rel_error')
    plt.tight_layout()
    plt.show()    

    inference_mode = "cumulative_inference"   
    workflow_outputs = []
    for l in range(7):
        w0_vec_l = np.pow(10, -np.linspace(start = -4, stop = -4+l, num = l+1))
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

    ax = plot_sigma_vs_ext_param_vec_size(workflow_outputs, int_params=['tau_s'], ext_param_name='w0', metric = 'std')
    plt.tight_layout()
    plt.show()


    ax = plot_sigma_vs_ext_param_vec_size(workflow_outputs, int_params=['tau_s'], ext_param_name='w0', metric = 'rel_error')
    plt.tight_layout()
    plt.show()


    # Bending & Shear Viscosities (Fixed Bending & Shear Elasticities)

    int_param_ranges = {'tau_b': [1.0], 'tau_s':[1e-3, 1e0, 1e3], 'Beta':[1.0]}
    A_vec = [1e-6]
    w0_vec = np.pow(10, -np.linspace(start = -4, stop = 2, num = 7))
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

    ax = plot_sigma_vs_ext_param(workflow_output, int_params=['tau_b', 'tau_s'], ext_param_name='w0', metric = 'std')
    plt.tight_layout()
    plt.show()    

    ax = plot_sigma_vs_ext_param(workflow_output, int_params=['tau_b', 'tau_s'], ext_param_name='w0', metric = 'rel_error')
    plt.tight_layout()
    plt.show()    

    inference_mode = "cumulative_inference"   
    workflow_outputs = []
    for l in range(7):
        w0_vec_l = np.pow(10, -np.linspace(start = -4, stop = -4+l, num = l+1))
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

    ax = plot_sigma_vs_ext_param_vec_size(workflow_outputs, int_params=['tau_b', 'tau_s'], ext_param_name='w0', metric = 'std')
    plt.tight_layout()
    plt.show()

    ax = plot_sigma_vs_ext_param_vec_size(workflow_outputs, int_params=['tau_b', 'tau_s'], ext_param_name='w0', metric = 'rel_error')
    plt.tight_layout()
    plt.show()
