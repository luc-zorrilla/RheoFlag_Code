# import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


import numpy as np
from pathlib import Path
from ViscoElasticFilament_Inferences import workflow_elastic_viscous_general

import pickle
import json

def plot_sigma_vs_ext_param(workflow_output, int_params, ext_param_name, metric='std'):
    """
    Plot metric vs external parameter value for a single inference run.
    Multiple internal parameters plotted on same axis for comparison.
    Color-codes by internal parameter values when they vary.
    
    Also plots combined metric (L2 norm) alongside individual metrics.
    
    Args:
        workflow_output: Workflow output dict from workflow_elastic_viscous_general
        int_params: List of internal parameter names (e.g., ['Sp4', 'Beta', 'eta'])
        ext_param_name: Name of external parameter (e.g., 'A', 'w0')
        metric: 'std' for Hessian standard error, 'rel_error' for relative error
    
    Returns:
        Plotly figure object
    """
    results_summary = workflow_output['results_summary']
    ext_param_ranges = workflow_output['ext_param_ranges']
    int_param_ranges = workflow_output['int_param_ranges']
    
    ext_param_vec = ext_param_ranges[ext_param_name]
    
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
    viridis = px.colors.sequential.Viridis
    param_color_palettes = {}
    for param_name in varying_params:
        num_values = len(varying_params[param_name])
        param_color_palettes[param_name] = [
            viridis[int(i * (len(viridis) - 1) / (num_values - 1))] if num_values > 1 else viridis[0]
            for i in range(num_values)
        ]
    
    # Color palette for fixed parameters
    tab20_colors = px.colors.qualitative.Light24
    
    fig = go.Figure()
    
    # Helper function to format parameter values
    def format_value(val):
        if isinstance(val, (int, float)):
            if val >= 1e3 or (val < 1e-2 and val > 0):
                return f'{val:.1e}'
            else:
                return f'{val}'
        return str(val)
    
    # Plot each internal parameter
    for param_idx, int_param_name in enumerate(int_params):
        ext_param_scatter = []
        metric_all = []
        label_to_data = {}  # Track data grouped by unique label
        
        for result in results_summary:
            task_key = result['task_key']
            
            # Parse task_key format: "int_{int_idx}_ext_{ext_idx}"
            parts = task_key.split('_')
            int_idx = int(parts[1])
            ext_idx = int(parts[3])
            
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
                    ext_param_val = ext_param_vec[ext_idx]
                    
                    # Determine color and label based on parameter type
                    if int_param_name in varying_params:
                        # Color by internal parameter value
                        param_values = varying_params[int_param_name]
                        color = param_color_palettes[int_param_name][int_idx]
                        param_value = param_values[int_idx]
                        label = f'{int_param_name}={format_value(param_value)}'
                    else:
                        # Fixed parameter
                        color = tab20_colors[param_idx % len(tab20_colors)]
                        param_value = fixed_params[int_param_name][0]
                        label = f'{int_param_name}={format_value(param_value)}'
                    
                    # Group data by label for legend deduplication
                    if label not in label_to_data:
                        label_to_data[label] = {
                            'ext_param': [],
                            'metric': [],
                            'color': color
                        }
                    label_to_data[label]['ext_param'].append(ext_param_val)
                    label_to_data[label]['metric'].append(metric_val)
        
        # Plot each label group separately (one legend entry per label)
        for label, data in label_to_data.items():
            fig.add_trace(go.Scatter(
                x=data['ext_param'],
                y=data['metric'],
                mode='markers',
                name=label,
                marker=dict(
                    size=8,
                    color=data['color'],
                    opacity=0.6,
                    line=dict(color='black', width=0.5)
                ),
                legendgroup=label,
                showlegend=True,
                hovertemplate=f'<b>{label}</b><br>Ext param: %{{x}}<br>Metric: %{{y:.4e}}<extra></extra>'
            ))
    
    # Plot combined metric if multiple parameters
    if len(int_params) > 1:
        combined_data_by_label = {}
        
        for result in results_summary:
            task_key = result['task_key']
            
            # Parse task_key
            parts = task_key.split('_')
            int_idx = int(parts[1])
            ext_idx = int(parts[3])
            
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
                ext_param_val = ext_param_vec[ext_idx]
                
                # Determine color for combined metric
                # Priority: use first varying parameter, or first fixed parameter
                if varying_params:
                    first_varying = list(varying_params.keys())[0]
                    param_values = varying_params[first_varying]
                    color = param_color_palettes[first_varying][int_idx]
                    param_value = param_values[int_idx]
                    label = f'Combined ({first_varying}={format_value(param_value)})'
                else:
                    color = '#808080'  # Gray
                    label = 'Combined (L2 norm)'
                
                if label not in combined_data_by_label:
                    combined_data_by_label[label] = {
                        'ext_param': [],
                        'metric': [],
                        'color': color
                    }
                combined_data_by_label[label]['ext_param'].append(ext_param_val)
                combined_data_by_label[label]['metric'].append(combined_metric)
        
        # Plot combined metric with diamond marker
        for label, data in combined_data_by_label.items():
            fig.add_trace(go.Scatter(
                x=data['ext_param'],
                y=data['metric'],
                mode='markers',
                name=label,
                marker=dict(
                    size=10,
                    color=data['color'],
                    opacity=0.7,
                    symbol='diamond',
                    line=dict(color='black', width=1)
                ),
                legendgroup=label,
                showlegend=True,
                hovertemplate=f'<b>{label}</b><br>Ext param: %{{x}}<br>Combined: %{{y:.4e}}<extra></extra>'
            ))
    
    # Determine x-axis scaling
    x_logscale = ext_param_name in ['A', 'w0', 'omega']
    
    # Set axis labels and title
    if metric == 'std':
        y_label = 'Sigma (Hessian Std. Error)'
        title_suffix = 'Standard Error'
        y_logscale = False
    else:
        y_label = 'Relative Error (L2 norm)'
        title_suffix = 'Relative Error'
        y_logscale = True
    
    fig.update_layout(
        title=dict(
            text=f'{title_suffix} vs {ext_param_name}',
            font=dict(size=16, color='black')
        ),
        xaxis=dict(
            title=dict(text=f'{ext_param_name}', font=dict(size=12)),
            type='log' if x_logscale else 'linear',
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            showgrid=True
        ),
        yaxis=dict(
            title=dict(text=y_label, font=dict(size=12)),
            type='log' if y_logscale else 'linear',
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            showgrid=True
        ),
        hovermode='closest',
        template='plotly_white',
        width=1000,
        height=600,
        font=dict(size=11),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black',
            borderwidth=1
        )
    )
    
    return fig

def plot_sigma_vs_ext_param_vec_size(workflow_outputs, int_params, ext_param_name, metric='std'):
    """
    Plot metric vs size of external parameter vector for cumulative inference runs.
    Multiple internal parameters plotted on same axis for comparison.
    Color-codes by internal parameter values when they vary.
    
    Plots all individual samples (no mean/std aggregation) with improved marker styling.
    Also plots combined metric (L2 norm) alongside individual metrics.
    
    Args:
        workflow_outputs: List of workflow output dicts from workflow_elastic_viscous_general
        int_params: List of internal parameter names (e.g., ['Sp4', 'Beta', 'eta'])
        ext_param_name: Name of external parameter vector
        metric: 'std' for Hessian standard error, 'rel_error' for relative error (L2 norm)
    
    Returns:
        Plotly figure object
    """
    # Get int_param_ranges from first workflow (assumes consistent across all runs)
    int_param_ranges = workflow_outputs[0]['int_param_ranges']
    
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
    viridis = px.colors.sequential.Viridis
    param_color_palettes = {}
    for param_name in varying_params:
        num_values = len(varying_params[param_name])
        param_color_palettes[param_name] = [
            viridis[int(i * (len(viridis) - 1) / (num_values - 1))] if num_values > 1 else viridis[0]
            for i in range(num_values)
        ]
    
    # Color palette for fixed parameters
    tab20_colors = px.colors.qualitative.Light24
    
    fig = go.Figure()
    
    # Helper function to format parameter values
    def format_value(val):
        if isinstance(val, (int, float)):
            if val >= 1e3 or (val < 1e-2 and val > 0):
                return f'{val:.1e}'
            else:
                return f'{val}'
        return str(val)
    
    # Plot each internal parameter
    for param_idx, int_param_name in enumerate(int_params):
        label_to_data = {}  # Track data grouped by unique label
        
        for k, workflow_output in enumerate(workflow_outputs):
            results_summary = workflow_output['results_summary']
            ext_param_ranges = workflow_output['ext_param_ranges']
            ext_vec = ext_param_ranges[ext_param_name]
            ext_vec_size = len(ext_vec)
            
            # Extract all metric values from this run
            for result in results_summary:
                task_key = result['task_key']
                
                # Parse task_key format: "int_{int_idx}_ext_{ext_idx}"
                parts = task_key.split('_')
                int_idx = int(parts[1])
                
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
                        # Determine color and label based on parameter type
                        if int_param_name in varying_params:
                            # Color by internal parameter value
                            param_values = varying_params[int_param_name]
                            color = param_color_palettes[int_param_name][int_idx]
                            param_value = param_values[int_idx]
                            label = f'{int_param_name}={format_value(param_value)}'
                        else:
                            # Fixed parameter—use fixed color
                            color = tab20_colors[param_idx % len(tab20_colors)]
                            param_value = fixed_params[int_param_name][0]
                            label = f'{int_param_name}={format_value(param_value)}'
                        
                        # Group data by label for legend deduplication
                        if label not in label_to_data:
                            label_to_data[label] = {
                                'ext_size': [],
                                'metric': [],
                                'color': color
                            }
                        label_to_data[label]['ext_size'].append(ext_vec_size)
                        label_to_data[label]['metric'].append(metric_val)
        
        # Plot each label group separately (one legend entry per label)
        for label, data in label_to_data.items():
            fig.add_trace(go.Scatter(
                x=data['ext_size'],
                y=data['metric'],
                mode='markers',
                name=label,
                marker=dict(
                    size=8,
                    color=data['color'],
                    opacity=0.6,
                    line=dict(color='black', width=0.5)
                ),
                legendgroup=label,
                showlegend=True,
                hovertemplate=f'<b>{label}</b><br>Vector size: %{{x}}<br>Metric: %{{y:.4e}}<extra></extra>'
            ))
    
    # Plot combined metric if multiple parameters
    if len(int_params) > 1:
        combined_data_by_label = {}
        
        for k, workflow_output in enumerate(workflow_outputs):
            results_summary = workflow_output['results_summary']
            ext_param_ranges = workflow_output['ext_param_ranges']
            ext_vec = ext_param_ranges[ext_param_name]
            ext_vec_size = len(ext_vec)
            
            for result in results_summary:
                task_key = result['task_key']
                
                # Parse task_key
                parts = task_key.split('_')
                int_idx = int(parts[1])
                
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
                    
                    # Determine color for combined metric
                    # Priority: use first varying parameter, or first fixed parameter
                    if varying_params:
                        # Use first varying parameter for color
                        first_varying = list(varying_params.keys())[0]
                        param_values = varying_params[first_varying]
                        color = param_color_palettes[first_varying][int_idx]
                        param_value = param_values[int_idx]
                        label = f'Combined ({first_varying}={format_value(param_value)})'
                    else:
                        color = '#808080'  # Gray
                        label = 'Combined (L2 norm)'
                    
                    if label not in combined_data_by_label:
                        combined_data_by_label[label] = {
                            'ext_size': [],
                            'metric': [],
                            'color': color
                        }
                    combined_data_by_label[label]['ext_size'].append(ext_vec_size)
                    combined_data_by_label[label]['metric'].append(combined_metric)
        
        # Plot combined metric with diamond marker
        for label, data in combined_data_by_label.items():
            fig.add_trace(go.Scatter(
                x=data['ext_size'],
                y=data['metric'],
                mode='markers',
                name=label,
                marker=dict(
                    size=10,
                    color=data['color'],
                    opacity=0.7,
                    symbol='diamond',
                    line=dict(color='black', width=1)
                ),
                legendgroup=label,
                showlegend=True,
                hovertemplate=f'<b>{label}</b><br>Vector size: %{{x}}<br>Combined: %{{y:.4e}}<extra></extra>'
            ))
    
    # Set axis labels and title
    if metric == 'std':
        y_label = 'Sigma (Hessian Std. Error)'
        title_suffix = 'Standard Error'
        y_logscale = False
    else:
        y_label = 'Relative Error (L2 norm)'
        title_suffix = 'Relative Error'
        y_logscale = True
    
    fig.update_layout(
        title=dict(
            text=f'{title_suffix} vs {ext_param_name} Vector Size (All Samples)',
            font=dict(size=16, color='black')
        ),
        xaxis=dict(
            title=dict(text=f'Size of {ext_param_name} vector (number of points)', font=dict(size=12)),
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            showgrid=True
        ),
        yaxis=dict(
            title=dict(text=y_label, font=dict(size=12)),
            type='log' if y_logscale else 'linear',
            gridwidth=1,
            gridcolor='rgba(128, 128, 128, 0.2)',
            showgrid=True
        ),
        hovermode='closest',
        template='plotly_white',
        width=1000,
        height=600,
        font=dict(size=11),
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black',
            borderwidth=1
        )
    )
    
    return fig

# Usage
if __name__ == "__main__":

    # Bending Elasticity - Sp4
    int_param_ranges = {'Sp4': [1e-3, 1e0, 1e3]}
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

    fig = plot_sigma_vs_ext_param(workflow_output, int_params=['Sp4'], ext_param_name='A', metric='std')
    fig.show()

    fig = plot_sigma_vs_ext_param(workflow_output, int_params=['Sp4'], ext_param_name='A', metric='rel_error')
    fig.show()

    # inference_mode = "cumulative_inference"
    # workflow_outputs = []
    # for k in range(6):
    #     A_vec_k = np.pow(10, np.linspace(start = -6, stop = -6+k, num = k+1))
    #     ext_param_ranges = {'A': A_vec_k}
    #     checkpoint_str = f"./Results/BendingElasticity/BendingElasticity_{k}"

    #     workflow_outputs.append(workflow_elastic_viscous_general(
    #         int_param_ranges=int_param_ranges,
    #         ext_param_ranges=ext_param_ranges,
    #         elastic_params_list = elastic_params_list,
    #         viscous_params_list = viscous_params_list,
    #         inference_mode = inference_mode,
    #         checkpoint_str=checkpoint_str,
    #         ))
    
    # fig = plot_sigma_vs_ext_param_vec_size(workflow_outputs, int_params=['Sp4'], ext_param_name='A', metric = 'std')
    # fig.show()

    # fig = plot_sigma_vs_ext_param_vec_size(workflow_outputs, int_params=['Sp4'], ext_param_name='A', metric = 'rel_error')
    # fig.show()

    # Shear Elasticity - Beta

    int_param_ranges = {'Beta': [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]}
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

    fig = plot_sigma_vs_ext_param(workflow_output, int_params=['Beta'], ext_param_name='A', metric = 'std')
    fig.show()    

    fig = plot_sigma_vs_ext_param(workflow_output, int_params=['Beta'], ext_param_name='A', metric = 'rel_error')
    fig.show()        

    # inference_mode = "cumulative_inference"   
    # workflow_outputs = []
    # for k in range(6):
    #     A_vec_k = np.pow(10, np.linspace(start = -6, stop = -6+k, num = k+1))
    #     ext_param_ranges = {'A': A_vec_k}
    #     checkpoint_str = f"./Results/ShearElasticity/ShearElasticity_{k}"

    #     workflow_outputs.append(workflow_elastic_viscous_general(
    #         int_param_ranges=int_param_ranges,
    #         ext_param_ranges=ext_param_ranges,
    #         elastic_params_list = elastic_params_list,
    #         viscous_params_list = viscous_params_list,
    #         inference_mode = inference_mode,
    #         checkpoint_str=checkpoint_str,
    #         ))

    # fig = plot_sigma_vs_ext_param_vec_size(workflow_outputs, int_params=['Beta'], ext_param_name = 'A', metric = 'std')
    # fig.show()

    # fig = plot_sigma_vs_ext_param_vec_size(workflow_outputs, int_params=['Beta'], ext_param_name = 'A', metric = 'rel_error')
    # fig.show()    

    exit()

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

    fig = plot_sigma_vs_ext_param(workflow_output, int_params=['Sp4', 'Beta'], ext_param_name = 'A', metric = 'std')
    fig.show() 

    fig = plot_sigma_vs_ext_param(workflow_output, int_params=['Sp4', 'Beta'], ext_param_name = 'A', metric = 'rel_error')
    fig.show()    

    # inference_mode = "cumulative_inference"   
    # workflow_outputs = []
    # for k in range(6):
    #     A_vec_k = np.pow(10, np.linspace(start = -6, stop = -6+k, num = k+1))
    #     ext_param_ranges = {'A': A_vec_k}
    #     checkpoint_str = f"./Results/BendingShearElasticity/BendingShearElasticity_{k}"

    #     workflow_outputs.append(workflow_elastic_viscous_general(
    #         int_param_ranges=int_param_ranges,
    #         ext_param_ranges=ext_param_ranges,
    #         elastic_params_list = elastic_params_list,
    #         viscous_params_list = viscous_params_list,
    #         inference_mode = inference_mode,
    #         checkpoint_str=checkpoint_str,
    #         ))
    
    # fig = plot_sigma_vs_ext_param_vec_size(workflow_outputs, int_params=['Sp4', 'Beta'], ext_param_name='A', metric = 'std')
    # fig.show()

    # fig = plot_sigma_vs_ext_param_vec_size(workflow_outputs, int_params=['Sp4', 'Beta'], ext_param_name='A', metric = 'rel_error')
    # fig.show()


    # Bending Viscosity (Fixed Bending Elasticity)

    int_param_ranges = {'tau_b': [1e-3, 1e0, 1e3]}
    A_vec = [1e-6]
    w0_vec = np.pow(10, -np.linspace(start = -4, stop = 5, num = 10))
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
    
    fig = plot_sigma_vs_ext_param(workflow_output, int_params=['tau_b'], ext_param_name='w0', metric = 'std')
    fig.show()    

    fig = plot_sigma_vs_ext_param(workflow_output, int_params=['tau_b'], ext_param_name='w0', metric = 'rel_error')
    fig.show()        

    # inference_mode = "cumulative_inference"  
    # workflow_outputs = [] 
    # for l in range(10):
    #     w0_vec_l = np.pow(10, -np.linspace(start = -4, stop = -4+l, num = l+1))
    #     ext_param_ranges = {'A': A_vec, 'w0':w0_vec_l}
    #     checkpoint_str = f"./Results/BendingViscosity/BendingViscosity_{l}"

    #     workflow_outputs.append(workflow_elastic_viscous_general(
    #         int_param_ranges=int_param_ranges,
    #         ext_param_ranges=ext_param_ranges,
    #         elastic_params_list = elastic_params_list,
    #         viscous_params_list = viscous_params_list,
    #         inference_mode = inference_mode,
    #         checkpoint_str=checkpoint_str,
    #         ))

    # fig = plot_sigma_vs_ext_param_vec_size(workflow_outputs, int_params=['tau_b'], ext_param_name='w0', metric = 'std')
    # fig.show()

    # fig = plot_sigma_vs_ext_param_vec_size(workflow_outputs, int_params=['tau_b'], ext_param_name='w0', metric = 'rel_error')
    # fig.show()

    # Shear Viscosity (Fixed Bending Elasticity & Shear Elasticity)

    int_param_ranges = {'tau_s': [1e-3, 1e0, 1e3], 'Beta':[1.0]}
    A_vec = [1e-6]
    w0_vec = np.pow(10, -np.linspace(start = -4, stop = 5, num = 10))
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

    fig = plot_sigma_vs_ext_param(workflow_output, int_params=['tau_s'], ext_param_name='w0', metric = 'std')
    fig.show()    

    fig = plot_sigma_vs_ext_param(workflow_output, int_params=['tau_s'], ext_param_name='w0', metric = 'rel_error')
    fig.show()    

    # inference_mode = "cumulative_inference"   
    # workflow_outputs = []
    # for l in range(10):
    #     w0_vec_l = np.pow(10, -np.linspace(start = -4, stop = -4+l, num = l+1))
    #     ext_param_ranges = {'A': A_vec, 'w0':w0_vec_l}
    #     checkpoint_str = f"./Results/ShearViscosity/ShearViscosity_{l}"

    #     workflow_outputs.append(workflow_elastic_viscous_general(
    #         int_param_ranges=int_param_ranges,
    #         ext_param_ranges=ext_param_ranges,
    #         elastic_params_list = elastic_params_list,
    #         viscous_params_list = viscous_params_list,
    #         inference_mode = inference_mode,
    #         checkpoint_str=checkpoint_str,
    #         ))

    # fig = plot_sigma_vs_ext_param_vec_size(workflow_outputs, int_params=['tau_s'], ext_param_name='w0', metric = 'std')
    # fig.show()


    # fig = plot_sigma_vs_ext_param_vec_size(workflow_outputs, int_params=['tau_s'], ext_param_name='w0', metric = 'rel_error')
    # fig.show()


    # Bending & Shear Viscosities (Fixed Bending & Shear Elasticities)

    int_param_ranges = {'tau_b': [1.0], 'tau_s':[1e0], 'Beta':[1.0]}
    A_vec = [1e-6]
    w0_vec = np.pow(10, -np.linspace(start = -4, stop = 5, num = 10))
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

    fig = plot_sigma_vs_ext_param(workflow_output, int_params=['tau_b', 'tau_s'], ext_param_name='w0', metric = 'std')
    fig.show()    

    fig = plot_sigma_vs_ext_param(workflow_output, int_params=['tau_b', 'tau_s'], ext_param_name='w0', metric = 'rel_error')
    fig.show()    

    # inference_mode = "cumulative_inference"   
    # workflow_outputs = []
    # for l in range(10):
    #     w0_vec_l = np.pow(10, -np.linspace(start = -4, stop = -4+l, num = l+1))
    #     ext_param_ranges = {'A': A_vec, 'w0':w0_vec_l}
    #     checkpoint_str = f"./Results/BendingShearViscosity/BendingShearViscosity_{l}"

    #     workflow_outputs.append(workflow_elastic_viscous_general(
    #         int_param_ranges=int_param_ranges,
    #         ext_param_ranges=ext_param_ranges,
    #         elastic_params_list = elastic_params_list,
    #         viscous_params_list = viscous_params_list,
    #         inference_mode = inference_mode,
    #         checkpoint_str=checkpoint_str,
    #         ))

    # fig = plot_sigma_vs_ext_param_vec_size(workflow_outputs, int_params=['tau_b', 'tau_s'], ext_param_name='w0', metric = 'std')
    # fig.show()

    # fig = plot_sigma_vs_ext_param_vec_size(workflow_outputs, int_params=['tau_b', 'tau_s'], ext_param_name='w0', metric = 'rel_error')
    # fig.show()
