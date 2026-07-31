# import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import itertools
import numpy as np
from pathlib import Path
from ViscoElasticFilament_Inferences import workflow_elastic_viscous_general

import pickle
import json

def plot_sigma_vs_ext_param(workflow_output, int_params, ext_param_name, metric='std'):
    """
    Plot metric vs external parameter value for a single inference run.
    Multiple internal parameters plotted on same axis for comparison.
    Color-codes by unique combinations of internal parameter values.
    
    Also plots combined metric (L2 norm) alongside individual metrics,
    with the same color as the corresponding parameter combination.
    
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
    
    # Build color palette for parameter combinations
    # Generate all unique combinations from varying parameters
    varying_param_names = list(varying_params.keys())
    if varying_param_names:
        param_combinations = list(itertools.product(*[varying_params[name] for name in varying_param_names]))
        num_combinations = len(param_combinations)
        
        # Use a color palette with enough distinct colors
        if num_combinations <= 10:
            color_palette = px.colors.qualitative.Light24[:num_combinations]
        elif num_combinations <= 24:
            color_palette = px.colors.qualitative.Light24
        else:
            # For more combinations, cycle through palette
            color_palette = px.colors.qualitative.Light24
        
        combination_colors = {combo: color_palette[i % len(color_palette)] for i, combo in enumerate(param_combinations)}
    else:
        combination_colors = {tuple(): px.colors.qualitative.Light24[0]}
    
    # Helper function to format parameter values
    def format_value(val):
        if isinstance(val, (int, float)):
            if val >= 1e3 or (val < 1e-2 and val > 0):
                return f'{val:.1e}'
            else:
                return f'{val}'
        return str(val)
    
    # Helper function to get internal parameter indices from task_key
    def parse_task_key(task_key, int_param_ranges):
        """
        Parse task_key format: "int_{int_idx}_ext_{ext_idx}" and return
        a mapping of int_param_name to its index value.
        """
        parts = task_key.split('_')
        int_idx = int(parts[1])
        ext_idx = int(parts[3])
        
        # Convert linear int_idx to multi-dimensional indices
        int_param_names = list(int_param_ranges.keys())
        int_param_counts = [len(int_param_ranges[name]) for name in int_param_names]
        
        # Unravel the linear index into multi-dimensional indices
        multi_idx = np.unravel_index(int_idx, tuple(int_param_counts))
        
        idx_mapping = {name: multi_idx[i] for i, name in enumerate(int_param_names)}
        return idx_mapping, ext_idx
    
    # Helper function to get parameter combination tuple
    def get_param_combination(idx_mapping, int_params, int_param_ranges):
        """Extract the parameter combination tuple for int_params in the requested order."""
        combo = tuple(
            int_param_ranges[param_name][idx_mapping[param_name]]
            for param_name in int_params
            if param_name in varying_params
        )
        return combo
    
    fig = go.Figure()
    
    # Track which combinations have been added to legend
    legend_added = set()
    
    # Plot each internal parameter
    for param_idx, int_param_name in enumerate(int_params):
        label_to_data = {}  # Track data grouped by unique label
        
        for result in results_summary:
            task_key = result['task_key']
            
            # Parse task_key to get indices for all internal parameters
            idx_mapping, ext_idx = parse_task_key(task_key, int_param_ranges)
            
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
                    
                    # Get the parameter combination for color coding
                    param_combo = get_param_combination(idx_mapping, int_params, int_param_ranges)
                    color = combination_colors[param_combo]
                    
                    # Build label with all int_params in the combination
                    label_parts = []
                    for pname in int_params:
                        if pname in varying_params:
                            pidx = idx_mapping[pname]
                            pval = varying_params[pname][pidx]
                            label_parts.append(f'{pname}={format_value(pval)}')
                        else:
                            pval = fixed_params[pname][0]
                            label_parts.append(f'{pname}={format_value(pval)}')
                    
                    label = ', '.join(label_parts)
                    
                    # Group data by label for legend deduplication
                    if label not in label_to_data:
                        label_to_data[label] = {
                            'ext_param': [],
                            'metric': [],
                            'color': color,
                            'combo': param_combo
                        }
                    label_to_data[label]['ext_param'].append(ext_param_val)
                    label_to_data[label]['metric'].append(metric_val)
        
        # Plot each label group separately (one legend entry per label)
        for label, data in label_to_data.items():
            combo_key = data['combo']
            show_legend = combo_key not in legend_added
            if show_legend:
                legend_added.add(combo_key)
            
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
                showlegend=show_legend,
                hovertemplate=f'<b>{label}</b><br>Ext param: %{{x}}<br>Metric: %{{y:.4e}}<extra></extra>'
            ))
    
    # Plot combined metric if multiple parameters
    if len(int_params) > 1:
        combined_data_by_combo = {}
        
        for result in results_summary:
            task_key = result['task_key']
            
            # Parse task_key
            idx_mapping, ext_idx = parse_task_key(task_key, int_param_ranges)
            
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
                
                # Get parameter combination and color
                param_combo = get_param_combination(idx_mapping, int_params, int_param_ranges)
                color = combination_colors[param_combo]
                
                # Build label for combined metric
                label_parts = []
                for pname in int_params:
                    if pname in varying_params:
                        pidx = idx_mapping[pname]
                        pval = varying_params[pname][pidx]
                        label_parts.append(f'{pname}={format_value(pval)}')
                    else:
                        pval = fixed_params[pname][0]
                        label_parts.append(f'{pname}={format_value(pval)}')
                
                label = 'Combined (' + ', '.join(label_parts) + ')'
                
                if param_combo not in combined_data_by_combo:
                    combined_data_by_combo[param_combo] = {
                        'ext_param': [],
                        'metric': [],
                        'color': color,
                        'label': label
                    }
                combined_data_by_combo[param_combo]['ext_param'].append(ext_param_val)
                combined_data_by_combo[param_combo]['metric'].append(combined_metric)
        
        # Plot combined metric with diamond marker
        for param_combo, data in combined_data_by_combo.items():
            fig.add_trace(go.Scatter(
                x=data['ext_param'],
                y=data['metric'],
                mode='markers',
                name=data['label'],
                marker=dict(
                    size=10,
                    color=data['color'],
                    opacity=0.7,
                    symbol='diamond',
                    line=dict(color='black', width=1)
                ),
                legendgroup=data['label'],
                showlegend=True,
                hovertemplate=f'<b>{data["label"]}</b><br>Ext param: %{{x}}<br>Combined: %{{y:.4e}}<extra></extra>'
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
    Color-codes by unique combinations of internal parameter values.
    
    Plots all individual samples (no mean/std aggregation) with improved marker styling.
    Also plots combined metric (L2 norm) alongside individual metrics,
    with the same color as the corresponding parameter combination.
    
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
    
    # Build color palette for parameter combinations
    varying_param_names = list(varying_params.keys())
    if varying_param_names:
        param_combinations = list(itertools.product(*[varying_params[name] for name in varying_param_names]))
        num_combinations = len(param_combinations)
        
        # Use a color palette with enough distinct colors
        if num_combinations <= 10:
            color_palette = px.colors.qualitative.Light24[:num_combinations]
        elif num_combinations <= 24:
            color_palette = px.colors.qualitative.Light24
        else:
            # For more combinations, cycle through palette
            color_palette = px.colors.qualitative.Light24
        
        combination_colors = {combo: color_palette[i % len(color_palette)] for i, combo in enumerate(param_combinations)}
    else:
        combination_colors = {tuple(): px.colors.qualitative.Light24[0]}
    
    fig = go.Figure()
    
    # Helper function to format parameter values
    def format_value(val):
        if isinstance(val, (int, float)):
            if val >= 1e3 or (val < 1e-2 and val > 0):
                return f'{val:.1e}'
            else:
                return f'{val}'
        return str(val)
    
    # Helper function to get parameter combination tuple
    def get_param_combination(int_idx, int_params, int_param_ranges, varying_params):
        """
        Extract the parameter combination tuple for int_params in the requested order.
        int_idx is the linear index into the int_param grid.
        """
        int_param_names = list(int_param_ranges.keys())
        int_param_counts = [len(int_param_ranges[name]) for name in int_param_names]
        
        # Unravel the linear index into multi-dimensional indices
        multi_idx = np.unravel_index(int_idx, tuple(int_param_counts))
        idx_mapping = {name: multi_idx[i] for i, name in enumerate(int_param_names)}
        
        # Build combo tuple with only varying params in the int_params order
        combo = tuple(
            int_param_ranges[param_name][idx_mapping[param_name]]
            for param_name in int_params
            if param_name in varying_params
        )
        return combo, idx_mapping
    
    # Track which combinations have been added to legend
    legend_added = set()
    
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
                        # Get parameter combination and color
                        param_combo, idx_mapping = get_param_combination(int_idx, int_params, int_param_ranges, varying_params)
                        color = combination_colors[param_combo]
                        
                        # Build label with all int_params in the combination
                        label_parts = []
                        for pname in int_params:
                            if pname in varying_params:
                                pidx = idx_mapping[pname]
                                pval = varying_params[pname][pidx]
                                label_parts.append(f'{pname}={format_value(pval)}')
                            else:
                                pval = fixed_params[pname][0]
                                label_parts.append(f'{pname}={format_value(pval)}')
                        
                        label = ', '.join(label_parts)
                        
                        # Group data by label for legend deduplication
                        if label not in label_to_data:
                            label_to_data[label] = {
                                'ext_size': [],
                                'metric': [],
                                'color': color,
                                'combo': param_combo
                            }
                        label_to_data[label]['ext_size'].append(ext_vec_size)
                        label_to_data[label]['metric'].append(metric_val)
        
        # Plot each label group separately (one legend entry per label)
        for label, data in label_to_data.items():
            combo_key = data['combo']
            show_legend = combo_key not in legend_added
            if show_legend:
                legend_added.add(combo_key)
            
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
                showlegend=show_legend,
                hovertemplate=f'<b>{label}</b><br>Vector size: %{{x}}<br>Metric: %{{y:.4e}}<extra></extra>'
            ))
    
    # Plot combined metric if multiple parameters
    if len(int_params) > 1:
        combined_data_by_combo = {}
        
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
                    
                    # Get parameter combination and color
                    param_combo, idx_mapping = get_param_combination(int_idx, int_params, int_param_ranges, varying_params)
                    color = combination_colors[param_combo]
                    
                    # Build label for combined metric
                    label_parts = []
                    for pname in int_params:
                        if pname in varying_params:
                            pidx = idx_mapping[pname]
                            pval = varying_params[pname][pidx]
                            label_parts.append(f'{pname}={format_value(pval)}')
                        else:
                            pval = fixed_params[pname][0]
                            label_parts.append(f'{pname}={format_value(pval)}')
                    
                    label = 'Combined (' + ', '.join(label_parts) + ')'
                    
                    if param_combo not in combined_data_by_combo:
                        combined_data_by_combo[param_combo] = {
                            'ext_size': [],
                            'metric': [],
                            'color': color,
                            'label': label
                        }
                    combined_data_by_combo[param_combo]['ext_size'].append(ext_vec_size)
                    combined_data_by_combo[param_combo]['metric'].append(combined_metric)
        
        # Plot combined metric with diamond marker
        for param_combo, data in combined_data_by_combo.items():
            fig.add_trace(go.Scatter(
                x=data['ext_size'],
                y=data['metric'],
                mode='markers',
                name=data['label'],
                marker=dict(
                    size=10,
                    color=data['color'],
                    opacity=0.7,
                    symbol='diamond',
                    line=dict(color='black', width=1)
                ),
                legendgroup=data['label'],
                showlegend=True,
                hovertemplate=f'<b>{data["label"]}</b><br>Vector size: %{{x}}<br>Combined: %{{y:.4e}}<extra></extra>'
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

    # Bending & Shear Elasticities - Sp4, Beta

    int_param_ranges = {'Sp4': [1e-3, 1e0, 1e3], 'Beta': [1e-3, 1e0, 1e3]}
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

    exit()

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
