# import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt # TODO: remove when replaced by plotly

import itertools
import numpy as np
from pathlib import Path

from ViscoElasticFilament_Models import X3N
from ViscoElasticFilament_Inferences import workflow_elastic_viscous_simulation, workflow_elastic_viscous_general, basinhopping_optimizer, dual_annealing_optimizer, rel_mse

import pickle
import json

def plot_model_distance_comparison(
    model_lists: Dict[int, ModelList],  # Changed from List to Dict
    reference_model_list: ModelList,
    distance_fn: Callable[[Dict[str, Any], Dict[str, Any]], float],
    int_param_name: str,
    int_params_metadata: List[Dict[str, Any]],
    ext_param_name: Optional[str] = None,
    ext_param_index: int = 0,
    title: Optional[str] = None,
    y_label: str = "Distance",
    log_scale: bool = False,
) -> go.Figure:
    """
    Plot the distance between simulations across varying internal parameters.
    
    Args:
        model_lists: Dict of ModelList objects {idx: ModelList}.
        reference_model_list: Reference ModelList to compare against.
        distance_fn: Function(sim_output_1, sim_output_2) -> float.
        int_param_name: Name of internal parameter to vary on x-axis (e.g., 'Sp4').
        int_params_metadata: List of metadata dicts, one per model_list.
        ext_param_name: Name of external parameter to filter/label by.
        ext_param_index: Index of external parameter set to plot (default 0).
        title: Plot title. If None, auto-generated.
        y_label: Label for y-axis.
        log_scale: If True, use log scale for both axes.
    
    Returns:
        plotly.graph_objects.Figure
    """
    
    # Extract internal parameter values and compute distances
    int_param_values = [metadata[int_param_name] for metadata in int_params_metadata]
    distances = []
    
    ref_model = reference_model_list.models[ext_param_index]
    ref_output = ref_model.sim_output
    
    # Iterate through dict values in sorted key order
    for idx in sorted(model_lists.keys()):
        model_list = model_lists[idx]
        model = model_list.models[ext_param_index]
        model_output = model.sim_output
        
        distance = distance_fn(model_output['value'], ref_output['value'])
        distances.append(distance)
    
    # Create figure
    fig = go.Figure()
    
    # Add scatter plot
    fig.add_trace(go.Scatter(
        x=int_param_values,
        y=distances,
        mode='lines+markers',
        name='Distance',
        line=dict(color='#0072B2', width=2),
        marker=dict(size=8, symbol='circle'),
        hovertemplate=(
            f"<b>{int_param_name}:</b> %{{x:.3e}}<br>"
            f"<b>{y_label}:</b> %{{y:.3e}}<br>"
            "<extra></extra>"
        ),
    ))
    
    # Update layout
    if title is None:
        title = f"Distance vs {int_param_name}"
    
    fig.update_layout(
        title=title,
        xaxis_title=int_param_name,
        yaxis_title=y_label,
        template='plotly_white',
        hovermode='closest',
        width=900,
        height=600,
    )
    
    if log_scale:
        fig.update_xaxes(type='log')
        fig.update_yaxes(type='log')
    
    return fig

def plot_model_distance_multi_external(
    model_lists: Dict[int, ModelList],  # Changed from List to Dict
    reference_model_list: ModelList,
    distance_fn: Callable[[Dict[str, Any], Dict[str, Any]], float],
    int_param_name: str,
    int_params_metadata: List[Dict[str, Any]],
    ext_params_list: List[Dict[str, Any]],
    ext_param_name: Optional[str] = None,
    title: Optional[str] = None,
    y_label: str = "Distance",
    log_scale: bool = False,
) -> go.Figure:
    """
    Plot distance across internal parameters for multiple external parameter values.
    
    Args:
        model_lists: Dict of ModelList objects {idx: ModelList}.
        reference_model_list: Reference ModelList to compare against.
        distance_fn: Function(sim_output_1, sim_output_2) -> float.
        int_param_name: Internal parameter name for x-axis.
        int_params_metadata: Metadata for each int param combo.
        ext_params_list: List of external parameter dicts.
        ext_param_name: External parameter name to vary across lines.
        title: Plot title. If None, auto-generated.
        y_label: Label for y-axis.
        log_scale: Use log scale for both axes.
    
    Returns:
        plotly.graph_objects.Figure
    """
    
    fig = go.Figure()
    
    # Extract internal parameter values
    int_param_values = [metadata[int_param_name] for metadata in int_params_metadata]
    
    # Get reference model outputs for all external params
    ref_models_outputs = [model.sim_output for model in reference_model_list.models]
    
    # Iterate through each external parameter set
    num_ext_params = len(ext_params_list)
    colors = [
        '#0072B2', '#E69F00', '#CC79A7', '#56B4E9',
        '#009E73', '#F0E442', '#D55E00', '#999999'
    ]
    
    for ext_idx in range(num_ext_params):
        distances = []
        
        ref_output = ref_models_outputs[ext_idx]
        
        # Iterate through dict values in sorted key order
        for idx in sorted(model_lists.keys()):
            model_list = model_lists[idx]
            model = model_list.models[ext_idx]
            model_output = model.sim_output
            distance = distance_fn(model_output['value'], ref_output['value'])
            distances.append(distance)
        
        # Create legend label
        if ext_param_name and ext_param_name in ext_params_list[ext_idx]:
            ext_value = ext_params_list[ext_idx][ext_param_name]
            label = f"{ext_param_name} = {ext_value:.3e}"
        else:
            label = f"External Set {ext_idx}"
        
        # Add trace
        fig.add_trace(go.Scatter(
            x=int_param_values,
            y=distances,
            mode='lines+markers',
            name=label,
            line=dict(color=colors[ext_idx % len(colors)], width=2),
            marker=dict(size=6, symbol='circle'),
            hovertemplate=(
                f"<b>{int_param_name}:</b> %{{x:.3e}}<br>"
                f"<b>{y_label}:</b> %{{y:.3e}}<br>"
                f"<b>{label}</b><extra></extra>"
            ),
        ))
    
    # Update layout
    if title is None:
        title = f"Distance vs {int_param_name} (Multiple External Parameters)"
    
    fig.update_layout(
        title=title,
        xaxis_title=int_param_name,
        yaxis_title=y_label,
        template='plotly_white',
        hovermode='closest',
        width=1000,
        height=650,
        legend=dict(x=0.02, y=0.98, bgcolor='rgba(255,255,255,0.8)'),
    )
    
    if log_scale:
        fig.update_xaxes(type='log')
        fig.update_yaxes(type='log')
    
    return fig

# def plot_final_output_vs_sp4_single_external(
#     model_lists: Dict[int, ModelList],
#     int_params_metadata: Dict[str, Any],
#     ext_param_index: int,
#     ext_params_list: List[float],
#     int_param_name: str = 'Sp4',
#     ext_param_name: str = 'A',
#     title: Optional[str] = None,
#     y_label: str = "X[-1]",
#     log_scale: bool = False,
# ) -> go.Figure:
#     """
#     Plot X[-1] vs Sp4 for a single external parameter (A).
    
#     Args:
#         model_lists: Dictionary mapping indices to ModelList objects
#         int_params_metadata: Metadata about internal parameters (e.g., values at each index)
#         ext_param_index: Index of the external parameter (e.g., A) to plot
#         ext_params_list: List of external parameter values for labeling
#         output_key: Key in sim_output dict (default 'X')
#         title: Plot title
#         y_label: Y-axis label
#         log_scale: Whether to use log scale for y-axis
    
#     Returns:
#         Plotly Figure
#     """
#     # Extract internal parameter values
#     int_param_values = [params_dict[int_param_name] for params_dict in int_params_metadata]    
    
#     # Extract X[-1] for all internal parameter indices at this external parameter
#     x_vals = []
#     y_vals = []
    
#     for idx in sorted(model_lists.keys()):
#         model_list = model_lists[idx]
#         if ext_param_index < len(model_list.models):
#             model = model_list.models[ext_param_index]
#             if model.sim_output is not None:
#                 output_data = model.sim_output
#                 # Handle both array and dict outputs
#                 if isinstance(output_data, dict) and 'value' in output_data:
#                     output_data = output_data['value']
                
#                 output_array = np.array(output_data)
#                 final_value = X3N(output_array) if len(output_array) > 0 else np.nan
                
#                 N = final_value.shape[0]//3
#                 x_vals.append(ext_param_values[ext_param_index])
#                 y_vals.append(final_value[N:int(2*N)][-1,0])
    
#     if not x_vals:
#         raise ValueError(f"No valid output found at ext_param_index={ext_param_index}")
    
#     # Create scatter plot
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         x=x_vals,
#         y=y_vals,
#         mode='lines+markers',
#         name=f"A = {ext_params_list[ext_param_index][ext_param_name]:.2e}",
#         hovertemplate=f"<b>{int_param_name}</b>: %{{x:.4e}}<br><b>{y_label}</b>: %{{y:.6e}}<extra></extra>",
#         line=dict(width=2),
#         marker=dict(size=8),
#     ))
    
#     fig.update_xaxes(title=int_param_name, type='log')
#     if log_scale:
#         fig.update_yaxes(title=y_label, type='log')
#     else:
#         fig.update_yaxes(title=y_label)
    
#     if title is None:
#         title = f"{y_label} vs {int_param_name} (A = {ext_params_list[ext_param_index][ext_param_name]:.2e})"
    
#     fig.update_layout(title=title, hovermode='closest')
    
#     return fig

def plot_final_output_vs_sp4_multi_external(
    model_lists: Dict[int, ModelList],
    int_params_metadata: List[Dict[str, Any]],
    ext_params_list: List[Dict[str, Any]],
    int_param_name: str = 'Sp4',
    ext_param_name: str = 'A',
    title: Optional[str] = None,
    y_label: str = "y",
    log_scale: bool = False,
    colorscale: str = 'Viridis',
) -> go.Figure:
    """
    Plot y vs A (external parameter) with multiple lines, one for each Sp4 value.
    Color-coded by internal parameter (Sp4).
    
    Args:
        model_lists: Dictionary mapping indices to ModelList objects
        int_params_metadata: List of dicts, each containing internal parameter values
        ext_params_list: List of external parameter dicts (each with 'A' key)
        int_param_name: Name of the internal parameter for coloring (e.g., 'Sp4')
        ext_param_name: Name of the external parameter on x-axis (e.g., 'A')
        title: Plot title
        y_label: Y-axis label
        log_scale: Whether to use log scale for y-axis
        colorscale: Plotly colorscale name (e.g., 'Viridis', 'Blues', 'Reds')
    
    Returns:
        Plotly Figure
    """
    # Extract internal parameter values
    int_param_values = [params_dict[int_param_name] for params_dict in int_params_metadata]
    # Extract external parameter values
    ext_param_values = [ext_dict[ext_param_name] for ext_dict in ext_params_list]

    fig = go.Figure()
    
    # Create a color scale for internal parameters
    n_internal = len(int_param_values)
    colors = px.colors.sample_colorscale(colorscale, [n / (n_internal - 1) for n in range(n_internal)])

    # Plot one line per internal parameter (Sp4)
    for int_idx in sorted(model_lists.keys()):
        x_vals = []
        y_vals = []
        
        model_list = model_lists[int_idx]
        
        # Iterate through all external parameters
        for ext_idx in range(len(model_list.models)):
            model = model_list.models[ext_idx]
            
            if model.sim_output is not None:
                output_data = model.sim_output
                # Handle both array and dict outputs
                if isinstance(output_data, dict) and 'value' in output_data:
                    output_data = output_data['value']
                
                output_array = np.array(output_data)
                
                if output_array.size > 1:
                    # Apply X3N transformation
                    X_3N = X3N(output_array)
                    N = X_3N.shape[0] // 3
                    
                    # Extract final y value: last element of [N:2*N]
                    final_y = X_3N[N:int(2*N)][-1, 0]
                    
                    x_vals.append(ext_param_values[ext_idx] * int_param_values[int_idx])
                    y_vals.append(final_y)
        
        if x_vals:
            fig.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='lines',
                name=f"{int_param_name} = {int_param_values[int_idx]:.4f}",
                hovertemplate=f"<b>{ext_param_name}</b>: %{{x:.2e}}<br><b>{y_label}</b>: %{{y:.6e}}<br><b>{int_param_name}</b>: {int_param_values[int_idx]:.4f}<extra></extra>",
                line=dict(width=2, color=colors[int_idx]),
                marker=dict(size=6, color=colors[int_idx]),
            ))
    
    fig.update_xaxes(title=ext_param_name, type='log')
    if log_scale:
        fig.update_yaxes(title=y_label, type='log')
    else:
        fig.update_yaxes(title=y_label)
    
    if title is None:
        title = f"{y_label} vs {ext_param_name} (color-coded by {int_param_name})"
    
    fig.update_layout(
        title=title,
        hovermode='closest',
        legend=dict(title=int_param_name)
    )
    
    return fig


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
        metric: 'std' for Hessian standard error (normalized by internal parameter), 
                'rel_error' for relative error
    
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
    varying_param_names = list(varying_params.keys())
    if varying_param_names:
        param_combinations = list(itertools.product(*[varying_params[name] for name in varying_param_names]))
        num_combinations = len(param_combinations)
        
        if num_combinations <= 10:
            color_palette = px.colors.qualitative.Light24[:num_combinations]
        elif num_combinations <= 24:
            color_palette = px.colors.qualitative.Light24
        else:
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
    
    # Helper function to parse task_key and extract internal parameter indices
    def parse_task_key(task_key, int_param_ranges):
        """
        Parse task_key format: "int_{int_idx}_ext_{ext_idx}" and return
        a mapping of int_param_name to its index value.
        """
        parts = task_key.split('_')
        int_idx = int(parts[1])
        ext_idx = int(parts[3])
        
        int_param_names = list(int_param_ranges.keys())
        int_param_counts = [len(int_param_ranges[name]) for name in int_param_names]
        
        multi_idx = np.unravel_index(int_idx, tuple(int_param_counts))
        
        idx_mapping = {name: multi_idx[i] for i, name in enumerate(int_param_names)}
        return idx_mapping, ext_idx
    
    # Helper function to get parameter combination tuple and ground truth values
    def get_param_combination_with_values(idx_mapping, int_params, int_param_ranges):
        """
        Extract the parameter combination tuple for int_params in the requested order.
        Returns (combo_tuple, ground_truth_dict) where ground_truth_dict maps 
        param_name -> actual parameter value.
        """
        combo = tuple(
            int_param_ranges[param_name][idx_mapping[param_name]]
            for param_name in int_params
            if param_name in varying_params
        )
        
        ground_truth_dict = {
            param_name: int_param_ranges[param_name][idx_mapping[param_name]]
            for param_name in int_params
        }
        
        return combo, ground_truth_dict
    
    # Helper function to normalize a single metric value
    def normalize_metric(metric_val, ground_truth_val):
        """Normalize metric by ground truth value."""
        if metric_val is None or not np.isfinite(metric_val):
            return None
        
        if isinstance(ground_truth_val, (list, np.ndarray)):
            gt_val = float(ground_truth_val[0]) if len(ground_truth_val) > 0 else None
        else:
            gt_val = float(ground_truth_val)
        
        if gt_val is not None and gt_val != 0:
            return metric_val / abs(gt_val)
        return None
    
    fig = go.Figure()
    legend_added = set()
    
    # Plot each internal parameter
    for param_idx, int_param_name in enumerate(int_params):
        label_to_data = {}
        
        for result in results_summary:
            task_key = result['task_key']
            idx_mapping, ext_idx = parse_task_key(task_key, int_param_ranges)
            
            # Get parameter combination and ground truth values
            param_combo, ground_truth_dict = get_param_combination_with_values(
                idx_mapping, int_params, int_param_ranges
            )
            
            # Extract metric key
            if metric == 'std':
                metric_key = f'{int_param_name}_sigma'
            elif metric == 'rel_error':
                metric_key = f'{int_param_name}_rel_error'
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            value = result.get(metric_key)
            
            if value is not None:
                if isinstance(value, (list, np.ndarray)):
                    metric_val = float(value[0]) if len(value) > 0 else None
                else:
                    metric_val = float(value)
                
                # Normalize by corresponding internal parameter (ground truth)
                if metric == 'std' and metric_val is not None:
                    ground_truth_val = ground_truth_dict[int_param_name]
                    metric_val = normalize_metric(metric_val, ground_truth_val)
                
                # Skip non-finite values
                if metric_val is not None and np.isfinite(metric_val):
                    ext_param_val = ext_param_vec[ext_idx]
                    color = combination_colors[param_combo]
                    
                    # Build label with all int_params
                    label_parts = []
                    for pname in int_params:
                        if pname in varying_params:
                            pval = ground_truth_dict[pname]
                            label_parts.append(f'{pname}={format_value(pval)}')
                        else:
                            pval = ground_truth_dict[pname]
                            label_parts.append(f'{pname}={format_value(pval)}')
                    
                    label = ', '.join(label_parts)
                    
                    if label not in label_to_data:
                        label_to_data[label] = {
                            'ext_param': [],
                            'metric': [],
                            'color': color,
                            'combo': param_combo
                        }
                    label_to_data[label]['ext_param'].append(ext_param_val)
                    label_to_data[label]['metric'].append(metric_val)
        
        # Plot each label group
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
            idx_mapping, ext_idx = parse_task_key(task_key, int_param_ranges)
            
            # Get parameter combination and ground truth values
            param_combo, ground_truth_dict = get_param_combination_with_values(
                idx_mapping, int_params, int_param_ranges
            )
            
            # Collect normalized metric values for all parameters
            normalized_metrics = []
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
                    
                    # Normalize by corresponding internal parameter
                    if metric == 'std' and metric_val is not None:
                        ground_truth_val = ground_truth_dict[int_param_name]
                        metric_val = normalize_metric(metric_val, ground_truth_val)
                    
                    if metric_val is not None and np.isfinite(metric_val):
                        normalized_metrics.append(metric_val)
                    else:
                        all_finite = False
                        break
                else:
                    all_finite = False
                    break
            
            # Compute combined metric as L2 norm of normalized metrics
            if all_finite and len(normalized_metrics) == len(int_params):
                combined_metric = np.sqrt(np.sum(np.array(normalized_metrics) ** 2))
                ext_param_val = ext_param_vec[ext_idx]
                color = combination_colors[param_combo]
                
                # Build label for combined metric
                label_parts = []
                for pname in int_params:
                    if pname in varying_params:
                        pval = ground_truth_dict[pname]
                        label_parts.append(f'{pname}={format_value(pval)}')
                    else:
                        pval = ground_truth_dict[pname]
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
        y_label = 'Normalized Sigma (Std. Error / Internal Parameter)'
        title_suffix = 'Normalized Standard Error'
        y_logscale = True
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
            x=1.02,
            y=1,
            xanchor='left',
            yanchor='top',
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
        metric: 'std' for Hessian standard error (normalized by internal parameter),
                'rel_error' for relative error (L2 norm)
    
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
        
        if num_combinations <= 10:
            color_palette = px.colors.qualitative.Light24[:num_combinations]
        elif num_combinations <= 24:
            color_palette = px.colors.qualitative.Light24
        else:
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
    
    # Helper function to parse task_key and extract internal parameter indices
    def parse_task_key_to_indices(task_key, int_param_ranges):
        """
        Parse task_key format: "int_{int_idx}_ext_{ext_idx}" and return
        mapping of int_param_name to its index value.
        """
        parts = task_key.split('_')
        int_idx = int(parts[1])
        
        int_param_names = list(int_param_ranges.keys())
        int_param_counts = [len(int_param_ranges[name]) for name in int_param_names]
        
        # Unravel the linear index into multi-dimensional indices
        multi_idx = np.unravel_index(int_idx, tuple(int_param_counts))
        idx_mapping = {name: multi_idx[i] for i, name in enumerate(int_param_names)}
        
        return idx_mapping
    
    # Helper function to get parameter combination tuple and ground truth values
    def get_param_combination_with_values(idx_mapping, int_params, int_param_ranges, varying_params):
        """
        Extract the parameter combination tuple for int_params in the requested order.
        Returns (combo_tuple, ground_truth_dict) where ground_truth_dict maps 
        param_name -> actual parameter value.
        """
        combo = tuple(
            int_param_ranges[param_name][idx_mapping[param_name]]
            for param_name in int_params
            if param_name in varying_params
        )
        
        ground_truth_dict = {
            param_name: int_param_ranges[param_name][idx_mapping[param_name]]
            for param_name in int_params
        }
        
        return combo, ground_truth_dict
    
    # Helper function to normalize a single metric value
    def normalize_metric(metric_val, ground_truth_val):
        """Normalize metric by ground truth value."""
        if metric_val is None or not np.isfinite(metric_val):
            return None
        
        if isinstance(ground_truth_val, (list, np.ndarray)):
            gt_val = float(ground_truth_val[0]) if len(ground_truth_val) > 0 else None
        else:
            gt_val = float(ground_truth_val)
        
        if gt_val is not None and gt_val != 0:
            return metric_val / abs(gt_val)
        return None
    
    # Track which combinations have been added to legend
    legend_added = set()
    
    # Plot each internal parameter
    for param_idx, int_param_name in enumerate(int_params):
        label_to_data = {}
        
        for k, workflow_output in enumerate(workflow_outputs):
            results_summary = workflow_output['results_summary']
            ext_param_ranges = workflow_output['ext_param_ranges']
            ext_vec = ext_param_ranges[ext_param_name]
            ext_vec_size = len(ext_vec)
            
            for result in results_summary:
                task_key = result['task_key']
                
                # Parse task_key to get indices
                idx_mapping = parse_task_key_to_indices(task_key, int_param_ranges)
                
                # Get parameter combination and ground truth values
                param_combo, ground_truth_dict = get_param_combination_with_values(
                    idx_mapping, int_params, int_param_ranges, varying_params
                )
                
                # Extract metric key
                if metric == 'std':
                    metric_key = f'{int_param_name}_sigma'
                elif metric == 'rel_error':
                    metric_key = f'{int_param_name}_rel_error'
                else:
                    raise ValueError(f"Unknown metric: {metric}")
                
                value = result.get(metric_key)
                
                if value is not None:
                    if isinstance(value, (list, np.ndarray)):
                        metric_val = float(value[0]) if len(value) > 0 else None
                    else:
                        metric_val = float(value)
                    
                    # Normalize by corresponding internal parameter (ground truth)
                    if metric == 'std' and metric_val is not None:
                        ground_truth_val = ground_truth_dict[int_param_name]
                        metric_val = normalize_metric(metric_val, ground_truth_val)
                    
                    # Skip non-finite values
                    if metric_val is not None and np.isfinite(metric_val):
                        color = combination_colors[param_combo]
                        
                        # Build label with all int_params
                        label_parts = []
                        for pname in int_params:
                            if pname in varying_params:
                                pval = ground_truth_dict[pname]
                                label_parts.append(f'{pname}={format_value(pval)}')
                            else:
                                pval = ground_truth_dict[pname]
                                label_parts.append(f'{pname}={format_value(pval)}')
                        
                        label = ', '.join(label_parts)
                        
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
                
                # Parse task_key to get indices
                idx_mapping = parse_task_key_to_indices(task_key, int_param_ranges)
                
                # Get parameter combination and ground truth values
                param_combo, ground_truth_dict = get_param_combination_with_values(
                    idx_mapping, int_params, int_param_ranges, varying_params
                )
                
                # Collect normalized metric values for all parameters
                normalized_metrics = []
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
                        
                        # Normalize by corresponding internal parameter
                        if metric == 'std' and metric_val is not None:
                            ground_truth_val = ground_truth_dict[int_param_name]
                            metric_val = normalize_metric(metric_val, ground_truth_val)
                        
                        if metric_val is not None and np.isfinite(metric_val):
                            normalized_metrics.append(metric_val)
                        else:
                            all_finite = False
                            break
                    else:
                        all_finite = False
                        break
                
                # Compute combined metric as L2 norm of normalized metrics
                if all_finite and len(normalized_metrics) == len(int_params):
                    combined_metric = np.sqrt(np.sum(np.array(normalized_metrics) ** 2))
                    color = combination_colors[param_combo]
                    
                    # Build label for combined metric
                    label_parts = []
                    for pname in int_params:
                        if pname in varying_params:
                            pval = ground_truth_dict[pname]
                            label_parts.append(f'{pname}={format_value(pval)}')
                        else:
                            pval = ground_truth_dict[pname]
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
        y_label = 'Normalized Sigma (Std. Error / Internal Parameter)'
        title_suffix = 'Normalized Standard Error'
        y_logscale = True
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
            x=1.02,
            y=1,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black',
            borderwidth=1
        )
    )
    
    return fig

if __name__ == "__main__":

    # --------------------------- #
    # -------- Simulations ------ #
    # --------------------------- #

    # Small logarithmic perturbations around 1
    epsilon = 0.01  # log-scale offset (in powers of 10)
    n_points = 5  # number of points on each side of 1
    log_offsets = np.linspace(-epsilon, epsilon, num=2*n_points + 1)
    Sp4_vec =  np.power(10, log_offsets)  
    int_param_ranges = {'Sp4': Sp4_vec}

    A_vec = np.pow(10, np.linspace(start=-6, stop=-1, num = 12))
    ext_param_ranges = {'A': A_vec}

    # Simulate
    simulation_output = workflow_elastic_viscous_simulation(
        int_param_ranges=int_param_ranges,
        ext_param_ranges=ext_param_ranges,
        param_keys_to_infer=['Sp4'],
        n_jobs_simulation=-1,
        checkpoint_str = "./distance",
    )

    model_lists = simulation_output['model_lists']
    int_params_metadata = simulation_output['int_params_metadata']
    ext_params_list = simulation_output['ext_params_list']

    # Plot

    # # Find the index where Sp4 = 1 (should be the middle point)
    Sp4_star = 1
    reference_index = np.argmin(np.abs(Sp4_vec - Sp4_star))
    reference_model_list = model_lists[reference_index]

    fig2 = plot_model_distance_multi_external(
        model_lists=model_lists,
        reference_model_list=reference_model_list,
        distance_fn=rel_mse,
        int_param_name='Sp4',
        int_params_metadata=int_params_metadata,
        ext_params_list=ext_params_list,
        ext_param_name='A',
        title="Distance vs Sp4 (All Amplitudes)",
        y_label="Relative L2 Error",
        log_scale=False,
    )
    fig2.show()

    log_offsets = np.linspace(-3, 3, num=7)
    Sp4_vec =  np.power(10, log_offsets)  
    int_param_ranges = {'Sp4': Sp4_vec}

    A_vec = np.pow(10, np.linspace(start=-6, stop=-1, num = 12))
    ext_param_ranges = {'A': A_vec}

    # Simulate
    simulation_output = workflow_elastic_viscous_simulation(
        int_param_ranges=int_param_ranges,
        ext_param_ranges=ext_param_ranges,
        param_keys_to_infer=['Sp4'],
        n_jobs_simulation=-1,
        checkpoint_str = "./tip",
    )
    model_lists = simulation_output['model_lists']
    int_params_metadata = simulation_output['int_params_metadata']
    ext_params_list = simulation_output['ext_params_list']    

    # All external parameters on one plot
    fig2 = plot_final_output_vs_sp4_multi_external(
        model_lists=model_lists,
        int_params_metadata=int_params_metadata,
        ext_params_list=ext_params_list,
        int_param_name='Sp4',
        colorscale='Viridis',
        log_scale=True,
    )
    fig2.show()

    exit()

    # --------------------------- #
    # --------- Inferences ------ #
    # --------------------------- #

    optimizer = dual_annealing_optimizer

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
        optimizer=optimizer,
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
    #         optimizer = optimizer,
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
        optimizer = optimizer,        
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
    #         optimizer = optimizer,    
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
        optimizer = optimizer,        
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
    #         optimizer = optimizer,    
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

    int_param_ranges = {'tau_b': [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]}
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
        optimizer = optimizer,
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
    #         optimizer = optimizer,
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

    int_param_ranges = {'tau_s': [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], 'Beta':[1.0]}
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
        optimizer = optimizer,        
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
    #         optimizer = optimizer,
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

    int_param_ranges = {'tau_b': [1e-3, 1e0, 1e3], 'tau_s':[1e-3, 1e0, 1e3], 'Beta':[1.0]}
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
        optimizer = optimizer,        
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
    #         optimizer = optimizer,
    #         elastic_params_list = elastic_params_list,
    #         viscous_params_list = viscous_params_list,
    #         inference_mode = inference_mode,
    #         checkpoint_str=checkpoint_str,
    #         ))

    # fig = plot_sigma_vs_ext_param_vec_size(workflow_outputs, int_params=['tau_b', 'tau_s'], ext_param_name='w0', metric = 'std')
    # fig.show()

    # fig = plot_sigma_vs_ext_param_vec_size(workflow_outputs, int_params=['tau_b', 'tau_s'], ext_param_name='w0', metric = 'rel_error')
    # fig.show()
