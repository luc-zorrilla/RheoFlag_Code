# import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import itertools
from pathlib import Path
import pickle
import json

import numpy as np
from scipy.spatial.distance import directed_hausdorff

from ViscoElasticFilament_Models import X3N
from ViscoElasticFilament_Inferences import workflow_elastic_viscous_simulation, workflow_elastic_viscous_general, basinhopping_optimizer, dual_annealing_optimizer, rel_mse

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

def plot_output_trajectory_subplots_int_param(
    model_lists: Dict[int, ModelList],
    int_params_metadata: List[Dict[str, Any]],
    ext_params_list: List[Dict[str, Any]],
    int_param_name: str = 'Sp4',
    ext_param_name: str = 'A',
    title: Optional[str] = None,
    x_label: str = "x",
    y_label: str = "y",
    log_scale: bool = False,
    colorscale: str = 'Viridis',
) -> go.Figure:
    """
    Plot phase-space trajectories (x vs y) in subplots, one per internal parameter.
    External parameter values are represented as a colormap along each trace.
    
    Args:
        model_lists: Dictionary mapping indices to ModelList objects
        int_params_metadata: List of dicts, each containing internal parameter values
        ext_params_list: List of external parameter dicts (each with 'A' key)
        int_param_name: Name of the internal parameter for subplots (e.g., 'Sp4')
        ext_param_name: Name of the external parameter for coloring (e.g., 'A')
        title: Plot title
        x_label: X-axis label
        y_label: Y-axis label
        log_scale: Whether to use log scale for axes
        colorscale: Plotly colorscale name (e.g., 'Viridis', 'Blues', 'Reds')
    
    Returns:
        Plotly Figure with subplots
    """
    # Extract internal and external parameter values
    int_param_values = [params_dict[int_param_name] for params_dict in int_params_metadata]
    ext_param_values = [ext_dict[ext_param_name] for ext_dict in ext_params_list]
    
    n_internal = len(int_param_values)
    
    # Create subplots: one per internal parameter
    fig = make_subplots(
        rows=1, cols=n_internal,
        subplot_titles=[f"{int_param_name} = {val:.4f}" for val in int_param_values],
        shared_yaxes=True,
    )
    
    if len(ext_param_values) > 1:
        # Normalize external parameter values for colormap
        ext_param_min = min(ext_param_values)
        ext_param_max = max(ext_param_values)
        ext_param_norm = [
            (val - ext_param_min) / (ext_param_max - ext_param_min) 
            for val in ext_param_values
        ]
    else: # Only one external parameter
        ext_param_norm = [0]

    # Get colors for external parameters
    colors_ext = px.colors.sample_colorscale(
        colorscale, 
        ext_param_norm
    )

    # Plot one trace per external parameter in each subplot
    for int_idx in sorted(model_lists.keys()):
        model_list = model_lists[int_idx]
        int_param_val = int_param_values[int_idx]
        col = int_idx + 1
        
        # Iterate through all external parameters
        for ext_idx in range(len(model_list.models)):
            model = model_list.models[ext_idx]
            ext_param_val = ext_param_values[ext_idx]
            
            if model.sim_output is not None:
                output_data = model.sim_output
                # Handle both array and dict outputs
                if isinstance(output_data, dict) and 'value' in output_data:
                    output_data = output_data['value']
                
                output_array = np.array(output_data)
                
                if output_array.size > 1:
                    output_array = np.atleast_1d(output_array)
                    
                    # Apply X3N transformation
                    X_3N = X3N(output_array)
                    N = X_3N.shape[0] // 3
                    
                    # Extract x and y trajectories
                    x_trajectory = X_3N[:N, 0] / N
                    y_trajectory = X_3N[N:int(2*N), 0] / N

                    if 'Beta' in int_param_name:
                        y_trajectory *= 1
                    
                    fig.add_trace(
                        go.Scatter(
                            x=x_trajectory,
                            y=y_trajectory,
                            mode='lines+markers',
                            name=f"{ext_param_name} = {ext_param_val:.2e}",
                            legendgroup=ext_idx,
                            showlegend=(int_idx == 0),  # Show legend only for first subplot
                            hovertemplate=(
                                f"<b>{x_label}</b>: %{{x:.6e}}<br>"
                                f"<b>{y_label}</b>: %{{y:.6e}}<br>"
                                f"<b>{int_param_name}</b>: {int_param_val:.4f}<br>"
                                f"<b>{ext_param_name}</b>: {ext_param_val:.2e}<extra></extra>"
                            ),
                            line=dict(width=2, color=colors_ext[ext_idx]),
                            marker=dict(size=4, color=colors_ext[ext_idx]),
                        ),
                        row=1, col=col
                    )
    
    # Update axes
    for col in range(1, n_internal + 1):
        fig.update_xaxes(title_text=x_label, row=1, col=col)
        # fig.update_yaxes(scaleanchor=f"x{col}" if col > 1 else "x", scaleratio=1, row=1, col=col)
    
    fig.update_yaxes(title_text=y_label, row=1, col=1)
    
    if log_scale:
        for col in range(1, n_internal + 1):
            fig.update_xaxes(type='log', row=1, col=col)
            fig.update_yaxes(type='log', row=1, col=col)
    
    if title is None:
        title = f"Phase-Space Trajectories by {int_param_name} (color-coded by {ext_param_name})"
    
    fig.update_layout(
        title=title,
        hovermode='closest',
        legend=dict(title=ext_param_name),
        height=400,
        width=400*n_internal,
    )
    
    return fig

def plot_output_trajectory_subplots_ext_param(
    model_lists: Dict[int, ModelList],
    int_params_metadata: List[Dict[str, Any]],
    ext_params_list: List[Dict[str, Any]],
    int_param_name: str = 'Sp4',
    ext_param_name: str = 'A',
    title: Optional[str] = None,
    x_label: str = "x",
    y_label: str = "y",
    log_scale: bool = False,
    colorscale: str = 'Viridis',
) -> go.Figure:
    """
    Plot phase-space trajectories (x vs y) in subplots, one per external parameter.
    Internal parameter values are represented as a colormap along each trace.
    
    Args:
        model_lists: Dictionary mapping indices to ModelList objects
        int_params_metadata: List of dicts, each containing internal parameter values
        ext_params_list: List of external parameter dicts (each with 'A' key)
        int_param_name: Name of the internal parameter for coloring (e.g., 'Sp4')
        ext_param_name: Name of the external parameter for subplots (e.g., 'A')
        title: Plot title
        x_label: X-axis label
        y_label: Y-axis label
        log_scale: Whether to use log scale for axes
        colorscale: Plotly colorscale name (e.g., 'Viridis', 'Blues', 'Reds')
    
    Returns:
        Plotly Figure with subplots
    """
    # Extract internal and external parameter values
    int_param_values = [params_dict[int_param_name] for params_dict in int_params_metadata]
    ext_param_values = [ext_dict[ext_param_name] for ext_dict in ext_params_list]
    
    n_external = len(ext_param_values)
    
    # Create subplots: one per external parameter
    fig = make_subplots(
        rows=1, cols=n_external,
        subplot_titles=[f"{ext_param_name} = {val:.2e}" for val in ext_param_values],
        shared_yaxes=True,
    )
    
    # Normalize internal parameter values for colormap
    int_param_min = min(int_param_values)
    int_param_max = max(int_param_values)
    int_param_norm = [
        (val - int_param_min) / (int_param_max - int_param_min) 
        for val in int_param_values
    ]
    
    # Get colors for internal parameters
    colors_int = px.colors.sample_colorscale(
        colorscale, 
        int_param_norm
    )
    
    # Plot one trace per internal parameter in each subplot
    for int_idx in sorted(model_lists.keys()):
        model_list = model_lists[int_idx]
        int_param_val = int_param_values[int_idx]
        
        # Iterate through all external parameters
        for ext_idx in range(len(model_list.models)):
            model = model_list.models[ext_idx]
            ext_param_val = ext_param_values[ext_idx]
            col = ext_idx + 1
            
            if model.sim_output is not None:
                output_data = model.sim_output
                # Handle both array and dict outputs
                if isinstance(output_data, dict) and 'value' in output_data:
                    output_data = output_data['value']
                
                output_array = np.array(output_data)
                
                if output_array.size > 1:
                    output_array = np.atleast_1d(output_array)
                    
                    # Apply X3N transformation
                    X_3N = X3N(output_array)
                    N = X_3N.shape[0] // 3
                    
                    # Extract x and y trajectories
                    x_trajectory = X_3N[:N, 0] / N
                    y_trajectory = X_3N[N:int(2*N), 0] / N

                    if 'Beta' in int_param_name:
                        y_trajectory /= ext_param_val
                    
                    fig.add_trace(
                        go.Scatter(
                            x=x_trajectory,
                            y=y_trajectory,
                            mode='lines+markers',
                            name=f"{int_param_name} = {int_param_val:.4f}",
                            legendgroup=int_idx,
                            showlegend=(ext_idx == 0),  # Show legend only for first subplot
                            hovertemplate=(
                                f"<b>{x_label}</b>: %{{x:.6e}}<br>"
                                f"<b>{y_label}</b>: %{{y:.6e}}<br>"
                                f"<b>{int_param_name}</b>: {int_param_val:.4f}<br>"
                                f"<b>{ext_param_name}</b>: {ext_param_val:.2e}<extra></extra>"
                            ),
                            line=dict(width=2, color=colors_int[int_idx]),
                            marker=dict(size=4, color=colors_int[int_idx]),
                        ),
                        row=1, col=col
                    )
    
    # Update axes
    for col in range(1, n_external + 1):
        fig.update_xaxes(title_text=x_label, row=1, col=col)
        # fig.update_yaxes(scaleanchor=f"x{col}" if col > 1 else "x", scaleratio=1, row=1, col=col)
    
    fig.update_yaxes(title_text=y_label, row=1, col=1)
    
    if log_scale:
        for col in range(1, n_external + 1):
            fig.update_xaxes(type='log', row=1, col=col)
            fig.update_yaxes(type='log', row=1, col=col)
    
    if title is None:
        title = f"Phase-Space Trajectories by {ext_param_name} (color-coded by {int_param_name})"
    
    fig.update_layout(
        title=title,
        hovermode='closest',
        legend=dict(title=int_param_name),
        height=400,
        width=400*n_external,
    )
    
    return fig

def plot_final_output_vs_ext_param_color_int_param(
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
                    
                    if 'Beta' in int_param_name:
                        final_x = ext_param_values[ext_idx] / int_param_values[int_idx]
                    elif 'Sp4' in int_param_name:
                        final_x = ext_param_values[ext_idx] * int_param_values[int_idx]
                    
                    x_vals.append(final_x)
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

def plot_fig_S1(
    model_lists: Dict[int, ModelList],
    int_params_metadata: List[Dict[str, Any]],
    ext_params_list: List[Dict[str, Any]],
    int_param_name: str = 'Sp4',
    ext_param_name: str = 'A',
    title: Optional[str] = None,
    x_label: str = "x",
    y_label: str = "y",
    log_scale: bool = False,
    colorscale: str = 'Viridis',
) -> go.Figure:
    """
    Args:
        model_lists: Dictionary mapping indices to ModelList objects
        int_params_metadata: List of dicts, each containing internal parameter values
        ext_params_list: List of external parameter dicts (each with 'A' key)
        int_param_name: Name of the internal parameter for subplots (e.g., 'Sp4')
        ext_param_name: Name of the external parameter for coloring (e.g., 'A')
        title: Plot title
        x_label: X-axis label
        y_label: Y-axis label
        log_scale: Whether to use log scale for axes
        colorscale: Plotly colorscale name (e.g., 'Viridis', 'Blues', 'Reds')
    
    Returns:
        Plotly Figure with subplots
    """
    # Extract internal and external parameter values
    int_param_values = [params_dict[int_param_name] for params_dict in int_params_metadata]
    ext_param_values = [ext_dict[ext_param_name] for ext_dict in ext_params_list]
    
    n_internal = len(int_param_values)
    
    # Create plot
    fig = go.Figure()

    # Second figure: L2 error between simulation and analytical solution for small uniform vertical flow
    fig2 = go.Figure()
    
    if len(int_param_values) > 1:
        # Normalize internal parameter values for colormap
        int_param_min = min(int_param_values)
        int_param_max = max(int_param_values)
        int_param_norm = [
            (val - int_param_min) / (int_param_max - int_param_min) 
            for val in int_param_values
        ]
    else: # Only one internal parameter
        int_param_norm = [0]

    # Get colors for external parameters
    colors_int = px.colors.sample_colorscale(
        colorscale, 
        int_param_norm,
    )

    N_array = []
    error_array = []
    # Plot one trace per external parameter in each subplot
    for int_idx in sorted(model_lists.keys()):
        model_list = model_lists[int_idx]
        int_param_val = int_param_values[int_idx]
        
        # Iterate through all external parameters
        for ext_idx in range(len(model_list.models)):
            model = model_list.models[ext_idx]
            ext_param_val = ext_param_values[ext_idx]
            
            if model.sim_output is not None:
                output_data = model.sim_output
                # Handle both array and dict outputs
                if isinstance(output_data, dict) and 'value' in output_data:
                    output_data = output_data['value']
                
                output_array = np.array(output_data)
                
                if output_array.size > 1:
                    output_array = np.atleast_1d(output_array)
                    
                    # Apply X3N transformation
                    X_3N = X3N(output_array)
                    N = X_3N.shape[0] // 3    
                    N_array.append(N)                
                    
                    # Extract x and y trajectories
                    x_trajectory = X_3N[:N, 0]/N
                    y_trajectory = X_3N[N:int(2*N), 0]/N

                    # Analytical Equilbrium Profile
                    n_eq = 1000
                    X_3N_eq = CheckEquilibrium(N, model.ext_params['A'], model.int_params['gamma'], model.int_params['Sp4'], n_L = model.int_params['n_L'], Lambdas=model.ext_params['Lambdas'], conditions = "vertical_flow_uniform", n_eq = n_eq)
                    
                    x_eq_trajectory = X_3N_eq[:n_eq, 0][X_3N_eq[:n_eq,0]<=N-1]/N
                    y_eq_trajectory = X_3N_eq[n_eq:2*n_eq,0][X_3N_eq[:n_eq,0]<=N-1]/N

                    # Concatenate x and y:
                    x_traj = np.array((x_trajectory, y_trajectory)).reshape((2,-1))
                    x_eq_traj = np.array((x_eq_trajectory, y_eq_trajectory)).reshape((2,-1))

                    # ============================================================
                    # Approach 1: Hausdorff Distance
                    # ============================================================
                    d1 = directed_hausdorff(x_traj.T, x_eq_traj.T)[0]
                    d2 = directed_hausdorff(x_eq_traj.T, x_traj.T)[0]
                    hausdorff_error = max(d1, d2)
                    error_array.append(hausdorff_error)

                    if 'Beta' in int_param_name:
                        y_trajectory *= 1
                    
                    fig.add_trace(
                        go.Scatter(
                            x=x_trajectory,
                            y=y_trajectory,
                            mode='lines+markers',
                            hovertemplate=(
                                f"<b>{x_label}</b>: %{{x:.6e}}<br>"
                                f"<b>{y_label}</b>: %{{y:.6e}}<br>"
                                f"<b>{int_param_name}</b>: {int_param_val:.4f}<br>"
                                f"<b>{ext_param_name}</b>: {ext_param_val:.2e}<extra></extra>"
                            ),
                            line=dict(width=2, color=colors_int[int_idx]),
                            marker=dict(size=4, color=colors_int[int_idx]),
                        ),
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=x_eq_trajectory,
                            y=y_eq_trajectory,
                            mode='lines',
                            hovertemplate=(
                                f"<b>{x_label}</b>: %{{x:.6e}}<br>"
                                f"<b>{y_label}</b>: %{{y:.6e}}<br>"
                                f"<b>{int_param_name}</b>: {int_param_val:.4f}<br>"
                                f"<b>{ext_param_name}</b>: {ext_param_val:.2e}<extra></extra>"
                            ),
                            line=dict(width=2, color="black"),
                        ),
                    )       

    N_array = np.array(N_array)
    error_array = np.array(error_array)
    fig2.add_trace(
        go.Scatter(
            x = 1/N_array,
            y = error_array,
            mode='markers+lines',
            line=dict(width=2, color="black"),
            marker=dict(size=4, color="black"),
        )
    )
    fig2.update_xaxes(type = "log")
    fig2.update_yaxes(type = "log")
    fig2.show()
    
    # Update axes
    fig.update_xaxes(title_text=x_label)
    fig.update_yaxes(title_text=y_label)
    
    if log_scale:
        fig.update_xaxes(type='log')
        fig.update_yaxes(type='log')
    
    if title is None:
        title = f"Phase-Space Trajectories color-coded by {int_param_name})"
    
    fig.update_layout(
        title=title,
        hovermode='closest',
        legend=dict(title=int_param_name),
        height=800,
        width=1400,
    )
    
    return fig    


def CheckEquilibrium(N, A, gamma, Sp4, n_L = [0,0], Lambdas=[[0,0]], conditions = "None", n_eq = 1000):

    """ Returns a figure with equilibrium position of a shape dynamics and computes analytical solution for small deflection: 
    - vertical point force at tip - "vertical_point_tip"
    - vertical density force on tip - "vertical_density_tip"
    - vertical uniform density force along the beam - "vertical_density_uniform"
    - vertical uniform flow - "vertical_flow_uniform"

    All analytical solutions are computed in the case of an initial horizontal beam, assuming equilibrium and small deflection.
    References for analytical solutions: [Felgner et. al. , Journal of Cell Science, 1996]
    
    Remark: total filament length = N but only N-1 segments are represented (N first points). 

    Returns X_eq with 2 * n_eq points:
        - n_eq points for x = s (small deflection approximation), such that s in [0,N] -> [0,L]
        - n_eq points for y, 
    """

    ## Analytical equilibrium solution
    X_eq = np.zeros((2*n_eq))
    theta_eq = np.zeros((n_eq))

    X_eq[:n_eq] = np.linspace(start = 0, stop = N, num = n_eq) # Arclength = horizontal position in the small amplitude regime
    x_eq_neq = N

    # For a vertical point force at distal end
    if conditions == "vertical_point_tip":
        F = n_L[1]
        X_eq[n_eq:] = (3 * (X_eq[:n_eq]/(N))**2 - (X_eq[:n_eq]/(N))**3 ) * F * ((N)**3) / 6
        y_eq_neq = (3 * (x_eq_neq/(N))**2 - (x_eq_neq/(N))**3 ) * F * ((N)**3) / 6

    # For a density force at distal segment
    elif conditions == "vertical_density_tip":
        F = Lambdas[-1][1]
        X_eq[n_eq:] = (3 * (X_eq[:n_eq]/N)**2 - (X_eq[:n_eq]/N)**3) * F * (N**3) / 6
        y_eq_neq = (3 * (x_eq_neq/N)**2 - (x_eq_neq/N)**3) * F * (N**3) / 6

    # For a uniform vertical force
    elif conditions == "vertical_density_uniform":
        f = Lambdas[0][1]
        X_eq[n_eq:] = ( (X_eq[:n_eq]/N)**4 - 4*(X_eq[:n_eq]/N)**3 + 6*(X_eq[:n_eq]/N)**2 ) * f * (N**4) / 24
        y_eq_neq = ( (x_eq_neq/N)**4 - 4*(x_eq_neq/N)**3 + 6*(x_eq_neq/N)**2 ) * f * (N**4) / 24

    # For a uniform small vertical flow
    elif conditions == "vertical_flow_uniform":
        X_eq[n_eq:] = ( (X_eq[:n_eq]/N)**4 - 4*(X_eq[:n_eq]/N)**3 + 6*(X_eq[:n_eq]/N)**2 ) * A * gamma * Sp4 *(N**4) / 24
        y_eq_neq = ( (x_eq_neq/N)**4 - 4*(x_eq_neq/N)**3 + 6*(x_eq_neq/N)**2 ) * A * gamma * Sp4 *(N**4) / 24
        
    else:
        print("No condition for exact solution has been specified.")
        return NameError
    
    theta_eq[:-1] = np.arctan2(X_eq[n_eq+1:], X_eq[1:n_eq])
    theta_eq[-1] = np.arctan2(y_eq_neq, x_eq_neq)
    X_3N_eq = np.vstack((X_eq.reshape((-1,1)), theta_eq.reshape(-1,1))) # Thetas are filled to zero here!
    return X_3N_eq


if __name__ == "__main__":
    
    # ---------------------------------- #
    # -------- I. Forward Problem ------ #
    # ---------------------------------- #

    # -------------------------------------------- #
    # Figure S1: Benchmark of analytical solutions #
    # -------------------------------------------- #

    N_vec =  np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100])
    int_param_ranges = {'N': N_vec}

    A_vec = np.array([1e-6, 1e-10]) # "A" quantifies the strength of the flow compared to the elastic forces, on one segment. It does not guarantee no movement at the whole filament scale, though. TODO: check Deborah number
    ext_param_ranges = {'A': A_vec}

    # Simulate
    simulation_output = workflow_elastic_viscous_simulation(
        int_param_ranges=int_param_ranges,
        ext_param_ranges=ext_param_ranges,
        param_keys_to_infer=[],
        n_jobs_simulation=-1,
        checkpoint_str = "./analytical_solution_constant_vertical_flow",
    )

    model_lists = simulation_output['model_lists']
    int_params_metadata = simulation_output['int_params_metadata']
    ext_params_list = simulation_output['ext_params_list']

    fig = plot_fig_S1(
        model_lists, int_params_metadata, ext_params_list,
        int_param_name='N', ext_param_name='A',
        x_label='x', y_label='y',
        colorscale='Viridis', log_scale=False
    )
    fig.write_image("Figures/analytical_solution_constant_vertical_flow.svg")
    fig.write_html("Figures/analytical_solution_constant_vertical_flow.html")
    fig.show()

if __name__ is None:
    
    # --------------------------- #
    # -------- Simulations ------ #
    # --------------------------- #

    # Bending Elasticity

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
        checkpoint_str = "./bending_distance",
    )

    model_lists = simulation_output['model_lists']
    int_params_metadata = simulation_output['int_params_metadata']
    ext_params_list = simulation_output['ext_params_list']

    # Plot

    # # Find the index where Sp4 = 1 (should be the middle point)
    Sp4_star = 1
    reference_index = np.argmin(np.abs(Sp4_vec - Sp4_star))
    reference_model_list = model_lists[reference_index]

    fig = plot_model_distance_multi_external(
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
    fig.write_image("Figures/loss_vs_Sp4_color_A.svg")
    fig.write_html("Figures/loss_vs_Sp4_color_A.html")
    fig.show()

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
        checkpoint_str = "./bending_tip",
    )
    model_lists = simulation_output['model_lists']
    int_params_metadata = simulation_output['int_params_metadata']
    ext_params_list = simulation_output['ext_params_list']    

    # All external parameters on one plot
    fig = plot_final_output_vs_ext_param_color_int_param(
        model_lists=model_lists,
        int_params_metadata=int_params_metadata,
        ext_params_list=ext_params_list,
        int_param_name='Sp4',
        ext_param_name='A',
        colorscale='Viridis',
        log_scale=True,
    )
    fig.write_image("Figures/tip_vs_A_color_Sp4.svg")
    fig.write_html("Figures/tip_vs_A_color_Sp4.html")
    fig.show()

    # Shear Elasticity

    # Small logarithmic perturbations around 1
    epsilon = 0.01  # log-scale offset (in powers of 10)
    n_points = 5  # number of points on each side of 1
    log_offsets = np.linspace(-epsilon, epsilon, num=2*n_points + 1)
    Beta_vec =  np.power(10, log_offsets)  
    int_param_ranges = {'Beta': Beta_vec}

    A_vec = np.pow(10, np.linspace(start=-4, stop=1, num = 12))
    ext_param_ranges = {'A': A_vec}

    # Simulate
    simulation_output = workflow_elastic_viscous_simulation(
        int_param_ranges=int_param_ranges,
        ext_param_ranges=ext_param_ranges,
        param_keys_to_infer=['Beta'],
        n_jobs_simulation=-1,
        checkpoint_str = "./shear_distance",
    )

    model_lists = simulation_output['model_lists']
    int_params_metadata = simulation_output['int_params_metadata']
    ext_params_list = simulation_output['ext_params_list']

    # Plot

    # # Find the index where Beta = 1 (should be the middle point)
    Beta_star = 1
    reference_index = np.argmin(np.abs(Beta_vec - Beta_star))
    reference_model_list = model_lists[reference_index]

    fig = plot_model_distance_multi_external(
        model_lists=model_lists,
        reference_model_list=reference_model_list,
        distance_fn=rel_mse,
        int_param_name='Beta',
        int_params_metadata=int_params_metadata,
        ext_params_list=ext_params_list,
        ext_param_name='A',
        title="Distance vs Beta (All Amplitudes)",
        y_label="Relative L2 Error",
        log_scale=False,
    )
    fig.write_image("Figures/loss_vs_Beta_color_A.svg")
    fig.write_html("Figures/loss_vs_Beta_color_A.html")
    fig.show()

    # Plot full trajectories for varying ext_param, one subplot per int_param

    # Small logarithmic perturbations around 1
    epsilon = 0.1  # log-scale offset (in powers of 10)
    n_points = 5  # number of points on each side of 1
    log_offsets = np.linspace(-epsilon, epsilon, num=2*n_points + 1)
    Beta_vec =  np.power(10, log_offsets)  
    int_param_ranges = {'Beta': Beta_vec}

    A_vec = np.pow(10, np.linspace(start=-4, stop=0, num = 16))
    ext_param_ranges = {'A': A_vec}

    # Simulate
    simulation_output = workflow_elastic_viscous_simulation(
        int_param_ranges=int_param_ranges,
        ext_param_ranges=ext_param_ranges,
        param_keys_to_infer=['Beta'],
        n_jobs_simulation=-1,
        checkpoint_str = "./shear_trajectory",
    )
    model_lists = simulation_output['model_lists']
    int_params_metadata = simulation_output['int_params_metadata']
    ext_params_list = simulation_output['ext_params_list']        

    fig = plot_output_trajectory_subplots_int_param(
        model_lists, int_params_metadata, ext_params_list,
        int_param_name='Beta', ext_param_name='A',
        x_label='x', y_label='y',
        colorscale='Viridis', log_scale=False
    )
    fig.write_image("Figures/filament_color_A_subplot_Beta.svg")
    fig.write_html("Figures/filament_color_A_subplot_Beta.html")
    fig.show()

    fig = plot_output_trajectory_subplots_ext_param(
        model_lists, int_params_metadata, ext_params_list,
        int_param_name='Beta', ext_param_name='A',
        x_label='x', y_label='y',
        colorscale='Viridis', log_scale=False
    )
    fig.write_image("Figures/filament_color_Beta_subplot_A.svg")
    fig.write_html("Figures/filament_color_Beta_subplot_A.html")
    fig.show()

    # Plot last element of X for varying int_params and ext_params
    log_offsets = np.linspace(-3, 3, num=70)
    Beta_vec = np.power(10, log_offsets)  
    int_param_ranges = {'Beta': Beta_vec}

    A_vec = np.pow(10, np.linspace(start=-4, stop=1, num = 12))
    ext_param_ranges = {'A': A_vec}

    # Simulate
    simulation_output = workflow_elastic_viscous_simulation(
        int_param_ranges=int_param_ranges,
        ext_param_ranges=ext_param_ranges,
        param_keys_to_infer=['Beta'],
        n_jobs_simulation=-1,
        checkpoint_str = "./shear_tip",
    )
    model_lists = simulation_output['model_lists']
    int_params_metadata = simulation_output['int_params_metadata']
    ext_params_list = simulation_output['ext_params_list']    

    # All external parameters on one plot
    fig = plot_final_output_vs_ext_param_color_int_param(
        model_lists=model_lists,
        int_params_metadata=int_params_metadata,
        ext_params_list=ext_params_list,
        int_param_name='Beta',
        ext_param_name='A',
        colorscale='Viridis',
        log_scale=True,
    )
    fig.write_image("Figures/tip_vs_A_color_Beta.svg")
    fig.write_html("Figures/tip_vs_A_color_Beta.html")
    fig.show()

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
    fig.write_image("Figures/std_Sp4_vs_A.svg")
    fig.write_html("Figures/std_Sp4_vs_A.html")
    fig.show()

    fig = plot_sigma_vs_ext_param(workflow_output, int_params=['Sp4'], ext_param_name='A', metric='rel_error')
    fig.write_image("Figures/err_Sp4_vs_A.svg")
    fig.write_html("Figures/err_Sp4_vs_A.html")
    fig.show()

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
    fig.write_image("Figures/std_Beta_vs_A.svg")
    fig.write_html("Figures/std_Beta_vs_A.html")
    fig.show()    

    fig = plot_sigma_vs_ext_param(workflow_output, int_params=['Beta'], ext_param_name='A', metric = 'rel_error')
    fig.write_image("Figures/err_Beta_vs_A.svg")
    fig.write_html("Figures/err_Beta_vs_A.html")
    fig.show()        

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
    fig.write_image("Figures/std_Sp4_Beta_vs_A.svg")
    fig.write_html("Figures/std_Sp4_Beta_vs_A.html")
    fig.show() 

    fig = plot_sigma_vs_ext_param(workflow_output, int_params=['Sp4', 'Beta'], ext_param_name = 'A', metric = 'rel_error')
    fig.write_image("Figures/err_Sp4_Beta_vs_A.svg")
    fig.write_html("Figures/err_Sp4_Beta_vs_A.html")
    fig.show()

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
    
    # Transform w0 -> tau_b * w0
    ## Extract tau_b values from legend entries
    tau_b_values = {}
    for trace in fig.data:
        if trace.name and 'tau_b' in trace.name:
            # Extract tau_b value from trace name (e.g., "tau_b = 0.5" -> 0.5)
            tau_b_str = trace.name.split('=')[-1].strip()
            tau_b = float(tau_b_str)
            tau_b_values[trace.name] = tau_b
    ## Modify x data for each trace
    for trace in fig.data:
        if trace.name in tau_b_values:
            tau_b = tau_b_values[trace.name]
            trace.x = tuple(x * tau_b for x in trace.x)
    ## Update x-axis label
    fig.update_xaxes(title_text="τ_b * w₀")

    fig.write_image("Figures/std_tau_b_vs_w0.svg")
    fig.write_html("Figures/std_tau_b_vs_w0.html")
    fig.show()    

    fig = plot_sigma_vs_ext_param(workflow_output, int_params=['tau_b'], ext_param_name='w0', metric = 'rel_error')

    # Transform w0 -> tau_b * w0
    ## Extract tau_b values from legend entries
    tau_b_values = {}
    for trace in fig.data:
        if trace.name and 'tau_b' in trace.name:
            # Extract tau_b value from trace name (e.g., "tau_b = 0.5" -> 0.5)
            tau_b_str = trace.name.split('=')[-1].strip()
            tau_b = float(tau_b_str)
            tau_b_values[trace.name] = tau_b
    ## Modify x data for each trace
    for trace in fig.data:
        if trace.name in tau_b_values:
            tau_b = tau_b_values[trace.name]
            trace.x = tuple(x * tau_b for x in trace.x)
    ## Update x-axis label
    fig.update_xaxes(title_text="τ_b * w₀")

    fig.write_image("Figures/err_tau_b_vs_w0.svg")
    fig.write_html("Figures/err_tau_b_vs_w0.html")
    fig.show()        

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

    # Transform w0 -> tau_s * w0
    ## Extract tau_b values from legend entries
    tau_s_values = {}
    for trace in fig.data:
        if trace.name and 'tau_s' in trace.name:
            # Extract tau_s value from trace name (e.g., "tau_s = 0.5" -> 0.5)
            tau_s_str = trace.name.split('=')[-1].strip()
            tau_s = float(tau_s_str)
            tau_s_values[trace.name] = tau_s
    ## Modify x data for each trace
    for trace in fig.data:
        if trace.name in tau_s_values:
            tau_s = tau_s_values[trace.name]
            trace.x = tuple(x * tau_s for x in trace.x)
    ## Update x-axis label
    fig.update_xaxes(title_text="τ_s * w₀")
    
    fig.write_image("Figures/std_tau_s_vs_w0.svg")
    fig.write_html("Figures/std_tau_s_vs_w0.html")
    fig.show()    

    fig = plot_sigma_vs_ext_param(workflow_output, int_params=['tau_s'], ext_param_name='w0', metric = 'rel_error')


    # Transform w0 -> tau_s * w0
    ## Extract tau_b values from legend entries
    tau_s_values = {}
    for trace in fig.data:
        if trace.name and 'tau_s' in trace.name:
            # Extract tau_s value from trace name (e.g., "tau_s = 0.5" -> 0.5)
            tau_s_str = trace.name.split('=')[-1].strip()
            tau_s = float(tau_s_str)
            tau_s_values[trace.name] = tau_s
    ## Modify x data for each trace
    for trace in fig.data:
        if trace.name in tau_s_values:
            tau_s = tau_s_values[trace.name]
            trace.x = tuple(x * tau_s for x in trace.x)
    ## Update x-axis label
    fig.update_xaxes(title_text="τ_s * w₀")

    fig.write_image("Figures/err_tau_s_vs_w0.svg")
    fig.write_html("Figures/err_tau_s_vs_w0.html")
    fig.show()    

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
    fig.write_image("Figures/std_tau_b_tau_s_vs_w0.svg")
    fig.write_html("Figures/std_tau_b_tau_s_vs_w0.html")
    fig.show()    

    fig = plot_sigma_vs_ext_param(workflow_output, int_params=['tau_b', 'tau_s'], ext_param_name='w0', metric = 'rel_error')
    fig.write_image("Figures/err_tau_b_tau_s_vs_w0.svg")
    fig.write_html("Figures/err_tau_b_tau_s_vs_w0.html")
    fig.show()    
