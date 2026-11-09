# import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import itertools
from pathlib import Path
import dill as pickle
import json

import numpy as np
from scipy.spatial.distance import directed_hausdorff
from scipy import stats
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

from ViscoElasticFilament_Models import X3N, Bend, StraightLine, SecondBend
from ViscoElasticFilament_Inferences import workflow_elastic_viscous_simulation, workflow_elastic_viscous_general, identity_optimizer, basinhopping_optimizer, dual_annealing_optimizer, dual_annealing_optimizer_three_pass, rel_mse

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
                hovertemplate=f"<b>{ext_param_name}</b>: %{{x:.2e}}<br><b>{y_label}</b>: %{{y:.2e}}<br><b>{int_param_name}</b>: {int_param_values[int_idx]:.4f}<extra></extra>",
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

def plot_fig_S1(
    model_lists: Dict[int, ModelList],
    int_params_metadata: List[Dict[str, Any]],
    ext_params_list: List[Dict[str, Any]],
    int_param_name: str = 'N',
    ext_param_name: str = 'A',
    x_label: str = "x (normalized)",
    y_label: str = "y (normalized)",
    log_scale: bool = False,
    colorscale: str = 'Turbo',
) -> Tuple[go.Figure, go.Figure]:
    """
    Generate publication-ready phase-space trajectory plots comparing simulations to analytical solutions.
    
    Args:
        model_lists: Dictionary mapping indices to ModelList objects
        int_params_metadata: List of dicts, each containing internal parameter values
        ext_params_list: List of external parameter dicts (each with 'A' key)
        int_param_name: Name of the internal parameter for colormapping (e.g., 'N')
        ext_param_name: Name of the external parameter (e.g., 'A')
        x_label: X-axis label
        y_label: Y-axis label
        log_scale: Whether to use log scale for axes
        colorscale: Plotly colorscale name (e.g., 'Viridis', 'Blues', 'Reds')
    
    Returns:
        Tuple of (fig_S1_a, fig_S1_b) - phase-space trajectories and convergence error plots
    """
    # Extract internal and external parameter values
    int_param_values = [params_dict[int_param_name] for params_dict in int_params_metadata]
    ext_param_values = [ext_dict[ext_param_name] for ext_dict in ext_params_list]
    
    # Filter to start from N=30
    filtered_indices = [i for i, val in enumerate(int_param_values) if val >= 30]
    int_param_values = [int_param_values[i] for i in filtered_indices]
    int_param_indices = filtered_indices
    
    # Create figures
    fig_S1_a = go.Figure()
    fig_S1_b = go.Figure()
    
    # Normalize internal parameter values for colormap
    if len(int_param_values) > 1:
        int_param_min = min(int_param_values)
        int_param_max = max(int_param_values)
        int_param_norm = [
            (val - int_param_min) / (int_param_max - int_param_min) 
            for val in int_param_values
        ]
    else:
        int_param_norm = [0]

    # Get colors for internal parameters
    colors_int = px.colors.sample_colorscale(colorscale, int_param_norm)

    N_array = []
    error_array = []
    
    # Plot one trace per external parameter in each subplot
    for pos, int_idx in enumerate(sorted([i for i in model_lists.keys() if i in int_param_indices])):
        model_list = model_lists[int_idx]
        int_param_val = int_param_values[pos]
        
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
                    x_trajectory = X_3N[:N, 0] / N
                    y_trajectory = X_3N[N:int(2*N), 0] / N

                    # Analytical Equilibrium Profile
                    n_eq = 1000
                    X_3N_eq = CheckEquilibrium(
                        N, model.ext_params['A'], model.int_params['gamma'], 
                        model.int_params['Sp4'], n_L=model.int_params['n_L'], 
                        Lambdas=model.ext_params['Lambdas'], 
                        conditions="vertical_flow_uniform", n_eq=n_eq
                    )
                    
                    x_eq_trajectory = X_3N_eq[:n_eq, 0][X_3N_eq[:n_eq, 0] <= N-1] / N
                    y_eq_trajectory = X_3N_eq[n_eq:2*n_eq, 0][X_3N_eq[:n_eq, 0] <= N-1] / N

                    # Concatenate x and y trajectories
                    x_traj = np.array((x_trajectory, y_trajectory)).reshape((2, -1))
                    x_eq_traj = np.array((x_eq_trajectory, y_eq_trajectory)).reshape((2, -1))

                    # Calculate Hausdorff Distance (convergence error metric)
                    d1 = directed_hausdorff(x_traj.T, x_eq_traj.T)[0]
                    d2 = directed_hausdorff(x_eq_traj.T, x_traj.T)[0]
                    hausdorff_error = max(d1, d2)
                    error_array.append(hausdorff_error)
                    
                    # Add simulation data trace (SOLID LINE with transparency)
                    fig_S1_a.add_trace(
                        go.Scatter(
                            x=x_trajectory,
                            y=y_trajectory,
                            mode='lines+markers',
                            name=f"{int_param_name} = {int_param_val}",
                            legendgroup=f"group_{pos}",
                            showlegend=False,
                            hovertemplate=(
                                f"<b>Simulation</b><br>"
                                f"<b>{x_label}</b>: %{{x:.6e}}<br>"
                                f"<b>{y_label}</b>: %{{y:.6e}}<br>"
                                f"<b>{int_param_name}</b>: {int_param_val}<br>"
                                f"<b>{ext_param_name}</b>: {ext_param_val:.2e}<extra></extra>"
                            ),
                            line=dict(width=2.5, color=colors_int[pos], dash='solid'),
                            marker=dict(size=5, color=colors_int[pos], opacity=0.7),
                            opacity=0.7,
                        ),
                    )

                    # Add analytical solution trace (DASHED BLACK LINE with transparency)
                    fig_S1_a.add_trace(
                        go.Scatter(
                            x=x_eq_trajectory,
                            y=y_eq_trajectory,
                            mode='lines',
                            name=f"{int_param_name} = {int_param_val}",
                            legendgroup=f"group_{pos}",
                            showlegend=True,
                            hovertemplate=(
                                f"<b>Analytical Solution</b><br>"
                                f"<b>{x_label}</b>: %{{x:.6e}}<br>"
                                f"<b>{y_label}</b>: %{{y:.6e}}<br>"
                                f"<b>{int_param_name}</b>: {int_param_val}<br>"
                                f"<b>{ext_param_name}</b>: {ext_param_val:.2e}<extra></extra>"
                            ),
                            line=dict(width=2.5, color="black", dash='dash'),
                            opacity=0.7,
                        ),
                    )

    # Build convergence error plot (fig_S1_b)
    N_array = np.array(N_array)
    error_array = np.array(error_array)

    # Fit in semi-log space: log(error) = a * log(1/N) + b
    log_N_inv = np.log10(1/N_array)
    log_error = np.log10(error_array)
    coeffs = np.polyfit(log_N_inv, log_error, 1)
    print(f"Linear fit slope (in log-log space): {coeffs[0]:.4f}")
    print(f"Linear fit intercept (in log-log space): {coeffs[1]:.4f}")

    fig_S1_b.add_trace(
        go.Scatter(
            x=1/N_array,
            y=error_array,
            mode='markers',
            name='Hausdorff Distance',
            marker=dict(size=12, color="black", opacity=0.8),
            hovertemplate=(
                f"<b>Grid Spacing</b>: 1/N = %{{x:.6e}}<br>"
                f"<b>Error (Hausdorff)</b>: %{{y:.6e}}<extra></extra>"
            ),
        )
    )
        
    # Add linear fit line in semi-log space, resampled over full range
    x_fit = np.linspace(0.008, 0.036, 100)
    log_x_fit = np.log10(x_fit)
    log_y_fit = coeffs[0] * log_x_fit + coeffs[1]
    y_fit = 10**log_y_fit

    fig_S1_b.add_trace(
        go.Scatter(
            x=x_fit,
            y=y_fit,
            mode='lines',
            name='Linear Fit',
            line=dict(width=2.5, color="black", dash='dash'),
            hovertemplate=(
                f"<b>Grid Spacing</b>: 1/N = %{{x:.6e}}<br>"
                f"<b>Fit</b>: %{{y:.6e}}<extra></extra>"
            ),
        )
    )

    # Update fig_S1_b axes
    fig_S1_b.update_xaxes(
        title_text="Grid Spacing (1/N)",
        type="log",
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgray",
    )
    fig_S1_b.update_yaxes(
        title_text="Hausdorff Distance (error)",
        type="log",
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgray",
    )
    
    fig_S1_b.update_layout(
        hovermode='closest',
        height=700,
        width=950,
        font=dict(size=11, family="Times New Roman"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False, 
    )
    
    # Update fig_S1_a axes
    fig_S1_a.update_xaxes(
        title_text=x_label,
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgray",
    )
    fig_S1_a.update_yaxes(
        title_text=y_label,
        showgrid=True,
        gridwidth=1,
        gridcolor="lightgray",
    )
    
    if log_scale:
        fig_S1_a.update_xaxes(type='log')
        fig_S1_a.update_yaxes(type='log')
    
    # Create colorbar manually using a hidden scatter trace
    colorbar_trace = go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(
            colorscale=colorscale,
            cmin=min(int_param_values),
            cmax=max(int_param_values),
            colorbar=dict(
                title=int_param_name,
                thickness=20,
                len=0.7,
                x=1.15,
            ),
            showscale=True,
        ),
        showlegend=False,
        hoverinfo='none',
    )
    fig_S1_a.add_trace(colorbar_trace)
    
    fig_S1_a.update_layout(
        hovermode='closest',
        height=700,
        width=1100,
        font=dict(size=11, family="Times New Roman"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        margin=dict(r=150),  # Extra right margin for colorbar
    )
    
    return fig_S1_a, fig_S1_b

def plot_fig_1(
    model_lists: Dict[int, ModelList],
    int_params_metadata: List[Dict[str, Any]],
    ext_params_list: List[Dict[str, Any]],
    int_param_name: str = 'Sp4',
    ext_param_name: str = 'A',
    title: Optional[str] = None,
    x_label: str = "x",
    y_label: str = "y",
    colorscale: str = 'RdYlBu_r',
    selected_int_param_values: Optional[List[float]] = None,
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
        colorscale: Plotly colorscale name (e.g., 'RdYlBu_r', 'Plasma')
        selected_int_param_values: List of specific parameter values to plot (e.g., [1e-1, 1e0, 1e1])
    
    Returns:
        Plotly Figure with subplots
    """
    # Extract internal and external parameter values
    int_param_values = [params_dict[int_param_name] for params_dict in int_params_metadata]
    ext_param_values = [ext_dict[ext_param_name] for ext_dict in ext_params_list]
    
    n_internal = len(int_param_values)
    
    # Create plot
    fig_1_a = go.Figure()
    fig_1_b = go.Figure()

    if len(int_param_values) > 1:
        # Normalize internal parameter values for colormap using log scale
        int_param_min = np.log10(min(int_param_values))
        int_param_max = np.log10(max(int_param_values))
        int_param_norm = [
            (np.log10(val) - int_param_min) / (int_param_max - int_param_min) 
            for val in int_param_values
        ]
    else: # Only one internal parameter
        int_param_norm = [0]

    all_theta_trajectories = []
    theta_matrix = []  # List to store rows for the heatmap
    y_axis_values = []  # Store corresponding int_param_values
    
    # Determine which indices to plot based on selected parameter values
    if selected_int_param_values is None:
        selected_indices = set(sorted(model_lists.keys()))
    else:
        selected_indices = set()
        for target_val in selected_int_param_values:
            # Find the closest index to each target value
            closest_idx = min(
                sorted(model_lists.keys()),
                key=lambda idx: abs(int_param_values[idx] - target_val)
            )
            selected_indices.add(closest_idx)

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
                    
                    # Extract alpha, x and y, and theta trajectories
                    # alpha_trajectory = output_array[2:]
                    x_trajectory = X_3N[:N, 0]/N
                    y_trajectory = X_3N[N:int(2*N), 0]/N
                    theta_trajectory = X_3N[int(2*N):int(3*N), 0]

                    all_theta_trajectories.append(theta_trajectory)

                    if 'Beta' in int_param_name:
                        y_trajectory *= int_param_val
                    
                    # Map internal parameter value to color using log-normalized scale
                    normalized_val = int_param_norm[int_idx]
                    color = px.colors.sample_colorscale(colorscale, [normalized_val])[0]
                    
                    # Only plot if this index is in the selected set
                    should_plot = int_idx in selected_indices
                    
                    fig_1_a.add_trace(
                        go.Scatter(
                            x = x_trajectory,
                            y = y_trajectory,
                            mode='lines',
                            hovertemplate=(
                                f"<b>{x_label}</b>: %{{x:.6e}}<br>"
                                f"<b>{y_label}</b>: %{{y:.6e}}<br>"
                                f"<b>{int_param_name}</b>: {int_param_val:.4f}<br>"
                                f"<b>{ext_param_name}</b>: {ext_param_val:.2e}<extra></extra>"
                            ),
                            line=dict(width=2.5, color=color),
                            showlegend=False,
                            visible=should_plot,
                        ),
                    )

                    # Store data for heatmap (only add once per int_param_val)
                    theta_matrix.append(np.log10(theta_trajectory))
                    y_axis_values.append(int_param_val)

    # Create parameter-specific colorbar label
    if int_param_name == 'Beta':
        colorbar_label = 'β'
    else:
        colorbar_label = int_param_name
        
    # Get the selected parameter values for colorbar display
    selected_param_values = sorted([int_param_values[idx] for idx in selected_indices])

    # Limit colorbar ticks to a reasonable number (e.g., 5-7)
    n_ticks = min(7, len(selected_param_values))  # Show max 7 ticks
    tick_indices = np.linspace(0, len(selected_param_values) - 1, n_ticks, dtype=int)
    colorbar_tickvals = [np.log10(selected_param_values[i]) for i in tick_indices]
    colorbar_ticktext = [f"{selected_param_values[i]:.0e}" for i in tick_indices]

    # Add invisible scatter trace for colorbar
    fig_1_a.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(
                colorscale=colorscale,
                cmin=min([np.log10(v) for v in int_param_values]),
                cmax=max([np.log10(v) for v in int_param_values]),
                colorbar=dict(
                    title=colorbar_label,
                    tickvals=colorbar_tickvals,
                    ticktext=colorbar_ticktext,
                ),
                showscale=True,
            ),
            showlegend=False,
            hoverinfo='skip',
        )
    )


    # Update axes for fig_1_a
    fig_1_a.update_xaxes(title_text=f"{x_label}/N")
    fig_1_a.update_yaxes(title_text=f"{y_label}/N")
    
    fig_1_a.update_layout(
        title="",
        hovermode='closest',
        height=800,
        width=1400,
        plot_bgcolor='white',
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
    )

    fig_1_b.add_trace(
            go.Heatmap(
                x=np.linspace(0, 1, len(theta_matrix[0])) if theta_matrix else [],
                y=y_axis_values,
                z=theta_matrix,
                colorscale="Viridis",
                hovertemplate=(
                    f"<b>s</b>: %{{x:.4f}}<br>"
                    f"<b>{int_param_name}</b>: %{{y:.4f}}<br>"
                    f"<b>log10(Θ)</b>: %{{z:.2f}}<extra></extra>"
                ),
                colorbar=dict(title="log10(Θ)"),
            )
        )

    fig_1_b.update_yaxes(type = "log")
    fig_1_b.update_layout(
        xaxis_title="s/L",
        yaxis_title=colorbar_label,
        autosize = False,
        height=800,
        width=1400,
        plot_bgcolor='white',
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='lightgray'),
    )
    
    return fig_1_a, fig_1_b

def plot_fig_2(
    model_lists: Dict[int, ModelList],
    int_params_metadata: List[Dict[str, Any]],
    ext_params_list: List[Dict[str, Any]],
    sim_params_list: List[Dict[str, Any]],
    int_param_name: str = 'tau_b',
    ext_param_name: str = 'A',
    sim_param_name: str = 'T_eval',
    x_label: str = "time",
    y_label: str = "y / y_max",
    colorscale: str = 'Viridis',
) -> Tuple[go.Figure, go.Figure]:
    """
    Args:
        model_lists: Dictionary mapping indices to ModelList objects
        int_params_metadata: List of dicts, each containing internal parameter values
        ext_params_list: List of external parameter dicts (each with 'A' key)
        sim_params_list: List of simulation parameter dicts (each with 'T_eval' key)
        int_param_name: Name of the internal parameter used for coloring (e.g., 'tau_b')
        ext_param_name: Name of the external parameter (e.g., 'A')
        sim_param_name: Name of the simulation parameter used as x-data (e.g., 'T_eval')
        x_label: X-axis label of the trajectory plot
        y_label: Y-axis label of the trajectory plot
        colorscale: Plotly colorscale name (e.g., 'Viridis', 'Blues', 'Reds')

    Returns:
        Tuple of two Plotly figures: trajectories (colored by tau_b on a log scale)
        and timescales (tau_b_inf versus tau_b_sim)
    """
    # Publication style shared by both panels
    font_family = "Times New Roman, Times, serif"
    axis_style = dict(
        showline=True,
        linecolor='black',
        linewidth=1.2,
        mirror=True,
        ticks='outside',
        tickwidth=1.2,
        ticklen=6,
        showgrid=False,
        zeroline=False,
        title_font=dict(size=16),
        tickfont=dict(size=14),
    )

    # Extract internal and external parameter values
    int_param_values = [params_dict[int_param_name] for params_dict in int_params_metadata]
    ext_param_values = [ext_dict[ext_param_name] for ext_dict in ext_params_list]
    sim_param_values = [sim_dict[sim_param_name] for sim_dict in sim_params_list]

    # ---- Logarithmic colormap (all tau_b values are positive) ----
    log_vals = [np.log10(v) for v in int_param_values]
    cmin = min(log_vals)
    cmax = max(log_vals)
    if cmax == cmin:
        cmax = cmin + 1.0  # avoid division by zero for a single value

    int_param_norm = [(c - cmin) / (cmax - cmin) for c in log_vals]
    colors_int = px.colors.sample_colorscale(colorscale, int_param_norm)

    # One colorbar tick per decade
    decade_min = int(np.ceil(cmin - 1e-9))
    decade_max = int(np.floor(cmax + 1e-9))
    tickvals = [float(k) for k in range(decade_min, decade_max + 1)]
    ticktext = [f"10<sup>{k}</sup>" for k in range(decade_min, decade_max + 1)]

    fig_2_a = go.Figure()
    fig_2_b = go.Figure()

    tau_b_list = []
    tau_fit_list = []

    for int_idx in sorted(model_lists.keys()):
        model_list = model_lists[int_idx]
        int_param_val = int_param_values[int_idx]

        for ext_idx in range(len(model_list.models)):
            model = model_list.models[ext_idx]
            ext_param_val = ext_param_values[ext_idx]
            sim_param_val = sim_param_values[ext_idx]

            if model.sim_output is None:
                continue

            output_data = model.sim_output
            if isinstance(output_data, dict) and 'value' in output_data:
                output_data = output_data['value']

            output_array = np.atleast_1d(np.array(output_data))
            if output_array.size <= 1:
                continue

            # Apply X3N transformation
            X_3N = np.transpose(
                np.array([X3N(output_array[:, t]) for t in range(output_array.shape[1])]).squeeze()
            )
            N = X_3N.shape[0] // 3

            # Tip trajectory normalized by its initial value
            y_tip = X_3N[int(2 * N - 1), :] / X_3N[int(2 * N - 1), 0]
            T_eval = sim_param_val

            # Normalize by the maximum of the trajectory
            y_max = np.max(y_tip)
            y_plot = y_tip / y_max

            # Timescale fit: log(y_tip) = -(1/tau) * T_eval + intercept
            slope, intercept, r_value, p_value, std_err = stats.linregress(T_eval, np.log(y_tip))
            tau_fit_list.append(-1 / slope)
            tau_b_list.append(int_param_val)

            fig_2_a.add_trace(
                go.Scatter(
                    x=T_eval,
                    y=y_plot,
                    mode='lines',
                    showlegend=False,
                    hovertemplate=(
                        f"<b>{x_label}</b>: %{{x:.6e}}<br>"
                        f"<b>{y_label}</b>: %{{y:.6e}}<br>"
                        f"<b>{int_param_name}</b>: {int_param_val:.4e}<br>"
                        f"<b>{ext_param_name}</b>: {ext_param_val:.2e}<extra></extra>"
                    ),
                    line=dict(width=1.8, color=colors_int[int_idx]),
                )
            )

    # Invisible trace that carries the continuous logarithmic colorbar
    fig_2_a.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            showlegend=False,
            hoverinfo='skip',
            marker=dict(
                colorscale=colorscale,
                cmin=cmin,
                cmax=cmax,
                color=[cmin],
                showscale=True,
                colorbar=dict(
                    title=dict(text="τ<sub>b</sub>", font=dict(size=16)),
                    tickmode='array',
                    tickvals=tickvals,
                    ticktext=ticktext,
                    tickfont=dict(size=14),
                    outlinewidth=1.2,
                    outlinecolor='black',
                ),
            ),
        )
    )

    # Timescale plot: tau_b_inf versus tau_b_sim
    if tau_fit_list:
        tau_eh = tau_fit_list[0]
        tau_b_array = np.logspace(-3, 9, int(1e3))
        tau_eh_array = np.ones_like(tau_b_array) * tau_eh
        tau_sum_array = tau_b_array + tau_eh_array

        # Black dots: fitted tau_b_inf from the simulations
        fig_2_b.add_trace(
            go.Scatter(
                x=tau_b_list,
                y=tau_fit_list,
                mode='markers',
                name="τ<sub>b,inf</sub>",
                marker=dict(color='black', size=8),
            )
        )
        # Dashed line: tau_b_inf = tau_eh
        fig_2_b.add_trace(
            go.Scatter(
                x=tau_b_array,
                y=tau_eh_array,
                mode='lines',
                name="τ<sub>b,inf</sub> = τ<sub>eh</sub>",
                line=dict(color='black', width=1.8, dash='dash'),
            )
        )
        # Different dashed line: tau_b_inf = tau_b
        fig_2_b.add_trace(
            go.Scatter(
                x=tau_b_array,
                y=tau_b_array,
                mode='lines',
                name="τ<sub>b,inf</sub> = τ<sub>b</sub>",
                line=dict(color='black', width=1.8, dash='dot'),
            )
        )
        # Red line with opacity 0.7: tau_b_inf = tau_eh + tau_b
        fig_2_b.add_trace(
            go.Scatter(
                x=tau_b_array,
                y=tau_sum_array,
                mode='lines',
                name="τ<sub>b,inf</sub> = τ<sub>eh</sub> + τ<sub>b</sub>",
                line=dict(color='red', width=2),
                opacity=0.7,
            )
        )

    # Axes for the trajectory plot
    fig_2_a.update_xaxes(title_text=x_label, **axis_style)
    fig_2_a.update_yaxes(title_text=y_label, type='log', exponentformat='power', **axis_style)

    # Axes for the timescale plot
    fig_2_b.update_xaxes(
        title_text="τ<sub>b,sim</sub>",
        type='log',
        exponentformat='power',
        **axis_style,
    )
    fig_2_b.update_yaxes(
        title_text="τ<sub>b,inf</sub>",
        type='log',
        exponentformat='power',
        **axis_style,
    )

    # Layout for both panels: no title, white background, serif font
    common_layout = dict(
        template='simple_white',
        font=dict(family=font_family, size=14, color='black'),
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=90, r=40, t=40, b=80),
    )

    fig_2_a.update_layout(
        **common_layout,
        showlegend=False,
        hovermode='closest',
        height=600,
        width=900,
    )

    fig_2_b.update_layout(
        **common_layout,
        showlegend=True,
        legend=dict(
            x=0.03,
            y=0.97,
            xanchor='left',
            yanchor='top',
            bgcolor='rgba(255,255,255,1)',
            bordercolor='black',
            borderwidth=1,
            font=dict(size=13),
        ),
        height=600,
        width=700,
    )

    return fig_2_a, fig_2_b

def plot_fig_3(
    model_lists,
    int_params_metadata,
    ext_params_list,
    sim_params_list,
    int_param_name='tau_b',
    ext_param_name='w0',
    sim_param_name='T_eval',
    x_label=r"$w_0$",
    y_label_a=r"$\frac{|y_{\max}|}{\max_{\tau_b} |y_{\max}|}$",
    y_label_b=r"$\tau_{b,\mathrm{inf}}$",
    x_label_b=r"$\tau_{b,\mathrm{sim}}$",
    cbar_label=r"$\tau_b$",
    colorscale='Viridis',
    M_peaks=20,
    font_family="Times New Roman",
    font_size=16,
    width_px=800,
    height_px=600,
    title=None,
):
    """
    Returns
    -------
    fig_3_a : go.Figure
        Normalised harmonic tip amplitude vs w0, coloured by tau_b (log colorbar).
    fig_3_b : go.Figure
        Fitted tau_b_inf vs simulated tau_b_sim (log-log).
    """
    # ------------------------------------------------------------------ #
    # Parameter extraction
    # ------------------------------------------------------------------ #
    int_vals = np.array([p[int_param_name] for p in int_params_metadata], dtype=float)
    ext_vals = np.array([e[ext_param_name] for e in ext_params_list], dtype=float)

    # Colour coordinate: log10(tau_b) mapped linearly to [0, 1]
    log_int = np.log10(int_vals)
    log_span = log_int.max() - log_int.min()
    if len(int_vals) > 1 and log_span > 0:
        int_norm = (log_int - log_int.min()) / log_span
    else:
        int_norm = np.zeros_like(log_int)
    colors_int = px.colors.sample_colorscale(colorscale, list(int_norm))

    # ------------------------------------------------------------------ #
    # Pass 1: extract tip amplitudes for every (tau_b, w0) pair
    # ------------------------------------------------------------------ #
    raw = {}  # int_idx -> (w0_array, ymax_array)
    for int_idx in sorted(model_lists.keys()):
        model_list = model_lists[int_idx]
        w0_list, peak_list = [], []

        for ext_idx, model in enumerate(model_list.models):
            if model.sim_output is None:
                continue

            output_data = model.sim_output
            if isinstance(output_data, dict) and 'value' in output_data:
                output_data = output_data['value']
            output_array = np.atleast_1d(np.array(output_data))
            if output_array.size <= 1:
                continue

            X_3N = np.transpose(
                np.array([X3N(output_array[:, t]) for t in range(output_array.shape[1])]).squeeze()
            )
            N = X_3N.shape[0] // 3
            y_tip = np.abs(X_3N[int(2 * N) - 1, :] / N)

            peaks, _ = find_peaks(y_tip)
            if peaks.size == 0:
                peak_vals = np.array([np.max(y_tip)])
            else:
                top = peaks[np.argsort(y_tip[peaks])[-M_peaks:]]
                peak_vals = y_tip[top]

            w0_list.append(ext_vals[ext_idx])
            peak_list.append(np.nanmean(peak_vals))

        raw[int_idx] = (np.array(w0_list), np.array(peak_list))

    # Global normalisation: |y_max| / max over all tau_b
    global_max = max(np.nanmax(ymax) for _, ymax in raw.values())

    # ------------------------------------------------------------------ #
    # Figure 3a
    # ------------------------------------------------------------------ #
    fig_3_a = go.Figure()
    tau_fit = {}

    def f(x, tau):
        return 1 / np.sqrt(1 + (tau * x) ** 2)

    for int_idx in sorted(raw.keys()):
        w0_arr, ymax_arr = raw[int_idx]
        y_norm = ymax_arr / global_max
        color = colors_int[int_idx]

        fig_3_a.add_trace(go.Scatter(
            x=w0_arr, y=y_norm,
            mode='markers',
            marker=dict(size=6, color=color, line=dict(width=0)),
            hovertemplate=(
                f"{x_label}: %{{x:.3e}}<br>"
                f"{y_label_a}: %{{y:.3e}}<br>"
                f"{int_param_name}: {int_vals[int_idx]:.3g}<extra></extra>"
            ),
        ))

        # Fit 1/sqrt(1+(tau*w0)^2) to valid points
        mask = np.isfinite(y_norm) & np.isfinite(w0_arr) & (w0_arr > 0)
        popt, _ = curve_fit(
            f, w0_arr[mask], y_norm[mask],
            p0=[1.0 / np.sqrt(w0_arr[mask].min() * w0_arr[mask].max())],
            bounds=(0, np.inf),
        )
        tau = popt[0]
        tau_fit[int_idx] = tau

        x_fit = np.logspace(np.log10(w0_arr[mask].min()), np.log10(w0_arr[mask].max()), 1000)
        fig_3_a.add_trace(go.Scatter(
            x=x_fit, y=f(x_fit, tau),
            mode='lines',
            line=dict(width=2.5, color=color),
            hoverinfo='skip',
        ))

    # Colorbar carrier trace (log-scaled tau_b ticks)
    k_min = int(np.ceil(log_int.min()))
    k_max = int(np.floor(log_int.max()))
    tick_k = np.arange(k_min, k_max + 1)
    if log_span > 0:
        tickvals = (tick_k - log_int.min()) / log_span
    else:
        tickvals = np.array([0.5])
    ticktext = [f"10<sup>{k}</sup>" for k in tick_k]

    fig_3_a.add_trace(go.Scatter(
        x=[np.nan], y=[np.nan],
        mode='markers',
        showlegend=False,
        hoverinfo='skip',
        marker=dict(
            color=[0],
            colorscale=colorscale,
            cmin=0, cmax=1,
            showscale=True,
            colorbar=dict(
                title=dict(text=cbar_label, side='right'),
                tickmode='array',
                tickvals=tickvals,
                ticktext=ticktext,
                len=0.9,
                thickness=18,
                outlinewidth=1,
                outlinecolor='black',
            ),
        ),
    ))

    fig_3_a.update_xaxes(
        title_text=x_label, type='log',
        showline=True, linewidth=1.2, linecolor='black', mirror=True,
        ticks='outside', showgrid=False,
    )
    fig_3_a.update_yaxes(
        title_text=y_label_a, type='log',
        showline=True, linewidth=1.2, linecolor='black', mirror=True,
        ticks='outside', showgrid=False,
    )
    fig_3_a.update_layout(
        template='simple_white',
        font=dict(family=font_family, size=font_size, color='black'),
        title=dict(text=title) if title else None,
        showlegend=False,
        hovermode='closest',
        width=width_px,
        height=height_px,
        margin=dict(l=90, r=110, t=50 if title else 20, b=75),
    )

    # ------------------------------------------------------------------ #
    # Figure 3b
    # ------------------------------------------------------------------ #
    tau_sim = np.array([int_vals[i] for i in sorted(tau_fit.keys())])
    tau_inf = np.array([tau_fit[i] for i in sorted(tau_fit.keys())])

    fig_3_b = go.Figure()
    fig_3_b.add_trace(go.Scatter(
        x=tau_sim, y=tau_inf,
        mode='markers',
        marker=dict(size=8, color='black', symbol='circle'),
        hovertemplate=(
            f"{x_label_b}: %{{x:.3g}}<br>"
            f"{y_label_b}: %{{y:.3g}}<extra></extra>"
        ),
    ))

    fig_3_b.update_xaxes(
        title_text=x_label_b, type='log',
        showline=True, linewidth=1.2, linecolor='black', mirror=True,
        ticks='outside', showgrid=False,
    )
    fig_3_b.update_yaxes(
        title_text=y_label_b, type='log',
        showline=True, linewidth=1.2, linecolor='black', mirror=True,
        ticks='outside', showgrid=False,
    )
    fig_3_b.update_layout(
        template='simple_white',
        font=dict(family=font_family, size=font_size, color='black'),
        title=dict(text=title) if title else None,
        showlegend=False,
        width=width_px,
        height=height_px,
        margin=dict(l=90, r=40, t=50 if title else 20, b=75),
    )

    return fig_3_a, fig_3_b

def plot_fig_5(
    workflow_output_1, workflow_output_2, workflow_output_3,
    font_family="Times New Roman",
    font_size=16,
    width_px=800,
    height_px=600,
) -> tuple[go.Figure, go.Figure]:
    """
    Create two publication-ready figures comparing the standard error and the
    signed relative error across three workflows.

    Args:
        workflow_output_1: Bending elasticity workflow output (Sp4)
        workflow_output_2: Shear elasticity workflow output (Beta)
        workflow_output_3: Combined bending-shear elasticity workflow output

    Returns:
        (fig_5a, fig_5b): fig_5a shows sigma(p_inf)/p_sim vs A,
                        fig_5b shows (p_inf - p_sim)/p_sim vs A.
    """

    Sp4_gt = 1e0
    Beta_gt = 1e0

    def to_scalar(v):
        """Convert a scalar, list or array entry to float (or None)."""
        if v is None:
            return None
        if isinstance(v, (list, np.ndarray)):
            return float(v[0]) if len(v) > 0 else None
        return float(v)

    def compute_signed_rel_error(measured_val, ground_truth_val):
        """(p_inf - p_sim) / p_sim"""
        measured = to_scalar(measured_val)
        gt = to_scalar(ground_truth_val)
        if measured is None or gt is None or gt == 0 or not np.isfinite(measured):
            return None
        return (measured - gt) / gt

    def normalise_sigma(sigma_val, ground_truth_val):
        """sigma(p_inf) / p_sim"""
        sigma = to_scalar(sigma_val)
        gt = to_scalar(ground_truth_val)
        if sigma is None or gt is None or gt == 0 or not np.isfinite(sigma):
            return None
        return sigma / abs(gt)

    def style_axes(fig, x_title, y_title, y_log=False):
        """Apply the shared publication style to a figure."""
        axis_common = dict(
            showline=True, linewidth=1.2, linecolor='black', mirror=True,
            ticks='outside', showgrid=False, zeroline=False,
            title_font=dict(size=font_size),
        )
        fig.update_xaxes(title_text=x_title, type='log', **axis_common)
        fig.update_yaxes(
            title_text=y_title,
            type='log' if y_log else 'linear',
            **axis_common,
        )
        fig.update_layout(
            template='simple_white',
            font=dict(family=font_family, size=font_size, color='black'),
            title=None,
            hovermode='closest',
            width=width_px,
            height=height_px,
            margin=dict(l=90, r=40, t=20, b=75),
            legend=dict(
                x=0.02, y=0.98,
                xanchor='left', yanchor='top',
                bgcolor='rgba(255, 255, 255, 0.85)',
                bordercolor='black',
                borderwidth=1,
                font=dict(size=font_size - 3),
            ),
        )

    # Legend labels including internal parameter values
    LBL_BEND = r"$\mathrm{Sp}^4=%g$" % Sp4_gt
    LBL_SHEAR = r"$\beta=%g$" % Beta_gt
    LBL_COMB = r"$(\mathrm{Sp}^4, \beta)=(%g, %g)$" % (Sp4_gt, Beta_gt)

    results_1 = workflow_output_1['results_summary']
    results_2 = workflow_output_2['results_summary']
    results_3 = workflow_output_3['results_summary']

    A_1 = workflow_output_1['ext_param_ranges']['A']
    A_2 = workflow_output_2['ext_param_ranges']['A']
    A_3 = workflow_output_3['ext_param_ranges']['A']

    # Ground-truth (simulated) values
    sp4_gt = 1e0
    beta_gt = 1e0

    series_style = {
        'bending': dict(color='#1f77b4', symbol='circle'),
        'shear': dict(color='#ff7f0e', symbol='circle'),
        'combined': dict(color='#2ca02c', symbol='diamond'),
    }

    # ================================================================== #
    # FIGURE 5a: sigma(p_inf) / p_sim
    # ================================================================== #
    fig_5a = go.Figure()

    sp4_sigma_1 = [normalise_sigma(r.get('Sp4_sigma'), sp4_gt) for r in results_1]
    fig_5a.add_trace(go.Scatter(
        x=A_1, y=sp4_sigma_1,
        mode='markers+lines', name=LBL_BEND,
        marker=dict(size=7, color=series_style['bending']['color'],
                    symbol=series_style['bending']['symbol']),
        line=dict(color=series_style['bending']['color'], width=2),
    ))

    beta_sigma_2 = [normalise_sigma(r.get('Beta_sigma'), beta_gt) for r in results_2]
    fig_5a.add_trace(go.Scatter(
        x=A_2, y=beta_sigma_2,
        mode='markers+lines', name=LBL_SHEAR,
        marker=dict(size=7, color=series_style['shear']['color'],
                    symbol=series_style['shear']['symbol']),
        line=dict(color=series_style['shear']['color'], width=2),
    ))

    sp4_sigma_3 = [normalise_sigma(r.get('Sp4_sigma'), sp4_gt) for r in results_3]
    beta_sigma_3 = [normalise_sigma(r.get('Beta_sigma'), beta_gt) for r in results_3]
    combined_sigma_3 = [
        np.sqrt(s4**2 + b**2)
        if s4 is not None and b is not None else None
        for s4, b in zip(sp4_sigma_3, beta_sigma_3)
    ]
    fig_5a.add_trace(go.Scatter(
        x=A_3, y=combined_sigma_3,
        mode='markers+lines', name=LBL_COMB,
        marker=dict(size=7, color=series_style['combined']['color'],
                    symbol=series_style['combined']['symbol']),
        line=dict(color=series_style['combined']['color'], width=2, dash='dash'),
    ))

    style_axes(
        fig_5a,
        x_title=r"$A$",
        y_title=r"$\sigma(p_{\mathrm{inf}})\,/\,p_{\mathrm{sim}}$",
        y_log=True,
    )

    # ================================================================== #
    # FIGURE 5b: (p_inf - p_sim) / p_sim
    # ================================================================== #
    fig_5b = go.Figure()

    sp4_meas_1 = [r.get('Sp4_inferred') for r in results_1]
    sp4_err_1 = [compute_signed_rel_error(m, sp4_gt) for m in sp4_meas_1]
    fig_5b.add_trace(go.Scatter(
        x=A_1, y=sp4_err_1,
        mode='markers+lines', name=LBL_BEND,
        marker=dict(size=7, color=series_style['bending']['color'],
                    symbol=series_style['bending']['symbol']),
        line=dict(color=series_style['bending']['color'], width=2),
    ))

    beta_meas_2 = [r.get('Beta_inferred') for r in results_2]
    beta_err_2 = [compute_signed_rel_error(m, beta_gt) for m in beta_meas_2]
    fig_5b.add_trace(go.Scatter(
        x=A_2, y=beta_err_2,
        mode='markers+lines', name=LBL_SHEAR,
        marker=dict(size=7, color=series_style['shear']['color'],
                    symbol=series_style['shear']['symbol']),
        line=dict(color=series_style['shear']['color'], width=2),
    ))

    sp4_meas_3 = [r.get('Sp4_inferred') for r in results_3]
    beta_meas_3 = [r.get('Beta_inferred') for r in results_3]
    sp4_err_3 = [compute_signed_rel_error(m, sp4_gt) for m in sp4_meas_3]
    beta_err_3 = [compute_signed_rel_error(m, beta_gt) for m in beta_meas_3]
    combined_err_3 = [
        np.sqrt(e4**2 + eb**2)
        if e4 is not None and eb is not None else None
        for e4, eb in zip(sp4_err_3, beta_err_3)
    ]
    fig_5b.add_trace(go.Scatter(
        x=A_3, y=combined_err_3,
        mode='markers+lines', name=LBL_COMB,
        marker=dict(size=7, color=series_style['combined']['color'],
                    symbol=series_style['combined']['symbol']),
        line=dict(color=series_style['combined']['color'], width=2, dash='dash'),
    ))

    style_axes(
        fig_5b,
        x_title=r"$A$",
        y_title=r"$(p_{\mathrm{inf}} - p_{\mathrm{sim}})\,/\,p_{\mathrm{sim}}$",
        y_log=False,
    )

    # Zero reference line for the signed error
    fig_5b.add_hline(y=0, line=dict(color='black', width=1, dash='dot'))

    return fig_5a, fig_5b

def plot_fig_S4(
    model_lists: Dict[int, ModelList],
    int_params_metadata: List[Dict[str, Any]],
    ext_params_list: List[Dict[str, Any]],
    reference_model_list: ModelList,
    int_param_name: str = 'Sp4',
    ext_param_name: str = 'A',
    distance_fn: Callable[[Dict[str, Any], Dict[str, Any]], float] = rel_mse,
    colorscale: str = 'Viridis',
    font_family: str = "Times New Roman",
    font_size: int = 16,
    width_px: int = 800,
    height_px: int = 600,
    x_tick_reference: float = 1.0,
    x_tick_scale: float = 100.0,
    x_tick_suffix: str = '%',
) -> tuple[go.Figure, go.Figure]:
    """
    Publication-ready plots of distance and final output across internal and
    external parameters. No figure titles; log-scale colorbars replace legends.

    Args:
        model_lists: Dictionary mapping indices to ModelList objects
        int_params_metadata: List of dicts, each containing internal parameter values
        ext_params_list: List of external parameter dicts
        reference_model_list: Reference ModelList to compare against
        int_param_name: Name of the internal parameter (e.g., 'Sp4', 'Beta')
        ext_param_name: Name of the external parameter (e.g., 'A')
        distance_fn: Distance function(output_1, output_2) -> float
        colorscale: Plotly colorscale name
        font_family: Font family for all text
        font_size: Base font size in points
        width_px: Figure width in pixels
        height_px: Figure height in pixels
        x_tick_reference: Truth value of the internal parameter, shown as 0 on
                          Figure A's x-axis (1.0 for Sp4 = 1)
        x_tick_scale: Scale applied to the relative deviation from the reference
                      (100 -> percent, 1000 -> per-mille)
        x_tick_suffix: Suffix appended to the scaled tick labels (e.g., '%')

    Returns:
        Tuple of (fig_a, fig_b) where:
            - fig_a: Normalized loss vs internal parameter (one line per external
                     param, colored by log10 of the external param)
            - fig_b: Tip displacement vs external parameter (markers only, colored
                     by log10 of the internal param, with a log-space linear fit
                     over the first two thirds of the points and a log-space
                     constant fit over the last tenth)
    """

    def _log_range(values):
        log_vals = np.log10(np.asarray(values, dtype=float))
        lo, hi = float(log_vals.min()), float(log_vals.max())
        if hi == lo:
            hi = lo + 1.0  # avoid division by zero for a single value
        return lo, hi

    def _log_colorbar(lo, hi, title):
        span = hi - lo
        exps = np.arange(np.ceil(lo), np.floor(hi) + 1)
        if len(exps) >= 2:
            tickvals = ((exps - lo) / span).tolist()
            ticktext = [f"10<sup>{int(e)}</sup>" for e in exps]
        else:
            tickvals = [0.0, 1.0]
            ticktext = [f"10<sup>{lo:.1f}</sup>", f"10<sup>{hi:.1f}</sup>"]
        return dict(
            title=dict(text=rf"${title}$", side='right'),
            tickmode='array',
            tickvals=tickvals,
            ticktext=ticktext,
            tickfont=dict(size=font_size - 2),
            ticks='outside',
            ticklen=4,
            tickcolor='black',
            outlinewidth=1,
            outlinecolor='black',
            len=0.9,
            thickness=18,
            x=1.02,
        )

    def _style_axes(fig, x_title, y_title, x_log=True, y_log=False):
        fig.update_xaxes(
            title_text=x_title, type='log' if x_log else 'linear',
            showline=True, linewidth=1.2, linecolor='black', mirror=True,
            ticks='outside', showgrid=False, zeroline=False,
        )
        fig.update_yaxes(
            title_text=y_title, type='log' if y_log else 'linear',
            showline=True, linewidth=1.2, linecolor='black', mirror=True,
            ticks='outside', showgrid=False, zeroline=False,
        )

    def _deviation_ticks(values):
        """
        Tick positions at the parameter values (plus the reference value),
        labelled as the relative deviation from the reference:
        (value - reference) * scale. The reference therefore reads 0%.
        """
        positions = np.unique(np.append(np.asarray(values, dtype=float)[::10], x_tick_reference))
        ticktext = []
        for v in positions:
            dev = (v - x_tick_reference) * x_tick_scale
            ticktext.append(f"{dev:.3g}{x_tick_suffix}" if abs(dev) > 1e-12 else f"0{x_tick_suffix}")
        return positions.tolist(), ticktext

    int_param_values = [params_dict[int_param_name] for params_dict in int_params_metadata]
    ext_param_values = [ext_dict[ext_param_name] for ext_dict in ext_params_list]

    num_ext_params = len(ext_params_list)
    sorted_int_keys = sorted(model_lists.keys())

    # ============================================================================
    # FIGURE A: LOSS VS INTERNAL PARAMETER (lines colored by log10 of A)
    # ============================================================================
    fig_a = go.Figure()

    ext_lo, ext_hi = _log_range(ext_param_values)
    ext_norm = [(np.log10(v) - ext_lo) / (ext_hi - ext_lo) for v in ext_param_values]
    colors_ext = px.colors.sample_colorscale(colorscale, ext_norm)

    ref_models_outputs = [model.sim_output for model in reference_model_list.models]

    # Compute all distances first so they can be normalized jointly
    distance_matrix = np.zeros((num_ext_params, len(sorted_int_keys)), dtype=float)
    for ext_idx in range(num_ext_params):
        ref_output = ref_models_outputs[ext_idx]
        for col, int_idx in enumerate(sorted_int_keys):
            model = model_lists[int_idx].models[ext_idx]
            distance_matrix[ext_idx, col] = distance_fn(
                model.sim_output['value'], ref_output['value']
            )

    # Normalize so the maximum amplitude across all curves combined is 1
    global_max = np.nanmax(np.abs(distance_matrix))
    if global_max > 0:
        distance_matrix = distance_matrix / global_max

    all_x, all_y, all_c, all_ext = [], [], [], []

    for ext_idx in range(num_ext_params):
        distances = distance_matrix[ext_idx].tolist()
        ext_value = ext_param_values[ext_idx]

        fig_a.add_trace(go.Scatter(
            x=int_param_values,
            y=distances,
            mode='lines',
            line=dict(color=colors_ext[ext_idx], width=2),
            showlegend=False,
            hoverinfo='skip',
        ))

        all_x.extend(int_param_values)
        all_y.extend(distances)
        all_c.extend([(np.log10(ext_value) - ext_lo) / (ext_hi - ext_lo)] * len(distances))
        all_ext.extend([ext_value] * len(distances))

    # One marker trace carries the colorbar
    fig_a.add_trace(go.Scatter(
        x=all_x,
        y=all_y,
        mode='markers',
        marker=dict(
            opacity=0,
            size=6,
            color=all_c,
            colorscale=colorscale,
            cmin=0,
            cmax=1,
            showscale=True,
            colorbar=_log_colorbar(ext_lo, ext_hi, ext_param_name),
        ),
        customdata=all_ext,
        showlegend=False,
        hovertemplate=(
            f"<b>{int_param_name}:</b> %{{x:.3e}}<br>"
            f"<b>Normalized Loss:</b> %{{y:.3e}}<br>"
            f"<b>{ext_param_name}:</b> %{{customdata:.3e}}<extra></extra>"
        ),
    ))

    fig_a.add_hline(
        y=0,
        line=dict(color='black', width=1.2, dash='dash'),
    )

    _style_axes(
        fig_a,
        x_title=(
            rf"$\mathrm{{{int_param_name}}} - {x_tick_reference:g}$ ({x_tick_suffix})"
        ),
        y_title=r'$\mathrm{Loss}(Sp^4,\ Sp^4=1)$ (normalized)',
        x_log=True,
        y_log=False,
    )
    tickvals_a, ticktext_a = _deviation_ticks(int_param_values)
    fig_a.update_xaxes(tickmode='array', tickvals=tickvals_a, ticktext=ticktext_a)
    fig_a.update_layout(
        template='simple_white',
        font=dict(family=font_family, size=font_size, color='black'),
        showlegend=False,
        hovermode='closest',
        width=width_px,
        height=height_px,
        margin=dict(l=90, r=120, t=20, b=75),
    )

    # ============================================================================
    # FIGURE B: TIP DISPLACEMENT VS EXTERNAL PARAMETER (colored by log10 of Sp4)
    # ============================================================================
    fig_b = go.Figure()

    int_lo, int_hi = _log_range(int_param_values)
    int_norm = [(np.log10(v) - int_lo) / (int_hi - int_lo) for v in int_param_values]
    colors_int = px.colors.sample_colorscale(colorscale, int_norm)

    all_x, all_y, all_c, all_int = [], [], [], []

    for int_idx in sorted_int_keys:
        x_vals, y_vals = [], []
        model_list = model_lists[int_idx]
        int_param_val = int_param_values[int_idx]

        for ext_idx in range(len(model_list.models)):
            model = model_list.models[ext_idx]
            ext_param_val = ext_param_values[ext_idx]

            if model.sim_output is not None:
                output_data = model.sim_output
                if isinstance(output_data, dict) and 'value' in output_data:
                    output_data = output_data['value']

                output_array = np.atleast_1d(np.array(output_data))

                if output_array.size > 1:
                    X_3N = X3N(output_array)
                    N = X_3N.shape[0] // 3
                    final_y = X_3N[N:int(2*N)][-1, 0]

                    if 'Beta' in int_param_name:
                        final_x = ext_param_val / int_param_val
                    else:  # Sp4 or other
                        final_x = ext_param_val * int_param_val

                    x_vals.append(final_x)
                    y_vals.append(final_y)

        if x_vals:
            # Markers only, no connecting lines
            fig_b.add_trace(go.Scatter(
                x=x_vals,
                y=y_vals,
                mode='markers',
                marker=dict(size=6, color=colors_int[int_idx]),
                showlegend=False,
                hoverinfo='skip',
            ))

            all_x.extend(x_vals)
            all_y.extend(y_vals)
            all_c.extend([(np.log10(int_param_val) - int_lo) / (int_hi - int_lo)] * len(x_vals))
            all_int.extend([int_param_val] * len(x_vals))

    # One marker trace carries the colorbar
    fig_b.add_trace(go.Scatter(
        x=all_x,
        y=all_y,
        mode='markers',
        marker=dict(
            size=6,
            color=all_c,
            colorscale=colorscale,
            cmin=0,
            cmax=1,
            showscale=True,
            colorbar=_log_colorbar(int_lo, int_hi, int_param_name),
        ),
        customdata=all_int,
        showlegend=False,
        hovertemplate=(
            f"<b>{ext_param_name}:</b> %{{x:.2e}}<br>"
            f"<b>Tip Displacement:</b> %{{y:.6e}}<br>"
            f"<b>{int_param_name}:</b> %{{customdata:.4f}}<extra></extra>"
        ),
    ))

    # Piecewise fits through all markers together, done in log10 space.
    # Points are sorted by x. Only strictly positive points are used.
    fit_x = np.asarray(all_x, dtype=float)
    fit_y = np.abs(np.asarray(all_y, dtype=float))
    mask = (fit_x > 0) & (fit_y > 0) & np.isfinite(fit_x) & np.isfinite(fit_y)

    if mask.sum() >= 2:
        order = np.argsort(fit_x[mask])
        log_x = np.log10(fit_x[mask])[order]
        log_y = np.log10(fit_y[mask])[order]
        n = len(log_x)

        # Data limits in log10 units, used for the fit range and axis limits
        lx_min, lx_max = float(log_x.min()), float(log_x.max())
        ly_min, ly_max = float(log_y.min()), float(log_y.max())

        # 1) Linear fit in log space over the first two thirds of the data,
        #    drawn across the full x-range of the data
        n_lin = int(np.ceil(2 * n / 3))
        if n_lin >= 2:
            slope, intercept = np.polyfit(log_x[:n_lin], log_y[:n_lin], 1)
            lin_log_x = np.linspace(lx_min, lx_max, 200)
            fig_b.add_trace(go.Scatter(
                x=10 ** lin_log_x,
                y=10 ** (slope * lin_log_x + intercept),
                mode='lines',
                line=dict(color='black', width=2, dash='dash'),
                showlegend=False,
                hoverinfo='skip',
            ))

        # 2) Constant fit in log space over the last tenth of the data,
        #    drawn across the full x-range of the data
        n_const = max(1, int(np.ceil(n / 10)))
        const_log_y = float(np.mean(log_y[-n_const:]))
        const_log_x = np.linspace(lx_min, lx_max, 2)
        fig_b.add_trace(go.Scatter(
            x=10 ** const_log_x,
            y=np.full(2, 10 ** const_log_y),
            mode='lines',
            line=dict(color='black', width=2, dash='dot'),
            showlegend=False,
            hoverinfo='skip',
        ))

    _style_axes(
        fig_b,
        x_title=f'${ext_param_name}$',
        y_title=r'$|y_{\mathrm{tip}}|$',
        x_log=True,
        y_log=True,
    )

    # Fix the axis limits to the data (plus a small margin) so that the fit
    # lines cannot extend the axes. Anything outside is clipped at the edges.
    if mask.sum() >= 2:
        pad_x = max(lx_max - lx_min, 1e-9) * 0.05
        pad_y = max(ly_max - ly_min, 1e-9) * 0.05
        fig_b.update_xaxes(range=[lx_min - pad_x, lx_max + pad_x])
        fig_b.update_yaxes(range=[ly_min - pad_y, ly_max + pad_y])

    fig_b.update_layout(
        template='simple_white',
        font=dict(family=font_family, size=font_size, color='black'),
        showlegend=False,
        hovermode='closest',
        width=width_px,
        height=height_px,
        margin=dict(l=90, r=120, t=20, b=75),
    )

    return fig_a, fig_b

def plot_fig_S5(
    model_lists: Dict[int, ModelList],
    int_params_metadata: List[Dict[str, Any]],
    ext_params_list: List[Dict[str, Any]],
    reference_model_list: ModelList,
    int_param_name: str = 'Beta',
    ext_param_name: str = 'A',
    distance_fn: Callable[[Dict[str, Any], Dict[str, Any]], float] = rel_mse,
    colorscale: str = 'Viridis',
) -> tuple[go.Figure, go.Figure]:
    """
    Publication-ready plots of distance and tip displacement across internal and
    external parameters. No figure titles; log-scale colorbars replace legends.

    Args:
        model_lists: Dictionary mapping indices to ModelList objects
        int_params_metadata: List of dicts, each containing internal parameter values
        ext_params_list: List of external parameter dicts
        reference_model_list: Reference ModelList to compare against
        int_param_name: Name of the internal parameter (e.g., 'Beta')
        ext_param_name: Name of the external parameter (e.g., 'A')
        distance_fn: Distance function(output_1, output_2) -> float
        colorscale: Plotly colorscale name

    Returns:
        Tuple of (fig_S5a, fig_S5b) where:
            - fig_S5a: Normalized loss vs relative deviation of the internal
                    parameter from 1 (%), one line per external parameter,
                    colored by log10 of the external parameter
            - fig_S5b: |y_tip| vs external parameter, markers only, colored by
                    log10 of the internal parameter
    """

    def _log_range(values):
        log_vals = np.log10(np.asarray(values, dtype=float))
        lo, hi = float(log_vals.min()), float(log_vals.max())
        if hi == lo:
            hi = lo + 1.0  # avoid division by zero for a single value
        return lo, hi

    def _log_colorbar(lo, hi, title):
        span = hi - lo
        exps = np.arange(np.ceil(lo), np.floor(hi) + 1)
        if len(exps) >= 2:
            tickvals = ((exps - lo) / span).tolist()
            ticktext = [f"10<sup>{int(e)}</sup>" for e in exps]
        else:
            tickvals = [0.0, 1.0]
            ticktext = [f"10<sup>{lo:.1f}</sup>", f"10<sup>{hi:.1f}</sup>"]
        return dict(
            title=dict(text=rf"${title}$", side='right'),
            tickmode='array',
            tickvals=tickvals,
            ticktext=ticktext,
            ticks='outside',
            ticklen=4,
            tickcolor='black',
            outlinewidth=1,
            outlinecolor='black',
            len=0.9,
            thickness=18,
            x=1.02,
        )

    def _style_axes(fig, x_title, y_title, x_log=True, y_log=False):
        fig.update_xaxes(
            title_text=x_title, type='log' if x_log else 'linear',
            showline=True, linewidth=1.2, linecolor='black', mirror=True,
            ticks='outside', showgrid=False, zeroline=False,
        )
        fig.update_yaxes(
            title_text=y_title, type='log' if y_log else 'linear',
            showline=True, linewidth=1.2, linecolor='black', mirror=True,
            ticks='outside', showgrid=False, zeroline=False,
        )

    int_param_values = [params_dict[int_param_name] for params_dict in int_params_metadata]
    ext_param_values = [ext_dict[ext_param_name] for ext_dict in ext_params_list]

    num_ext_params = len(ext_params_list)
    sorted_int_keys = sorted(model_lists.keys())

    # ============================================================================
    # FIGURE S5A: LOSS VS RELATIVE DEVIATION OF INTERNAL PARAMETER FROM 1
    # ============================================================================
    fig_S5a = go.Figure()

    ext_lo, ext_hi = _log_range(ext_param_values)
    ext_norm = [(np.log10(v) - ext_lo) / (ext_hi - ext_lo) for v in ext_param_values]
    colors_ext = px.colors.sample_colorscale(colorscale, ext_norm)

    ref_models_outputs = [model.sim_output for model in reference_model_list.models]

    # Distances for all curves first, so they can be normalized jointly
    distance_matrix = np.zeros((num_ext_params, len(sorted_int_keys)), dtype=float)
    for ext_idx in range(num_ext_params):
        ref_output = ref_models_outputs[ext_idx]
        for col, int_idx in enumerate(sorted_int_keys):
            model = model_lists[int_idx].models[ext_idx]
            distance_matrix[ext_idx, col] = distance_fn(
                model.sim_output['value'], ref_output['value']
            )

    # Normalize by the maximum amplitude of all data points
    global_max = np.nanmax(np.abs(distance_matrix))
    if global_max > 0:
        distance_matrix = distance_matrix / global_max

    # Relative deviation from the truth value (Beta = 1), in percent
    rel_dev_pct = [(np.asarray(int_param_values[i], dtype=float) - 1.0) * 100.0
                for i in sorted_int_keys]

    for ext_idx in range(num_ext_params):
        fig_S5a.add_trace(go.Scatter(
            x=rel_dev_pct,
            y=distance_matrix[ext_idx].tolist(),
            mode='lines',
            line=dict(color=colors_ext[ext_idx], width=2),
            showlegend=False,
            hoverinfo='skip',
        ))

    # Invisible carrier trace: contains no points, only the log colorbar for A
    fig_S5a.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(
            color=[0.0],
            colorscale=colorscale,
            cmin=0,
            cmax=1,
            showscale=True,
            colorbar=_log_colorbar(ext_lo, ext_hi, ext_param_name),
        ),
        showlegend=False,
        hoverinfo='skip',
    ))

    _style_axes(
        fig_S5a,
        x_title=r'$\beta_{\mathrm{sim}}$ deviation from $\beta = 1$ (%)',
        y_title=r'$\mathrm{Loss}(\beta_{sim},\ \beta=1)$ (normalized)',
        x_log=False,
        y_log=False,
    )

    rel_grid = np.round(np.asarray(rel_dev_pct, dtype=float), 6)
    tick_lo = int(np.floor(rel_grid.min()))
    tick_hi = int(np.ceil(rel_grid.max()))
    tickvals_a = list(range(tick_lo, tick_hi + 1))
    fig_S5a.update_xaxes(
        tickmode='array',
        tickvals=tickvals_a,
        ticktext=[f"{v}%" for v in tickvals_a],
    )
    fig_S5a.update_layout(
        template='plotly_white',
        showlegend=False,
        hovermode='closest',
        width=1000,
        height=650,
        margin=dict(l=90, r=120, t=20, b=75),
    )

    # ============================================================================
    # FIGURE S5B: |y_tip| VS EXTERNAL PARAMETER (markers, colored by log10 Beta)
    # ============================================================================
    fig_S5b = go.Figure()

    int_lo, int_hi = _log_range(int_param_values)

    all_x, all_y, all_c, all_int = [], [], [], []

    for int_idx in sorted_int_keys:
        model_list = model_lists[int_idx]
        int_param_val = int_param_values[int_idx]
        color_norm = (np.log10(int_param_val) - int_lo) / (int_hi - int_lo)

        for ext_idx in range(len(model_list.models)):
            model = model_list.models[ext_idx]
            ext_param_val = ext_param_values[ext_idx]

            if model.sim_output is None:
                continue

            output_data = model.sim_output
            if isinstance(output_data, dict) and 'value' in output_data:
                output_data = output_data['value']

            output_array = np.atleast_1d(np.array(output_data))
            if output_array.size <= 1:
                continue

            X_3N = X3N(output_array)
            N = X_3N.shape[0] // 3
            final_y = X_3N[int(2 * N) - 1, 0]

            if 'Beta' in int_param_name:
                final_x = ext_param_val / int_param_val
            else:  # Sp4 or other
                final_x = ext_param_val * int_param_val

            all_x.append(final_x)
            all_y.append(final_y)
            all_c.append(color_norm)
            all_int.append(int_param_val)

    fig_S5b.add_trace(go.Scatter(
        x=all_x,
        y=np.abs(all_y).tolist(),
        mode='markers',
        marker=dict(
            size=6,
            color=all_c,
            colorscale=colorscale,
            cmin=0,
            cmax=1,
            showscale=True,
            colorbar=_log_colorbar(int_lo, int_hi, r'$\beta$'),
        ),
        customdata=all_int,
        showlegend=False,
        hovertemplate=(
            f"<b>{ext_param_name}:</b> %{{x:.2e}}<br>"
            f"<b>|y<sub>tip</sub>|:</b> %{{y:.6e}}<br>"
            f"<b>{int_param_name}:</b> %{{customdata:.4f}}<extra></extra>"
        ),
    ))

    _style_axes(
        fig_S5b,
        x_title=r'$ A / \beta $',
        y_title=r'$|y_{\mathrm{tip}}|$',
        x_log=True,
        y_log=True,
    )
    fig_S5b.update_layout(
        template='plotly_white',
        showlegend=False,
        hovermode='closest',
        width=1000,
        height=650,
        margin=dict(l=90, r=120, t=20, b=75),
    )

    return fig_S5a, fig_S5b


def plot_vid_S5c(
    model_lists: Dict[int, ModelList],
    int_params_metadata: List[Dict[str, Any]],
    ext_params_list: List[Dict[str, Any]],
    int_param_name: str = 'Beta',
    ext_param_name: str = 'A',
    title: Optional[str] = None,
    x_label: str = "x",
    y_label: str = "y",
    log_scale: bool = False,
    colorscale: str = 'Viridis',
) -> go.Figure:
    """
    Create an animated 2D plot of phase-space trajectories with a slider.
    Each frame corresponds to a different external parameter value.
    All internal parameter trajectories are shown simultaneously, colored by internal parameter.
    Axis ranges are fixed across all frames.
    
    Args:
        model_lists: Dictionary mapping indices to ModelList objects
        int_params_metadata: List of dicts, each containing internal parameter values
        ext_params_list: List of external parameter dicts
        int_param_name: Name of the internal parameter (e.g., 'Beta')
        ext_param_name: Name of the external parameter for animation (e.g., 'A')
        title: Plot title
        x_label: X-axis label
        y_label: Y-axis label
        log_scale: Whether to use log scale for axes
        colorscale: Plotly colorscale name (e.g., 'Viridis', 'Blues', 'Reds')
    
    Returns:
        Plotly Figure with animation and slider
    """
    # Extract internal and external parameter values
    int_param_values = np.array([params_dict[int_param_name] for params_dict in int_params_metadata])
    ext_param_values = np.array([ext_dict[ext_param_name] for ext_dict in ext_params_list])
    
    # Normalize internal parameter values for colormap (log scale)
    int_param_log = np.log10(int_param_values)
    int_param_min = int_param_log.min()
    int_param_max = int_param_log.max()
    int_param_norm = (int_param_log - int_param_min) / (int_param_max - int_param_min)
    
    # Get colors for internal parameters
    colors_int = px.colors.sample_colorscale(
        colorscale, 
        int_param_norm.tolist()
    )
    
    # Pre-compute global axis ranges across all frames
    x_min, x_max = np.inf, -np.inf
    y_min, y_max = np.inf, -np.inf
    sorted_int_indices = sorted(model_lists.keys())
    
    for ext_idx in range(len(ext_param_values)):
        ext_param_val = ext_param_values[ext_idx]
        for int_idx in sorted_int_indices:
            model_list = model_lists[int_idx]
            model = model_list.models[ext_idx]
            
            if model.sim_output is not None:
                output_data = model.sim_output
                if isinstance(output_data, dict) and 'value' in output_data:
                    output_data = output_data['value']
                
                output_array = np.array(output_data)
                
                if output_array.size > 1:
                    output_array = np.atleast_1d(output_array)
                    X_3N = X3N(output_array)
                    N = X_3N.shape[0] // 3
                    
                    x_trajectory = X_3N[:N, 0] / N
                    y_trajectory = X_3N[N:int(2*N), 0] / N

                    if int_param_name == 'Beta':
                        y_trajectory /= ext_param_val
                    
                    x_min = min(x_min, x_trajectory.min())
                    x_max = max(x_max, x_trajectory.max())
                    y_min = min(y_min, y_trajectory.min())
                    y_max = max(y_max, y_trajectory.max())
    
    # Add small padding to the ranges
    x_padding = (x_max - x_min) * 0.05
    y_padding = (y_max - y_min) * 0.05
    x_min -= x_padding
    x_max += x_padding
    y_min -= y_padding
    y_max += y_padding
    
    # Create frames for animation
    frames = []
    
    for ext_idx, ext_param_val in enumerate(ext_param_values):
        frame_traces = []
        
        # Add trace for each internal parameter
        for int_idx in sorted_int_indices:
            model_list = model_lists[int_idx]
            int_param_val = int_param_values[int_idx]
            model = model_list.models[ext_idx]
            
            if model.sim_output is not None:
                output_data = model.sim_output
                if isinstance(output_data, dict) and 'value' in output_data:
                    output_data = output_data['value']
                
                output_array = np.array(output_data)
                
                if output_array.size > 1:
                    output_array = np.atleast_1d(output_array)
                    
                    X_3N = X3N(output_array)
                    N = X_3N.shape[0] // 3
                    
                    x_trajectory = X_3N[:N, 0] / N
                    y_trajectory = X_3N[N:int(2*N), 0] / N

                    if int_param_name == 'Beta':
                        y_trajectory /= ext_param_val
                    
                    frame_traces.append(
                        go.Scatter(
                            x=x_trajectory,
                            y=y_trajectory,
                            mode='lines+markers',
                            name=f"{int_param_name} = {int_param_val:.4f}",
                            legendgroup=f"int_{int_idx}",
                            showlegend=(ext_idx == 0),  # Only show legend on first frame
                            hovertemplate=(
                                f"<b>{x_label}</b>: %{{x:.6e}}<br>"
                                f"<b>{y_label}</b>: %{{y:.6e}}<br>"
                                f"<b>{ext_param_name}</b>: {ext_param_val:.2e}<br>"
                                f"<b>{int_param_name}</b>: {int_param_val:.4f}<extra></extra>"
                            ),
                            line=dict(width=2, color=colors_int[int_idx]),
                            marker=dict(size=4, color=colors_int[int_idx]),
                        )
                    )
        
        frames.append(go.Frame(data=frame_traces, name=str(ext_idx)))
    
    # Create initial figure with first frame
    initial_traces = frames[0].data if frames else []
    
    fig = go.Figure(
        data=initial_traces,
        frames=frames,
    )
    
    # Create slider steps
    slider_steps = []
    for ext_idx, ext_param_val in enumerate(ext_param_values):
        slider_steps.append(
            dict(
                args=[[str(ext_idx)], {
                    "frame": {"duration": 300, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 300},
                }],
                method="animate",
                label=f"{ext_param_name} = {ext_param_val:.2e}",
            )
        )
    
    # Add slider
    sliders = [
        dict(
            active=0,
            yanchor="top",
            y=0,
            xanchor="left",
            x=0.1,
            len=0.8,
            transition={"duration": 300},
            pad={"b": 10, "t": 50},
            currentvalue=dict(
                prefix=f"<b>{ext_param_name} = </b>",
                visible=True,
                xanchor="right",
                font=dict(size=12),
            ),
            steps=slider_steps,
        )
    ]
    
    if title is None:
        title = f"Filament Trajectories Across {ext_param_name} (colored by {int_param_name})"
    
    # Configure 2D layout with fixed axis ranges
    fig.update_layout(
        title=title,
        xaxis=dict(
            title=x_label,
            type='log' if log_scale else 'linear',
            range=[np.log10(x_min) if log_scale else x_min, 
                np.log10(x_max) if log_scale else x_max],
        ),
        yaxis=dict(
            title=y_label,
            type='log' if log_scale else 'linear',
            range=[np.log10(y_min) if log_scale else y_min, 
                np.log10(y_max) if log_scale else y_max],
        ),
        hovermode='closest',
        legend=dict(title=int_param_name, x=0.02, y=0.98),
        height=700,
        width=900,
        template='plotly_white',
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=[
                    dict(
                        args=[None, {
                            "frame": {"duration": 500, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 300},
                        }],
                        label="▶ Play",
                        method="animate",
                    ),
                    dict(
                        args=[[None], {
                            "frame": {"duration": 0, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        }],
                        label="⏸ Pause",
                        method="animate",
                    ),
                ],
                pad={"t": 87},
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.15,
                yanchor="top",
            ),
        ],
        sliders=sliders,
    )
    
    return fig

def plot_fig_6(
    workflow_output_1, workflow_output_2, workflow_output_3
) -> tuple[go.Figure, go.Figure]:

    tau_b_gt = 1e0
    tau_s_gt = 1e0
    beta_gt = 1.0

    def _first(v):
        if v is None:
            return None
        if isinstance(v, (list, np.ndarray)):
            return float(v[0]) if len(v) > 0 else None
        return float(v)

    def _norm(v, scale):
        return v / abs(scale) if v is not None and np.isfinite(v) else None

    def _combine(a, b):
        if a is None or b is None or not (np.isfinite(a) and np.isfinite(b)):
            return None
        return float(np.sqrt(a**2 + b**2))

    def compute_signed_rel_error(measured_val, ground_truth_val):
        """(measured - ground_truth) / ground_truth"""
        measured = _first(measured_val)
        gt = _first(ground_truth_val)
        if measured is None or gt is None or gt == 0:
            return None
        return np.abs(measured - gt) / gt

    def _trace(x, y, name, color, symbol, dash=None):
        line = dict(color=color, width=2.5)
        if dash is not None:
            line['dash'] = dash
        return go.Scatter(
            x=x, y=y, mode='markers+lines', name=name,
            marker=dict(size=9, color=color, symbol=symbol,
                        line=dict(color='black', width=0.5)),
            line=line,
        )

    def _style(fig, x_title, y_title, y_log):
        axis_common = dict(
            showline=True, linecolor='black', linewidth=1.5, mirror=True,
            ticks='outside', tickwidth=1.2, ticklen=6,
            showgrid=False, zeroline=False,
        )
        fig.update_layout(
            template='plotly_white',
            width=800, height=600,
            font=dict(family='Times New Roman', size=16, color='black'),
            margin=dict(l=90, r=30, t=30, b=80),
            hovermode='closest',
            legend=dict(
                x=0.98, y=0.98, xanchor='right', yanchor='top',
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='black', borderwidth=1,
                font=dict(size=14),
            ),
        )
        fig.update_xaxes(
            title=dict(text=x_title, font=dict(size=18)),
            type='log', exponentformat='power', **axis_common,
        )
        fig.update_yaxes(
            title=dict(text=y_title, font=dict(size=18)),
            type='log' if y_log else 'linear', **axis_common,
        )

    # Legend labels including internal parameter values
    LBL_B = r"Bending<br>($\tau_b={:g}$)".format(tau_b_gt)
    LBL_S = r"Shear<br>($\tau_s={:g}$)".format(tau_s_gt)
    LBL_C = r"Combined<br>($(\tau_b, \tau_s)=({:g}, {:g})$)".format(tau_b_gt, tau_s_gt)

    X_TITLE = r"$w_0$"

    results_1 = workflow_output_1['results_summary']
    results_2 = workflow_output_2['results_summary']
    results_3 = workflow_output_3['results_summary']

    w0_1 = workflow_output_1['ext_param_ranges']['w0']
    w0_2 = workflow_output_2['ext_param_ranges']['w0']
    w0_3 = workflow_output_3['ext_param_ranges']['w0']

    # ---------------- Figure A: sigma(tau_inf) / tau_sim ---------------- #
    tau_b_sig_1 = [_norm(_first(r.get('tau_b_sigma')), tau_b_gt) for r in results_1]
    tau_s_sig_2 = [_norm(_first(r.get('tau_s_sigma')), tau_s_gt) for r in results_2]
    tau_b_sig_3 = [_norm(_first(r.get('tau_b_sigma')), tau_b_gt) for r in results_3]
    tau_s_sig_3 = [_norm(_first(r.get('tau_s_sigma')), tau_s_gt) for r in results_3]
    comb_sig_3 = [_combine(b, s) for b, s in zip(tau_b_sig_3, tau_s_sig_3)]

    fig_a = go.Figure()
    fig_a.add_trace(_trace(w0_1, tau_b_sig_1, LBL_B, '#1f77b4', 'circle'))
    fig_a.add_trace(_trace(w0_2, tau_s_sig_2, LBL_S, '#ff7f0e', 'square'))
    fig_a.add_trace(_trace(w0_3, comb_sig_3, LBL_C, '#2ca02c', 'diamond', dash='dash'))

    _style(
        fig_a,
        x_title=X_TITLE,
        y_title=r"$\sigma(\tau_{\mathrm{inf}})/\tau_{\mathrm{sim}}$",
        y_log=True,
    )

    # ---------------- Figure B: (tau_inf - tau_sim) / tau_sim ---------- #
    tau_b_err_1 = [compute_signed_rel_error(r.get('tau_b_inferred'), tau_b_gt) for r in results_1]
    tau_s_err_2 = [compute_signed_rel_error(r.get('tau_s_inferred'), tau_s_gt) for r in results_2]
    tau_b_err_3 = [compute_signed_rel_error(r.get('tau_b_inferred'), tau_b_gt) for r in results_3]
    tau_s_err_3 = [compute_signed_rel_error(r.get('tau_s_inferred'), tau_s_gt) for r in results_3]
    comb_err_3 = [_combine(b, s) for b, s in zip(tau_b_err_3, tau_s_err_3)]

    fig_b = go.Figure()
    fig_b.add_trace(_trace(w0_1, tau_b_err_1, LBL_B, '#1f77b4', 'circle'))
    fig_b.add_trace(_trace(w0_2, tau_s_err_2, LBL_S, '#ff7f0e', 'square'))
    fig_b.add_trace(_trace(w0_3, comb_err_3, LBL_C, '#2ca02c', 'diamond', dash='dash'))

    _style(
        fig_b,
        x_title=X_TITLE,
        y_title=r"$|\tau_{\mathrm{inf}}-\tau_{\mathrm{sim}}|/\tau_{\mathrm{sim}}$",
        y_log=True,
    )

    # Zoom the y-axis
    fig_b.update_yaxes(type='log')

    return fig_a, fig_b


def plot_fig_S6a(workflow_output, int_param_name, ext_param_name='w0', metric='std') -> go.Figure:
    """
    Plot normalized standard error or relative error vs scaled external parameter.
    Directly plots results for a single internal parameter with multiple values,
    scaling the x-axis by the internal parameter value.
    
    Parameters:
    -----------
    workflow_output : dict
        Output from workflow_elastic_viscous_general
    int_param_name : str
        Internal parameter name (e.g., 'tau_b', 'tau_s')
    ext_param_name : str
        External parameter name (e.g., 'w0', 'A')
    metric : str
        Metric to plot ('std' for normalized standard error, 'rel_error' for relative error)
    
    Returns:
    --------
    go.Figure
        Plotly figure object
    """
    
    results_summary = workflow_output['results_summary']
    ext_param_ranges = workflow_output['ext_param_ranges']
    int_param_ranges = workflow_output['int_param_ranges']
    
    ext_param_vec = ext_param_ranges[ext_param_name]
    int_param_vec = int_param_ranges[int_param_name]
    
    # Define metric key based on requested metric
    if metric == 'std':
        metric_key = f'{int_param_name}_sigma'
        y_label = 'Normalized Sigma (Std. Error / Internal Parameter)'
        title_suffix = 'Normalized Standard Error'
    elif metric == 'rel_error':
        metric_key = f'{int_param_name}_rel_error'
        y_label = 'Relative Error'
        title_suffix = 'Relative Error'
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    def parse_task_key(task_key, int_param_ranges):
        """Parse task key to get internal and external parameter indices."""
        parts = task_key.split('_')
        int_idx = int(parts[1])
        ext_idx = int(parts[3])
        
        int_param_names = list(int_param_ranges.keys())
        int_param_counts = [len(int_param_ranges[name]) for name in int_param_names]
        
        multi_idx = np.unravel_index(int_idx, tuple(int_param_counts))
        
        idx_mapping = {name: multi_idx[i] for i, name in enumerate(int_param_names)}
        return idx_mapping, ext_idx
    
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
    
    def format_value(val):
        """Format value for display."""
        if isinstance(val, (int, float)):
            if val >= 1e3 or (val < 1e-2 and val > 0):
                return f'{val:.1e}'
            else:
                return f'{val}'
        return str(val)
    
    # Color palette for different parameter values
    num_params = len(int_param_vec)
    if num_params <= 10:
        color_palette = px.colors.qualitative.Light24[:num_params]
    else:
        color_palette = px.colors.qualitative.Light24
    
    # Organize data by internal parameter value
    data_by_param = {}
    
    for result in results_summary:
        task_key = result['task_key']
        idx_mapping, ext_idx = parse_task_key(task_key, int_param_ranges)
        
        # Get the internal parameter value for this result
        param_idx = idx_mapping[int_param_name]
        param_val = int_param_vec[param_idx]
        
        # Get the metric value
        value = result.get(metric_key)
        
        if value is not None:
            if isinstance(value, (list, np.ndarray)):
                metric_val = float(value[0]) if len(value) > 0 else None
            else:
                metric_val = float(value)
            
            # Normalize if it's standard error metric
            if metric == 'std' and metric_val is not None:
                metric_val = normalize_metric(metric_val, param_val)
            
            if metric_val is not None and np.isfinite(metric_val):
                ext_param_val = ext_param_vec[ext_idx]
                
                # Scale x-axis by internal parameter value
                scaled_x = ext_param_val * param_val
                
                if param_val not in data_by_param:
                    data_by_param[param_val] = {
                        'x': [],
                        'y': [],
                        'color': color_palette[param_idx % len(color_palette)],
                        'label': f'{int_param_name} = {format_value(param_val)}'
                    }
                
                data_by_param[param_val]['x'].append(scaled_x)
                data_by_param[param_val]['y'].append(metric_val)
    
    # Sort by parameter value for consistent plotting order
    sorted_params = sorted(data_by_param.keys())
    
    fig = go.Figure()
    
    for param_val in sorted_params:
        data = data_by_param[param_val]
        
        fig.add_trace(go.Scatter(
            x=data['x'],
            y=data['y'],
            mode='markers+lines',
            name=data['label'],
            marker=dict(
                size=8,
                color=data['color'],
                opacity=0.7,
                line=dict(color='black', width=0.5)
            ),
            line=dict(color=data['color'], width=2),
            hovertemplate=f'<b>{data["label"]}</b><br>' + 
                         f'{int_param_name} * {ext_param_name}: %{{x:.4e}}<br>' +
                        f'Metric: %{{y:.4e}}<extra></extra>'
        ))
    
    # Determine if log scale is appropriate
    x_logscale = True  # Scale is typically log for w0*tau_b or similar
    y_logscale = True  # Standard error and relative error typically plotted on log scale
    
    fig.update_layout(
        title=dict(
            text=f'{title_suffix} vs {int_param_name} * {ext_param_name}',
            font=dict(size=16, color='black')
        ),
        xaxis=dict(
            title=dict(text=f'{int_param_name} × {ext_param_name}', font=dict(size=12)),
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
            x=0.98, y=0.98,
            xanchor='right', yanchor='top',
            bgcolor='rgba(255, 255, 255, 0.8)',
            bordercolor='black',
            borderwidth=1
        )
    )
    
    return fig

def plot_fig_S6b(workflow_output, int_param_name, ext_param_name, metric='std'):
    """
    Plot parameter sweep with x-axis scaled by internal parameter value.
    
    Parameters
    ----------
    workflow_output : dict
        Output from workflow_elastic_viscous_general containing results_summary,
        ext_param_ranges, and int_param_ranges.
    int_param_name : str
        Name of the internal parameter to sweep (e.g., 'tau_s', 'tau_b').
    ext_param_name : str
        Name of the external parameter (e.g., 'w0', 'A', 'omega').
    metric : str, default='std'
        Metric to plot: 'std' for normalized standard error or 'rel_error' for 
        signed relative error.
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Plotly figure with scaled x-axis.
    """
    results_summary = workflow_output['results_summary']
    ext_param_ranges = workflow_output['ext_param_ranges']
    int_param_ranges = workflow_output['int_param_ranges']
    
    ext_param_vec = ext_param_ranges[ext_param_name]
    int_param_vec = int_param_ranges[int_param_name]
    
    # Validate metric
    if metric not in ['std', 'rel_error']:
        raise ValueError(f"Unknown metric: {metric}. Use 'std' or 'rel_error'.")
    
    # Define metric key
    metric_key = f'{int_param_name}_sigma' if metric == 'std' else f'{int_param_name}_rel_error'
    
    def parse_task_key(task_key, int_param_ranges):
        """Parse task key to extract internal and external indices."""
        parts = task_key.split('_')
        int_idx = int(parts[1])
        ext_idx = int(parts[3])
        
        int_param_names = list(int_param_ranges.keys())
        int_param_counts = [len(int_param_ranges[name]) for name in int_param_names]
        
        multi_idx = np.unravel_index(int_idx, tuple(int_param_counts))
        
        idx_mapping = {name: multi_idx[i] for i, name in enumerate(int_param_names)}
        return idx_mapping, ext_idx
    
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
    
    # Organize data by internal parameter value
    data_by_int_param = {}
    
    for result in results_summary:
        task_key = result['task_key']
        idx_mapping, ext_idx = parse_task_key(task_key, int_param_ranges)
        
        # Get internal parameter value
        int_param_idx = idx_mapping[int_param_name]
        int_param_val = int_param_vec[int_param_idx]
        
        # Get metric value
        value = result.get(metric_key)
        
        if value is not None:
            if isinstance(value, (list, np.ndarray)):
                metric_val = float(value[0]) if len(value) > 0 else None
            else:
                metric_val = float(value)
            
            # Normalize if std metric
            if metric == 'std' and metric_val is not None:
                metric_val = normalize_metric(metric_val, int_param_val)
            
            if metric_val is not None and np.isfinite(metric_val):
                ext_param_val = ext_param_vec[ext_idx]
                # Scale x-axis by internal parameter value
                scaled_x = ext_param_val * int_param_val
                
                if int_param_val not in data_by_int_param:
                    data_by_int_param[int_param_val] = {
                        'ext_param': [],
                        'scaled_x': [],
                        'metric': []
                    }
                
                data_by_int_param[int_param_val]['ext_param'].append(ext_param_val)
                data_by_int_param[int_param_val]['scaled_x'].append(scaled_x)
                data_by_int_param[int_param_val]['metric'].append(metric_val)
    
    # Create figure
    fig = go.Figure()
    
    # Color palette
    num_params = len(data_by_int_param)
    if num_params <= 10:
        color_palette = px.colors.qualitative.Light24[:num_params]
    elif num_params <= 24:
        color_palette = px.colors.qualitative.Light24
    else:
        color_palette = px.colors.qualitative.Light24
    
    # Sort by internal parameter value for consistent coloring
    sorted_int_params = sorted(data_by_int_param.keys())
    
    # Format function for legend labels
    def format_value(val):
        if isinstance(val, (int, float)):
            if val >= 1e3 or (val < 1e-2 and val > 0):
                return f'{val:.1e}'
            else:
                return f'{val:.3g}'
        return str(val)
    
    # Add traces
    for i, int_param_val in enumerate(sorted_int_params):
        data = data_by_int_param[int_param_val]
        color = color_palette[i % len(color_palette)]
        
        label = f'{int_param_name}={format_value(int_param_val)}'
        
        fig.add_trace(go.Scatter(
            x=data['scaled_x'],
            y=data['metric'],
            mode='markers+lines',
            name=label,
            marker=dict(
                size=8,
                color=color,
                opacity=0.7,
                line=dict(color='black', width=0.5)
            ),
            line=dict(color=color, width=1.5, dash='solid'),
            hovertemplate=(
                f'<b>{label}</b><br>'
                f'{ext_param_name}: %{{customdata[0]:.3e}}<br>'
                f'Scaled x ({int_param_name}*{ext_param_name}): %{{x:.3e}}<br>'
                f'Metric: %{{y:.3e}}<extra></extra>'
            ),
            customdata=[[ext_param_vec[data_by_int_param[int_param_val]['ext_param'].index(x)]] 
                       for x in data['scaled_x']]
        ))
    
    # Determine axis scaling
    x_logscale = ext_param_name in ['A', 'w0', 'omega']
    y_logscale = True
    
    # Update layout
    if metric == 'std':
        y_label = 'Normalized Sigma (Std. Error / Internal Parameter)'
        title_suffix = 'Normalized Standard Error'
    else:
        y_label = 'Signed Relative Error'
        title_suffix = 'Signed Relative Error'
    
    scaled_x_label = f'{int_param_name} × {ext_param_name}'
    
    fig.update_layout(
        title=dict(
            text=f'{title_suffix} vs {scaled_x_label}',
            font=dict(size=16, color='black')
        ),
        xaxis=dict(
            title=dict(text=scaled_x_label, font=dict(size=12)),
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

def plot_fig_S6c(workflow_output, int_params, ext_param_name, metric='std'):
    """
    Plot combined error metric for multiple internal parameters.
    
    Displays only the combined L2 norm of normalized errors across all
    internal parameters, without individual parameter traces.
    
    Parameters
    ----------
    workflow_output : dict
        Output from workflow_elastic_viscous_general containing results_summary,
        ext_param_ranges, and int_param_ranges.
    int_params : list of str
        Names of internal parameters to include in combined metric
        (e.g., ['tau_b', 'tau_s']).
    ext_param_name : str
        Name of the external parameter (e.g., 'w0', 'A', 'omega').
    metric : str, default='std'
        Metric to plot: 'std' for normalized standard error or 'rel_error' for 
        signed relative error.
    
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Plotly figure with combined error traces only.
    """
    results_summary = workflow_output['results_summary']
    ext_param_ranges = workflow_output['ext_param_ranges']
    int_param_ranges = workflow_output['int_param_ranges']
    
    ext_param_vec = ext_param_ranges[ext_param_name]
    
    # Validate metric
    if metric not in ['std', 'rel_error']:
        raise ValueError(f"Unknown metric: {metric}. Use 'std' or 'rel_error'.")
    
    # Identify varying and fixed parameters
    varying_params = {
        name: values for name, values in int_param_ranges.items()
        if name in int_params and len(values) > 1
    }
    fixed_params = {
        name: values for name, values in int_param_ranges.items()
        if name in int_params and len(values) == 1
    }
    
    varying_param_names = list(varying_params.keys())
    
    # Create parameter combinations for coloring
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
    
    def format_value(val):
        """Format numerical values for display."""
        if isinstance(val, (int, float)):
            if val >= 1e3 or (val < 1e-2 and val > 0):
                return f'{val:.1e}'
            else:
                return f'{val:.3g}'
        return str(val)
    
    def parse_task_key(task_key, int_param_ranges):
        """Parse task key to extract internal and external indices."""
        parts = task_key.split('_')
        int_idx = int(parts[1])
        ext_idx = int(parts[3])
        
        int_param_names = list(int_param_ranges.keys())
        int_param_counts = [len(int_param_ranges[name]) for name in int_param_names]
        
        multi_idx = np.unravel_index(int_idx, tuple(int_param_counts))
        
        idx_mapping = {name: multi_idx[i] for i, name in enumerate(int_param_names)}
        return idx_mapping, ext_idx
    
    def get_param_combination_with_values(idx_mapping, int_params, int_param_ranges):
        """Extract parameter combinations and ground truth values."""
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
    
    # Organize combined data by parameter combination
    combined_data_by_combo = {}
    
    for result in results_summary:
        task_key = result['task_key']
        idx_mapping, ext_idx = parse_task_key(task_key, int_param_ranges)
        
        param_combo, ground_truth_dict = get_param_combination_with_values(
            idx_mapping, int_params, int_param_ranges
        )
        
        # Collect normalized metrics for all internal parameters
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
                
                # Normalize if std metric
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
        
        # Compute combined L2 norm
        if all_finite and len(normalized_metrics) == len(int_params):
            combined_metric = np.sqrt(np.sum(np.array(normalized_metrics) ** 2))
            ext_param_val = ext_param_vec[ext_idx]
            color = combination_colors[param_combo]
            
            # Build label from all parameters
            label_parts = []
            for pname in int_params:
                pval = ground_truth_dict[pname]
                label_parts.append(f'{pname}={format_value(pval)}')
            
            label = ', '.join(label_parts)
            
            if param_combo not in combined_data_by_combo:
                combined_data_by_combo[param_combo] = {
                    'ext_param': [],
                    'metric': [],
                    'color': color,
                    'label': label
                }
            combined_data_by_combo[param_combo]['ext_param'].append(ext_param_val)
            combined_data_by_combo[param_combo]['metric'].append(combined_metric)
    
    # Create figure
    fig = go.Figure()
    
    # Add traces for each parameter combination
    for param_combo, data in combined_data_by_combo.items():
        # Sort by external parameter for better line rendering
        sorted_indices = np.argsort(data['ext_param'])
        sorted_ext_param = [data['ext_param'][i] for i in sorted_indices]
        sorted_metric = [data['metric'][i] for i in sorted_indices]
        
        fig.add_trace(go.Scatter(
            x=sorted_ext_param,
            y=sorted_metric,
            mode='markers+lines',
            name=data['label'],
            marker=dict(
                size=10,
                color=data['color'],
                opacity=0.7,
                symbol='circle',
                line=dict(color='black', width=1)
            ),
            line=dict(color=data['color'], width=2, dash='solid'),
            legendgroup=data['label'],
            showlegend=True,
            hovertemplate=(
                f'<b>{data["label"]}</b><br>'
                f'{ext_param_name}: %{{x:.3e}}<br>'
                f'Combined Error: %{{y:.3e}}<extra></extra>'
            )
        ))
    
    # Determine axis scaling
    x_logscale = ext_param_name in ['A', 'w0', 'omega']
    y_logscale = True
    
    # Set labels and title based on metric
    if metric == 'std':
        y_label = 'Combined Normalized Sigma (L2 norm)'
        title_suffix = 'Combined Normalized Standard Error'
    else:
        y_label = 'Combined Relative Error (L2 norm)'
        title_suffix = 'Combined Signed Relative Error'
    
    fig.update_layout(
        title=dict(
            text=f'{title_suffix} vs {ext_param_name}',
            font=dict(size=16, color='black')
        ),
        xaxis=dict(
            title=dict(text=ext_param_name, font=dict(size=12)),
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

if __name__ == "__main__":
    
    # ---------------------------------- #
    # -------- I. Forward Problem ------ #
    # ---------------------------------- #

    # -------------------------------------------- #
    # Figure S1: Benchmark of analytical solutions #
    # -------------------------------------------- #

    N_vec =  np.array([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100])
    int_param_ranges = {'N': N_vec}

    A_vec = np.array([1e-10]) # "A" quantifies the strength of the flow compared to the elastic forces, on one segment. It does not guarantee no movement at the whole filament scale, though.
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

    # fig_S1_a, fig_S1_b = plot_fig_S1(
    #     model_lists, int_params_metadata, ext_params_list,
    #     int_param_name='N', ext_param_name='A',
    #     x_label='x', y_label='y',
    #     colorscale='Turbo', log_scale=False
    # )

    # fig_S1_a.write_image("Figures/fig_S1a.pdf")
    # fig_S1_a.write_html("Figures/fig_S1a.html")
    # fig_S1_a.show()

    # fig_S1_b.write_image("Figures/fig_S1b.pdf")
    # fig_S1_b.write_html("Figures/fig_S1b.html")
    # fig_S1_b.show()

    # --------------------- #
    # Figure 1: Counterbend #
    # --------------------- #

    N = 100
    N_vec = np.array([N])
    Beta_vec = np.logspace(start = -1, stop = 1, num = 100)
    X_0 = [StraightLine(N)]
    int_param_ranges = {'N': N_vec, 'Beta': Beta_vec, 'X_0':X_0}
    
    Zetas = [0]*N
    Zetas[N//2] = 1e-2
    Zetas_vec = [Zetas]
    A_vec = np.array([0])
    ext_param_ranges = {'A':A_vec, 'Zetas':Zetas_vec}

    # Simulate
    simulation_output = workflow_elastic_viscous_simulation(
        int_param_ranges=int_param_ranges,
        ext_param_ranges=ext_param_ranges,
        param_keys_to_infer=[],
        n_jobs_simulation=-1,
        checkpoint_str = "./counterbend",
    )

    model_lists = simulation_output['model_lists']
    int_params_metadata = simulation_output['int_params_metadata']
    ext_params_list = simulation_output['ext_params_list']

    # fig_1_a, fig_1_b = plot_fig_1(
    #     model_lists, int_params_metadata, ext_params_list,
    #     int_param_name='Beta', ext_param_name='A',
    #     x_label='x', y_label='y',
    #     colorscale='RdYlBu_r',
    # )

    # fig_1_a.write_image("Figures/fig_1a.pdf")
    # fig_1_a.write_html("Figures/fig_1a.html")
    # fig_1_a.show()

    # fig_1_b.write_image("Figures/fig_1b.pdf")
    # fig_1_b.write_html("Figures/fig_1b.html")
    # fig_1_b.show()

    # ------------------------------------------ #
    # Figure 2: Relaxation of a bending filament #
    # ------------------------------------------ #

    # Take X_0 the solution to a bending problem

    N = 10
    N_vec =  np.array([N])
    int_param_ranges = {'N': N_vec}

    A_vec = np.array([1e-3])
    ext_param_ranges = {'A': A_vec}

    # Simulate
    simulation_output = workflow_elastic_viscous_simulation(
        int_param_ranges=int_param_ranges,
        ext_param_ranges=ext_param_ranges,
        param_keys_to_infer=[],
        n_jobs_simulation=-1,
        checkpoint_str = "./bending_problem_to_get_X0",
    )

    model_lists = simulation_output['model_lists']
    int_params_metadata = simulation_output['int_params_metadata']
    ext_params_list = simulation_output['ext_params_list']

    # Extract X_0 here
    int_param_name = 'N'
    ext_param_name = 'A'
    int_param_values = [params_dict[int_param_name] for params_dict in int_params_metadata]
    ext_param_values = [ext_dict[ext_param_name] for ext_dict in ext_params_list]
    for int_idx in sorted(model_lists.keys()):
            model_list = model_lists[int_idx]
            int_param_val = int_param_values[int_idx]

            if int_param_val == N:

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
                            X_0_fig_2 = output_array

    N_vec = np.array([N])
    tau_b_vec = np.array([1e-3, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5])
    X_0 = [X_0_fig_2]
    int_param_ranges = {'N': N_vec, 'tau_b': tau_b_vec, 'X_0':X_0}
    
    A_vec = np.array([0])
    ext_param_ranges = {'A':A_vec}

    T_span_vec = np.array([(1e3, 1e4)])
    T_eval_vec = np.array([np.linspace(1e3, 1e4, int(1e2))])
    method_vec = np.array(['BDF'])
    T_sim_max_vec = np.array([360])
    sim_param_ranges = {'T_span':T_span_vec, 'T_eval':T_eval_vec, 'method':method_vec, 'T_sim_max':T_sim_max_vec}

    # Simulate
    simulation_output = workflow_elastic_viscous_simulation(
        int_param_ranges=int_param_ranges,
        ext_param_ranges=ext_param_ranges,
        sim_param_ranges=sim_param_ranges,
        param_keys_to_infer=[],
        n_jobs_simulation=-1,
        checkpoint_str = "./bending_relaxation",
    )

    model_lists = simulation_output['model_lists']
    int_params_metadata = simulation_output['int_params_metadata']
    ext_params_list = simulation_output['ext_params_list']
    sim_params_list = simulation_output['sim_params_list']

    # fig_2_a, fig_2_b = plot_fig_2(
    #     model_lists, int_params_metadata, ext_params_list, sim_params_list,
    #     int_param_name='tau_b', ext_param_name='A',
    #     x_label='time', y_label='y / y_max',
    #     colorscale='Viridis',
    # )

    # fig_2_a.write_image("Figures/fig_2a.pdf")
    # fig_2_a.write_html("Figures/fig_2a.html")
    # fig_2_a.show()

    # fig_2_b.write_image("Figures/fig_2b.pdf")
    # fig_2_b.write_html("Figures/fig_2a.html")
    # fig_2_b.show()

    # ------------------------------------------------- #
    # -------- II. Spring-Dashpot Simplification ------ #
    # ------------------------------------------------- #

    # ---------------------------------------------------------------- #
    # Figure 3: Harmonic tip amplitude response for a bending filament #
    # ---------------------------------------------------------------- #

    tau_b_vec = np.logspace(start = 0, stop = 6, num = 7)
    int_param_ranges = {'tau_b': tau_b_vec}
    
    A_vec = np.array([1e-5])
    w0_vec = np.logspace(start = -9, stop = 0, num = 100)
    ext_param_ranges = {'A':A_vec, 'w0':w0_vec}

    # Simulate
    simulation_output = workflow_elastic_viscous_simulation(
        int_param_ranges=int_param_ranges,
        ext_param_ranges=ext_param_ranges,
        param_keys_to_infer=[],
        n_jobs_simulation=-1,
        checkpoint_str = "./harmonic_response_tip_amplitude",
    )

    model_lists = simulation_output['model_lists']
    int_params_metadata = simulation_output['int_params_metadata']
    ext_params_list = simulation_output['ext_params_list']
    sim_params_list = simulation_output['sim_params_list']

    # fig_3_a, fig_3_b = plot_fig_3(
    #     model_lists, int_params_metadata, ext_params_list, sim_params_list,
    #     int_param_name='tau_b', 
    #     ext_param_name='w0',
    # )

    # fig_3_a.write_image("Figures/fig_3a.pdf")
    # fig_3_a.write_html("Figures/fig_3a.html")
    # fig_3_a.show()

    # fig_3_b.write_image("Figures/fig_3b.pdf")
    # fig_3_b.write_html("Figures/fig_3b.html")
    # fig_3_b.show()    

    # ----------------------------------------- #
    # -------- II. Algorithmic Inversion ------ #
    # ----------------------------------------- #

    # --------------------------------------------- #
    # Figure 4: Schematics of the Inference Process #
    # --------------------------------------------- #

    # Made in excalidraw

    # --------------------- #
    # A. Elastic Inferences #
    # --------------------- #

    # ----------------------------------- #
    # Figure 5: Inference standard errors #
    # ----------------------------------- #

    optimizer = dual_annealing_optimizer

    # Bending Elasticity - Sp4
    int_param_ranges = {'Sp4': [1e0]}
    A_vec = np.pow(10, np.linspace(start=-6, stop=-1, num=6))
    ext_param_ranges = {'A': A_vec}
    elastic_params_list = ['Sp4']
    viscous_params_list = []

    inference_mode = "single_inference"
    checkpoint_str = "./Results/BendingElasticity/BendingElasticity"

    workflow_output_bending = workflow_elastic_viscous_general(
        int_param_ranges=int_param_ranges,
        ext_param_ranges=ext_param_ranges,
        optimizer=optimizer,
        elastic_params_list = elastic_params_list,
        viscous_params_list = viscous_params_list,
        inference_mode = inference_mode,
        checkpoint_str=checkpoint_str,
        )

    # Shear Elasticity - Beta

    int_param_ranges = {'Beta': [1e0]}
    A_vec = np.pow(10, np.linspace(start = -6, stop = -1, num = 6))
    ext_param_ranges = {'A': A_vec}
    elastic_params_list = ['Beta']
    viscous_params_list = []

    inference_mode = "single_inference"
    checkpoint_str = "./Results/ShearElasticity/ShearElasticity"

    workflow_output_shear = workflow_elastic_viscous_general(
        int_param_ranges=int_param_ranges,
        ext_param_ranges=ext_param_ranges,
        optimizer = optimizer,        
        elastic_params_list = elastic_params_list,
        viscous_params_list = viscous_params_list,
        inference_mode = inference_mode,
        checkpoint_str=checkpoint_str,
        )     

    # Bending & Shear Elasticities - Sp4, Beta

    int_param_ranges = {'Sp4': [1e0], 'Beta': [1e0]}
    A_vec = np.pow(10, np.linspace(start = -6, stop = -1, num = 6))
    ext_param_ranges = {'A': A_vec}
    elastic_params_list = ['Sp4', 'Beta']
    viscous_params_list = []

    inference_mode = "single_inference"
    checkpoint_str = "./Results/BendingShearElasticity/BendingShearElasticity"

    workflow_output_bending_shear = workflow_elastic_viscous_general(
        int_param_ranges=int_param_ranges,
        ext_param_ranges=ext_param_ranges,
        optimizer = optimizer,        
        elastic_params_list = elastic_params_list,
        viscous_params_list = viscous_params_list,
        inference_mode = inference_mode,
        checkpoint_str=checkpoint_str,
        )

    fig_5_a, fig_5_b = plot_fig_5(
        workflow_output_bending, 
        workflow_output_shear, 
        workflow_output_bending_shear
        )

    fig_5_a.write_image("Figures/fig_5a.pdf")
    fig_5_a.write_html("Figures/fig_5a.html")
    fig_5_a.show()

    fig_5_b.write_image("Figures/fig_5b.pdf")
    fig_5_b.write_html("Figures/fig_5b.html")
    fig_5_b.show()

    # ------------------------------------------------ #
    # Figure S4: Hessian decrease for bending filament #
    # ------------------------------------------------ #

    # Small logarithmic perturbations around 1
    epsilon = 0.001  # log-scale offset (in powers of 10)
    # Linear perturbations around Sp4 = 1: 0.1% steps, up to 5% on each side
    Sp4_vec = 1 + epsilon* np.arange(-50, 51)   # 0.950 ... 1.050, 101 points
    int_param_ranges = {'Sp4': Sp4_vec}

    A_vec = np.pow(10, np.linspace(start=-6, stop=-1, num=12))
    ext_param_ranges = {'A': A_vec}

    # Simulate
    simulation_output = workflow_elastic_viscous_simulation(
        int_param_ranges=int_param_ranges,
        ext_param_ranges=ext_param_ranges,
        param_keys_to_infer=['Sp4'],
        n_jobs_simulation=-1,
        checkpoint_str="./bending_distance",
    )

    model_lists = simulation_output['model_lists']
    int_params_metadata = simulation_output['int_params_metadata']
    ext_params_list = simulation_output['ext_params_list']

    # Find the index where Sp4 = 1 (should be the middle point)
    Sp4_star = 1
    reference_index = np.argmin(np.abs(Sp4_vec - Sp4_star))
    reference_model_list = model_lists[reference_index]

    # fig_S4a, _ = plot_fig_S4(
    #     model_lists=model_lists,
    #     int_params_metadata=int_params_metadata,
    #     ext_params_list=ext_params_list,
    #     reference_model_list=reference_model_list,
    #     int_param_name='Sp4',
    #     ext_param_name='A',
    #     distance_fn=rel_mse,
    # )

    # fig_S4a.write_image("Figures/fig_S4a.pdf")
    # fig_S4a.write_html("Figures/fig_S4a.html")
    # fig_S4a.show()

    # ============================================================================
    # FIGURE S4B: Tip displacement vs A (color-coded by Sp4)
    # ============================================================================

    log_offsets = np.linspace(-3, 3, num=7)
    Sp4_vec = np.power(10, log_offsets)  
    int_param_ranges = {'Sp4': Sp4_vec}

    A_vec = np.pow(10, np.linspace(start=-6, stop=-1, num=12))
    ext_param_ranges = {'A': A_vec}

    # Simulate
    simulation_output = workflow_elastic_viscous_simulation(
        int_param_ranges=int_param_ranges,
        ext_param_ranges=ext_param_ranges,
        param_keys_to_infer=['Sp4'],
        n_jobs_simulation=-1,
        checkpoint_str="./bending_tip",
    )

    model_lists = simulation_output['model_lists']
    int_params_metadata = simulation_output['int_params_metadata']
    ext_params_list = simulation_output['ext_params_list']

    # _, fig_S4b = plot_fig_S4(
    #     model_lists=model_lists,
    #     int_params_metadata=int_params_metadata,
    #     ext_params_list=ext_params_list,
    #     reference_model_list=reference_model_list,  # Not needed for fig B
    #     int_param_name='Sp4',
    #     ext_param_name='A',
    #     colorscale='Viridis',
    # )

    # fig_S4b.write_image("Figures/fig_S4b.pdf")
    # fig_S4b.write_html("Figures/fig_S4b.html")
    # fig_S4b.show()

    # ---------------------------------------------- #
    # Figure S5: Hessian decrease for shear filament #
    # ---------------------------------------------- #

    # Linear perturbations around Beta = 1: 0.1% steps, up to 5% on each side
    epsilon = 0.001
    Beta_vec = 1 + epsilon * np.arange(-50, 51)   # 0.950 ... 1.050, 101 points
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

    # ============================================================================
    # FIGURE S5A: Distance vs Beta (All Amplitudes)
    # ============================================================================
    Beta_star = 1
    reference_index = np.argmin(np.abs(Beta_vec - Beta_star))
    reference_model_list = model_lists[reference_index]

    # fig_S5a, _ = plot_fig_S5(
    #     model_lists=model_lists,
    #     int_params_metadata=int_params_metadata,
    #     ext_params_list=ext_params_list,
    #     reference_model_list=reference_model_list,
    #     int_param_name='Beta',
    #     ext_param_name='A',
    #     distance_fn=rel_mse,
    #     colorscale='Viridis',
    # )

    # fig_S5a.write_image("Figures/fig_S5a.pdf")
    # fig_S5a.write_html("Figures/fig_S5a.html")
    # fig_S5a.show()

    # ============================================================================
    # FIGURE S5B: Tip Displacement vs A (Color-coded by Beta)
    # ============================================================================
    log_offsets = np.linspace(-3, 3, num=70)
    Beta_vec = np.power(10, log_offsets)  
    int_param_ranges = {'Beta': Beta_vec}

    A_vec = np.pow(10, np.linspace(start=-4, stop=1, num=12))
    ext_param_ranges = {'A': A_vec}

    simulation_output = workflow_elastic_viscous_simulation(
        int_param_ranges=int_param_ranges,
        ext_param_ranges=ext_param_ranges,
        param_keys_to_infer=['Beta'],
        n_jobs_simulation=-1,
        checkpoint_str="./shear_tip",
    )
    model_lists = simulation_output['model_lists']
    int_params_metadata = simulation_output['int_params_metadata']
    ext_params_list = simulation_output['ext_params_list']

    # _, fig_S5b = plot_fig_S5(
    #     model_lists=model_lists,
    #     int_params_metadata=int_params_metadata,
    #     ext_params_list=ext_params_list,
    #     reference_model_list=reference_model_list,
    #     int_param_name='Beta',
    #     ext_param_name='A',
    #     distance_fn=rel_mse,
    #     colorscale='Viridis',
    # )

    # fig_S5b.write_image("Figures/fig_S5b.pdf")
    # fig_S5b.write_html("Figures/fig_S5b.html")
    # fig_S5b.show()

    # =========
    # VIDEO S5c
    # =========

    # Small logarithmic perturbations around 1
    epsilon = 0.1  # log-scale offset (in powers of 10)
    n_points = 5  # number of points on each side of 1
    log_offsets = np.linspace(-epsilon, epsilon, num=2*n_points + 1)
    Beta_vec =  np.power(10, log_offsets)  
    int_param_ranges = {'Beta': Beta_vec}

    A_vec = np.pow(10, np.linspace(start=-4, stop=0, num=16))
    ext_param_ranges = {'A': A_vec}

    # Simulate
    simulation_output = workflow_elastic_viscous_simulation(
        int_param_ranges=int_param_ranges,
        ext_param_ranges=ext_param_ranges,
        param_keys_to_infer=['Beta'],
        n_jobs_simulation=-1,
        checkpoint_str="./shear_trajectory",
    )
    model_lists = simulation_output['model_lists']
    int_params_metadata = simulation_output['int_params_metadata']
    ext_params_list = simulation_output['ext_params_list']        

    # Create figure
    # fig_S5c = plot_vid_S5c(
    #     model_lists=model_lists,
    #     int_params_metadata=int_params_metadata,
    #     ext_params_list=ext_params_list,
    #     int_param_name='Beta',
    #     ext_param_name='A',
    #     x_label='x',
    #     y_label='y',
    #     colorscale='Viridis',
    #     log_scale=False,
    # )

    # Save and display
    # fig_S5c.write_image("Figures/filament_color_Beta_subplot_A.pdf")
    # fig_S5c.write_html("Figures/filament_color_Beta_subplot_A.html")
    # fig_S5c.show()

    # --------------------- #
    # B. Viscous Inferences #
    # --------------------- #

    # ----------- #
    # Figure 6    #
    # ----------- #

    optimizer = dual_annealing_optimizer_three_pass

    # Bending Viscosity (tau_b)
    int_param_ranges = {'tau_b': [1e0]}
    A_vec = [1e-10]
    w0_vec = np.pow(10, -np.linspace(start=-6, stop=6, num=13))
    ext_param_ranges = {'A': A_vec, 'w0': w0_vec}
    elastic_params_list = []
    viscous_params_list = ['tau_b']

    inference_mode = "single_inference"
    checkpoint_str = "./Results/BendingViscosity/BendingViscosity"

    workflow_output_bending_visc = workflow_elastic_viscous_general(
        int_param_ranges=int_param_ranges,
        ext_param_ranges=ext_param_ranges,
        optimizer=optimizer,
        elastic_params_list=elastic_params_list,
        viscous_params_list=viscous_params_list,
        inference_mode=inference_mode,
        checkpoint_str=checkpoint_str,
        n_jobs_per_pass=1,
    )

    # Shear Viscosity (tau_s)
    int_param_ranges = {'tau_s': [1e0], 'Beta': [1.0]}
    A_vec = [1e-10]
    w0_vec = np.pow(10, -np.linspace(start=-6, stop=6, num=13))
    ext_param_ranges = {'A': A_vec, 'w0': w0_vec}
    elastic_params_list = []
    viscous_params_list = ['tau_s']

    inference_mode = "single_inference"
    checkpoint_str = "./Results/ShearViscosity/ShearViscosity"

    workflow_output_shear_visc = workflow_elastic_viscous_general(
        int_param_ranges=int_param_ranges,
        ext_param_ranges=ext_param_ranges,
        optimizer=optimizer,
        elastic_params_list=elastic_params_list,
        viscous_params_list=viscous_params_list,
        inference_mode=inference_mode,
        checkpoint_str=checkpoint_str,
        n_jobs_per_pass=1,
    )

    # Combined Bending and Shear Viscosity (tau_b + tau_s)
    int_param_ranges = {'tau_b': [1e0], 'tau_s': [1e0], 'Beta': [1.0]}
    A_vec = [1e-10]
    w0_vec = np.pow(10, -np.linspace(start=-6, stop=6, num=13))
    ext_param_ranges = {'A': A_vec, 'w0': w0_vec}
    elastic_params_list = []
    viscous_params_list = ['tau_b', 'tau_s']

    inference_mode = "single_inference"
    checkpoint_str = "./Results/BendingShearViscosity/BendingShearViscosity"

    workflow_output_bending_shear_visc = workflow_elastic_viscous_general(
        int_param_ranges=int_param_ranges,
        ext_param_ranges=ext_param_ranges,
        optimizer=optimizer,
        elastic_params_list=elastic_params_list,
        viscous_params_list=viscous_params_list,
        inference_mode=inference_mode,
        checkpoint_str=checkpoint_str,
        n_jobs_per_pass=1,
    )

    fig_6_a, fig_6_b = plot_fig_6(
        workflow_output_bending_visc,
        workflow_output_shear_visc,
        workflow_output_bending_shear_visc,
    )

    fig_6_a.write_image("Figures/fig_6a.pdf")
    fig_6_a.write_html("Figures/fig_6a.html")
    fig_6_a.show()

    fig_6_b.write_image("Figures/fig_6b.pdf")
    fig_6_b.write_html("Figures/fig_6b.html")
    fig_6_b.show()
    
    exit()

    # ----------- #
    # Figure S6   #
    # ----------- #

    # ============ Figure S6a: Viscous Parameter Sweeps ============

    optimizer = dual_annealing_optimizer_three_pass

    # Bending Viscosity (tau_b with multiple values)
    int_param_ranges = {'tau_b': [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]}
    A_vec = [1e-10]
    w0_vec = np.pow(10, -np.linspace(start=-6, stop=6, num=13))
    ext_param_ranges = {'A': A_vec, 'w0': w0_vec}
    elastic_params_list = []
    viscous_params_list = ['tau_b']

    inference_mode = "single_inference"
    checkpoint_str = "./Results/BendingViscosity/Full/BendingViscosity"

    workflow_output_tau_b = workflow_elastic_viscous_general(
        int_param_ranges=int_param_ranges,
        ext_param_ranges=ext_param_ranges,
        optimizer=optimizer,
        elastic_params_list=elastic_params_list,
        viscous_params_list=viscous_params_list,
        inference_mode=inference_mode,
        checkpoint_str=checkpoint_str,
        n_jobs_per_pass=1,
    )

    # Plot bending viscosity standard error
    fig_S6a_std = plot_fig_S6a(
        workflow_output_tau_b,
        int_param_name='tau_b',
        ext_param_name='w0',
        metric='std'
    )

    fig_S6a_std.write_image("Figures/fig_S6a_1.pdf")
    fig_S6a_std.write_html("Figures/fig_S6a_1.html")
    fig_S6a_std.show()

    # Plot bending viscosity relative error
    fig_S6a_err = plot_fig_S6a(
        workflow_output_tau_b,
        int_param_name='tau_b',
        ext_param_name='w0',
        metric='rel_error'
    )

    fig_S6a_err.write_image("Figures/fig_S6a_2.pdf")
    fig_S6a_err.write_html("Figures/fig_S6a_2.html")
    fig_S6a_err.show()

    # Shear Viscosity (Fixed Shear Elasticity)
    int_param_ranges = {'tau_s': [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], 'Beta': [1.0]}
    A_vec = [1e-10]
    w0_vec = np.pow(10, -np.linspace(start=-6, stop=6, num=13))
    ext_param_ranges = {'A': A_vec, 'w0': w0_vec}
    elastic_params_list = []
    viscous_params_list = ['tau_s']

    inference_mode = "single_inference"
    checkpoint_str = "./Results/ShearViscosity/Full/ShearViscosity"

    workflow_output = workflow_elastic_viscous_general(
        int_param_ranges=int_param_ranges,
        ext_param_ranges=ext_param_ranges,
        optimizer=optimizer,
        elastic_params_list=elastic_params_list,
        viscous_params_list=viscous_params_list,
        inference_mode=inference_mode,
        checkpoint_str=checkpoint_str,
        n_jobs_per_pass=1,
    )

    # Standard error with scaled x-axis
    fig = plot_fig_S6b(workflow_output, int_param_name='tau_s', ext_param_name='w0', metric='std')
    fig.write_image("Figures/fig_S6b_1.pdf")
    fig.write_html("Figures/fig_S6b_1.html")
    fig.show()

    # Relative error with scaled x-axis
    fig = plot_fig_S6b(workflow_output, int_param_name='tau_s', ext_param_name='w0', metric='rel_error')
    fig.write_image("Figures/fig_S6b_2.pdf")
    fig.write_html("Figures/fig_S6b_2.html")
    fig.show()

    # Bending & Shear Viscosities (Fixed Bending & Shear Elasticities)
    int_param_ranges = {'tau_b': [1e-3, 1e0, 1e3], 'tau_s': [1e-3, 1e0, 1e3], 'Beta': [1.0]}
    A_vec = [1e-10]
    w0_vec = np.pow(10, -np.linspace(start=-6, stop=6, num=13))
    ext_param_ranges = {'A': A_vec, 'w0': w0_vec}
    elastic_params_list = []
    viscous_params_list = ['tau_b', 'tau_s']

    inference_mode = "single_inference"
    checkpoint_str = "./Results/BendingShearViscosity/Full/BendingShearViscosity"

    workflow_output = workflow_elastic_viscous_general(
        int_param_ranges=int_param_ranges,
        ext_param_ranges=ext_param_ranges,
        optimizer=optimizer,
        elastic_params_list=elastic_params_list,
        viscous_params_list=viscous_params_list,
        inference_mode=inference_mode,
        checkpoint_str=checkpoint_str,
        n_jobs_per_pass=1,
    )

    # Combined standard error with scaled x-axis
    fig = plot_fig_S6c(workflow_output, int_params=['tau_b', 'tau_s'], ext_param_name='w0', metric='std')
    fig.write_image("Figures/fig_S6c_1.pdf")
    fig.write_html("Figures/fig_S6c_1.html")
    fig.show()

    # Combined relative error with scaled x-axis
    fig = plot_fig_S6c(workflow_output, int_params=['tau_b', 'tau_s'], ext_param_name='w0', metric='rel_error')
    fig.write_image("Figures/fig_S6c_2.pdf")
    fig.write_html("Figures/fig_S6c_2.html")
    fig.show()