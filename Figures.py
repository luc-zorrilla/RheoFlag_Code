from pathlib import Path
import dill as pickle
import numpy as np
from typing import Dict, List, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load results
def load_results(writing_path: Path, k_max: int) -> Tuple[Dict, Dict]:
    """Load all result and result_cumul files."""
    results = {}
    results_cumul = {}
    
    for k in range(k_max + 1):
        # Load result_k
        filename = str((writing_path / f"result_k={k}.pkl").resolve())
        try:
            with open(filename, 'rb') as f:
                results[k] = pickle.load(f)
        except FileNotFoundError:
            print(f"Warning: {filename} not found")
        
        # Load result_cumul_k
        filename = str((writing_path / f"result_cumul_k_min=0_k_max={k}.pkl").resolve())
        try:
            with open(filename, 'rb') as f:
                results_cumul[k] = pickle.load(f)
        except FileNotFoundError:
            print(f"Warning: {filename} not found")
    
    return results, results_cumul

# Extract data for plotting
def extract_data(results_dict: Dict) -> Tuple[List, Dict, Dict, Dict]:
    """Extract k values, params, loss, and std_errors from results dict."""
    k_values = sorted(results_dict.keys())
    
    # Initialize param names from first result
    first_result = results_dict[k_values[0]]
    param_names = list(first_result.params.keys())
    
    # Initialize storage
    params_data = {name: [] for name in param_names}
    loss_data = []
    std_errors_data = {name: [] for name in param_names}
    
    # Populate data
    for k in k_values:
        result = results_dict[k]
        loss_data.append(result.loss)
        
        for name in param_names:
            params_data[name].append(result.params[name])
            if result.std_errors is not None:
                # Map param name to std_error index if needed
                param_idx = list(result.params.keys()).index(name)
                std_errors_data[name].append(result.std_errors[param_idx])
    
    return k_values, params_data, loss_data, std_errors_data


if __name__ == "__main__":

    """ Elastic Inference: Bending Filament (Sp4) """

    # Define your paths and parameters
    writing_path = (Path(__file__).resolve().parent.parent / 'Inference' / 'FromSimulationData' / 'ElasticInference_BendingElasticity' / 'VaryingA')
    A_vec = np.pow(10, np.linspace(start = -6, stop = -2, num = 100))
    true_params = {'Sp4':1}
    k_max_overall = 99 # 50 - 1

    # Load data
    results, results_cumul = load_results(writing_path, k_max_overall)
    k_values, params, losses, std_errors = extract_data(results)
    k_values_cumul, params_cumul, losses_cumul, std_errors_cumul = extract_data(results_cumul)

    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Result: Centered and Normalized Parameters vs A",
            "Result_cumul: Centered and Normalized Parameters vs (k_max - k_min)"
        )
    )

    # Colors for differentiation
    colors_list = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]

    def hex_to_rgba(hex_color, alpha):
        """Convert hex color to rgba string."""
        hex_color = hex_color.lstrip('#')
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        return f'rgba({r}, {g}, {b}, {alpha})'

    # 1. Result: params vs A (with log scale on x, Centered and Normalized, transparent error bands)
    A_values = A_vec[np.array(k_values)]

    for idx, (param_name, param_vals) in enumerate(params.items()):
        param_array = np.array(param_vals)
        std_err_array = np.array(std_errors[param_name])
        
        # Center and normalize to true param
        cent_norm_params = (param_array - true_params[param_name]) / true_params[param_name]
        std_norm_params = std_err_array / true_params[param_name]

        # Calculate error bounds
        upper_bound = cent_norm_params + std_norm_params
        lower_bound = cent_norm_params - std_norm_params
        
        color = colors_list[idx % len(colors_list)]
        color_rgba = hex_to_rgba(color, 0.3)  # 30% opacity
        
        # Add shaded error band
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([A_values, A_values[::-1]]),
                y=np.concatenate([upper_bound, lower_bound[::-1]]),
                fill='toself',
                fillcolor=color_rgba,
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                hoverinfo='skip',
                name=param_name
            ),
            row=1, col=1
        )
        
        # Add main line
        fig.add_trace(
            go.Scatter(
                x=A_values,
                y=cent_norm_params,
                mode='lines+markers',
                name=param_name,
                line=dict(color=color, width=2),
                marker=dict(size=8),
                legendgroup='result',
                hovertemplate=f'{param_name}: %{{y:.6f}}<br>A: %{{x:.6f}}<br>Error: ±%{{customdata:.6f}}<extra></extra>',
                customdata=std_norm_params
            ),
            row=1, col=1
        )

    # 2. Result_cumul: params vs (k_max - k_min) (centered and normalized, transparent error bands)
    k_ranges = np.array(k_values_cumul)

    for idx, (param_name, param_vals) in enumerate(params_cumul.items()):
        param_array = np.array(param_vals)
        std_err_array = np.array(std_errors_cumul[param_name])

        # Center and normalize to true param
        cent_norm_params = (param_array - true_params[param_name]) / true_params[param_name]
        std_norm_params = std_err_array / true_params[param_name]        
        
        # Calculate error bounds
        upper_bound = cent_norm_params + std_norm_params
        lower_bound = cent_norm_params - std_norm_params
        
        color = colors_list[idx % len(colors_list)]
        color_rgba = hex_to_rgba(color, 0.3)  # 30% opacity
        
        # Add shaded error band
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([k_ranges, k_ranges[::-1]]),
                y=np.concatenate([upper_bound, lower_bound[::-1]]),
                fill='toself',
                fillcolor=color_rgba,
                line=dict(color='rgba(255,255,255,0)'),
                showlegend=False,
                hoverinfo='skip',
                name=param_name
            ),
            row=1, col=2
        )
        
        # Add main line
        fig.add_trace(
            go.Scatter(
                x=k_ranges,
                y=cent_norm_params,
                mode='lines+markers',
                name=param_name,
                line=dict(color=color, width=2),
                marker=dict(size=8),
                legendgroup='result_cumul',
                showlegend=False,
                hovertemplate=f'{param_name}: %{{y:.6f}}<br>k_range: %{{x}}<br>Error: ±%{{customdata:.6f}}<extra></extra>',
                customdata=std_norm_params
            ),
            row=1, col=2
        )


        # Add main line
        fig.add_trace(
            go.Scatter(
                x=k_ranges,
                y=std_norm_params[0]/np.sqrt(k_ranges + 1),
                mode='lines',
                name=param_name,
                line=dict(color="red", width=2),
                legendgroup='uncertainty_line',
                showlegend=True,
            ),
            row=1, col=2
        )

        # Update x-axes (log scale for A, linear for k_range)
        fig.update_xaxes(
            title_text="A (log scale)",
            type='log',
            row=1, col=1
        )
        fig.update_xaxes(
            title_text="k_max - k_min",
            row=1, col=2
        )

        # Update y-axes
        fig.update_yaxes(
            title_text="Centered and Normalized Parameter Value",
            row=1, col=1,
        )
        fig.update_yaxes(
            title_text="Centered and Normalized Parameter Value",
            row=1, col=2,
        )

        # Update layout
        fig.update_layout(
            title_text="Parameter Inference Analysis (Normalized Error Bands)",
            height=600,
            width=1600,
            showlegend=True,
            hovermode='closest',
            font=dict(size=12),
            legend=dict(
                x=1.02,
                y=1,
                xanchor='left',
                yanchor='top'
            )
        )

        # Save and show
        fig.write_html('analysis_results.html')
        fig.show()

        print("Analysis complete. Interactive plot saved to 'analysis_results.html'")

exit()


fig_nbr = 1
panel_nbr = 0

##############################
# Model chapter ------------ #
##############################

# TODO: complete it from file D01_RF_plots.py

##############################
# Inference chapter -------- #
##############################

## Figure 1: 
## Figure 2:

if __name__ == '__main__':
    print("")
