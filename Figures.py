"""
Figures.py - Visualization of inference results across different elasticity modes.
"""

import json
import dill as pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class InferenceResultsVisualizer:
    
    def __init__(self, mode_path: Path):
        self.mode_path = Path(mode_path)
        self.mode = self.mode_path.name  # 'SingleExtParams' or 'CumulativeExtParams'
        self.manifest = self._load_manifest()
        self.param_ref = self._load_param_reference()
    
    def _load_manifest(self) -> Dict:
        manifest_file = self.mode_path / 'results_manifest.json'
        if not manifest_file.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_file}")
        with open(manifest_file, 'r') as f:
            return json.load(f)
    
    def _load_param_reference(self) -> Dict:
        ref_file = self.mode_path / 'parameter_reference.json'
        if not ref_file.exists():
            raise FileNotFoundError(f"Parameter reference not found: {ref_file}")
        with open(ref_file, 'r') as f:
            return json.load(f)
    
    def _load_result_file(self, result_file_path: str):
        full_path = self.mode_path / result_file_path
        with open(full_path, 'rb') as f:
            return pickle.load(f)

    def _extract_results_single_mode(self) -> Tuple[Dict, Dict, Dict, Dict]:
        int_params_dict = {}
        ext_params_dict = {}
        uncertainties_dict = {}
        loss_dict = {}
        
        for entry in self.manifest['entries']:
            result = self._load_result_file(entry['result_file'])
            
            # Create a hashable key from ground truth int_params
            entry_key = tuple(sorted(entry['int_params'].items()))
            
            # Initialize nested dicts for this ground truth combo if needed
            for param_name, param_value in result.params.items():
                if param_name not in int_params_dict:
                    int_params_dict[param_name] = {}
                if entry_key not in int_params_dict[param_name]:
                    int_params_dict[param_name][entry_key] = {}
                
                # Use external parameter value as the secondary key
                ext_param_key = None
                if entry['ext_params']:
                    # Assuming single external parameter; adjust if multiple
                    ext_param = self._get_external_params()[0]
                    ext_param_key = entry['ext_params'].get(ext_param)
                
                int_params_dict[param_name][entry_key][ext_param_key] = param_value
            
            # Store external parameters
            if entry['ext_params']:
                for param_name, param_value in entry['ext_params'].items():
                    if param_name not in ext_params_dict:
                        ext_params_dict[param_name] = {}
                    if entry_key not in ext_params_dict[param_name]:
                        ext_params_dict[param_name][entry_key] = {}
                    
                    ext_param_key = entry['ext_params'].get(self._get_external_params()[0])
                    ext_params_dict[param_name][entry_key][ext_param_key] = param_value
            
            # Store uncertainties (standard errors)
            if result.std_errors is not None:
                for i, param_name in enumerate(result.params.keys()):
                    if param_name not in uncertainties_dict:
                        uncertainties_dict[param_name] = {}
                    if entry_key not in uncertainties_dict[param_name]:
                        uncertainties_dict[param_name][entry_key] = {}
                    
                    ext_param_key = None
                    if entry['ext_params']:
                        ext_param = self._get_external_params()[0]
                        ext_param_key = entry['ext_params'].get(ext_param)
                    
                    uncertainties_dict[param_name][entry_key][ext_param_key] = result.std_errors[i]
            
            # Store loss
            if entry_key not in loss_dict:
                loss_dict[entry_key] = {}
            ext_param_key = None
            if entry['ext_params']:
                ext_param = self._get_external_params()[0]
                ext_param_key = entry['ext_params'].get(ext_param)
            loss_dict[entry_key][ext_param_key] = result.loss
        
        return int_params_dict, ext_params_dict, uncertainties_dict, loss_dict

    def _extract_results_cumulative_mode(self) -> Tuple[Dict, Dict, Dict, Dict]:
        int_params_dict = {}
        ext_params_dict = {}
        uncertainties_dict = {}
        cumul_indices_dict = {}
        
        for entry in self.manifest['entries']:
            result = self._load_result_file(entry['result_file'])
            
            # Create a hashable key from ground truth int_params
            entry_key = tuple(sorted(entry['int_params'].items()))
            
            # Initialize nested dicts for this ground truth combo if needed
            for param_name, param_value in result.params.items():
                if param_name not in int_params_dict:
                    int_params_dict[param_name] = {}
                if entry_key not in int_params_dict[param_name]:
                    int_params_dict[param_name][entry_key] = {}
                
                # Use cumulative index as the secondary key
                cumul_index_key = None
                if entry['cumul_indices']:
                    ext_param = self._get_external_params()[0]
                    if ext_param in entry['cumul_indices']:
                        start, end = entry['cumul_indices'][ext_param]
                        cumul_index_key = end + 1  # end is inclusive, so num_params = end + 1
                
                int_params_dict[param_name][entry_key][cumul_index_key] = param_value
            
            # Store external parameters
            if entry['ext_params']:
                for param_name, param_value in entry['ext_params'].items():
                    if param_name not in ext_params_dict:
                        ext_params_dict[param_name] = {}
                    if entry_key not in ext_params_dict[param_name]:
                        ext_params_dict[param_name][entry_key] = {}
                    
                    cumul_index_key = None
                    if entry['cumul_indices']:
                        ext_param = self._get_external_params()[0]
                        if ext_param in entry['cumul_indices']:
                            start, end = entry['cumul_indices'][ext_param]
                            cumul_index_key = end + 1
                    
                    ext_params_dict[param_name][entry_key][cumul_index_key] = param_value
            
            # Store uncertainties (standard errors)
            if result.std_errors is not None:
                for i, param_name in enumerate(result.params.keys()):
                    if param_name not in uncertainties_dict:
                        uncertainties_dict[param_name] = {}
                    if entry_key not in uncertainties_dict[param_name]:
                        uncertainties_dict[param_name][entry_key] = {}
                    
                    cumul_index_key = None
                    if entry['cumul_indices']:
                        ext_param = self._get_external_params()[0]
                        if ext_param in entry['cumul_indices']:
                            start, end = entry['cumul_indices'][ext_param]
                            cumul_index_key = end + 1
                    
                    uncertainties_dict[param_name][entry_key][cumul_index_key] = result.std_errors[i]
            
            # Store cumulative indices
            if entry['cumul_indices']:
                if entry_key not in cumul_indices_dict:
                    cumul_indices_dict[entry_key] = {}
                cumul_indices_dict[entry_key] = entry['cumul_indices']
        
        return int_params_dict, ext_params_dict, uncertainties_dict, cumul_indices_dict

    def _get_inferred_params(self) -> List[str]:
        """Identify which parameters were inferred."""
        if not self.manifest['entries']:
            return []
        
        first_entry = self.manifest['entries'][0]
        result = self._load_result_file(first_entry['result_file'])
        return list(result.params.keys())
    
    def _get_external_params(self) -> List[str]:
        """Identify external parameters."""
        if not self.manifest['entries']:
            return []
        
        # For cumulative mode, extract external params from cumul_indices
        if self.mode == 'CumulativeExtParams':
            first_entry = self.manifest['entries'][0]
            if 'cumul_indices' in first_entry and first_entry['cumul_indices']:
                return list(first_entry['cumul_indices'].keys())
            return []
        
        # For single mode, get them from ext_params
        first_entry = self.manifest['entries'][0]
        return list(first_entry['ext_params'].keys()) if first_entry['ext_params'] else []
    
    def _get_ground_truth_combinations(self, inferred_param: str) -> Dict:
        """
        Get all unique ground truth int_params combinations and their display names.
        Returns dict: {entry_key -> {param_name: value, ...}}
        """
        combinations = {}
        for entry in self.manifest['entries']:
            entry_key = tuple(sorted(entry['int_params'].items()))
            if entry_key not in combinations:
                combinations[entry_key] = dict(entry['int_params'])
        return combinations
    
    def _format_ground_truth_label(self, int_params_dict: Dict) -> str:
        """Format ground truth parameters as a readable label."""
        parts = [f"{k}={v}" for k, v in sorted(int_params_dict.items())]
        return ", ".join(parts)

    def plot_single_mode(self, inference_name: str):
        """..."""
        try:
            int_params_dict, ext_params_dict, uncertainties_dict, loss_dict = \
                self._extract_results_single_mode()
        except Exception as e:
            print(f"Error extracting single-mode results: {e}")
            return

        inferred_params = self._get_inferred_params()
        external_params = self._get_external_params()

        if not inferred_params or not external_params:
            print(f"Warning: No results found for {inference_name} (single mode)")
            return

        ext_param = external_params[0]

        # Create ONE figure with subplots for each inferred parameter
        fig = make_subplots(
            rows=len(inferred_params) + 1, cols=2,
            subplot_titles= ['L2 Inference Error', 'Total Variance'] + [
                title for inferred_param in inferred_params
                for title in (
                    f'({inferred_param} - GT) / GT vs {ext_param}',
                    f'σ / GT vs {ext_param}'
                )
            ],
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}]] + [[{'type': 'scatter'}, {'type': 'scatter'}] for _ in inferred_params]
        )

        colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]

        # Get unique internal parameters from the first entry
        first_entry = self.manifest['entries'][0]
        internal_param_names = list(first_entry['int_params'].keys())
        
        # Only add combined traces if there are multiple internal parameters
        if len(internal_param_names) > 1:
            ground_truth_combinations = self._get_ground_truth_combinations(inferred_params[0])
            
            for color_idx, entry_key in enumerate(sorted(ground_truth_combinations.keys())):
                color = colors[color_idx % len(colors)]
                label = self._format_ground_truth_label(dict(entry_key))
                
                # Collect external parameter values
                ext_vals_set = set()
                for inferred_param in inferred_params:
                    if inferred_param in int_params_dict and entry_key in int_params_dict[inferred_param]:
                        ext_vals_set.update(int_params_dict[inferred_param][entry_key].keys())
                
                ext_vals = sorted(ext_vals_set)
                
                # For each external value, sum across all inferred parameters
                sum_relative_error_squared = []
                sum_normalized_sigma_squared = []
                
                for ext_val in ext_vals:
                    sum_rel_err_squared = 0.0
                    sum_norm_sig_squared = 0.0
                    
                    for inferred_param in inferred_params:
                        if inferred_param not in int_params_dict or entry_key not in int_params_dict[inferred_param]:
                            continue
                        if ext_val not in int_params_dict[inferred_param][entry_key]:
                            continue
                        
                        int_params = dict(entry_key)
                        gt_value = int_params.get(inferred_param)
                        
                        if gt_value is None or gt_value == 0:
                            continue
                        
                        inferred_val = int_params_dict[inferred_param][entry_key][ext_val]
                        relative_error = (inferred_val - gt_value) / gt_value
                        sum_rel_err_squared += relative_error**2
                        
                        if inferred_param in uncertainties_dict and entry_key in uncertainties_dict[inferred_param]:
                            sigma = uncertainties_dict[inferred_param][entry_key].get(ext_val, np.nan)
                            if np.isfinite(sigma).all():
                                normalized_sigma = sigma / gt_value
                                sum_norm_sig_squared += normalized_sigma**2
                    
                    sum_relative_error_squared.append(sum_rel_err_squared)
                    sum_normalized_sigma_squared.append(sum_norm_sig_squared)
                
                ext_vals_array = np.array(ext_vals)
                sum_relative_error_squared = np.array(sum_relative_error_squared)
                sum_normalized_sigma_squared = np.array(sum_normalized_sigma_squared)
                
                # Add combined trace only to first row (column 1)
                fig.add_trace(
                    go.Scatter(
                        x=ext_vals_array,
                        y=sum_relative_error_squared,
                        mode='lines+markers',
                        name=f'{label} (Combined)',
                        line=dict(color=color, width=3, dash='solid'),
                        marker=dict(size=10),
                        legendgroup=label,
                        showlegend=True
                    ),
                    row=1, col=1
                )
                
                # Add combined trace only to first row (column 2)
                valid_indices = [i for i, s in enumerate(sum_normalized_sigma_squared) if np.isfinite(s) and s > 0]
                if valid_indices:
                    fig.add_trace(
                        go.Scatter(
                            x=ext_vals_array[valid_indices],
                            y=sum_normalized_sigma_squared[valid_indices],
                            mode='lines+markers',
                            name=f'{label} (Combined)',
                            line=dict(color=color, width=3, dash='solid'),
                            marker=dict(size=10),
                            legendgroup=label,
                            showlegend=False
                        ),
                        row=1, col=2
                    )

                fig.update_yaxes(type = "log", row = 1)
        
        # Then add individual parameter traces
        for row_idx, inferred_param in enumerate(inferred_params, start=2):
            if inferred_param not in int_params_dict:
                continue

            for color_idx, (entry_key, ext_param_data) in enumerate(sorted(int_params_dict[inferred_param].items())):
                color = colors[color_idx % len(colors)]

                int_params = dict(entry_key)
                gt_value = int_params.get(inferred_param)
                label = self._format_ground_truth_label(int_params)

                if gt_value is None or gt_value == 0:
                    print(f"Warning: Skipping normalization for {label} (GT value is {gt_value})")
                    continue

                ext_vals = []
                normalized_inferred_vals = []
                normalized_sigma_vals = []

                for ext_val in sorted(ext_param_data.keys()):
                    inferred_val = ext_param_data[ext_val]
                    ext_vals.append(ext_val)

                    normalized_val = (inferred_val - gt_value) / gt_value
                    normalized_inferred_vals.append(float(normalized_val))

                    if inferred_param in uncertainties_dict and entry_key in uncertainties_dict[inferred_param]:
                        sigma = uncertainties_dict[inferred_param][entry_key].get(ext_val, np.nan)
                        if np.isfinite(sigma).all():
                            normalized_sigma = sigma / gt_value
                            normalized_sigma_vals.append(float(normalized_sigma))
                        else:
                            normalized_sigma_vals.append(np.nan)
                    else:
                        normalized_sigma_vals.append(np.nan)

                ext_vals = np.array(ext_vals)
                normalized_inferred_vals = np.array(normalized_inferred_vals)
                normalized_sigma_vals = np.array(normalized_sigma_vals)

                fig.add_trace(
                    go.Scatter(
                        x=ext_vals,
                        y=normalized_inferred_vals,
                        mode='lines+markers',
                        name=label,
                        line=dict(color=color, width=2),
                        marker=dict(size=8),
                        legendgroup=label,
                        showlegend=(row_idx == 2)
                    ),
                    row=row_idx, col=1
                )

                valid_indices = [i for i, s in enumerate(normalized_sigma_vals) if np.isfinite(s)]
                if valid_indices:
                    fig.add_trace(
                        go.Scatter(
                            x=ext_vals[valid_indices],
                            y=normalized_sigma_vals[valid_indices],
                            mode='lines+markers',
                            name=label,
                            line=dict(color=color, width=2, dash='dash'),
                            marker=dict(size=8, symbol='square'),
                            legendgroup=label,
                            showlegend=False
                        ),
                        row=row_idx, col=2
                    )


        fig.update_layout(
            title_text=f'{inference_name} - Single Mode (Normalized by GT)',
            height=300 * len(inferred_params),
            width=1400,
            hovermode='x unified',
            template='plotly_white'
        )

        output_path = self.mode_path / f'Fig_single_all_params.html'
        fig.write_html(str(output_path))
        print(f"Saved: {output_path}")

    def plot_cumulative_mode(self, inference_name: str):
        """
        Plot cumulative-mode inference results using Plotly.
        Creates separate traces for each ground truth int_params combination.
        Each trace has multiple points (one per cumulative index).
        Inferred values are centered and normalized by their ground truth value.
        """
        try:
            int_params_dict, ext_params_dict, uncertainties_dict, cumul_indices_dict = \
                self._extract_results_cumulative_mode()
        except Exception as e:
            print(f"Error extracting cumulative-mode results: {e}")
            return

        inferred_params = self._get_inferred_params()
        external_params = self._get_external_params()

        if not inferred_params or not external_params:
            print(f"Warning: No results found for {inference_name} (cumulative mode)")
            return

        ext_param = external_params[0]

        fig = make_subplots(
            rows=len(inferred_params) + 1, cols=2,
            subplot_titles=tuple(
                ['Combined L2 Inference Error', 'Combined Total Variance'] +
                [
                    title for inferred_param in inferred_params
                    for title in (
                        f'({inferred_param} - GT) / GT vs # Cumulated {ext_param}',
                        f'σ / GT vs # Cumulated {ext_param}'
                    )
                ]
            ),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}]] + 
                [[{'type': 'scatter'}, {'type': 'scatter'}] for _ in inferred_params]
        )

        colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]

        first_entry = self.manifest['entries'][0]
        internal_param_names = list(first_entry['int_params'].keys())

        # Combined metrics across all inferred params (top row)
        if len(internal_param_names) > 1:
            ground_truth_combinations = self._get_ground_truth_combinations(inferred_params[0])

            for color_idx, entry_key in enumerate(sorted(ground_truth_combinations.keys())):
                color = colors[color_idx % len(colors)]
                label = self._format_ground_truth_label(dict(entry_key))

                num_params_set = set()
                for inferred_param in inferred_params:
                    if inferred_param in int_params_dict and entry_key in int_params_dict[inferred_param]:
                        num_params_set.update(int_params_dict[inferred_param][entry_key].keys())

                num_params = sorted(num_params_set)

                sum_relative_error_squared = []
                sum_normalized_sigma_squared = []

                for num_param in num_params:
                    sum_rel_err_squared = 0.0
                    sum_norm_sig_squared = 0.0

                    for inferred_param in inferred_params:
                        if inferred_param not in int_params_dict or entry_key not in int_params_dict[inferred_param]:
                            continue
                        if num_param not in int_params_dict[inferred_param][entry_key]:
                            continue

                        int_params = dict(entry_key)
                        gt_value = int_params.get(inferred_param)

                        if gt_value is None or gt_value == 0:
                            continue

                        inferred_val = int_params_dict[inferred_param][entry_key][num_param]
                        relative_error = (inferred_val - gt_value) / gt_value
                        sum_rel_err_squared += relative_error**2

                        if inferred_param in uncertainties_dict and entry_key in uncertainties_dict[inferred_param]:
                            sigma = uncertainties_dict[inferred_param][entry_key].get(num_param, np.nan)
                            if np.isfinite(sigma).all():
                                normalized_sigma = sigma / gt_value
                                sum_norm_sig_squared += normalized_sigma**2

                    sum_relative_error_squared.append(sum_rel_err_squared)
                    sum_normalized_sigma_squared.append(sum_norm_sig_squared)

                num_params_array = np.array(num_params)
                sum_relative_error_squared = np.array(sum_relative_error_squared)
                sum_normalized_sigma_squared = np.array(sum_normalized_sigma_squared)

                fig.add_trace(
                    go.Scatter(
                        x=num_params_array,
                        y=sum_relative_error_squared,
                        mode='lines+markers',
                        name=f'{label} (Combined)',
                        line=dict(color=color, width=3, dash='solid'),
                        marker=dict(size=10),
                        legendgroup=label,
                        showlegend=True
                    ),
                    row=1, col=1
                )

                valid_indices = [i for i, s in enumerate(sum_normalized_sigma_squared) if np.isfinite(s) and s > 0]
                if valid_indices:
                    fig.add_trace(
                        go.Scatter(
                            x=num_params_array[valid_indices],
                            y=sum_normalized_sigma_squared[valid_indices],
                            mode='lines+markers',
                            name=f'{label} (Combined)',
                            line=dict(color=color, width=3, dash='solid'),
                            marker=dict(size=10),
                            legendgroup=label,
                            showlegend=False
                        ),
                        row=1, col=2
                    )

                fig.update_yaxes(type = "log", row = 1)    

        # Individual inferred parameter rows
        for row_idx, inferred_param in enumerate(inferred_params, start=2):
            if inferred_param not in int_params_dict:
                continue

            for color_idx, (entry_key, cumul_index_data) in enumerate(sorted(int_params_dict[inferred_param].items())):
                color = colors[color_idx % len(colors)]

                int_params = dict(entry_key)
                gt_value = int_params.get(inferred_param)
                label = self._format_ground_truth_label(int_params)

                if gt_value is None or gt_value == 0:
                    print(f"Warning: Skipping normalization for {label} (GT value is {gt_value})")
                    continue

                num_params_list = []
                normalized_inferred_vals = []
                normalized_sigma_vals = []

                for num_params in sorted(cumul_index_data.keys()):
                    inferred_val = cumul_index_data[num_params]
                    num_params_list.append(num_params)

                    normalized_val = (inferred_val - gt_value) / gt_value
                    normalized_inferred_vals.append(float(normalized_val))

                    if inferred_param in uncertainties_dict and entry_key in uncertainties_dict[inferred_param]:
                        sigma = uncertainties_dict[inferred_param][entry_key].get(num_params, np.nan)
                        if np.isfinite(sigma).all():
                            normalized_sigma = sigma / gt_value
                            normalized_sigma_vals.append(float(normalized_sigma))
                        else:
                            normalized_sigma_vals.append(np.nan)
                    else:
                        normalized_sigma_vals.append(np.nan)

                num_params_list = np.array(num_params_list)
                normalized_inferred_vals = np.array(normalized_inferred_vals)
                normalized_sigma_vals = np.array(normalized_sigma_vals)

                fig.add_trace(
                    go.Scatter(
                        x=num_params_list,
                        y=normalized_inferred_vals,
                        mode='lines+markers',
                        name=label,
                        line=dict(color=color, width=2),
                        marker=dict(size=8),
                        legendgroup=label,
                        showlegend=(row_idx == 2)
                    ),
                    row=row_idx, col=1
                )

                valid_indices = [i for i, s in enumerate(normalized_sigma_vals) if np.isfinite(s)]
                if valid_indices:
                    fig.add_trace(
                        go.Scatter(
                            x=num_params_list[valid_indices],
                            y=normalized_sigma_vals[valid_indices],
                            mode='lines+markers',
                            name=label,
                            line=dict(color=color, width=2, dash='dash'),
                            marker=dict(size=8, symbol='square'),
                            legendgroup=label,
                            showlegend=False
                        ),
                        row=row_idx, col=2
                    )

            fig.update_xaxes(title_text=f'# Cumulated {ext_param}', row=row_idx, col=1)
            fig.update_yaxes(title_text=f'({inferred_param} - GT) / GT', row=row_idx, col=1)
            fig.update_xaxes(title_text=f'# Cumulated {ext_param}', row=row_idx, col=2)
            fig.update_yaxes(type='log', title_text=f'σ / GT', row=row_idx, col=2)

        fig.update_layout(
            title_text=f'{inference_name} - Cumulative Mode (Normalized by GT)',
            height=300 * (len(inferred_params) + 1),
            width=1400,
            hovermode='x unified',
            template='plotly_white'
        )

        output_path = self.mode_path / f'Fig_cumulative_all_params.html'
        fig.write_html(str(output_path))
        print(f"Saved: {output_path}")


    def plot_all(self, inference_name: str):
        """Generate all plots for this inference."""
        print(f"\nGenerating plots for {inference_name}...")
        self.plot_single_mode(inference_name)
        self.plot_cumulative_mode(inference_name)

def main():
    base_inference_path = Path(__file__).resolve().parent.parent / 'Inference' / 'FromSimulationData'
    
    inferences = [
        ('BendingElasticity', 'ElasticInference_BendingElasticity'),
        ('ShearElasticity', 'ElasticInference_ShearElasticity'),
        ('BendingShearElasticity', 'ElasticInference_BendingShearElasticity'),
    ]
    
    for inference_name, dir_name in inferences:
        inference_base_path = base_inference_path / dir_name
        
        for mode in ['SingleExtParams', 'CumulativeExtParams']:
            # Navigate to the mode subdirectory
            mode_path = inference_base_path / mode
            
            if not mode_path.exists():
                print(f"Warning: Path not found: {mode_path}")
                continue
            
            try:
                visualizer = InferenceResultsVisualizer(mode_path)
                if mode == 'SingleExtParams':
                    visualizer.plot_single_mode(inference_name)
                elif mode == 'CumulativeExtParams':
                    visualizer.plot_cumulative_mode(inference_name)

            except Exception as e:
                import traceback
                traceback.print_exc()



if __name__ == "__main__":
    main()
