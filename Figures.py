"""
Figures.py - Visualization of inference results across different elasticity modes.
"""

import json
import dill as pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

class InferenceResultsVisualizer:
    """Generalized visualizer for inference results."""
    
    def __init__(self, mode_path: Path):
        """
        Initialize visualizer with path to a specific inference mode (SingleExtParams or CumulativeExtParams).
        
        Args:
            mode_path: Path to SingleExtParams or CumulativeExtParams directory
        """
        self.mode_path = Path(mode_path)
        self.mode = self.mode_path.name  # 'SingleExtParams' or 'CumulativeExtParams'
        self.manifest = self._load_manifest()
        self.param_ref = self._load_param_reference()
    
    def _load_manifest(self) -> Dict:
        """Load results manifest."""
        manifest_file = self.mode_path / 'results_manifest.json'
        if not manifest_file.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_file}")
        with open(manifest_file, 'r') as f:
            return json.load(f)
    
    def _load_param_reference(self) -> Dict:
        """Load parameter reference file."""
        ref_file = self.mode_path / 'parameter_reference.json'
        if not ref_file.exists():
            raise FileNotFoundError(f"Parameter reference not found: {ref_file}")
        with open(ref_file, 'r') as f:
            return json.load(f)
    
    def _load_result_file(self, result_file_path: str):
        """Load an InferenceResult object from pickle."""
        full_path = self.mode_path / result_file_path
        with open(full_path, 'rb') as f:
            return pickle.load(f)
    
    def _extract_results_single_mode(self) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        Extract and organize single-mode results.
        
        Returns:
            Tuple of (int_params_dict, ext_params_dict, uncertainties_dict, loss_dict)
            where dicts map: {param_name: {entry_key: values}}
        """
        int_params_dict = {}
        ext_params_dict = {}
        uncertainties_dict = {}
        loss_dict = {}
        
        for entry in self.manifest['entries']:
            result = self._load_result_file(entry['result_file'])
            
            # Create a hashable key from ground truth int_params
            entry_key = tuple(sorted(entry['int_params'].items()))
            ext_key = tuple(sorted(entry['ext_params'].items())) if entry['ext_params'] else None
            
            # Store inferred internal parameters
            for param_name, param_value in result.params.items():
                if param_name not in int_params_dict:
                    int_params_dict[param_name] = {}
                int_params_dict[param_name][entry_key] = param_value
            
            # Store external parameters
            if entry['ext_params']:
                for param_name, param_value in entry['ext_params'].items():
                    if param_name not in ext_params_dict:
                        ext_params_dict[param_name] = {}
                    ext_params_dict[param_name][entry_key] = param_value
            
            # Store uncertainties (standard errors)
            if result.std_errors is not None:
                for i, param_name in enumerate(result.params.keys()):
                    if param_name not in uncertainties_dict:
                        uncertainties_dict[param_name] = {}
                    uncertainties_dict[param_name][entry_key] = result.std_errors[i]
            
            # Store loss
            loss_dict[entry_key] = result.loss
        
        return int_params_dict, ext_params_dict, uncertainties_dict, loss_dict
    
    def _extract_results_cumulative_mode(self) -> Tuple[Dict, Dict, Dict, Dict]:
        """
        Extract and organize cumulative-mode results.
        
        Returns:
            Tuple of (int_params_dict, ext_params_dict, uncertainties_dict, cumul_indices_dict)
        """
        int_params_dict = {}
        ext_params_dict = {}
        uncertainties_dict = {}
        cumul_indices_dict = {}
        
        for entry in self.manifest['entries']:
            result = self._load_result_file(entry['result_file'])
            
            # Create a hashable key from ground truth int_params
            entry_key = tuple(sorted(entry['int_params'].items()))
            pass_idx = entry['pass_idx']
            
            # Store inferred internal parameters
            for param_name, param_value in result.params.items():
                if param_name not in int_params_dict:
                    int_params_dict[param_name] = {}
                int_params_dict[param_name][entry_key] = param_value
            
            # Store uncertainties
            if result.std_errors is not None:
                for i, param_name in enumerate(result.params.keys()):
                    if param_name not in uncertainties_dict:
                        uncertainties_dict[param_name] = {}
                    uncertainties_dict[param_name][entry_key] = result.std_errors[i]
            
            # Store cumul_indices for reference
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
        first_entry = self.manifest['entries'][0]
        return list(first_entry['ext_params'].keys()) if first_entry['ext_params'] else []
    
    def _sort_by_external_param(self, ext_param: str, 
                                data_dict: Dict) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Sort data by a specific external parameter value.
        
        Args:
            ext_param: Name of external parameter to sort by
            data_dict: Dictionary mapping entry_key -> values for multiple parameters
        
        Returns:
            Tuple of (sorted_ext_values, {param_name: sorted_values})
        """
        # Extract external param values and organize by entry_key
        ext_values = {}
        for entry in self.manifest['entries']:
            entry_key = tuple(sorted(entry['int_params'].items()))
            if entry['ext_params'] and ext_param in entry['ext_params']:
                ext_values[entry_key] = entry['ext_params'][ext_param]
        
        # Sort by external parameter value
        sorted_keys = sorted(ext_values.keys(), key=lambda k: ext_values[k])
        sorted_ext_values = np.array([ext_values[k] for k in sorted_keys])
        
        sorted_data = {}
        for param_name, value_dict in data_dict.items():
            sorted_data[param_name] = np.array([value_dict[k] if np.isfinite(value_dict[k]) else np.inf for k in sorted_keys])
        
        return sorted_ext_values, sorted_data
    
    def plot_single_mode(self, inference_name: str, figsize: Tuple[int, int] = (16, 10)):
        """
        Plot single-mode inference results.
        
        Creates one figure per inferred parameter, showing:
        - Panel 1: Inferred value vs external parameter
        - Panel 2: Uncertainty vs external parameter
        """
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
        
        # Use primary external parameter (usually A or the first one)
        ext_param = external_params[0]
        
        for inferred_param in inferred_params:
            if inferred_param not in int_params_dict:
                continue
            
            fig = plt.figure(figsize=figsize)
            gs = GridSpec(1, 2, figure=fig, hspace=0.3, wspace=0.3)
            
            # Prepare data
            data_dict = {
                inferred_param: int_params_dict[inferred_param],
            }
            if inferred_param in uncertainties_dict:
                data_dict['sigma'] = uncertainties_dict[inferred_param]
            
            # Sort by external parameter
            ext_values, sorted_data = self._sort_by_external_param(ext_param, data_dict)
            inferred_values = sorted_data[inferred_param]
            sigmas = sorted_data.get('sigma', np.full_like(inferred_values, np.nan))
            
            # Panel 1: Inferred parameter vs external parameter
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.loglog(ext_values, inferred_values, 'o-', linewidth=2, markersize=8, label='Inferred value')
            ax1.set_xlabel(f'{ext_param}', fontsize=12, fontweight='bold')
            ax1.set_ylabel(f'{inferred_param}', fontsize=12, fontweight='bold')
            ax1.set_title(f'{inferred_param} vs {ext_param}', fontsize=13, fontweight='bold')
            ax1.grid(True, alpha=0.3, which='both')
            ax1.legend(fontsize=10)
            
            # Panel 2: Uncertainty vs external parameter
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.loglog(ext_values, sigmas, 's-', linewidth=2, markersize=8, color='red', label='σ')
            ax2.set_xlabel(f'{ext_param}', fontsize=12, fontweight='bold')
            ax2.set_ylabel(f'σ({inferred_param})', fontsize=12, fontweight='bold')
            ax2.set_title(f'Uncertainty vs {ext_param}', fontsize=13, fontweight='bold')
            ax2.grid(True, alpha=0.3, which='both')
            ax2.legend(fontsize=10)
            
            fig.suptitle(f'{inference_name} - Single Mode', fontsize=14, fontweight='bold', y=0.995)
            
            # Save figure
            output_path = self.mode_path / f'Fig_single_{inferred_param}.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {output_path}")
            plt.close(fig)
    
    def plot_cumulative_mode(self, inference_name: str, figsize: Tuple[int, int] = (16, 10)):
        """
        Plot cumulative-mode inference results.
        
        Creates one figure per inferred parameter, showing:
        - Panel 1: Inferred value vs number of cumulated external parameters
        - Panel 2: Uncertainty vs number of cumulated external parameters
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
        
        for inferred_param in inferred_params:
            if inferred_param not in int_params_dict:
                continue
            
            fig = plt.figure(figsize=figsize)
            gs = GridSpec(1, 2, figure=fig, hspace=0.3, wspace=0.3)
            
            # Prepare data: organize by number of cumulated parameters
            cumul_counts = {}
            cumul_inferred = {}
            cumul_sigmas = {}
            
            for entry_key in int_params_dict[inferred_param]:
                if entry_key not in cumul_indices_dict:
                    continue
                
                cumul_indices = cumul_indices_dict[entry_key]
                
                if ext_param in cumul_indices:
                    start, end = cumul_indices[ext_param]
                    num_params = end + 1  # end is inclusive
                    
                    inferred_val = int_params_dict[inferred_param][entry_key]
                    sigma = uncertainties_dict[inferred_param].get(entry_key, np.nan)
                    
                    if num_params not in cumul_counts:
                        cumul_counts[num_params] = []
                        cumul_inferred[num_params] = []
                        cumul_sigmas[num_params] = []
                    
                    cumul_counts[num_params].append(entry_key)
                    cumul_inferred[num_params].append(inferred_val)
                    cumul_sigmas[num_params].append(sigma)
            
            if not cumul_counts:
                print(f"Warning: No cumulative data for {inferred_param}")
                continue
            
            # Aggregate by averaging over multiple int_param combinations
            num_params_sorted = sorted(cumul_counts.keys())
            avg_inferred = [np.mean(cumul_inferred[n]) for n in num_params_sorted]
            std_inferred = [np.std(cumul_inferred[n]) for n in num_params_sorted]
            avg_sigmas = [np.mean(cumul_sigmas[n]) for n in num_params_sorted]
            std_sigmas = [np.std(cumul_sigmas[n]) for n in num_params_sorted]
            
            # Panel 1: Inferred parameter vs number of cumulated parameters
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.errorbar(num_params_sorted, avg_inferred, yerr=std_inferred, 
                         fmt='o-', linewidth=2, markersize=8, capsize=5, label='Inferred value')
            ax1.set_xlabel(f'# Cumulated {ext_param}', fontsize=12, fontweight='bold')
            ax1.set_ylabel(f'{inferred_param}', fontsize=12, fontweight='bold')
            ax1.set_title(f'{inferred_param} vs # Cumulated {ext_param}', fontsize=13, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend(fontsize=10)
            
            # Panel 2: Uncertainty vs number of cumulated parameters
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.errorbar(num_params_sorted, avg_sigmas, yerr=std_sigmas, 
                         fmt='s-', linewidth=2, markersize=8, capsize=5, color='red', label='σ')
            ax2.set_xlabel(f'# Cumulated {ext_param}', fontsize=12, fontweight='bold')
            ax2.set_ylabel(f'σ({inferred_param})', fontsize=12, fontweight='bold')
            ax2.set_title(f'Uncertainty vs # Cumulated {ext_param}', fontsize=13, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend(fontsize=10)
            
            fig.suptitle(f'{inference_name} - Cumulative Mode', fontsize=14, fontweight='bold', y=0.995)
            
            # Save figure
            output_path = self.mode_path / f'Fig_cumulative_{inferred_param}.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {output_path}")
            plt.close(fig)
    
    def plot_all(self, inference_name: str):
        """Generate all plots for this inference."""
        print(f"\nGenerating plots for {inference_name}...")
        self.plot_single_mode(inference_name)
        self.plot_cumulative_mode(inference_name)


def main():
    """Main execution: Generate plots for all inferences."""
    
    base_inference_path = Path(__file__).resolve().parent.parent / 'Inference' / 'FromSimulationData'
    
    inferences = [
        ('BendingElasticity', 'ElasticInference_BendingElasticity'),
        # ('ShearElasticity', 'ElasticInference_ShearElasticity'),
        # ('BendingShearElasticity', 'ElasticInference_BendingShearElasticity'),
    ]
    
    for inference_name, dir_name in inferences:

        for mode in ['SingleExtParams', 'CumulativeExtParams']:
            inference_path = base_inference_path / dir_name / mode
            
            if not inference_path.exists():
                print(f"Warning: Path not found: {inference_path}")
                continue
            
            try:
                visualizer = InferenceResultsVisualizer(inference_path)
                if 'SingleExtParams' in mode:
                    visualizer.plot_single_mode(inference_name)
                elif 'CumulativeExtParams' in mode:
                    visualizer.plot_cumulative_mode(inference_name)
                    
            except Exception as e:
                print(f"Error processing {inference_name}: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()
