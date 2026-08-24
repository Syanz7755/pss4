import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

OMEGA_COLUMN = 'omega (rad/s)'
SPECTRAL_COLUMN = 'spectral_flux (W/m2/K/(rad/s))'

def _empty_result_like(df: pd.DataFrame, temperature: float = None) -> pd.DataFrame:
    result = pd.DataFrame({
        OMEGA_COLUMN: df[OMEGA_COLUMN].values,
        'transmission': np.zeros(len(df)),
    })
    if temperature is not None:
        result[SPECTRAL_COLUMN] = np.zeros(len(df))
    return result

def _add_result_frame(accum: Optional[pd.DataFrame], df: pd.DataFrame,
                      temperature: float = None) -> pd.DataFrame:
    if accum is None:
        accum = _empty_result_like(df, temperature)
    merged = pd.merge(accum, df, on=OMEGA_COLUMN, how='outer',
                      suffixes=('_accum', '_new'))
    merged['transmission'] = (
        merged['transmission_accum'].fillna(0.0)
        + merged['transmission_new'].fillna(0.0)
    )
    keep_cols = [OMEGA_COLUMN, 'transmission']
    if temperature is not None:
        accum_col = f'{SPECTRAL_COLUMN}_accum'
        new_col = f'{SPECTRAL_COLUMN}_new'
        accum_values = merged[accum_col] if accum_col in merged.columns else 0.0
        new_values = merged[new_col] if new_col in merged.columns else 0.0
        if hasattr(accum_values, 'fillna'):
            accum_values = accum_values.fillna(0.0)
        if hasattr(new_values, 'fillna'):
            new_values = new_values.fillna(0.0)
        merged[SPECTRAL_COLUMN] = accum_values + new_values
        keep_cols.append(SPECTRAL_COLUMN)
    return merged[keep_cols].sort_values(OMEGA_COLUMN).reset_index(drop=True)

def _frame_to_result_dict(df: pd.DataFrame) -> Dict:
    return {col: df[col].values for col in df.columns}

def _integrate_trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, 'trapezoid'):
        return np.trapezoid(y, x)
    return np.trapz(y, x)

def aggregate_probe_results(out_dir: str, tasks: List[Tuple[str, list]], 
                           omega_grid: np.ndarray, temperature: float = None,
                           group_defs: Optional[List[Dict]] = None):
    """Aggregate results for each probe from all source contributions.
    
    Args:
        out_dir: Output directory containing task CSV files
        tasks: List of (task_id, task_layers) tuples
        omega_grid: Frequency grid
        temperature: Temperature if specified
        group_defs: Optional list of group definitions with 'name' and 'ids' keys
    """
    probe_frames = {}
    
    if group_defs is not None:
        return _aggregate_by_groups(out_dir, tasks, omega_grid, temperature, group_defs)
    
    for task_id, task_layers in tasks:
        probe_id = task_id
        
        task_file = os.path.join(out_dir, f"{task_id}.csv")
        if os.path.exists(task_file):
            df = pd.read_csv(task_file)
            probe_frames[probe_id] = _add_result_frame(probe_frames.get(probe_id), df, temperature)
    
    probe_results = {}
    for probe_id, df in probe_frames.items():
        output_file = os.path.join(out_dir, f"tot_{probe_id}.csv")
        df.to_csv(output_file, index=False, float_format='%.15e')
        probe_results[probe_id] = _frame_to_result_dict(df)
        print(f"Saved aggregated results to {output_file}")
    
    return probe_results

def _aggregate_by_groups(out_dir: str, tasks: List[Tuple[str, list]],
                         omega_grid: np.ndarray, temperature: float,
                         group_defs: List[Dict]) -> Dict:
    """Aggregate per-source-layer results by group definitions.
    
    For each probe P{x}, reads P{x}_S{y}.csv files for each source layer y,
    sums according to group defs, and outputs sum_{group_name}_P{x}.csv.
    """
    grouped_frames = {}
    
    for task_id, _ in tasks:
        probe_id = task_id  
        for group in group_defs:
            group_name = group['name']
            source_ids = group['ids']
            sum_key = f"sum_{group_name}_{probe_id}"
            
            for src_id in source_ids:
                src_task_id = f"{probe_id}_S{src_id}"
                src_file = os.path.join(out_dir, f"{src_task_id}.csv")
                if os.path.exists(src_file):
                    df = pd.read_csv(src_file)
                    grouped_frames[sum_key] = _add_result_frame(grouped_frames.get(sum_key), df, temperature)
                    print(f"  Added source S{src_id} contribution to {sum_key}")
                else:
                    print(f"  WARNING: Source file {src_file} not found")
    
    grouped_results = {}
    for sum_key, df in grouped_frames.items():
        output_file = os.path.join(out_dir, f"{sum_key}.csv")
        df.to_csv(output_file, index=False, float_format='%.15e')
        grouped_results[sum_key] = _frame_to_result_dict(df)
        print(f"Saved grouped result to {output_file}")
    
    tot_probe_results = {}
    for task_id, _ in tasks:
        probe_id = task_id
        tot_file = os.path.join(out_dir, f"{probe_id}.csv")
        if os.path.exists(tot_file):
            df = pd.read_csv(tot_file)
            df = df.sort_values(OMEGA_COLUMN).reset_index(drop=True)
            tot_probe_results[probe_id] = _frame_to_result_dict(df)
            output_file = os.path.join(out_dir, f"tot_{probe_id}.csv")
            df.to_csv(output_file, index=False, float_format='%.15e')
            print(f"Saved total probe result to {output_file}")
    
    all_results = {}
    all_results.update(grouped_results)
    all_results.update(tot_probe_results)
    
    return all_results

def plot_probe_results(out_dir: str, probe_results: Dict, temperature: float = None):
    """Generate plots for each probe's spectral flux.
    
    Args:
        out_dir: Output directory for plots
        probe_results: Dictionary of probe results
        temperature: Temperature if specified
    """
    for probe_id, results in probe_results.items():
        if temperature is not None and 'spectral_flux (W/m2/K/(rad/s))' in results:
            plt.figure(figsize=(10, 6))
            plt.plot(results['omega (rad/s)'], results['spectral_flux (W/m2/K/(rad/s))'])
            plt.xlabel('omega (rad/s)')
            plt.ylabel('spectral_flux (W/m2/K/(rad/s))')
            plt.title(f'Spectral Flux for {probe_id}')
            plt.xscale('log')
            plt.yscale('log')
            plt.grid(True)
            
            plot_file = os.path.join(out_dir, f"tot_{probe_id}_plot.png")
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved plot to {plot_file}")

def generate_final_report(out_dir: str, probe_results: Dict, args: dict, 
                         execution_time: float, parsed_layers: list):
    """Generate final report with high precision formatting.
    
    Args:
        out_dir: Output directory
        probe_results: Dictionary of probe results
        args: Command line arguments
        execution_time: Total execution time in seconds
        parsed_layers: Parsed layer information
    """
    report_file = os.path.join(out_dir, "final_report.txt")
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("FED Solver Final Report\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("PARAMETERS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Temperature: {args.get('temperature', 'Not specified')} K\n")
        f.write(f"All frequencies: {args.get('all_freq', False)}\n")
        f.write(f"Output directory: {args.get('out_dir', 'results/')}\n")
        f.write(f"Execution time: {execution_time:.6f} seconds\n\n")
        
        f.write("STACK GEOMETRY\n")
        f.write("-" * 40 + "\n")
        for layer in parsed_layers:
            f.write(f"Layer {layer['layer_id']}: {layer['material']}, "
                   f"thickness={layer['thickness']:.6e} m, "
                   f"source={layer['is_source']}, probe={layer['is_probe']}\n")
        f.write("\n")
        
        f.write("PROBE RESULTS\n")
        f.write("-" * 40 + "\n")
        
        for probe_id, results in probe_results.items():
            f.write(f"\n{probe_id}:\n")
            
            omega = results['omega (rad/s)']
            trans = results['transmission']
            
            f.write(f"  First 5 rows:\n")
            f.write(f"  {'omega (rad/s)':<20} {'transmission':<20}\n")
            for i in range(min(5, len(omega))):
                f.write(f"  {omega[i]:.15e}  {trans[i]:.15e}\n")
            
            f.write(f"\n  Last 5 rows:\n")
            f.write(f"  {'omega (rad/s)':<20} {'transmission':<20}\n")
            for i in range(max(0, len(omega)-5), len(omega)):
                f.write(f"  {omega[i]:.15e}  {trans[i]:.15e}\n")
            
            if 'spectral_flux (W/m2/K/(rad/s))' in results:
                spectral_flux = results['spectral_flux (W/m2/K/(rad/s))']
                integrated_flux = _integrate_trapezoid(spectral_flux, omega)
                f.write(f"\n  Integrated heat transfer coefficient: {integrated_flux:.15e} W/m²/K\n")
            
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"Saved final report to {report_file}")

def run_post_processing(out_dir: str, tasks: List[Tuple[str, list]], 
                        omega_grid: np.ndarray, args: dict, 
                        execution_time: float, parsed_layers: list,
                        group_defs: Optional[List[Dict]] = None):
    """Run complete post-processing pipeline.
    
    Args:
        out_dir: Output directory
        tasks: List of (task_id, task_layers) tuples
        omega_grid: Frequency grid
        args: Command line arguments
        execution_time: Total execution time
        parsed_layers: Layer information
        group_defs: Optional group definitions for source-layer grouping
    """
    temperature = args.get('temperature')
    
    probe_results = aggregate_probe_results(out_dir, tasks, omega_grid, temperature, group_defs=group_defs)
    
    if temperature is not None:
        plot_probe_results(out_dir, probe_results, temperature)
    
    generate_final_report(out_dir, probe_results, args, execution_time, parsed_layers)
