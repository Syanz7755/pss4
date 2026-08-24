import os
import json
import time
import numpy as np
import pandas as pd
from typing import Dict, Optional, List
import logging

from config_parser import parse_stack_file
from materials import load_materials_and_grid
from grid_builder import build_k_grid
from scattering import compute_kz
from fed_engine import classify_prop_evan_modes, compute_observables, compute_transmission
import tau_grid
from utils import c, setup_logger
from post_processing import run_post_processing

logger = logging.getLogger("fed_solver")

POLARIZATIONS = ('s', 'p')
OMEGA_COLUMN = 'omega (rad/s)'

def make_single_polarization_tau_array(pol: str, tau_values: np.ndarray) -> np.ndarray:
    """Return a two-row tau array with only one polarization populated."""
    tau_array = np.zeros((2, len(tau_values)))
    pol_index = POLARIZATIONS.index(pol)
    tau_array[pol_index] = tau_values
    return tau_array

def parse_metadata_items(items: Optional[List[str]]) -> Dict[str, str]:
    """Parse CLI metadata entries without giving them solver semantics."""
    metadata = {}
    for item in items or []:
        if '=' not in item:
            raise ValueError(f"Invalid metadata item '{item}'. Expected KEY=VALUE.")
        key, value = item.split('=', 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid metadata item '{item}'. Metadata key is empty.")
        metadata[key] = value.strip()
    return metadata

def parse_int_items(value: Optional[str]) -> Optional[List[int]]:
    """Parse comma-separated integers or a file containing integers."""
    if not value:
        return None
    if os.path.exists(value):
        with open(value, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        text = value
    items = []
    for token in text.replace('\n', ',').split(','):
        token = token.strip()
        if token:
            items.append(int(token))
    return items

def parse_float_file(path: Optional[str]) -> Optional[List[float]]:
    """Parse one angular frequency per line, or comma-separated values."""
    if not path:
        return None
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    values = []
    for token in text.replace('\n', ',').split(','):
        token = token.strip()
        if token:
            values.append(float(token))
    return values

def omega_key(omega: float) -> str:
    return f"{float(omega):.15e}"

def read_existing_omega_keys(path: str) -> set:
    if not os.path.exists(path):
        return set()
    df = pd.read_csv(path, usecols=[OMEGA_COLUMN])
    return {omega_key(value) for value in df[OMEGA_COLUMN].values}

def merge_result_csv(path: str, new_df: pd.DataFrame) -> pd.DataFrame:
    """Merge new omega rows with an existing result file, keeping sorted uniques."""
    frames = []
    if os.path.exists(path):
        frames.append(pd.read_csv(path))
    if not new_df.empty:
        frames.append(new_df)
    if frames:
        merged = pd.concat(frames, ignore_index=True)
        merged = merged.drop_duplicates(subset=[OMEGA_COLUMN], keep='last')
        merged = merged.sort_values(OMEGA_COLUMN).reset_index(drop=True)
    else:
        merged = new_df
    merged.to_csv(path, index=False, float_format='%.15e')
    return merged

def write_run_metadata(out_dir: str, input_path: str, args: dict, parsed_layers: List[dict],
                       group_defs: List[dict], omega_grid: np.ndarray,
                       k_par_grid: np.ndarray, gap_thickness: Optional[float]) -> None:
    """Write solver metadata while keeping external geometry as passive tags."""
    metadata = {
        'input_path': input_path,
        'solver_scope': 'local_1d_periodic_stack_fed',
        'external_geometry_metadata': args.get('metadata', {}),
        'temperature_K': args.get('temperature'),
        'all_freq': args.get('all_freq', False),
        'freq_skip': args.get('freq_skip', 1),
        'freq_offset': args.get('freq_offset', 0),
        'freq_indices': args.get('freq_indices'),
        'freq_values_file': args.get('freq_values_file'),
        'resume': args.get('resume', False),
        'requested_k_parallel_points': args.get('k_grid_points', 1000),
        'vacuum_gap_m': gap_thickness,
        'omega_points': int(len(omega_grid)),
        'omega_min_rad_s': float(omega_grid[0]),
        'omega_max_rad_s': float(omega_grid[-1]),
        'k_parallel_points': int(len(k_par_grid)),
        'k_parallel_min_1_m': float(k_par_grid[0]),
        'k_parallel_max_1_m': float(k_par_grid[-1]),
        'layers': parsed_layers,
        'groups': group_defs,
        'notes': [
            'pss4_project computes the local sidewall FED coefficient h_W.',
            'Device-scale geometry metadata is recorded here but not interpreted by the solver.',
        ],
    }
    path = os.path.join(out_dir, 'run_metadata.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

# Keep these names available from main.py for older callers, but delegate the
# implementation to the reusable tau_grid module.
compute_unit_cell_s_matrix = tau_grid.compute_unit_cell_s_matrix
compute_bloch_reflection_forward = tau_grid.compute_bloch_reflection_forward
compute_bloch_reflection_backward_local = tau_grid.compute_bloch_reflection_backward_local
compute_R_L_and_R_R_circular_shift = tau_grid.compute_R_L_and_R_R_circular_shift

def run_solver(input_path: str, args: dict):
    """Run the FED solver for all probe layers using unit cell invariance.
    
    Args:
        input_path: Path to input stack file
        args: Dictionary with temperature, all_freq, out_dir
    """
    start_time = time.time()
    
    temperature = args.get('temperature')
    all_freq = args.get('all_freq', False)
    freq_skip = args.get('freq_skip', 1)
    freq_offset = args.get('freq_offset', 0)
    freq_indices = args.get('freq_indices')
    freq_values_file = args.get('freq_values_file')
    freq_values = parse_float_file(freq_values_file)
    resume = args.get('resume', False)
    out_dir = args.get('out_dir', 'results/')
    
    os.makedirs(out_dir, exist_ok=True)
    logger = setup_logger(out_dir)
    
    logger.info(f"Parsing stack file: {input_path}")
    parsed_layers, group_defs = parse_stack_file(input_path)
    logger.info(f"Parsed {len(parsed_layers)} layers")
    if group_defs:
        logger.info(f"Parsed {len(group_defs)} group definitions:")
        for g in group_defs:
            logger.info(f"  Group '{g['name']}': ids={g['ids']}")
    else:
        logger.info("No group definitions found in input file")
    
    logger.info("Loading materials and frequency grid")
    omega_grid, material_eps = load_materials_and_grid(
        parsed_layers,
        all_freq,
        freq_skip=freq_skip,
        freq_offset=freq_offset,
        freq_indices=freq_indices,
        freq_values=freq_values,
    )
    logger.info(f"Frequency grid: {len(omega_grid)} points, range [{omega_grid[0]:.3e}, {omega_grid[-1]:.3e}] rad/s")
    if resume:
        logger.info("Resume mode enabled: existing omega rows in output CSVs will be skipped and merged")
    
    source_indices = []
    for i, layer in enumerate(parsed_layers):
        if layer['is_source']:
            source_indices.append(i)
    
    logger.info(f"Found {len(source_indices)} source layers at indices: {source_indices}")
    
    probe_indices = []
    for i, layer in enumerate(parsed_layers):
        if layer['is_probe']:
            probe_indices.append(i)
    
    logger.info(f"Found {len(probe_indices)} probe layers at indices: {probe_indices}")
    
    has_groups = len(group_defs) > 0
    if has_groups:
        logger.info("Group mode enabled: will compute per-source-layer contributions for grouping")
    
    # parse_stack_file already converts input thicknesses from micrometres to metres.
    # Keep this value in SI units; multiplying by 1e-6 again would corrupt the
    # near-field k-grid scaling by six orders of magnitude.
    gap_thickness = None
    for layer in parsed_layers:
        if layer['material'] == 'Vacuum':
            gap_thickness = layer['thickness']
            logger.info(f"Detected vacuum gap thickness: {gap_thickness:.3e} m")
            break
    
    logger.info("Building k_parallel grid")
    k_grid_points = args.get('k_grid_points', 1000)
    k_par_grid, k_weights = build_k_grid(
        omega_grid[0], gap_thickness, num_points=k_grid_points)
    logger.info(f"k_parallel grid: {len(k_par_grid)} points, range [{k_par_grid[0]:.3e}, {k_par_grid[-1]:.3e}] 1/m")
    if gap_thickness:
        logger.info(f"k_max * gap = {k_par_grid[-1] * gap_thickness:.1f} (should be >> 1 for near-field)")
    write_run_metadata(out_dir, input_path, args, parsed_layers, group_defs, omega_grid, k_par_grid, gap_thickness)
    
    for probe_idx in probe_indices:
        probe_layer = parsed_layers[probe_idx]
        probe_task_id = f"P{probe_layer['layer_id']}"
        
        logger.info(f"Running task: {probe_task_id}")
        logger.info(f"Probe layer: index={probe_idx}, material={probe_layer['material']}, thickness={probe_layer['thickness']}")
        
        per_source_results = {}
        for src_idx in source_indices:
            src_layer = parsed_layers[src_idx]
            src_task_id = f"{probe_task_id}_S{src_layer['layer_id']}"
            per_source_results[src_task_id] = {
                OMEGA_COLUMN: [],
                'transmission': [],
                'src_layer_idx': src_idx
            }
            if temperature is not None:
                per_source_results[src_task_id]['spectral_flux (W/m2/K/(rad/s))'] = []
        
        output_file_total = os.path.join(out_dir, f"{probe_task_id}.csv")
        existing_omega_keys = read_existing_omega_keys(output_file_total) if resume else set()
        if resume and existing_omega_keys:
            logger.info(f"{probe_task_id}: found {len(existing_omega_keys)} existing omega rows")
        total_results = {OMEGA_COLUMN: [], 'transmission': []}
        if temperature is not None:
            total_results['spectral_flux (W/m2/K/(rad/s))'] = []
        
        for i, omega in enumerate(omega_grid):
            if resume and omega_key(omega) in existing_omega_keys:
                continue
            tau_per_src = {}
            for src_idx in source_indices:
                tau_per_src[src_idx] = 0.0
            spectral_flux_per_src = {}
            for src_idx in source_indices:
                spectral_flux_per_src[src_idx] = 0.0
            tau_total = 0.0
            spectral_flux_total = 0.0
            
            if i == 0:
                logger.info(f"First frequency omega[0] = {omega:.6e} rad/s")
            
            for pol_idx, pol in enumerate(POLARIZATIONS):
                tau_array_total = np.zeros(len(k_par_grid))
                tau_arrays_per_src = {src_idx: np.zeros(len(k_par_grid)) for src_idx in source_indices}
                
                if i == 0 and pol_idx == 0:
                    logger.info(f"Polarization: {pol}")
                    
                    eps_vac = material_eps.get('Vacuum')
                    if eps_vac is not None:
                        if isinstance(eps_vac, np.ndarray):
                            eps_v = eps_vac[i]
                        else:
                            eps_v = eps_vac
                        propagating_mask, evanescent_mask, k_boundary = classify_prop_evan_modes(
                            k_par_grid, omega, eps_v)
                        logger.info(f"Critical k_par (k0*sqrt(eps_vac)) = {k_boundary:.6e} 1/m")
                        logger.info(f"k_par grid range: [{k_par_grid[0]:.6e}, {k_par_grid[-1]:.6e}] 1/m")
                        n_propagating = np.sum(propagating_mask)
                        n_evanescent = np.sum(evanescent_mask)
                        logger.info(f"Propagating modes: {n_propagating}, Evanescent modes: {n_evanescent}")
                
                for k_idx, k_par in enumerate(k_par_grid):
                    eps_probe = material_eps[probe_layer['material']]
                    if isinstance(eps_probe, np.ndarray):
                        eps_p = eps_probe[i]
                    else:
                        eps_p = eps_probe
                    
                    kz_probe = compute_kz(eps_p, omega, k_par, c)
                    d_gap = probe_layer['thickness']
                    
                    is_evanescent = np.imag(kz_probe) > 1e-6 * abs(np.real(kz_probe))
                    
                    debug_this = False
                    if i == 0:
                        if k_idx == 0:
                            debug_this = True
                            logger.info(f"Probe material: {probe_layer['material']}, eps[0] = {eps_p}")
                        elif k_idx == len(k_par_grid) - 1:
                            debug_this = True
                        elif k_idx == len(k_par_grid) // 2 and is_evanescent:
                            debug_this = True
                    
                    if debug_this:
                        logger.info(f"--- k_par[{k_idx}] = {k_par:.6e} (evanescent={is_evanescent}) ---")
                        logger.info(f"  kz_probe = {kz_probe}")
                        logger.info(f"  Im(kz_probe) = {np.imag(kz_probe):.6e}, d_gap = {d_gap:.6e} m")
                        logger.info(f"  exp(-2*Im(kz)*d) = {np.exp(-2 * np.imag(kz_probe) * d_gap):.6e}")
                    
                    R_L, R_R = tau_grid.compute_R_L_and_R_R_circular_shift(
                        parsed_layers, probe_idx, omega, k_par, pol, material_eps, i,
                        kz_gap=kz_probe, debug=debug_this)
                    
                    tau = compute_transmission(kz_probe, d_gap, R_L, R_R)
                    tau_array_total[k_idx] = tau
                    
                    if debug_this:
                        logger.info(f"  R_L = {R_L}, Im(R_L) = {np.imag(R_L):.6e}")
                        logger.info(f"  R_R = {R_R}, Im(R_R) = {np.imag(R_R):.6e}")
                        logger.info(f"  tau = {tau:.6e}")
                    
                    if has_groups:
                        n_sources = len(source_indices)
                        for s_idx, src_idx in enumerate(source_indices):
                            tau_arrays_per_src[src_idx][k_idx] = tau / n_sources
                
                if i == 0:
                    logger.info(f"tau_array stats: min={np.min(tau_array_total):.6e}, max={np.max(tau_array_total):.6e}, mean={np.mean(tau_array_total):.6e}")
                    
                    eps_vac = material_eps.get('Vacuum')
                    if eps_vac is not None:
                        if isinstance(eps_vac, np.ndarray):
                            eps_v = eps_vac[i]
                        else:
                            eps_v = eps_vac
                        propagating_mask, evanescent_mask, _ = classify_prop_evan_modes(
                            k_par_grid, omega, eps_v)
                        
                        if np.any(propagating_mask):
                            tau_prop = tau_array_total[propagating_mask]
                            logger.info(f"Propagating tau: min={np.min(tau_prop):.6e}, max={np.max(tau_prop):.6e}, mean={np.mean(tau_prop):.6e}")
                        if np.any(evanescent_mask):
                            tau_evan = tau_array_total[evanescent_mask]
                            logger.info(f"Evanescent tau: min={np.min(tau_evan):.6e}, max={np.max(tau_evan):.6e}, mean={np.mean(tau_evan):.6e}")
                
                obs_total = compute_observables(
                    omega,
                    temperature,
                    make_single_polarization_tau_array(pol, tau_array_total),
                    k_par_grid,
                    k_weights,
                )
                if i == 0:
                    logger.info(f"Observable transmission for pol {pol}: {obs_total['transmission']:.6e}")
                
                tau_total += obs_total['transmission']
                if temperature is not None:
                    spectral_flux_total += obs_total.get('spectral_flux', 0.0)
                
                if has_groups:
                    for src_idx in source_indices:
                        obs_src = compute_observables(
                            omega,
                            temperature,
                            make_single_polarization_tau_array(pol, tau_arrays_per_src[src_idx]),
                            k_par_grid,
                            k_weights,
                        )
                        tau_per_src[src_idx] += obs_src['transmission']
                        if temperature is not None:
                            spectral_flux_per_src[src_idx] += obs_src.get('spectral_flux', 0.0)
            
            if i == 0:
                logger.info(f"Total tau_total for omega[0]: {tau_total:.6e}")
            
            total_results[OMEGA_COLUMN].append(omega)
            total_results['transmission'].append(tau_total)
            if temperature is not None:
                total_results['spectral_flux (W/m2/K/(rad/s))'].append(spectral_flux_total)
            
            if has_groups:
                for src_idx in source_indices:
                    src_layer = parsed_layers[src_idx]
                    src_task_id = f"{probe_task_id}_S{src_layer['layer_id']}"
                    per_source_results[src_task_id][OMEGA_COLUMN].append(omega)
                    per_source_results[src_task_id]['transmission'].append(tau_per_src[src_idx])
                    if temperature is not None:
                        per_source_results[src_task_id]['spectral_flux (W/m2/K/(rad/s))'].append(spectral_flux_per_src[src_idx])
        
        df_total = pd.DataFrame(total_results)
        merged_total = merge_result_csv(output_file_total, df_total)
        logger.info(f"Saved total results to {output_file_total} ({len(df_total)} new, {len(merged_total)} total rows)")
        
        if has_groups:
            for src_task_id, src_data in per_source_results.items():
                df_src = pd.DataFrame({k: v for k, v in src_data.items() if k != 'src_layer_idx'})
                output_file_src = os.path.join(out_dir, f"{src_task_id}.csv")
                merged_src = merge_result_csv(output_file_src, df_src)
                logger.info(f"Saved source results to {output_file_src} ({len(df_src)} new, {len(merged_src)} total rows)")
            logger.info(f"Saved {len(per_source_results)} per-source-layer result files for probe {probe_task_id}")
    
    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(f"Total execution time: {execution_time:.2f} seconds")
    
    tasks = [(f"P{parsed_layers[i]['layer_id']}", []) for i in probe_indices]
    run_post_processing(out_dir, tasks, omega_grid, args, execution_time, parsed_layers,
                       group_defs=group_defs if has_groups else None)
    
    return {
        'execution_time': execution_time,
        'omega_grid': omega_grid,
        'tasks': tasks,
        'args': args
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="1D-Periodic Planar FED Solver")
    parser.add_argument("--temperature", type=float, default=None, help="Source temperature in Kelvin")
    parser.add_argument("--all-freq", action="store_true", default=False, help="Use full frequency grid")
    parser.add_argument("--freq-skip", type=int, default=1,
                        help="Use every Nth frequency from the selected material table")
    parser.add_argument("--freq-offset", type=int, default=0,
                        help="Starting offset for --freq-skip subsets; must satisfy 0 <= offset < skip")
    parser.add_argument("--freq-indices", type=str, default=None,
                        help="Comma-separated indices or a text file of indices into the selected material table")
    parser.add_argument("--freq-values-file", type=str, default=None,
                        help="Text file of explicit omega values in rad/s; each value must exist in the material tables")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Skip omega rows already present in output CSVs, then merge sorted unique outputs")
    parser.add_argument("--k-grid-points", type=int, default=1000,
                        help="Number of logarithmically spaced k_parallel integration points (default: 1000)")
    parser.add_argument("--out-dir", type=str, default="results/", help="Output directory")
    parser.add_argument("--meta", action="append", default=[], metavar="KEY=VALUE",
                        help="Passive run metadata tag; recorded in run_metadata.json but not used by the solver")
    parser.add_argument("input_path", type=str, help="Path to input stack file")
    
    args = parser.parse_args()
    
    args_dict = {
        'temperature': args.temperature,
        'all_freq': args.all_freq,
        'freq_skip': args.freq_skip,
        'freq_offset': args.freq_offset,
        'freq_indices': parse_int_items(args.freq_indices),
        'freq_values_file': args.freq_values_file,
        'resume': args.resume,
        'k_grid_points': args.k_grid_points,
        'out_dir': args.out_dir,
        'metadata': parse_metadata_items(args.meta),
    }
    
    run_solver(args.input_path, args_dict)
