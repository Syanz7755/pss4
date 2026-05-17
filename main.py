import os
import time
import numpy as np
import pandas as pd
from typing import Dict, Optional, List, Tuple
import logging

from config_parser import parse_stack_file
from materials import load_materials_and_grid
from grid_builder import build_k_grid
from scattering import compute_kz, fresnel_reflection_s, fresnel_transmission_s
from scattering import fresnel_reflection_p, fresnel_transmission_p
from scattering import redheffer_star, compute_bloch_reflection
from scattering import compute_interface_s_matrix, compute_propagation_s_matrix
from scattering import compute_bloch_reflection_backward
from fed_engine import compute_transmission, compute_observables
from utils import c, setup_logger
from post_processing import run_post_processing

logger = logging.getLogger("fed_solver")

def compute_unit_cell_s_matrix(layers: List[dict], omega: float, k_par: float, pol: str, 
                                material_eps: dict, freq_idx: int) -> Tuple[complex, complex, complex, complex]:
    """Compute the S-matrix for a sequence of layers.
    
    For a periodic structure, we compute:
    1. Propagation through each layer
    2. Interface S-matrix between consecutive layers
    3. Boundary interface (from last material to first material for periodic)
    
    Args:
        layers: list of layer dictionaries
        omega: angular frequency
        k_par: parallel wavevector
        pol: polarization ('s' or 'p')
        material_eps: dictionary of material permittivities
        freq_idx: frequency index for material permittivity array
    
    Returns:
        S_UC: unit cell S-matrix (R11, T12, T21, R22)
    """
    if len(layers) == 0:
        return (0.0, 1.0, 1.0, 0.0)
    
    def get_eps(layer):
        eps_array = material_eps[layer['material']]
        if isinstance(eps_array, np.ndarray):
            return eps_array[freq_idx]
        return eps_array
    
    def get_kz(eps):
        return compute_kz(eps, omega, k_par, c)
    
    def get_interface_s(eps_i, eps_j, kz_i, kz_j):
        if pol == 's':
            r_ij = fresnel_reflection_s(kz_i, kz_j)
            t_ij = fresnel_transmission_s(kz_i, kz_j)
            r_ji = fresnel_reflection_s(kz_j, kz_i)
            t_ji = fresnel_transmission_s(kz_j, kz_i)
        else:
            r_ij = fresnel_reflection_p(eps_i, eps_j, kz_i, kz_j)
            t_ij = fresnel_transmission_p(eps_i, eps_j, kz_i, kz_j)
            r_ji = fresnel_reflection_p(eps_j, eps_i, kz_j, kz_i)
            t_ji = fresnel_transmission_p(eps_j, eps_i, kz_j, kz_i)
        return (r_ij, t_ij, t_ji, r_ji)
    
    eps_list = [get_eps(layer) for layer in layers]
    kz_list = [get_kz(eps) for eps in eps_list]
    
    S_total = None
    
    for i, layer in enumerate(layers):
        kz = kz_list[i]
        S_prop = compute_propagation_s_matrix(kz, layer['thickness'])
        
        if S_total is None:
            S_total = S_prop
        else:
            S_total = redheffer_star(S_total, S_prop)
        
        if i < len(layers) - 1:
            S_interface = get_interface_s(eps_list[i], eps_list[i+1], kz_list[i], kz_list[i+1])
            S_total = redheffer_star(S_total, S_interface)
    
    if len(layers) > 1:
        first_eps = eps_list[0]
        last_eps = eps_list[-1]
        first_kz = kz_list[0]
        last_kz = kz_list[-1]
        
        S_boundary = get_interface_s(last_eps, first_eps, last_kz, first_kz)
        S_total = redheffer_star(S_total, S_boundary)
    
    if S_total is None:
        return (0.0, 1.0, 1.0, 0.0)
    
    return S_total

def compute_bloch_reflection_forward(S_UC: Tuple[complex, complex, complex, complex],
                                      kz_gap: complex = None,
                                      debug: bool = False) -> complex:
    """Compute forward Bloch reflection: R_R = R11 + T12 * R_R * (1 - R22 * R_R)^(-1) * T21
    
    This is the standard Bloch reflection for light incident from the left.
    Solves the quadratic: R22 * R^2 + (T12*T21 - R11*R22 - 1) * R + R11 = 0
    """
    return compute_bloch_reflection(S_UC, kz_gap=kz_gap, debug=debug)

def compute_bloch_reflection_backward_local(S_UC: Tuple[complex, complex, complex, complex],
                                             kz_gap: complex = None,
                                             debug: bool = False) -> complex:
    """Compute backward Bloch reflection: R_L = R22 + T21 * R_L * (1 - R11 * R_L)^(-1) * T12
    
    This is the Bloch reflection for light incident from the right.
    Solves the quadratic: R11 * R^2 + (T12*T21 - R11*R22 - 1) * R + R22 = 0
    """
    return compute_bloch_reflection_backward(S_UC, kz_gap=kz_gap, debug=debug)

def compute_R_L_and_R_R_circular_shift(all_layers: List[dict], probe_idx: int, 
                                        omega: float, k_par: float, pol: str,
                                        material_eps: dict, freq_idx: int,
                                        kz_gap: complex = None,
                                        debug: bool = False) -> Tuple[complex, complex]:
    """Compute R_L and R_R using Circular Shift Algorithm.
    
    For a probe at index p in a unit cell of length N:
    - R_R: shifted sequence = Layer(p+1) to Layer(N-1), then Layer(0) to Layer(p)
    - R_L: shifted sequence = Layer(p) to Layer(N-1), then Layer(0) to Layer(p-1)
    
    Args:
        all_layers: list of all layer dictionaries in the unit cell
        probe_idx: index of the probe layer
        omega: angular frequency
        k_par: parallel wavevector
        pol: polarization ('s' or 'p')
        material_eps: dictionary of material permittivities
        freq_idx: frequency index
        kz_gap: z-component of wavevector in the gap (for root selection)
        debug: if True, print debug information
    
    Returns:
        (R_L, R_R): tuple of complex reflection coefficients
    """
    N = len(all_layers)
    
    if debug:
        logger.info(f"  Circular Shift Algorithm: probe_idx={probe_idx}, N={N}")
        logger.info(f"  All layers: {[(l['layer_id'], l['material']) for l in all_layers]}")
    
    right_sequence = []
    for i in range(probe_idx + 1, N):
        right_sequence.append(all_layers[i])
    for i in range(0, probe_idx + 1):
        right_sequence.append(all_layers[i])
    
    if debug:
        logger.info(f"  Right sequence (for R_R): {[(l['layer_id'], l['material']) for l in right_sequence]}")
    
    S_right = compute_unit_cell_s_matrix(right_sequence, omega, k_par, pol, material_eps, freq_idx)
    R_R = compute_bloch_reflection_forward(S_right, kz_gap=kz_gap, debug=debug)
    
    if debug:
        R11, T12, T21, R22 = S_right
        logger.info(f"  S_right: R11={R11:.6e}, T12={T12:.6e}, T21={T21:.6e}, R22={R22:.6e}")
        logger.info(f"  R_R (forward Bloch) = {R_R:.6e}, Im(R_R)={np.imag(R_R):.6e}")
    
    left_sequence = []
    for i in range(probe_idx, N):
        left_sequence.append(all_layers[i])
    for i in range(0, probe_idx):
        left_sequence.append(all_layers[i])
    
    if debug:
        logger.info(f"  Left sequence (for R_L): {[(l['layer_id'], l['material']) for l in left_sequence]}")
    
    S_left = compute_unit_cell_s_matrix(left_sequence, omega, k_par, pol, material_eps, freq_idx)
    R_L = compute_bloch_reflection_backward_local(S_left, kz_gap=kz_gap, debug=debug)
    
    if debug:
        R11, T12, T21, R22 = S_left
        logger.info(f"  S_left: R11={R11:.6e}, T12={T12:.6e}, T21={T21:.6e}, R22={R22:.6e}")
        logger.info(f"  R_L (backward Bloch) = {R_L:.6e}, Im(R_L)={np.imag(R_L):.6e}")
    
    return R_L, R_R

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
    omega_grid, material_eps = load_materials_and_grid(parsed_layers, all_freq, freq_skip=freq_skip)
    logger.info(f"Frequency grid: {len(omega_grid)} points, range [{omega_grid[0]:.3e}, {omega_grid[-1]:.3e}] rad/s")
    
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
    
    logger.info("Building k_parallel grid")
    k_par_grid, k_weights = build_k_grid(omega_grid[0])
    logger.info(f"k_parallel grid: {len(k_par_grid)} points, range [{k_par_grid[0]:.3e}, {k_par_grid[-1]:.3e}] 1/m")
    
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
                'omega (rad/s)': omega_grid,
                'transmission': [],
                'src_layer_idx': src_idx
            }
            if temperature is not None:
                per_source_results[src_task_id]['spectral_flux (W/m2/K/(rad/s))'] = []
        
        total_results = {'omega (rad/s)': omega_grid, 'transmission': []}
        if temperature is not None:
            total_results['spectral_flux (W/m2/K/(rad/s))'] = []
        
        for i, omega in enumerate(omega_grid):
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
            
            for pol_idx, pol in enumerate(['s', 'p']):
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
                        k0 = omega / c
                        k_critical = k0 * np.sqrt(eps_v)
                        logger.info(f"Critical k_par (k0*sqrt(eps_vac)) = {k_critical:.6e} 1/m")
                        logger.info(f"k_par grid range: [{k_par_grid[0]:.6e}, {k_par_grid[-1]:.6e}] 1/m")
                        n_propagating = np.sum(k_par_grid < np.real(k_critical))
                        n_evanescent = np.sum(k_par_grid >= np.real(k_critical))
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
                    
                    R_L, R_R = compute_R_L_and_R_R_circular_shift(
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
                        k0 = omega / c
                        k_critical = k0 * np.sqrt(eps_v)
                        
                        propagating_mask = k_par_grid < np.real(k_critical)
                        evanescent_mask = k_par_grid >= np.real(k_critical)
                        
                        if np.any(propagating_mask):
                            tau_prop = tau_array_total[propagating_mask]
                            logger.info(f"Propagating tau: min={np.min(tau_prop):.6e}, max={np.max(tau_prop):.6e}, mean={np.mean(tau_prop):.6e}")
                        if np.any(evanescent_mask):
                            tau_evan = tau_array_total[evanescent_mask]
                            logger.info(f"Evanescent tau: min={np.min(tau_evan):.6e}, max={np.max(tau_evan):.6e}, mean={np.mean(tau_evan):.6e}")
                
                obs_total = compute_observables(omega, temperature, 
                                         np.array([tau_array_total, tau_array_total]),
                                         k_par_grid, k_weights)
                if i == 0:
                    logger.info(f"Observable transmission for pol {pol}: {obs_total['transmission']:.6e}")
                
                tau_total += obs_total['transmission']
                if temperature is not None:
                    spectral_flux_total += obs_total.get('spectral_flux', 0.0)
                
                if has_groups:
                    for src_idx in source_indices:
                        obs_src = compute_observables(omega, temperature,
                                               np.array([tau_arrays_per_src[src_idx], tau_arrays_per_src[src_idx]]),
                                               k_par_grid, k_weights)
                        tau_per_src[src_idx] += obs_src['transmission']
                        if temperature is not None:
                            spectral_flux_per_src[src_idx] += obs_src.get('spectral_flux', 0.0)
            
            if i == 0:
                logger.info(f"Total tau_total for omega[0]: {tau_total:.6e}")
            
            total_results['transmission'].append(tau_total)
            if temperature is not None:
                total_results['spectral_flux (W/m2/K/(rad/s))'].append(spectral_flux_total)
            
            if has_groups:
                for src_idx in source_indices:
                    src_layer = parsed_layers[src_idx]
                    src_task_id = f"{probe_task_id}_S{src_layer['layer_id']}"
                    per_source_results[src_task_id]['transmission'].append(tau_per_src[src_idx])
                    if temperature is not None:
                        per_source_results[src_task_id]['spectral_flux (W/m2/K/(rad/s))'].append(spectral_flux_per_src[src_idx])
        
        df_total = pd.DataFrame(total_results)
        output_file_total = os.path.join(out_dir, f"{probe_task_id}.csv")
        df_total.to_csv(output_file_total, index=False, float_format='%.15e')
        logger.info(f"Saved total results to {output_file_total}")
        
        if has_groups:
            for src_task_id, src_data in per_source_results.items():
                df_src = pd.DataFrame({k: v for k, v in src_data.items() if k != 'src_layer_idx'})
                output_file_src = os.path.join(out_dir, f"{src_task_id}.csv")
                df_src.to_csv(output_file_src, index=False, float_format='%.15e')
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
    parser.add_argument("--out-dir", type=str, default="results/", help="Output directory")
    parser.add_argument("input_path", type=str, help="Path to input stack file")
    
    args = parser.parse_args()
    
    args_dict = {
        'temperature': args.temperature,
        'all_freq': args.all_freq,
        'out_dir': args.out_dir
    }
    
    run_solver(args.input_path, args_dict)
