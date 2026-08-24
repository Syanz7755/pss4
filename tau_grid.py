import logging
from typing import Dict, Iterable, List, Tuple

import numpy as np

from fed_engine import compute_prop_evan_boundary, compute_transmission
from scattering import compute_kz, fresnel_reflection_s, fresnel_transmission_s
from scattering import fresnel_reflection_p, fresnel_transmission_p
from scattering import redheffer_star, compute_bloch_reflection
from scattering import compute_propagation_s_matrix, compute_bloch_reflection_backward
from utils import c

logger = logging.getLogger("fed_solver")

POLARIZATIONS = ("s", "p")


def resolve_layer_eps(layer: Dict, material_eps: Dict[str, np.ndarray], freq_idx: int) -> complex:
    """Resolve a layer permittivity for one frequency index."""
    eps_array = material_eps[layer["material"]]
    if isinstance(eps_array, np.ndarray):
        return eps_array[freq_idx]
    return eps_array


def compute_unit_cell_s_matrix(layers: List[dict], omega: float, k_par: float, pol: str,
                               material_eps: dict, freq_idx: int) -> Tuple[complex, complex, complex, complex]:
    """Compute the S-matrix for a sequence of layers."""
    if len(layers) == 0:
        return (0.0, 1.0, 1.0, 0.0)

    def get_eps(layer):
        return resolve_layer_eps(layer, material_eps, freq_idx)

    def get_kz(eps):
        return compute_kz(eps, omega, k_par, c)

    def get_interface_s(eps_i, eps_j, kz_i, kz_j):
        if pol == "s":
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
        S_prop = compute_propagation_s_matrix(kz, layer["thickness"])

        if S_total is None:
            S_total = S_prop
        else:
            S_total = redheffer_star(S_total, S_prop)

        if i < len(layers) - 1:
            S_interface = get_interface_s(eps_list[i], eps_list[i + 1], kz_list[i], kz_list[i + 1])
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
    """Compute forward Bloch reflection for light incident from the left."""
    return compute_bloch_reflection(S_UC, kz_gap=kz_gap, debug=debug)


def compute_bloch_reflection_backward_local(S_UC: Tuple[complex, complex, complex, complex],
                                            kz_gap: complex = None,
                                            debug: bool = False) -> complex:
    """Compute backward Bloch reflection for light incident from the right."""
    return compute_bloch_reflection_backward(S_UC, kz_gap=kz_gap, debug=debug)


def compute_R_L_and_R_R_circular_shift(all_layers: List[dict], probe_idx: int,
                                       omega: float, k_par: float, pol: str,
                                       material_eps: dict, freq_idx: int,
                                       kz_gap: complex = None,
                                       debug: bool = False) -> Tuple[complex, complex]:
    """Compute left and right Bloch reflections using the circular shift algorithm."""
    n_layers = len(all_layers)

    if debug:
        logger.info(f"  Circular Shift Algorithm: probe_idx={probe_idx}, N={n_layers}")
        logger.info(f"  All layers: {[(l['layer_id'], l['material']) for l in all_layers]}")

    right_sequence = []
    for i in range(probe_idx + 1, n_layers):
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
    for i in range(probe_idx, n_layers):
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


def compute_tau_grid(parsed_layers: List[dict], probe_idx: int,
                     omega_grid: np.ndarray, material_eps: Dict[str, np.ndarray],
                     k_par_grid: np.ndarray,
                     polarizations: Iterable[str] = POLARIZATIONS,
                     debug_first_frequency: bool = False) -> Dict:
    """Compute tau(omega, k_parallel) arrays for one probe layer.

    Returns a dictionary with ``tau[pol]`` arrays shaped
    ``(len(omega_grid), len(k_par_grid))`` and a frequency-dependent
    ``prop_evan_boundary`` array for heatmap overlays.
    """
    probe_layer = parsed_layers[probe_idx]
    if not probe_layer["is_probe"]:
        raise ValueError(f"Layer index {probe_idx} is not marked as a probe layer")

    selected_pols = tuple(polarizations)
    invalid_pols = sorted(set(selected_pols) - set(POLARIZATIONS))
    if invalid_pols:
        raise ValueError(f"Invalid polarizations: {invalid_pols}")

    omega_grid = np.asarray(omega_grid, dtype=float)
    k_par_grid = np.asarray(k_par_grid, dtype=float)
    tau_by_pol = {
        pol: np.zeros((len(omega_grid), len(k_par_grid)), dtype=float)
        for pol in selected_pols
    }
    prop_evan_boundary = np.zeros(len(omega_grid), dtype=float)

    for freq_idx, omega in enumerate(omega_grid):
        eps_probe = resolve_layer_eps(probe_layer, material_eps, freq_idx)
        prop_evan_boundary[freq_idx] = compute_prop_evan_boundary(omega, eps_probe)

        for pol in selected_pols:
            for k_idx, k_par in enumerate(k_par_grid):
                kz_probe = compute_kz(eps_probe, omega, k_par, c)
                debug_this = debug_first_frequency and freq_idx == 0 and k_idx in (0, len(k_par_grid) - 1)
                R_L, R_R = compute_R_L_and_R_R_circular_shift(
                    parsed_layers, probe_idx, omega, k_par, pol, material_eps, freq_idx,
                    kz_gap=kz_probe, debug=debug_this)
                tau_by_pol[pol][freq_idx, k_idx] = compute_transmission(
                    kz_probe, probe_layer["thickness"], R_L, R_R)

    return {
        "omega": omega_grid,
        "k_parallel": k_par_grid,
        "tau": tau_by_pol,
        "polarizations": selected_pols,
        "prop_evan_boundary": prop_evan_boundary,
        "probe_layer_id": probe_layer["layer_id"],
    }
