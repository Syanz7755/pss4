import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger("fed_solver")

def compute_kz(eps: complex, omega: float, k_par: float, c: float = 299792458.0) -> complex:
    """Compute z-component of wavevector with strict branch cut.
    
    Enforces Im(kz) >= 0. If Im(kz) == 0, enforces Re(kz) >= 0.
    """
    k0 = omega / c
    kz_sq = eps * k0**2 - k_par**2
    kz = np.sqrt(kz_sq)
    
    if np.imag(kz) < 0:
        kz = -kz
    elif np.imag(kz) == 0 and np.real(kz) < 0:
        kz = -kz
    
    return kz

def fresnel_reflection_s(kz_i: complex, kz_j: complex) -> complex:
    """Compute s-polarization Fresnel reflection coefficient at interface i->j."""
    r_s = (kz_i - kz_j) / (kz_i + kz_j)
    return r_s

def fresnel_transmission_s(kz_i: complex, kz_j: complex) -> complex:
    """Compute s-polarization Fresnel transmission coefficient at interface i->j."""
    t_s = 2 * kz_i / (kz_i + kz_j)
    return t_s

def fresnel_reflection_p(eps_i: complex, eps_j: complex, kz_i: complex, kz_j: complex) -> complex:
    """Compute p-polarization Fresnel reflection coefficient at interface i->j."""
    r_p = (eps_j * kz_i - eps_i * kz_j) / (eps_j * kz_i + eps_i * kz_j)
    return r_p

def fresnel_transmission_p(eps_i: complex, eps_j: complex, kz_i: complex, kz_j: complex) -> complex:
    """Compute p-polarization Fresnel transmission coefficient at interface i->j."""
    t_p = 2 * np.sqrt(eps_i * eps_j) * kz_i / (eps_j * kz_i + eps_i * kz_j)
    return t_p

def compute_layer_s_matrix(eps: complex, thickness: float, omega: float, k_par: float, 
                           polarization: str, c: float = 299792458.0) -> Tuple[complex, complex, complex, complex]:
    """Compute S-matrix for a single layer.
    
    Returns (R11, T12, T21, R22) where:
    - R11: reflection from left
    - T12: transmission left to right
    - T21: transmission right to left
    - R22: reflection from right
    """
    kz = compute_kz(eps, omega, k_par, c)
    phase = np.exp(1j * kz * thickness)
    
    R11 = 0.0
    T12 = phase
    T21 = phase
    R22 = 0.0
    
    return (R11, T12, T21, R22)

def redheffer_star(S_A: Tuple[complex, complex, complex, complex], 
                   S_B: Tuple[complex, complex, complex, complex]) -> Tuple[complex, complex, complex, complex]:
    """Combine two S-matrices using Redheffer Star Product.
    
    S_A = (R11_A, T12_A, T21_A, R22_A)
    S_B = (R11_B, T12_B, T21_B, R22_B)
    
    Returns combined S-matrix (R11, T12, T21, R22).
    """
    R11_A, T12_A, T21_A, R22_A = S_A
    R11_B, T12_B, T21_B, R22_B = S_B
    
    denom1 = 1 - R22_A * R11_B
    denom2 = 1 - R11_B * R22_A
    denom3 = 1 - R11_A * R22_B
    denom4 = 1 - R22_B * R11_A
    
    R11 = R11_A + T12_A * R11_B * (1 / denom1) * T21_A
    T12 = T12_A * (1 / denom2) * T12_B
    T21 = T21_B * (1 / denom3) * T21_A
    R22 = R22_B + T21_B * R22_A * (1 / denom4) * T12_B
    
    return (R11, T12, T21, R22)

def compute_interface_s_matrix(kz_i: complex, kz_j: complex, eps_i: complex = None, 
                                eps_j: complex = None, polarization: str = 's') -> Tuple[complex, complex, complex, complex]:
    """Compute S-matrix for an interface from medium i to medium j.
    
    Returns (R11, T12, T21, R22) where:
    - R11: reflection from i side
    - T12: transmission from i to j
    - T21: transmission from j to i
    - R22: reflection from j side
    """
    if polarization == 's':
        r_ij = fresnel_reflection_s(kz_i, kz_j)
        t_ij = fresnel_transmission_s(kz_i, kz_j)
        r_ji = fresnel_reflection_s(kz_j, kz_i)
        t_ji = fresnel_transmission_s(kz_j, kz_i)
    else:  # p-polarization
        r_ij = fresnel_reflection_p(eps_i, eps_j, kz_i, kz_j)
        t_ij = fresnel_transmission_p(eps_i, eps_j, kz_i, kz_j)
        r_ji = fresnel_reflection_p(eps_j, eps_i, kz_j, kz_i)
        t_ji = fresnel_transmission_p(eps_j, eps_i, kz_j, kz_i)
    
    R11 = r_ij
    T12 = t_ij
    T21 = t_ji
    R22 = r_ji
    
    return (R11, T12, T21, R22)

def compute_propagation_s_matrix(kz: complex, thickness: float) -> Tuple[complex, complex, complex, complex]:
    """Compute S-matrix for propagation through a layer.
    
    Returns (R11, T12, T21, R22) where R11=R22=0 and T12=T21=phase.
    """
    phase = np.exp(1j * kz * thickness)
    return (0.0, phase, phase, 0.0)

def compute_layer_s_matrix_full(eps: complex, thickness: float, omega: float, k_par: float,
                                 polarization: str, c: float = 299792458.0,
                                 eps_prev: complex = None, eps_next: complex = None,
                                 kz_prev: complex = None, kz_next: complex = None) -> Tuple[complex, complex, complex, complex]:
    """Compute full S-matrix for a layer including interface effects.
    
    This combines: front interface + propagation + back interface.
    
    Args:
        eps: permittivity of this layer
        thickness: layer thickness
        omega: angular frequency
        k_par: parallel wavevector
        polarization: 's' or 'p'
        c: speed of light
        eps_prev: permittivity of previous layer (for front interface)
        eps_next: permittivity of next layer (for back interface)
        kz_prev: kz of previous layer
        kz_next: kz of next layer
    
    Returns:
        (R11, T12, T21, R22) S-matrix tuple
    """
    kz = compute_kz(eps, omega, k_par, c)
    phase = np.exp(1j * kz * thickness)
    
    R11 = 0.0
    T12 = phase
    T21 = phase
    R22 = 0.0
    
    return (R11, T12, T21, R22)

def compute_bloch_reflection(S_UC: Tuple[complex, complex, complex, complex], 
                             kz_gap: complex = None,
                             tol: float = 1e-8, max_iter: int = 1000,
                             prefer_positive_imag: bool = True,
                             debug: bool = False) -> complex:
    """Compute reflection of infinite periodic stack using analytical solution.
    
    For a unit cell S-matrix, the Bloch reflection can be computed analytically.
    
    The equation is: R_inf = R11 + T12 * R_inf * (1 - R22 * R_inf)^(-1) * T21
    
    This can be rewritten as a quadratic equation:
    R22 * R_inf^2 + (T12*T21 - R11*R22 - 1) * R_inf + R11 = 0
    
    Root selection logic based on kz_gap:
    - For evanescent waves (Im(kz_gap) > 0): select root with larger Im(R)
      Physical meaning: energy absorbed by medium, Im(R) > 0
    - For propagating waves (Im(kz_gap) ≈ 0): select root with smaller |R|
      Physical meaning: energy conservation, |R| ≤ 1
    
    Args:
        S_UC: unit cell S-matrix (R11, T12, T21, R22)
        kz_gap: z-component of wavevector in the gap (used for root selection)
        tol: tolerance for convergence (unused in analytical solution)
        max_iter: maximum iterations (unused in analytical solution)
        prefer_positive_imag: legacy parameter, ignored if kz_gap is provided
        debug: if True, log debug information
    
    Returns:
        R_inf: converged reflection coefficient
    """
    R11, T12, T21, R22 = S_UC
    
    if abs(R22) < 1e-15:
        if abs(T12 * T21 - 1) < 1e-10:
            R_inf = R11
        else:
            R_inf = R11 / (1 - T12 * T21)
        if debug:
            logger.info(f"    [Bloch] Degenerate case: R22≈0, returning R_inf={R_inf:.6e}")
        return R_inf
    
    a = R22
    b = T12 * T21 - R11 * R22 - 1
    c_coef = R11
    
    discriminant = b * b - 4 * a * c_coef
    sqrt_disc = np.sqrt(discriminant + 0j)
    
    R1 = (-b + sqrt_disc) / (2 * a)
    R2 = (-b - sqrt_disc) / (2 * a)
    
    im1 = np.imag(R1)
    im2 = np.imag(R2)
    abs1 = abs(R1)
    abs2 = abs(R2)
    
    if kz_gap is not None:
        imag_kz = np.imag(kz_gap)
        real_kz = np.real(kz_gap)
        is_evanescent = imag_kz > 1e-6 * abs(real_kz)
        
        if is_evanescent:
            if im1 > im2:
                R_inf = R1
                if debug:
                    logger.info(f"    [Bloch] EVANESCENT: Im(kz)={imag_kz:.6e}>0, selecting R with larger Im(R)")
                    logger.info(f"    [Bloch] R1={R1:.6e} (Im={im1:.6e}, |R|={abs1:.6e})")
                    logger.info(f"    [Bloch] R2={R2:.6e} (Im={im2:.6e}, |R|={abs2:.6e})")
                    logger.info(f"    [Bloch] Selected R1 (Im(R1)={im1:.6e} > Im(R2)={im2:.6e})")
            else:
                R_inf = R2
                if debug:
                    logger.info(f"    [Bloch] EVANESCENT: Im(kz)={imag_kz:.6e}>0, selecting R with larger Im(R)")
                    logger.info(f"    [Bloch] R1={R1:.6e} (Im={im1:.6e}, |R|={abs1:.6e})")
                    logger.info(f"    [Bloch] R2={R2:.6e} (Im={im2:.6e}, |R|={abs2:.6e})")
                    logger.info(f"    [Bloch] Selected R2 (Im(R2)={im2:.6e} > Im(R1)={im1:.6e})")
        else:
            if abs1 < abs2:
                R_inf = R1
                if debug:
                    logger.info(f"    [Bloch] PROPAGATING: Im(kz)={imag_kz:.6e}≈0, selecting R with smaller |R|")
                    logger.info(f"    [Bloch] R1={R1:.6e} (Im={im1:.6e}, |R|={abs1:.6e})")
                    logger.info(f"    [Bloch] R2={R2:.6e} (Im={im2:.6e}, |R|={abs2:.6e})")
                    logger.info(f"    [Bloch] Selected R1 (|R1|={abs1:.6e} < |R2|={abs2:.6e})")
            else:
                R_inf = R2
                if debug:
                    logger.info(f"    [Bloch] PROPAGATING: Im(kz)={imag_kz:.6e}≈0, selecting R with smaller |R|")
                    logger.info(f"    [Bloch] R1={R1:.6e} (Im={im1:.6e}, |R|={abs1:.6e})")
                    logger.info(f"    [Bloch] R2={R2:.6e} (Im={im2:.6e}, |R|={abs2:.6e})")
                    logger.info(f"    [Bloch] Selected R2 (|R2|={abs2:.6e} < |R1|={abs1:.6e})")
        
        return R_inf
    
    if prefer_positive_imag:
        if abs(abs(R1) - 1) < 1e-6 and abs(abs(R2) - 1) < 1e-6:
            if im1 >= 0 and im2 < 0:
                return R1
            elif im2 >= 0 and im1 < 0:
                return R2
            elif im1 >= im2:
                return R1
            else:
                return R2
    
    if abs(R1) <= 1 and abs(R2) > 1:
        return R1
    elif abs(R2) <= 1 and abs(R1) > 1:
        return R2
    elif abs(R1) <= abs(R2):
        return R1
    else:
        return R2


def compute_bloch_reflection_backward(S_UC: Tuple[complex, complex, complex, complex],
                                       kz_gap: complex = None,
                                       debug: bool = False) -> complex:
    """Compute backward Bloch reflection: R_L = R22 + T21 * R_L * (1 - R11 * R_L)^(-1) * T12
    
    This is the Bloch reflection for light incident from the right.
    Solves the quadratic: R11 * R^2 + (T12*T21 - R11*R22 - 1) * R + R22 = 0
    
    Root selection logic based on kz_gap:
    - For evanescent waves (Im(kz_gap) > 0): select root with larger Im(R)
      Physical meaning: energy absorbed by medium, Im(R) > 0
    - For propagating waves (Im(kz_gap) ≈ 0): select root with smaller |R|
      Physical meaning: energy conservation, |R| ≤ 1
    
    Args:
        S_UC: unit cell S-matrix (R11, T12, T21, R22)
        kz_gap: z-component of wavevector in the gap (used for root selection)
        debug: if True, log debug information
    
    Returns:
        R_inf: converged reflection coefficient
    """
    R11, T12, T21, R22 = S_UC
    
    if abs(R11) < 1e-15:
        if abs(T12 * T21 - 1) < 1e-10:
            R_inf = R22
        else:
            R_inf = R22 / (1 - T12 * T21)
        if debug:
            logger.info(f"    [Bloch_backward] Degenerate case: R11≈0, returning R_inf={R_inf:.6e}")
        return R_inf
    
    a = R11
    b = T12 * T21 - R11 * R22 - 1
    c_coef = R22
    
    discriminant = b * b - 4 * a * c_coef
    sqrt_disc = np.sqrt(discriminant + 0j)
    
    R1 = (-b + sqrt_disc) / (2 * a)
    R2 = (-b - sqrt_disc) / (2 * a)
    
    im1 = np.imag(R1)
    im2 = np.imag(R2)
    abs1 = abs(R1)
    abs2 = abs(R2)
    
    if kz_gap is not None:
        imag_kz = np.imag(kz_gap)
        real_kz = np.real(kz_gap)
        is_evanescent = imag_kz > 1e-6 * abs(real_kz)
        
        if is_evanescent:
            if im1 > im2:
                R_inf = R1
                if debug:
                    logger.info(f"    [Bloch_backward] EVANESCENT: Im(kz)={imag_kz:.6e}>0, selecting R with larger Im(R)")
                    logger.info(f"    [Bloch_backward] R1={R1:.6e} (Im={im1:.6e}, |R|={abs1:.6e})")
                    logger.info(f"    [Bloch_backward] R2={R2:.6e} (Im={im2:.6e}, |R|={abs2:.6e})")
                    logger.info(f"    [Bloch_backward] Selected R1 (Im(R1)={im1:.6e} > Im(R2)={im2:.6e})")
            else:
                R_inf = R2
                if debug:
                    logger.info(f"    [Bloch_backward] EVANESCENT: Im(kz)={imag_kz:.6e}>0, selecting R with larger Im(R)")
                    logger.info(f"    [Bloch_backward] R1={R1:.6e} (Im={im1:.6e}, |R|={abs1:.6e})")
                    logger.info(f"    [Bloch_backward] R2={R2:.6e} (Im={im2:.6e}, |R|={abs2:.6e})")
                    logger.info(f"    [Bloch_backward] Selected R2 (Im(R2)={im2:.6e} > Im(R1)={im1:.6e})")
        else:
            if abs1 < abs2:
                R_inf = R1
                if debug:
                    logger.info(f"    [Bloch_backward] PROPAGATING: Im(kz)={imag_kz:.6e}≈0, selecting R with smaller |R|")
                    logger.info(f"    [Bloch_backward] R1={R1:.6e} (Im={im1:.6e}, |R|={abs1:.6e})")
                    logger.info(f"    [Bloch_backward] R2={R2:.6e} (Im={im2:.6e}, |R|={abs2:.6e})")
                    logger.info(f"    [Bloch_backward] Selected R1 (|R1|={abs1:.6e} < |R2|={abs2:.6e})")
            else:
                R_inf = R2
                if debug:
                    logger.info(f"    [Bloch_backward] PROPAGATING: Im(kz)={imag_kz:.6e}≈0, selecting R with smaller |R|")
                    logger.info(f"    [Bloch_backward] R1={R1:.6e} (Im={im1:.6e}, |R|={abs1:.6e})")
                    logger.info(f"    [Bloch_backward] R2={R2:.6e} (Im={im2:.6e}, |R|={abs2:.6e})")
                    logger.info(f"    [Bloch_backward] Selected R2 (|R2|={abs2:.6e} < |R1|={abs1:.6e})")
        
        return R_inf
    
    if abs(R1) <= 1 and abs(R2) > 1:
        return R1
    elif abs(R2) <= 1 and abs(R1) > 1:
        return R2
    elif abs(R1) <= abs(R2):
        return R1
    else:
        return R2
