import numpy as np
from typing import Dict, Optional, Tuple
import logging
from utils import hbar, k_B
from utils import c

logger = logging.getLogger("fed_solver")

def compute_prop_evan_boundary(omega: float, eps_gap: complex = 1.0) -> float:
    """Return the propagating/evanescent k_parallel boundary for a gap medium."""
    return float(np.real((omega / c) * np.sqrt(eps_gap)))

def classify_prop_evan_modes(k_par_grid: np.ndarray, omega: float,
                             eps_gap: complex = 1.0) -> Tuple[np.ndarray, np.ndarray, float]:
    """Classify k_parallel samples as propagating or evanescent in the gap."""
    k_boundary = compute_prop_evan_boundary(omega, eps_gap)
    propagating_mask = k_par_grid < k_boundary
    evanescent_mask = ~propagating_mask
    return propagating_mask, evanescent_mask, k_boundary

def compute_transmission(kz_gap: complex, d_gap: float, R_L: complex, R_R: complex) -> float:
    """Compute dimensionless energy transmission probability.
    
    Args:
        kz_gap: z-component of wavevector in gap
        d_gap: gap thickness in meters
        R_L: reflection coefficient from left side
        R_R: reflection coefficient from right side
    
    Returns:
        tau: dimensionless transmission probability (clamped to >= 0)
    """
    D = 1 - R_L * R_R * np.exp(2j * kz_gap * d_gap)
    
    imag_kz = np.imag(kz_gap)
    real_kz = np.real(kz_gap)
    
    if abs(imag_kz) < 1e-6 * abs(real_kz):
        tau = (1 - abs(R_L)**2) * (1 - abs(R_R)**2) / abs(D)**2
    else:
        tau = 4 * np.imag(R_L) * np.imag(R_R) * np.exp(-2 * imag_kz * d_gap) / abs(D)**2
    
    if tau < 0:
        logger.debug(f"Transmission tau clamped from {tau} to 0.0")
        tau = 0.0
    
    return tau

def planck_derivative(omega: float, T: float) -> float:
    """Calculate temperature derivative of Planck oscillator energy.
    
    Args:
        omega: angular frequency in rad/s
        T: temperature in Kelvin
    
    Returns:
        dTheta_dT: temperature derivative in J/K
    """
    x = hbar * omega / (k_B * T)
    
    if x > 700:
        return 0.0
    
    exp_x = np.exp(x)
    denom = exp_x - 1
    
    if denom == 0:
        return 0.0
    
    dTheta_dT = (hbar * omega)**2 * exp_x / (k_B * T**2 * denom**2)
    
    return dTheta_dT

def compute_observables(omega: float, T: Optional[float], tau_array: np.ndarray,
                        k_par_grid: np.ndarray, k_weights: np.ndarray) -> Dict:
    """Compute spectral observables by integrating over k_parallel.
    
    In the macroscopic model, tau_array represents the total dimensionless flux.
    Absorptivity is implicitly handled by the transmission formula.
    
    Args:
        omega: angular frequency in rad/s
        T: temperature in Kelvin (optional)
        tau_array: transmission array of shape (2, N_k) for s and p polarizations
        k_par_grid: k_parallel grid in 1/m
        k_weights: integration weights for k_parallel
    
    Returns:
        dict with 'transmission' and optionally 'spectral_flux'
    """
    tau_s, tau_p = tau_array
    
    integrand_s = tau_s * k_par_grid * k_weights / (2 * np.pi)
    integrand_p = tau_p * k_par_grid * k_weights / (2 * np.pi)
    
    Phi = np.sum(integrand_s) + np.sum(integrand_p)
    
    result = {'transmission': Phi}
    
    if T is not None:
        dTheta_dT = planck_derivative(omega, T)
        spectral_flux = dTheta_dT * Phi / (2 * np.pi)
        result['spectral_flux'] = spectral_flux
    
    return result
