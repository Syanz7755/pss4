import numpy as np
from typing import Tuple
from utils import c

def build_k_grid(omega_min: float) -> Tuple[np.ndarray, np.ndarray]:
    """Build continuous k_parallel grid with integration weights"""
    num_points = 500
    k_max = 1e8
    
    k_min = 1e-4 * (omega_min / c)
    
    k_par_grid = np.logspace(np.log10(k_min), np.log10(k_max), num_points)
    
    k_weights = np.zeros_like(k_par_grid)
    k_weights[0] = (k_par_grid[1] - k_par_grid[0]) / 2
    k_weights[-1] = (k_par_grid[-1] - k_par_grid[-2]) / 2
    k_weights[1:-1] = (k_par_grid[2:] - k_par_grid[:-2]) / 2
    
    return k_par_grid, k_weights
