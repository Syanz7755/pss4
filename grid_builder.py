import numpy as np
from typing import Tuple
from utils import c

def build_k_grid(omega_min: float, gap_thickness: float = None,
                 num_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Build continuous k_parallel grid with integration weights
    
    For near-field radiative heat transfer, the evanescent wave contribution
    is significant at k_parallel > 1/gap_thickness. The grid must extend
    sufficiently high to capture this near-field enhancement.
    
    Args:
        omega_min: minimum angular frequency
        gap_thickness: vacuum gap thickness in meters (for scaling k_max)
        num_points: number of logarithmically spaced integration points
    
    Returns:
        k_par_grid: array of k_parallel values
        k_weights: integration weights for each k_parallel
    """
    if num_points < 2:
        raise ValueError("num_points must be >= 2")
    
    # k_max should be at least 10/gap to capture near-field contribution
    # For smallest gaps (~10 nm), need k_max ~ 1e9 or higher
    if gap_thickness and gap_thickness > 0:
        k_max = max(1e10, 100.0 / gap_thickness)  # 100/gap ensures capturing near-field
    else:
        k_max = 1e10  # Default for smallest gaps
    
    k_min = 1e-4 * (omega_min / c)
    
    k_par_grid = np.logspace(np.log10(k_min), np.log10(k_max), num_points)
    
    k_weights = np.zeros_like(k_par_grid)
    k_weights[0] = (k_par_grid[1] - k_par_grid[0]) / 2
    k_weights[-1] = (k_par_grid[-1] - k_par_grid[-2]) / 2
    k_weights[1:-1] = (k_par_grid[2:] - k_par_grid[:-2]) / 2
    
    return k_par_grid, k_weights
