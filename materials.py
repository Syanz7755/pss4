import numpy as np
import os
from typing import List, Dict, Tuple

def load_materials_and_grid(parsed_layers: List[Dict], all_freq: bool, freq_skip: int = 1) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Load materials and sync to the primary frequency grid.
    
    Args:
        parsed_layers: List of layer dictionaries
        all_freq: If True, use full frequency grid; otherwise use sparse_53
        freq_skip: Skip factor for frequency grid (1=all points, 2=half points, etc.)
    """
    # Resolve materials directory relative to this file
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    _mat_dir = os.path.join(_this_dir, 'materials')

    if all_freq:
        material_files = {
            'Silica': os.path.join(_mat_dir, 'aligned_SiO2-Franta-300C.txt'),
            'SiO2': os.path.join(_mat_dir, 'aligned_SiO2-Franta-300C.txt'),
            'Vacuum': os.path.join(_mat_dir, 'aligned_Vacuum.txt'),
            'Si3N4': os.path.join(_mat_dir, 'aligned_Si3N4-Luke(true).txt'),
            'SiN': os.path.join(_mat_dir, 'aligned_Si3N4-Luke(true).txt'),
            'SiC': os.path.join(_mat_dir, 'aligned_SiC_Lorentz-sparse_53.txt'),
        }
    else:
        material_files = {
            'Silica': os.path.join(_mat_dir, 'aligned-SiO2_Franta-300C-sparse_53.txt'),
            'SiO2': os.path.join(_mat_dir, 'aligned-SiO2_Franta-300C-sparse_53.txt'),
            'Vacuum': os.path.join(_mat_dir, 'aligned-Vacuum-sparse_53.txt'),
            'Si3N4': os.path.join(_mat_dir, 'aligned_Si3N4-Luke(true)-sparse_53.txt'),
            'SiN': os.path.join(_mat_dir, 'aligned_Si3N4-Luke(true)-sparse_53.txt'),
            'SiC': os.path.join(_mat_dir, 'aligned_SiC_Lorentz-sparse_53.txt'),
        }
    
    materials = set()
    for layer in parsed_layers:
        materials.add(layer['material'])
    
    primary_material = None
    for mat in ['Silica', 'SiO2', 'SiC', 'SiN', 'Si3N4']:
        if mat in materials:
            primary_material = mat
            break

    if primary_material is None:
        raise ValueError("Primary material (Silica/SiO2/SiC/SiN) not found in parsed layers")
    
    primary_file = material_files.get(primary_material)
    if not primary_file or not os.path.exists(primary_file):
        raise ValueError(f"Primary material file not found: {primary_file}")
    
    primary_data = np.loadtxt(primary_file)
    omega_grid = primary_data[:, 0]
    
    # Apply frequency skip to reduce grid density
    if freq_skip > 1:
        omega_grid = omega_grid[::freq_skip]
    
    material_eps = {}
    
    for material in materials:
        file_path = material_files.get(material)
        if not file_path or not os.path.exists(file_path):
            raise ValueError(f"Material file not found: {file_path}")
        
        data = np.loadtxt(file_path)
        file_omega = data[:, 0]
        eps_real = data[:, 1]
        eps_imag = data[:, 2]
        
        freq_to_idx = {}
        for i, freq in enumerate(file_omega):
            freq_to_idx[freq] = i
        
        eps = np.zeros_like(omega_grid, dtype=np.complex128)
        missing_freqs = []
        
        for i, omega in enumerate(omega_grid):
            if omega in freq_to_idx:
                eps_idx = freq_to_idx[omega]
                eps[i] = eps_real[eps_idx] + 1j * eps_imag[eps_idx]
            else:
                missing_freqs.append(omega)
        
        if missing_freqs:
            raise ValueError("Material dielectric input error: missing frequencies.")
        
        material_eps[material] = eps
    
    return omega_grid, material_eps
