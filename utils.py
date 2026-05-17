import logging
import os

# Physical constants in SI units
c = 299792458.0  # Speed of light in vacuum (m/s)
hbar = 1.0545718e-34  # Reduced Planck constant (J·s)
k_B = 1.380649e-23  # Boltzmann constant (J/K)
eps0 = 8.854187e-12  # Vacuum permittivity (F/m)

def setup_logger(out_dir: str):
    """Set up logger that outputs INFO to console and DEBUG to execution.log"""
    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("fed_solver")
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)
    
    # Console handler (INFO level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (DEBUG level)
    file_handler = logging.FileHandler(os.path.join(out_dir, "execution.log"))
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger
