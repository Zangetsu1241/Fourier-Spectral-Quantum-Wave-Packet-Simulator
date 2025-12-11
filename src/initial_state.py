import numpy as np

def gaussian_packet(x: np.ndarray, x0: float = 0.0, k0: float = 2.0, sigma: float = 2.0) -> np.ndarray:
    """
    Creates a Gaussian wave packet.
    
    Args:
        x: Spatial grid
        x0: Initial position
        k0: Initial momentum (wavenumber)
        sigma: Width of the packet
    
    Returns:
        Normalized complex wavefunction
    """
    # Normalization factor (1 / (pi * sigma^2))^0.25
    norm = (1 / (np.pi * sigma**2))**0.25
    
    # Gaussian envelope * Plane wave
    return norm * np.exp(-(x - x0)**2 / (2 * sigma**2)) * np.exp(1j * k0 * x)
