import numpy as np

def V_free(x: np.ndarray) -> np.ndarray:
    """Free particle potential (V=0)."""
    return np.zeros_like(x)

def V_barrier(x: np.ndarray, height: float = 8.0, width: float = 5.0) -> np.ndarray:
    """Rectangular potential barrier."""
    return height * (np.abs(x) < width)

def V_harmonic(x: np.ndarray, omega: float = 0.04) -> np.ndarray:
    """Harmonic oscillator potential."""
    return 0.5 * omega**2 * x**2
