from dataclasses import dataclass
import numpy as np

@dataclass
class SimulationConfig:
    """Configuration parameters for the quantum simulation."""
    N: int = 2048           # Grid points
    L: float = 200.0        # Domain size [-L/2, L/2]
    dt: float = 0.005       # Time step
    steps: int = 5000       # Number of time steps
    use_gpu: bool = False   # GPU acceleration flag
    
    @property
    def dx(self) -> float:
        return self.L / self.N
    
    @property
    def x(self) -> np.ndarray:
        return np.linspace(-self.L/2, self.L/2, self.N, endpoint=False)
    
    @property
    def k(self) -> np.ndarray:
        return 2 * np.pi * np.fft.fftfreq(self.N, d=self.dx)
