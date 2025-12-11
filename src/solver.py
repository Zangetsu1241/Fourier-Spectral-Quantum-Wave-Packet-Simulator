import numpy as np

class SplitStepSolver:
    """
    Solves the time-dependent Schrödinger equation using the Split-Step Fourier Method.
    Supports both CPU (NumPy) and GPU (PyTorch) execution.
    """
    def __init__(self, config):
        self.config = config
        self.use_gpu = config.use_gpu
        
        if self.use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    self.device = torch.device("cuda")
                    print(f"GPU Acceleration Enabled: Using {torch.cuda.get_device_name(0)}")
                else:
                    print("Warning: CUDA not available. Falling back to CPU (PyTorch).")
                    self.device = torch.device("cpu")
            except ImportError:
                print("Warning: PyTorch not installed. Falling back to NumPy.")
                self.use_gpu = False

    def solve(self, psi0: np.ndarray, V: np.ndarray) -> np.ndarray:
        """
        Run the simulation.
        Delegates to _solve_gpu or _solve_cpu based on config.
        """
        if self.use_gpu:
            return self._solve_gpu(psi0, V)
        else:
            return self._solve_cpu(psi0, V)

    def _solve_cpu(self, psi0: np.ndarray, V: np.ndarray) -> np.ndarray:
        N = self.config.N
        dt = self.config.dt
        steps = self.config.steps
        k2 = self.config.k**2
        
        # Kinetic operator (Full step)
        expK = np.exp(-1j * k2 * dt)
        # Potential operator (Half step)
        expV_half = np.exp(-1j * V * dt / 2)
        
        psi = psi0.astype(np.complex128)
        results = []
        
        for _ in range(steps):
            psi *= expV_half
            psi_k = np.fft.fft(psi)
            psi_k *= expK
            psi = np.fft.ifft(psi_k)
            psi *= expV_half
            results.append(np.abs(psi)**2)
            
        return np.array(results)

    def _solve_gpu(self, psi0: np.ndarray, V: np.ndarray) -> np.ndarray:
        import torch
        
        # Convert inputs to tensors
        # Use complex128 (double precision) for stability
        psi = torch.from_numpy(psi0).to(self.device, dtype=torch.complex128)
        V_tensor = torch.from_numpy(V).to(self.device, dtype=torch.complex128)
        k2 = torch.from_numpy(self.config.k**2).to(self.device, dtype=torch.complex128)
        
        dt = self.config.dt
        steps = self.config.steps
        N = self.config.N
        
        # Precompute operators
        # Kinetic (Full step)
        expK = torch.exp(-1j * k2 * dt)
        # Potential (Half step)
        expV_half = torch.exp(-1j * V_tensor * dt / 2)
        
        # Pre-allocate output tensor on CPU to save GPU memory if steps is huge
        # But for typical sizes (5000x2048), accumulation on GPU is faster and fine (~80MB)
        # However, to be safe for very large runs, let's keep on GPU and stack at end? 
        # Actually pre-allocating a CPU tensor and copying typically requires sync overhead.
        # Best for speed: Append to GPU list or pre-allocate GPU tensor, then headers transfer once.
        
        # We will collect everything on GPU then move at end.
        prob_density_gpu = torch.empty((steps, N), device=self.device, dtype=torch.float64)
        
        # Main loop
        for i in range(steps):
            psi *= expV_half
            psi = torch.fft.fft(psi)
            psi *= expK
            psi = torch.fft.ifft(psi)
            psi *= expV_half
            
            # Store probability density directly into pre-allocated GPU tensor
            prob_density_gpu[i] = torch.abs(psi)**2
            
        # Single transfer at the end
        return prob_density_gpu.cpu().numpy()
