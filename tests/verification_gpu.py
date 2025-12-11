import unittest
import numpy as np
import torch
from src.config import SimulationConfig
from src.solver import SplitStepSolver
from src.potentials import V_free
from src.initial_state import gaussian_packet

class TestGPUQuantumSimulation(unittest.TestCase):
    
    def setUp(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
            
    def test_gpu_consistency_with_cpu(self):
        print("\nTesting GPU vs CPU Consistency...")
        # Small grid for quick comparison
        config_cpu = SimulationConfig(N=512, L=100.0, dt=0.01, steps=100, use_gpu=False)
        config_gpu = SimulationConfig(N=512, L=100.0, dt=0.01, steps=100, use_gpu=True)
        
        psi0 = gaussian_packet(config_cpu.x)
        V = V_free(config_cpu.x)
        
        solver_cpu = SplitStepSolver(config_cpu)
        solver_gpu = SplitStepSolver(config_gpu)
        
        res_cpu = solver_cpu.solve(psi0, V)
        res_gpu = solver_gpu.solve(psi0, V)
        
        # Compare final state probability density
        diff = np.abs(res_cpu[-1] - res_gpu[-1])
        max_diff = np.max(diff)
        
        print(f"Max Difference (CPU vs GPU): {max_diff}")
        self.assertTrue(max_diff < 1e-10, f"GPU results diverge from CPU! Max diff: {max_diff}")

    def test_gpu_norm_conservation(self):
        print("\nTesting GPU Norm Conservation...")
        config = SimulationConfig(steps=100, dt=0.01, use_gpu=True)
        solver = SplitStepSolver(config)
        V = V_free(config.x)
        psi0 = gaussian_packet(config.x)
        
        prob_density = solver.solve(psi0, V)
        
        final_prob = prob_density[-1]
        norm = np.sum(final_prob) * config.dx
        
        print(f"Final Norm (GPU): {norm}")
        self.assertTrue(np.abs(norm - 1.0) < 1e-4, f"Norm not conserved on GPU: {norm}")

if __name__ == '__main__':
    unittest.main()
