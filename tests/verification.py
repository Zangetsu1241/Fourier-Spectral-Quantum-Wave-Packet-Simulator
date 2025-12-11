import unittest
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from src.config import SimulationConfig
from src.solver import SplitStepSolver
from src.initial_state import gaussian_packet
from src.potentials import V_free

class TestQuantumSimulation(unittest.TestCase):
    
    def test_norm_conservation(self):
        """Verify that wavefunction norm is conserved (Unitary evolution)."""
        print("\nTesting Norm Conservation...")
        config = SimulationConfig(steps=100, dt=0.01)
        solver = SplitStepSolver(config)
        V = V_free(config.x)
        psi0 = gaussian_packet(config.x)
        
        prob_density = solver.solve(psi0, V)
        
        # Check normalization at last step
        # sum |psi|^2 * dx should be 1
        final_prob = prob_density[-1]
        norm = np.sum(final_prob) * config.dx
        
        print(f"Final Norm: {norm}")
        self.assertTrue(np.abs(norm - 1.0) < 1e-4, f"Norm not conserved: {norm}")

    def test_free_particle_spreading(self):
        """
        Verify analytical spreading of a Gaussian wave packet.
        For H = -d^2/dx^2, sigma(t) = sigma_0 * sqrt(1 + (t/sigma_0^2)^2)
        """
        print("\nTesting Free Particle Spreading...")
        sigma0 = 2.0
        t_final = 10.0
        dt = 0.05
        steps = int(t_final / dt)
        
        config = SimulationConfig(steps=steps, dt=dt, L=100.0, N=1024)
        solver = SplitStepSolver(config)
        V = V_free(config.x)
        psi0 = gaussian_packet(config.x, sigma=sigma0, k0=1.0) # k0=1 gives some motion
        
        prob_density = solver.solve(psi0, V)
        
        # Calculate width at final time
        # sigma^2 = <x^2> - <x>^2
        final_prob = prob_density[-1]
        x = config.x
        
        x_mean = np.sum(final_prob * x) * config.dx
        x2_mean = np.sum(final_prob * x**2) * config.dx
        sigma_sim = np.sqrt(x2_mean - x_mean**2)
        
        # Analytical
        # For H = -d^2/dx^2, E = k^2, omega = k^2
        # Group velocity dispersion w'' = 2.
        # For a Gaussian |psi|^2 ~ exp(-x^2 / sigma_prob^2), 
        # sigma_prob(t)^2 = sigma_prob(0)^2 + (w'' * t / (2 * sigma_prob(0)))^2
        # Note: sigma0 in gaussian_packet is width of Amplitude, so sigma_prob(0) = sigma0 / sqrt(2)
        
        sigma_prob_0 = sigma0 / np.sqrt(2)
        w_double_prime = 2.0
        
        sigma_analytical = np.sqrt(sigma_prob_0**2 + (w_double_prime * t_final / (2 * sigma_prob_0))**2)
        
        print(f"Time: {t_final}")
        print(f"Simulated Width: {sigma_sim:.4f}")
        print(f"Analytical Width: {sigma_analytical:.4f}")
        
        # Allow small error (numerical integration + finite grid)
        err = np.abs(sigma_sim - sigma_analytical)
        self.assertTrue(err < 0.05, f"Width mismatch. Sim: {sigma_sim}, Ref: {sigma_analytical}, Err: {err}")

if __name__ == "__main__":
    unittest.main()
