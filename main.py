import argparse
import numpy as np
from src.config import SimulationConfig
from src.potentials import V_free, V_barrier, V_harmonic
from src.initial_state import gaussian_packet
from src.solver import SplitStepSolver
from src.visualizer import QuantumVisualizer

def run_simulation(scenario: str, use_gpu: bool = False):
    print(f"--- Running Scenario: {scenario} (GPU={use_gpu}) ---")
    config = SimulationConfig(use_gpu=use_gpu)
    x = config.x
    
    # 1. Setup Potential and Initial State
    if scenario == "free":
        V = V_free(x)
        # Standard packet
        psi0 = gaussian_packet(x, x0=-50, k0=4.0, sigma=2.0)
        title = "Free Particle"
        
    elif scenario == "barrier":
        V = V_barrier(x, height=8.0, width=5.0)
        # Packet incoming from left
        psi0 = gaussian_packet(x, x0=-50, k0=4.0, sigma=2.0)
        title = "Barrier Scattering"
        
    elif scenario == "harmonic":
        V = V_harmonic(x, omega=0.04)
        # Displaced packet
        psi0 = gaussian_packet(x, x0=-50, k0=0.0, sigma=2.0)
        title = "Harmonic Oscillator"
        
    else:
        raise ValueError(f"Unknown scenario: {scenario}")
        
    # 2. Solve
    solver = SplitStepSolver(config)
    print("Solving Schrödinger equation...")
    # shape: (steps, N)
    prob_density = solver.solve(psi0, V)
    print("Simulation complete.")
    
    # 3. Visualize
    viz = QuantumVisualizer(config)
    print("Generating animation...")
    viz.create_animation(prob_density, V, title, filename=scenario)
    
    print("Plotting observables...")
    barrier_width = 5.0 if scenario == "barrier" else 0
    viz.plot_observables(prob_density, filename_prefix=scenario, barrier_width=barrier_width)
    print(f"--- Scenario {scenario} Finished ---\n")

def main():
    parser = argparse.ArgumentParser(description="Complex Spectral-PDE Quantum Propagation Simulator")
    parser.add_argument("scenarios", nargs="*", default=["free", "barrier", "harmonic"], 
                        help="Scenarios to run: free, barrier, harmonic")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU acceleration (requires PyTorch & CUDA)")
    args = parser.parse_args()
    
    # Update global config for all scenarios (or pass it down)
    # Since run_simulation instantiates config, we should pass the flag or modify run_simulation
    
    for scenario in args.scenarios:
        run_simulation(scenario, use_gpu=args.gpu)

if __name__ == "__main__":
    main()
