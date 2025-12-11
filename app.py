import streamlit as st
import numpy as np
import os
import torch
from src.config import SimulationConfig
from src.potentials import V_free, V_barrier, V_harmonic
from src.initial_state import gaussian_packet
from src.solver import SplitStepSolver
from src.visualizer import QuantumVisualizer

# Set page layout
st.set_page_config(page_title="Quantum Wave Packet Simulator", layout="wide")

st.title("Complex Spectral–PDE Quantum Propagation Simulator")
st.markdown("""
Simulates the time-dependent Schrödinger equation using the **Split-Step Fourier Method**.
""")

# --- Sidebar Configuration ---
st.sidebar.header("Simulation Parameters")

scenario = st.sidebar.selectbox(
    "Scenario",
    ["Free Particle", "Barrier Scattering", "Harmonic Oscillator"]
)

N = st.sidebar.slider("Grid Points (N)", 512, 4096, 2048, step=512)
L = st.sidebar.number_input("Domain Size (L)", value=200.0)
dt = st.sidebar.number_input("Time Step (dt)", value=0.005, format="%.4f")
steps = st.sidebar.number_input("Time Steps", value=5000, step=100)

use_gpu = st.sidebar.checkbox("Use GPU Acceleration", value=False)
if use_gpu and not torch.cuda.is_available():
    st.sidebar.warning("CUDA not available. Will fall back to CPU.")

# --- Helper Function ---
def run_simulation_ui(scenario_name, N, L, dt, steps, use_gpu):
    config = SimulationConfig(N=N, L=L, dt=dt, steps=steps, use_gpu=use_gpu)
    x = config.x
    
    # 1. Setup Potential and Initial State
    if scenario_name == "Free Particle":
        V = V_free(x)
        psi0 = gaussian_packet(x, x0=-50, k0=4.0, sigma=2.0)
        scenario_key = "free"
        
    elif scenario_name == "Barrier Scattering":
        V = V_barrier(x, height=8.0, width=5.0)
        psi0 = gaussian_packet(x, x0=-50, k0=4.0, sigma=2.0)
        scenario_key = "barrier"
        
    elif scenario_name == "Harmonic Oscillator":
        V = V_harmonic(x, omega=0.04)
        psi0 = gaussian_packet(x, x0=-50, k0=0.0, sigma=2.0)
        scenario_key = "harmonic"
        
    # 2. Solve
    solver = SplitStepSolver(config)
    
    with st.spinner("Solving Schrödinger equation..."):
        prob_density = solver.solve(psi0, V)
    
    st.success("Simulation Complete!")
    
    # 3. Visualize
    viz = QuantumVisualizer(config, output_dir="stream_output")
    
    # Generate GIF
    with st.spinner("Rendering Animation..."):
        viz.create_animation(prob_density, V, scenario_name, filename=scenario_key)
        
    # Generate Plots
    barrier_width = 5.0 if scenario_key == "barrier" else 0
    viz.plot_observables(prob_density, filename_prefix=scenario_key, barrier_width=barrier_width)
    
    return scenario_key

# --- Main Area ---

if st.button("Run Simulation"):
    scenario_key = run_simulation_ui(scenario, N, L, dt, steps, use_gpu)
    
    # Display Results
    st.subheader("Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Wavefunction Evolution")
        gif_path = f"stream_output/{scenario_key}.gif"
        if os.path.exists(gif_path):
            st.image(gif_path, caption="Animation")
        else:
            st.error("Animation file not found.")
            
    with col2:
        st.markdown("### Observables")
        pos_plot = f"stream_output/{scenario_key}_position.png"
        scat_plot = f"stream_output/{scenario_key}_scattering.png"
        
        if os.path.exists(pos_plot):
            st.image(pos_plot, caption="Expected Position <x>")
            
        if scenario_key == "barrier" and os.path.exists(scat_plot):
            st.image(scat_plot, caption="Transmission / Reflection")

st.markdown("---")
st.markdown("Generated outputs are saved in `stream_output/`.")
