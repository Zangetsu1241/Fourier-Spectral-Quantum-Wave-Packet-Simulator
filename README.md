# Complex Spectral–PDE Quantum Propagation Simulator

A high-performance spectral method solver for the time-dependent Schrödinger equation, featuring **GPU acceleration** and an **Interactive Web UI**.

## 🚀 Features

- **Split-Step Fourier Method (SSFM)**: High-accuracy $O(N \log N)$ spectral solver with Strang Splitting ($O(dt^2)$).
- **Hybrid CPU/GPU Engine**: 
  - Standard **NumPy** backend for compatibility.
  - **PyTorch/CUDA** backend for massive speedups on NVIDIA GPUs.
- **Interactive Frontend**: **Streamlit** dashboard for real-time simulation configuration and visualization.
- **Modular Architecture**: Clean separation of physics, solver, and visualization components.
- **Optimized Rendering**: Adaptive downsampling for smooth animations regardless of simulation length.

---

## 📦 Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-repo/quantum-simulator.git
    cd quantum-simulator
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **(Optional) Enable GPU Support**:
    If you have an NVIDIA GPU, install the CUDA-enabled version of PyTorch:
    ```bash
    pip install torch --index-url https://download.pytorch.org/whl/cu121
    ```

---

## 🖥️ Usage

### 1. Interactive GUI (Recommended)
Launch the Streamlit web interface:
```bash
streamlit run app.py
```
- Open **http://localhost:8501** in your browser.
- Select scenarios, adjust grid/time parameters, and toggle **GPU Acceleration**.

### 2. Command Line Interface (CLI)
Run specific simulation scenarios directly from the terminal.

**Run all scenarios:**
```bash
python main.py
```

**Run specific scenarios with GPU:**
```bash
python main.py free barrier --gpu
```
*Available scenarios: `free`, `barrier`, `harmonic`*

---

## 🧪 Verification

The simulator includes a rigorous test suite to ensure physical correctness.

**Run verification tests:**
```bash
python -m tests.verification       # CPU tests
python -m tests.verification_gpu   # GPU tests & consistency check
```

**What is tested?**
- **Norm Conservation**: $\int |\psi|^2 dx = 1.0$ (Unitary evolution).
- **Dispersion**: Free particle spreading matches analytical solution ($H = -\nabla^2$).
- **Consistency**: GPU results match CPU results to machine precision (~$10^{-15}$).

---

## 📂 Project Structure

- `src/`
  - `config.py`: Simulation configuration parameters.
  - `potentials.py`: Potential definitions ($V_{free}, V_{barrier}, V_{harmonic}$).
  - `initial_state.py`: Wave packet generation.
  - `solver.py`: Core `SplitStepSolver` (Hybrid CPU/GPU).
  - `visualizer.py`: Animation and plotting logic.
- `app.py`: Streamlit frontend application.
- `main.py`: CLI entry point.
- `tests/`: Verification scripts.

---

## 🔬 Physics Scenarios

1.  **Free Particle**: Visualizes wave packet dispersion and group velocity.
2.  **Barrier Scattering**: Demonstrates quantum tunneling and reflection ($R+T=1$).
3.  **Harmonic Oscillator**: Shows coherent states breathing and oscillating in a parabolic potential.
