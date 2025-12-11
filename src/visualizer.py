import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os

class QuantumVisualizer:
    def __init__(self, config, output_dir="output"):
        self.config = config
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def create_animation(self, data: np.ndarray, V: np.ndarray, title: str, filename: str):
        """
        Creates and saves an animation of the wave packet evolution.
        """
        x = self.config.x
        dt = self.config.dt
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot potential (scaled for visibility)
        max_prob = np.max(data)
        max_V = np.max(np.abs(V))
        if max_V > 1e-6:
            V_scaled = V / max_V * max_prob
            ax.plot(x, V_scaled, 'r--', label="Potential (Scaled)", alpha=0.5)
        
        # Plot wavefunction
        line, = ax.plot(x, data[0], 'b-', lw=2, label="|ψ|²")
        
        ax.set_xlim(-self.config.L/2, self.config.L/2)
        ax.set_ylim(0, max_prob * 1.2)
        ax.set_xlabel("Position (x)")
        ax.set_ylabel("Probability Density")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        
        def update(frame):
            line.set_ydata(data[frame])
            time_text.set_text(f"Time: {frame * dt:.2f}")
            ax.set_title(f"{title}")
            return line, time_text
        
        # Downsample frames for speed if very long
        # Target max frames for GIF to keep rendering fast (~5-10 seconds)
        max_frames = 150
        total_steps = len(data)
        stride = max(1, total_steps // max_frames)
        
        frames = list(range(0, total_steps, stride))
        print(f"Rendering {len(frames)} frames (stride={stride})...")
        
        ani = FuncAnimation(fig, update, frames=frames, interval=30, blit=True)
        
        save_path = os.path.join(self.output_dir, filename)
        print(f"Saving animation to {save_path}.gif ...")
        # Save as GIF using Pillow (more portable)
        ani.save(save_path + ".gif", writer='pillow', fps=30)
        plt.close(fig)

    def plot_observables(self, data: np.ndarray, filename_prefix: str, barrier_width: float = 0):
        """
        Plots observables like expected position <x> and transmission/reflection.
        """
        x = self.config.x
        t = np.arange(len(data)) * self.config.dt
        
        # 1. Expected Position <x>
        # Use trapezoidal integration
        # <x> = integral(x * |psi|^2)
        x_mean = np.sum(data * x[None, :], axis=1) * self.config.dx
        
        plt.figure(figsize=(10, 5))
        plt.plot(t, x_mean)
        plt.axhline(0, color='r', linestyle='--', alpha=0.5)
        plt.xlabel("Time")
        plt.ylabel("<x>")
        plt.title(f"Expected Position: {filename_prefix}")
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, f"{filename_prefix}_position.png"))
        plt.close()
        
        # 2. Transmission/Reflection (if barrier present)
        if barrier_width > 0:
            # Probability on left (Reflection) vs Right (Transmission)
            # define boundary at x=0 (center of barrier usually)
            
            # Mask for x < 0 and x > 0
            left_mask = x < 0
            right_mask = x > 0
            
            # Sum * dx
            P_left = np.sum(data[:, left_mask], axis=1) * self.config.dx
            P_right = np.sum(data[:, right_mask], axis=1) * self.config.dx
            
            plt.figure(figsize=(10, 5))
            plt.plot(t, P_left, label="Reflection (x < 0)")
            plt.plot(t, P_right, label="Transmission (x > 0)")
            plt.xlabel("Time")
            plt.ylabel("Probability")
            plt.title(f"Scattering Probabilities: {filename_prefix}")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.output_dir, f"{filename_prefix}_scattering.png"))
            plt.close()
