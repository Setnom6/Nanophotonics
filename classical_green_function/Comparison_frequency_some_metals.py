import numpy as np
import matplotlib.pyplot as plt
import os
from MetallicSlab import MetallicSlab, Constants, Metals
import time

start = time.time()

# Fixed parameters
epsilon1 = 1.0
epsilon3 = 1.0
t = 20 * Constants.NM.value  # Thickness of the slab
z = 5 * Constants.NM.value  # Distance from the source
cutOff = 15
eps_rel = 1e-3
limit = 50

# Create directories for saving data and plots
os.makedirs("./data/", exist_ok=True)
os.makedirs("./plots/", exist_ok=True)

# Colors for each metal
colors = ["b", "r", "g", "m", "c"]
metals = list(Metals)

# Create figure with 4 subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Initialize y-axis limits
y_min, y_max = float('inf'), float('-inf')

for metal, color in zip(metals, colors):
    print(f"Computing: {metal.name}")
    plasmon_resonance = metal.plasmaFrequency / np.sqrt(metal.epsilonB + 1)
    num_first_values = 50
    omega_values = np.concatenate([
        np.linspace(0.01, 0.9 * plasmon_resonance, num_first_values),  # Avoid divergence at omega -> 0
        np.linspace(0.9 * plasmon_resonance, 1.1 * plasmon_resonance, 50),  # High resolution near resonance
        np.linspace(1.1 * plasmon_resonance, 2 * plasmon_resonance, 50)  # Extend up to 2 times the resonance
    ]) * Constants.EV.value

    Gxx_values = np.zeros(len(omega_values), dtype=complex)
    Gzz_values = np.zeros(len(omega_values), dtype=complex)

    # Calculate the Green's function for each frequency
    for i, omega in enumerate(omega_values):
        slab = MetallicSlab(metal, epsilon1, epsilon3, omega, t, z)
        G = slab.calculateNormalizedGreenFunctionReflected(cutOff=cutOff, eps_rel=eps_rel, limit=limit)

        Gxx_values[i] = G[0, 0]  # Gxx = Gyy
        Gzz_values[i] = G[2, 2]  # Gzz

    # Save data
    np.savez(f"./data/{metal.name}_frequency_green_function.npz",
             omega=omega_values, Gxx=Gxx_values, Gzz=Gzz_values)

    # Update y-axis limits
    y_min = min(y_min, np.min(np.real(Gxx_values[num_first_values:])), np.min(np.imag(Gxx_values[num_first_values:])),
                np.min(np.real(Gzz_values[num_first_values:])), np.min(np.imag(Gzz_values[num_first_values:])))
    y_max = max(y_max, np.max(np.real(Gxx_values[num_first_values:])), np.max(np.imag(Gxx_values[num_first_values:])),
                np.max(np.real(Gzz_values[num_first_values:])), np.max(np.imag(Gzz_values[num_first_values:])))

    # Plot Gxx
    axes[0, 0].plot(omega_values / Constants.EV.value, np.real(Gxx_values), label=f"Re(Gxx) - {metal.name}",
                    color=color)
    axes[0, 1].plot(omega_values / Constants.EV.value, np.imag(Gxx_values), label=f"Im(Gxx) - {metal.name}",
                    color=color)

    # Plot Gzz
    axes[1, 0].plot(omega_values / Constants.EV.value, np.real(Gzz_values), label=f"Re(Gzz) - {metal.name}",
                    color=color)
    axes[1, 1].plot(omega_values / Constants.EV.value, np.imag(Gzz_values), label=f"Im(Gzz) - {metal.name}",
                    color=color)

    # Vertical line at plasmon resonance
    for ax in axes.flatten():
        ax.axvline(x=plasmon_resonance, color=color, linestyle="dotted")

# Labels and legends
axes[0, 0].set(title="Re(Gxx) vs Frequency", xlabel="Frequency (eV)", ylabel="Re(Gxx)")
axes[0, 1].set(title="Im(Gxx) vs Frequency", xlabel="Frequency (eV)", ylabel="Im(Gxx)")
axes[1, 0].set(title="Re(Gzz) vs Frequency", xlabel="Frequency (eV)", ylabel="Re(Gzz)")
axes[1, 1].set(title="Im(Gzz) vs Frequency", xlabel="Frequency (eV)", ylabel="Im(Gzz)")

# Adjust y-axis limits automatically
for ax in axes.flatten():
    ax.legend()
    ax.grid(True)
    ax.set_xlim(0, 1.1 * max(metal.plasmaFrequency / np.sqrt(metal.epsilonB + 1) for metal in metals))  # Adjust x-axis
    ax.set_ylim(y_min, y_max)  # Adjust y-axis automatically

plt.tight_layout()

# Save plot
plt.savefig("./plots/different_metals_vs_frequency.png")

end = time.time()
elapsed_time = end - start
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"Execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

plt.show()


