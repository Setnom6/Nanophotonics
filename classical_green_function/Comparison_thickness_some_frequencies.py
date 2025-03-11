import numpy as np
import matplotlib.pyplot as plt
import os
from MetallicSlab import MetallicSlab, Constants, Metals
import time

start = time.time()

# Fixed parameters
epsilon1 = 1.0
epsilon3 = 1.0
z = 5 * Constants.NM.value  # Distance from the source
cutOff = 15
eps_rel = 1e-3
limit = 50

# Select a single metal
metal = Metals.SILVER
plasmon_resonance = metal.plasmaFrequency / np.sqrt(metal.epsilonB + 1)

# Thickness values to study
omega_values = [0.5, 0.8, 1.0, 1.2, 1.5]  # in nm
omega_values = [w * Constants.EV.value * plasmon_resonance for w in omega_values]

# Create directories for saving data and plots
os.makedirs("./data/", exist_ok=True)
os.makedirs("./plots/", exist_ok=True)

# Colors for each thickness
colors = ["b", "r", "g", "m", "c"]

# Create figure with 4 subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Initialize y-axis limits
y_min, y_max = float('inf'), float('-inf')

for omega, color in zip(omega_values, colors):
    print(f"Computing: frequency {omega / (Constants.EV.value * plasmon_resonance):.2f} times the plasmon resonance")
    thickness_values = np.linspace(5, 30, 300) * Constants.NM.value

    Gxx_values = np.zeros(len(thickness_values), dtype=complex)
    Gzz_values = np.zeros(len(thickness_values), dtype=complex)

    # Calculate the Green's function for each frequency
    for i, t in enumerate(thickness_values):
        slab = MetallicSlab(metal, epsilon1, epsilon3, omega, t, z)
        G = slab.calculateNormalizedGreenFunctionReflected(cutOff=cutOff, eps_rel=eps_rel, limit=limit)

        Gxx_values[i] = G[0, 0]  # Gxx = Gyy
        Gzz_values[i] = G[2, 2]  # Gzz

    # Save data
    np.savez(f"./data/{metal.name}_frequency_{int(omega / Constants.EV.value)}eV_green_function.npz",
             t=thickness_values, Gxx=Gxx_values, Gzz=Gzz_values)

    # Update y-axis limits
    y_min = min(y_min, np.min(np.real(Gxx_values)), np.min(np.imag(Gxx_values)),
                np.min(np.real(Gzz_values)), np.min(np.imag(Gzz_values)))
    y_max = max(y_max, np.max(np.real(Gxx_values)), np.max(np.imag(Gxx_values)),
                np.max(np.real(Gzz_values)), np.max(np.imag(Gzz_values)))

    # Plot Gxx
    axes[0, 0].plot(thickness_values / Constants.NM.value, np.real(Gxx_values), label=f"frequency={omega/(plasmon_resonance*Constants.EV.value):.2f} "+ "$\omega_{ps}$",
                    color=color)
    axes[0, 1].plot(thickness_values / Constants.NM.value, np.imag(Gxx_values), label=f"Im(Gxx) - frequency={omega/Constants.EV.value:.2f} eV",
                    color=color)

    # Plot Gzz
    axes[1, 0].plot(thickness_values / Constants.NM.value, np.real(Gzz_values), label=f"Re(Gzz) - frequency={omega/Constants.EV.value:.2f} eV",
                    color=color)
    axes[1, 1].plot(thickness_values / Constants.NM.value, np.imag(Gzz_values), label=f"Im(Gzz) - frequency={omega/Constants.EV.value:.2f} eV",
                    color=color)

    # Vertical line at plasmon resonance
    for ax in axes.flatten():
        ax.axvline(x=plasmon_resonance, color="k", linestyle="dotted")

# Labels and legends
axes[0, 0].set(title="Re(Gxx) vs Thickness", xlabel="Thickness (nm)", ylabel="Re(Gxx)")
axes[0, 1].set(title="Im(Gxx) vs Thickness", xlabel="Thickness (nm)", ylabel="Im(Gxx")
axes[1, 0].set(title="Re(Gzz) vs Thickness", xlabel="Thickness (nm)", ylabel="Re(Gzz")
axes[1, 1].set(title="Im(Gzz) vs Thickness", xlabel="Thickness (nm)", ylabel="Im(Gzz")

# Adjust y-axis limits automatically
fig.legend(handles=axes[0, 0].get_legend_handles_labels()[0], loc='upper center', ncol=len(omega_values))

# Adjust axes limits
for ax in axes.flatten():
    ax.grid(True)
    ax.set_xlim(5,30)
    ax.grid(True)
    ax.set_xlim(5,30)  # Adjust x-axis
      # Adjust y-axis automatically

plt.tight_layout(rect=tuple((0.0, 0.0, 1.0, 0.92)))

# Save plot
plt.savefig(f"./plots/{metal.name}_frequency_variation.png")

end = time.time()
elapsed_time = end - start
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"Execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

plt.show()
