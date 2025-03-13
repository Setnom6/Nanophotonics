import numpy as np
import matplotlib.pyplot as plt
from MetallicSlab import MetallicSlab, Constants, Metals

# Fixed parameters
epsilon1 = 1.0
epsilon3 = 1.0
t = 1.0 * Constants.NM.value  # Thickness in meters
z = 5 * Constants.NM.value  # Distance from the source

# Select a single metal
metal = Metals.SILVER
plasmon_resonance = metal.plasmaFrequency / np.sqrt(metal.epsilonB + 1)
omega = 0.01 * plasmon_resonance * Constants.EV.value

# Define range for cutOff and limit
cutOff_values = np.arange(5, 60, 5)  # Varying cutOff from 5 to 60
limit = 50  # Fixed limit
eps_rel = 1e-6  # Fixed eps_rel

# Arrays to store values
Gxx_real = np.zeros(len(cutOff_values))
Gzz_real = np.zeros(len(cutOff_values))
Gxx_imag = np.zeros(len(cutOff_values))
Gzz_imag = np.zeros(len(cutOff_values))

# Compute Green's function values for different cutOff
for i, cutOff in enumerate(cutOff_values):
    slab = MetallicSlab(metal, epsilon1, epsilon3, omega, t, z)
    G = slab.calculateNormalizedGreenFunctionReflected(cutOff=cutOff, eps_rel=eps_rel, limit=limit)

    Gxx_real[i] = np.real(G[0, 0])
    Gzz_real[i] = np.real(G[2, 2])
    Gxx_imag[i] = np.imag(G[0, 0])
    Gzz_imag[i] = np.imag(G[2, 2])

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

titles = [r"Re$(G_{xx})$", r"Re$(G_{zz})$", r"Im$(G_{xx})$", r"Im$(G_{zz})$"]
data = [Gxx_real, Gzz_real, Gxx_imag, Gzz_imag]

for ax, title, values in zip(axes.ravel(), titles, data):
    ax.plot(cutOff_values, values, marker='o', linestyle='-')
    ax.set_xlabel("cutOff")
    ax.set_ylabel(title)
    ax.set_title(title)
    ax.grid(True)

fig.suptitle(
    f"Effect of cutOff on Green's Function (t={t / Constants.NM.value:.2f} nm, z={z / Constants.NM.value:.2f} nm)")
plt.tight_layout()
plt.savefig(
    f"./plots/Green_Function_cutOff_t_{t / Constants.NM.value:.0f}_omega_{omega / (plasmon_resonance * Constants.EV.value) * 100:.0f}.png")
plt.show()