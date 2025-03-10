import numpy as np
import matplotlib.pyplot as plt
from MetallicSlab import MetallicSlab, Constants, Metals

# Fixed parameters
epsilon1 = 1.0
epsilon3 = 1.0
metal = Metals.GOLD
plasmon_resonance = metal.plasmaFrequency / np.sqrt(metal.epsilonB +1)
omega = 1.0*plasmon_resonance*Constants.EV.value
z = 5 * Constants.NM.value  # Source distance from the slab
cutOff = 15
intervalsLimit = 1000

# Thickness range (1 nm to 30 nm)

thickness_values = np.linspace(1, 30, 100) * Constants.NM.value

# Arrays to store results
Gxx_values = np.zeros(len(thickness_values), dtype=complex)
Gzz_values = np.zeros(len(thickness_values), dtype=complex)

# Compute Green's function for each frequency
for i, t in enumerate(thickness_values):
    slab = MetallicSlab(metal, epsilon1, epsilon3, omega, t, z)
    G = slab.calculateNormalizedGreenFunctionReflected(cutOff=cutOff, limit=intervalsLimit)

    Gxx_values[i] = G[0, 0]  # Gxx = Gyy
    Gzz_values[i] = G[2, 2]  # Gzz

# Create the figure
plt.figure(figsize=(12, 5))

# Plot Gxx
plt.subplot(1, 2, 1)
plt.plot(thickness_values / Constants.NM.value, np.real(Gxx_values), label="Re(Gxx)", color="b")
plt.plot(thickness_values / Constants.NM.value, np.imag(Gxx_values), label="Im(Gxx)", linestyle="dashed", color="r")
plt.xlabel("Slab thickness (nm)")
plt.ylabel("Normalized Gxx")
plt.title("Gxx vs Thickness")
plt.legend()

# Plot Gzz
plt.subplot(1, 2, 2)
plt.plot(thickness_values / Constants.NM.value, np.real(Gzz_values), label="Re(Gzz)", color="b")
plt.plot(thickness_values / Constants.NM.value, np.imag(Gzz_values), label="Im(Gzz)", linestyle="dashed", color="r")
plt.xlabel("Slab thickness (nm)")
plt.ylabel("Normalized Gzz")
plt.title("Gzz vs Thickness")
plt.legend()

# Add a text box with fixed parameters
params_text = (f"Fixed Parameters:\n"
               f"ε₁ = {epsilon1}, ε₃ = {epsilon3}\n"
               f"ωₚ = {metal.plasmaFrequency:.2f} eV\n"
               f"εB = {metal.epsilonB}, γ = {metal.gamma:.3f} eV\n"
               f"$\\omega$ = {omega / (Constants.EV.value*plasmon_resonance):.1f}"+ "$\\omega_{sp}$"+f", z = {z / Constants.NM.value} nm")

plt.gcf().text(0.02, 0.95, params_text, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()