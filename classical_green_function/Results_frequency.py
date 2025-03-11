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
eps_rel = 1e-6
limit = 100

# Select a single metal and thickness
metal = Metals.SILVER
t = 20 * Constants.NM.value  # Fixed thickness
plasmon_resonance = metal.plasmaFrequency / np.sqrt(metal.epsilonB + 1)

# Create directories for saving data and plots
os.makedirs("./data/", exist_ok=True)
os.makedirs("./plots/", exist_ok=True)

# Frequency range
min_omega = 0.95 * plasmon_resonance
max_omega = 1.05 * plasmon_resonance
omega_values = np.linspace(min_omega, max_omega, 200) * Constants.EV.value

# Arrays to store results
Gxx_values = np.zeros(len(omega_values), dtype=complex)
Gzz_values = np.zeros(len(omega_values), dtype=complex)

# Calculate the Green's function for each frequency
print(f"Computing for thickness {t / Constants.NM.value} nm")
index_good_omega = 0
count = 0
for i, omega in enumerate(omega_values):
    slab = MetallicSlab(metal, epsilon1, epsilon3, omega, t, z)
    G = slab.calculateNormalizedGreenFunctionReflected(cutOff=cutOff, eps_rel=eps_rel, limit=limit)
    Gxx_values[i] = G[0, 0]  # Gxx = Gyy
    Gzz_values[i] = G[2, 2]  # Gzz
    if omega > 0.5*plasmon_resonance*Constants.EV.value and count==0:
        index_good_omega = i
        count += 1


y_min_real = min(np.min(np.real(Gxx_values[index_good_omega:])),
            np.min(np.real(Gzz_values[index_good_omega:])))
y_max_real = max(np.max(np.real(Gxx_values[index_good_omega:])),
            np.max(np.real(Gzz_values[index_good_omega:])))
y_min_imag = min(np.min(np.imag(Gxx_values[index_good_omega:])),
            np.min(np.imag(Gzz_values[index_good_omega:])))
y_max_imag = max(np.max(np.imag(Gxx_values[index_good_omega:])),
                 np.max(np.imag(Gzz_values[index_good_omega:])))


# Save data
np.savez(f"./data/{metal.name}_thickness_{int(t / Constants.NM.value)}nm_single_green_function.npz",
         omega=omega_values, Gxx=Gxx_values, Gzz=Gzz_values)

# Create figure with two subplots (aligned horizontally)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))  # 1 row, 2 columns

# Plot Real Parts
axes[0].plot(omega_values / Constants.EV.value, np.real(Gxx_values), label="Re(Gxx)", color="b")
axes[0].plot(omega_values / Constants.EV.value, np.real(Gzz_values), label="Re(Gzz)", color="r")
axes[0].axvline(x=plasmon_resonance, color="k", linestyle="dotted", label="Plasmon Resonance")
axes[0].set(title=f"Real Part of Green's Function for {metal.name}, t={t / Constants.NM.value} nm",
            xlabel="Frequency (eV)", ylabel="Re(G)")
axes[0].legend()
axes[0].grid(True)
axes[0].set_xlim(min_omega,max_omega)
axes[0].set_ylim(1.2*y_min_real,1.2*y_max_real)

# Plot Imaginary Parts
axes[1].plot(omega_values / Constants.EV.value, np.imag(Gxx_values), label="Im(Gxx)", linestyle="dashed", color="b")
axes[1].plot(omega_values / Constants.EV.value, np.imag(Gzz_values), label="Im(Gzz)", linestyle="dashed", color="r")
axes[1].axvline(x=plasmon_resonance, color="k", linestyle="dotted", label="Plasmon Resonance")
axes[1].set(title=f"Imaginary Part of Green's Function for {metal.name}, t={t / Constants.NM.value:.2f} nm",
            xlabel="Frequency (eV)", ylabel="Im(G)")
axes[1].legend()
axes[1].grid(True)
axes[1].set_xlim(min_omega, max_omega)
axes[1].set_ylim(1.2*y_min_imag,1.2*y_max_imag)

plt.tight_layout()  # Ajusta el diseño para evitar solapamientos

# Save plot
plt.savefig(f"./plots/{metal.name}_thickness_{int(t / Constants.NM.value)}nm_real_imag_green_function_zoom.png")

end = time.time()
elapsed_time = end - start
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"Execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")

plt.show()


