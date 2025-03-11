import numpy as np
import matplotlib.pyplot as plt
from MetallicSlab import MetallicSlab, Constants, Metals

# Fixed parameters
epsilon1 = 1.0
epsilon3 = 1.0
t = 20 * Constants.NM.value  # Thickness in meters
z = 5 * Constants.NM.value  # Distance from the source

# Select a single metal
metal = Metals.SILVER
plasmon_resonance = metal.plasmaFrequency / np.sqrt(metal.epsilonB + 1)
omega = 2.0 * plasmon_resonance * Constants.EV.value

# Define range for limit and eps_rel
limit_values = np.arange(5, 60, 3)  # Varying limit from 5 to 200
eps_rel_values = np.logspace(-12, 1, 20)  # Varying eps_rel from 1e-8 to 1e3

# Matrices para almacenar valores
Gxx_real = np.zeros((len(eps_rel_values), len(limit_values)))
Gzz_real = np.zeros((len(eps_rel_values), len(limit_values)))
Gxx_imag = np.zeros((len(eps_rel_values), len(limit_values)))
Gzz_imag = np.zeros((len(eps_rel_values), len(limit_values)))

# Compute Green's function values
for i, eps_rel in enumerate(eps_rel_values):
    for j, limit in enumerate(limit_values):
        slab = MetallicSlab(metal, epsilon1, epsilon3, omega, t, z)
        G = slab.calculateNormalizedGreenFunctionReflected(cutOff=15, eps_rel=eps_rel, limit=limit)

        Gxx_real[i, j] = np.real(G[0, 0])
        Gzz_real[i, j] = np.real(G[2, 2])
        Gxx_imag[i, j] = np.imag(G[0, 0])
        Gzz_imag[i, j] = np.imag(G[2, 2])

# Plot Contour Plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

titles = [r"Re$(G_{xx})$", r"Re$(G_{zz})$", r"Im$(G_{xx})$", r"Im$(G_{zz})$"]
data = [Gxx_real, Gzz_real, Gxx_imag, Gzz_imag]

for ax, title, Z in zip(axes.ravel(), titles, data):
    contour = ax.contourf(limit_values, eps_rel_values, Z, levels=50, cmap="viridis")
    ax.set_xscale("linear")
    ax.set_yscale("log")
    ax.set_xlabel("limit")
    ax.set_ylabel(r"$\epsilon_{rel}$")
    ax.set_title(title)
    fig.colorbar(contour, ax=ax)

fig.suptitle(f"Green-Function Integral Convergence for t= {t/Constants.NM.value:.2f} nm, omega/plasmon resonance = {omega/(plasmon_resonance*Constants.EV.value):.3f}, z = {z/Constants.NM.value:.2f} nm")

plt.tight_layout()
plt.savefig(f"./plots/Integral_convergence_t_{t/Constants.NM.value:.0f}_omega_{omega/(plasmon_resonance*Constants.EV.value)*100:.0f}.png")

# Convergence threshold (adjustable)
threshold = 1e-5


def find_optimal_parameters(Z, limit_values, eps_rel_values, threshold):
    """Finds the smallest limit and largest eps_rel before a significant jump in relative change."""
    best_limit, best_eps = limit_values[-1], eps_rel_values[0]  # Default to most precise values

    for i in range(len(eps_rel_values) - 1):  # From low to high eps_rel
        for j in range(len(limit_values) - 1, 0, -1):  # From high to low limit
            # Compute relative differences
            delta_limit = abs(Z[i, j] - Z[i, j - 1]) / (abs(Z[i, j - 1]) + 1e-10)
            delta_eps = abs(Z[i, j] - Z[i + 1, j]) / (abs(Z[i + 1, j]) + 1e-10)

            if delta_limit > threshold or delta_eps > threshold:  # If a jump is detected
                return best_limit, best_eps  # Return the last stable values

            best_limit, best_eps = limit_values[j], eps_rel_values[i]  # Update stable values

    return best_limit, best_eps  # If no jump is found, return the most precise values


# Apply the improved criterion
opt_limit_xx_real, opt_eps_xx_real = find_optimal_parameters(Gxx_real, limit_values, eps_rel_values, threshold)
opt_limit_zz_real, opt_eps_zz_real = find_optimal_parameters(Gzz_real, limit_values, eps_rel_values, threshold)
opt_limit_xx_imag, opt_eps_xx_imag = find_optimal_parameters(Gxx_imag, limit_values, eps_rel_values, threshold)
opt_limit_zz_imag, opt_eps_zz_imag = find_optimal_parameters(Gzz_imag, limit_values, eps_rel_values, threshold)

# Print results
print(f"Optimal for Re(Gxx): limit = {opt_limit_xx_real}, eps_rel = {opt_eps_xx_real}")
print(f"Optimal for Re(Gzz): limit = {opt_limit_zz_real}, eps_rel = {opt_eps_zz_real}")
print(f"Optimal for Im(Gxx): limit = {opt_limit_xx_imag}, eps_rel = {opt_eps_xx_imag}")
print(f"Optimal for Im(Gzz): limit = {opt_limit_zz_imag}, eps_rel = {opt_eps_zz_imag}")

plt.show()

