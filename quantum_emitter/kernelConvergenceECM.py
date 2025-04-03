import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import time
from QuantumEmitter import QuantumEmitter, SpectralDensityType
from classical_green_function.MetallicSlab import Constants, MetallicSlab, Metals

start_time = time.time()

def meanSquaredError(reference, test):
    """Calculate the mean squared error between reference and test arrays."""
    return np.mean((reference - test) ** 2)

def convergenceRate(errors):
    """Calculate the convergence rate based on the errors."""
    return np.abs(np.diff(errors) / errors[:-1])

def savePlot(figure, name):
    """Save the plot in the 'plots' folder."""
    if not os.path.exists('plots'):
        os.makedirs('plots')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    figure.savefig(f'plots/{name}_{timestamp}.png', dpi=300, bbox_inches='tight')
    figure.savefig(f'plots/{name}_{timestamp}.pdf', bbox_inches='tight')

# Fixed parameters
metal = Metals.SILVER
slab_thickness = 20  # nm
emitter_wavelength = 300  # nm
dipole_moment = 1.0  # e·nm
distances = [200]  # Single value for simplicity
emitter_frequency = 1240 / emitter_wavelength  # eV
T = 10 / emitter_frequency
dt = 0.001 / emitter_frequency
fixed_num_points = 100  # Fixed number of points
fixed_cut_off1 = 5  # Fixed value for the second part

# Calculate reference with high cutOff1
high_cut_off1 = 50  # High reference value
high_num_points = 1000
phys_distance = distances[0] * Constants.NM.value
metallic_slab = MetallicSlab(
    metal, 1.0, 1.0, emitter_frequency * Constants.EV.value,
    slab_thickness * Constants.NM.value, phys_distance
)

reference_emitter = QuantumEmitter(SpectralDensityType.METALLIC_SLAB, {
    'metalSlab': metallic_slab,
    'dipole': np.array([1, 1, 1]) * dipole_moment,
    'omega0': emitter_frequency
}, cutOff=[15, high_cut_off1], numPoints=high_num_points)

times, reference_prob = reference_emitter.computeProbability(T, dt)

# Convergence analysis for cutOff[1]
cut_off1_values = np.logspace(np.log10(1), np.log10(high_cut_off1), 20)
errors_cut_off1 = []
for cut in cut_off1_values:
    emitter = QuantumEmitter(SpectralDensityType.METALLIC_SLAB, {
        'metalSlab': metallic_slab,
        'dipole': np.array([1, 1, 1]) * dipole_moment,
        'omega0': emitter_frequency
    }, cutOff=[15, cut], numPoints=fixed_num_points)
    _, test_prob = emitter.computeProbability(T, dt)
    errors_cut_off1.append(meanSquaredError(reference_prob, test_prob))

rates_cut_off1 = convergenceRate(errors_cut_off1)

# Convergence analysis for numPoints
num_points_values = np.logspace(np.log10(20), np.log10(high_num_points), 20, dtype=int)
errors_num_points = []
for num in num_points_values:
    emitter = QuantumEmitter(SpectralDensityType.METALLIC_SLAB, {
        'metalSlab': metallic_slab,
        'dipole': np.array([1, 1, 1]) * dipole_moment,
        'omega0': emitter_frequency
    }, cutOff=[15, fixed_cut_off1], numPoints=num)
    _, test_prob = emitter.computeProbability(T, dt)
    errors_num_points.append(meanSquaredError(reference_prob, test_prob))

rates_num_points = convergenceRate(errors_num_points)

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

axes[0, 0].plot(cut_off1_values, errors_cut_off1, 'o-', label='MSE vs cutOff[1]')
axes[0, 0].set_xlabel('cutOff[1]')
axes[0, 0].set_ylabel('Mean Squared Error')
axes[0, 0].set_title('Convergence with respect to cutOff[1]')
axes[0, 0].grid()
axes[0, 0].legend()

axes[0, 1].plot(num_points_values, errors_num_points, 'o-', label='MSE vs numPoints')
axes[0, 1].set_xlabel('numPoints')
axes[0, 1].set_ylabel('Mean Squared Error')
axes[0, 1].set_title('Convergence with respect to numPoints')
axes[0, 1].grid()
axes[0, 1].legend()

axes[1, 0].plot(cut_off1_values[:-1], rates_cut_off1, 's-', color='r', label='Rate vs cutOff[1]')
axes[1, 0].set_xlabel('cutOff[1]')
axes[1, 0].set_ylabel('Convergence Rate')
axes[1, 0].set_title('Convergence Rate for cutOff[1]')
axes[1, 0].grid()
axes[1, 0].legend()

axes[1, 1].plot(num_points_values[:-1], rates_num_points, 's-', color='r', label='Rate vs numPoints')
axes[1, 1].set_xlabel('numPoints')
axes[1, 1].set_ylabel('Convergence Rate')
axes[1, 1].set_title('Convergence Rate for numPoints')
axes[1, 1].grid()
axes[1, 1].legend()

plt.tight_layout()
savePlot(fig, 'kernel_convergence_analysis')

elapsed_time = time.time() - start_time
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"Total execution time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")

plt.show()
