import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import time
from QuantumEmitter import QuantumEmitter, SpectralDensityType
from classical_green_function.MetallicSlab import Constants, MetallicSlab, Metals

start_time = time.time()


def savePlot(figure, name):
    """Save the plot in the 'plots' folder."""
    if not os.path.exists('plots'):
        os.makedirs('plots')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    figure.savefig(f'plots/{name}_{timestamp}.png', dpi=300, bbox_inches='tight')
    figure.savefig(f'plots/{name}_{timestamp}.pdf', bbox_inches='tight')


def computeSpectralDensity(cutOff, numPoints, frequenciesEV, distance, metal, slabThickness, dipoleMoment,
                           emitterFrequency):
    """Compute the spectral density for a given configuration."""
    metallicSlab = MetallicSlab(metal=metal, epsilon1=1.0, epsilon3=1.0,
                                omega=emitterFrequency * Constants.EV.value, t=slabThickness, z=distance)
    params = {'metalSlab': metallicSlab, 'dipole': np.array([1, 1, 1]) * dipoleMoment, 'omega0': emitterFrequency}
    emitter = QuantumEmitter(SpectralDensityType.METALLIC_SLAB, params, cutOff=cutOff, numPoints=numPoints)
    return np.array([emitter.spectralDensity(omegaEV) for omegaEV in frequenciesEV])


# Fixed parameters
metal = Metals.SILVER
slab_thickness_nm = 20
emitter_wavelength_nm = 300
dipole_moment = 1.0
slab_thickness = slab_thickness_nm * Constants.NM.value
emitter_frequency = 1240 / emitter_wavelength_nm
distances_nm = [10]
distance = distances_nm[0] * Constants.NM.value
frequencies_eV = np.linspace(0.1 * emitter_frequency, 3 * emitter_frequency, 300)

# Reference values
cutOff_ref = [200, 5]
numPoints_ref = 2000
spectral_density_ref = computeSpectralDensity(cutOff_ref, numPoints_ref, frequencies_eV, distance, metal,
                                              slab_thickness, dipole_moment, emitter_frequency)

# Convergence test for cutOff[0]
cutOff_values = [10, 20, 50, 100, 150, 200]
errors_cutOff = []
convergence_rate_cutOff = []

for i, cutOff_0 in enumerate(cutOff_values):
    spectral_density = computeSpectralDensity([cutOff_0, 5], numPoints_ref, frequencies_eV, distance, metal,
                                              slab_thickness, dipole_moment, emitter_frequency)
    error = np.mean((spectral_density - spectral_density_ref) ** 2)
    errors_cutOff.append(error)

    if i > 0:
        convergence_rate_cutOff.append(errors_cutOff[i] / errors_cutOff[i - 1])
    else:
        convergence_rate_cutOff.append(np.nan)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(cutOff_values, errors_cutOff, 'o-', label='MSE vs cutOff[0]')
plt.xlabel('cutOff[0]')
plt.ylabel('Mean Squared Error')
plt.title('Convergence of cutOff[0] in Spectral Density')
plt.grid()
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(cutOff_values[1:], convergence_rate_cutOff[1:], 's-', label='Convergence Rate')
plt.xlabel('cutOff[0]')
plt.ylabel('Convergence Rate')
plt.title('Convergence Rate of cutOff[0]')
plt.grid()
plt.legend()

savePlot(plt.gcf(), 'spectralDensity_convergence_cutOff')

# Convergence test for numPoints
numPoints_values = [100, 200, 500, 1000, 1500, 2000]
errors_numPoints = []
convergence_rate_numPoints = []

for i, numPoints in enumerate(numPoints_values):
    spectral_density = computeSpectralDensity(cutOff_ref, numPoints, frequencies_eV, distance, metal, slab_thickness,
                                              dipole_moment, emitter_frequency)
    error = np.mean((spectral_density - spectral_density_ref) ** 2)
    errors_numPoints.append(error)

    if i > 0:
        convergence_rate_numPoints.append(errors_numPoints[i] / errors_numPoints[i - 1])
    else:
        convergence_rate_numPoints.append(np.nan)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(numPoints_values, errors_numPoints, 'o-', label='MSE vs numPoints')
plt.xlabel('numPoints')
plt.ylabel('Mean Squared Error')
plt.title('Convergence of numPoints in Spectral Density')
plt.grid()
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(numPoints_values[1:], convergence_rate_numPoints[1:], 's-', label='Convergence Rate')
plt.xlabel('numPoints')
plt.ylabel('Convergence Rate')
plt.title('Convergence Rate of numPoints')
plt.grid()
plt.legend()

savePlot(plt.gcf(), 'spectralDensity_convergence_numPoints')

elapsed_time = time.time() - start_time
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"Total execution time: {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}")
